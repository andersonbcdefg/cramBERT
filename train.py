# Training notes:
# We use a one-cycle learning rate schedule.
# Gradient accumulation is used to increase the effective batch size.
# Gradient clipping is used to stabilize training.
# We use Adam with weight decay. (TODO: Use apex for fused AdamW, 
    # or 8-bit adam or adafactor to save memory) https://arxiv.org/pdf/2110.02861.pdf
# Use automatic mixed precision training to save memory.
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# TODO: consider gradient checkpointing
# TODO: from Cramming, sort training data by prevalence (pseudo-curriculum learning)
import os
import time
import sys
import wandb
import fire
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from data import BERTDataset
from dataclasses import dataclass
from model import BERT, BERTConfig

@dataclass
class TrainConfig:
    # training budget
    max_train_seqs: int # max number of training samples to use
    max_val_seqs: int # max number of validation samples to use
    gpus: int # number of gpus to use
    train_workers: int # number of workers for train_dataloader

    # data
    train_path: str
    val_path: str
    mask_token_id: int # id of mask token in vocab
    vocab_size: int # vocab size for data/tokenizer
    micro_batch_size: int # 128 or 256 whatever fits in memory
    max_batch_size: int # recommended 4096
    anneal_batch_size: bool # whether to anneal batch size
    batch_size_anneal_frac: float # what fraction of training examples to use for annealing

    # one-cycle lr schedule -- total steps calculated dynamically
    pct_start: float
    max_lr: float
    start_div_factor: float
    end_div_factor: float

    # optimizer
    optimizer: str
    b1: float # adam beta1
    b2: float # adam beta2
    weight_decay: float
    max_grad_norm: float
    fused: bool # whether to use fused adam (adamw not supported in stable pytorch yet)

    # logging, eval, & checkpointing
    use_wandb: bool
    log_interval: int
    val_interval: int
    save_interval: int
    save_dir: str

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_dict(self):
        return self.__dict__

def train_bert(bert_config, train_config):
    print(f"BERT Config: {bert_config}")
    print(f"Train Config: {train_config}")

    # If provided strings, load configs from yaml files
    if isinstance(bert_config, str):
        bert_config = BERTConfig.from_yaml(bert_config)
    if isinstance(train_config, str):
        train_config = TrainConfig.from_yaml(train_config)
    
    # Initialize data
    train_dataset = BERTDataset(
        train_config.train_path, 
        train_config.vocab_size,
        bert_config.max_seq_len,
        train_config.mask_token_id,
        max_seqs = train_config.max_train_seqs,
        loop = True
    )
    val_dataset = BERTDataset(
        train_config.val_path, 
        train_config.vocab_size,
        bert_config.max_seq_len,
        train_config.mask_token_id,
        max_seqs = train_config.max_val_seqs,
        loop = False
    )

    # Error check and calculate batch size schedule, total steps
    n_train_seqs = train_dataset.n_seqs
    assert train_config.max_batch_size % train_config.micro_batch_size == 0,\
        f"Batch size {train_config.max_batch_size} must be divisible by micro batch size {train_config.micro_batch_size}"
    if train_config.anneal_batch_size:
        anneal_budget = n_train_seqs * train_config.batch_size_anneal_frac
        average_annealed_batch_size = (train_config.max_batch_size + train_config.micro_batch_size) // 2
        anneal_steps = int(anneal_budget // average_annealed_batch_size)
        batch_size_schedule = np.linspace(train_config.micro_batch_size, train_config.max_batch_size, anneal_steps)
        batch_size_schedule = np.round(batch_size_schedule / train_config.micro_batch_size) * train_config.micro_batch_size
        batch_size_schedule = batch_size_schedule.astype(int)
        remaining_seqs = n_train_seqs - np.sum(batch_size_schedule)
        remaining_steps = remaining_seqs // train_config.max_batch_size
        batch_size_schedule = np.concatenate([batch_size_schedule, np.ones(remaining_steps) * train_config.max_batch_size])
    else:
        batch_size_schedule = np.ones(n_train_seqs // train_config.max_batch_size) * train_config.max_batch_size
    train_config.batch_size_schedule = batch_size_schedule
    train_config.total_steps = len(batch_size_schedule)
    print(f"Training for {train_config.total_steps} steps.")
    
    # Initialize model & data loaders
    num_gpus = min(train_config.gpus, torch.cuda.device_count())
    if num_gpus > 1:
        raise NotImplementedError("Multi-GPU training not implemented.")
    device = torch.device('cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
    model = BERT(bert_config)
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=max(4, 4 * num_gpus), pin_memory=num_gpus > 0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=max(4, 4 * num_gpus), pin_memory=num_gpus > 0)

    # Initialize optimizer and scheduler
    assert train_config.optimizer in ["Adam", "AdamW"], "Only Adam and AdamW optimizers currently supported."
    # group parameters by weight decay
    # https://github.com/karpathy/nanoGPT/blob/master/model.py
    # https://github.com/JonasGeiping/cramming/blob/main/cramming/backend/utils.py
    # https://huggingface.co/transformers/v3.3.1/training.html
    optim_groups = model.get_optim_groups(train_config.weight_decay)
    if train_config.optimizer == "Adam":
        optimizer = torch.optim.Adam(optim_groups)
    elif train_config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optim_groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=train_config.max_lr, 
        total_steps=train_config.total_steps, 
        pct_start=train_config.pct_start, 
        div_factor=train_config.start_div_factor, 
        final_div_factor=train_config.end_div_factor,
        anneal_strategy='linear',
        three_phase=False
    )

    # Initialize wandb
    if train_config.use_wandb:
        wandb.init(project="cramming", 
        config={
            "bert_config": bert_config.to_dict(),
            "train_config": train_config.to_dict()
        })

    # Training loop, with gradient accumulation
    training_step = 0
    micro_batches = 0
    accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
    model.train()
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        micro_batch_loss = model(x, targets=y, mask=mask)
        if train_config.use_wandb:
            wandb.log({
                "microbatch_train_loss": micro_batch_loss.item()
            })
        normalized_loss = micro_batch_loss / accum_iters
        normalized_loss.backward()
        micro_batches += 1
        # Once microbatches accumulated, take a step
        if micro_batches == accum_iters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if train_config.use_wandb:
                wandb.log({
                    "accumulated_train_loss": normalized_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "batch_size": train_config.batch_size_schedule[training_step]
                })
            if training_step % train_config.log_interval == 0:
                print(f"Step {training_step} | Train loss: {normalized_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.4f} | Batch size: {train_config.batch_size_schedule[training_step]}")
            if training_step % train_config.val_interval == 0:
                model.eval()
                val_steps = 0
                val_loss = 0
                # start time
                start = time.time()
                with torch.no_grad():
                    for x, y, mask in tqdm(val_loader):
                        val_steps += 1
                        x, y, mask = x.to(device), y.to(device), mask.to(device)
                        loss = model(x, targets=y, mask=mask)
                        val_loss += loss.item()
                val_loss /= val_steps
                # end time
                end = time.time()
                if train_config.use_wandb:
                    wandb.log({
                        "val_loss": val_loss
                    })
                print(f"Step {training_step} | Val loss: {val_loss:.4f}")
                print(f"Validation took {end - start:.2f} seconds for {val_steps} steps.")
                print("Time per micro-batch: ", (end - start) / val_steps, " seconds")
                print("Saving model...")
                if not os.path.exists(train_config.save_dir):
                    os.mkdir(train_config.save_dir)
                torch.save(model.state_dict(), f"{train_config.save_dir}/{training_step}.pt")
                model.train()
            training_step += 1
            micro_batches = 0
            accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
            if training_step == train_config.total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), f"{train_config.save_dir}/final.pt")
    

if __name__ == '__main__':
    fire.Fire(train_bert)