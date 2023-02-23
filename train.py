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
from data import BERTDataset, load_tokenizer
from dataclasses import dataclass
from model import BERT, HuggingFaceRoBERTa, BERTConfig
import bitsandbytes as bnb

@dataclass
class TrainConfig:
    # model
    model: str # either "mine" my BERT, or "huggingface" RoBERTa

    # training budget
    max_train_seqs: int # max number of training samples to use
    do_eval: bool # whether to evaluate on validation set
    max_val_seqs: int # max number of validation samples to use
    gpus: int # number of gpus to use
    train_workers: int # number of workers for train_dataloader

    # other training configs
    use_amp: bool # whether to use automatic mixed precision training
    use_checkpointing: bool # whether to use gradient checkpointing

    # data
    train_path: str
    val_path: str
    loop_train_data: bool # whether to loop over training data, or single-epoch
    tokenizer_path: str
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
    eight_bit: bool # whether to use 8-bit adam

    # logging, eval, & checkpointing
    use_wandb: bool
    wandb_project: str
    wandb_watch: bool
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
    
    # Get tokenizer
    tokenizer = load_tokenizer(train_config.tokenizer_path)

    # Initialize data
    train_dataset = BERTDataset(
        train_config.train_path, 
        tokenizer,
        bert_config.max_seq_len,
        max_seqs = train_config.max_train_seqs,
        loop = train_config.loop_train_data
    )

    if train_config.do_eval:
        val_dataset = BERTDataset(
            train_config.val_path, 
            tokenizer,
            bert_config.max_seq_len,
            max_seqs = train_config.max_val_seqs,
            loop = False
        )

    # Error check and calculate batch size schedule, total steps
    n_train_seqs = train_dataset.n_seqs if not train_config.loop_train_data else train_config.max_train_seqs
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
    train_config.total_microbatches = int(np.sum(batch_size_schedule) // train_config.micro_batch_size)
    print(f"Training for {train_config.total_steps} steps.")
    
    # Initialize model & data loaders
    num_gpus = min(train_config.gpus, torch.cuda.device_count())
    if num_gpus > 1:
        raise NotImplementedError("Multi-GPU training not implemented.")
    device = torch.device('cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
    if train_config.model == "mine":
        model = BERT(bert_config)
    elif train_config.model == "huggingface":
        model = HuggingFaceRoBERTa(bert_config)
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=train_config.train_workers, pin_memory=num_gpus > 0)
    val_loader = None
    if train_config.do_eval:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=train_config.train_workers, pin_memory=num_gpus > 0)

    # Initialize optimizer, scheduler, and scaler
    assert train_config.optimizer in ["Adam", "AdamW"], "Only Adam and AdamW optimizers currently supported."
    # group parameters by weight decay
    # https://github.com/karpathy/nanoGPT/blob/master/model.py
    # https://github.com/JonasGeiping/cramming/blob/main/cramming/backend/utils.py
    # https://huggingface.co/transformers/v3.3.1/training.html
    optim_groups = model.get_optim_groups(train_config.weight_decay)
    if train_config.eight_bit and train_config.optimizer == "Adam":
        optimizer = bnb.optim.Adam8bit(optim_groups, betas=(train_config.b1, train_config.b2))
    elif train_config.eight_bit and train_config.optimizer == "AdamW":
        optimizer = bnb.optim.AdamW8bit(optim_groups, betas=(train_config.b1, train_config.b2))
    elif train_config.optimizer == "Adam" and train_config.fused:
        optimizer = torch.optim.Adam(optim_groups, betas=(train_config.b1, train_config.b2), fused=True)
    elif train_config.optimizer == "Adam":
        optimizer = torch.optim.Adam(optim_groups, betas=(train_config.b1, train_config.b2))
    elif train_config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optim_groups, betas=(train_config.b1, train_config.b2))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=train_config.max_lr, 
        total_steps=train_config.total_microbatches, 
        pct_start=train_config.pct_start, 
        div_factor=train_config.start_div_factor, 
        final_div_factor=train_config.end_div_factor,
        anneal_strategy='linear',
        three_phase=False
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)

    # Initialize wandb
    if train_config.use_wandb:
        wandb.init(
            project=train_config.wandb_project, 
            config={
                "bert_config": bert_config.to_dict(),
                "train_config": train_config.to_dict()
            }
        )
    if train_config.wandb_watch:
        wandb.watch(model, log="all")

    # Training loop, with gradient accumulation
    training_step = 0
    micro_batches = 0
    running_batch_loss = 0
    accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
    print("initial accum_iters", accum_iters)
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_amp):
            micro_batch_loss = model(x, targets=y)
            if train_config.use_wandb:
                wandb.log({
                    "microbatch_train_loss": micro_batch_loss.item()
                })
            normalized_loss = micro_batch_loss / accum_iters
            running_batch_loss += normalized_loss.item()
        scaler.scale(normalized_loss).backward()
        micro_batches += 1
        scheduler.step()
        del x, y, micro_batch_loss, normalized_loss
        # Once microbatches accumulated, take a step
        if micro_batches == accum_iters:
            print("taking a step")
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if train_config.use_wandb:
                wandb.log({
                    "batch_train_loss": running_batch_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "batch_size": train_config.batch_size_schedule[training_step]
                })
            if training_step % train_config.log_interval == 0:
                print(f"Step {training_step} | Train loss: {running_batch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.4f} | Batch size: {train_config.batch_size_schedule[training_step]}")
            del running_batch_loss
            if training_step % train_config.val_interval == 0 and train_config.do_eval:
                model.eval()
                val_steps = 0
                val_loss = 0
                # start time
                start = time.time()
                with torch.no_grad():
                    for x, y in val_loader:
                        val_steps += 1
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_amp):
                            x, y = x.to(device), y.to(device)
                            loss = model(x, targets=y)
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
                del x, y, loss, val_loss
            training_step += 1
            micro_batches = 0
            running_batch_loss = 0
            print("Training step: ", training_step)
            accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
            print("new accum_iters", accum_iters")
            if training_step == train_config.total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), f"{train_config.save_dir}/final.pt")
    

if __name__ == '__main__':
    fire.Fire(train_bert)