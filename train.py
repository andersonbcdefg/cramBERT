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
import glob
import pathlib
import time
import sys
import wandb
import fire
import yaml
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from data import InMemoryBERTDataset, BERTDataset, load_tokenizer
from dataclasses import dataclass
from model import BERT, HuggingFaceRoBERTa, BERTConfig
import bitsandbytes as bnb

@dataclass
class TrainConfig:
    # training budget
    max_train_seqs: int # max number of training samples to use
    epochs: int # number of epochs to train
    do_eval: bool # whether to evaluate on validation set
    max_val_seqs: int # max number of validation samples to use
    gpus: int # number of gpus to use
    train_workers: int # number of workers for train_dataloader

    # other training configs
    use_amp: bool # whether to use automatic mixed precision training
    use_checkpointing: bool # whether to use gradient checkpointing -- not implemented yet

    # data
    train_path: str
    val_path: str
    seq_len: int
    tokenizer_path: str
    in_memory: bool # whether to load all data from current fileinto CPU memory
    micro_batch_size: int # 128 or 256 whatever fits in memory
    max_batch_size: int # recommended 4096
    anneal_batch_size: bool # whether to anneal batch size
    batch_size_anneal_frac: float # what fraction of training examples to use for annealing

    # one-cycle lr schedule -- total steps calculated dynamically
    pct_start: float
    max_lr: float
    start_div_factor: float
    end_div_factor: float
    anneal_strategy: str # linear or cosine

    # optimizer
    optimizer: str
    b1: float # adam beta1
    b2: float # adam beta2
    weight_decay: float
    max_grad_norm: float
    fused: bool # whether to use fused adam (adamw not supported in stable pytorch yet)
    eight_bit: bool # whether to use 8-bit adam
    loss_spike_threshold: float # threshold for detecting loss spikes

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

    # Calculate steps and prepare batch and LR schedules
    train_is_split = pathlib.Path(train_config.train_path).is_dir()
    bytes_per_seq = train_config.seq_len * 2
    if train_is_split:
        train_files = sorted(glob.glob(os.path.join(train_config.train_path, "*.bin")))
        n_train_seqs = sum([
            os.path.getsize(f) // bytes_per_seq for f in train_files
        ])
    else:
        n_train_seqs = os.path.getsize(train_config.train_path) // bytes_per_seq
        train_files = [train_config.train_path]
    n_train_seqs = min(n_train_seqs, train_config.max_train_seqs)
    n_train_seq_steps = n_train_seqs * train_config.epochs
    assert train_config.max_batch_size % train_config.micro_batch_size == 0,\
        f"Batch size {train_config.max_batch_size} must be divisible by micro batch size {train_config.micro_batch_size}"
    if train_config.anneal_batch_size:
        anneal_budget = n_train_seq_steps * train_config.batch_size_anneal_frac
        average_annealed_batch_size = (train_config.max_batch_size + train_config.micro_batch_size) // 2
        anneal_steps = int(anneal_budget // average_annealed_batch_size)
        batch_size_schedule = np.linspace(train_config.micro_batch_size, train_config.max_batch_size, anneal_steps)
        batch_size_schedule = np.round(batch_size_schedule / train_config.micro_batch_size) * train_config.micro_batch_size
        batch_size_schedule = batch_size_schedule.astype(int)
        remaining_seq_steps = n_train_seq_steps - np.sum(batch_size_schedule)
        remaining_batch_steps = remaining_seq_steps // train_config.max_batch_size
        batch_size_schedule = np.concatenate([batch_size_schedule, np.ones(remaining_batch_steps) * train_config.max_batch_size])
    else:
        batch_size_schedule = np.ones(n_train_seq_steps // train_config.max_batch_size) * train_config.max_batch_size
    train_config.batch_size_schedule = batch_size_schedule
    train_config.total_steps = len(batch_size_schedule)
    train_config.total_microbatches = int(np.sum(batch_size_schedule) // train_config.micro_batch_size)
    print(f"Training for {train_config.total_steps} steps.")
    
    # Initialize model
    num_gpus = min(train_config.gpus, torch.cuda.device_count())
    if num_gpus > 1:
        raise NotImplementedError("Multi-GPU training not implemented.")
    device = torch.device('cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
    if bert_config.model == "BERT":
        model = BERT(bert_config)
    elif train_config.model == "RoBERTa":
        model = HuggingFaceRoBERTa(bert_config)
    model.to(device)

    # If doing eval, set up val dataset/dataloader
    if train_config.do_eval:
        if train_config.val_path is None:
            raise ValueError("Validation path must be provided if do_eval is True.")
        val_dataset = BERTDataset(
            train_config.val_path, 
            tokenizer,
            train_config.seq_len, 
            max_seqs = train_config.max_val_seqs
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=train_config.micro_batch_size,
            shuffle=False,
            num_workers=train_config.train_workers,
            pin_memory=True
        )

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
        anneal_strategy="cos" if train_config.anneal_strategy == "cosine" else "linear",
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

    # Training loop
    train_dataset = None
    train_loader = None
    running_previous_loss = -math.log(1.0 / bert_config.vocab_size) # initialize to maximum entropy
    training_step = 0
    micro_batches = 0
    running_batch_loss = 0
    accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
    model.train()
    for epoch in range(train_config.epochs):
        train_seqs_so_far = 0
        for train_file in train_files:
            print("Training on file: ", train_file)
            del train_dataset
            del train_loader
            if train_seqs_so_far >= train_config.max_train_seqs:
                break
            if train_config.in_memory:
                train_dataset = InMemoryBERTDataset(
                    train_file, 
                    tokenizer,
                    train_config.seq_len
                )
            else:
                train_dataset = BERTDataset(
                    train_file, 
                    tokenizer,
                    train_config.seq_len
                )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=train_config.micro_batch_size,
                shuffle=True,
                num_workers=train_config.train_workers,
                pin_memory=True
            )
            for x, y in train_loader:
                if train_seqs_so_far >= train_config.max_train_seqs:
                    break
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_amp):
                    micro_batch_loss = model(x, targets=y)
                    if train_config.use_wandb:
                        wandb.log({
                            "microbatch_train_loss": micro_batch_loss.item()
                        })
                # Skip microbatch if loss spikes / NaNs
                if micro_batch_loss.item() > running_previous_loss * train_config.loss_spike_threshold or torch.isnan(micro_batch_loss).item():
                    print(f"Loss spike detected, skipping microbatch.")
                    continue
                else:
                    normalized_loss = micro_batch_loss / accum_iters
                    running_batch_loss += normalized_loss.item()
                    scaler.scale(normalized_loss).backward()
                    micro_batches += 1
                
                # Scheduler always takes a step, because it's based on total amount of data
                scheduler.step()
                
                # Once enough microbatches accumulated, take a step. No loss spike check here, since
                # we already checked for spikes in the microbatches.
                if micro_batches == accum_iters:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update previous loss using exponential moving average
                    running_previous_loss = running_previous_loss * 0.5 + running_batch_loss * 0.5
                    
                    # Handle logging and validation
                    if train_config.use_wandb:
                        wandb.log({
                            "batch_train_loss": running_batch_loss,
                            "lr": scheduler.get_last_lr()[0],
                            "batch_size": train_config.batch_size_schedule[training_step]
                        })
                    if training_step % train_config.log_interval == 0:
                        print(f"Step {training_step} | Train loss: {running_batch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f} | Batch size: {train_config.batch_size_schedule[training_step]}")
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
                    train_seqs_so_far += train_config.batch_size_schedule[training_step]
                    if training_step == train_config.total_steps:
                        break
                    if train_seqs_so_far >= train_config.max_train_seqs:
                        break
                    micro_batches = 0
                    running_batch_loss = 0
                    accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
                    optimizer.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), f"{train_config.save_dir}/final.pt") 

if __name__ == '__main__':
    fire.Fire(train_bert)