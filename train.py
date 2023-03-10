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
from utils import get_batch_size_schedule, get_optimizer_and_scheduler, get_dataloader
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
    loss_spike_mult_threshold: float # threshold for detecting loss spikes (ratio of current loss to previous loss)
    loss_spike_add_threshold: float # threshold for detecting loss spikes (absolute value of current loss)
    max_microbatch_skips: int # max number of microbatches to skip before stopping training

    # logging, eval, & checkpointing
    use_wandb: bool
    wandb_project: str
    wandb_watch: bool
    log_interval: int
    val_interval: int
    save_interval: int
    save_dir: str
    recovery_ckpt_path: str # path to checkpoint to recover from

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_dict(self):
        return self.__dict__

def micro_batch_step(x, y, model, train_config, running_previous_loss):
    x, y = x.to(train_config.device), y.to(train_config.device)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_amp):
        micro_batch_loss = model(x, targets=y)
        if micro_batch_loss.item() > min(running_previous_loss * train_config.loss_spike_mult_threshold,
                                                running_previous_loss + train_config.loss_spike_add_threshold):
            print(f"Loss spike detected, skipping microbatch. (Reason: Loss {micro_batch_loss.item()} exceeded threshold)")
            return None
        elif torch.isnan(micro_batch_loss).item():
            print(f"Loss spike detected, skipping microbatch. (Reason: Loss = NaN)")
            return None
        else:
            if train_config.use_wandb:
                wandb.log({"microbatch_train_loss": micro_batch_loss.item()})
            return micro_batch_loss

def step_optimizer():
    pass

def evaluate_and_save_model(model, training_step, val_loader, train_config):
    model.eval()
    val_steps = 0
    val_loss = 0                
    start = time.time()
    with torch.no_grad():
        for x, y in val_loader:
            val_steps += 1
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_config.use_amp):
                x, y = x.to(train_config.device), y.to(train_config.device)
                loss = model(x, targets=y)
                val_loss += loss.item()
        val_loss /= val_steps                    
    end = time.time()
    if train_config.use_wandb:
        wandb.log({"val_loss": val_loss})
    print(f"==== VALIDATION LOSS: {val_loss:.4f} ====")
    print(f"Validation took {end - start:.2f} seconds for {val_steps} steps, {(end - start) / val_steps:.2f} seconds per microbatch.")                   
    print("Saving model...")
    if not os.path.exists(train_config.save_dir):
        os.mkdir(train_config.save_dir)
    torch.save(model.state_dict(), f"{train_config.save_dir}/{training_step}.pt")
    torch.save(model.state_dict(), train_config.recovery_ckpt_path)
    model.train()
    del x, y, loss
    return val_loss.item()

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
    assert train_config.max_batch_size % train_config.micro_batch_size == 0,\
        f"Batch size {train_config.max_batch_size} must be divisible by micro batch size {train_config.micro_batch_size}"
    train_is_split = pathlib.Path(train_config.train_path).is_dir()
    if train_is_split:
        train_files = sorted(glob.glob(os.path.join(train_config.train_path, "*.bin")))
    else:
        train_files = [train_config.train_path]
    train_config.batch_size_schedule = get_batch_size_schedule(train_config)
    train_config.total_steps = len(train_config.batch_size_schedule)
    train_config.total_microbatches = int(np.sum(train_config.batch_size_schedule) // train_config.micro_batch_size)
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
    train_config.device = device

    # If doing eval, set up val dataset/dataloader
    if train_config.do_eval:
        if train_config.val_path is None:
            raise ValueError("Validation path must be provided if do_eval is True.")
        val_loader = get_dataloader(train_config.val_path, tokenizer, train_config, max_seqs=train_config.max_val_seqs)

    # Initialize optimizer, scheduler, and scaler
    optimizer, scheduler = get_optimizer_and_scheduler(model, train_config)
    scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)

    # Initialize wandb
    if train_config.use_wandb:
        wandb.init(
            project=train_config.wandb_project, 
            config={"bert_config": bert_config.to_dict(), "train_config": train_config.to_dict()}
        )
    if train_config.wandb_watch: wandb.watch(model, log="all")

    # Training loop
    train_dataset = None
    train_loader = None
    microbatch_skips = 0
    ckpt_recovery_attempted = False
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
            train_loader = get_dataloader(train_file, tokenizer, train_config, max_seqs=0)
            for x, y in train_loader:
                if train_seqs_so_far >= train_config.max_train_seqs:
                    break
                micro_batch_loss = micro_batch_step(x, y, model, train_config, running_previous_loss)
                if micro_batch_loss is None:
                    microbatch_skips += 1
                else:
                    normalized_loss = micro_batch_loss / accum_iters
                    running_batch_loss += normalized_loss.item()
                    scaler.scale(normalized_loss).backward()
                    micro_batches += 1
                    microbatch_skips = 0
                    del normalized_loss
                if microbatch_skips >= train_config.max_microbatch_skips:
                    if not ckpt_recovery:
                        print("Too many microbatch skips. Attempting to recover from checkpoint.")
                        ckpt_recovery_attempted = True
                        model.load_weights_from_checkpoint(train_config.recovery_ckpt_path)
                        microbatch_skips = 0

                        # Reset optimizer restart the batch
                        print("Resetting optimizer and scheduler.")
                        micro_batches = 0
                        running_batch_loss = 0
                        optimizer.zero_grad(set_to_none=True)

                    else:
                        print("Unable to stabilize training. Exiting.")
                        sys.exit(1)
                del x, y, micro_batch_loss

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
                        evaluate_and_save_model(model, training_step, val_loader, train_config)
                        ckpt_recovery_attempted = False
                    train_seqs_so_far += train_config.batch_size_schedule[training_step]
                    training_step += 1
                    if training_step >= train_config.total_steps:
                        break
                    if train_seqs_so_far >= train_config.max_train_seqs:
                        break
                    micro_batches = 0
                    running_batch_loss = 0
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        accum_iters =  train_config.batch_size_schedule[training_step] // train_config.micro_batch_size
                    except:
                        print("Reached end of batch size schedule.")
                        break
    # Take final step with remaining microbatches
    if micro_batches > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if train_config.use_wandb:
            wandb.log({
                "batch_train_loss": running_batch_loss,
                "lr": scheduler.get_last_lr()[0],
                "batch_size": train_config.batch_size_schedule[training_step]
            })
        print(f"Step {training_step} | Train loss: {running_batch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f} | Batch size: {train_config.batch_size_schedule[training_step]}")
    if train_config.do_eval:
        final_val_loss = evaluate_and_save_model(model, training_step, val_loader, train_config)
        print(f"Final validation loss: {final_val_loss:.4f}")    
    print("Saving final model...")
    if not os.path.exists(train_config.save_dir):
        os.mkdir(train_config.save_dir)
    torch.save(model.state_dict(), f"{train_config.save_dir}/final.pt")

if __name__ == '__main__':
    fire.Fire(train_bert)