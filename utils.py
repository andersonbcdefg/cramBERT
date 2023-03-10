import os
import glob
import pathlib
import numpy as np
from data import *

def get_batch_size_schedule(train_config, train_files):
    """Get the batch size schedule for training.
    Args:
        train_config (dict): The training configuration.
    Returns:
        batch_size_schedule (list): The batch size schedule.
    """
    n_train_seqs = sum([os.path.getsize(f) // bytes_per_seq for f in train_files])
    bytes_per_seq = train_config.seq_len * 2
    n_train_seqs = min(n_train_seqs, train_config.max_train_seqs)
    n_train_seq_steps = n_train_seqs * train_config.epochs
    
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
    return batch_size_schedule

def get_optimizer_and_scheduler(model, train_config):
    """Get the optimizer.
    Args:
        model (nn.Module): The model.
        train_config (dict): The training configuration.
    Returns:
        optimizer (torch.optim.Optimizer): The optimizer.
    """
     assert train_config.optimizer in ["Adam", "AdamW"], "Only Adam and AdamW optimizers currently supported."
    # group parameters by weight decay
    # https://github.com/karpathy/nanoGPT/blob/master/model.py
    # https://github.com/JonasGeiping/cramming/blob/main/cramming/backend/utils.py
    # https://huggingface.co/transformers/v3.3.1/training.html
    optim_groups = model.get_optim_groups(train_config.weight_decay)
    if train_config.eight_bit:
        if train_config.fused:
            raise ValueError("Fused optimizer not supported with 8-bit optimizer.")
        if train_config.optimizer == "Adam":
            optimizer = bnb.optim.Adam8bit(optim_groups, betas=(train_config.b1, train_config.b2))
        elif train_config.optimizer == "AdamW":
            optimizer = bnb.optim.AdamW8bit(optim_groups, betas=(train_config.b1, train_config.b2))
    elif train_config.fused:
        if train_config.optimizer == "Adam":
            optimizer = torch.optim.Adam(optim_groups, betas=(train_config.b1, train_config.b2), fused=True)
        elif train_config.optimizer == "AdamW":
            raise ValueError("AdamW fused optimizer not supported.")
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
    return optimizer, scheduler

def get_dataloader(data_file, tokenizer, train_config, max_seqs=0):
    """Get the dataloader.
    Args:
        data_file (str): The data file.
        train_config (dict): The training configuration.
    Returns:
        dataloader (torch.utils.data.DataLoader): The dataloader.
    """
    if train_config.in_memory:
        dataset = InMemoryBERTDataset(
            data_file, 
            tokenizer,
            train_config.seq_len,
            max_seqs=max_seqs
        )
    else:
        dataset = BERTDataset(
            data_file, 
            tokenizer,
            train_config.seq_len,
            max_seqs=max_seqs
        )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=train_config.micro_batch_size,
        shuffle=False,
        num_workers=train_config.train_workers,
        pin_memory=True
    )
    return loader
    