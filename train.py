# Training notes:
# We use a one-cycle learning rate schedule.
# Gradient accumulation is used to increase the effective batch size.
# Gradient clipping is used to stabilize training.
# We use Adam with weight decay. (TODO: Use apex for fused AdamW, 
    # or 8-bit adam or adafactor to save memory) https://arxiv.org/pdf/2110.02861.pdf
# Use automatic mixed precision training to save memory.
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# TODO: use mmap for loading data
# TODO: consider gradient checkpointing
# TODO: from Cramming, sort training data by prevalence (pseudo-curriculum learning)
# TODO: add wandb logging

import fire
import yaml
import torch
import numpy as np
from data import BERTDataset
from dataclasses import dataclass
from model import BERT, BERTConfig


@dataclass
class TrainConfig:
    # data
    data_path: str
    vocab_size: int # vocab size for data/tokenizer
    micro_batch_size: int # 128 or 256 whatever fits in memory
    batch_size: int # recommended 4096
    val_size: float # fraction of training data size to use for validation
    
    # lr schedule
    total_steps: int
    pct_start: float
    max_lr: float

    # optimizer
    weight_decay: float
    max_grad_norm: float
    log_interval: int
    save_interval: int
    save_dir: str
    gpus: int

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

def train_bert(bert_config, train_config):
    # initialize model
    model = BERT(bert_config)

    # initialize data
    full_dataset = np.load(train_config.data_path)
    np.random.shuffle(full_dataset)
    train_samples_needed = train_config.total_steps * train_config.batch_size
    val_samples_needed = train_config.val_size * train_samples_needed
    assert len(full_dataset) >= train_samples_needed + val_samples_needed, 
    f"Not enough data for training: need {train_samples_needed + val_samples_needed} samples, but only have {len(full_dataset)}"
    train_samples = full_dataset[:train_samples_needed]
    val_samples = full_dataset[train_samples_needed:train_samples_needed + val_samples_needed]
    train_dataset = BERTDataset(train_samples, train_config.vocab_size)
    val_dataset = BERTDataset(val_samples, train_config.vocab_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=4 * train_config.gpus, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config.micro_batch_size, shuffle=False, num_workers=4 * train_config.gpus, pin_memory=True)

    # sanity checks
    ## batch size is divisible by micro batch size
    assert train_config.batch_size % train_config.micro_batch_size == 0, f"Batch size {train_config.batch_size} must be divisible by micro batch size {train_config.micro_batch_size}"
    ## data sequence length is equal to model sequence length
    assert train_dataset[0].shape[1] == bert_config.max_seq_len, 
        f"Data sequence length {train_dataset[0].shape[1]} must be equal to model sequence length {config.max_seq_len}"
    ## dataset vocab size is less than or equal to model vocab size
    train_config.vocab_size 
    
    # group parameters by weight decay
    # https://github.com/karpathy/nanoGPT/blob/master/model.py
    # https://github.com/JonasGeiping/cramming/blob/main/cramming/backend/utils.py
    # https://huggingface.co/transformers/v3.3.1/training.html

    # initialize optimizer and scheduler


    optimizer = torch.optim.AdamW()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=train_config.max_lr, 
        total_steps=train_config.total_steps,
        anneal_strategy='linear',
        pct_start=0.5,
        three_phase=False
    )

    # training loop
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # make sure to use optimizer.zero_grad with set_to_none=True
    

if __name__ == '__main__':
    fire.Fire(train_bert)