"""
Routine to download GLUE data adapted from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
Eval adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
"""
import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile
import fire
from model import BERTConfig, BERT
from dataclasses import dataclass
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
from data import load_tokenizer

@dataclass
class FinetuneConfig:
    tasks: list
    num_epochs: int
    batch_size: int
    learning_rate: float
    dropout: float

class FineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, sentence1s, sentence2s, labels, num_classes, tokenizer, max_len=128):
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_ids = self.tokenizer(self.sentence1s[idx], max_length=self.max_len, padding=False)['input_ids']
        if self.sentence2s is not None:
            input_ids.extend(self.tokenizer.sep_token_id)
            input_ids.extend(self.tokenizer(self.sentence2s[idx], max_length=self.max_len, padding=False)['input_ids'])
        input_ids.extend(self.tokenizer.sep_token_id)
        # truncate if too long
        input_ids = input_ids[:self.max_len]
        # pad if too short
        input_ids.extend([self.tokenizer.pad_token_id] * (self.max_len - len(input_ids)))
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]
        label = self.labels[idx]
        if self.num_classes == 1:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'label': label
        }



# Supports binary, multiclass, and regression tasks
class BERTForFineTuning(nn.Module):
    def __init__(self, bert, num_classes, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(bert.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # (bsz, seq_len, hidden_size)
        pooled = torch.mean(outputs, dim=1) # (bsz, hidden_size)
        logits = self.classifier(self.dropout(pooled_output)) # (bsz, num_classes)
        return logits

def download_glue(metadata_file="glue_metadata.yaml", data_dir="glue"):
    metadata = yaml.safe_load(open(metadata_file, 'r'))
    for task in metadata['tasks']:
        print(f"Downloading and extracting {task}...")
        data_file = f"{task}.zip"
        urllib.request.urlretrieve(metadata['task_urls'][task], data_file)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(data_file)
        if task == "CoLA":
            # add header to CoLA train, dev
            cola_train_df = pd.read_csv(os.path.join(data_dir, "CoLA", "train.tsv"), sep='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
            cola_eval_df = pd.read_csv(os.path.join(data_dir, "CoLA", "dev.tsv"), sep='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
            cola_train_df.to_csv(os.path.join(data_dir, "CoLA", "train.tsv"), sep='\t', index=False)
            cola_eval_df.to_csv(os.path.join(data_dir, "CoLA", "dev.tsv"), sep='\t', index=False)
            print("Added header to CoLA train & dev.")
    print("Done!")

def parse_mnli_line(idx, line):
    items = line.strip().split('\t')
    if len(items) != 12 and len(items) != 16:
        print(f"Invalid line: {idx}", end=', ')
        return None
    premise = items[8].strip()
    hypothesis = items[9].strip()
    gold_label = items[-1].strip()
    if gold_label not in ["entailment", "contradiction", "neutral"]:
        print(f"Invalid gold label: {gold_label}")
        return None
    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': 0 if gold_label == "entailment" else 1 if gold_label == "neutral" else 2
    }

def parse_stsb_line(idx, line):
    items = line.strip().split('\t')
    if len(items) != 10:
        print(f"Invalid line: {idx}")
        return None
    sentence1 = items[7].strip()
    sentence2 = items[8].strip()
    score = items[-1].strip()
    try:
        score = float(score)
    except:
        print(f"Invalid label: {score}")
        return None
    return {
        'sentence1': sentence1,
        'sentence2': sentence2,
        'score': score
    }

def parse_qnli_line(idx, line):
    items = line.strip().split('\t')
    if len(items) != 4:
        print(f"Invalid line: {idx}")
        return None
    question = items[1].strip()
    sentence = items[2].strip()
    label = items[-1].strip()
    if label not in ["entailment", "not_entailment"]:
        print(f"Invalid label: {label}")
        return None
    return {
        'question': question,
        'sentence': sentence,
        'label': 0 if label == "entailment" else 1
    }

def load_mnli(data_dir="glue", split="train"):
    records = []
    with open(os.path.join(data_dir, "MNLI", f"{split}.tsv"), 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        record = parse_mnli_line(idx, line)
        if record is not None:
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

def load_stsb(data_dir="glue", split="train"):
    records = []
    with open(os.path.join(data_dir, "STS-B", f"{split}.tsv"), 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        record = parse_stsb_line(idx, line)
        if record is not None:
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

def load_qnli(data_dir="glue", split="train"):
    records = []
    with open(os.path.join(data_dir, "QNLI", f"{split}.tsv"), 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        record = parse_qnli_line(idx, line)
        if record is not None:
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

def load_rte(data_dir="glue", split="train"):
    df = pd.read_csv(os.path.join(data_dir, "RTE", f"{split}.tsv"), sep='\t', header=0)
    df.label = df.label.apply(lambda x: 0 if x == "entailment" else 1)
    return df

# load data, output (sentence1, sentence2, label) lists
def load_task(task, glue_metadata, data_dir="glue", split="train"):
    if task == "MNLI":
        df = load_mnli(data_dir, split)
    elif task == "STS-B":
        df = load_stsb(data_dir, split)
    elif task == "QNLI":
        df = load_qnli(data_dir, split)
    elif task == "RTE":
        df = load_rte(data_dir, split)
    else:
        df = pd.read_csv(os.path.join(data_dir, task, f"{split}.tsv"), sep='\t', header=0)
    sentence1_key = glue_metadata['task_cols'][task]['sentence1']
    sentence2_key = glue_metadata['task_cols'][task]['sentence2']
    label_key = glue_metadata['task_cols'][task]['label']
    sentence1s = df[sentence1_key].values
    sentence2s = df[sentence2_key].values if sentence2_key is not None else None
    labels = df[label_key].values
    return sentence1s, sentence2s, labels

def test_load_data():
    tokenizer = load_tokenizer()
    metadata = yaml.safe_load(open("glue_metadata.yaml", 'r'))
    for task in metadata['tasks']:
        for split in ["train", "dev", "dev_matched", "dev_mismatched"]:
            if os.path.exists(os.path.join("glue", task, f"{split}.tsv")):
                print(f"Loading {task} {split}...")
                sentence1s, sentence2s, labels = load_task(task, metadata, split=split)
                dataset = FineTuneDataset(sentence1s, sentence2s, labels, 
                    metadata['num_classes'][task], tokenizer, max_len=128)
                print(f"Loaded {len(dataset)} examples")

def finetune_and_eval(base_model, task, finetune_config, glue_metadata):
    # Load data
    train_sentence1s, train_sentence2s, train_labels = load_task(task, glue_metadata, split="train")
    train_dataset = FineTuneDataset(train_sentence1s, train_sentence2s, train_labels, 
            glue_metadata['num_classes'][task], tokenizer, max_len=128)
    train_dataloader = DataLoader(train_dataset, batch_size=finetune_config['batch_size'], shuffle=True)
    if task != "MNLI":
        dev_sentence1s, dev_sentence2s, dev_labels = load_task(task, glue_metadata, split="dev")
        dev_dataset = FineTuneDataset(dev_sentence1s, dev_sentence2s, dev_labels,
            glue_metadata['num_classes'][task], tokenizer, max_len=128)
        dev_dataloader = DataLoader(dev_dataset, batch_size=finetune_config['batch_size'], shuffle=False)
    else:
        dev_matched_sentence1s, dev_matched_sentence2s, dev_matched_labels = load_task(task, glue_metadata, split="dev_matched")
        dev_matched_dataset = FineTuneDataset(dev_matched_sentence1s, dev_matched_sentence2s, dev_matched_labels,
            glue_metadata['num_classes'][task], tokenizer, max_len=128)
        dev_matched_dataloader = DataLoader(dev_matched_dataset, batch_size=finetune_config['batch_size'], shuffle=False)
        dev_mismatched_sentence1s, dev_mismatched_sentence2s, dev_mismatched_labels = load_task(task, glue_metadata, split="dev_mismatched")
        dev_mismatched_dataset = FineTuneDataset(dev_mismatched_sentence1s, dev_mismatched_sentence2s, dev_mismatched_labels,
            glue_metadata['num_classes'][task], tokenizer, max_len=128)
        dev_mismatched_dataloader = DataLoader(dev_mismatched_dataset, batch_size=finetune_config['batch_size'], shuffle=False)
    
    
    # If configs are paths, load them from yaml


def run_glue(model_config, finetune_config):
    # Load model from config

    # Get optimizer and scheduler

    # Train model

    # Evaluate model

# Load model from checkpoint

# Get optimizer and scheduler

# Train model

# Evaluate model

if __name__ == "__main__":
    fire.Fire(test_load_data)