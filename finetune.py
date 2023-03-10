# The code here supports finetuning BERT on downstream tasks that can
# be framed as sentence or sentence pair classification (or regression).
import torch
import torch.nn as nn
import yaml
from dataclasses import dataclass

@dataclass
class FineTuneConfig:
    tasks: list
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


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
        input_ids = self.tokenizer(self.sentence1s[idx], max_length=self.max_len, truncation=True, padding=False)['input_ids']
        if self.sentence2s is not None:
            input_ids.append(self.tokenizer.sep_token_id)
            input_ids.extend(self.tokenizer(self.sentence2s[idx], max_length=self.max_len, truncation=True, padding=False)['input_ids'])
        input_ids.append(self.tokenizer.sep_token_id)
        # truncate if too long
        input_ids = input_ids[:self.max_len]
        # pad if too short
        input_ids.extend([self.tokenizer.pad_token_id] * (self.max_len - len(input_ids)))
        # attention mask - 1 for tokens that are not padding, 0 for padding tokens
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]
        label = self.labels[idx]
        if self.num_classes == 1:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return (
            torch.LongTensor(input_ids), 
            label, 
            torch.BoolTensor(attention_mask)
        )

# Supports binary, multiclass, and regression tasks
class BERTForFineTuning(nn.Module):
    def __init__(self, bert, num_classes, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(bert.d_model, num_classes)
    def forward(self, input_ids, targets, attention_mask):
        outputs = self.bert(input_ids, mask=attention_mask) # (bsz, seq_len, hidden_size)
        print("output_shape:", outputs.shape)
        pooled = torch.mean(outputs, dim=1) # (bsz, hidden_size)
        logits = self.output_head(self.dropout(pooled)) # (bsz, num_classes)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss