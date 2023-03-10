import yaml
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TransformerBlock, LayerNorm
import bitsandbytes as bnb
import transformers
# TODO: add dropout for finetuning

@dataclass
class BERTConfig:
    model: str
    vocab_size: int
    n_layers: int
    max_seq_len: int
    d_model: int
    d_qkv: int
    n_heads: int
    ffn_geglu: bool
    ffn_hidden_size: int
    tie_weights: bool
    dropout: float
    linear_bias: bool
    layernorm_bias: bool
    initializer_range: float = 0.02
    checkpoint_path: str = None

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_dict(self):
        return self.__dict__


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.token_emb = bnb.nn.StableEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(
            config.d_model, 
            config.d_qkv, 
            config.n_heads, 
            config.ffn_geglu,
            config.ffn_hidden_size,
            dropout=config.dropout
        ) for _ in range(config.n_layers)])
        self.norm = LayerNorm(config.d_model, weight=True, bias=False)
        self.fc = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.initializer_range = config.initializer_range
        self.tie_weights = config.tie_weights

        if config.tie_weights:
            self.fc.weight = self.token_emb.weight

        n_params = (sum(p.numel() for p in self.token_emb.parameters()) +
                    self.pos_emb.numel() +
                    sum(p.numel() for p in self.blocks.parameters()) +
                    sum(p.numel() for p in self.norm.parameters()) +
                    sum(p.numel() for p in self.fc.parameters())
        )
        if config.tie_weights:
            n_params -= self.fc.weight.numel()
        print("Number of parameters: ~%.0fM" % (n_params/1e6,))
        
        # Initialize model parameters
        self.apply(self._init_weights)

    def load_weights_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if name in ckpt:
                param.data.copy_(ckpt[name])
            else:
                print(f"Parameter {name} not found in checkpoint.")
        if self.tie_weights:
            self.fc.weight = self.token_emb.weight
        del ckpt

    # Borrowed from Karpathy's nanoGPT 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (LayerNorm)) or isinstance(module, (nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # Adapted from Karpathy's nanoGPT
    def get_optim_groups(self, weight_decay):
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                no_decay.add(name)
            elif name.endswith(".norm.weight"):
                no_decay.add(name)
            elif "pos_emb" in name or "token_emb" in name:
                no_decay.add(name)
            else:
                decay.add(name)
        decay = sorted(list(decay))
        no_decay = sorted(list(no_decay))
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        return [
            {'params': [param_dict[pn] for pn in decay], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in no_decay], 'weight_decay': 0.0},
        ] 

    # Targets must be masked with -100 at non-masked indices that should be ignored
    def forward(self, X, targets=None, mask=None):
        token_embs = self.token_emb(X)
        pos_embs = self.pos_emb[:, :X.shape[1], :]
        X = self.token_emb(X) + self.pos_emb[:, :X.shape[1], :]
        # X = self.emb_norm(X) ==> bnb.nn.StableEmbedding already has LayerNorm
        X = self.emb_dropout(X)
        for block in self.blocks:
            X = block(X, mask=mask)
        

        if targets is not None:
            logits = self.fc(self.norm(X))
            loss = F.cross_entropy(
                torch.flatten(logits, start_dim=0, end_dim=1), 
                torch.flatten(targets)
            )
            return loss
        else:
            return X

class HuggingFaceRoBERTa(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.roberta_config = transformers.RobertaPreLayerNormConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            intermediate_size=config.ffn_hidden_size,
            hidden_act="gelu",
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            max_position_embeddings=config.max_seq_len + 2,
            type_vocab_size=1,
            initializer_range=config.initializer_range,
        )
        self.roberta = transformers.RobertaPreLayerNormForMaskedLM(self.roberta_config)
        n_params = (sum(p.numel() for p in self.roberta.parameters()))
        print("Number of parameters: ~%.0fM" % (n_params / 1e6,))

    def get_optim_groups(self, weight_decay):
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                no_decay.add(name)
            elif "LayerNorm" in name:
                no_decay.add(name)
            elif "roberta.embeddings" in name:
                no_decay.add(name)
            else:
                decay.add(name)
        decay = sorted(list(decay))
        no_decay = sorted(list(no_decay))
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        return [
            {'params': [param_dict[pn] for pn in decay], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in no_decay], 'weight_decay': 0.0},
        ]   
    
    def forward(self, X, targets=None):
        logits = self.roberta(X).logits
        if targets is not None:
            loss = F.cross_entropy(
                torch.flatten(logits, start_dim=0, end_dim=1), 
                torch.flatten(targets)
            )
            return loss
        else:
            return logits