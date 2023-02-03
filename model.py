import yaml
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TransformerBlock
# TODO: add dropout for finetuning

@dataclass
class BERTConfig:
    vocab_size: int
    n_layers: int
    max_seq_len: int
    d_model: int
    d_qkv: int
    n_heads: int
    ffn_geglu: bool
    ffn_hidden_size: int
    tie_weights: bool

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(
            config.d_model, 
            config.d_qkv, 
            config.n_heads, 
            config.ffn_geglu,
            config.ffn_hidden_size) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.fc.weight = self.token_emb.weight

        n_params = (sum(p.numel() for p in self.token_emb.parameters()) +
                    self.pos_emb.numel() +
                    sum(p.numel() for p in self.blocks.parameters()) +
                    sum(p.numel() for p in self.ln.parameters()) +
                    sum(p.numel() for p in self.fc.parameters())
        )
        if config.tie_weights:
            n_params -= self.fc.weight.numel()
        print("Number of parameters: ~%.0fM" % (n_params/1e6,))
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, name):
        raise NotImplementedError

    # Borrowed from nanoGPT
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        

    def forward(self, X, targets=None, mask=None):
        token_embs = self.token_emb(X)
        pos_embs = self.pos_emb[:, :X.shape[1], :]
        X = self.token_emb(X) + self.pos_emb[:, :X.shape[1], :]
        for block in self.blocks:
            X = block(X)
        logits = self.fc(self.ln(X))
        if targets is not None:
            if mask is None:
                raise ValueError("Mask is required when targets are provided.")
            targets.masked_fill_(~mask, -100)
            # what's the loss for totally uniform predictions?
            # uniform = torch.full(logits.shape, 1/self.vocab_size, device=logits.device)
            # logits = logits * 0.0000001 + uniform * 0.9999999
            loss = F.cross_entropy(
                torch.flatten(logits, start_dim=0, end_dim=1), 
                torch.flatten(targets)
            )
            return loss
        else:
            return logits