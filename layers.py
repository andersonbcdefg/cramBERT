import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from Andrej Karpathy's "nanoGPT"
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional weight & bias. PyTorch doesn't support simply bias=False,
    this module allows both, neither, or just one of [weight, bias]. """

    def __init__(self, d_model, weight=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model)) if weight else None
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, (self.d_model,), self.weight, self.bias, 1e-5)

class PreNormAndAdd(nn.Module):
    def __init__(self, d_model, sublayer):
        super().__init__()
        self.norm = LayerNorm(d_model, bias=False)
        self.sublayer = sublayer
    
    def forward(self, X):
        return X + self.sublayer(self.norm(X))

class Attention(nn.Module):
    def __init__(self, d_model, d_qkv, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_qkv = d_qkv
        self.scale = d_qkv**0.5
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(d_model, 3 * d_qkv * n_heads, bias=False)
        self.out_proj = nn.Linear(d_qkv * n_heads, d_model, bias=False)

    def forward(self, X):
        b, s, _ = X.shape
        q, k, v = self.to_qkv(X).view(b, s, self.n_heads, self.d_qkv, 3).unbind(dim=-1)
        attn_scores = q.transpose(1, 2) @ k.permute((0, 2, 3, 1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v.transpose(1, 2)
        return self.out_proj(attn_out.transpose(1, 2).flatten(-2))

# Easy mode: einsum (used as reference for correctness)
class EinsumAttention(nn.Module):
    def __init__(self, d_model, d_qkv, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_qkv = d_qkv
        self.scale = d_qkv**0.5
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(d_model, 3 * d_qkv * n_heads, bias=False)
        self.out_proj = nn.Linear(d_qkv * n_heads, d_model, bias=False)

    def forward(self, X):
        b, s, _ = X.shape
        q, k, v = self.to_qkv(X).view(b, s, self.n_heads, self.d_qkv, 3).unbind(dim=-1)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        return self.out_proj(attn_out.flatten(-2))

class FFN(nn.Module):
    def __init__(self, geglu, d_model, hidden_size):
        super().__init__()
        self.geglu = geglu
        if geglu:
            self.fc1 = nn.Linear(d_model, 2 * hidden_size, bias=False)
        else: 
            self.fc1 = nn.Linear(d_model, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, X):
        if self.geglu:
            a, b = self.fc1(X).chunk(2, dim=-1)
            return self.fc2(a * F.gelu(b, approximate='tanh'))
        else:
            return self.fc2(F.gelu(self.fc1(X), approximate='tanh'))



class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_qkv, n_heads, ffn_geglu, ffn_hidden_size):
        super().__init__()
        self.attn = PreNormAndAdd(d_model, Attention(d_model, d_qkv, n_heads))
        self.ffn =  PreNormAndAdd(d_model, FFN(ffn_geglu, d_model, ffn_hidden_size))

    def forward(self, X):
        return self.ffn(self.attn(X))