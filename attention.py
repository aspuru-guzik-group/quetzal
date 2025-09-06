# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # qk-layernorm
        if config.qk_norm:
            self.q_norm = LayerNorm(config.n_embd // config.n_head, bias=config.bias)
            self.k_norm = LayerNorm(config.n_embd // config.n_head, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.causal = causal

    def forward(self, x, block_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if hasattr(self, 'q_norm'):
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        if block_mask is None:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=self.causal)
        else:
            y = flex_attention(q, k, v, block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/swiglu_ffn.py
class SwiGLUFFN(nn.Module):
    def __init__(self, channels, bias):
        super().__init__()
        self.w12 = nn.Linear(channels, 2*channels*2, bias=bias)
        self.w3 = nn.Linear(2*channels, channels, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Block(nn.Module):

    def __init__(self, config, causal=True):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config, causal=causal)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.trunk_mlp == "mlp":
            self.mlp = MLP(config)
        elif config.trunk_mlp == "swiglu":
            self.mlp = SwiGLUFFN(config.n_embd, bias=config.bias)

    def forward(self, x, block_mask=None):
        x = x + self.attn(self.ln_1(x), block_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
