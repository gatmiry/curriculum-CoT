import torch.nn as nn
import torch.nn.functional as F
import torch
import math
device = 'cuda'
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc_2 = nn.Linear(config.n_embd * 3, config.n_embd)
        self.NANO_SCALE_GPT = True
    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.block_size = config.block_size # Store block_size
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Removed bias registration
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x, last_k_no_attend=0, window_size=0):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        attn = q @ k.transpose(-1,-2) * (k.size(-1)) ** -0.5
        # Create bias mask directly in forward and ensure it's on the correct device
        bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(bias == 0, float('-inf'))
        if last_k_no_attend > 0:
          last_k_no_attend = min(last_k_no_attend, T)
          attn[:,-last_k_no_attend:] = float('-inf')
          if window_size > 0:
            window_size = min(window_size, T - last_k_no_attend)
            i = torch.arange(T).view(T, 1)
            j = torch.arange(T).view(1, T)
            recent_band = (i - j) >= 0 & (i - j) <= window_size & i >= T - last_k_no_attend
            attn[recent_band] = float('-inf')

        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, last_k_no_attend=0, window_size=0):
        x = x + self.c_attn(self.ln_1(x), last_k_no_attend=0, window_size=0)
        return x + self.c_fc(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.linear_head = nn.Linear(config.n_embd, 1, bias=False)
        self.apply(self._init_weights)
        self.config = config # Store config
        self.const_dir_idx = torch.tensor(2, device=device)        

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, idx, cts_tokens, targets=None):
        B, input_length = idx.size()
        T = input_length + cts_tokens.size(1)
        assert T <= self.config.block_size, "Block size is too small"
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device) # Ensure pe is on the same device as idx
        pe_vecs = self.transformer.wpe(pe)
        

        #print( 'shape of self.transformer.wte(idx)', self.transformer.wte(idx).type(), ' shape of cts_tokens', cts_tokens.type())
        #print( 'shape of self.transformer.wte(idx)', self.transformer.wte(self.const_dir_idx).shape, ' shape of cts_tokens', cts_tokens.shape)
        x = pe_vecs + torch.cat([self.transformer.wte(idx), cts_tokens.unsqueeze(2) @ self.transformer.wte(self.const_dir_idx).unsqueeze(0).unsqueeze(0)], dim=1)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        output_vals = self.linear_head(x[:, input_length - 1:-1]).squeeze(-1)
        #print( 'shape of output_vals', output_vals.shape, 'shape of cts_tokens', cts_tokens.shape)
        loss = F.mse_loss(output_vals[:, :], cts_tokens[:, :])
        return output_vals, loss

    def generate(self, idx):
        B, input_length = idx.size()
        cot_tokens = torch.empty(B, 0, self.config.n_embd, device=idx.device)
        for _ in range(self.config.cot_length):
            output_vals, _ = self.forward(idx, torch.cat([cot_tokens, torch.zeros(B, 1, self.config.n_embd, device=idx.device)], dim=1))
            cot_tokens = torch.cat([cot_tokens, torch.outer(output_vals[:, -1], self.transformer.wte(self.const_dir_idx)).unsqueeze(1)], dim=1)
        return output_vals[:, -1]

class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers = 2
    n_heads = 1
    n_embd = 64
    dropout = 0.01
    cot_length = 0

    def __init__(self, block_size, vocab_size, n_layers, n_heads, n_embd, dropout, cot_length):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.cot_length = cot_length