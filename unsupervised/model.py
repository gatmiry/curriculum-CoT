import torch.nn as nn
import torch.nn.functional as F
import torch
import math
device = 'cuda'

def precompute_rope_freqs(head_dim, max_seq_len, theta=10000.0, device=None):
    """Precompute cosine and sine frequencies for RoPE."""
    # Compute frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Compute position indices
    t = torch.arange(max_seq_len, device=device)
    # Outer product to get (seq_len, head_dim // 2)
    freqs = torch.outer(t, freqs)
    # Compute cos and sin
    cos = torch.cos(freqs)  # (seq_len, head_dim // 2)
    sin = torch.sin(freqs)  # (seq_len, head_dim // 2)
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary positional embeddings to x.
    
    Args:
        x: (B, n_heads, T, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
    
    Returns:
        rotated x with same shape
    """
    T = x.shape[2]
    head_dim = x.shape[-1]
    
    # Get the cos/sin for the current sequence length
    cos = cos[:T]  # (T, head_dim // 2)
    sin = sin[:T]  # (T, head_dim // 2)
    
    # Split x into two halves along head_dim
    x1 = x[..., : head_dim // 2]  # (B, n_heads, T, head_dim // 2)
    x2 = x[..., head_dim // 2 :]  # (B, n_heads, T, head_dim // 2)
    
    # Reshape cos/sin for broadcasting: (1, 1, T, head_dim // 2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return rotated.type_as(x)

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
        self.head_dim = config.n_embd // config.n_heads
        self.block_size = config.block_size # Store block_size
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Removed bias registration
        self.c_proj.NANOGPT_SCALE_INIT = True
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(self.head_dim, config.block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x, last_k_no_attend=0, window_size=0):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        
        # Apply RoPE to q and k
        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)
        
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
            # No wpe needed - using RoPE for positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        # Create linear head(s)
        self.separate_heads = getattr(config, 'separate_heads', False)
        if self.separate_heads:
            # Create a separate linear head for each number of CoT tokens (0 to cot_length)
            # Index i corresponds to using i CoT tokens
            self.linear_heads = nn.ModuleList([
                nn.Linear(config.n_embd, 1, bias=False) 
                for _ in range(config.cot_length + 1)
            ])
        else:
            self.linear_head = nn.Linear(config.n_embd, 1, bias=False)
        
        # Special embedding vector added at the end of cot_tokens to produce final output
        self.special_embedding = nn.Parameter(torch.randn(config.n_embd) * 0.08)  # 4x larger std
        self.apply(self._init_weights)
        self.config = config # Store config
        self.const_dir_idx = torch.tensor(2, device=device)        

    def _init_weights(self, module):
        std = 0.08  # 4x larger than standard 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def forward_with_hidden(self, idx, cot_embeddings):
        """
        Forward pass that returns the last hidden embedding.
        Uses RoPE for positional encoding (applied in attention layers).
        
        Args:
            idx: input token indices (B, input_length)
            cot_embeddings: continuous cot token embeddings (B, num_cot, n_embd)
        
        Returns:
            hidden_states: the hidden states after layer norm (B, T, n_embd)
        """
        B, input_length = idx.size()
        num_cot = cot_embeddings.size(1)
        T = input_length + num_cot
        assert T <= self.config.block_size, "Block size is too small"
        
        # Concatenate input embeddings with cot embeddings (no additive positional embedding)
        # RoPE handles positional information in the attention layers
        input_embeds = self.transformer.wte(idx)  # (B, input_length, n_embd)
        x = torch.cat([input_embeds, cot_embeddings], dim=1)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def forward(self, idx, cts_tokens, targets=None):
        B, input_length = idx.size()
        T = input_length + cts_tokens.size(1)
        assert T <= self.config.block_size, "Block size is too small"
        
        # No additive positional embeddings - RoPE handles positions in attention
        x = torch.cat([self.transformer.wte(idx), cts_tokens.unsqueeze(2) @ self.transformer.wte(self.const_dir_idx).unsqueeze(0).unsqueeze(0)], dim=1)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        output_vals = self.linear_head(x[:, input_length - 1:-1]).squeeze(-1)
        loss = F.mse_loss(output_vals[:, :], cts_tokens[:, :])
        return output_vals, loss

    def generate(self, idx, num_cot_tokens, target=None):
        """
        Generate cot tokens autoregressively using hidden embeddings.
        
        Args:
            idx: input token indices (B, input_length)
            num_cot_tokens: number of cot tokens to generate
            target: optional target for computing loss (B,) - a real number per sample
        
        Returns:
            output: final output (B,) - a real number per sample
            loss: MSE loss if target is provided, else None
        """
        B, input_length = idx.size()
        
        # Get truncated backprop settings from config
        truncate_backprop = getattr(self.config, 'truncate_backprop', False)
        backprop_steps = getattr(self.config, 'backprop_steps', num_cot_tokens + 1)
        
        # Initialize cot_embeddings as empty
        cot_embeddings = torch.empty(B, 0, self.config.n_embd, device=idx.device)
        
        # Total forward passes = num_cot_tokens + 1 (including final pass)
        # If truncate_backprop is enabled, only backprop through last backprop_steps passes
        # Detach at iteration i if i < (num_cot_tokens + 1 - backprop_steps)
        detach_until = num_cot_tokens + 1 - backprop_steps if truncate_backprop else 0
        
        # Generate cot tokens autoregressively
        for i in range(num_cot_tokens):
            # Truncate backprop: detach cot_embeddings if we're before the backprop window
            if truncate_backprop and i < detach_until:
                cot_embeddings = cot_embeddings.detach()
            
            # Use special_embedding as the placeholder for the next token position
            special_emb = self.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, n_embd)
            current_cot = torch.cat([cot_embeddings, special_emb], dim=1)
            
            # Forward pass to get hidden states
            hidden_states = self.forward_with_hidden(idx, current_cot)
            
            # Get the last hidden embedding (corresponds to the placeholder position)
            next_cot_token = hidden_states[:, -1:, :]  # (B, 1, n_embd)
            
            # Append to cot_embeddings
            cot_embeddings = torch.cat([cot_embeddings, next_cot_token], dim=1)
        
        # Detach before final pass if needed
        if truncate_backprop and num_cot_tokens < detach_until:
            cot_embeddings = cot_embeddings.detach()
        
        # Append special embedding at the end
        special_emb = self.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, n_embd)
        final_cot = torch.cat([cot_embeddings, special_emb], dim=1)
        
        # Final forward pass with cot_embeddings + special embedding
        hidden_states = self.forward_with_hidden(idx, final_cot)
        
        # Get the hidden embedding at the special token position (last position)
        special_hidden = hidden_states[:, -1, :]  # (B, n_embd)
        
        # Apply linear head to get final output
        if self.separate_heads:
            # Use the linear head corresponding to num_cot_tokens
            output = self.linear_heads[num_cot_tokens](special_hidden).squeeze(-1)  # (B,)
        else:
            output = self.linear_head(special_hidden).squeeze(-1)  # (B,)
        
        # Compute loss if target is provided
        loss = None
        if target is not None:
            loss = F.mse_loss(output, target)
        
        return output, loss

class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers = 2
    n_heads = 1
    n_embd = 64
    dropout = 0.01
    cot_length = 0
    separate_heads = False
    truncate_backprop = False
    backprop_steps = 1

    def __init__(self, block_size, vocab_size, n_layers, n_heads, n_embd, dropout, cot_length, 
                 separate_heads=False, truncate_backprop=False, backprop_steps=1):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.cot_length = cot_length
        self.separate_heads = separate_heads
        self.truncate_backprop = truncate_backprop
        self.backprop_steps = backprop_steps

