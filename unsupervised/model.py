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
        
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe_vecs = self.transformer.wpe(pe)
        
        # Concatenate input embeddings with cot embeddings
        input_embeds = self.transformer.wte(idx)  # (B, input_length, n_embd)
        x = pe_vecs + torch.cat([input_embeds, cot_embeddings], dim=1)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

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
        
        # Initialize cot_embeddings as empty
        cot_embeddings = torch.empty(B, 0, self.config.n_embd, device=idx.device)
        
        # Generate cot tokens autoregressively
        for _ in range(num_cot_tokens):
            # Use special_embedding as the placeholder for the next token position
            special_emb = self.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, n_embd)
            current_cot = torch.cat([cot_embeddings, special_emb], dim=1)
            
            # Forward pass to get hidden states
            hidden_states = self.forward_with_hidden(idx, current_cot)
            
            # Get the last hidden embedding (corresponds to the placeholder position)
            next_cot_token = hidden_states[:, -1:, :]  # (B, 1, n_embd)
            
            # Append to cot_embeddings
            cot_embeddings = torch.cat([cot_embeddings, next_cot_token], dim=1)
        
        # Append special embedding at the end
        special_emb = self.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, n_embd)
        final_cot = torch.cat([cot_embeddings, special_emb], dim=1)
        
        # Final forward pass with cot_embeddings + special embedding
        hidden_states = self.forward_with_hidden(idx, final_cot)
        
        # Get the hidden embedding at the special token position (last position)
        special_hidden = hidden_states[:, -1, :]  # (B, n_embd)
        
        # Apply linear head to get final output
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

    def __init__(self, block_size, vocab_size, n_layers, n_heads, n_embd, dropout, cot_length):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.cot_length = cot_length

