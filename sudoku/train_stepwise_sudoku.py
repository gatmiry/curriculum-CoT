import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import wandb
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== Wandb ==============

def wandb_login():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, '.wandb_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    wandb.login(key=line)
                    return True
    return False

# ============== Sudoku Generation & Validation ==============

def generate_solved_sudoku(t):
    n = t * t
    grid = np.zeros((n, n), dtype=int)
    
    def is_valid(grid, row, col, num):
        if num in grid[row]: return False
        if num in grid[:, col]: return False
        box_row, box_col = (row // t) * t, (col // t) * t
        if num in grid[box_row:box_row+t, box_col:box_col+t]: return False
        return True
    
    def solve(grid):
        for i in range(n):
            for j in range(n):
                if grid[i, j] == 0:
                    nums = list(range(1, n + 1))
                    random.shuffle(nums)
                    for num in nums:
                        if is_valid(grid, i, j, num):
                            grid[i, j] = num
                            if solve(grid): return True
                            grid[i, j] = 0
                    return False
        return True
    
    solve(grid)
    return grid

def is_valid_placement(grid, t, loc, num):
    """Check if placing num at loc creates valid partial sudoku."""
    n = t * t
    row, col = loc // n, loc % n
    
    # Check row
    for j in range(n):
        if j != col and grid[row * n + j] == num:
            return False
    # Check column
    for i in range(n):
        if i != row and grid[i * n + col] == num:
            return False
    # Check box
    box_row, box_col = (row // t) * t, (col // t) * t
    for i in range(box_row, box_row + t):
        for j in range(box_col, box_col + t):
            if (i != row or j != col) and grid[i * n + j] == num:
                return False
    return True

def is_puzzle_solvable(grid, t):
    """Check if puzzle has at least one valid solution."""
    n = t * t
    grid_2d = grid.reshape(n, n).copy()
    
    def is_valid(g, row, col, num):
        if num in g[row]: return False
        if num in g[:, col]: return False
        br, bc = (row // t) * t, (col // t) * t
        if num in g[br:br+t, bc:bc+t]: return False
        return True
    
    def solve(g):
        for i in range(n):
            for j in range(n):
                if g[i, j] == 0:
                    for num in range(1, n + 1):
                        if is_valid(g, i, j, num):
                            g[i, j] = num
                            if solve(g): return True
                            g[i, j] = 0
                    return False
        return True
    
    return solve(grid_2d)

def create_puzzle(solved_grid, num_empty):
    n = solved_grid.shape[0]
    puzzle = solved_grid.copy()
    positions = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(positions)
    for i, j in positions[:num_empty]:
        puzzle[i, j] = 0
    return puzzle

def generate_batch(t, batch_size, max_empty=None):
    """Generate puzzles (all valid/solvable)."""
    n = t * t
    total_cells = n * n
    puzzles = []
    is_valid = []
    
    for _ in range(batch_size):
        solved = generate_solved_sudoku(t)
        if max_empty is not None:
            num_empty = random.randint(1, max(1, max_empty))
        else:
            num_empty = random.randint(1, total_cells // 2)
        puzzle = create_puzzle(solved, num_empty)
        puzzles.append(puzzle.flatten())
        is_valid.append(True)
    
    puzzles_tensor = torch.tensor(np.array(puzzles), dtype=torch.long, device=device)
    is_valid_tensor = torch.tensor(is_valid, dtype=torch.bool, device=device)
    return puzzles_tensor, is_valid_tensor

# ============== Model ==============

def precompute_rope_freqs(head_dim, max_seq_len, theta=10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(pos, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    T = x.shape[2]
    head_dim = x.shape[-1]
    cos, sin = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        cos, sin = precompute_rope_freqs(self.head_dim, block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)
        attn = (q @ k.transpose(-1, -2)) * (k.size(-1) ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        return self.c_proj((attn @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class StepwiseSudokuGPT(nn.Module):
    def __init__(self, t, n_layers, n_heads, n_embd, max_cot_tokens, truncate_backprop=False, backprop_steps=1):
        super().__init__()
        self.t = t
        self.n = t * t
        self.num_cells = self.n * self.n
        self.vocab_size = self.n + 1
        self.n_embd = n_embd
        self.max_cot = max_cot_tokens
        self.truncate_backprop = truncate_backprop
        self.backprop_steps = backprop_steps
        
        block_size = self.num_cells + max_cot_tokens + 2
        self.wte = nn.Embedding(self.vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.special_emb = nn.Parameter(torch.randn(n_embd) * 0.02)
        
        # Validity head: 2 classes (No=0, Yes=1)
        self.validity_heads = nn.ModuleList([
            nn.Linear(n_embd, 2, bias=False) for _ in range(max_cot_tokens + 1)
        ])
        # Location heads: num_cells classes
        self.location_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(n_embd, self.num_cells, bias=False) for _ in range(k)
            ]) for k in range(max_cot_tokens + 1)
        ])
        # Number heads: n classes (1 to n, 0-indexed as 0 to n-1)
        self.number_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(n_embd, self.n, bias=False) for _ in range(k)
            ]) for k in range(max_cot_tokens + 1)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def get_hidden(self, idx, num_cot_tokens):
        """Generate CoT tokens and return final hidden state."""
        B = idx.size(0)
        x = self.wte(idx)
        
        detach_until = num_cot_tokens + 1 - self.backprop_steps if self.truncate_backprop else 0
        cot_embs = torch.empty(B, 0, self.n_embd, device=idx.device)
        
        for i in range(num_cot_tokens):
            if self.truncate_backprop and i < detach_until:
                cot_embs = cot_embs.detach()
            special = self.special_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            combined = torch.cat([x, cot_embs, special], dim=1)
            for block in self.blocks:
                combined = block(combined)
            combined = self.ln_f(combined)
            cot_embs = torch.cat([cot_embs, combined[:, -1:, :]], dim=1)
        
        if self.truncate_backprop and num_cot_tokens < detach_until:
            cot_embs = cot_embs.detach()
        
        special = self.special_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        combined = torch.cat([x, cot_embs, special], dim=1)
        for block in self.blocks:
            combined = block(combined)
        hidden = self.ln_f(combined)
        return hidden[:, -1, :]

    def forward_rl(self, puzzles, num_cot_tokens, wt=1.0, is_valid_puzzle=None):
        """
        RL-style forward: model generates autoregressively, loss based on validity.
        
        Args:
            puzzles: (B, num_cells) input puzzles with 0s for empty cells
            num_cot_tokens: number of CoT tokens / max steps
            wt: penalty weight for invalid moves
            is_valid_puzzle: (B,) bool tensor indicating if each puzzle is solvable
        
        Returns:
            loss, metrics_dict
        """
        B = puzzles.size(0)
        hidden = self.get_hidden(puzzles, num_cot_tokens)
        
        # Check if puzzles are solvable
        if is_valid_puzzle is None:
            is_valid_puzzle = torch.ones(B, dtype=torch.bool, device=puzzles.device)
        
        # Validity prediction
        validity_logits = self.validity_heads[num_cot_tokens](hidden)  # (B, 2)
        validity_log_probs = torch.log_softmax(validity_logits, dim=-1)
        
        # Initialize loss and tracking
        total_loss = torch.zeros(B, device=puzzles.device)
        num_valid_steps = torch.zeros(B, device=puzzles.device)
        num_total_steps = torch.zeros(B, device=puzzles.device)
        completed = torch.zeros(B, dtype=torch.bool, device=puzzles.device)
        made_mistake = torch.zeros(B, dtype=torch.bool, device=puzzles.device)
        
        # Copy puzzles to track current state
        current_grids = puzzles.clone()
        
        # For invalid puzzles: loss = log P(Yes) - log P(No)
        # Minimizing this decreases P(Yes) and increases P(No)
        invalid_puzzle_mask = ~is_valid_puzzle
        if invalid_puzzle_mask.any():
            total_loss[invalid_puzzle_mask] = (validity_log_probs[invalid_puzzle_mask, 1] - 
                                                validity_log_probs[invalid_puzzle_mask, 0])
            completed[invalid_puzzle_mask] = True
        
        # For valid puzzles: start with validity loss (should predict Yes)
        valid_puzzle_mask = is_valid_puzzle & ~completed
        validity_ce = -validity_log_probs[:, 1]  # -log P(Yes)
        
        # Separate tracking for correct losses (to average) and penalty (not averaged)
        correct_losses = [[] for _ in range(B)]  # Will be averaged
        penalty_loss = [None for _ in range(B)]  # Just added with wt, not averaged
        
        if valid_puzzle_mask.any():
            for b in range(B):
                if valid_puzzle_mask[b]:
                    correct_losses[b].append(validity_ce[b])
        
        # Get heads for this cot level
        loc_heads = self.location_heads[num_cot_tokens]
        num_heads = self.number_heads[num_cot_tokens]
        
        max_steps = min(num_cot_tokens, len(loc_heads))
        
        # Track dense penalty for filled cells (accumulated across steps)
        total_filled_penalty = torch.zeros(B, device=puzzles.device)
        
        for step in range(max_steps):
            # Skip if all samples are done
            active = valid_puzzle_mask & ~completed & ~made_mistake
            if not active.any():
                break
            
            # Predict location
            loc_logits = loc_heads[step](hidden)  # (B, num_cells)
            loc_log_probs = torch.log_softmax(loc_logits, dim=-1)
            pred_loc = loc_logits.argmax(dim=-1)  # (B,)
            
            # DIRECT SUPERVISION on empty cell locations
            # Cross-entropy: -log P(empty_cell) for each empty cell
            empty_mask = (current_grids == 0)  # (B, num_cells) - True where cell is empty
            # Sum -log P(empty) over all empty cells (encourages model to assign high prob to empty cells)
            location_supervision = -(loc_log_probs * empty_mask.float()).sum(dim=-1)  # (B,)
            total_filled_penalty = total_filled_penalty + location_supervision
            
            # Predict number
            num_logits = num_heads[step](hidden)  # (B, n)
            num_log_probs = torch.log_softmax(num_logits, dim=-1)
            pred_num = num_logits.argmax(dim=-1) + 1  # (B,) 1-indexed
            
            for b in range(B):
                if not active[b]:
                    continue
                
                loc_b = pred_loc[b].item()
                num_b = pred_num[b].item()
                grid_b = current_grids[b].cpu().numpy()
                
                num_total_steps[b] += 1
                
                # Check if location is valid (empty cell)
                if grid_b[loc_b] != 0:
                    # Trying to fill non-empty cell - invalid move
                    # To DISCOURAGE: add log P (negative) → minimizing pushes P down
                    penalty_loss[b] = wt * loc_log_probs[b, loc_b]
                    made_mistake[b] = True
                    continue
                
                # Check if number placement is valid
                if not is_valid_placement(grid_b, self.t, loc_b, num_b):
                    # Invalid number for this location - penalize both
                    penalty_loss[b] = wt * (loc_log_probs[b, loc_b] + num_log_probs[b, num_b - 1])
                    made_mistake[b] = True
                    continue
                
                # Valid move! Encourage it by adding -log P (will be averaged)
                valid_loss = -loc_log_probs[b, loc_b] - num_log_probs[b, num_b - 1]
                correct_losses[b].append(valid_loss)
                num_valid_steps[b] += 1
                
                # Update grid
                current_grids[b, loc_b] = num_b
                
                # Check if puzzle is complete
                if (current_grids[b] != 0).all():
                    completed[b] = True
        
        # Compute final loss: average correct losses + penalty (not averaged) + dense filled cell penalty
        for b in range(B):
            # Average of correct losses (if any)
            if len(correct_losses[b]) > 0:
                total_loss[b] = torch.stack(correct_losses[b]).mean()
            # Add penalty loss (not averaged, just added)
            if penalty_loss[b] is not None:
                total_loss[b] = total_loss[b] + penalty_loss[b]
            # Add dense penalty for all filled cells (always applied)
            total_loss[b] = total_loss[b] + wt * total_filled_penalty[b]
        
        # Compute metrics
        validity_acc = (validity_logits.argmax(dim=-1) == is_valid_puzzle.long()).float().mean().item()
        avg_valid_steps = num_valid_steps.mean().item()
        avg_total_steps = num_total_steps.mean().item()
        completion_rate = completed.float().mean().item()
        mistake_rate = made_mistake.float().mean().item()
        
        return total_loss.mean(), {
            'validity_acc': validity_acc,
            'avg_valid_steps': avg_valid_steps,
            'avg_total_steps': avg_total_steps,
            'completion_rate': completion_rate,
            'mistake_rate': mistake_rate
        }

# ============== Training ==============

def is_valid_complete_sudoku(grid, t):
    """Check if grid is a valid complete sudoku."""
    n = t * t
    if (grid == 0).any():
        return False
    grid_2d = grid.reshape(n, n)
    # Check rows
    for i in range(n):
        if len(set(grid_2d[i])) != n:
            return False
    # Check columns
    for j in range(n):
        if len(set(grid_2d[:, j])) != n:
            return False
    # Check boxes
    for bi in range(t):
        for bj in range(t):
            box = grid_2d[bi*t:(bi+1)*t, bj*t:(bj+1)*t].flatten()
            if len(set(box)) != n:
                return False
    return True

def generate_invalid_puzzle(t):
    """Generate an invalid (unsolvable) puzzle by creating conflicts."""
    n = t * t
    total_cells = n * n
    
    # Start with a valid solved grid
    solved = generate_solved_sudoku(t)
    puzzle = solved.copy()
    
    # Create a conflict: put same number twice in a row/col/box
    # Remove some cells first
    positions = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(positions)
    num_empty = random.randint(n, total_cells // 2)
    for i, j in positions[:num_empty]:
        puzzle[i, j] = 0
    
    # Now create a conflict by placing a number that makes it unsolvable
    # Find an empty cell and put a number that conflicts
    empty_cells = [(i, j) for i in range(n) for j in range(n) if puzzle[i, j] == 0]
    if len(empty_cells) >= 2:
        # Pick two empty cells in same row and force them to need same number
        row = random.randint(0, n-1)
        empty_in_row = [j for j in range(n) if puzzle[row, j] == 0]
        if len(empty_in_row) >= 2:
            # Find a number not in the row
            used = set(puzzle[row])
            available = [x for x in range(1, n+1) if x not in used]
            if len(available) >= 1 and len(empty_in_row) > len(available):
                # More empty cells than available numbers = unsolvable
                pass  # Already unsolvable
            else:
                # Force conflict: put same number requirement
                j1, j2 = empty_in_row[0], empty_in_row[1]
                # Fill the column and box of both to leave only one option
                num = available[0] if available else 1
                # Put num in the column of j1 (not in row 'row')
                for i in range(n):
                    if i != row and puzzle[i, j1] == 0:
                        puzzle[i, j1] = num
                        break
    
    return puzzle.flatten()

def print_example(t, puzzles, filled_grids, model, phase, valid_mask):
    """Print an example puzzle and model's prediction."""
    n = t * t
    
    # Find first valid puzzle
    for b in range(puzzles.size(0)):
        if valid_mask[b]:
            puzzle = puzzles[b].cpu().numpy().reshape(n, n)
            filled = filled_grids[b].cpu().numpy().reshape(n, n)
            
            # Get model's predictions for this example
            hidden = model.get_hidden(puzzles[b:b+1], phase)
            loc_logits = model.location_heads[phase][0](hidden)
            num_logits = model.number_heads[phase][0](hidden)
            pred_loc = loc_logits.argmax(dim=-1).item()
            pred_num = num_logits.argmax(dim=-1).item() + 1
            
            # Find actual empty cell location and correct number
            empty_locs = np.where(puzzle.flatten() == 0)[0]
            
            print(f"\n  === Example (t={t}, {n}x{n} grid) ===")
            print(f"  Input puzzle (0=empty):")
            for row in puzzle:
                print(f"    {row}")
            
            # Show location probabilities
            loc_probs = torch.softmax(loc_logits, dim=-1)[0].cpu().numpy()
            
            print(f"\n  Empty cell(s) at: {empty_locs.tolist()}")
            print(f"  Prob of empty cell: {loc_probs[empty_locs[0]]:.4f}")
            print(f"  Model predicts: loc={pred_loc}, num={pred_num}")
            top_locs = np.argsort(loc_probs)[-5:][::-1]
            print(f"  Top 5 location probs: {[(int(l), f'{loc_probs[l]:.3f}') for l in top_locs]}")
            
            # Show number probabilities
            num_probs = torch.softmax(num_logits, dim=-1)[0].cpu().numpy()
            print(f"  Number probs (1-{n}): {[f'{p:.3f}' for p in num_probs]}")
            
            print(f"\n  Filled result:")
            for row in filled:
                print(f"    {row}")
            print()
            break

@torch.no_grad()
def evaluate(model, t, phase, eval_batch_size=100, wt=1.0):
    """
    Evaluate solve accuracy:
    - For valid puzzles: did model complete it to a valid sudoku?
    - For invalid puzzles: did model correctly predict No?
    """
    model.eval()
    n = t * t
    num_cells = n * n
    
    # Generate mostly valid puzzles, some invalid
    num_invalid = max(1, eval_batch_size // 10)  # 10% invalid
    num_valid = eval_batch_size - num_invalid
    
    valid_puzzles, _ = generate_batch(t, num_valid, max_empty=phase)
    invalid_puzzles = torch.stack([
        torch.tensor(generate_invalid_puzzle(t), dtype=torch.long, device=device)
        for _ in range(num_invalid)
    ])
    
    all_puzzles = torch.cat([valid_puzzles, invalid_puzzles], dim=0)
    is_valid_puzzle = torch.cat([
        torch.ones(num_valid, dtype=torch.bool, device=device),
        torch.zeros(num_invalid, dtype=torch.bool, device=device)
    ])
    
    # Get hidden state
    hidden = model.get_hidden(all_puzzles, phase)
    
    # Validity prediction
    validity_logits = model.validity_heads[phase](hidden)
    pred_valid = validity_logits.argmax(dim=-1) == 1  # 1 = Yes
    
    # Track solve accuracy
    correct = torch.zeros(eval_batch_size, dtype=torch.bool, device=device)
    
    # For invalid puzzles: correct if predicted No
    invalid_mask = ~is_valid_puzzle
    correct[invalid_mask] = ~pred_valid[invalid_mask]  # Correct if predicted No (0)
    
    # For valid puzzles: need to check if model solves correctly
    valid_mask = is_valid_puzzle
    current_grids = all_puzzles.clone()
    
    loc_heads = model.location_heads[phase]
    num_heads = model.number_heads[phase]
    max_steps = min(phase, len(loc_heads))
    
    solved_correctly = torch.zeros(eval_batch_size, dtype=torch.bool, device=device)
    
    for step in range(max_steps):
        loc_logits = loc_heads[step](hidden)
        pred_loc = loc_logits.argmax(dim=-1)
        
        num_logits = num_heads[step](hidden)
        pred_num = num_logits.argmax(dim=-1) + 1  # 1-indexed
        
        for b in range(eval_batch_size):
            if not valid_mask[b]:
                continue
            loc_b = pred_loc[b].item()
            num_b = pred_num[b].item()
            
            # Only fill if cell is empty
            if current_grids[b, loc_b] == 0:
                current_grids[b, loc_b] = num_b
    
    # Check if valid puzzles are solved correctly
    for b in range(eval_batch_size):
        if valid_mask[b]:
            grid_np = current_grids[b].cpu().numpy()
            if is_valid_complete_sudoku(grid_np, t):
                solved_correctly[b] = True
    
    correct[valid_mask] = solved_correctly[valid_mask]
    
    solve_accuracy = correct.float().mean().item()
    valid_solve_acc = solved_correctly[valid_mask].float().mean().item() if valid_mask.any() else 0.0
    invalid_reject_acc = correct[invalid_mask].float().mean().item() if invalid_mask.any() else 0.0
    
    # Print example for phase 1
    if phase == 1:
        print_example(t, all_puzzles, current_grids, model, phase, valid_mask)
    
    # Also compute loss for logging (only on valid puzzles)
    valid_is_valid = torch.ones(num_valid, dtype=torch.bool, device=device)
    loss, train_metrics = model.forward_rl(all_puzzles[:num_valid], phase, wt, valid_is_valid)
    
    metrics = {
        'solve_accuracy': solve_accuracy,
        'valid_solve_acc': valid_solve_acc,
        'invalid_reject_acc': invalid_reject_acc,
        **train_metrics
    }
    
    return loss.item(), metrics

def train_phase(model, optimizer, t, phase, max_iterations, batch_size,
                eval_interval=100, target_accuracy=0.9, remember_rate=0.2,
                global_step=0, training_history=None, eval_batch_size=100, 
                use_wandb=True, wt=1.0):
    model.train()
    
    iter_num = 0
    eval_loss = float('inf')
    eval_solve_acc = 0.0  # Start at 0% solve accuracy
    train_losses = []
    
    pbar = tqdm(total=max_iterations, desc=f"Phase {phase}")
    
    # Advance phase when solve accuracy exceeds target
    while eval_solve_acc < target_accuracy and iter_num < max_iterations:
        # Multilevel training
        if remember_rate > random.random():
            current_cot = np.random.randint(1, phase + 1)
        else:
            current_cot = phase
        
        puzzles, is_valid = generate_batch(t, batch_size, max_empty=current_cot)
        
        optimizer.zero_grad()
        loss, _ = model.forward_rl(puzzles, current_cot, wt, is_valid)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'eval': f'{eval_loss:.4f}', 'cot': current_cot})
        pbar.update(1)
        
        if (iter_num + 1) % eval_interval == 0:
            eval_loss, metrics = evaluate(model, t, phase, eval_batch_size, wt)
            eval_solve_acc = metrics['solve_accuracy']
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_losses = []
            
            if use_wandb:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "phase": phase,
                    **metrics,
                    "iteration": global_step + iter_num + 1
                })
            
            if training_history is not None:
                training_history.append({
                    "iteration": global_step + iter_num + 1,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "phase": phase,
                    **metrics
                })
            
            print(f"  Phase {phase}, Iter {iter_num + 1}: Loss={eval_loss:.4f}, "
                  f"SolveAcc={metrics['solve_accuracy']:.2%}, ValidSteps={metrics['avg_valid_steps']:.2f}, MistakeRate={metrics['mistake_rate']:.2%}, "
                  f"InvalidReject={metrics['invalid_reject_acc']:.2%}")
            model.train()
            
            if eval_solve_acc >= target_accuracy:
                print(f"  ✓ Target accuracy {target_accuracy:.1%} reached!")
                break
        
        iter_num += 1
    
    pbar.close()
    
    eval_loss, metrics = evaluate(model, t, phase, eval_batch_size, wt)
    print(f"Phase {phase} Complete: SolveAcc={metrics['solve_accuracy']:.2%}, "
          f"ValidSolve={metrics['valid_solve_acc']:.2%} (after {iter_num} iterations)\n")
    
    return eval_loss, metrics, iter_num, global_step + iter_num

# ============== Plotting ==============

def plot_training_curves(training_history, args, save_path='training_curves.png'):
    if not training_history:
        return
    
    iterations = [h['iteration'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    eval_losses = [h['eval_loss'] for h in training_history]
    solve_accs = [h['solve_accuracy'] for h in training_history]
    valid_solve_accs = [h['valid_solve_acc'] for h in training_history]
    invalid_reject_accs = [h['invalid_reject_acc'] for h in training_history]
    phases = [h['phase'] for h in training_history]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(iterations, train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(iterations, eval_losses, 'r-', linewidth=1.5, label='Eval Loss', alpha=0.8)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    phase_changes = [iterations[0]]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append(iterations[i])
    for pc in phase_changes[1:]:
        ax1.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    ax2 = axes[1]
    ax2.plot(iterations, solve_accs, 'g-', linewidth=1.5, label='Solve Accuracy', alpha=0.8)
    ax2.plot(iterations, valid_solve_accs, 'b-', linewidth=1.5, label='Valid Solve Acc', alpha=0.8)
    ax2.plot(iterations, invalid_reject_accs, 'r-', linewidth=1.5, label='Invalid Reject Acc', alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Solve Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    for pc in phase_changes[1:]:
        ax2.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    ax3 = axes[2]
    ax3.step(iterations, phases, 'c-', linewidth=2, label='Phase', alpha=0.8, where='post')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Phase', fontsize=12)
    ax3.set_title('Training Phase', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    if phases:
        ax3.set_yticks(range(1, max(phases) + 1))
    
    n = args.t * args.t
    fig.suptitle(f'Stepwise Sudoku {n}×{n} (t={args.t}) RL Training, wt={args.wt}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()

def save_training_data(training_history, results, args, save_path='training_data.json'):
    data = {'config': vars(args), 'training_history': training_history, 'phase_results': results}
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Training data saved to {save_path}")

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='Train Stepwise Sudoku with RL-style loss')
    parser.add_argument('--t', type=int, default=2, help='Sudoku parameter (grid is t² × t²)')
    parser.add_argument('--max_cot', type=int, default=8, help='Maximum CoT tokens (phases)')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--iterations_per_phase', type=int, default=5000, help='Max iterations per phase')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Evaluation batch size')
    parser.add_argument('--target_accuracy', type=float, default=0.9, help='Target solve accuracy to advance phase')
    parser.add_argument('--truncate_backprop', action='store_true', default=True, help='Truncate backprop')
    parser.add_argument('--backprop_steps', type=int, default=1, help='Backprop steps')
    parser.add_argument('--remember_rate', type=float, default=0.2, help='Rate to train on previous phases')
    parser.add_argument('--wt', type=float, default=1.0, help='Penalty weight for invalid moves')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plots_dir', type=str, default='plots_stepwise', help='Directory for plots')
    parser.add_argument('--plot_data_dir', type=str, default='plot_data_stepwise', help='Directory for data')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    t = args.t
    n = t * t
    
    print(f"Stepwise Sudoku RL: {n}×{n} (t={t})")
    print(f"Max CoT / phases: {args.max_cot}")
    print(f"Penalty weight (wt): {args.wt}")
    print(f"Device: {device}")
    print()
    
    use_wandb = not args.no_wandb
    if use_wandb:
        if wandb_login():
            wandb.init(project="curriculum-cot-sudoku-stepwise-rl", config=vars(args))
        else:
            try:
                wandb.init(project="curriculum-cot-sudoku-stepwise-rl", config=vars(args), mode="offline")
            except:
                use_wandb = False
    
    model = StepwiseSudokuGPT(
        t=t,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        max_cot_tokens=args.max_cot,
        truncate_backprop=args.truncate_backprop,
        backprop_steps=args.backprop_steps
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    results = []
    training_history = []
    global_step = 0
    
    for phase in range(1, args.max_cot + 1):
        print(f"{'='*60}")
        print(f"PHASE {phase}: RL Training with {phase} CoT tokens, ≤{phase} empty cells")
        print(f"{'='*60}")
        
        eval_loss, metrics, iterations, global_step = train_phase(
            model=model,
            optimizer=optimizer,
            t=t,
            phase=phase,
            max_iterations=args.iterations_per_phase,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            target_accuracy=args.target_accuracy,
            remember_rate=args.remember_rate,
            global_step=global_step,
            training_history=training_history,
            eval_batch_size=args.eval_batch_size,
            use_wandb=use_wandb,
            wt=args.wt
        )
        
        results.append({'phase': phase, 'loss': eval_loss, **metrics, 'iterations': iterations})
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    total_iterations = 0
    for r in results:
        print(f"Phase {r['phase']}: SolveAcc={r['solve_accuracy']:.2%}, "
              f"ValidSolve={r['valid_solve_acc']:.2%}, InvalidReject={r['invalid_reject_acc']:.2%}, "
              f"Iters={r['iterations']}")
        total_iterations += r['iterations']
    print(f"{'='*60}")
    print(f"Total iterations: {total_iterations}")
    
    # Save outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = args.plots_dir if os.path.isabs(args.plots_dir) else os.path.join(script_dir, args.plots_dir)
    plot_data_dir = args.plot_data_dir if os.path.isabs(args.plot_data_dir) else os.path.join(script_dir, args.plot_data_dir)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_training_curves(training_history, args, save_path=os.path.join(plots_dir, f'training_curves_{timestamp}.png'))
    save_training_data(training_history, results, args, save_path=os.path.join(plot_data_dir, f'training_data_{timestamp}.json'))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        't': t,
        'args': args,
        'results': results,
        'training_history': training_history
    }, os.path.join(script_dir, 'stepwise_sudoku_model.pt'))
    print(f"\nModel saved to stepwise_sudoku_model.pt")
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
