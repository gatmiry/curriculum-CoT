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
    """Login to wandb using API key from .wandb_key file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, '.wandb_key')
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    wandb.login(key=line)
                    print("Logged in to wandb using .wandb_key file")
                    return True
    
    print("Warning: .wandb_key file not found. Using default wandb authentication.")
    return False

# ============== Sudoku Generation ==============

def generate_solved_sudoku(t):
    """Generate a valid solved t² × t² Sudoku grid."""
    n = t * t
    grid = np.zeros((n, n), dtype=int)
    
    def is_valid(grid, row, col, num):
        if num in grid[row]:
            return False
        if num in grid[:, col]:
            return False
        box_row, box_col = (row // t) * t, (col // t) * t
        if num in grid[box_row:box_row+t, box_col:box_col+t]:
            return False
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
                            if solve(grid):
                                return True
                            grid[i, j] = 0
                    return False
        return True
    
    solve(grid)
    return grid

def create_puzzle(solved_grid, num_empty):
    """Create a puzzle by removing num_empty cells from solved grid."""
    n = solved_grid.shape[0]
    puzzle = solved_grid.copy()
    positions = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(positions)
    for i, j in positions[:num_empty]:
        puzzle[i, j] = 0
    return puzzle

def generate_batch(t, batch_size, max_empty=None):
    """Generate a batch of Sudoku puzzles and solutions.
    
    Args:
        t: Sudoku parameter (grid is t² × t²)
        batch_size: number of puzzles to generate
        max_empty: maximum number of empty cells (for curriculum learning)
                   If None, uses 30-70% of cells as empty
    """
    n = t * t
    total_cells = n * n
    puzzles, solutions = [], []
    
    for _ in range(batch_size):
        solved = generate_solved_sudoku(t)
        if max_empty is not None:
            # Curriculum: 1 to max_empty empty cells
            num_empty = random.randint(1, max(1, max_empty))
        else:
            # Default: 30-70% empty
            num_empty = random.randint(int(total_cells * 0.3), int(total_cells * 0.7))
        puzzle = create_puzzle(solved, num_empty)
        puzzles.append(puzzle.flatten())
        solutions.append(solved.flatten())
    
    return (torch.tensor(np.array(puzzles), dtype=torch.long, device=device),
            torch.tensor(np.array(solutions), dtype=torch.long, device=device))

# ============== Model ==============

def precompute_rope_freqs(head_dim, max_seq_len, theta=10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
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
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class SudokuGPT(nn.Module):
    def __init__(self, t, n_layers, n_heads, n_embd, max_cot_tokens, truncate_backprop=False, backprop_steps=1):
        super().__init__()
        self.t = t
        self.n = t * t
        self.num_cells = self.n * self.n
        self.vocab_size = self.n + 1  # 0 to n
        self.n_embd = n_embd
        self.max_cot = max_cot_tokens
        self.truncate_backprop = truncate_backprop
        self.backprop_steps = backprop_steps
        
        block_size = self.num_cells + max_cot_tokens + 2
        self.wte = nn.Embedding(self.vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.special_emb = nn.Parameter(torch.randn(n_embd) * 0.02)
        
        # Separate output heads for each (cot_length, cell_position)
        # heads[cot][cell] predicts logits for cell at cot tokens
        self.output_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(n_embd, self.vocab_size, bias=False) for _ in range(self.num_cells)
            ]) for _ in range(max_cot_tokens + 1)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward_with_cot(self, idx, num_cot_tokens, targets=None):
        """Forward pass with chain-of-thought tokens and optional truncated backprop."""
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
        
        final_hidden = hidden[:, -1, :]
        
        # Use heads for this specific cot length
        heads = self.output_heads[num_cot_tokens]
        logits = torch.stack([head(final_hidden) for head in heads], dim=1)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss

# ============== Training ==============

@torch.no_grad()
def evaluate(model, t, phase, num_cot_tokens, detect_threshold=0.2, eval_batch_size=100, show_examples=0):
    """Evaluate model and compute r (lowest phase with loss > threshold)."""
    model.eval()
    n = t * t
    
    # Evaluate with puzzles having at most `phase` empty cells
    puzzles, solutions = generate_batch(t, eval_batch_size, max_empty=phase)
    
    losses_by_cot = {}
    logits_phase, targets_phase = None, None
    
    for cot in range(1, phase + 1):
        logits, loss = model.forward_with_cot(puzzles, cot, solutions)
        losses_by_cot[cot] = loss.item()
        if cot == phase:
            logits_phase = logits
            targets_phase = solutions
    
    loss = losses_by_cot[phase]
    preds = logits_phase.argmax(dim=-1)
    
    cell_acc = (preds == targets_phase).float().mean().item()
    puzzle_acc = (preds == targets_phase).all(dim=1).float().mean().item()
    
    mask = (puzzles == 0)
    unknown_acc = (preds[mask] == targets_phase[mask]).float().mean().item() if mask.any() else 1.0
    
    # Compute r: first cot level with loss > threshold
    r = phase
    for cot in range(1, phase + 1):
        if losses_by_cot[cot] > detect_threshold:
            r = cot
            break
    
    if show_examples > 0:
        print(f"\n  Sample predictions (phase {phase}, {phase} CoT tokens):")
        for i in range(min(show_examples, eval_batch_size)):
            puzzle = puzzles[i].cpu().numpy().reshape(n, n)
            pred = preds[i].cpu().numpy().reshape(n, n)
            sol = targets_phase[i].cpu().numpy().reshape(n, n)
            correct = (pred == sol).all()
            print(f"  Example {i+1}: {'✓' if correct else '✗'}")
    
    return loss, cell_acc, puzzle_acc, unknown_acc, r

def train_phase(model, optimizer, t, phase, max_iterations, batch_size,
                eval_interval=100, target_loss=0.5, remember_rate=0.2,
                detect_threshold=0.2, global_step=0, training_history=None, eval_batch_size=100,
                use_wandb=True):
    """Train model for a single phase with curriculum learning."""
    model.train()
    
    iter_num = 0
    eval_loss = float('inf')
    r_current = 1
    train_losses = []
    
    pbar = tqdm(total=max_iterations, desc=f"Phase {phase}")
    
    while eval_loss > target_loss and iter_num < max_iterations:
        # Multilevel training with remember_rate
        if remember_rate > random.random():
            # Remember: sample from 1 to phase
            current_cot = np.random.randint(1, phase + 1)
        else:
            # Focus on frontier: sample from r_current to phase
            r_floor = max(1, min(r_current, phase))
            current_cot = np.random.randint(r_floor, phase + 1)
        
        # Generate puzzles with at most current_cot empty cells
        puzzles, solutions = generate_batch(t, batch_size, max_empty=current_cot)
        
        optimizer.zero_grad()
        _, loss = model.forward_with_cot(puzzles, current_cot, solutions)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'eval_loss': f'{eval_loss:.4f}', 'cot': current_cot})
        pbar.update(1)
        
        if (iter_num + 1) % eval_interval == 0:
            eval_loss, cell_acc, puzzle_acc, unknown_acc, r_current = evaluate(
                model, t, phase, phase, detect_threshold, eval_batch_size, show_examples=0
            )
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_losses = []
            
            if use_wandb:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "phase": phase,
                    "cell_acc": cell_acc,
                    "puzzle_acc": puzzle_acc,
                    "unknown_acc": unknown_acc,
                    "r": r_current,
                    "iteration": global_step + iter_num + 1
                })
            
            if training_history is not None:
                training_history.append({
                    "iteration": global_step + iter_num + 1,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "phase": phase,
                    "cell_acc": cell_acc,
                    "puzzle_acc": puzzle_acc,
                    "unknown_acc": unknown_acc,
                    "r": r_current
                })
            
            print(f"  Phase {phase}, Iter {iter_num + 1}: Loss={eval_loss:.4f}, "
                  f"Cell={cell_acc:.2%}, Puzzle={puzzle_acc:.2%}, Unknown={unknown_acc:.2%}, r={r_current}")
            model.train()
            
            if eval_loss <= target_loss:
                print(f"  ✓ Target loss {target_loss} reached!")
                break
        
        iter_num += 1
    
    pbar.close()
    
    eval_loss, cell_acc, puzzle_acc, unknown_acc, _ = evaluate(
        model, t, phase, phase, detect_threshold, eval_batch_size, show_examples=2
    )
    print(f"Phase {phase} Complete: Loss={eval_loss:.4f}, Cell={cell_acc:.2%}, "
          f"Puzzle={puzzle_acc:.2%} (after {iter_num} iterations)\n")
    
    return eval_loss, cell_acc, puzzle_acc, unknown_acc, iter_num, global_step + iter_num

# ============== Plotting ==============

def plot_training_curves(training_history, args, save_path='training_curves.png'):
    if not training_history:
        return
    
    iterations = [h['iteration'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    eval_losses = [h['eval_loss'] for h in training_history]
    cell_accs = [h['cell_acc'] for h in training_history]
    puzzle_accs = [h['puzzle_acc'] for h in training_history]
    phases = [h['phase'] for h in training_history]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(iterations, train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(iterations, eval_losses, 'r-', linewidth=1.5, label='Eval Loss', alpha=0.8)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    phase_changes = [iterations[0]]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append(iterations[i])
    for pc in phase_changes[1:]:
        ax1.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    ax2 = axes[1]
    ax2.plot(iterations, cell_accs, 'g-', linewidth=1.5, label='Cell Accuracy', alpha=0.8)
    ax2.plot(iterations, puzzle_accs, 'm-', linewidth=1.5, label='Puzzle Accuracy', alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    for pc in phase_changes[1:]:
        ax2.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    ax3 = axes[2]
    ax3.step(iterations, phases, 'c-', linewidth=2, label='Phase', alpha=0.8, where='post')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Phase', fontsize=12)
    ax3.set_title('Training Phase', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks(range(1, max(phases) + 1))
    
    n = args.t * args.t
    fig.suptitle(f'Sudoku {n}×{n} (t={args.t}) Training: {args.n_layers} layers, lr={args.lr}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()

def save_training_data(training_history, results, args, save_path='training_data.json'):
    data = {
        'config': vars(args),
        'training_history': training_history,
        'phase_results': results
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Training data saved to {save_path}")

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='Train Sudoku solver with curriculum CoT learning')
    parser.add_argument('--t', type=int, default=2, help='Sudoku parameter (grid is t² × t²)')
    parser.add_argument('--max_cot', type=int, default=16, help='Maximum CoT tokens (also number of phases)')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--iterations_per_phase', type=int, default=5000, help='Max iterations per phase')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Evaluation batch size')
    parser.add_argument('--target_loss', type=float, default=0.5, help='Target loss to advance phase')
    parser.add_argument('--truncate_backprop', action='store_true', default=True, help='Truncate backprop')
    parser.add_argument('--backprop_steps', type=int, default=1, help='Backprop steps (r)')
    parser.add_argument('--remember_rate', type=float, default=0.2, help='Rate to train on previous phases')
    parser.add_argument('--detect_threshold', type=float, default=0.5, help='Loss threshold for r detection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory for plots')
    parser.add_argument('--plot_data_dir', type=str, default='plot_data', help='Directory for plot data')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    t = args.t
    n = t * t
    num_cells = n * n
    
    print(f"Sudoku size: {n}×{n} (t={t})")
    print(f"Number of cells: {num_cells}")
    print(f"Vocabulary size: {n + 1} (0=empty, 1-{n}=values)")
    print(f"Max CoT tokens / phases: {args.max_cot}")
    print(f"Device: {device}")
    print(f"Truncate backprop: {args.truncate_backprop}, steps: {args.backprop_steps}")
    print(f"Remember rate: {args.remember_rate}, detect threshold: {args.detect_threshold}")
    print()
    
    use_wandb = not args.no_wandb
    if use_wandb:
        if wandb_login():
            wandb.init(project="curriculum-cot-sudoku", config=vars(args))
        else:
            try:
                wandb.init(project="curriculum-cot-sudoku", config=vars(args), mode="offline")
            except:
                use_wandb = False
                print("Wandb disabled - running without logging")
    
    model = SudokuGPT(
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
        print(f"PHASE {phase}: Learning Sudoku with {phase} CoT tokens")
        print(f"{'='*60}")
        
        eval_loss, cell_acc, puzzle_acc, unknown_acc, iterations, global_step = train_phase(
            model=model,
            optimizer=optimizer,
            t=t,
            phase=phase,
            max_iterations=args.iterations_per_phase,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            target_loss=args.target_loss,
            remember_rate=args.remember_rate,
            detect_threshold=args.detect_threshold,
            global_step=global_step,
            training_history=training_history,
            eval_batch_size=args.eval_batch_size,
            use_wandb=use_wandb
        )
        
        results.append({
            'phase': phase,
            'loss': eval_loss,
            'cell_acc': cell_acc,
            'puzzle_acc': puzzle_acc,
            'unknown_acc': unknown_acc,
            'iterations': iterations
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    total_iterations = 0
    for r in results:
        print(f"Phase {r['phase']}: Loss={r['loss']:.4f}, Cell={r['cell_acc']:.2%}, "
              f"Puzzle={r['puzzle_acc']:.2%}, Iters={r['iterations']}")
        total_iterations += r['iterations']
    print(f"{'='*60}")
    print(f"Total iterations: {total_iterations}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (all phases)")
    print(f"{'='*60}")
    
    for phase in range(1, args.max_cot + 1):
        loss, cell_acc, puzzle_acc, unknown_acc, _ = evaluate(
            model, t, phase, phase, args.detect_threshold, args.eval_batch_size, show_examples=0
        )
        print(f"Phase {phase} ({phase} CoT, ≤{phase} empty): Loss={loss:.4f}, Cell={cell_acc:.2%}, Puzzle={puzzle_acc:.2%}")
    
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
    }, os.path.join(script_dir, 'sudoku_model.pt'))
    print(f"\nModel saved to sudoku_model.pt")
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
