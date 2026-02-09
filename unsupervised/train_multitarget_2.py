import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT, GPTConfig
import numpy as np
import argparse
from tqdm import tqdm
import wandb
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def wandb_login():
    """Login to wandb using API key from .wandb_key file."""
    # Look for .wandb_key in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, '.wandb_key')
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            lines = f.readlines()
            # Skip comment lines and get the API key
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    api_key = line
                    wandb.login(key=api_key)
                    print("Logged in to wandb using .wandb_key file")
                    return True
    
    print("Warning: .wandb_key file not found or empty. Using default wandb authentication.")
    return False

def generate_all_inputs(n_bits, max_samples=None):
    """
    Generate inputs on the hypercube.
    Each input is a sequence of n_bits tokens, where each token is 0 or 1.
    (0 represents -1, 1 represents +1 in the parity calculation)
    
    If max_samples is None and 2^n_bits is feasible (<=2^20), generate all.
    Otherwise, sample max_samples random inputs.
    """
    num_possible = 2 ** n_bits
    
    # If feasible to enumerate all, do so
    if max_samples is None and n_bits <= 16:
        inputs = []
        for i in range(num_possible):
            bits = [(i >> j) & 1 for j in range(n_bits)]
            inputs.append(bits)
        return torch.tensor(inputs, dtype=torch.long, device=device)
    
    # Otherwise, sample random inputs
    num_samples = max_samples if max_samples else min(num_possible, 200000)
    inputs = torch.randint(0, 2, (num_samples, n_bits), dtype=torch.long, device=device)
    return inputs

def sample_random_batch(n_bits, batch_size):
    """Generate a random batch of inputs (for large n_bits where we can't store all)."""
    return torch.randint(0, 2, (batch_size, n_bits), dtype=torch.long, device=device)

def compute_parity(inputs, k, subset_indices_arr=None, coefficients=None, flipping_bits=None):
    """
    Compute parity of bits for each input and return a modified batch where
    irrelevant bits (index >= k) are set to 1.
    
    Args:
        inputs: (B, n_bits) tensor of 0/1 values
        k: number of bits to consider (first k bit positions)
        subset_indices_arr: list of index lists, each defining a parity subset
        coefficients: list of real coefficients for each subset
        flipping_bits: list of bit indices to flip for additional targets
    
    Parity is the product of selected bits (treating 0 as -1, 1 as +1).
    For each subset, use only indices < k (assume 1 for the rest),
    then sum parities weighted by coefficients.

    Returns:
        targets: tensor of shape (B, 1 + len(flipping_bits)), where the first column
                 is the original target and each subsequent column corresponds to flipping
                 one bit from flipping_bits.
        modified_batch: copy of inputs where bits with index >= k are set to 1
    """
    # Create modified_batch with bits >= k set to 1
    modified_batch = inputs.clone()
    if k < inputs.size(1):
        modified_batch[:, k:] = 1
    
    def compute_single_parity(values):
        if k == 0:
            return torch.ones(values.size(0), device=values.device)
        if subset_indices_arr is None or len(subset_indices_arr) == 0:
            local_subsets = [list(range(k))]
        else:
            local_subsets = subset_indices_arr
        if coefficients is None or len(coefficients) == 0:
            local_coeffs = [1.0] * len(local_subsets)
        else:
            local_coeffs = coefficients
        if len(local_coeffs) != len(local_subsets):
            raise ValueError("coefficients must have the same length as subset_indices_arr")

        result = torch.zeros(values.size(0), device=values.device)
        for subset_indices, coeff in zip(local_subsets, local_coeffs):
            valid_indices = [i for i in subset_indices if i < k]
            if len(valid_indices) == 0:
                parity = torch.ones(values.size(0), device=values.device)
            else:
                selected = values[:, valid_indices]  # (B, len(valid_indices))
                signs = 2 * selected.float() - 1
                parity = signs.prod(dim=1)  # (B,)
            result += float(coeff) * parity
        return result

    base_target = compute_single_parity(inputs)
    flip_indices = flipping_bits or []
    flipped_targets = []
    for bit_idx in flip_indices:
        if 0 <= bit_idx < inputs.size(1):
            flipped = inputs.clone()
            flipped[:, bit_idx] = 1 - flipped[:, bit_idx]
            flipped_targets.append(compute_single_parity(flipped))
        else:
            flipped_targets.append(base_target)
    return torch.stack([base_target] + flipped_targets, dim=1), modified_batch

def generate_random_subsets(n_bits, max_subset_size, num_subsets, seed=None):
    """
    Generate multiple random subsets of indices.
    All phases will use these same subsets for computing parity.
    
    Args:
        n_bits: total number of bits
        max_subset_size: maximum size of each subset
        num_subsets: number of subsets to generate
        seed: random seed for reproducibility
    
    Returns:
        list of index lists (each sorted)
    """
    if seed is not None:
        np.random.seed(seed)
    
    max_subset_size = max(1, min(max_subset_size, n_bits))
    subsets = []
    for _ in range(num_subsets):
        subset_size = np.random.randint(1, max_subset_size + 1)
        indices = np.random.choice(n_bits, size=subset_size, replace=False).tolist()
        subsets.append(sorted(indices))
    return subsets

def sample_batch(all_inputs, batch_size):
    """Sample a random batch from all inputs."""
    num_samples = all_inputs.size(0)
    indices = torch.randint(0, num_samples, (batch_size,), device=all_inputs.device)
    return all_inputs[indices]

def train_phase(model, optimizer, all_inputs, phase, num_cot_tokens,
                max_iterations, batch_size, n_bits, eval_interval=100, target_loss=0.2,
                multilevel_training=False, subset_indices_arr=None, coefficients=None, flipping_bits=None,
                heads=None, flipping_ratio=1.0, global_step=0, training_history=None,
                remember_rate=0.5, detect_threshold=0.2, eval_batch_size=10000):
    """
    Train the model for a single phase until loss drops below target_loss.
    
    Args:
        model: GPT model
        optimizer: optimizer
        all_inputs: all possible inputs or sampled inputs for evaluation
        phase: current phase number (1 to k), also the number of bits for parity
        num_cot_tokens: number of cot tokens to use (equals phase)
        max_iterations: maximum number of training iterations
        batch_size: batch size
        n_bits: number of input bits (used for random sampling when n_bits > 16)
        eval_interval: how often to evaluate on a random batch
        eval_batch_size: batch size for evaluation
        target_loss: stop training when eval loss drops below this value
        multilevel_training: if True, randomly sample i from 1 to phase and train on
                            parity of first i bits with i CoT tokens
        subset_indices_arr: list of bit-index subsets to use for parity (Fourier terms)
        coefficients: coefficients for each subset
        flipping_bits: list of bit indices to flip for additional targets
        heads: per-(cot, target) linear heads
        flipping_ratio: relative probability of each flipped target vs original
        global_step: global step counter across all phases (for wandb logging)
        training_history: list to append training metrics to (for plotting)
    
    Returns:
        eval_loss, accuracy, iter_num, updated global_step
    """
    model.train()
    
    iter_num = 0
    eval_loss = float('inf')
    use_random_sampling = n_bits > 16  # For large n_bits, sample random batches
    r_current = 1  # Updated at each evaluation based on detect_threshold
    
    # Track train loss over eval_interval
    train_losses = []
    
    pbar = tqdm(total=max_iterations, desc=f"Phase {phase}")
    
    while eval_loss > target_loss and iter_num < max_iterations:
        # Sample batch - use random sampling for large n_bits
        if use_random_sampling:
            batch = sample_random_batch(n_bits, batch_size)
        else:
            batch = sample_batch(all_inputs, batch_size)
        
        if multilevel_training:
            # Randomly pick i from 1 to phase (inclusive)
            import random
            if remember_rate > random.random():
                i = np.random.randint(1, phase + 1)
            else:
                r_floor = max(1, min(r_current, phase))
                i = np.random.randint(r_floor, phase + 1)
            # Compute target parity and get modified batch (use same subset_indices for all levels)
            targets_all, modified_batch = compute_parity(
                batch,
                i,
                subset_indices_arr=subset_indices_arr,
                coefficients=coefficients,
                flipping_bits=flipping_bits
            )
            relevant_flip_indices = [
                idx for idx, bit in enumerate(flipping_bits or []) if bit < i
            ]
            if not relevant_flip_indices:
                target_idx = 0
            else:
                weights = [1.0] + [flipping_ratio] * len(relevant_flip_indices)
                probs = np.array(weights, dtype=np.float64)
                probs /= probs.sum()
                choice = np.random.choice(len(probs), p=probs)
                target_idx = 0 if choice == 0 else (relevant_flip_indices[choice - 1] + 1)
            target = targets_all[:, target_idx]
            # Use i CoT tokens
            current_num_cot = i
        else:
            # Compute target parity and get modified batch for this phase
            targets_all, modified_batch = compute_parity(
                batch,
                phase,
                subset_indices_arr=subset_indices_arr,
                coefficients=coefficients,
                flipping_bits=flipping_bits
            )
            relevant_flip_indices = [
                idx for idx, bit in enumerate(flipping_bits or []) if bit < phase
            ]
            if not relevant_flip_indices:
                target_idx = 0
            else:
                weights = [1.0] + [flipping_ratio] * len(relevant_flip_indices)
                probs = np.array(weights, dtype=np.float64)
                probs /= probs.sum()
                choice = np.random.choice(len(probs), p=probs)
                target_idx = 0 if choice == 0 else (relevant_flip_indices[choice - 1] + 1)
            target = targets_all[:, target_idx]
            current_num_cot = num_cot_tokens
        
        # Forward pass through generate (use modified_batch instead of batch)
        optimizer.zero_grad()
        head = heads[current_num_cot][target_idx]
        output, loss = generate_with_head(model, modified_batch, current_num_cot, head, target=target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'eval_loss': f'{eval_loss:.4f}'})
        pbar.update(1)
        
        # Evaluate on full dataset periodically (always evaluate on the current phase target)
        if (iter_num + 1) % eval_interval == 0:
            eval_loss, accuracy, r_current = evaluate(
                model,
                all_inputs,
                phase,
                num_cot_tokens,
                detect_threshold=detect_threshold,
                show_examples=6,
                eval_batch_size=eval_batch_size,
                subset_indices_arr=subset_indices_arr,
                coefficients=coefficients,
                flipping_bits=flipping_bits,
                heads=heads
            )
            
            # Compute average train loss over the interval
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_losses = []  # Reset for next interval
            
            # Log to wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "eval_loss": eval_loss,
                "phase": phase,
                "accuracy": accuracy,
                "r": r_current,
                "iteration": global_step + iter_num + 1
            })
            
            # Save to training history for local plotting
            if training_history is not None:
                training_history.append({
                    "iteration": global_step + iter_num + 1,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "phase": phase,
                    "accuracy": accuracy,
                    "r": r_current
                })
            
            print(f"  Phase {phase}, Iter {iter_num + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {eval_loss:.4f}, MSE = {accuracy:.4f}, r = {r_current}")
            model.train()
            
            if eval_loss <= target_loss:
                print(f"  ✓ Target loss {target_loss} reached!")
                break
        
        iter_num += 1
    
    pbar.close()
    
    # Final evaluation with examples
    eval_loss, accuracy, _ = evaluate(
        model,
        all_inputs,
        phase,
        num_cot_tokens,
        detect_threshold=detect_threshold,
        show_examples=8,
        eval_batch_size=eval_batch_size,
        subset_indices_arr=subset_indices_arr,
        coefficients=coefficients,
        flipping_bits=flipping_bits,
        heads=heads
    )
    print(f"Phase {phase} Complete: Final Loss = {eval_loss:.4f}, MSE = {accuracy:.4f} (after {iter_num} iterations)\n")
    
    return eval_loss, accuracy, iter_num, global_step + iter_num

def bits_to_str(bits, k=None):
    """Convert bits tensor to readable string. Highlight first k bits if specified."""
    bits_list = bits.tolist()
    if k is not None:
        # Show first k bits in brackets, rest after
        first_k = ''.join(['+' if b == 1 else '-' for b in bits_list[:k]])
        rest = ''.join(['+' if b == 1 else '-' for b in bits_list[k:]])
        return f"[{first_k}]{rest}"
    return ''.join(['+' if b == 1 else '-' for b in bits_list])

def format_fourier_expression(subset_indices_arr, coefficients):
    """Format a sum of monomials like 0.1 x1x2x3 + 0.4 x4x7."""
    if not subset_indices_arr or not coefficients:
        return "0"
    terms = []
    for subset_indices, coeff in zip(subset_indices_arr, coefficients):
        if subset_indices:
            vars_part = "".join([f"x{i+1}" for i in subset_indices])
        else:
            vars_part = "1"
        terms.append((coeff, vars_part))
    expr_parts = []
    for idx, (coeff, vars_part) in enumerate(terms):
        sign = "-" if coeff < 0 else "+"
        coeff_abs = abs(coeff)
        if idx == 0:
            expr_parts.append(f"{coeff:.4f} {vars_part}")
        else:
            expr_parts.append(f"{sign} {coeff_abs:.4f} {vars_part}")
    return " ".join(expr_parts)

def generate_with_head(model, idx, num_cot_tokens, head, target=None):
    """Generate output using a specific linear head for a target."""
    B, input_length = idx.size()
    truncate_backprop = getattr(model.config, 'truncate_backprop', False)
    backprop_steps = getattr(model.config, 'backprop_steps', num_cot_tokens + 1)
    cot_embeddings = torch.empty(B, 0, model.config.n_embd, device=idx.device)
    detach_until = num_cot_tokens + 1 - backprop_steps if truncate_backprop else 0

    for i in range(num_cot_tokens):
        if truncate_backprop and i < detach_until:
            cot_embeddings = cot_embeddings.detach()
        special_emb = model.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        current_cot = torch.cat([cot_embeddings, special_emb], dim=1)
        hidden_states = model.forward_with_hidden(idx, current_cot)
        next_cot_token = hidden_states[:, -1:, :]
        cot_embeddings = torch.cat([cot_embeddings, next_cot_token], dim=1)

    if truncate_backprop and num_cot_tokens < detach_until:
        cot_embeddings = cot_embeddings.detach()

    special_emb = model.special_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    final_cot = torch.cat([cot_embeddings, special_emb], dim=1)
    hidden_states = model.forward_with_hidden(idx, final_cot)
    special_hidden = hidden_states[:, -1, :]
    output = head(special_hidden).squeeze(-1)

    loss = None
    if target is not None:
        loss = nn.functional.mse_loss(output, target)
    return output, loss

@torch.no_grad()
def evaluate(model, all_inputs, phase, num_cot_tokens, detect_threshold=0.2, show_examples=0, eval_batch_size=1024, subset_indices_arr=None, coefficients=None, flipping_bits=None, heads=None):
    """Evaluate model on the full dataset in batches to save memory.
    
    Args:
        model: GPT model
        all_inputs: all possible inputs
        phase: current phase (number of bits for parity)
        num_cot_tokens: number of cot tokens
        show_examples: number of examples to display (0 = none)
        eval_batch_size: batch size for evaluation (random batch size)
        subset_indices_arr: list of bit-index subsets to use for parity (Fourier terms)
        coefficients: coefficients for each subset
        flipping_bits: list of bit indices to flip for additional targets
        heads: per-(cot, target) linear heads
    """
    model.eval()

    print("  Fourier target:", format_fourier_expression(subset_indices_arr, coefficients))
    
    losses_by_cot = {}
    outputs_phase = None
    targets_phase = None
    modified_inputs_phase = None
    num_samples = all_inputs.size(0)
    eval_indices = torch.randint(0, num_samples, (min(eval_batch_size, num_samples),), device=all_inputs.device)
    eval_inputs = all_inputs[eval_indices]
    
    for cot_tokens in range(1, phase + 1):
        targets_all, modified_inputs = compute_parity(
            eval_inputs,
            cot_tokens,
            subset_indices_arr=subset_indices_arr,
            coefficients=coefficients,
            flipping_bits=flipping_bits
        )
        targets = targets_all[:, 0]
        head = heads[cot_tokens][0]
        outputs, loss = generate_with_head(model, modified_inputs, cot_tokens, head, target=targets)
        losses_by_cot[cot_tokens] = loss.item()
        if cot_tokens == phase:
            outputs_phase = outputs
            targets_phase = targets
            modified_inputs_phase = modified_inputs
    
    loss = losses_by_cot[phase]
    
    # Use MSE as the accuracy indicator for real-valued targets
    accuracy = loss
    
    r = phase
    for cot_tokens in range(1, phase + 1):
        if losses_by_cot[cot_tokens] > detect_threshold:
            r = cot_tokens
            break
    
    # Show some examples
    if show_examples > 0:
        print(f"\n  Sample predictions (phase {phase}, parity of first {phase} bits):")
        print(f"  {'Input':<20} {'Target':>8} {'Output':>10} {'Pred':>6} {'Correct':>8}")
        print(f"  {'-'*54}")
        
        # Show a mix of correct and incorrect predictions
        errors = (outputs_phase - targets_phase).abs()
        incorrect_indices = torch.where(errors > 0)[0]
        correct_indices = torch.where(errors == 0)[0]
        
        # Sample some examples
        n_incorrect = min(show_examples // 2, len(incorrect_indices))
        n_correct = min(show_examples - n_incorrect, len(correct_indices))
        
        if n_incorrect > 0:
            sample_incorrect = incorrect_indices[torch.randperm(len(incorrect_indices))[:n_incorrect]]
        else:
            sample_incorrect = torch.tensor([], dtype=torch.long, device=all_inputs.device)
        
        if n_correct > 0:
            sample_correct = correct_indices[torch.randperm(len(correct_indices))[:n_correct]]
        else:
            sample_correct = torch.tensor([], dtype=torch.long, device=all_inputs.device)
        
        sample_indices = torch.cat([sample_incorrect, sample_correct])
        
        for idx in sample_indices:
            inp = modified_inputs_phase[idx]
            tgt = targets_phase[idx].item()
            out = outputs_phase[idx].item()
            pred = outputs_phase[idx].item()
            correct = "✓" if pred == tgt else "✗"
            
            inp_str = bits_to_str(inp, k=phase)
            tgt_str = f"{tgt:+.0f}"
            out_str = f"{out:+.4f}"
            pred_str = f"{pred:+.4f}"
            
            print(f"  {inp_str:<20} {tgt_str:>8} {out_str:>10} {pred_str:>6} {correct:>8}")
    
    return loss, accuracy, r

def plot_training_curves(training_history, args, save_path='training_curves.png'):
    """
    Plot training curves from the collected training history.
    
    Args:
        training_history: list of dicts with keys: iteration, train_loss, eval_loss, phase, accuracy
        args: command line arguments (for title info)
        save_path: path to save the plot
    """
    if not training_history:
        print("No training history to plot.")
        return
    
    iterations = [h['iteration'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    eval_losses = [h['eval_loss'] for h in training_history]
    accuracies = [h['accuracy'] for h in training_history]
    phases = [h['phase'] for h in training_history]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Train and Eval Loss
    ax1 = axes[0]
    ax1.plot(iterations, train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(iterations, eval_losses, 'r-', linewidth=1.5, label='Eval Loss', alpha=0.8)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Evaluation Loss over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add phase transition markers
    phase_changes = [iterations[0]]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append(iterations[i])
    for pc in phase_changes[1:]:
        ax1.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    ax2.plot(iterations, accuracies, 'g-', linewidth=1.5, label='Accuracy', alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Evaluation Accuracy over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    
    # Add phase transition markers
    for pc in phase_changes[1:]:
        ax2.axvline(x=pc, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Phase
    ax3 = axes[2]
    ax3.plot(iterations, phases, 'm-', linewidth=2, label='Training Phase', alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Phase', fontsize=12)
    ax3.set_title('Training Phase over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks(range(1, max(phases) + 1))
    
    # Overall title
    fig.suptitle(f'Curriculum CoT Training: {args.n_bits} bits, {args.n_layers} layers, lr={args.lr}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()

def save_training_data(training_history, results, args, save_path='training_data.json'):
    """
    Save training history and results to JSON file for later analysis.
    
    Args:
        training_history: list of training metrics over time
        results: final results per phase
        args: command line arguments
        save_path: path to save the JSON file
    """
    data = {
        'config': vars(args),
        'training_history': training_history,
        'phase_results': results
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Training data saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train parity function with curriculum CoT learning')
    parser.add_argument('--n_bits', type=int, default=20, help='Number of input bits')
    parser.add_argument('--k_phases', type=int, default=10, help='Number of phases (max parity bits)')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--iterations_per_phase', type=int, default=20000, help='Training iterations per phase')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--eval_batch_size', type=int, default=10000, help='Evaluation batch size')
    parser.add_argument('--target_loss', type=float, default=0.05, help='Target loss to stop each phase')
    parser.add_argument('--multilevel', action='store_true', default=False, help='Enable multilevel training (train on parity of 1 to phase bits)')
    parser.add_argument('--separate_heads', action='store_true', default=True, help='Use separate linear head for each number of CoT tokens')
    parser.add_argument('--truncate_backprop', action='store_true', default=False, help='Enable truncated backprop through only last r forward passes')
    parser.add_argument('--backprop_steps', type=int, default=1, help='Number of last forward passes to backprop through (r)')
    parser.add_argument('--random_subset', action='store_true', default=True, help='Use random subsets for Fourier parity terms')
    parser.add_argument('--fourier_num', type=int, default=2, help='Number of parity subsets (Fourier terms) to sample')
    parser.add_argument('--seed', type=int, default=54, help='Random seed')
    parser.add_argument('--remember_rate', type=float, default=0.1, help='Rate at which to remember the previous phases')
    parser.add_argument('--detect_threshold', type=float, default=0.0, help='Loss threshold used to compute r during evaluation')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory for saving plots')
    parser.add_argument('--plot_data_dir', type=str, default='plot_data', help='Directory for saving plot data')
    parser.add_argument('--flipping_bits', type=str, default='0,2', help='Comma-separated bit indices to flip for extra targets')
    parser.add_argument('--flipping_ratio', type=float, default=0.5, help='Relative probability of each flipped target vs original')

    args = parser.parse_args()
    flipping_bits = []
    if args.flipping_bits.strip():
        flipping_bits = [int(x) for x in args.flipping_bits.split(',') if x.strip() != '']
    print('multilevel training: ', args.multilevel)
    print('separate heads: ', args.separate_heads)
    print('truncate backprop: ', args.truncate_backprop)
    print('backprop steps (r): ', args.backprop_steps)
    print('random subset: ', args.random_subset)
    if args.n_embd % args.n_heads != 0:
        raise ValueError(f"n_embd ({args.n_embd}) must be divisible by n_heads ({args.n_heads})")
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Login and initialize wandb
    wandb_login()
    wandb.init(
        project="curriculum-cot-parity",
        config={
            "n_bits": args.n_bits,
            "k_phases": args.k_phases,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "n_embd": args.n_embd,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "iterations_per_phase": args.iterations_per_phase,
            "target_loss": args.target_loss,
            "eval_batch_size": args.eval_batch_size,
            "multilevel": args.multilevel,
            "separate_heads": args.separate_heads,
            "truncate_backprop": args.truncate_backprop,
            "backprop_steps": args.backprop_steps,
            "random_subset": args.random_subset,
            "fourier_num": args.fourier_num,
            "seed": args.seed,
            "detect_threshold": args.detect_threshold,
            "flipping_bits": flipping_bits,
            "flipping_ratio": args.flipping_ratio,
        }
    )
    
    print(f"Training parity function on {args.n_bits} bits with {args.n_bits} phases")
    print(f"Device: {device}")
    print(f"Config: n_layers={args.n_layers}, n_heads={args.n_heads}, n_embd={args.n_embd}")
    print(f"Training: batch_size={args.batch_size}, lr={args.lr}, max_iterations={args.iterations_per_phase}, target_loss={args.target_loss}, multilevel={args.multilevel}")
    print()
    
    # Create config
    # vocab_size = 2 (for -1 and +1, represented as 0 and 1)
    # block_size needs to accommodate input + max cot tokens + special token
    # max cot tokens = n_bits (one per phase)
    block_size = args.n_bits + args.n_bits + 2
    
    num_targets = 1 + len(flipping_bits)
    config = GPTConfig(
        block_size=block_size,
        vocab_size=2,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=0.0,
        cot_length=args.n_bits,  # max cot tokens = n_bits
        separate_heads=args.separate_heads,
        truncate_backprop=args.truncate_backprop,
        backprop_steps=args.backprop_steps,
        num_targets=num_targets
    )
    
    # Create model
    model = GPT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use multitarget heads created inside the model
    heads = model.multitarget_heads
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Generate inputs (all if n_bits <= 16, otherwise sample for evaluation)
    if args.n_bits <= 16:
        all_inputs = generate_all_inputs(args.n_bits)
        print(f"Total inputs (exhaustive): {all_inputs.size(0)}")
    else:
        # For large n_bits, use sampled inputs for evaluation only
        eval_samples = 1000000
        all_inputs = generate_all_inputs(args.n_bits, max_samples=eval_samples)
        print(f"Total inputs (sampled for eval): {all_inputs.size(0)} (training uses fresh random samples)")
    print()
    
    # Generate parity subsets and coefficients (same for ALL phases)
    if args.random_subset:
        subset_indices_arr = generate_random_subsets(
            args.n_bits,
            args.k_phases,
            args.fourier_num,
            seed=args.seed
        )
        coefficients = np.random.uniform(-1.0, 1.0, size=args.fourier_num).tolist()
        print("Using RANDOM SUBSETS for parity (same for all phases):")
        print(f"  Subset count: {len(subset_indices_arr)}")
        print(f"  Coefficients: {coefficients}")
        print()
    else:
        subset_indices_arr = [list(range(args.n_bits))]
        coefficients = [1.0]
    
    # Training phases (number of phases = n_bits)
    results = []
    training_history = []  # Collect metrics for plotting
    global_step = 0
    
    for phase in range(1, args.n_bits + 1):
        print(f"{'='*60}")
        if args.random_subset:
            print(f"PHASE {phase}: Learning Fourier parity with {phase} CoT tokens")
        else:
            print(f"PHASE {phase}: Learning parity of first {phase} bits with {phase} CoT tokens")
        print(f"{'='*60}")
        
        # Train this phase until loss < target_loss
        loss, accuracy, iterations, global_step = train_phase(
            model=model,
            optimizer=optimizer,
            all_inputs=all_inputs,
            phase=phase,
            num_cot_tokens=phase,  # num_cot_tokens equals phase number
            max_iterations=args.iterations_per_phase,
            batch_size=args.batch_size,
            n_bits=args.n_bits,
            eval_interval=args.eval_interval,
            target_loss=args.target_loss,
            multilevel_training=args.multilevel,
            subset_indices_arr=subset_indices_arr,
            coefficients=coefficients,
            flipping_bits=flipping_bits,
            heads=heads,
            flipping_ratio=args.flipping_ratio,
            global_step=global_step,
            training_history=training_history,
            remember_rate=args.remember_rate,
            detect_threshold=args.detect_threshold,
            eval_batch_size=args.eval_batch_size
        )
        
        results.append({
            'phase': phase,
            'loss': loss,
            'accuracy': accuracy,
            'iterations': iterations
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    total_iterations = 0
    for r in results:
        print(f"Phase {r['phase']}: Loss = {r['loss']:.4f}, MSE = {r['accuracy']:.4f}, Iterations = {r['iterations']}")
        total_iterations += r['iterations']
    print(f"{'='*60}")
    print(f"Total iterations across all phases: {total_iterations}")
    
    # Test generalization: evaluate all phases with their respective cot tokens
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (all phases)")
    print(f"{'='*60}")
    
    for phase in range(1, args.n_bits + 1):
        loss, accuracy, _ = evaluate(
            model,
            all_inputs,
            phase,
            num_cot_tokens=phase,
            detect_threshold=args.detect_threshold,
            show_examples=6,
            eval_batch_size=args.eval_batch_size,
            subset_indices_arr=subset_indices_arr,
            coefficients=coefficients,
            flipping_bits=flipping_bits,
            heads=heads
        )
        if args.random_subset:
            print(f"Phase {phase} (Fourier parity, {phase} CoT tokens): Loss = {loss:.4f}, MSE = {accuracy:.4f}\n")
        else:
            print(f"Phase {phase} (parity of first {phase} bits, {phase} CoT tokens): Loss = {loss:.4f}, MSE = {accuracy:.4f}\n")
    
    # Create output folders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = args.plots_dir if os.path.isabs(args.plots_dir) else os.path.join(script_dir, args.plots_dir)
    plot_data_dir = args.plot_data_dir if os.path.isabs(args.plot_data_dir) else os.path.join(script_dir, args.plot_data_dir)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    
    # Generate timestamp for filenames and run folders
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = f"{timestamp}_flips_{len(flipping_bits)}"
    plots_run_dir = os.path.join(plots_dir, run_folder)
    plot_data_run_dir = os.path.join(plot_data_dir, run_folder)
    os.makedirs(plots_run_dir, exist_ok=True)
    os.makedirs(plot_data_run_dir, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(training_history, args, save_path=os.path.join(plots_run_dir, f'training_curves_{timestamp}.png'))
    
    # Save training data to JSON
    save_training_data(training_history, results, args, save_path=os.path.join(plot_data_run_dir, f'training_data_{timestamp}.json'))

    # Save run configuration alongside outputs
    with open(os.path.join(plots_run_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(plot_data_run_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
        'training_history': training_history,
        'args': args
    }, 'parity_model.pt')
    print("\nModel saved to parity_model.pt")
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
