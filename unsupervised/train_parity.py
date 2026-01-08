import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT, GPTConfig
import numpy as np
import argparse
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    num_samples = max_samples if max_samples else min(num_possible, 100000)
    inputs = torch.randint(0, 2, (num_samples, n_bits), dtype=torch.long, device=device)
    return inputs

def sample_random_batch(n_bits, batch_size):
    """Generate a random batch of inputs (for large n_bits where we can't store all)."""
    return torch.randint(0, 2, (batch_size, n_bits), dtype=torch.long, device=device)

def compute_parity(inputs, k, indices=None):
    """
    Compute parity of bits for each input.
    
    Args:
        inputs: (B, n_bits) tensor of 0/1 values
        k: number of bits to consider (first k bit positions)
        indices: optional list of bit indices for random subset mode
                 If provided, computes parity over intersection of indices and first k bits
    
    Parity is the product of selected bits (treating 0 as -1, 1 as +1).
    Returns +1 or -1.
    """
    if indices is not None:
        # Intersect indices with first k bits (indices that are < k)
        valid_indices = [i for i in indices if i < k]
        if len(valid_indices) == 0:
            return torch.ones(inputs.size(0), device=inputs.device)
        selected = inputs[:, valid_indices]  # (B, len(valid_indices))
    else:
        # Use first k bits
        if k == 0:
            return torch.ones(inputs.size(0), device=inputs.device)
        selected = inputs[:, :k]  # (B, k)
    
    # Convert 0/1 to -1/+1
    signs = 2 * selected.float() - 1
    # Parity is the product
    parity = signs.prod(dim=1)  # (B,)
    return parity

def generate_random_subset(n_bits, subset_size, seed=None):
    """
    Generate a single random subset of indices.
    All phases will use this same subset for computing parity.
    
    Args:
        n_bits: total number of bits
        subset_size: size of the subset to generate
        seed: random seed for reproducibility
    
    Returns:
        list of indices (sorted)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Randomly select `subset_size` indices from 0 to n_bits-1
    indices = np.random.choice(n_bits, size=subset_size, replace=False).tolist()
    return sorted(indices)

def sample_batch(all_inputs, batch_size):
    """Sample a random batch from all inputs."""
    num_samples = all_inputs.size(0)
    indices = torch.randint(0, num_samples, (batch_size,), device=all_inputs.device)
    return all_inputs[indices]

def train_phase(model, optimizer, all_inputs, phase, num_cot_tokens, 
                max_iterations, batch_size, n_bits, eval_interval=100, target_loss=0.2,
                multilevel_training=False, subset_indices=None):
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
        eval_interval: how often to evaluate on full dataset
        target_loss: stop training when eval loss drops below this value
        multilevel_training: if True, randomly sample i from 1 to phase and train on
                            parity of first i bits with i CoT tokens
        subset_indices: optional list of bit indices to use for parity (random subset mode)
    """
    model.train()
    
    iter_num = 0
    eval_loss = float('inf')
    use_random_sampling = n_bits > 16  # For large n_bits, sample random batches
    
    pbar = tqdm(total=max_iterations, desc=f"Phase {phase}")
    
    while eval_loss > target_loss and iter_num < max_iterations:
        # Sample batch - use random sampling for large n_bits
        if use_random_sampling:
            batch = sample_random_batch(n_bits, batch_size)
        else:
            batch = sample_batch(all_inputs, batch_size)
        
        if multilevel_training:
            # Randomly pick i from 1 to phase (inclusive)
            i = np.random.randint(1, phase + 1)
            # Compute target parity (use same subset_indices for all levels)
            target = compute_parity(batch, i, indices=subset_indices)
            # Use i CoT tokens
            current_num_cot = i
        else:
            # Compute target parity for this phase
            target = compute_parity(batch, phase, indices=subset_indices)
            current_num_cot = num_cot_tokens
        
        # Forward pass through generate
        optimizer.zero_grad()
        output, loss = model.generate(batch, num_cot_tokens=current_num_cot, target=target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'eval_loss': f'{eval_loss:.4f}'})
        pbar.update(1)
        
        # Evaluate on full dataset periodically (always evaluate on the current phase target)
        if (iter_num + 1) % eval_interval == 0:
            eval_loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=6, subset_indices=subset_indices)
            print(f"  Phase {phase}, Iter {iter_num + 1}: Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.2%}")
            model.train()
            
            if eval_loss <= target_loss:
                print(f"  ✓ Target loss {target_loss} reached!")
                break
        
        iter_num += 1
    
    pbar.close()
    
    # Final evaluation with examples
    eval_loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=8, subset_indices=subset_indices)
    print(f"Phase {phase} Complete: Final Loss = {eval_loss:.4f}, Accuracy = {accuracy:.2%} (after {iter_num} iterations)\n")
    
    return eval_loss, accuracy, iter_num

def bits_to_str(bits, k=None):
    """Convert bits tensor to readable string. Highlight first k bits if specified."""
    bits_list = bits.tolist()
    if k is not None:
        # Show first k bits in brackets, rest after
        first_k = ''.join(['+' if b == 1 else '-' for b in bits_list[:k]])
        rest = ''.join(['+' if b == 1 else '-' for b in bits_list[k:]])
        return f"[{first_k}]{rest}"
    return ''.join(['+' if b == 1 else '-' for b in bits_list])

@torch.no_grad()
def evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=0, eval_batch_size=1024, subset_indices=None):
    """Evaluate model on the full dataset in batches to save memory.
    
    Args:
        model: GPT model
        all_inputs: all possible inputs
        phase: current phase (number of bits for parity)
        num_cot_tokens: number of cot tokens
        show_examples: number of examples to display (0 = none)
        eval_batch_size: batch size for evaluation (to avoid OOM)
        subset_indices: optional list of bit indices for parity (random subset mode)
    """
    model.eval()
    
    # Compute targets
    targets = compute_parity(all_inputs, phase, indices=subset_indices)
    
    # Forward pass in batches to save memory
    num_samples = all_inputs.size(0)
    all_outputs = []
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, num_samples, eval_batch_size):
        batch_inputs = all_inputs[i:i+eval_batch_size]
        batch_targets = targets[i:i+eval_batch_size]
        
        outputs, loss = model.generate(batch_inputs, num_cot_tokens=num_cot_tokens, target=batch_targets)
        all_outputs.append(outputs)
        total_loss += loss.item()
        num_batches += 1
    
    outputs = torch.cat(all_outputs, dim=0)
    loss = total_loss / num_batches
    
    # Compute accuracy (output should be close to +1 or -1)
    predictions = torch.sign(outputs)
    accuracy = (predictions == targets).float().mean().item()
    
    # Show some examples
    if show_examples > 0:
        print(f"\n  Sample predictions (phase {phase}, parity of first {phase} bits):")
        print(f"  {'Input':<20} {'Target':>8} {'Output':>10} {'Pred':>6} {'Correct':>8}")
        print(f"  {'-'*54}")
        
        # Show a mix of correct and incorrect predictions
        correct_mask = (predictions == targets)
        incorrect_indices = torch.where(~correct_mask)[0]
        correct_indices = torch.where(correct_mask)[0]
        
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
            inp = all_inputs[idx]
            tgt = targets[idx].item()
            out = outputs[idx].item()
            pred = predictions[idx].item()
            correct = "✓" if pred == tgt else "✗"
            
            inp_str = bits_to_str(inp, k=phase)
            tgt_str = f"{tgt:+.0f}"
            out_str = f"{out:+.4f}"
            pred_str = f"{pred:+.0f}"
            
            print(f"  {inp_str:<20} {tgt_str:>8} {out_str:>10} {pred_str:>6} {correct:>8}")
    
    return loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train parity function with curriculum CoT learning')
    parser.add_argument('--n_bits', type=int, default=20, help='Number of input bits')
    parser.add_argument('--k_phases', type=int, default=10, help='Number of phases (max parity bits)')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--iterations_per_phase', type=int, default=15000, help='Training iterations per phase')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--target_loss', type=float, default=0.2, help='Target loss to stop each phase')
    parser.add_argument('--multilevel', action='store_true', default=False, help='Enable multilevel training (train on parity of 1 to phase bits)')
    parser.add_argument('--separate_heads', action='store_true', default=True, help='Use separate linear head for each number of CoT tokens')
    parser.add_argument('--truncate_backprop', action='store_true', default=False, help='Enable truncated backprop through only last r forward passes')
    parser.add_argument('--backprop_steps', type=int, default=1, help='Number of last forward passes to backprop through (r)')
    parser.add_argument('--random_subset', action='store_true', default=True, help='Use random subset of bits for parity instead of first k bits')
    parser.add_argument('--seed', type=int, default=54, help='Random seed')
    
    args = parser.parse_args()
    print('multilevel training: ', args.multilevel)
    print('separate heads: ', args.separate_heads)
    print('truncate backprop: ', args.truncate_backprop)
    print('backprop steps (r): ', args.backprop_steps)
    print('random subset: ', args.random_subset)
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
        backprop_steps=args.backprop_steps
    )
    
    # Create model
    model = GPT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Generate inputs (all if n_bits <= 16, otherwise sample for evaluation)
    if args.n_bits <= 16:
        all_inputs = generate_all_inputs(args.n_bits)
        print(f"Total inputs (exhaustive): {all_inputs.size(0)}")
    else:
        # For large n_bits, use sampled inputs for evaluation only
        eval_samples = 10000
        all_inputs = generate_all_inputs(args.n_bits, max_samples=eval_samples)
        print(f"Total inputs (sampled for eval): {all_inputs.size(0)} (training uses fresh random samples)")
    print()
    
    # Generate random subset if enabled (same subset used for ALL phases)
    if args.random_subset:
        subset_indices = generate_random_subset(args.n_bits, args.k_phases, seed=args.seed)
        print(f"Using RANDOM SUBSET for parity (same for all phases):")
        print(f"  Parity bits: {subset_indices}")
        print()
    else:
        subset_indices = None
    
    # Training phases (number of phases = n_bits)
    results = []
    
    for phase in range(1, args.n_bits + 1):
        print(f"{'='*60}")
        if args.random_subset:
            print(f"PHASE {phase}: Learning parity of bits {subset_indices} with {phase} CoT tokens")
        else:
            print(f"PHASE {phase}: Learning parity of first {phase} bits with {phase} CoT tokens")
        print(f"{'='*60}")
        
        # Train this phase until loss < target_loss
        loss, accuracy, iterations = train_phase(
            model=model,
            optimizer=optimizer,
            all_inputs=all_inputs,
            phase=phase,
            num_cot_tokens=1,  # num_cot_tokens equals phase number
            max_iterations=args.iterations_per_phase,
            batch_size=args.batch_size,
            n_bits=args.n_bits,
            eval_interval=args.eval_interval,
            target_loss=args.target_loss,
            multilevel_training=args.multilevel,
            subset_indices=subset_indices
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
        print(f"Phase {r['phase']}: Loss = {r['loss']:.4f}, Accuracy = {r['accuracy']:.2%}, Iterations = {r['iterations']}")
        total_iterations += r['iterations']
    print(f"{'='*60}")
    print(f"Total iterations across all phases: {total_iterations}")
    
    # Test generalization: evaluate all phases with their respective cot tokens
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (all phases)")
    print(f"{'='*60}")
    
    for phase in range(1, args.n_bits + 1):
        loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens=phase, show_examples=6, subset_indices=subset_indices)
        if args.random_subset:
            print(f"Phase {phase} (parity of bits {subset_indices}, {phase} CoT tokens): Loss = {loss:.4f}, Accuracy = {accuracy:.2%}\n")
        else:
            print(f"Phase {phase} (parity of first {phase} bits, {phase} CoT tokens): Loss = {loss:.4f}, Accuracy = {accuracy:.2%}\n")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
        'args': args
    }, 'parity_model.pt')
    print("\nModel saved to parity_model.pt")

if __name__ == '__main__':
    main()

