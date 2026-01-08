import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT, GPTConfig
import numpy as np
import argparse
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_all_inputs(n_bits):
    """
    Generate all 2^n_bits possible inputs on the hypercube.
    Each input is a sequence of n_bits tokens, where each token is 0 or 1.
    (0 represents -1, 1 represents +1 in the parity calculation)
    """
    num_samples = 2 ** n_bits
    inputs = []
    for i in range(num_samples):
        bits = [(i >> j) & 1 for j in range(n_bits)]
        inputs.append(bits)
    return torch.tensor(inputs, dtype=torch.long, device=device)

def compute_parity(inputs, k):
    """
    Compute parity of first k bits for each input.
    Parity is the product of the first k bits (treating 0 as -1, 1 as +1).
    Returns +1 or -1.
    """
    if k == 0:
        return torch.ones(inputs.size(0), device=inputs.device)
    
    # Convert 0/1 to -1/+1
    signs = 2 * inputs[:, :k].float() - 1  # (B, k)
    # Parity is the product
    parity = signs.prod(dim=1)  # (B,)
    return parity

def sample_batch(all_inputs, batch_size):
    """Sample a random batch from all inputs."""
    num_samples = all_inputs.size(0)
    indices = torch.randint(0, num_samples, (batch_size,), device=all_inputs.device)
    return all_inputs[indices]

def train_phase(model, optimizer, all_inputs, phase, num_cot_tokens, 
                num_iterations, batch_size, eval_interval=100):
    """
    Train the model for a single phase.
    
    Args:
        model: GPT model
        optimizer: optimizer
        all_inputs: all possible inputs (2^n_bits, n_bits)
        phase: current phase number (1 to k), also the number of bits for parity
        num_cot_tokens: number of cot tokens to use (equals phase)
        num_iterations: number of training iterations
        batch_size: batch size
        eval_interval: how often to evaluate on full dataset
    """
    model.train()
    
    pbar = tqdm(range(num_iterations), desc=f"Phase {phase}")
    
    for iter_num in pbar:
        # Sample batch
        batch = sample_batch(all_inputs, batch_size)
        
        # Compute target parity for first `phase` bits
        target = compute_parity(batch, phase)
        
        # Forward pass through generate
        optimizer.zero_grad()
        output, loss = model.generate(batch, num_cot_tokens=num_cot_tokens, target=target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate on full dataset periodically
        if (iter_num + 1) % eval_interval == 0:
            eval_loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=6)
            print(f"  Phase {phase}, Iter {iter_num + 1}: Eval Loss = {eval_loss:.4f}, Accuracy = {accuracy:.2%}")
            model.train()
    
    # Final evaluation with examples
    eval_loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=8)
    print(f"Phase {phase} Complete: Final Loss = {eval_loss:.4f}, Accuracy = {accuracy:.2%}\n")
    
    return eval_loss, accuracy

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
def evaluate(model, all_inputs, phase, num_cot_tokens, show_examples=0):
    """Evaluate model on the full dataset.
    
    Args:
        model: GPT model
        all_inputs: all possible inputs
        phase: current phase (number of bits for parity)
        num_cot_tokens: number of cot tokens
        show_examples: number of examples to display (0 = none)
    """
    model.eval()
    
    # Compute targets
    targets = compute_parity(all_inputs, phase)
    
    # Forward pass
    outputs, loss = model.generate(all_inputs, num_cot_tokens=num_cot_tokens, target=targets)
    
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
    
    return loss.item(), accuracy

def main():
    parser = argparse.ArgumentParser(description='Train parity function with curriculum CoT learning')
    parser.add_argument('--n_bits', type=int, default=8, help='Number of input bits')
    parser.add_argument('--k_phases', type=int, default=8, help='Number of phases (max parity bits)')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--iterations_per_phase', type=int, default=2000, help='Training iterations per phase')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Training parity function on {args.n_bits} bits with {args.k_phases} phases")
    print(f"Device: {device}")
    print(f"Config: n_layers={args.n_layers}, n_heads={args.n_heads}, n_embd={args.n_embd}")
    print(f"Training: batch_size={args.batch_size}, lr={args.lr}, iterations_per_phase={args.iterations_per_phase}")
    print()
    
    # Create config
    # vocab_size = 2 (for -1 and +1, represented as 0 and 1)
    # block_size needs to accommodate input + max cot tokens + special token
    block_size = args.n_bits + args.k_phases + 2
    
    config = GPTConfig(
        block_size=block_size,
        vocab_size=2,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=0.0,
        cot_length=args.k_phases
    )
    
    # Create model
    model = GPT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Generate all possible inputs
    all_inputs = generate_all_inputs(args.n_bits)
    print(f"Total inputs: {all_inputs.size(0)}")
    print()
    
    # Training phases
    results = []
    
    for phase in range(1, args.k_phases + 1):
        print(f"{'='*60}")
        print(f"PHASE {phase}: Learning parity of first {phase} bits with {phase} CoT tokens")
        print(f"{'='*60}")
        
        # Train this phase
        loss, accuracy = train_phase(
            model=model,
            optimizer=optimizer,
            all_inputs=all_inputs,
            phase=phase,
            num_cot_tokens=phase,  # num_cot_tokens equals phase number
            num_iterations=args.iterations_per_phase,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval
        )
        
        results.append({
            'phase': phase,
            'loss': loss,
            'accuracy': accuracy
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"Phase {r['phase']}: Loss = {r['loss']:.4f}, Accuracy = {r['accuracy']:.2%}")
    
    # Test generalization: evaluate all phases with their respective cot tokens
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (all phases)")
    print(f"{'='*60}")
    
    for phase in range(1, args.k_phases + 1):
        loss, accuracy = evaluate(model, all_inputs, phase, num_cot_tokens=phase, show_examples=6)
        print(f"Phase {phase} (parity of {phase} bits, {phase} CoT tokens): Loss = {loss:.4f}, Accuracy = {accuracy:.2%}\n")
    
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

