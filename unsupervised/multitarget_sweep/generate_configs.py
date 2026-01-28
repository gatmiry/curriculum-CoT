#!/usr/bin/env python3
"""Generate sweep configurations for train_multitarget.py"""

import json
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_bits = 20  # Default n_bits in train_multitarget.py
k_phases = 10  # Default k_phases
base_seed = 54  # Seed used for training
fourier_nums = [1, 2, 3, 4, 5]

def generate_random_subsets(n_bits, max_subset_size, num_subsets, seed):
    """
    Generate multiple random subsets of indices.
    Mirrors the function in train_multitarget.py
    """
    np.random.seed(seed)
    max_subset_size = max(1, min(max_subset_size, n_bits))
    subsets = []
    for _ in range(num_subsets):
        subset_size = np.random.randint(1, max_subset_size + 1)
        indices = np.random.choice(n_bits, size=subset_size, replace=False).tolist()
        subsets.append(sorted(indices))
    return subsets

def format_fourier_expression(subset_indices_arr, coefficients):
    """Format a sum of monomials like 0.12x1x3+0.45x2x5."""
    if not subset_indices_arr or not coefficients:
        return "0"
    terms = []
    for subset_indices, coeff in zip(subset_indices_arr, coefficients):
        if subset_indices:
            vars_part = "".join([f"x{i}" for i in subset_indices])
        else:
            vars_part = "1"
        # Format coefficient with sign
        if coeff >= 0:
            sign = "+" if terms else ""
            terms.append(f"{sign}{coeff:.2f}{vars_part}")
        else:
            terms.append(f"{coeff:.2f}{vars_part}")
    return "".join(terms)

def format_fourier_for_filename(subset_indices_arr, coefficients):
    """Format Fourier expression for use in filename (no special chars)."""
    if not subset_indices_arr or not coefficients:
        return "0"
    terms = []
    for subset_indices, coeff in zip(subset_indices_arr, coefficients):
        if subset_indices:
            vars_part = "".join([f"x{i}" for i in subset_indices])
        else:
            vars_part = "1"
        # Format coefficient - use 'p' for positive, 'n' for negative
        coeff_abs = abs(coeff)
        sign = "p" if coeff >= 0 else "n"
        terms.append(f"{sign}{coeff_abs:.2f}{vars_part}")
    return "_".join(terms).replace(".", "d")  # Replace dots with 'd' for filenames

# Generate random flipping_bits options
def generate_random_bits(n_bits, count, num_bits_each):
    """Generate 'count' random combinations of 'num_bits_each' bit indices."""
    results = []
    seen = set()
    attempts = 0
    max_attempts = 1000
    
    while len(results) < count and attempts < max_attempts:
        bits = sorted(np.random.choice(n_bits, size=num_bits_each, replace=False).tolist())
        bits_str = ','.join(map(str, bits))
        if bits_str not in seen:
            seen.add(bits_str)
            results.append(bits_str)
        attempts += 1
    
    return results

# Generate all flipping_bits options
flipping_bits_options = []

# 4 single bits
singles = generate_random_bits(n_bits, 4, 1)
flipping_bits_options.extend(singles)
print(f"Singles: {singles}")

# 4 pairs (x,y)
pairs = generate_random_bits(n_bits, 4, 2)
flipping_bits_options.extend(pairs)
print(f"Pairs: {pairs}")

# 4 triples (x,y,z)
triples = generate_random_bits(n_bits, 4, 3)
flipping_bits_options.extend(triples)
print(f"Triples: {triples}")

# 4 quadruples (x,y,z,t)
quadruples = generate_random_bits(n_bits, 4, 4)
flipping_bits_options.extend(quadruples)
print(f"Quadruples: {quadruples}")

# 4 quintuples (x,y,z,t,r)
quintuples = generate_random_bits(n_bits, 4, 5)
flipping_bits_options.extend(quintuples)
print(f"Quintuples: {quintuples}")

print(f"\nTotal flipping_bits options: {len(flipping_bits_options)}")
print(f"Total fourier_num options: {len(fourier_nums)}")
print(f"Total configurations: {len(fourier_nums) * len(flipping_bits_options)}")

# Pre-generate the Fourier functions for each fourier_num
fourier_functions = {}
for fourier_num in fourier_nums:
    # Reset seed to match what train_multitarget.py will do
    np.random.seed(base_seed)
    subsets = generate_random_subsets(n_bits, k_phases, fourier_num, base_seed)
    np.random.seed(base_seed)  # Reset again for coefficients
    # Skip the subset generation random calls
    for _ in range(fourier_num):
        _ = np.random.randint(1, k_phases + 1)
        _ = np.random.choice(n_bits, size=np.random.randint(1, k_phases + 1), replace=False)
    coefficients = np.random.uniform(-1.0, 1.0, size=fourier_num).tolist()
    
    # Actually, let's just regenerate properly
    np.random.seed(base_seed)
    subsets = []
    for _ in range(fourier_num):
        subset_size = np.random.randint(1, k_phases + 1)
        indices = np.random.choice(n_bits, size=subset_size, replace=False).tolist()
        subsets.append(sorted(indices))
    coefficients = np.random.uniform(-1.0, 1.0, size=fourier_num).tolist()
    
    fourier_functions[fourier_num] = {
        'subsets': subsets,
        'coefficients': coefficients,
        'expression': format_fourier_expression(subsets, coefficients),
        'filename_safe': format_fourier_for_filename(subsets, coefficients)
    }
    print(f"\nFourier num {fourier_num}: {fourier_functions[fourier_num]['expression']}")
    print(f"  Filename: {fourier_functions[fourier_num]['filename_safe']}")

# Generate all configurations
configs = []
config_id = 1

for fourier_num in fourier_nums:
    for flipping_bits in flipping_bits_options:
        fourier_info = fourier_functions[fourier_num]
        config = {
            "id": config_id,
            "n_bits": 20,
            "k_phases": 10,
            "n_layers": 1,
            "n_heads": 4,
            "n_embd": 64,
            "batch_size": 128,
            "iterations_per_phase": 20000,
            "lr": 1e-5,
            "eval_interval": 200,
            "eval_batch_size": 10000,
            "target_loss": 0.1,
            "multilevel": True,
            "separate_heads": True,
            "truncate_backprop": True,
            "backprop_steps": 1,
            "random_subset": True,
            "fourier_num": fourier_num,
            "seed": base_seed,
            "remember_rate": 0.1,
            "detect_threshold": 0.1,
            "flipping_bits": flipping_bits,
            "flipping_ratio": 0.5,
            # Add Fourier function info
            "fourier_expression": fourier_info['expression'],
            "fourier_filename": fourier_info['filename_safe'],
            "fourier_subsets": fourier_info['subsets'],
            "fourier_coefficients": fourier_info['coefficients']
        }
        configs.append(config)
        config_id += 1

# Save to JSON
output_file = "sweep_configs.json"
with open(output_file, 'w') as f:
    json.dump(configs, f, indent=2)

print(f"\nSaved {len(configs)} configurations to {output_file}")

# Also save a summary
summary = {
    "total_configs": len(configs),
    "fourier_nums": fourier_nums,
    "fourier_functions": {str(k): v for k, v in fourier_functions.items()},
    "flipping_bits_options": flipping_bits_options,
    "flipping_bits_breakdown": {
        "singles": singles,
        "pairs": pairs,
        "triples": triples,
        "quadruples": quadruples,
        "quintuples": quintuples
    }
}

with open("sweep_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved sweep summary to sweep_summary.json")
