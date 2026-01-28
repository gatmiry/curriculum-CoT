#!/usr/bin/env python3
"""
Generate sweep configurations for train_parity_linear_decay.py
Varying detect_threshold from 0.025 to 0.4 in steps of 0.025
"""

import json
import os

# detect_threshold values: 0.025, 0.05, 0.075, ..., 0.4
detect_thresholds = [round(0.025 * i, 3) for i in range(1, 17)]  # 16 values

# Base configuration (defaults from train_parity_linear_decay.py)
base_config = {
    "n_bits": 20,
    "k_phases": 10,
    "n_layers": 1,
    "n_heads": 1,
    "n_embd": 256,
    "batch_size": 256,
    "iterations_per_phase": 15000,
    "lr": 1e-5,
    "eval_interval": 200,
    "target_loss": 0.1,
    "multilevel": True,
    "separate_heads": True,
    "truncate_backprop": True,
    "backprop_steps": 1,
    "random_subset": True,
    "seed": 54,
    "remember_rate": 0.2,
}

configs = []
for i, detect_threshold in enumerate(detect_thresholds, start=1):
    config = base_config.copy()
    config["id"] = i
    config["detect_threshold"] = detect_threshold
    configs.append(config)

# Save configs
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "sweep_configs.json")

with open(config_path, "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {len(configs)} configurations")
print(f"Saved to: {config_path}")

# Also save a summary
summary = {
    "total_configs": len(configs),
    "detect_threshold_values": detect_thresholds,
    "base_config": base_config,
}

summary_path = os.path.join(script_dir, "sweep_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_path}")
print(f"\ndetect_threshold values: {detect_thresholds}")

