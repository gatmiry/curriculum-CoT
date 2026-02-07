#!/usr/bin/env python3
"""
Plot comparison of accuracy for CoT vs No CoT vs No Curriculum sweeps.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path("/mnt/task_runtime/curriculum-CoT/unsupervised/comparison_sweep_1")
COT_DIR = BASE_DIR / "cot_outputs"
NOCOT_DIR = BASE_DIR / "nocot_outputs"
NOCURRICULUM_DIR = BASE_DIR / "nocurriculum_outputs"
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_sweep_data_with_seeds(sweep_dir):
    """Load accuracy data from all runs, returning seed info."""
    all_data = []
    
    if not sweep_dir.exists():
        return None
    
    run_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]
    
    for run_dir in sorted(run_dirs):
        # Extract seed from folder name
        folder_name = run_dir.name
        seed = None
        if '_seed_' in folder_name:
            seed = int(folder_name.split('_seed_')[-1])
        
        # Look in plot_data subfolder
        plot_data_dir = run_dir / "plot_data"
        if plot_data_dir.exists():
            json_files = list(plot_data_dir.glob("training_data_*.json"))
        else:
            json_files = list(run_dir.glob("training_data_*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            phase_results = data.get('phase_results', [])
            if phase_results:
                phases = []
                accuracies = []
                losses = []
                for r in sorted(phase_results, key=lambda x: x['phase']):
                    phases.append(r['phase'])
                    accuracies.append(r['accuracy'] * 100)
                    losses.append(r['loss'])
                
                all_data.append({
                    'seed': seed,
                    'phases': phases,
                    'accuracies': accuracies,
                    'losses': losses
                })
    
    return all_data if all_data else None

def compute_mean_std(all_data, metric='accuracies'):
    """Compute mean and std across all runs."""
    if not all_data:
        return None, None, None
    
    all_phases = set()
    for d in all_data:
        all_phases.update(d['phases'])
    phases = sorted(all_phases)
    
    means = []
    stds = []
    for phase in phases:
        phase_vals = []
        for d in all_data:
            if phase in d['phases']:
                idx = d['phases'].index(phase)
                phase_vals.append(d[metric][idx])
        if phase_vals:
            means.append(np.mean(phase_vals))
            stds.append(np.std(phase_vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return phases, np.array(means), np.array(stds)

# Load data
print("Loading CoT data...")
cot_data = load_sweep_data_with_seeds(COT_DIR)
print(f"Loaded {len(cot_data) if cot_data else 0} CoT runs")

print("Loading No CoT data...")
nocot_data = load_sweep_data_with_seeds(NOCOT_DIR)
print(f"Loaded {len(nocot_data) if nocot_data else 0} No CoT runs")

print("Loading No Curriculum data...")
nocurriculum_data = load_sweep_data_with_seeds(NOCURRICULUM_DIR)
print(f"Loaded {len(nocurriculum_data) if nocurriculum_data else 0} No Curriculum runs")

plt.style.use('seaborn-v0_8-whitegrid')

# ==================== Plot 1: Accuracy Comparison ====================
fig, ax = plt.subplots(figsize=(12, 7))

colors = {
    'cot': '#2E86AB',           # Blue
    'nocot': '#E63946',         # Red
    'nocurriculum': '#7B2D8E',  # Purple
}

# Plot CoT
if cot_data:
    phases_cot, means_cot, stds_cot = compute_mean_std(cot_data, 'accuracies')
    ax.plot(phases_cot, means_cot, 'o-', color=colors['cot'], linewidth=2.5, 
            markersize=8, label=f'CoT (n={len(cot_data)})')
    ax.fill_between(phases_cot, means_cot - stds_cot, means_cot + stds_cot, 
                    alpha=0.2, color=colors['cot'])

# Plot No CoT
if nocot_data:
    phases_nocot, means_nocot, stds_nocot = compute_mean_std(nocot_data, 'accuracies')
    ax.plot(phases_nocot, means_nocot, 's-', color=colors['nocot'], linewidth=2.5, 
            markersize=8, label=f'No CoT (n={len(nocot_data)})')
    ax.fill_between(phases_nocot, means_nocot - stds_nocot, means_nocot + stds_nocot, 
                    alpha=0.2, color=colors['nocot'])

# Plot No Curriculum (horizontal line since it only trains on full n_bits)
if nocurriculum_data:
    phases_nocurr, means_nocurr, stds_nocurr = compute_mean_std(nocurriculum_data, 'accuracies')
    # Draw as horizontal dashed line across all phases
    if phases_cot:
        x_range = [min(phases_cot), max(phases_cot)]
    elif phases_nocot:
        x_range = [min(phases_nocot), max(phases_nocot)]
    else:
        x_range = [1, 8]
    mean_acc = means_nocurr[0] if len(means_nocurr) > 0 else 50
    std_acc = stds_nocurr[0] if len(stds_nocurr) > 0 else 0
    ax.axhline(y=mean_acc, color=colors['nocurriculum'], linewidth=2.5, linestyle='--',
               label=f'No Curriculum (n={len(nocurriculum_data)})')
    ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=colors['nocurriculum'])

ax.set_xlabel('Phase (k)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Comparison: CoT vs No CoT vs No Curriculum\n(batch=128, n_layers=1, k_phases=8)', fontsize=14)
ax.legend(fontsize=11, loc='lower left')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'accuracy_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved: accuracy_comparison.png/pdf")

# ==================== Plot 2: All Trajectories with Mean ====================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# CoT trajectories
ax = axes[0]
cmap = plt.cm.tab20
if cot_data:
    cot_data_sorted = sorted(cot_data, key=lambda x: x['seed'] if x['seed'] else 999)
    for i, d in enumerate(cot_data_sorted):
        color = cmap(i % 20)
        ax.plot(d['phases'], d['accuracies'], '-', color=color, 
                linewidth=1, alpha=0.5)
        if d['phases'] and d['accuracies']:
            ax.annotate(f"s{d['seed']}", (d['phases'][-1] + 0.15, d['accuracies'][-1]), 
                       fontsize=7, color=color, alpha=0.8)
    
    ax.plot(phases_cot, means_cot, 'o-', color='black', linewidth=3, 
            markersize=8, label='Mean', zorder=10)
    ax.fill_between(phases_cot, means_cot - stds_cot, means_cot + stds_cot, 
                    alpha=0.3, color='gray')

ax.set_xlabel('Phase (k)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('CoT: All Trajectories', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3)

# No CoT trajectories
ax = axes[1]
if nocot_data:
    nocot_data_sorted = sorted(nocot_data, key=lambda x: x['seed'] if x['seed'] else 999)
    for i, d in enumerate(nocot_data_sorted):
        color = cmap(i % 20)
        ax.plot(d['phases'], d['accuracies'], '-', color=color, 
                linewidth=1, alpha=0.5)
        if d['phases'] and d['accuracies']:
            ax.annotate(f"s{d['seed']}", (d['phases'][-1] + 0.15, d['accuracies'][-1]), 
                       fontsize=7, color=color, alpha=0.8)
    
    ax.plot(phases_nocot, means_nocot, 'o-', color='black', linewidth=3, 
            markersize=8, label='Mean', zorder=10)
    ax.fill_between(phases_nocot, means_nocot - stds_nocot, means_nocot + stds_nocot, 
                    alpha=0.3, color='gray')

ax.set_xlabel('Phase (k)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('No CoT: All Trajectories', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3)

# No Curriculum trajectories (all at single phase 20)
ax = axes[2]
if nocurriculum_data:
    nocurriculum_data_sorted = sorted(nocurriculum_data, key=lambda x: x['seed'] if x['seed'] else 999)
    for i, d in enumerate(nocurriculum_data_sorted):
        color = cmap(i % 20)
        # Plot as scatter point at phase 20
        if d['accuracies']:
            final_acc = d['accuracies'][-1]
            ax.scatter([20], [final_acc], color=color, s=50, alpha=0.7)
            ax.annotate(f"s{d['seed']}", (20.3, final_acc), fontsize=7, color=color, alpha=0.8)
    
    if phases_nocurr is not None and len(means_nocurr) > 0:
        ax.axhline(y=means_nocurr[0], color='black', linewidth=3, label='Mean', zorder=10)
        ax.axhspan(means_nocurr[0] - stds_nocurr[0], means_nocurr[0] + stds_nocurr[0], 
                   alpha=0.3, color='gray')

ax.set_xlabel('Phase (k)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('No Curriculum: Final Accuracy (phase=20)', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim([0, 105])
ax.set_xlim([18, 22])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'accuracy_all_trajectories.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'accuracy_all_trajectories.pdf', bbox_inches='tight')
plt.close()
print("Saved: accuracy_all_trajectories.png/pdf")

# ==================== Plot 3: Eval Loss Comparison ====================
fig, ax = plt.subplots(figsize=(12, 7))

if cot_data:
    phases_cot, means_cot_loss, stds_cot_loss = compute_mean_std(cot_data, 'losses')
    ax.plot(phases_cot, means_cot_loss, 'o-', color=colors['cot'], linewidth=2.5, 
            markersize=8, label=f'CoT (n={len(cot_data)})')
    ax.fill_between(phases_cot, 
                    np.maximum(0, means_cot_loss - stds_cot_loss), 
                    means_cot_loss + stds_cot_loss, 
                    alpha=0.2, color=colors['cot'])

if nocot_data:
    phases_nocot, means_nocot_loss, stds_nocot_loss = compute_mean_std(nocot_data, 'losses')
    ax.plot(phases_nocot, means_nocot_loss, 's-', color=colors['nocot'], linewidth=2.5, 
            markersize=8, label=f'No CoT (n={len(nocot_data)})')
    ax.fill_between(phases_nocot, 
                    np.maximum(0, means_nocot_loss - stds_nocot_loss), 
                    means_nocot_loss + stds_nocot_loss, 
                    alpha=0.2, color=colors['nocot'])

# No Curriculum loss as horizontal line
if nocurriculum_data:
    phases_nocurr_loss, means_nocurr_loss, stds_nocurr_loss = compute_mean_std(nocurriculum_data, 'losses')
    if means_nocurr_loss is not None and len(means_nocurr_loss) > 0:
        mean_loss = means_nocurr_loss[0]
        std_loss = stds_nocurr_loss[0]
        ax.axhline(y=mean_loss, color=colors['nocurriculum'], linewidth=2.5, linestyle='--',
                   label=f'No Curriculum (n={len(nocurriculum_data)})')
        ax.axhspan(max(0, mean_loss - std_loss), mean_loss + std_loss, 
                   alpha=0.15, color=colors['nocurriculum'])

ax.set_xlabel('Phase (k)', fontsize=12)
ax.set_ylabel('Eval Loss', fontsize=12)
ax.set_title('Evaluation Loss Comparison: CoT vs No CoT vs No Curriculum\n(batch=128, n_layers=1, k_phases=8)', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'eval_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'eval_loss_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved: eval_loss_comparison.png/pdf")

# Save summary stats
summary = {}
if cot_data:
    summary['cot'] = {
        'n_runs': len(cot_data),
        'phases': phases_cot,
        'mean_accuracy': means_cot.tolist() if hasattr(means_cot, 'tolist') else list(means_cot),
        'std_accuracy': stds_cot.tolist() if hasattr(stds_cot, 'tolist') else list(stds_cot),
        'mean_loss': means_cot_loss.tolist() if hasattr(means_cot_loss, 'tolist') else list(means_cot_loss),
        'std_loss': stds_cot_loss.tolist() if hasattr(stds_cot_loss, 'tolist') else list(stds_cot_loss)
    }
if nocot_data:
    summary['nocot'] = {
        'n_runs': len(nocot_data),
        'phases': phases_nocot,
        'mean_accuracy': means_nocot.tolist() if hasattr(means_nocot, 'tolist') else list(means_nocot),
        'std_accuracy': stds_nocot.tolist() if hasattr(stds_nocot, 'tolist') else list(stds_nocot),
        'mean_loss': means_nocot_loss.tolist() if hasattr(means_nocot_loss, 'tolist') else list(means_nocot_loss),
        'std_loss': stds_nocot_loss.tolist() if hasattr(stds_nocot_loss, 'tolist') else list(stds_nocot_loss)
    }
if nocurriculum_data:
    summary['nocurriculum'] = {
        'n_runs': len(nocurriculum_data),
        'phases': phases_nocurr if phases_nocurr else [20],
        'mean_accuracy': means_nocurr.tolist() if hasattr(means_nocurr, 'tolist') else list(means_nocurr) if means_nocurr is not None else [],
        'std_accuracy': stds_nocurr.tolist() if hasattr(stds_nocurr, 'tolist') else list(stds_nocurr) if stds_nocurr is not None else [],
        'mean_loss': means_nocurr_loss.tolist() if hasattr(means_nocurr_loss, 'tolist') else list(means_nocurr_loss) if means_nocurr_loss is not None else [],
        'std_loss': stds_nocurr_loss.tolist() if hasattr(stds_nocurr_loss, 'tolist') else list(stds_nocurr_loss) if stds_nocurr_loss is not None else []
    }

with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: summary_stats.json")

print("\nDone!")
