#!/usr/bin/env python3
"""
Generate single-column width plot for two-column paper format.
Standard single-column width: 3.5 inches (8.89 cm)
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

def load_sweep_data_with_seeds(sweep_dir):
    """Load accuracy data from all runs, returning seed info."""
    all_data = []
    
    if not sweep_dir.exists():
        return None
    
    run_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]
    
    for run_dir in sorted(run_dirs):
        folder_name = run_dir.name
        seed = None
        if '_seed_' in folder_name:
            seed = int(folder_name.split('_seed_')[-1])
        
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
print("Loading data...")
cot_data = load_sweep_data_with_seeds(COT_DIR)
nocot_data = load_sweep_data_with_seeds(NOCOT_DIR)
nocurriculum_data = load_sweep_data_with_seeds(NOCURRICULUM_DIR)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

colors = {
    'cot': '#2E86AB',           # Blue
    'nocot': '#E63946',         # Red
    'nocurriculum': '#7B2D8E',  # Purple
}

# ==================== Single Column Accuracy Plot ====================
# Standard single-column width: 3.5 inches, height ~2.8 inches for good aspect ratio
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Plot CoT
if cot_data:
    phases_cot, means_cot, stds_cot = compute_mean_std(cot_data, 'accuracies')
    ax.plot(phases_cot, means_cot, 'o-', color=colors['cot'], linewidth=1.5, 
            markersize=4, label='CoT with Curriculum')
    ax.fill_between(phases_cot, means_cot - stds_cot, means_cot + stds_cot, 
                    alpha=0.2, color=colors['cot'])

# Plot No CoT
if nocot_data:
    phases_nocot, means_nocot, stds_nocot = compute_mean_std(nocot_data, 'accuracies')
    ax.plot(phases_nocot, means_nocot, 's-', color=colors['nocot'], linewidth=1.5, 
            markersize=4, label='No CoT with Curriculum')
    ax.fill_between(phases_nocot, means_nocot - stds_nocot, means_nocot + stds_nocot, 
                    alpha=0.2, color=colors['nocot'])

# Plot No Curriculum
if nocurriculum_data:
    phases_nocurr, means_nocurr, stds_nocurr = compute_mean_std(nocurriculum_data, 'accuracies')
    mean_acc = means_nocurr[0] if len(means_nocurr) > 0 else 50
    std_acc = stds_nocurr[0] if len(stds_nocurr) > 0 else 0
    ax.axhline(y=mean_acc, color=colors['nocurriculum'], linewidth=1.5, linestyle='--',
               label='No CoT No Curriculum')
    ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=colors['nocurriculum'])

ax.set_xlabel('Phase (k)')
ax.set_ylabel('Accuracy (%)')
ax.legend(loc='lower left', framealpha=0.9)
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout(pad=0.5)
plt.savefig(OUTPUT_DIR / 'accuracy_comparison_single_col.png', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'accuracy_comparison_single_col.pdf', bbox_inches='tight')
plt.close()
print("Saved: accuracy_comparison_single_col.png/pdf")

# ==================== Single Column Eval Loss Plot ====================
# Slightly different aesthetic: warmer tones, different markers, subtle background
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Slightly different color palette for loss plot (teal/orange theme)
loss_colors = {
    'cot': '#2A9D8F',           # Teal
    'nocot': '#E76F51',         # Coral/burnt orange  
    'nocurriculum': '#5E2750',  # Deeper purple
}

# Add subtle warm background tint
ax.set_facecolor('#FDFBF7')

if cot_data:
    phases_cot, means_cot_loss, stds_cot_loss = compute_mean_std(cot_data, 'losses')
    ax.plot(phases_cot, means_cot_loss, '^-', color=loss_colors['cot'], linewidth=1.5, 
            markersize=5, label='CoT with Curriculum', markeredgecolor='white', markeredgewidth=0.3)
    ax.fill_between(phases_cot, 
                    np.maximum(0, means_cot_loss - stds_cot_loss), 
                    means_cot_loss + stds_cot_loss, 
                    alpha=0.15, color=loss_colors['cot'])

if nocot_data:
    phases_nocot, means_nocot_loss, stds_nocot_loss = compute_mean_std(nocot_data, 'losses')
    ax.plot(phases_nocot, means_nocot_loss, 'D-', color=loss_colors['nocot'], linewidth=1.5, 
            markersize=4, label='No CoT with Curriculum', markeredgecolor='white', markeredgewidth=0.3)
    ax.fill_between(phases_nocot, 
                    np.maximum(0, means_nocot_loss - stds_nocot_loss), 
                    means_nocot_loss + stds_nocot_loss, 
                    alpha=0.15, color=loss_colors['nocot'])

if nocurriculum_data:
    phases_nocurr_loss, means_nocurr_loss, stds_nocurr_loss = compute_mean_std(nocurriculum_data, 'losses')
    if means_nocurr_loss is not None and len(means_nocurr_loss) > 0:
        mean_loss = means_nocurr_loss[0]
        std_loss = stds_nocurr_loss[0]
        ax.axhline(y=mean_loss, color=loss_colors['nocurriculum'], linewidth=1.5, linestyle=':',
                   label='No CoT No Curriculum')
        ax.axhspan(max(0, mean_loss - std_loss), mean_loss + std_loss, 
                   alpha=0.12, color=loss_colors['nocurriculum'])

ax.set_xlabel('Phase (k)')
ax.set_ylabel('Eval Loss')
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, alpha=0.4, linewidth=0.4, linestyle=':')

plt.tight_layout(pad=0.5)
plt.savefig(OUTPUT_DIR / 'eval_loss_comparison_single_col.png', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'eval_loss_comparison_single_col.pdf', bbox_inches='tight')
plt.close()
print("Saved: eval_loss_comparison_single_col.png/pdf")

print("\nDone! Single-column plots saved.")
