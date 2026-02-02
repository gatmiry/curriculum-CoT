#!/usr/bin/env python3
"""
Plot comparison of eval loss per phase for both sweeps:
- train_parity_2 (num_cot_tokens=1)
- train_parity_2_cot (num_cot_tokens=phase)
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP1_DIR = os.path.join(SCRIPT_DIR, "outputs")
SWEEP2_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "train_parity_2_cot_sweep", "outputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "final_plots")

def load_phase_results(sweep_dir):
    """Load phase results from all runs in a sweep directory."""
    all_results = []
    
    # Find all training_data JSON files
    for run_dir in sorted(glob.glob(os.path.join(sweep_dir, "*_seed_*"))):
        json_files = glob.glob(os.path.join(run_dir, "training_data_*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                phase_results = data.get('phase_results', [])
                if phase_results:
                    all_results.append(phase_results)
    
    return all_results

def extract_losses_per_phase(all_results):
    """Extract eval losses per phase from all runs."""
    if not all_results:
        return {}, {}
    
    # Get number of phases from first result
    num_phases = len(all_results[0])
    
    losses_per_phase = {i: [] for i in range(1, num_phases + 1)}
    accuracies_per_phase = {i: [] for i in range(1, num_phases + 1)}
    
    for run_results in all_results:
        for result in run_results:
            phase = result['phase']
            loss = result['loss']
            accuracy = result['accuracy']
            losses_per_phase[phase].append(loss)
            accuracies_per_phase[phase].append(accuracy)
    
    return losses_per_phase, accuracies_per_phase

def compute_stats(values_per_phase):
    """Compute mean and std for each phase."""
    phases = sorted(values_per_phase.keys())
    means = []
    stds = []
    
    for phase in phases:
        values = values_per_phase[phase]
        if values:
            means.append(np.mean(values))
            stds.append(np.std(values))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return phases, np.array(means), np.array(stds)

def main():
    print("Loading sweep 1 (num_cot_tokens=1)...")
    sweep1_results = load_phase_results(SWEEP1_DIR)
    print(f"  Found {len(sweep1_results)} runs")
    
    print("Loading sweep 2 (num_cot_tokens=phase)...")
    sweep2_results = load_phase_results(SWEEP2_DIR)
    print(f"  Found {len(sweep2_results)} runs")
    
    # Extract losses per phase
    sweep1_losses, sweep1_acc = extract_losses_per_phase(sweep1_results)
    sweep2_losses, sweep2_acc = extract_losses_per_phase(sweep2_results)
    
    # Compute statistics
    phases1, means1, stds1 = compute_stats(sweep1_losses)
    phases2, means2, stds2 = compute_stats(sweep2_losses)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot sweep 1 (num_cot_tokens=1)
    ax.plot(phases1, means1, 'o-', color='#2ecc71', linewidth=2, markersize=8, 
            label='num_cot_tokens = 1', alpha=0.9)
    ax.fill_between(phases1, means1 - stds1, means1 + stds1, 
                    color='#2ecc71', alpha=0.2)
    
    # Plot sweep 2 (num_cot_tokens=phase)
    ax.plot(phases2, means2, 's-', color='#e74c3c', linewidth=2, markersize=8,
            label='num_cot_tokens = phase', alpha=0.9)
    ax.fill_between(phases2, means2 - stds2, means2 + stds2,
                    color='#e74c3c', alpha=0.2)
    
    # Styling
    ax.set_xlabel('Phase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Eval Loss (MSE)', fontsize=14, fontweight='bold')
    ax.set_title('Eval Loss per Phase: Comparison of CoT Token Strategies\n(Shaded region = ±1 std across 20 seeds)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    # ax.set_yscale('log')  # Use linear scale instead
    
    # Set x-axis ticks
    ax.set_xticks(phases1)
    ax.set_xlim(0.5, max(phases1) + 0.5)
    
    # Add horizontal line at target loss
    ax.axhline(y=0.02, color='gray', linestyle='--', alpha=0.7, label='Target loss (0.02)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'eval_loss_per_phase_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF for high quality
    pdf_path = os.path.join(OUTPUT_DIR, 'eval_loss_per_phase_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()
    
    # ========== ACCURACY PLOT ==========
    # Compute accuracy statistics
    acc_phases1, acc_means1, acc_stds1 = compute_stats(sweep1_acc)
    acc_phases2, acc_means2, acc_stds2 = compute_stats(sweep2_acc)
    
    # Create accuracy plot
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Plot sweep 1 (num_cot_tokens=1)
    ax2.plot(acc_phases1, acc_means1, 'o-', color='#2ecc71', linewidth=2, markersize=8, 
            label='num_cot_tokens = 1', alpha=0.9)
    ax2.fill_between(acc_phases1, acc_means1 - acc_stds1, acc_means1 + acc_stds1, 
                    color='#2ecc71', alpha=0.2)
    
    # Plot sweep 2 (num_cot_tokens=phase)
    ax2.plot(acc_phases2, acc_means2, 's-', color='#e74c3c', linewidth=2, markersize=8,
            label='num_cot_tokens = phase', alpha=0.9)
    ax2.fill_between(acc_phases2, acc_means2 - acc_stds2, acc_means2 + acc_stds2,
                    color='#e74c3c', alpha=0.2)
    
    # Styling
    ax2.set_xlabel('Phase', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_title('Accuracy per Phase: Comparison of CoT Token Strategies\n(Shaded region = ±1 std across 20 seeds)', 
                 fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Set axis limits
    ax2.set_xticks(acc_phases1)
    ax2.set_xlim(0.5, max(acc_phases1) + 0.5)
    ax2.set_ylim(0, 1.05)
    
    # Add horizontal lines for reference
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Random chance (0.5)')
    
    plt.tight_layout()
    
    # Save accuracy plot
    acc_output_path = os.path.join(OUTPUT_DIR, 'accuracy_per_phase_comparison.png')
    plt.savefig(acc_output_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy plot saved to: {acc_output_path}")
    
    acc_pdf_path = os.path.join(OUTPUT_DIR, 'accuracy_per_phase_comparison.pdf')
    plt.savefig(acc_pdf_path, bbox_inches='tight')
    print(f"Accuracy PDF saved to: {acc_pdf_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - EVAL LOSS")
    print("="*80)
    print(f"\n{'Phase':<8} {'CoT=1 (mean±std)':<25} {'CoT=phase (mean±std)':<25}")
    print("-"*60)
    for i, phase in enumerate(phases1):
        if i < len(means2):
            print(f"{phase:<8} {means1[i]:.4f} ± {stds1[i]:.4f}         {means2[i]:.4f} ± {stds2[i]:.4f}")
        else:
            print(f"{phase:<8} {means1[i]:.4f} ± {stds1[i]:.4f}         N/A")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - ACCURACY")
    print("="*80)
    print(f"\n{'Phase':<8} {'CoT=1 (mean±std)':<25} {'CoT=phase (mean±std)':<25}")
    print("-"*60)
    for i, phase in enumerate(acc_phases1):
        if i < len(acc_means2):
            print(f"{phase:<8} {acc_means1[i]:.4f} ± {acc_stds1[i]:.4f}         {acc_means2[i]:.4f} ± {acc_stds2[i]:.4f}")
        else:
            print(f"{phase:<8} {acc_means1[i]:.4f} ± {acc_stds1[i]:.4f}         N/A")
    
    # Save summary to JSON
    summary = {
        'sweep1_num_cot_tokens_1': {
            'phases': phases1,
            'mean_loss': means1.tolist(),
            'std_loss': stds1.tolist(),
            'num_seeds': len(sweep1_results)
        },
        'sweep2_num_cot_tokens_phase': {
            'phases': phases2,
            'mean_loss': means2.tolist(),
            'std_loss': stds2.tolist(),
            'num_seeds': len(sweep2_results)
        }
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'summary_stats.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == '__main__':
    main()

