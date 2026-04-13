#!/usr/bin/env python3
"""Generate figures for the paper."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

# Figure 1: Layer Sweep
def fig_layer_sweep():
    layers = list(range(15))
    cure_rates = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 80, 95]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1f77b4' if r < 50 else '#d62728' for r in cure_rates]
    bars = ax.bar(layers, cure_rates, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, label='Baseline (40%)')
    ax.set_xlabel('KV Cache Entry')
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Sycophancy Cure Rate by KV Cache Entry (Gemma-4 2B)')
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Annotate key layers
    ax.annotate('Entry 13\n80%', xy=(13, 80), xytext=(11, 90),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')
    ax.annotate('Entry 14\n95%', xy=(14, 95), xytext=(14, 85),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_layer_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_layer_sweep.png")

# Figure 2: K vs V Decomposition
def fig_kv_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mixed set
    conditions = ['Baseline', 'K-only', 'V-only', 'K+V']
    mixed_rates = [40, 39, 73, 80]
    mixed_ci = [0, 9, 8.5, 8]  # approximate CI widths
    
    ax = axes[0]
    colors = ['#7f7f7f', '#ff7f0e', '#2ca02c', '#9467bd']
    bars = ax.bar(conditions, mixed_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(conditions, mixed_rates, yerr=mixed_ci, fmt='none', color='black', capsize=5)
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Mixed Question Set (n=100)')
    ax.set_ylim(0, 100)
    
    # Add significance markers
    ax.annotate('n.s.', xy=(1, 48), ha='center', fontsize=10)
    ax.annotate('***', xy=(2, 82), ha='center', fontsize=12)
    
    # Hard set
    hard_rates = [40, 20, 72, 75]
    hard_ci = [9.5, 8, 9, 9]
    
    ax = axes[1]
    bars = ax.bar(conditions, hard_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(conditions, hard_rates, yerr=hard_ci, fmt='none', color='black', capsize=5)
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Hard Question Set (n=100)')
    ax.set_ylim(0, 100)
    
    # Add significance markers
    ax.annotate('***\n(harmful)', xy=(1, 10), ha='center', fontsize=10, color='red')
    ax.annotate('***', xy=(2, 82), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_kv_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_kv_decomposition.png")

# Figure 3: Cross-Question Patching
def fig_cross_question():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    conditions = ['Baseline', 'Same-Q V', 'Cross-Q V\n(Entity)', 'Cross-Q V\n(Date)']
    rates = [40, 73, 74, 23]
    ci_low = [31, 64, 65, 16]
    ci_high = [50, 81, 82, 32]
    ci = [(r - l, h - r) for r, l, h in zip(rates, ci_low, ci_high)]
    
    colors = ['#7f7f7f', '#2ca02c', '#2ca02c', '#d62728']
    bars = ax.bar(conditions, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(conditions, rates, yerr=np.array(ci).T, fmt='none', color='black', capsize=5)
    
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Cross-Question V Patching: Transfer Depends on Answer Type (n=100)')
    ax.set_ylim(0, 100)
    
    # Annotations
    ax.annotate('Same rate!', xy=(1.5, 77), ha='center', fontsize=10,
                arrowprops=dict(arrowstyle='-', color='green', lw=2),
                xytext=(1.5, 90))
    ax.annotate('Harmful', xy=(3, 23), xytext=(3, 8), ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_cross_question.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_cross_question.png")

# Figure 4: Bidirectional Causality
def fig_bidirectional():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conditions = ['Clean prompt\n+ Clean KV', 'Clean prompt\n+ Hint KV', 'Hint prompt\n+ Hint KV']
    rates = [9, 41, 60]
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax.bar(conditions, rates, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Sycophancy Rate (%)')
    ax.set_title('Bidirectional Causality: KV Contamination Induces Sycophancy')
    ax.set_ylim(0, 80)
    
    # Add annotation for causal contribution
    ax.annotate('', xy=(1, 41), xytext=(0, 9),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(0.5, 25, '+32pp\n(63% of\ntotal effect)', ha='center', fontsize=10)
    
    ax.annotate('', xy=(2, 60), xytext=(1, 41),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(1.5, 50, '+19pp', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_bidirectional.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_bidirectional.png")

# Figure 5: Interpolation Gradient (recreate cleaner version)
def fig_interpolation():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Data from interpolation experiment
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    cure_rates = [46, 40, 33, 26, 20]
    
    ax.plot(alphas, cure_rates, 'o-', color='#1f77b4', linewidth=2, markersize=10)
    ax.fill_between(alphas, [r-8 for r in cure_rates], [r+8 for r in cure_rates], 
                    alpha=0.2, color='#1f77b4')
    
    ax.set_xlabel('α (0 = Pure Entity V, 1 = Pure Date V)')
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Behavioral Effect Scales Continuously with V Vector Mixture')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 60)
    ax.set_xticks(alphas)
    
    # Annotations
    ax.annotate('Entity donors\n(beneficial)', xy=(0, 46), xytext=(0.15, 52),
                fontsize=10, ha='center')
    ax.annotate('Date donors\n(harmful)', xy=(1, 20), xytext=(0.85, 12),
                fontsize=10, ha='center')
    ax.annotate('26pp monotonic\ngradient', xy=(0.5, 33), xytext=(0.5, 45),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_interpolation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_interpolation.png")

# Figure 6: Cross-Architecture Comparison
def fig_cross_arch():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(3)
    width = 0.35
    
    gemma = [73, 39, 80]
    qwen = [50, 25, 40]  # coherent-only rates
    
    bars1 = ax.bar(x - width/2, gemma, width, label='Gemma-4 2B (n=100)', color='#1f77b4')
    bars2 = ax.bar(x + width/2, qwen, width, label='Qwen2.5-3B (n=20, coherent)', color='#ff7f0e')
    
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Cure Rate (%)')
    ax.set_title('Cross-Architecture Comparison: V-Advantage Replicates')
    ax.set_xticks(x)
    ax.set_xticklabels(['V-only', 'K-only', 'Baseline'])
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add ratio annotations
    ax.annotate('2× advantage', xy=(0, 50), xytext=(0.3, 65),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e'),
                fontsize=10, color='#ff7f0e')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_cross_arch.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig6_cross_arch.png")

if __name__ == "__main__":
    fig_layer_sweep()
    fig_kv_decomposition()
    fig_cross_question()
    fig_bidirectional()
    fig_interpolation()
    fig_cross_arch()
    print(f"\nAll figures saved to: {output_dir}")
