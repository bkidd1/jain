#!/usr/bin/env python3
"""
Generate ALL figures for the paper, including method diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def create_method_diagram():
    """
    Figure 1: Method overview diagram showing the full pipeline.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    blue = '#3498db'
    orange = '#e67e22'
    green = '#27ae60'
    gray = '#7f8c8d'
    light_blue = '#d5e8f7'
    light_orange = '#fdebd0'
    light_green = '#d5f5e3'
    
    # === TRAINING PHASE (top) ===
    # Background box
    training_bg = FancyBboxPatch((0.3, 4.2), 11.4, 3.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2)
    ax.add_patch(training_bg)
    ax.text(6, 7.4, 'TRAINING PHASE', fontsize=14, fontweight='bold', 
            ha='center', color='#495057')
    
    # Llama box
    llama_box = FancyBboxPatch((0.8, 5.2), 2.2, 1.5, 
                                boxstyle="round,pad=0.05",
                                facecolor=light_blue, edgecolor=blue, linewidth=2)
    ax.add_patch(llama_box)
    ax.text(1.9, 6.1, 'Llama 3.1 8B', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.9, 5.6, '(Open Model)', fontsize=9, ha='center', color='gray')
    
    # Arrow 1
    ax.annotate('', xy=(3.3, 5.95), xytext=(3.0, 5.95),
                arrowprops=dict(arrowstyle='->', color=blue, lw=2))
    
    # Logit Lens box
    lens_box = FancyBboxPatch((3.5, 5.2), 2.4, 1.5,
                               boxstyle="round,pad=0.05",
                               facecolor=light_blue, edgecolor=blue, linewidth=2)
    ax.add_patch(lens_box)
    ax.text(4.7, 6.1, 'Logit Lens +', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.7, 5.6, 'Act. Patching', fontsize=10, ha='center')
    
    # Arrow 2
    ax.annotate('', xy=(6.2, 5.95), xytext=(5.9, 5.95),
                arrowprops=dict(arrowstyle='->', color=blue, lw=2))
    
    # Ground Truth box
    gt_box = FancyBboxPatch((6.4, 5.0), 2.6, 1.9,
                             boxstyle="round,pad=0.05",
                             facecolor=light_green, edgecolor=green, linewidth=2)
    ax.add_patch(gt_box)
    ax.text(7.7, 6.35, 'Ground Truth', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.7, 5.9, 'Traces', fontsize=10, ha='center')
    ax.text(7.7, 5.35, 'Dallas→Texas→Austin', fontsize=8, ha='center', 
            color='#27ae60', style='italic')
    
    # Arrow 3 (down)
    ax.annotate('', xy=(7.7, 4.8), xytext=(7.7, 5.0),
                arrowprops=dict(arrowstyle='->', color=green, lw=2))
    
    # RTP Training box
    rtp_box = FancyBboxPatch((9.3, 5.2), 2.0, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor=light_orange, edgecolor=orange, linewidth=2)
    ax.add_patch(rtp_box)
    ax.text(10.3, 6.1, 'Train RTP', fontsize=10, fontweight='bold', ha='center')
    ax.text(10.3, 5.6, '(TinyLlama)', fontsize=9, ha='center', color='gray')
    
    # Arrow to RTP
    ax.annotate('', xy=(9.1, 5.95), xytext=(9.0, 5.95),
                arrowprops=dict(arrowstyle='->', color=orange, lw=2))
    
    # === INFERENCE PHASE (bottom) ===
    # Background box
    inference_bg = FancyBboxPatch((0.3, 0.5), 11.4, 3.3,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#fff9f0', edgecolor='#f5cba7', linewidth=2)
    ax.add_patch(inference_bg)
    ax.text(6, 3.5, 'INFERENCE PHASE (Black-Box)', fontsize=14, fontweight='bold',
            ha='center', color='#935116')
    
    # Mistral box (grayed out - closed model)
    mistral_box = FancyBboxPatch((0.8, 1.2), 2.2, 1.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#eaecee', edgecolor=gray, linewidth=2)
    ax.add_patch(mistral_box)
    ax.text(1.9, 2.1, 'Mistral 7B', fontsize=11, fontweight='bold', ha='center', color='#5d6d7e')
    ax.text(1.9, 1.6, '(or Closed Model)', fontsize=9, ha='center', color='gray')
    
    # Arrow
    ax.annotate('', xy=(3.3, 1.95), xytext=(3.0, 1.95),
                arrowprops=dict(arrowstyle='->', color=gray, lw=2))
    
    # Input/Output only box
    io_box = FancyBboxPatch((3.5, 1.2), 2.4, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#fdfefe', edgecolor=gray, linewidth=2)
    ax.add_patch(io_box)
    ax.text(4.7, 2.1, 'Input/Output', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.7, 1.6, 'Only', fontsize=10, ha='center', color='gray')
    
    # Arrow
    ax.annotate('', xy=(6.2, 1.95), xytext=(5.9, 1.95),
                arrowprops=dict(arrowstyle='->', color=orange, lw=2))
    
    # Trained RTP box
    rtp_infer_box = FancyBboxPatch((6.4, 1.2), 2.4, 1.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=light_orange, edgecolor=orange, linewidth=2)
    ax.add_patch(rtp_infer_box)
    ax.text(7.6, 2.1, 'Trained RTP', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.6, 1.6, '(from above)', fontsize=9, ha='center', color='gray')
    
    # Arrow from training to inference
    ax.annotate('', xy=(7.6, 2.7), xytext=(10.3, 4.5),
                arrowprops=dict(arrowstyle='->', color=orange, lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    
    # Arrow
    ax.annotate('', xy=(9.1, 1.95), xytext=(8.8, 1.95),
                arrowprops=dict(arrowstyle='->', color=green, lw=2))
    
    # Predicted Trace box
    pred_box = FancyBboxPatch((9.3, 1.0), 2.0, 1.9,
                               boxstyle="round,pad=0.05",
                               facecolor=light_green, edgecolor=green, linewidth=2)
    ax.add_patch(pred_box)
    ax.text(10.3, 2.35, 'Predicted', fontsize=10, fontweight='bold', ha='center')
    ax.text(10.3, 1.9, 'Trace', fontsize=10, ha='center')
    ax.text(10.3, 1.35, '40% F1!', fontsize=11, ha='center', 
            color='#27ae60', fontweight='bold')
    
    # Key insight box
    key_box = FancyBboxPatch((3.5, 0.1), 5.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor='#fef9e7', edgecolor='#f4d03f', linewidth=1)
    ax.add_patch(key_box)
    ax.text(6, 0.4, '🔑 No access to Mistral internals needed!', 
            fontsize=10, ha='center', fontweight='bold', color='#9a7d0a')
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig1_method_overview.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig("docs/figures/fig1_method_overview.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved fig1_method_overview")
    plt.close()


def create_logit_lens_heatmap():
    """
    Figure 2: Logit lens showing "Austin" emerging across layers.
    """
    layers = [8, 16, 24, 31]
    layer_labels = ["Layer 8\n(25%)", "Layer 16\n(50%)", "Layer 24\n(75%)", "Layer 31\n(Final)"]
    concepts = ["Austin", "Texas", "capital", "a", "the"]
    
    probs = np.array([
        [0.02, 0.08, 0.42, 0.16],
        [0.01, 0.05, 0.15, 0.08],
        [0.03, 0.04, 0.05, 0.04],
        [0.04, 0.02, 0.10, 0.16],
        [0.05, 0.03, 0.03, 0.02],
    ])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    
    for i in range(len(concepts)):
        for j in range(len(layers)):
            color = "white" if probs[i, j] > 0.25 else "black"
            ax.text(j, i, f"{probs[i,j]:.2f}", ha="center", va="center", 
                   color=color, fontsize=10, fontweight='bold' if probs[i,j] > 0.3 else 'normal')
    
    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Predicted Token", fontsize=12)
    ax.set_title('Logit Lens: "The capital of Texas is ___"', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability", fontsize=11)
    
    rect = plt.Rectangle((1.5, -0.5), 1, 1, fill=False, edgecolor='green', linewidth=3)
    ax.add_patch(rect)
    ax.annotate('Key: "Austin" peaks\nat layer 24', 
                xy=(2, 0), xytext=(2.8, 1.8),
                fontsize=10, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig2_logit_lens_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig("docs/figures/fig2_logit_lens_heatmap.pdf", bbox_inches='tight')
    print("✓ Saved fig2_logit_lens_heatmap")
    plt.close()


def create_training_loss_curve():
    """Figure 3: Training loss curve."""
    epochs = np.array([0.71, 1.0, 1.43, 2.0, 2.14, 2.86, 3.0, 3.57, 4.0, 4.29, 5.0])
    train_losses = [3.83, 3.04, 2.71, 1.75, 1.86, 1.30, 1.34, 1.00, 1.13, 0.91, 0.96]
    eval_epochs = [1.0, 2.0, 3.0, 4.0, 5.0]
    eval_losses = [3.04, 1.75, 1.34, 1.13, 1.10]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Training Loss', 
            marker='o', markersize=5, alpha=0.8)
    ax.plot(eval_epochs, eval_losses, 'r--', linewidth=2.5, label='Validation Loss',
            marker='s', markersize=7)
    
    ax.fill_between(epochs, train_losses, alpha=0.1, color='blue')
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("RTP Training Progress", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 4.5)
    
    ax.annotate('75% reduction', xy=(3, 1.0), xytext=(3.5, 2.0),
                fontsize=10, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig3_training_loss.png", dpi=150, bbox_inches='tight')
    plt.savefig("docs/figures/fig3_training_loss.pdf", bbox_inches='tight')
    print("✓ Saved fig3_training_loss")
    plt.close()


def create_transfer_f1_chart():
    """Figure 4: Cross-model transfer F1 scores."""
    prompts = ["Capital of\nTexas", "Dallas →\nstate", "Largest city\nin CA", 
               "Windows\ncompany", "Apple\nfounder"]
    f1_scores = [100, 100, 67, 67, 0]
    precision = [100, 100, 100, 100, 0]
    recall = [100, 100, 50, 50, 0]
    
    x = np.arange(len(prompts))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel("Test Prompt", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Cross-Model Transfer: Llama-trained RTP → Mistral", 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 115)
    
    ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(4.5, 43, 'Avg F1: 40%', fontsize=11, color='red', fontweight='bold')
    
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{int(height)}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig4_transfer_f1.png", dpi=150, bbox_inches='tight')
    plt.savefig("docs/figures/fig4_transfer_f1.pdf", bbox_inches='tight')
    print("✓ Saved fig4_transfer_f1")
    plt.close()


def create_concept_comparison():
    """Figure 5: Side-by-side concept comparison Llama vs Mistral."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    prompts = ["Capital of TX", "Dallas→state", "Largest CA", "Windows co.", "Apple founder"]
    llama_concepts = ["Austin", "Texas", "Los", "Microsoft", "Jobs"]
    mistral_concepts = ["Austin", "Texas", "Los", "Microsoft", "Apple"]
    matches = [True, True, True, True, False]
    
    # Llama side
    ax1 = axes[0]
    colors1 = ['#27ae60' if m else '#e74c3c' for m in matches]
    bars1 = ax1.barh(prompts, [1]*5, color=colors1, alpha=0.3)
    for i, (prompt, concept) in enumerate(zip(prompts, llama_concepts)):
        ax1.text(0.5, i, concept, ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_title("Llama 3.1 8B\n(Training Source)", fontsize=12, fontweight='bold', color='#3498db')
    ax1.set_xticks([])
    ax1.invert_yaxis()
    
    # Mistral side
    ax2 = axes[1]
    colors2 = ['#27ae60' if m else '#e74c3c' for m in matches]
    bars2 = ax2.barh(prompts, [1]*5, color=colors2, alpha=0.3)
    for i, (prompt, concept) in enumerate(zip(prompts, mistral_concepts)):
        ax2.text(0.5, i, concept, ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_title("Mistral 7B\n(Transfer Target)", fontsize=12, fontweight='bold', color='#9b59b6')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', alpha=0.5, label='Match (4/5)'),
                       Patch(facecolor='#e74c3c', alpha=0.5, label='Mismatch (1/5)')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, 
               bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle("Internal Concept Comparison: What Do Models \"Think\"?", 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig("docs/figures/fig5_concept_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig("docs/figures/fig5_concept_comparison.pdf", bbox_inches='tight')
    print("✓ Saved fig5_concept_comparison")
    plt.close()


def main():
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating ALL paper figures")
    print("="*60 + "\n")
    
    create_method_diagram()
    create_logit_lens_heatmap()
    create_training_loss_curve()
    create_transfer_f1_chart()
    create_concept_comparison()
    
    print("\n✓ All 5 figures saved to docs/figures/")


if __name__ == "__main__":
    main()
