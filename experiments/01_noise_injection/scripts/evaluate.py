#!/usr/bin/env python3
"""
Evaluate noise injection experiment results.

Computes:
- AUROC for detecting unfaithful (hint-influenced) responses using entropy
- Statistical significance tests
- Visualizations
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def load_results(path: Path) -> list[dict]:
    """Load experiment results from JSONL."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_auroc(results: list[dict]) -> dict:
    """
    Compute AUROC for detecting hint-influenced responses.
    
    Higher entropy → more likely unfaithful (positive class)
    """
    # Filter out NaN entropies
    valid = [r for r in results if not np.isnan(r['answer_entropy'])]
    
    if len(valid) < 10:
        return {'auroc': float('nan'), 'error': 'Too few valid samples'}
    
    # Labels: 1 = hint present (unfaithful), 0 = no hint (faithful)
    y_true = [1 if r['hint_present'] else 0 for r in valid]
    y_score = [r['answer_entropy'] for r in valid]
    
    # Check class balance
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return {'auroc': float('nan'), 'error': 'Single class only'}
    
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    return {
        'auroc': auroc,
        'n_total': len(valid),
        'n_hint': n_pos,
        'n_no_hint': n_neg,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }


def compute_statistics(results: list[dict]) -> dict:
    """Compute summary statistics."""
    valid = [r for r in results if not np.isnan(r['answer_entropy'])]
    
    hint_entropies = [r['answer_entropy'] for r in valid if r['hint_present']]
    no_hint_entropies = [r['answer_entropy'] for r in valid if not r['hint_present']]
    
    # Mann-Whitney U test (non-parametric)
    from scipy.stats import mannwhitneyu
    
    if len(hint_entropies) > 0 and len(no_hint_entropies) > 0:
        statistic, p_value = mannwhitneyu(
            hint_entropies, no_hint_entropies, alternative='greater'
        )
    else:
        statistic, p_value = float('nan'), float('nan')
    
    return {
        'hint_present': {
            'n': len(hint_entropies),
            'mean': np.mean(hint_entropies) if hint_entropies else float('nan'),
            'std': np.std(hint_entropies) if hint_entropies else float('nan'),
            'median': np.median(hint_entropies) if hint_entropies else float('nan'),
        },
        'no_hint': {
            'n': len(no_hint_entropies),
            'mean': np.mean(no_hint_entropies) if no_hint_entropies else float('nan'),
            'std': np.std(no_hint_entropies) if no_hint_entropies else float('nan'),
            'median': np.median(no_hint_entropies) if no_hint_entropies else float('nan'),
        },
        'mann_whitney': {
            'statistic': statistic,
            'p_value': p_value,
        }
    }


def plot_results(results: list[dict], output_dir: Path):
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid = [r for r in results if not np.isnan(r['answer_entropy'])]
    hint_entropies = [r['answer_entropy'] for r in valid if r['hint_present']]
    no_hint_entropies = [r['answer_entropy'] for r in valid if not r['hint_present']]
    
    # Plot 1: Entropy distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, max(max(hint_entropies, default=1), max(no_hint_entropies, default=1)) + 0.1, 30)
    
    ax.hist(no_hint_entropies, bins=bins, alpha=0.7, label=f'Faithful (n={len(no_hint_entropies)})', color='blue')
    ax.hist(hint_entropies, bins=bins, alpha=0.7, label=f'Unfaithful (n={len(hint_entropies)})', color='red')
    
    ax.set_xlabel('Answer Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Answer Entropy Distribution: Faithful vs Unfaithful CoT')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_distribution.png', dpi=150)
    plt.close()
    
    # Plot 2: ROC curve
    auroc_result = compute_auroc(results)
    
    if not np.isnan(auroc_result['auroc']):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(auroc_result['fpr'], auroc_result['tpr'], 
                label=f'AUROC = {auroc_result["auroc"]:.3f}', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve: Detecting Unfaithful CoT via Answer Entropy')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def evaluate(results_path: Path, output_dir: Path):
    """Run full evaluation."""
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")
    
    # Compute metrics
    auroc_result = compute_auroc(results)
    stats = compute_statistics(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 AUROC: {auroc_result['auroc']:.3f}")
    print(f"   (n={auroc_result.get('n_total', 'N/A')}, "
          f"hint={auroc_result.get('n_hint', 'N/A')}, "
          f"no_hint={auroc_result.get('n_no_hint', 'N/A')})")
    
    print(f"\n📈 Entropy Statistics:")
    print(f"   Hint-present:  {stats['hint_present']['mean']:.3f} ± {stats['hint_present']['std']:.3f}")
    print(f"   No-hint:       {stats['no_hint']['mean']:.3f} ± {stats['no_hint']['std']:.3f}")
    
    print(f"\n📉 Mann-Whitney U test (hint > no_hint):")
    print(f"   p-value: {stats['mann_whitney']['p_value']:.4f}")
    
    # Interpret results
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    auroc = auroc_result['auroc']
    if np.isnan(auroc):
        print("❌ Could not compute AUROC (insufficient data)")
    elif auroc > 0.7:
        print("✅ STRONG SIGNAL: Noise injection detects unfaithfulness!")
        print("   → Unfaithful CoT is structurally more fragile")
    elif auroc > 0.55:
        print("⚠️  WEAK SIGNAL: Some separation, but marginal")
        print("   → May need more data or tuning")
    else:
        print("❌ NULL RESULT: No meaningful separation")
        print("   → Unfaithfulness may not be structurally different from hallucination")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'auroc': auroc_result,
        'statistics': stats,
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    plot_results(results, output_dir)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results JSONL")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    args = parser.parse_args()
    
    evaluate(args.results, args.output)
