#!/usr/bin/env python3
"""
Generate publication-quality interpolation gradient figure.
Shows monotonic cure rate decrease as V-vectors shift from entity to date donors.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('../data/results/30_interpolation_n100.json') as f:
    data = json.load(f)

# Extract data points
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
cure_rates = []
ci_lows = []
ci_highs = []

for alpha in alphas:
    key = f"alpha_{alpha:.2f}"
    r = data['results'][key]
    cure_rates.append(r['cure_rate'] * 100)  # Convert to percentage
    ci_lows.append(r['ci_low'] * 100)
    ci_highs.append(r['ci_high'] * 100)

cure_rates = np.array(cure_rates)
ci_lows = np.array(ci_lows)
ci_highs = np.array(ci_highs)
errors = np.array([cure_rates - ci_lows, ci_highs - cure_rates])

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot with error bars
ax.errorbar(alphas, cure_rates, yerr=errors, 
            fmt='o-', color='#2E86AB', linewidth=2.5, markersize=10,
            capsize=6, capthick=2, elinewidth=2,
            label='Cure rate (n=100 per point)')

# Add baseline reference
baseline = 40  # ~40% baseline from validation experiments
ax.axhline(y=baseline, color='#E94F37', linestyle='--', linewidth=1.5, 
           alpha=0.7, label=f'Sycophantic baseline (~{baseline}%)')

# Shade the "harmful" region (below baseline)
ax.axhspan(0, baseline, alpha=0.1, color='#E94F37')
ax.axhspan(baseline, 100, alpha=0.1, color='#2E86AB')

# Annotations
ax.annotate('Entity V-vectors\n(neutral)', xy=(0, 47), xytext=(0.08, 58),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.annotate('Date V-vectors\n(harmful)', xy=(1.0, 23), xytext=(0.75, 12),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Labels and formatting
ax.set_xlabel('Interpolation α  (0 = entity, 1 = date)', fontsize=12)
ax.set_ylabel('Cure Rate (%)', fontsize=12)
ax.set_title('V-Vector Interpolation: Cure Rate Tracks Geometric Mixture', 
             fontsize=13, fontweight='bold')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 70)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0\n(100% entity)', '0.25', '0.5', '0.75', '1.0\n(100% date)'])

ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Add spread annotation
spread = cure_rates[0] - cure_rates[-1]
ax.annotate('', xy=(1.05, cure_rates[-1]), xytext=(1.05, cure_rates[0]),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
ax.text(1.08, (cure_rates[0] + cure_rates[-1])/2, f'{spread:.0f}pp\nspread', 
        fontsize=9, va='center')

plt.tight_layout()

# Save
output_path = '../figures/interpolation_gradient.png'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../figures/interpolation_gradient.pdf', bbox_inches='tight', facecolor='white')
print(f"Saved to {output_path}")

# Also show the data
print("\nInterpolation Results (n=100 per point):")
print("-" * 50)
for i, alpha in enumerate(alphas):
    print(f"α={alpha:.2f}: {cure_rates[i]:.0f}% [{ci_lows[i]:.0f}-{ci_highs[i]:.0f}%]")
print(f"\nMonotonic: {data['is_monotonic']}")
print(f"Spread: {spread:.0f}pp")
