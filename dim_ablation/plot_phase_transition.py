#!/usr/bin/env python3
"""
Phase transition plot for §3.1.2
Two crossing lines with 5-seed confidence bands
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('exp2_phase_transition.csv')

# Aggregate by step (across all entities and seeds)
grouped = df.groupby('step').agg({
    'p_ratio': ['mean', 'std'],
    'correct_ratio': ['mean', 'std']
}).reset_index()

grouped.columns = ['step', 'p_mean', 'p_std', 'correct_mean', 'correct_std']

# Fill NaN std with 0
grouped['p_std'] = grouped['p_std'].fillna(0)
grouped['correct_std'] = grouped['correct_std'].fillna(0)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

steps = grouped['step'].values

# Plot P-ratio (starts high, decreases)
ax.plot(steps, grouped['p_mean'], 'b-', linewidth=2, label='P-dominant')
ax.fill_between(steps, 
                grouped['p_mean'] - grouped['p_std'],
                grouped['p_mean'] + grouped['p_std'],
                alpha=0.2, color='blue')

# Plot correct-ratio (starts low, increases)
ax.plot(steps, grouped['correct_mean'], 'r-', linewidth=2, label='Correct')
ax.fill_between(steps,
                grouped['correct_mean'] - grouped['correct_std'],
                grouped['correct_mean'] + grouped['correct_std'],
                alpha=0.2, color='red')

# Find crossing point (approximate)
diff = grouped['p_mean'].values - grouped['correct_mean'].values
for i in range(len(diff) - 1):
    if diff[i] > 0 and diff[i+1] <= 0:
        # Linear interpolation for crossing point
        x1, x2 = steps[i], steps[i+1]
        y1_p, y2_p = grouped['p_mean'].iloc[i], grouped['p_mean'].iloc[i+1]
        y1_c, y2_c = grouped['correct_mean'].iloc[i], grouped['correct_mean'].iloc[i+1]
        # Solve: y1_p + t*(y2_p - y1_p) = y1_c + t*(y2_c - y1_c)
        t = (y1_p - y1_c) / ((y2_c - y1_c) - (y2_p - y1_p))
        cross_x = x1 + t * (x2 - x1)
        cross_y = y1_p + t * (y2_p - y1_p)
        ax.axvline(x=cross_x, color='gray', linestyle='--', alpha=0.5)
        ax.plot(cross_x, cross_y, 'ko', markersize=8)
        ax.annotate(f'Transition\n(step ≈ {int(cross_x)})', 
                    xy=(cross_x, cross_y), xytext=(cross_x - 300, cross_y + 0.25),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        break

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Ratio', fontsize=12)
ax.set_title('Phase Transition: P-dominant → Correct (5 seeds)', fontsize=13)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, max(steps) + 50)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_transition.png', dpi=150, bbox_inches='tight')
plt.savefig('phase_transition.pdf', bbox_inches='tight')
print('Saved: phase_transition.png, phase_transition.pdf')

# Also print statistics
print(f"\nCrossing point: step ≈ {int(cross_x)}")
print(f"Final P-dominant: {grouped['p_mean'].iloc[-1]:.3f} ± {grouped['p_std'].iloc[-1]:.3f}")
print(f"Final Correct: {grouped['correct_mean'].iloc[-1]:.3f} ± {grouped['correct_std'].iloc[-1]:.3f}")
