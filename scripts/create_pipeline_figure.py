#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))

# Define colors
blue = '#4472C4'
green = '#70AD47'
orange = '#ED7D31'
red = '#C5504B'
gray = '#A5A5A5'

# Pipeline steps
steps = [
    ("Natural Language\nPrompt", 1, 7, blue),
    ("Text-to-Constraint\nMapping", 3, 7, green),
    ("Constraint-Guided\nSampling", 5, 7, orange),
    ("ESM-2 LM\nRescoring", 7, 7, red),
    ("Safety\nScreening", 9, 7, gray),
    ("Final\nPeptides", 11, 7, blue)
]

# Draw boxes and text
for step, x, y, color in steps:
    box = FancyBboxPatch((x-0.7, y-0.8), 1.4, 1.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, 
                         edgecolor='black',
                         alpha=0.7)
    ax.add_patch(box)
    ax.text(x, y, step, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Draw arrows
arrow_props = dict(arrowstyle='->', lw=2, color='black')
for i in range(len(steps)-1):
    x1 = steps[i][1] + 0.7
    x2 = steps[i+1][1] - 0.7
    y = 7
    ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=arrow_props)

# Add details below each step
details = [
    '"cationic amphipathic\nhelix"',
    'charge: +3 to +8\nμH: >0.35\nGRAVY: -0.2 to 0.6',
    'Two-phase search:\n1. Reach charge\n2. Optimize μH/GRAVY',
    'ESM-2-t6-8M\navg log-likelihood',
    'Length, charge,\nhomopolymers,\ncysteines',
    'Top sequences\nwith properties'
]

for i, (detail, (_, x, _, _)) in enumerate(zip(details, steps)):
    ax.text(x, 4.5, detail, ha='center', va='top', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

# Add title and labels
ax.text(6, 9, 'Prompt2Peptide Pipeline', ha='center', va='center', 
        fontsize=16, fontweight='bold')

# Add example sequences
example_box = FancyBboxPatch((2, 1), 8, 2, 
                           boxstyle="round,pad=0.2", 
                           facecolor='lightblue', 
                           edgecolor='blue',
                           alpha=0.3)
ax.add_patch(example_box)

ax.text(6, 2.5, 'Example Generated Sequences', ha='center', va='center', 
        fontsize=12, fontweight='bold')

examples = [
    'KLALKLALKLALKLAL (μH=0.89, charge=+8, GRAVY=0.12)',
    'RLALRLALRLALRLAL (μH=0.91, charge=+8, GRAVY=0.15)',
    'KWFWKWFWKWFWKWFW (μH=0.82, charge=+8, GRAVY=0.98)'
]

for i, example in enumerate(examples):
    ax.text(6, 2.0 - i*0.3, example, ha='center', va='center', 
            fontsize=9, fontfamily='monospace')

# Set limits and remove axes
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()
plt.savefig('styles/pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

print("Pipeline figure created: styles/pipeline.png")
