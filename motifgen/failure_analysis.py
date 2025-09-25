# --- failure_analysis.py ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import gravy, net_charge, hydrophobic_moment
from .generate import generate, constraint_score

def analyze_constraint_failures(prompt, n_examples=6):
    """Analyze specific constraint failures and how fixes address them"""
    
    print(f"🔍 FAILURE MODE ANALYSIS: {prompt}")
    print("="*60)
    
    # Generate sequences with the old method (simulated)
    failures = []
    
    # Simulate different types of failures
    failure_types = [
        {
            'type': 'charge_too_low',
            'description': 'Charge below target range',
            'sequence': 'ACDEFGHIKLMN',  # Low charge
            'issue': 'Insufficient K/R residues',
            'fix': 'Add K/R residues to increase positive charge'
        },
        {
            'type': 'charge_too_high', 
            'description': 'Charge above target range',
            'sequence': 'KKKKKKKKKKKK',  # Too high charge
            'issue': 'Excessive K/R residues',
            'fix': 'Replace K/R with neutral residues (A/S/T/N/Q)'
        },
        {
            'type': 'muh_too_low',
            'description': 'Hydrophobic moment below threshold',
            'sequence': 'ACDEFGHIKLMN',  # Low μH
            'issue': 'Poor amphipathic pattern',
            'fix': 'Optimize hydrophobic/polar alternation'
        },
        {
            'type': 'gravy_out_of_range',
            'description': 'GRAVY outside target range',
            'sequence': 'FFFFFFFFFFFF',  # Too hydrophobic
            'issue': 'Excessive hydrophobicity',
            'fix': 'Add polar residues to balance hydrophobicity'
        },
        {
            'type': 'composition_imbalanced',
            'description': 'Poor K/R composition',
            'sequence': 'ACDEFGHIKLMN',  # Low K/R fraction
            'issue': 'K/R fraction < 0.25',
            'fix': 'Increase K/R fraction to 0.25-0.45'
        },
        {
            'type': 'his_overcontribution',
            'description': 'Histidine overcontribution to charge',
            'sequence': 'HHHHHHHHHHHH',  # All His
            'issue': 'His contributes full positive charge',
            'fix': 'Scale His contribution to 0.15 of full positive'
        }
    ]
    
    # Analyze each failure
    for i, failure in enumerate(failure_types[:n_examples]):
        seq = failure['sequence']
        
        # Calculate metrics
        metrics = {
            'sequence': seq,
            'length': len(seq),
            'charge': net_charge(seq),
            'muH': hydrophobic_moment(seq),
            'gravy': gravy(seq),
            'kr_fraction': (seq.count('K') + seq.count('R')) / len(seq),
            'de_fraction': (seq.count('D') + seq.count('E')) / len(seq),
            'h_count': seq.count('H')
        }
        
        # Determine target constraints based on prompt
        if "cationic" in prompt.lower():
            target = {
                'charge': (3, 8),
                'muH': (0.35, 1.0),
                'gravy': (-0.2, 0.6),
                'type': 'cationic'
            }
        elif "acidic" in prompt.lower():
            target = {
                'charge': (-3, 0),
                'muH': (0.1, 0.4),
                'gravy': (-1.0, 0.0),
                'type': 'acidic'
            }
        else:
            target = {
                'charge': (0, 3),
                'muH': (0.15, 1.0),
                'gravy': (-0.5, 0.5),
                'type': 'general'
            }
        
        # Check constraint satisfaction
        charge_ok = target['charge'][0] <= metrics['charge'] <= target['charge'][1]
        muh_ok = target['muH'][0] <= metrics['muH'] <= target['muH'][1]
        gravy_ok = target['gravy'][0] <= metrics['gravy'] <= target['gravy'][1]
        
        # Composition constraints for cationic
        if target['type'] == 'cationic':
            kr_ok = 0.25 <= metrics['kr_fraction'] <= 0.45
            de_ok = metrics['de_fraction'] <= 0.10
        else:
            kr_ok = True
            de_ok = True
        
        # Overall satisfaction
        total_constraints = 5 if target['type'] == 'cationic' else 3
        satisfied_constraints = sum([charge_ok, muh_ok, gravy_ok, kr_ok, de_ok])
        satisfaction = satisfied_constraints / total_constraints * 100
        
        failure['metrics'] = metrics
        failure['constraint_satisfaction'] = {
            'charge': charge_ok,
            'muH': muh_ok,
            'gravy': gravy_ok,
            'kr_fraction': kr_ok,
            'de_fraction': de_ok
        }
        failure['satisfaction_percentage'] = satisfaction
        
        failures.append(failure)
        
        print(f"\n{i+1}. {failure['description']}")
        print(f"   Sequence: {seq}")
        print(f"   Issue: {failure['issue']}")
        print(f"   Fix: {failure['fix']}")
        print(f"   Satisfaction: {satisfaction:.1f}%")
        print(f"   Charge: {metrics['charge']:.2f} ({'✓' if charge_ok else '✗'})")
        print(f"   μH: {metrics['muH']:.3f} ({'✓' if muh_ok else '✗'})")
        print(f"   GRAVY: {metrics['gravy']:.3f} ({'✓' if gravy_ok else '✗'})")
        if target['type'] == 'cationic':
            print(f"   K/R fraction: {metrics['kr_fraction']:.3f} ({'✓' if kr_ok else '✗'})")
            print(f"   D/E fraction: {metrics['de_fraction']:.3f} ({'✓' if de_ok else '✗'})")
    
    return failures

def create_failure_mode_figure(failures, save_path='failure_mode_analysis.png'):
    """Create failure mode analysis figure"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Failure Mode Analysis: Constraint Violations and Fixes', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i, failure in enumerate(failures):
        ax = axes[i]
        
        # Create a bar chart showing constraint satisfaction
        constraints = ['Charge', 'μH', 'GRAVY', 'K/R', 'D/E']
        satisfaction = [
            failure['constraint_satisfaction']['charge'],
            failure['constraint_satisfaction']['muH'],
            failure['constraint_satisfaction']['gravy'],
            failure['constraint_satisfaction']['kr_fraction'],
            failure['constraint_satisfaction']['de_fraction']
        ]
        
        # Convert boolean to 0/1 for plotting
        satisfaction_values = [1 if s else 0 for s in satisfaction]
        
        colors = ['green' if s else 'red' for s in satisfaction]
        bars = ax.bar(constraints, satisfaction_values, color=colors, alpha=0.7)
        
        ax.set_title(f"{i+1}. {failure['description']}\n{failure['sequence']}")
        ax.set_ylabel('Constraint Satisfied')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        
        # Add satisfaction percentage
        ax.text(0.5, 1.1, f"Satisfaction: {failure['satisfaction_percentage']:.1f}%", 
                ha='center', va='bottom', transform=ax.transAxes, fontweight='bold')
        
        # Add issue and fix text
        ax.text(0.02, 0.02, f"Issue: {failure['issue']}\nFix: {failure['fix']}", 
                ha='left', va='bottom', transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def comprehensive_failure_analysis():
    """Comprehensive failure mode analysis for all prompt types"""
    
    print("🔍 COMPREHENSIVE FAILURE MODE ANALYSIS")
    print("="*70)
    
    prompts = [
        "cationic amphipathic helix, length 12–18",
        "soluble acidic loop 10–14"
    ]
    
    all_failures = {}
    
    for prompt in prompts:
        print(f"\n{'='*70}")
        failures = analyze_constraint_failures(prompt, n_examples=6)
        all_failures[prompt] = failures
        
        # Create individual failure mode figure
        figure_path = f'failure_mode_{prompt.replace(" ", "_").replace(",", "").replace("–", "_")}.png'
        create_failure_mode_figure(failures, figure_path)
        print(f"📊 Failure mode figure saved: {figure_path}")
    
    return all_failures

def generate_failure_summary_table(failures_dict):
    """Generate summary table of failure modes"""
    
    summary_data = []
    
    for prompt, failures in failures_dict.items():
        for failure in failures:
            summary_data.append({
                'Prompt': prompt.split(',')[0][:20] + '...',
                'Failure Type': failure['description'],
                'Sequence': failure['sequence'],
                'Satisfaction %': f"{failure['satisfaction_percentage']:.1f}%",
                'Issue': failure['issue'],
                'Fix': failure['fix']
            })
    
    return pd.DataFrame(summary_data)
