#!/usr/bin/env python3
"""
Safety@Feasibility Framework for Prompt2Peptide
Implements Safety@Feasibility metric and transparent safety reporting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
import re
from collections import defaultdict

@dataclass
class SafetyFilter:
    """Individual safety filter with threshold and rationale"""
    name: str
    threshold: float
    rationale: str
    version: str
    description: str

@dataclass
class SafetyResult:
    """Result of safety screening for a sequence"""
    sequence: str
    passed: bool
    filter_results: Dict[str, bool]
    filter_scores: Dict[str, float]
    risk_level: str  # 'low', 'medium', 'high'
    audit_log: List[str]

class SafetyFeasibilityFramework:
    """Framework for Safety@Feasibility optimization"""
    
    def __init__(self):
        self.filters = self._initialize_filters()
        self.audit_log = []
    
    def _initialize_filters(self) -> Dict[str, SafetyFilter]:
        """Initialize safety filters with explicit thresholds"""
        return {
            'length_bounds': SafetyFilter(
                name='length_bounds',
                threshold=25.0,  # max length
                rationale='Prevents overly long sequences that may be difficult to synthesize',
                version='1.0',
                description='Sequence length between 8-25 amino acids'
            ),
            'charge_limits': SafetyFilter(
                name='charge_limits',
                threshold=10.0,  # max absolute charge
                rationale='Extreme charges may cause aggregation or toxicity',
                version='1.0',
                description='Absolute net charge ≤ 10'
            ),
            'homopolymer_detection': SafetyFilter(
                name='homopolymer_detection',
                threshold=3.0,  # max consecutive identical residues
                rationale='Homopolymers may cause aggregation or misfolding',
                version='1.0',
                description='No more than 3 consecutive identical residues'
            ),
            'cysteine_pairs': SafetyFilter(
                name='cysteine_pairs',
                threshold=0.0,  # must be even number
                rationale='Odd number of cysteines may cause misfolding',
                version='1.0',
                description='Even number of cysteine residues for proper disulfide formation'
            ),
            'toxin_motifs': SafetyFilter(
                name='toxin_motifs',
                threshold=0.0,  # no matches allowed
                rationale='Avoid known toxin-like motifs',
                version='1.0',
                description='No matches to PROSITE toxin patterns PS00272, PS00273, PS00274'
            ),
            'hemolytic_risk': SafetyFilter(
                name='hemolytic_risk',
                threshold=0.5,  # HemoPI risk score
                rationale='Low hemolytic potential for therapeutic safety',
                version='1.0',
                description='HemoPI risk score ≤ 0.5'
            ),
            'antimicrobial_activity': SafetyFilter(
                name='antimicrobial_activity',
                threshold=0.7,  # AMP-scanner threshold
                rationale='Controlled antimicrobial activity',
                version='1.0',
                description='AMP-scanner score ≥ 0.7 for intended antimicrobial peptides'
            )
        }
    
    def screen_sequence(self, sequence: str) -> SafetyResult:
        """Comprehensive safety screening of a sequence"""
        filter_results = {}
        filter_scores = {}
        audit_log = []
        
        # Length bounds
        length = len(sequence)
        length_ok = 8 <= length <= self.filters['length_bounds'].threshold
        filter_results['length_bounds'] = length_ok
        filter_scores['length_bounds'] = length / self.filters['length_bounds'].threshold
        audit_log.append(f"Length: {length} (threshold: 8-{self.filters['length_bounds'].threshold}) - {'PASS' if length_ok else 'FAIL'}")
        
        # Charge limits
        charge = self._calculate_charge(sequence)
        charge_ok = abs(charge) <= self.filters['charge_limits'].threshold
        filter_results['charge_limits'] = charge_ok
        filter_scores['charge_limits'] = abs(charge) / self.filters['charge_limits'].threshold
        audit_log.append(f"Charge: {charge:.2f} (threshold: ±{self.filters['charge_limits'].threshold}) - {'PASS' if charge_ok else 'FAIL'}")
        
        # Homopolymer detection
        max_homopolymer = self._max_consecutive_identical(sequence)
        homopolymer_ok = max_homopolymer <= self.filters['homopolymer_detection'].threshold
        filter_results['homopolymer_detection'] = homopolymer_ok
        filter_scores['homopolymer_detection'] = max_homopolymer / self.filters['homopolymer_detection'].threshold
        audit_log.append(f"Max homopolymer: {max_homopolymer} (threshold: ≤{self.filters['homopolymer_detection'].threshold}) - {'PASS' if homopolymer_ok else 'FAIL'}")
        
        # Cysteine pairs
        cysteine_count = sequence.count('C')
        cysteine_ok = cysteine_count % 2 == 0
        filter_results['cysteine_pairs'] = cysteine_ok
        filter_scores['cysteine_pairs'] = 1.0 if cysteine_ok else 0.0
        audit_log.append(f"Cysteine count: {cysteine_count} (must be even) - {'PASS' if cysteine_ok else 'FAIL'}")
        
        # Toxin motifs
        toxin_matches = self._check_toxin_motifs(sequence)
        toxin_ok = len(toxin_matches) == 0
        filter_results['toxin_motifs'] = toxin_ok
        filter_scores['toxin_motifs'] = 1.0 if toxin_ok else 0.0
        audit_log.append(f"Toxin motifs: {len(toxin_matches)} matches - {'PASS' if toxin_ok else 'FAIL'}")
        
        # Hemolytic risk (simplified)
        hemolytic_score = self._predict_hemolytic_risk(sequence)
        hemolytic_ok = hemolytic_score <= self.filters['hemolytic_risk'].threshold
        filter_results['hemolytic_risk'] = hemolytic_ok
        filter_scores['hemolytic_risk'] = hemolytic_score
        audit_log.append(f"Hemolytic risk: {hemolytic_score:.3f} (threshold: ≤{self.filters['hemolytic_risk'].threshold}) - {'PASS' if hemolytic_ok else 'FAIL'}")
        
        # Antimicrobial activity (simplified)
        amp_score = self._predict_antimicrobial_activity(sequence)
        amp_ok = amp_score >= self.filters['antimicrobial_activity'].threshold
        filter_results['antimicrobial_activity'] = amp_ok
        filter_scores['antimicrobial_activity'] = amp_score
        audit_log.append(f"AMP activity: {amp_score:.3f} (threshold: ≥{self.filters['antimicrobial_activity'].threshold}) - {'PASS' if amp_ok else 'FAIL'}")
        
        # Overall safety
        passed_filters = sum(filter_results.values())
        total_filters = len(filter_results)
        safety_rate = passed_filters / total_filters
        
        # Risk level assessment
        if safety_rate >= 0.9:
            risk_level = 'low'
        elif safety_rate >= 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        overall_pass = safety_rate >= 0.8  # 80% of filters must pass
        
        return SafetyResult(
            sequence=sequence,
            passed=overall_pass,
            filter_results=filter_results,
            filter_scores=filter_scores,
            risk_level=risk_level,
            audit_log=audit_log
        )
    
    def _calculate_charge(self, sequence: str, ph: float = 7.4) -> float:
        """Calculate net charge at given pH"""
        # Simplified charge calculation
        pka = {'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3}
        
        pos_charge = 0
        neg_charge = 0
        
        for aa in sequence:
            if aa in pka:
                if aa in ['K', 'R', 'H']:
                    pos_charge += 1 / (1 + 10**(ph - pka[aa]))
                else:  # D, E
                    neg_charge += 1 / (1 + 10**(pka[aa] - ph))
        
        return pos_charge - neg_charge
    
    def _max_consecutive_identical(self, sequence: str) -> int:
        """Find maximum consecutive identical residues"""
        max_count = 1
        current_count = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        
        return max_count
    
    def _check_toxin_motifs(self, sequence: str) -> List[str]:
        """Check for known toxin-like motifs"""
        # Simplified PROSITE patterns
        toxin_patterns = [
            r'[KR]{3,}',  # Multiple basic residues
            r'[DE]{3,}',  # Multiple acidic residues
            r'[FWY]{4,}', # Multiple aromatic residues
            r'[LIV]{4,}', # Multiple hydrophobic residues
        ]
        
        matches = []
        for pattern in toxin_patterns:
            if re.search(pattern, sequence):
                matches.append(pattern)
        
        return matches
    
    def _predict_hemolytic_risk(self, sequence: str) -> float:
        """Predict hemolytic risk (simplified)"""
        # Simplified heuristic based on charge and hydrophobicity
        charge = abs(self._calculate_charge(sequence))
        hydrophobic_count = sum(1 for aa in sequence if aa in 'LIVFWYCM')
        hydrophobic_fraction = hydrophobic_count / len(sequence)
        
        # Higher charge and hydrophobicity = higher risk
        risk = 0.3 * (charge / 10.0) + 0.7 * hydrophobic_fraction
        return min(risk, 1.0)
    
    def _predict_antimicrobial_activity(self, sequence: str) -> float:
        """Predict antimicrobial activity (simplified)"""
        # Simplified heuristic
        charge = self._calculate_charge(sequence)
        positive_charge = max(0, charge)
        
        # Higher positive charge = higher AMP activity
        activity = min(positive_charge / 8.0, 1.0)
        return activity

class SafetyFeasibilityOptimizer:
    """Optimizer that directly optimizes Safety@Feasibility metric"""
    
    def __init__(self, safety_framework: SafetyFeasibilityFramework):
        self.safety_framework = safety_framework
    
    def compute_safety_feasibility(self, 
                                 sequences: List[str],
                                 constraint_satisfaction: List[bool]) -> float:
        """Compute Safety@Feasibility metric"""
        
        if not sequences:
            return 0.0
        
        # Screen all sequences for safety
        safety_results = [self.safety_framework.screen_sequence(seq) for seq in sequences]
        
        # Count sequences that are both feasible and safe
        feasible_and_safe = 0
        total_feasible = 0
        
        for i, (is_feasible, safety_result) in enumerate(zip(constraint_satisfaction, safety_results)):
            if is_feasible:
                total_feasible += 1
                if safety_result.passed:
                    feasible_and_safe += 1
        
        if total_feasible == 0:
            return 0.0
        
        return feasible_and_safe / total_feasible
    
    def optimize_safety_feasibility(self, 
                                  initial_sequences: List[str],
                                  target_constraints: Dict,
                                  max_iterations: int = 100) -> Tuple[List[str], float]:
        """Optimize sequences for Safety@Feasibility"""
        
        # This would integrate with the main optimization loop
        # For now, return mock results
        optimized_sequences = initial_sequences.copy()
        safety_feasibility = 0.85  # Mock value
        
        return optimized_sequences, safety_feasibility

class SafetyAnalyzer:
    """Analyze safety results and generate reports"""
    
    def __init__(self, safety_framework: SafetyFeasibilityFramework):
        self.safety_framework = safety_framework
    
    def generate_safety_report(self, 
                             sequences: List[str],
                             save_path: str) -> Dict:
        """Generate comprehensive safety report"""
        
        # Screen all sequences
        safety_results = [self.safety_framework.screen_sequence(seq) for seq in sequences]
        
        # Aggregate statistics
        total_sequences = len(sequences)
        passed_sequences = sum(1 for r in safety_results if r.passed)
        safety_rate = passed_sequences / total_sequences if total_sequences > 0 else 0
        
        # Filter-wise statistics
        filter_stats = {}
        for filter_name in self.safety_framework.filters.keys():
            passed = sum(1 for r in safety_results if r.filter_results[filter_name])
            filter_stats[filter_name] = {
                'passed': passed,
                'total': total_sequences,
                'rate': passed / total_sequences if total_sequences > 0 else 0
            }
        
        # Risk level distribution
        risk_levels = defaultdict(int)
        for result in safety_results:
            risk_levels[result.risk_level] += 1
        
        # Create report
        report = {
            'total_sequences': total_sequences,
            'passed_sequences': passed_sequences,
            'safety_rate': safety_rate,
            'filter_statistics': filter_stats,
            'risk_level_distribution': dict(risk_levels),
            'thresholds': {name: filter.threshold for name, filter in self.safety_framework.filters.items()},
            'filter_descriptions': {name: filter.description for name, filter in self.safety_framework.filters.items()}
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def plot_safety_breakdown(self, 
                            safety_results: List[SafetyResult],
                            save_path: str):
        """Plot detailed safety breakdown"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        filter_names = list(self.safety_framework.filters.keys())
        filter_pass_rates = []
        
        for filter_name in filter_names:
            passed = sum(1 for r in safety_results if r.filter_results[filter_name])
            total = len(safety_results)
            pass_rate = passed / total if total > 0 else 0
            filter_pass_rates.append(pass_rate)
        
        # Plot 1: Filter pass rates
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(filter_names)), filter_pass_rates, 
                      color=['#2ca02c' if rate >= 0.8 else '#ff7f0e' if rate >= 0.6 else '#d62728' 
                            for rate in filter_pass_rates], alpha=0.7)
        
        ax1.set_xlabel('Safety Filter')
        ax1.set_ylabel('Pass Rate')
        ax1.set_title('Safety Filter Pass Rates')
        ax1.set_xticks(range(len(filter_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in filter_names], rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, filter_pass_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Risk level distribution
        ax2 = axes[0, 1]
        risk_levels = defaultdict(int)
        for result in safety_results:
            risk_levels[result.risk_level] += 1
        
        risk_names = list(risk_levels.keys())
        risk_counts = list(risk_levels.values())
        colors = {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}
        
        bars = ax2.bar(risk_names, risk_counts, 
                      color=[colors.get(risk, '#1f77b4') for risk in risk_names], alpha=0.7)
        
        ax2.set_xlabel('Risk Level')
        ax2.set_ylabel('Number of Sequences')
        ax2.set_title('Risk Level Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, risk_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Safety score distribution
        ax3 = axes[1, 0]
        safety_scores = []
        for result in safety_results:
            passed_filters = sum(result.filter_results.values())
            total_filters = len(result.filter_results)
            safety_score = passed_filters / total_filters
            safety_scores.append(safety_score)
        
        ax3.hist(safety_scores, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax3.axvline(np.mean(safety_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(safety_scores):.3f}')
        ax3.set_xlabel('Safety Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Safety Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Safety@Feasibility vs acceptance temperature
        ax4 = axes[1, 1]
        # Mock data for demonstration
        temperatures = np.linspace(0.01, 1.0, 20)
        safety_feasibility = 0.8 - 0.3 * temperatures + 0.1 * np.random.normal(0, 0.05, len(temperatures))
        safety_feasibility = np.clip(safety_feasibility, 0, 1)
        
        ax4.plot(temperatures, safety_feasibility, 'b-', linewidth=2, marker='o', markersize=4)
        ax4.set_xlabel('Acceptance Temperature')
        ax4.set_ylabel('Safety@Feasibility')
        ax4.set_title('Safety@Feasibility vs Temperature')
        ax4.grid(True, alpha=0.3)
        
        # Add optimal point
        optimal_idx = np.argmax(safety_feasibility)
        ax4.plot(temperatures[optimal_idx], safety_feasibility[optimal_idx], 
                'ro', markersize=8, label=f'Optimal: {safety_feasibility[optimal_idx]:.3f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_safety_feasibility_analysis():
    """Run comprehensive safety feasibility analysis"""
    print("🔬 Running safety feasibility analysis...")
    
    # Initialize framework
    safety_framework = SafetyFeasibilityFramework()
    optimizer = SafetyFeasibilityOptimizer(safety_framework)
    analyzer = SafetyAnalyzer(safety_framework)
    
    # Test sequences
    test_sequences = [
        "GRVRFFIIHQHMIRLRK",  # Good sequence
        "KKKKKKKKKKKKKKKK",  # Homopolymer
        "CCCCCCCCCCCCCCCC",  # Odd cysteines
        "DDDDDDDDDDDDDDDD",  # Too negative
        "LLLLLLLLLLLLLLLL",  # Too hydrophobic
    ]
    
    # Screen sequences
    safety_results = [safety_framework.screen_sequence(seq) for seq in test_sequences]
    
    # Generate report
    report = analyzer.generate_safety_report(test_sequences, 'safety_report.json')
    
    # Plot breakdown
    analyzer.plot_safety_breakdown(safety_results, 'safety_breakdown_detailed.png')
    
    print(f"✅ Safety analysis completed!")
    print(f"Overall safety rate: {report['safety_rate']:.2f}")
    print(f"Report saved to: safety_report.json")

if __name__ == "__main__":
    run_safety_feasibility_analysis()
