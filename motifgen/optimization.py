#!/usr/bin/env python3
"""
Principled Optimization Framework for Prompt2Peptide
Implements formal objective J with curriculum analysis and time-to-feasibility
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import seaborn as sns

@dataclass
class ConstraintWeights:
    """Weights for the optimization objective J"""
    charge: float = 1.0
    muh: float = 1.0
    gravy: float = 1.0
    composition: float = 1.0
    length: float = 0.5
    safety: float = 2.0  # Higher weight for safety

@dataclass
class OptimizationResult:
    """Result of optimization run"""
    sequence: str
    final_score: float
    constraint_scores: Dict[str, float]
    feasibility_time: float
    iterations: int
    trajectory: List[Dict[str, float]]
    is_feasible: bool

class OptimizationObjective:
    """Formal optimization objective J = w_c*C + w_h*H + w_g*G + w_comp*P"""
    
    def __init__(self, weights: ConstraintWeights = None):
        self.weights = weights or ConstraintWeights()
        
    def compute_score(self, 
                     sequence: str, 
                     target_constraints: Dict[str, Tuple[float, float]],
                     metrics_calculator) -> Tuple[float, Dict[str, float]]:
        """Compute objective J and individual constraint scores"""
        
        # Calculate biophysical metrics
        metrics = metrics_calculator.compute_all_metrics(sequence)
        
        # Individual constraint scores (0-1, higher is better)
        constraint_scores = {}
        
        # Charge constraint
        charge = metrics['charge']
        charge_min, charge_max = target_constraints['charge']
        if charge_min <= charge <= charge_max:
            constraint_scores['charge'] = 1.0
        else:
            distance = min(abs(charge - charge_min), abs(charge - charge_max))
            constraint_scores['charge'] = max(0.0, 1.0 - distance / max(abs(charge_min), abs(charge_max)))
        
        # μH constraint
        muh = metrics['muh']
        muh_min, muh_max = target_constraints['muh']
        if muh_min <= muh <= muh_max:
            constraint_scores['muh'] = 1.0
        else:
            distance = min(abs(muh - muh_min), abs(muh - muh_max))
            constraint_scores['muh'] = max(0.0, 1.0 - distance / max(abs(muh_min), abs(muh_max)))
        
        # GRAVY constraint
        gravy = metrics['gravy']
        gravy_min, gravy_max = target_constraints['gravy']
        if gravy_min <= gravy <= gravy_max:
            constraint_scores['gravy'] = 1.0
        else:
            distance = min(abs(gravy - gravy_min), abs(gravy - gravy_max))
            constraint_scores['gravy'] = max(0.0, 1.0 - distance / max(abs(gravy_min), abs(gravy_max)))
        
        # Composition constraint (K/R fraction)
        kr_fraction = metrics['kr_fraction']
        if 'composition' in target_constraints:
            comp_min, comp_max = target_constraints['composition']
            if comp_min <= kr_fraction <= comp_max:
                constraint_scores['composition'] = 1.0
            else:
                distance = min(abs(kr_fraction - comp_min), abs(kr_fraction - comp_max))
                constraint_scores['composition'] = max(0.0, 1.0 - distance / max(abs(comp_min), abs(comp_max)))
        else:
            constraint_scores['composition'] = 1.0
        
        # Length constraint
        length = len(sequence)
        if 'length' in target_constraints:
            len_min, len_max = target_constraints['length']
            if len_min <= length <= len_max:
                constraint_scores['length'] = 1.0
            else:
                distance = min(abs(length - len_min), abs(length - len_max))
                constraint_scores['length'] = max(0.0, 1.0 - distance / max(len_min, len_max))
        else:
            constraint_scores['length'] = 1.0
        
        # Safety constraint (placeholder - would integrate with safety module)
        constraint_scores['safety'] = 1.0  # Assume safe for now
        
        # Compute weighted objective J
        J = (self.weights.charge * constraint_scores['charge'] +
             self.weights.muh * constraint_scores['muh'] +
             self.weights.gravy * constraint_scores['gravy'] +
             self.weights.composition * constraint_scores['composition'] +
             self.weights.length * constraint_scores['length'] +
             self.weights.safety * constraint_scores['safety'])
        
        return J, constraint_scores

class CurriculumOptimizer:
    """Charge-first curriculum optimizer with two-phase strategy"""
    
    def __init__(self, 
                 objective: OptimizationObjective,
                 metrics_calculator,
                 temperature_schedule: str = "exponential"):
        self.objective = objective
        self.metrics_calculator = metrics_calculator
        self.temperature_schedule = temperature_schedule
        
    def optimize(self, 
                initial_sequence: str,
                target_constraints: Dict[str, Tuple[float, float]],
                max_iterations: int = 500,
                phase1_ratio: float = 0.6) -> OptimizationResult:
        """Run curriculum optimization with two-phase strategy"""
        
        current_sequence = initial_sequence
        trajectory = []
        start_time = time.time()
        feasibility_time = None
        
        # Phase 1: Charge-directed curriculum (reach feasible charge manifold)
        phase1_iterations = int(max_iterations * phase1_ratio)
        phase2_iterations = max_iterations - phase1_iterations
        
        for iteration in range(max_iterations):
            # Determine phase and temperature
            if iteration < phase1_iterations:
                phase = "charge"
                # Higher temperature for exploration in phase 1
                temperature = self._get_temperature(iteration, phase1_iterations, base_temp=1.0)
            else:
                phase = "optimization"
                # Lower temperature for exploitation in phase 2
                temp_iter = iteration - phase1_iterations
                temperature = self._get_temperature(temp_iter, phase2_iterations, base_temp=0.1)
            
            # Generate candidate
            candidate = self._generate_candidate(current_sequence, target_constraints, phase)
            
            # Evaluate objective
            current_score, current_constraints = self.objective.compute_score(
                current_sequence, target_constraints, self.metrics_calculator
            )
            candidate_score, candidate_constraints = self.objective.compute_score(
                candidate, target_constraints, self.metrics_calculator
            )
            
            # Check feasibility
            is_feasible = self._is_feasible(candidate_constraints)
            if is_feasible and feasibility_time is None:
                feasibility_time = time.time() - start_time
            
            # Accept/reject based on temperature
            if self._accept_candidate(candidate_score, current_score, temperature):
                current_sequence = candidate
                current_score = candidate_score
                current_constraints = candidate_constraints
            
            # Record trajectory
            trajectory.append({
                'iteration': iteration,
                'phase': phase,
                'score': current_score,
                'temperature': temperature,
                'is_feasible': is_feasible,
                'constraints': current_constraints.copy()
            })
        
        total_time = time.time() - start_time
        
        return OptimizationResult(
            sequence=current_sequence,
            final_score=current_score,
            constraint_scores=current_constraints,
            feasibility_time=feasibility_time or total_time,
            iterations=max_iterations,
            trajectory=trajectory,
            is_feasible=is_feasible
        )
    
    def _get_temperature(self, iteration: int, max_iterations: int, base_temp: float) -> float:
        """Get temperature based on schedule"""
        if self.temperature_schedule == "exponential":
            return base_temp * (0.995 ** iteration)
        elif self.temperature_schedule == "linear":
            return base_temp * (1.0 - iteration / max_iterations)
        else:
            return base_temp
    
    def _generate_candidate(self, 
                          sequence: str, 
                          target_constraints: Dict[str, Tuple[float, float]],
                          phase: str) -> str:
        """Generate candidate sequence based on phase"""
        # This would integrate with the existing mutation logic
        # For now, implement simplified version
        
        if phase == "charge":
            # Charge-directed mutations
            return self._charge_directed_mutation(sequence, target_constraints)
        else:
            # General optimization mutations
            return self._general_mutation(sequence)
    
    def _charge_directed_mutation(self, sequence: str, target_constraints: Dict[str, Tuple[float, float]]) -> str:
        """Charge-directed mutation for phase 1"""
        # Simplified implementation - would integrate with existing generate.py
        import random
        AA = "ACDEFGHIKLMNPQRSTVWY"
        
        # Get current charge
        metrics = self.metrics_calculator.compute_all_metrics(sequence)
        current_charge = metrics['charge']
        charge_min, charge_max = target_constraints['charge']
        
        # Mutate to move toward charge target
        seq_list = list(sequence)
        pos = random.randint(0, len(seq_list) - 1)
        
        if current_charge < charge_min:
            # Need more positive charge
            seq_list[pos] = random.choice('KR')
        elif current_charge > charge_max:
            # Need less positive charge
            seq_list[pos] = random.choice('ASTNQ')
        else:
            # Random mutation
            seq_list[pos] = random.choice(AA.replace(seq_list[pos], ""))
        
        return ''.join(seq_list)
    
    def _general_mutation(self, sequence: str) -> str:
        """General mutation for phase 2"""
        import random
        AA = "ACDEFGHIKLMNPQRSTVWY"
        
        seq_list = list(sequence)
        pos = random.randint(0, len(seq_list) - 1)
        seq_list[pos] = random.choice(AA.replace(seq_list[pos], ""))
        
        return ''.join(seq_list)
    
    def _accept_candidate(self, candidate_score: float, current_score: float, temperature: float) -> bool:
        """Accept candidate based on Metropolis criterion"""
        if candidate_score > current_score:
            return True
        
        import random
        delta = candidate_score - current_score
        probability = np.exp(delta / temperature)
        return random.random() < probability
    
    def _is_feasible(self, constraint_scores: Dict[str, float], threshold: float = 0.8) -> bool:
        """Check if solution is feasible"""
        return all(score >= threshold for score in constraint_scores.values())

class OptimizationAnalyzer:
    """Analyze optimization trajectories and performance"""
    
    def __init__(self):
        pass
    
    def plot_trajectories(self, results: List[OptimizationResult], save_path: str):
        """Plot optimization trajectories"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Score trajectories
        ax1 = axes[0, 0]
        for i, result in enumerate(results):
            scores = [step['score'] for step in result.trajectory]
            ax1.plot(scores, alpha=0.7, label=f'Run {i+1}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Score J')
        ax1.set_title('Optimization Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature schedules
        ax2 = axes[0, 1]
        for i, result in enumerate(results):
            temps = [step['temperature'] for step in result.trajectory]
            ax2.plot(temps, alpha=0.7, label=f'Run {i+1}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Schedules')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feasibility over time
        ax3 = axes[1, 0]
        for i, result in enumerate(results):
            feasible = [step['is_feasible'] for step in result.trajectory]
            ax3.plot(feasible, alpha=0.7, label=f'Run {i+1}')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Feasible')
        ax3.set_title('Feasibility Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Constraint satisfaction
        ax4 = axes[1, 1]
        constraint_names = ['charge', 'muh', 'gravy', 'composition']
        for constraint in constraint_names:
            scores = []
            for result in results:
                final_score = result.constraint_scores.get(constraint, 0)
                scores.append(final_score)
            ax4.bar(constraint, np.mean(scores), alpha=0.7, label=constraint)
        ax4.set_ylabel('Average Constraint Score')
        ax4.set_title('Final Constraint Satisfaction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feasibility_cdf(self, results: List[OptimizationResult], save_path: str):
        """Plot time-to-feasibility CDF"""
        feasibility_times = [r.feasibility_time for r in results if r.feasibility_time is not None]
        
        if not feasibility_times:
            print("No feasible solutions found")
            return
        
        plt.figure(figsize=(10, 6))
        
        # CDF plot
        sorted_times = np.sort(feasibility_times)
        y = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        
        plt.plot(sorted_times, y, 'b-', linewidth=2, label='Time-to-Feasibility CDF')
        plt.axvline(np.mean(feasibility_times), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(feasibility_times):.2f}s')
        plt.axvline(np.median(feasibility_times), color='g', linestyle='--', 
                   label=f'Median: {np.median(feasibility_times):.2f}s')
        
        plt.xlabel('Time to Feasibility (seconds)')
        plt.ylabel('Cumulative Probability')
        plt.title('Time-to-Feasibility Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_baselines(self, 
                         curriculum_results: List[OptimizationResult],
                         baseline_results: Dict[str, List[OptimizationResult]],
                         save_path: str):
        """Compare curriculum optimizer with baselines"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Feasibility rates
        ax1 = axes[0]
        methods = ['Curriculum'] + list(baseline_results.keys())
        feasibility_rates = []
        
        # Curriculum feasibility rate
        curriculum_feasible = sum(1 for r in curriculum_results if r.is_feasible)
        feasibility_rates.append(curriculum_feasible / len(curriculum_results))
        
        # Baseline feasibility rates
        for method, results in baseline_results.items():
            feasible = sum(1 for r in results if r.is_feasible)
            feasibility_rates.append(feasible / len(results))
        
        bars = ax1.bar(methods, feasibility_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        for bar, rate in zip(bars, feasibility_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Feasibility Rate')
        ax1.set_title('Feasibility Rate Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time to feasibility
        ax2 = axes[1]
        all_times = []
        all_labels = []
        
        # Curriculum times
        curriculum_times = [r.feasibility_time for r in curriculum_results if r.feasibility_time is not None]
        all_times.extend(curriculum_times)
        all_labels.extend(['Curriculum'] * len(curriculum_times))
        
        # Baseline times
        for method, results in baseline_results.items():
            times = [r.feasibility_time for r in results if r.feasibility_time is not None]
            all_times.extend(times)
            all_labels.extend([method] * len(times))
        
        # Box plot
        import pandas as pd
        df = pd.DataFrame({'Time': all_times, 'Method': all_labels})
        sns.boxplot(data=df, x='Method', y='Time', ax=ax2)
        ax2.set_ylabel('Time to Feasibility (s)')
        ax2.set_title('Time-to-Feasibility Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final objective scores
        ax3 = axes[2]
        all_scores = []
        all_labels = []
        
        # Curriculum scores
        curriculum_scores = [r.final_score for r in curriculum_results]
        all_scores.extend(curriculum_scores)
        all_labels.extend(['Curriculum'] * len(curriculum_scores))
        
        # Baseline scores
        for method, results in baseline_results.items():
            scores = [r.final_score for r in results]
            all_scores.extend(scores)
            all_labels.extend([method] * len(scores))
        
        # Box plot
        df_scores = pd.DataFrame({'Score': all_scores, 'Method': all_labels})
        sns.boxplot(data=df_scores, x='Method', y='Score', ax=ax3)
        ax3.set_ylabel('Final Objective Score J')
        ax3.set_title('Final Score Distribution')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_optimization_analysis():
    """Run comprehensive optimization analysis"""
    print("🔬 Running optimization analysis...")
    
    # This would integrate with the existing codebase
    # For now, create a demonstration
    
    # Create mock results for demonstration
    results = []
    for i in range(10):
        # Mock optimization result
        result = OptimizationResult(
            sequence="MOCKSEQUENCE",
            final_score=0.8 + np.random.normal(0, 0.1),
            constraint_scores={'charge': 0.9, 'muh': 0.8, 'gravy': 0.7, 'composition': 0.85},
            feasibility_time=10 + np.random.exponential(5),
            iterations=500,
            trajectory=[],  # Would contain full trajectory
            is_feasible=True
        )
        results.append(result)
    
    # Analyze results
    analyzer = OptimizationAnalyzer()
    analyzer.plot_trajectories(results, 'optimization_trajectories.png')
    analyzer.plot_feasibility_cdf(results, 'feasibility_cdf.png')
    
    print("✅ Optimization analysis completed!")

if __name__ == "__main__":
    run_optimization_analysis()
