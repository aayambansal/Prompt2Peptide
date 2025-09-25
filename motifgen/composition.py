#!/usr/bin/env python3
"""
Multi-Prompt Composition for Prompt2Peptide
Enables composition of multiple prompts with Pareto optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from itertools import combinations, product
import json

@dataclass
class PromptComposition:
    """Represents a composition of multiple prompts"""
    prompts: List[str]
    weights: List[float]
    operation: str  # 'and', 'or', 'weighted_sum'
    
    def __post_init__(self):
        if len(self.prompts) != len(self.weights):
            raise ValueError("Number of prompts must match number of weights")
        if self.operation == 'weighted_sum':
            # Normalize weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]

@dataclass
class ConstraintConflict:
    """Represents a conflict between constraints"""
    constraint_name: str
    prompt1_value: float
    prompt2_value: float
    conflict_type: str  # 'incompatible_ranges', 'opposite_directions'

class PromptComposer:
    """Composes multiple prompts into unified constraints"""
    
    def __init__(self, prompt_encoder=None):
        self.prompt_encoder = prompt_encoder
        self.conflict_resolution_strategies = {
            'incompatible_ranges': self._resolve_incompatible_ranges,
            'opposite_directions': self._resolve_opposite_directions
        }
    
    def compose_prompts(self, composition: PromptComposition) -> Dict[str, Tuple[float, float]]:
        """Compose multiple prompts into unified constraints"""
        
        # Get constraints for each prompt
        prompt_constraints = []
        for prompt in composition.prompts:
            if self.prompt_encoder:
                # Use learned encoder
                constraints = self.prompt_encoder.predict_constraints([prompt])[0]
                # Convert to range format
                constraint_ranges = {
                    'charge': (constraints['charge_min'], constraints['charge_max']),
                    'muh': (constraints['muh_min'], constraints['muh_max']),
                    'gravy': (constraints['gravy_min'], constraints['gravy_max']),
                    'length': (constraints['length_min'], constraints['length_max'])
                }
            else:
                # Use rule-based mapping
                constraint_ranges = self._rule_based_constraints(prompt)
            
            prompt_constraints.append(constraint_ranges)
        
        # Detect conflicts
        conflicts = self._detect_conflicts(prompt_constraints)
        
        # Resolve conflicts
        resolved_constraints = self._resolve_conflicts(prompt_constraints, conflicts, composition)
        
        # Compose based on operation
        if composition.operation == 'and':
            return self._intersection_composition(prompt_constraints, composition.weights)
        elif composition.operation == 'or':
            return self._union_composition(prompt_constraints, composition.weights)
        elif composition.operation == 'weighted_sum':
            return self._weighted_composition(prompt_constraints, composition.weights)
        else:
            raise ValueError(f"Unknown operation: {composition.operation}")
    
    def _rule_based_constraints(self, prompt: str) -> Dict[str, Tuple[float, float]]:
        """Rule-based constraint mapping (fallback)"""
        prompt_lower = prompt.lower()
        
        if "cationic" in prompt_lower and "amphipathic" in prompt_lower:
            return {
                'charge': (3.0, 8.0),
                'muh': (0.35, 1.0),
                'gravy': (-0.2, 0.6),
                'length': (12, 18)
            }
        elif "soluble" in prompt_lower and "acidic" in prompt_lower:
            return {
                'charge': (-3.0, 0.0),
                'muh': (0.1, 0.4),
                'gravy': (-1.0, 0.0),
                'length': (10, 14)
            }
        elif "hydrophobic" in prompt_lower and "sheet" in prompt_lower:
            return {
                'charge': (-1.0, 2.0),
                'muh': (0.1, 0.3),
                'gravy': (0.5, 1.5),
                'length': (10, 14)
            }
        else:
            # Default constraints
            return {
                'charge': (0.0, 3.0),
                'muh': (0.15, 1.0),
                'gravy': (-0.5, 0.5),
                'length': (8, 20)
            }
    
    def _detect_conflicts(self, prompt_constraints: List[Dict[str, Tuple[float, float]]]) -> List[ConstraintConflict]:
        """Detect conflicts between prompt constraints"""
        conflicts = []
        
        for i, j in combinations(range(len(prompt_constraints)), 2):
            constraints1 = prompt_constraints[i]
            constraints2 = prompt_constraints[j]
            
            for constraint_name in constraints1.keys():
                range1 = constraints1[constraint_name]
                range2 = constraints2[constraint_name]
                
                # Check for incompatible ranges (no overlap)
                if range1[1] < range2[0] or range2[1] < range1[0]:
                    conflicts.append(ConstraintConflict(
                        constraint_name=constraint_name,
                        prompt1_value=(range1[0] + range1[1]) / 2,
                        prompt2_value=(range2[0] + range2[1]) / 2,
                        conflict_type='incompatible_ranges'
                    ))
                
                # Check for opposite directions (for optimization)
                elif (range1[0] > range2[1] and range1[1] > range2[0]) or \
                     (range2[0] > range1[1] and range2[1] > range1[0]):
                    conflicts.append(ConstraintConflict(
                        constraint_name=constraint_name,
                        prompt1_value=(range1[0] + range1[1]) / 2,
                        prompt2_value=(range2[0] + range2[1]) / 2,
                        conflict_type='opposite_directions'
                    ))
        
        return conflicts
    
    def _resolve_conflicts(self, 
                          prompt_constraints: List[Dict[str, Tuple[float, float]]],
                          conflicts: List[ConstraintConflict],
                          composition: PromptComposition) -> List[Dict[str, Tuple[float, float]]]:
        """Resolve conflicts between prompt constraints"""
        
        resolved_constraints = [constraints.copy() for constraints in prompt_constraints]
        
        for conflict in conflicts:
            strategy = self.conflict_resolution_strategies.get(conflict.conflict_type)
            if strategy:
                resolved_constraints = strategy(resolved_constraints, conflict, composition)
        
        return resolved_constraints
    
    def _resolve_incompatible_ranges(self, 
                                   constraints: List[Dict[str, Tuple[float, float]]],
                                   conflict: ConstraintConflict,
                                   composition: PromptComposition) -> List[Dict[str, Tuple[float, float]]]:
        """Resolve incompatible range conflicts by expanding ranges"""
        
        # Find the constraint ranges that conflict
        constraint_name = conflict.constraint_name
        
        # Get all ranges for this constraint
        ranges = [c[constraint_name] for c in constraints]
        
        # Find the union of all ranges
        min_val = min(r[0] for r in ranges)
        max_val = max(r[1] for r in ranges)
        
        # Expand all ranges to include the union
        for constraints_dict in constraints:
            constraints_dict[constraint_name] = (min_val, max_val)
        
        return constraints
    
    def _resolve_opposite_directions(self, 
                                   constraints: List[Dict[str, Tuple[float, float]]],
                                   conflict: ConstraintConflict,
                                   composition: PromptComposition) -> List[Dict[str, Tuple[float, float]]]:
        """Resolve opposite direction conflicts by creating trade-off ranges"""
        
        # For opposite directions, create a range that spans both
        # This will require multi-objective optimization
        constraint_name = conflict.constraint_name
        
        ranges = [c[constraint_name] for c in constraints]
        min_val = min(r[0] for r in ranges)
        max_val = max(r[1] for r in ranges)
        
        # Create a wider range that encompasses both
        range_width = max_val - min_val
        expanded_min = min_val - 0.1 * range_width
        expanded_max = max_val + 0.1 * range_width
        
        for constraints_dict in constraints:
            constraints_dict[constraint_name] = (expanded_min, expanded_max)
        
        return constraints
    
    def _intersection_composition(self, 
                                prompt_constraints: List[Dict[str, Tuple[float, float]]],
                                weights: List[float]) -> Dict[str, Tuple[float, float]]:
        """Compose constraints using intersection (AND operation)"""
        
        composed_constraints = {}
        
        for constraint_name in prompt_constraints[0].keys():
            # Find intersection of all ranges
            min_vals = [c[constraint_name][0] for c in prompt_constraints]
            max_vals = [c[constraint_name][1] for c in prompt_constraints]
            
            # Intersection is the overlap of all ranges
            intersection_min = max(min_vals)
            intersection_max = min(max_vals)
            
            # If no intersection, use weighted average
            if intersection_min > intersection_max:
                weighted_min = sum(w * c[constraint_name][0] for w, c in zip(weights, prompt_constraints))
                weighted_max = sum(w * c[constraint_name][1] for w, c in zip(weights, prompt_constraints))
                composed_constraints[constraint_name] = (weighted_min, weighted_max)
            else:
                composed_constraints[constraint_name] = (intersection_min, intersection_max)
        
        return composed_constraints
    
    def _union_composition(self, 
                         prompt_constraints: List[Dict[str, Tuple[float, float]]],
                         weights: List[float]) -> Dict[str, Tuple[float, float]]:
        """Compose constraints using union (OR operation)"""
        
        composed_constraints = {}
        
        for constraint_name in prompt_constraints[0].keys():
            # Find union of all ranges
            min_vals = [c[constraint_name][0] for c in prompt_constraints]
            max_vals = [c[constraint_name][1] for c in prompt_constraints]
            
            union_min = min(min_vals)
            union_max = max(max_vals)
            
            composed_constraints[constraint_name] = (union_min, union_max)
        
        return composed_constraints
    
    def _weighted_composition(self, 
                            prompt_constraints: List[Dict[str, Tuple[float, float]]],
                            weights: List[float]) -> Dict[str, Tuple[float, float]]:
        """Compose constraints using weighted average"""
        
        composed_constraints = {}
        
        for constraint_name in prompt_constraints[0].keys():
            weighted_min = sum(w * c[constraint_name][0] for w, c in zip(weights, prompt_constraints))
            weighted_max = sum(w * c[constraint_name][1] for w, c in zip(weights, prompt_constraints))
            
            composed_constraints[constraint_name] = (weighted_min, weighted_max)
        
        return composed_constraints

class ParetoOptimizer:
    """Multi-objective optimization for prompt composition"""
    
    def __init__(self, composer: PromptComposer):
        self.composer = composer
    
    def generate_pareto_front(self, 
                            composition: PromptComposition,
                            n_points: int = 100) -> List[Dict[str, float]]:
        """Generate Pareto front for multi-objective optimization"""
        
        # Get composed constraints
        composed_constraints = self.composer.compose_prompts(composition)
        
        # Generate trade-off points
        pareto_points = []
        
        for i in range(n_points):
            # Create trade-off weights
            alpha = i / (n_points - 1)
            
            # For demonstration, create trade-off between two objectives
            # In practice, this would be more sophisticated
            trade_off_constraints = {}
            
            for constraint_name, (min_val, max_val) in composed_constraints.items():
                # Create trade-off within the range
                trade_off_min = min_val + alpha * (max_val - min_val) * 0.3
                trade_off_max = max_val - alpha * (max_val - min_val) * 0.3
                
                trade_off_constraints[constraint_name] = (trade_off_min, trade_off_max)
            
            # Evaluate this trade-off point
            pareto_point = {
                'alpha': alpha,
                'constraints': trade_off_constraints,
                'charge_center': (trade_off_constraints['charge'][0] + trade_off_constraints['charge'][1]) / 2,
                'muh_center': (trade_off_constraints['muh'][0] + trade_off_constraints['muh'][1]) / 2,
                'gravy_center': (trade_off_constraints['gravy'][0] + trade_off_constraints['gravy'][1]) / 2
            }
            
            pareto_points.append(pareto_point)
        
        return pareto_points
    
    def plot_pareto_front(self, 
                         pareto_points: List[Dict[str, float]],
                         save_path: str):
        """Plot Pareto front for trade-off analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        alphas = [p['alpha'] for p in pareto_points]
        charge_centers = [p['charge_center'] for p in pareto_points]
        muh_centers = [p['muh_center'] for p in pareto_points]
        gravy_centers = [p['gravy_center'] for p in pareto_points]
        
        # Plot 1: Charge vs μH trade-off
        ax1 = axes[0, 0]
        scatter = ax1.scatter(charge_centers, muh_centers, c=alphas, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Charge (center)')
        ax1.set_ylabel('μH (center)')
        ax1.set_title('Charge vs μH Trade-off')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Trade-off Parameter α')
        
        # Plot 2: Charge vs GRAVY trade-off
        ax2 = axes[0, 1]
        scatter = ax2.scatter(charge_centers, gravy_centers, c=alphas, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Charge (center)')
        ax2.set_ylabel('GRAVY (center)')
        ax2.set_title('Charge vs GRAVY Trade-off')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Trade-off Parameter α')
        
        # Plot 3: μH vs GRAVY trade-off
        ax3 = axes[1, 0]
        scatter = ax3.scatter(muh_centers, gravy_centers, c=alphas, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('μH (center)')
        ax3.set_ylabel('GRAVY (center)')
        ax3.set_title('μH vs GRAVY Trade-off')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Trade-off Parameter α')
        
        # Plot 4: 3D trade-off surface
        ax4 = axes[1, 1]
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        scatter = ax4.scatter(charge_centers, muh_centers, gravy_centers, c=alphas, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Charge')
        ax4.set_ylabel('μH')
        ax4.set_zlabel('GRAVY')
        ax4.set_title('3D Trade-off Surface')
        plt.colorbar(scatter, ax=ax4, label='Trade-off Parameter α')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class CompositionAnalyzer:
    """Analyze prompt composition results"""
    
    def __init__(self):
        pass
    
    def analyze_composition_coverage(self, 
                                   compositions: List[PromptComposition],
                                   results: List[Dict],
                                   save_path: str):
        """Analyze coverage of composed prompts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        composition_names = [f"{'+'.join(c.prompts[:2])}" for c in compositions]
        feasibility_rates = [r.get('feasibility_rate', 0) for r in results]
        constraint_satisfaction = [r.get('constraint_satisfaction', 0) for r in results]
        novelty_rates = [r.get('novelty_rate', 0) for r in results]
        safety_rates = [r.get('safety_rate', 0) for r in results]
        
        # Plot 1: Feasibility rates
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(composition_names)), feasibility_rates, 
                      color='#1f77b4', alpha=0.7)
        ax1.set_xlabel('Composition')
        ax1.set_ylabel('Feasibility Rate')
        ax1.set_title('Feasibility Rates by Composition')
        ax1.set_xticks(range(len(composition_names)))
        ax1.set_xticklabels(composition_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, feasibility_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Constraint satisfaction
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(composition_names)), constraint_satisfaction, 
                      color='#ff7f0e', alpha=0.7)
        ax2.set_xlabel('Composition')
        ax2.set_ylabel('Constraint Satisfaction')
        ax2.set_title('Constraint Satisfaction by Composition')
        ax2.set_xticks(range(len(composition_names)))
        ax2.set_xticklabels(composition_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, constraint_satisfaction):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Multi-metric comparison
        ax3 = axes[1, 0]
        x = np.arange(len(composition_names))
        width = 0.2
        
        ax3.bar(x - 1.5*width, feasibility_rates, width, label='Feasibility', alpha=0.7)
        ax3.bar(x - 0.5*width, constraint_satisfaction, width, label='Constraint Sat.', alpha=0.7)
        ax3.bar(x + 0.5*width, novelty_rates, width, label='Novelty', alpha=0.7)
        ax3.bar(x + 1.5*width, safety_rates, width, label='Safety', alpha=0.7)
        
        ax3.set_xlabel('Composition')
        ax3.set_ylabel('Rate')
        ax3.set_title('Multi-Metric Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(composition_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Coverage vs complexity
        ax4 = axes[1, 1]
        complexity = [len(c.prompts) for c in compositions]
        coverage = [f * c for f, c in zip(feasibility_rates, constraint_satisfaction)]
        
        scatter = ax4.scatter(complexity, coverage, s=100, alpha=0.7, c=range(len(compositions)), cmap='viridis')
        ax4.set_xlabel('Composition Complexity (number of prompts)')
        ax4.set_ylabel('Coverage (feasibility × constraint satisfaction)')
        ax4.set_title('Coverage vs Complexity')
        ax4.grid(True, alpha=0.3)
        
        # Add labels
        for i, name in enumerate(composition_names):
            ax4.annotate(name, (complexity[i], coverage[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_composition_analysis():
    """Run comprehensive composition analysis"""
    print("🔬 Running composition analysis...")
    
    # Create composer
    composer = PromptComposer()
    pareto_optimizer = ParetoOptimizer(composer)
    analyzer = CompositionAnalyzer()
    
    # Define test compositions
    compositions = [
        PromptComposition(
            prompts=["cationic amphipathic helix", "protease resistant"],
            weights=[0.7, 0.3],
            operation="weighted_sum"
        ),
        PromptComposition(
            prompts=["soluble acidic loop", "membrane permeable"],
            weights=[0.6, 0.4],
            operation="weighted_sum"
        ),
        PromptComposition(
            prompts=["hydrophobic beta sheet", "thermostable"],
            weights=[0.8, 0.2],
            operation="weighted_sum"
        )
    ]
    
    # Generate Pareto fronts
    for i, composition in enumerate(compositions):
        pareto_points = pareto_optimizer.generate_pareto_front(composition, n_points=50)
        pareto_optimizer.plot_pareto_front(pareto_points, f'pareto_front_{i}.png')
    
    # Mock results for analysis
    results = [
        {'feasibility_rate': 0.75, 'constraint_satisfaction': 0.82, 'novelty_rate': 1.0, 'safety_rate': 0.85},
        {'feasibility_rate': 0.68, 'constraint_satisfaction': 0.78, 'novelty_rate': 1.0, 'safety_rate': 0.92},
        {'feasibility_rate': 0.71, 'constraint_satisfaction': 0.85, 'novelty_rate': 1.0, 'safety_rate': 0.88}
    ]
    
    # Analyze composition coverage
    analyzer.analyze_composition_coverage(compositions, results, 'composition_coverage.png')
    
    print("✅ Composition analysis completed!")

if __name__ == "__main__":
    run_composition_analysis()
