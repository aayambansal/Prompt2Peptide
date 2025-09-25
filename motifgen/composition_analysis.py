#!/usr/bin/env python3
"""
Composition as First-Class Results
Pareto fronts with hypervolume and dominated fraction vs baselines
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

@dataclass
class ParetoPoint:
    """A point on the Pareto front"""
    charge: float
    muh: float
    gravy: float
    feasibility: float
    safety: float
    sequence: str
    dominated: bool = False

class ParetoAnalyzer:
    """Comprehensive Pareto front analysis"""
    
    def __init__(self):
        self.objectives = ['charge', 'muh', 'gravy', 'feasibility', 'safety']
    
    def generate_pareto_fronts(self, 
                             compositions: List[str],
                             n_points: int = 100) -> Dict[str, List[ParetoPoint]]:
        """Generate Pareto fronts for different compositions"""
        
        pareto_fronts = {}
        
        for composition in compositions:
            print(f"  Generating Pareto front for: {composition}")
            
            # Generate trade-off points
            pareto_points = []
            
            for i in range(n_points):
                # Create trade-off parameter
                alpha = i / (n_points - 1)
                
                # Generate sequence with trade-off
                sequence = self._generate_tradeoff_sequence(composition, alpha)
                
                # Evaluate objectives
                objectives = self._evaluate_objectives(sequence, composition)
                
                pareto_point = ParetoPoint(
                    charge=objectives['charge'],
                    muh=objectives['muh'],
                    gravy=objectives['gravy'],
                    feasibility=objectives['feasibility'],
                    safety=objectives['safety'],
                    sequence=sequence
                )
                
                pareto_points.append(pareto_point)
            
            # Compute Pareto dominance
            pareto_points = self._compute_pareto_dominance(pareto_points)
            pareto_fronts[composition] = pareto_points
        
        return pareto_fronts
    
    def _generate_tradeoff_sequence(self, composition: str, alpha: float) -> str:
        """Generate sequence with specific trade-off parameter"""
        
        # Mock sequence generation based on composition and trade-off
        AA = "ACDEFGHIKLMNPQRSTVWY"
        length = np.random.randint(12, 19)
        
                # Adjust composition based on trade-off
        if "cationic" in composition.lower():
            # Trade-off between charge and other properties
            if alpha < 0.3:
                # High charge, lower other properties
                seq = ''.join(np.random.choice(['K', 'R'], length))
            elif alpha < 0.7:
                # Balanced
                seq = ''.join(np.random.choice(list(AA), length))
            else:
                # Lower charge, higher other properties
                seq = ''.join(np.random.choice(['A', 'L', 'V'], length))
        else:
            # Default generation
            seq = ''.join(np.random.choice(list(AA), length))
        
        return seq
    
    def _evaluate_objectives(self, sequence: str, composition: str) -> Dict[str, float]:
        """Evaluate multiple objectives for a sequence"""
        
        # Mock objective evaluation
        charge = self._calculate_charge(sequence)
        muh = self._calculate_muh(sequence)
        gravy = self._calculate_gravy(sequence)
        
        # Feasibility based on composition constraints
        if "cationic" in composition.lower():
            feasibility = 1.0 if 3 <= charge <= 8 else 0.5
        elif "acidic" in composition.lower():
            feasibility = 1.0 if -3 <= charge <= 0 else 0.5
        else:
            feasibility = 0.8
        
        # Safety based on sequence properties
        safety = 0.9 if len(sequence) <= 20 and abs(charge) <= 5 else 0.7
        
        return {
            'charge': charge,
            'muh': muh,
            'gravy': gravy,
            'feasibility': feasibility,
            'safety': safety
        }
    
    def _calculate_charge(self, sequence: str) -> float:
        """Calculate net charge"""
        pos_aa = 'KRH'
        neg_aa = 'DE'
        pos_count = sum(1 for aa in sequence if aa in pos_aa)
        neg_count = sum(1 for aa in sequence if aa in neg_aa)
        return pos_count - neg_count
    
    def _calculate_muh(self, sequence: str) -> float:
        """Calculate hydrophobic moment"""
        return np.random.uniform(0, 1)
    
    def _calculate_gravy(self, sequence: str) -> float:
        """Calculate GRAVY score"""
        return np.random.uniform(-1, 1)
    
    def _compute_pareto_dominance(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Compute Pareto dominance for points"""
        
        n = len(points)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if point i dominates point j
                    if self._dominates(points[i], points[j]):
                        points[j].dominated = True
        
        return points
    
    def _dominates(self, point1: ParetoPoint, point2: ParetoPoint) -> bool:
        """Check if point1 dominates point2"""
        
        # For maximization objectives (feasibility, safety)
        obj1 = [point1.charge, point1.muh, point1.gravy, point1.feasibility, point1.safety]
        obj2 = [point2.charge, point2.muh, point2.gravy, point2.feasibility, point2.safety]
        
        # Point1 dominates point2 if it's better in at least one objective
        # and not worse in any objective
        better_in_some = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        worse_in_none = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        
        return better_in_some and worse_in_none
    
    def compute_hypervolume(self, pareto_front: List[ParetoPoint], reference_point: List[float]) -> float:
        """Compute hypervolume indicator for Pareto front"""
        
        # Filter non-dominated points
        non_dominated = [p for p in pareto_front if not p.dominated]
        
        if not non_dominated:
            return 0.0
        
        # Convert to numpy array
        points = np.array([[p.charge, p.muh, p.gravy, p.feasibility, p.safety] for p in non_dominated])
        ref_point = np.array(reference_point)
        
        # Compute hypervolume (simplified 2D version for visualization)
        # In practice, would use proper hypervolume computation
        volume = 0.0
        
        for point in points:
            # Compute volume contribution
            volume += np.prod(np.maximum(0, point - ref_point))
        
        return volume
    
    def compute_dominated_fraction(self, pareto_front: List[ParetoPoint]) -> float:
        """Compute fraction of dominated points"""
        
        total_points = len(pareto_front)
        dominated_points = sum(1 for p in pareto_front if p.dominated)
        
        return dominated_points / total_points if total_points > 0 else 0.0
    
    def create_composition_plots(self, 
                               pareto_fronts: Dict[str, List[ParetoPoint]],
                               save_path: str):
        """Create comprehensive composition analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: 2D Pareto fronts (Charge vs μH)
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(pareto_fronts)))
        
        for i, (composition, points) in enumerate(pareto_fronts.items()):
            non_dominated = [p for p in points if not p.dominated]
            dominated = [p for p in points if p.dominated]
            
            if non_dominated:
                charges = [p.charge for p in non_dominated]
                muhs = [p.muh for p in non_dominated]
                ax1.scatter(charges, muhs, c=[colors[i]], label=composition, alpha=0.7, s=50)
            
            if dominated:
                charges = [p.charge for p in dominated]
                muhs = [p.muh for p in dominated]
                ax1.scatter(charges, muhs, c=[colors[i]], alpha=0.3, s=20)
        
        ax1.set_xlabel('Charge')
        ax1.set_ylabel('μH')
        ax1.set_title('Pareto Fronts: Charge vs μH')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 2D Pareto fronts (μH vs GRAVY)
        ax2 = axes[0, 1]
        
        for i, (composition, points) in enumerate(pareto_fronts.items()):
            non_dominated = [p for p in points if not p.dominated]
            
            if non_dominated:
                muhs = [p.muh for p in non_dominated]
                gravies = [p.gravy for p in non_dominated]
                ax2.scatter(muhs, gravies, c=[colors[i]], label=composition, alpha=0.7, s=50)
        
        ax2.set_xlabel('μH')
        ax2.set_ylabel('GRAVY')
        ax2.set_title('Pareto Fronts: μH vs GRAVY')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hypervolume comparison
        ax3 = axes[0, 2]
        compositions = list(pareto_fronts.keys())
        hypervolumes = []
        
        reference_point = [0, 0, -1, 0, 0]  # Reference point for hypervolume
        
        for composition in compositions:
            hv = self.compute_hypervolume(pareto_fronts[composition], reference_point)
            hypervolumes.append(hv)
        
        bars = ax3.bar(range(len(compositions)), hypervolumes, 
                      color=colors, alpha=0.7)
        ax3.set_xlabel('Composition')
        ax3.set_ylabel('Hypervolume')
        ax3.set_title('Hypervolume Comparison')
        ax3.set_xticks(range(len(compositions)))
        ax3.set_xticklabels([c.replace(' ', '\n') for c in compositions], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, hv in zip(bars, hypervolumes):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{hv:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Dominated fraction
        ax4 = axes[1, 0]
        dominated_fractions = []
        
        for composition in compositions:
            df = self.compute_dominated_fraction(pareto_fronts[composition])
            dominated_fractions.append(df)
        
        bars = ax4.bar(range(len(compositions)), dominated_fractions, 
                      color=colors, alpha=0.7)
        ax4.set_xlabel('Composition')
        ax4.set_ylabel('Dominated Fraction')
        ax4.set_title('Dominated Fraction Comparison')
        ax4.set_xticks(range(len(compositions)))
        ax4.set_xticklabels([c.replace(' ', '\n') for c in compositions], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, df in zip(bars, dominated_fractions):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{df:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: 3D Pareto front
        ax5 = axes[1, 1]
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        
        for i, (composition, points) in enumerate(pareto_fronts.items()):
            non_dominated = [p for p in points if not p.dominated]
            
            if non_dominated:
                charges = [p.charge for p in non_dominated]
                muhs = [p.muh for p in non_dominated]
                gravies = [p.gravy for p in non_dominated]
                ax5.scatter(charges, muhs, gravies, c=[colors[i]], label=composition, alpha=0.7)
        
        ax5.set_xlabel('Charge')
        ax5.set_ylabel('μH')
        ax5.set_zlabel('GRAVY')
        ax5.set_title('3D Pareto Front')
        ax5.legend()
        
        # Plot 6: Coverage vs diversity
        ax6 = axes[1, 2]
        coverages = []
        diversities = []
        
        for composition in compositions:
            points = pareto_fronts[composition]
            non_dominated = [p for p in points if not p.dominated]
            
            # Coverage: fraction of non-dominated points
            coverage = len(non_dominated) / len(points) if points else 0
            
            # Diversity: spread of points
            if len(non_dominated) > 1:
                coords = np.array([[p.charge, p.muh, p.gravy] for p in non_dominated])
                distances = pdist(coords)
                diversity = np.mean(distances)
            else:
                diversity = 0
            
            coverages.append(coverage)
            diversities.append(diversity)
        
        scatter = ax6.scatter(coverages, diversities, c=colors, s=100, alpha=0.7)
        ax6.set_xlabel('Coverage (Non-dominated Fraction)')
        ax6.set_ylabel('Diversity (Mean Pairwise Distance)')
        ax6.set_title('Coverage vs Diversity')
        ax6.grid(True, alpha=0.3)
        
        # Add labels
        for i, composition in enumerate(compositions):
            ax6.annotate(composition, (coverages[i], diversities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'hypervolumes': dict(zip(compositions, hypervolumes)),
            'dominated_fractions': dict(zip(compositions, dominated_fractions)),
            'coverages': dict(zip(compositions, coverages)),
            'diversities': dict(zip(compositions, diversities))
        }

def run_composition_analysis():
    """Run comprehensive composition analysis"""
    print("🔬 Running composition analysis...")
    
    analyzer = ParetoAnalyzer()
    
    compositions = [
        'cationic amphipathic helix',
        'soluble acidic loop',
        'hydrophobic beta sheet',
        'polar flexible linker',
        'basic nuclear localization'
    ]
    
    # Generate Pareto fronts
    pareto_fronts = analyzer.generate_pareto_fronts(compositions, n_points=50)
    
    # Create analysis plots
    results = analyzer.create_composition_plots(pareto_fronts, 'composition_pareto_analysis.png')
    
    print("✅ Composition analysis completed!")
    print(f"Hypervolumes: {results['hypervolumes']}")
    print(f"Dominated fractions: {results['dominated_fractions']}")
    
    return results

if __name__ == "__main__":
    run_composition_analysis()
