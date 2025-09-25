#!/usr/bin/env python3
"""
Structure and Interpretability Visualizations
Helical-wheel snapshots and ESMFold pLDDT trends
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge
import seaborn as sns

class HelicalWheelVisualizer:
    """Create helical wheel visualizations for amphipathic peptides"""
    
    def __init__(self):
        self.aa_properties = {
            'A': {'hydrophobic': True, 'charge': 0, 'color': '#FF6B6B'},
            'R': {'hydrophobic': False, 'charge': 1, 'color': '#4ECDC4'},
            'N': {'hydrophobic': False, 'charge': 0, 'color': '#45B7D1'},
            'D': {'hydrophobic': False, 'charge': -1, 'color': '#96CEB4'},
            'C': {'hydrophobic': True, 'charge': 0, 'color': '#FFEAA7'},
            'Q': {'hydrophobic': False, 'charge': 0, 'color': '#DDA0DD'},
            'E': {'hydrophobic': False, 'charge': -1, 'color': '#98D8C8'},
            'G': {'hydrophobic': False, 'charge': 0, 'color': '#F7DC6F'},
            'H': {'hydrophobic': False, 'charge': 0.5, 'color': '#BB8FCE'},
            'I': {'hydrophobic': True, 'charge': 0, 'color': '#85C1E9'},
            'L': {'hydrophobic': True, 'charge': 0, 'color': '#F8C471'},
            'K': {'hydrophobic': False, 'charge': 1, 'color': '#82E0AA'},
            'M': {'hydrophobic': True, 'charge': 0, 'color': '#F1948A'},
            'F': {'hydrophobic': True, 'charge': 0, 'color': '#D7BDE2'},
            'P': {'hydrophobic': False, 'charge': 0, 'color': '#A9DFBF'},
            'S': {'hydrophobic': False, 'charge': 0, 'color': '#AED6F1'},
            'T': {'hydrophobic': False, 'charge': 0, 'color': '#A3E4D7'},
            'W': {'hydrophobic': True, 'charge': 0, 'color': '#F9E79F'},
            'Y': {'hydrophobic': True, 'charge': 0, 'color': '#D5DBDB'},
            'V': {'hydrophobic': True, 'charge': 0, 'color': '#FADBD8'}
        }
    
    def create_helical_wheel(self, sequence: str, title: str = "", save_path: str = None):
        """Create helical wheel visualization for a sequence"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Helical wheel parameters
        n_residues = len(sequence)
        radius = 3.0
        center_x, center_y = 0, 0
        
        # Calculate positions for each residue
        positions = []
        for i, aa in enumerate(sequence):
            # Helical wheel: 100° per residue
            angle = i * 100 * np.pi / 180
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions.append((x, y, angle, aa))
        
        # Draw circles for different property regions
        self._draw_property_regions(ax, center_x, center_y, radius)
        
        # Draw residues
        for x, y, angle, aa in positions:
            self._draw_residue(ax, x, y, aa, angle)
        
        # Add title and labels
        ax.set_title(f'Helical Wheel: {title}\n{sequence}', fontsize=14, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        self._add_legend(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _draw_property_regions(self, ax, center_x, center_y, radius):
        """Draw regions for different amino acid properties"""
        
        # Hydrophobic region (top half)
        hydrophobic_wedge = Wedge((center_x, center_y), radius + 0.5, 0, 180, 
                                width=0.5, alpha=0.2, color='red', label='Hydrophobic')
        ax.add_patch(hydrophobic_wedge)
        
        # Hydrophilic region (bottom half)
        hydrophilic_wedge = Wedge((center_x, center_y), radius + 0.5, 180, 360, 
                                width=0.5, alpha=0.2, color='blue', label='Hydrophilic')
        ax.add_patch(hydrophilic_wedge)
        
        # Main circle
        main_circle = Circle((center_x, center_y), radius, fill=False, 
                           linewidth=2, color='black')
        ax.add_patch(main_circle)
    
    def _draw_residue(self, ax, x, y, aa, angle):
        """Draw a single amino acid residue"""
        
        properties = self.aa_properties.get(aa, {'hydrophobic': False, 'charge': 0, 'color': '#CCCCCC'})
        
        # Draw residue circle
        residue_circle = Circle((x, y), 0.3, color=properties['color'], 
                              edgecolor='black', linewidth=1)
        ax.add_patch(residue_circle)
        
        # Add amino acid letter
        ax.text(x, y, aa, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add charge indicator
        if properties['charge'] != 0:
            charge_symbol = '+' if properties['charge'] > 0 else '-'
            charge_x = x + 0.4 * np.cos(angle)
            charge_y = y + 0.4 * np.sin(angle)
            ax.text(charge_x, charge_y, charge_symbol, ha='center', va='center', 
                   fontsize=8, color='red' if properties['charge'] > 0 else 'blue')
    
    def _add_legend(self, ax):
        """Add legend for amino acid properties"""
        
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.2, label='Hydrophobic'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.2, label='Hydrophilic'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#4ECDC4', label='Positive Charge'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#96CEB4', label='Negative Charge'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', label='Neutral')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    def create_multiple_helical_wheels(self, sequences: List[Tuple[str, str]], save_path: str):
        """Create multiple helical wheels in a grid"""
        
        n_sequences = len(sequences)
        cols = min(3, n_sequences)
        rows = (n_sequences + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (sequence, title) in enumerate(sequences):
            if i < len(axes):
                ax = axes[i]
                
                # Create helical wheel for this sequence
                self._create_helical_wheel_subplot(ax, sequence, title)
        
        # Hide unused subplots
        for i in range(n_sequences, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_helical_wheel_subplot(self, ax, sequence: str, title: str):
        """Create helical wheel in a subplot"""
        
        # Helical wheel parameters
        n_residues = len(sequence)
        radius = 2.0
        center_x, center_y = 0, 0
        
        # Calculate positions for each residue
        positions = []
        for i, aa in enumerate(sequence):
            # Helical wheel: 100° per residue
            angle = i * 100 * np.pi / 180
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions.append((x, y, angle, aa))
        
        # Draw property regions
        hydrophobic_wedge = Wedge((center_x, center_y), radius + 0.3, 0, 180, 
                                width=0.3, alpha=0.2, color='red')
        ax.add_patch(hydrophobic_wedge)
        
        hydrophilic_wedge = Wedge((center_x, center_y), radius + 0.3, 180, 360, 
                                width=0.3, alpha=0.2, color='blue')
        ax.add_patch(hydrophilic_wedge)
        
        # Main circle
        main_circle = Circle((center_x, center_y), radius, fill=False, 
                           linewidth=1.5, color='black')
        ax.add_patch(main_circle)
        
        # Draw residues
        for x, y, angle, aa in positions:
            properties = self.aa_properties.get(aa, {'hydrophobic': False, 'charge': 0, 'color': '#CCCCCC'})
            
            # Draw residue circle
            residue_circle = Circle((x, y), 0.2, color=properties['color'], 
                                  edgecolor='black', linewidth=0.5)
            ax.add_patch(residue_circle)
            
            # Add amino acid letter
            ax.text(x, y, aa, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Set title and limits
        ax.set_title(f'{title}\n{sequence}', fontsize=10, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')

class ESMFoldAnalyzer:
    """Analyze ESMFold confidence trends"""
    
    def __init__(self):
        pass
    
    def analyze_plddt_trends(self, sequences: List[str], save_path: str):
        """Analyze pLDDT confidence trends for sequences"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mock pLDDT data (in practice would use actual ESMFold)
        plddt_data = {}
        for seq in sequences:
            # Simulate pLDDT values with some structure
            length = len(seq)
            base_confidence = np.random.uniform(0.6, 0.9)
            
            # Add some structure-dependent variation
            plddt_values = []
            for i in range(length):
                # Helical regions tend to have higher confidence
                if i % 3 == 0:  # Mock helical pattern
                    confidence = base_confidence + np.random.uniform(0, 0.2)
                else:
                    confidence = base_confidence + np.random.uniform(-0.1, 0.1)
                
                plddt_values.append(min(1.0, max(0.0, confidence)))
            
            plddt_data[seq] = plddt_values
        
        # Plot 1: pLDDT trends for individual sequences
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sequences)))
        
        for i, (seq, plddt_values) in enumerate(plddt_data.items()):
            ax1.plot(range(len(plddt_values)), plddt_values, 
                    color=colors[i], alpha=0.7, linewidth=2, label=seq[:10] + '...')
        
        ax1.set_xlabel('Residue Position')
        ax1.set_ylabel('pLDDT Confidence')
        ax1.set_title('ESMFold pLDDT Trends by Sequence')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Average pLDDT by position
        ax2 = axes[0, 1]
        max_length = max(len(seq) for seq in sequences)
        position_means = []
        position_stds = []
        
        for pos in range(max_length):
            position_values = []
            for seq, plddt_values in plddt_data.items():
                if pos < len(plddt_values):
                    position_values.append(plddt_values[pos])
            
            if position_values:
                position_means.append(np.mean(position_values))
                position_stds.append(np.std(position_values))
            else:
                position_means.append(0)
                position_stds.append(0)
        
        positions = range(len(position_means))
        ax2.plot(positions, position_means, 'b-', linewidth=2, label='Mean pLDDT')
        ax2.fill_between(positions, 
                        np.array(position_means) - np.array(position_stds),
                        np.array(position_means) + np.array(position_stds),
                        alpha=0.3, color='blue', label='±1 std')
        
        ax2.set_xlabel('Residue Position')
        ax2.set_ylabel('Average pLDDT')
        ax2.set_title('Average pLDDT by Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: pLDDT distribution
        ax3 = axes[1, 0]
        all_plddt = []
        for plddt_values in plddt_data.values():
            all_plddt.extend(plddt_values)
        
        ax3.hist(all_plddt, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(all_plddt), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_plddt):.3f}')
        ax3.set_xlabel('pLDDT Confidence')
        ax3.set_ylabel('Frequency')
        ax3.set_title('pLDDT Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confidence vs sequence properties
        ax4 = axes[1, 1]
        mean_confidences = [np.mean(plddt_values) for plddt_values in plddt_data.values()]
        sequence_lengths = [len(seq) for seq in sequences]
        
        scatter = ax4.scatter(sequence_lengths, mean_confidences, 
                            c=colors, s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(sequence_lengths, mean_confidences, 1)
        p = np.poly1d(z)
        ax4.plot(sequence_lengths, p(sequence_lengths), "r--", alpha=0.8)
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Mean pLDDT')
        ax4.set_title('Confidence vs Sequence Length')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class FailureModeAnalyzer:
    """Analyze failure modes with Sankey diagram"""
    
    def __init__(self):
        pass
    
    def create_failure_sankey(self, save_path: str):
        """Create Sankey diagram showing constraint failures pre/post curriculum"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Mock failure data
        pre_curriculum_failures = {
            'Charge': 45,
            'μH': 38,
            'GRAVY': 42,
            'Composition': 35,
            'Length': 28
        }
        
        post_curriculum_failures = {
            'Charge': 15,
            'μH': 25,
            'GRAVY': 30,
            'Composition': 20,
            'Length': 18
        }
        
        # Create simplified Sankey-like visualization
        constraints = list(pre_curriculum_failures.keys())
        y_positions = np.linspace(0.8, 0.2, len(constraints))
        
        # Pre-curriculum bars
        for i, (constraint, failures) in enumerate(pre_curriculum_failures.items()):
            # Left side (pre-curriculum)
            rect = patches.Rectangle((0.1, y_positions[i] - 0.05), 0.3, 0.1, 
                                   facecolor='red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.25, y_positions[i], f'{constraint}\n{failures}%', 
                   ha='center', va='center', fontweight='bold')
        
        # Post-curriculum bars
        for i, (constraint, failures) in enumerate(post_curriculum_failures.items()):
            # Right side (post-curriculum)
            rect = patches.Rectangle((0.6, y_positions[i] - 0.05), 0.3, 0.1, 
                                   facecolor='green', alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.75, y_positions[i], f'{constraint}\n{failures}%', 
                   ha='center', va='center', fontweight='bold')
        
        # Add arrows showing improvement
        for i, constraint in enumerate(constraints):
            pre_fail = pre_curriculum_failures[constraint]
            post_fail = post_curriculum_failures[constraint]
            improvement = pre_fail - post_fail
            
            # Arrow from pre to post
            ax.annotate('', xy=(0.6, y_positions[i]), xytext=(0.4, y_positions[i]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            
            # Improvement label
            ax.text(0.5, y_positions[i] + 0.08, f'-{improvement}%', 
                   ha='center', va='center', fontweight='bold', color='blue')
        
        # Add titles
        ax.text(0.25, 0.95, 'Pre-Curriculum\nFailure Rates', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        ax.text(0.75, 0.95, 'Post-Curriculum\nFailure Rates', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Constraint Failure Analysis: Pre vs Post Curriculum', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_structure_visualization():
    """Run comprehensive structure visualization analysis"""
    print("🔬 Running structure visualization analysis...")
    
    # Create helical wheel visualizer
    hw_visualizer = HelicalWheelVisualizer()
    
    # Example sequences for helical wheels
    sequences = [
        ("GRVRFFIIHQHMIRLRK", "Cationic Amphipathic Helix"),
        ("CHAFRARTFARGRIKLV", "Cationic Amphipathic Helix"),
        ("KKLLKLLKKLLKLLKK", "Synthetic Amphipathic"),
        ("DDEEEDDEEEDDEEED", "Acidic Loop")
    ]
    
    # Create multiple helical wheels
    hw_visualizer.create_multiple_helical_wheels(sequences, 'helical_wheels_analysis.png')
    
    # Create ESMFold analyzer
    esm_analyzer = ESMFoldAnalyzer()
    esm_analyzer.analyze_plddt_trends([seq for seq, _ in sequences], 'esmfold_plddt_analysis.png')
    
    # Create failure mode analyzer
    failure_analyzer = FailureModeAnalyzer()
    failure_analyzer.create_failure_sankey('failure_mode_sankey.png')
    
    print("✅ Structure visualization analysis completed!")

if __name__ == "__main__":
    run_structure_visualization()
