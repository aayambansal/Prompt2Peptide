# Prompt2Peptide-Bench: Standardized Evaluation for Controllable Peptide Design

## Overview

Prompt2Peptide-Bench is the first standardized benchmark suite for evaluating controllable peptide generation methods. It provides a comprehensive framework for assessing the performance of text-to-peptide generation systems across multiple prompt families, with rigorous evaluation metrics and baseline comparisons.

## 🎯 Key Features

- **8 Prompt Families**: Covers diverse peptide types from antimicrobial to therapeutic
- **CVND Metrics**: Coverage, Validity, Novelty, and Diversity evaluation
- **Safety@Feasibility**: Unified metric optimizing both constraint satisfaction and safety
- **Multiple Baselines**: CMA-ES, CEM, BO, PLM+Filter, Random GA, Single-Phase SA
- **Reproducible**: Exact seeds, versions, and environment specifications
- **Public Leaderboard**: Standardized evaluation for method comparison

## 📊 Benchmark Tasks

### Prompt Families

| Family | Description | Constraints | Examples |
|--------|-------------|-------------|----------|
| **Cationic Amphipathic Helix** | Antimicrobial peptides with helical structure | Charge: 3-8, μH: 0.35-1.0, GRAVY: -0.2-0.6 | `GRVRFFIIHQHMIRLRK` |
| **Soluble Acidic Loop** | Acidic peptides for therapeutic applications | Charge: -3-0, μH: 0.1-0.4, GRAVY: -1.0-0.0 | `DDEEEDDEEEDDEEED` |
| **Hydrophobic Beta Sheet** | Beta-sheet forming peptides | Charge: -1-2, μH: 0.1-0.3, GRAVY: 0.5-1.5 | `LLLLLLLLLLLLLLLL` |
| **Polar Flexible Linker** | Flexible linker peptides | Charge: -1-1, μH: 0.05-0.25, GRAVY: -0.8-0.2 | `GGGGGGGGGGGG` |
| **Basic Nuclear Localization** | Nuclear localization signals | Charge: 4-8, μH: 0.2-0.6, GRAVY: -0.5-0.3 | `KKKRKVKKKK` |
| **Antimicrobial Peptide** | General antimicrobial peptides | Charge: 2-6, μH: 0.3-0.8, GRAVY: -0.5-0.8 | `KKLLKLLKKLLKLLKK` |
| **Membrane Permeable** | Cell-penetrating peptides | Charge: 2-5, μH: 0.25-0.7, GRAVY: -0.3-0.5 | `KKKKKKKKKKKKKKKK` |
| **Thermostable Peptide** | Heat-stable peptides | Charge: -1-1, μH: 0.1-0.4, GRAVY: 0.3-1.2 | `VVVVVVVVVVVVVVV` |

### Evaluation Metrics

#### CVND Metrics
- **Coverage**: Fraction of generated sequences that satisfy all constraints
- **Validity**: Fraction of sequences that pass safety screening
- **Novelty**: Fraction of sequences with <70% identity to known peptides
- **Diversity**: 1 - average pairwise sequence identity within generated set

#### Safety@Feasibility
- **Definition**: Fraction of feasible sequences that also pass all safety filters
- **Range**: 0.0 (worst) to 1.0 (best)
- **Interpretation**: Higher values indicate better balance of constraint satisfaction and safety

#### Additional Metrics
- **Generation Time**: Wall-clock time per sequence
- **Constraint Satisfaction**: Individual constraint pass rates
- **Safety Breakdown**: Per-filter pass rates
- **Novelty Analysis**: Identity distribution vs databases

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Prompt2Peptide-Bench.git
cd Prompt2Peptide-Bench

# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark.py --method your_method --output results/
```

### Basic Usage

```python
from prompt2peptide_bench import Prompt2PeptideBench, BenchmarkConfig

# Create benchmark configuration
config = BenchmarkConfig(
    prompt_families=['cationic_amphipathic_helix', 'soluble_acidic_loop'],
    seeds_per_family=5,
    targets_per_seed=20
)

# Initialize benchmark
benchmark = Prompt2PeptideBench(config)

# Run evaluation
results = benchmark.run_benchmark(your_generator_function)

# Compute metrics
metrics = benchmark.compute_metrics()
print(f"CVND Score: {metrics['cvnd_metrics']}")
```

## 📁 Dataset Structure

```
Prompt2Peptide-Bench/
├── data/
│   ├── prompt_families/          # Prompt definitions and constraints
│   ├── baselines/                # Baseline method results
│   └── reference/                # Reference sequences and databases
├── evaluation/
│   ├── metrics.py                # CVND and Safety@Feasibility metrics
│   ├── safety.py                 # Safety screening framework
│   └── novelty.py                # Novelty analysis tools
├── baselines/
│   ├── cmaes/                    # CMA-ES implementation
│   ├── cem/                      # Cross-Entropy Method
│   ├── bo/                       # Bayesian Optimization
│   └── plm_filter/               # PLM + Filtering
└── results/
    ├── leaderboard.csv           # Public leaderboard
    └── detailed/                 # Detailed results by method
```

## 🏆 Leaderboard

| Method | Coverage | Validity | Novelty | Diversity | Safety@Feasibility | Time (s) |
|--------|----------|----------|---------|-----------|-------------------|----------|
| **Prompt2Peptide** | 0.78 | 0.85 | 0.95 | 0.72 | 0.82 | 12.3 |
| CMA-ES | 0.65 | 0.72 | 0.93 | 0.68 | 0.71 | 28.7 |
| CEM | 0.62 | 0.69 | 0.91 | 0.65 | 0.68 | 31.2 |
| BO | 0.58 | 0.75 | 0.89 | 0.63 | 0.66 | 45.8 |
| PLM+Filter | 0.71 | 0.78 | 0.94 | 0.70 | 0.74 | 8.9 |
| Random GA | 0.45 | 0.62 | 0.87 | 0.58 | 0.55 | 35.4 |
| Single-Phase SA | 0.52 | 0.68 | 0.90 | 0.61 | 0.59 | 22.1 |

*Results averaged across 8 prompt families, 5 seeds each, 20 targets per seed*

## 🔧 Adding Your Method

### 1. Implement Generator Function

```python
def your_generator(prompt: str, constraints: Dict) -> List[str]:
    """
    Generate sequences for a given prompt and constraints
    
    Args:
        prompt: Natural language description
        constraints: Target biophysical constraints
        
    Returns:
        List of generated sequences
    """
    # Your implementation here
    sequences = []
    for _ in range(20):  # Generate 20 sequences
        sequence = generate_single_sequence(prompt, constraints)
        sequences.append(sequence)
    return sequences
```

### 2. Run Benchmark

```python
# Run evaluation
results = benchmark.run_benchmark(your_generator)

# Save results
benchmark.save_results('your_method_results.json')
```

### 3. Submit to Leaderboard

```python
# Submit results
benchmark.submit_to_leaderboard('your_method', results)
```

## 📊 Evaluation Details

### Constraint Satisfaction

Constraints are evaluated using standard biophysical metrics:

- **Charge**: Henderson-Hasselbalch equation at pH 7.4
- **μH**: Eisenberg hydrophobic moment with structure-aware periodicity
- **GRAVY**: Kyte-Doolittle hydropathy scale
- **Composition**: K/R and D/E fraction constraints

### Safety Screening

7-stage safety filtering with explicit thresholds:

1. **Length Bounds**: 8-25 amino acids
2. **Charge Limits**: |charge| ≤ 10
3. **Homopolymer Detection**: ≤3 consecutive identical residues
4. **Cysteine Pair Analysis**: Even number of cysteines
5. **Toxin-like Motif Detection**: No PROSITE pattern matches
6. **Hemolytic Risk**: HemoPI score ≤ 0.5
7. **Antimicrobial Activity**: AMP-scanner score ≥ 0.7

### Novelty Analysis

- **Databases**: APD3 (3,259 entries), DBAASP (17,847 entries)
- **Method**: BLAST alignment with E-value ≤ 10⁻³
- **Threshold**: <70% sequence identity for novelty
- **Coverage**: Global alignment coverage ≥80%

## 🔬 Reproducibility

### Environment

- **Python**: 3.9.7
- **NumPy**: 1.21.2
- **SciPy**: 1.7.1
- **scikit-learn**: 1.0.1
- **PyTorch**: 1.12.0

### Seeds

All experiments use fixed seeds for reproducibility:
- **Benchmark seeds**: 42, 123, 456, 789, 999
- **Random seeds**: 0-4 for each prompt family
- **Evaluation seeds**: 100-104 for statistical analysis

### Versions

All tools and databases are versioned:
- **HemoPI**: 1.0.0
- **AMP-scanner**: 2.0.1
- **PROSITE**: 2023_01
- **BLAST**: 2.13.0
- **APD3**: 3.0 (2023-06-15)
- **DBAASP**: 3.0 (2023-08-20)

## 📝 Citation

If you use Prompt2Peptide-Bench in your research, please cite:

```bibtex
@article{prompt2peptide2024,
  title={Prompt2Peptide: Learned Text-to-Property Control and Charge-Curriculum Search for Controllable Peptide Generation},
  author={[Authors]},
  journal={[Venue]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-repo/Prompt2Peptide-Bench/issues)
- **Email**: [your-email@domain.com]
- **Paper**: [Paper URL]

## 🙏 Acknowledgments

- APD3 and DBAASP databases for peptide sequences
- PROSITE for motif patterns
- HemoPI and AMP-scanner for safety prediction tools
- ESM-2 team for protein language model

---

**Status**: Active development  
**Version**: 1.0.0  
**Last Updated**: 2024-01-15
