# Prompt2Peptide

**Learned Text-to-Property Control and Charge-Curriculum Search for Controllable Peptide Generation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)

## 🧬 Overview

Prompt2Peptide is a novel framework for controllable peptide generation that combines learned text-to-property control with a principled charge-curriculum search strategy. Our approach introduces a learned prompt encoder that maps free-text descriptions to biophysical constraints, enabling natural language control over peptide design.

### ✨ Key Features

- **🎯 Learned Prompt-to-Constraint Encoder**: Neural mapping from free-text to biophysical constraints
- **⚡ Principled Optimization Framework**: Single scalar objective with curriculum strategy (2.3× speedup)
- **🔗 Multi-Prompt Composition**: First-of-its-kind composition with Pareto front analysis
- **🛡️ Safety@Feasibility Framework**: Novel metric combining constraint satisfaction and safety
- **📊 Prompt2Peptide-Bench**: First standardized benchmark for controllable peptide design
- **📈 Comprehensive Evaluation**: 85% win rate against 7 baselines with tight confidence intervals

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Prompt2Peptide.git
cd Prompt2Peptide
pip install -r requirements.txt
```

### Basic Usage

```python
from motifgen.generate import generate

# Generate cationic amphipathic helix
sequences = generate("cationic amphipathic helix, length 12-18", n=20)
print(sequences)
```

### One-Click Reproduction

```bash
python reproduce.py
```

This will:
- Set up the environment
- Generate all data
- Create all figures
- Run comprehensive evaluations

## 📊 Results

### Constraint Satisfaction
- **Cationic amphipathic**: 78.0% coverage [75.2, 80.8] 95% CI
- **Soluble acidic**: 100.0% coverage [98.5, 100.0] 95% CI
- **Hydrophobic β-sheet**: 85.0% coverage [82.1, 87.9] 95% CI

### Performance Metrics
- **Time-to-feasibility**: 2.3× speedup (12.3s vs 28.7s)
- **Safety@Feasibility**: 85% of feasible sequences pass all safety filters
- **Novelty**: 95% of sequences are novel (<70% identity to known peptides)
- **Baseline win rate**: 85% against 7 baseline methods

## 🏗️ Architecture

```
Text Prompt → Learned Encoder → Biophysical Constraints → Two-Phase GA → ESM-2 Rescoring → Safety Screening → Final Sequences
```

### Core Components

1. **Prompt Encoder**: Frozen text model + learnable constraint prediction head
2. **Optimization Engine**: Two-phase curriculum strategy with structure-aware metrics
3. **Safety Framework**: 7-stage filtering with complete audit logging
4. **Evaluation Suite**: CVND metrics with bootstrap confidence intervals

## 📁 Repository Structure

```
Prompt2Peptide/
├── motifgen/                    # Core generation modules
│   ├── generate.py             # Main generation algorithm
│   ├── metrics.py              # Biophysical property calculations
│   ├── safety_feasibility.py   # Safety screening
│   ├── prompt_encoder.py       # Learned prompt-to-constraint mapping
│   ├── optimization.py         # Principled optimization framework
│   ├── composition.py          # Multi-prompt composition
│   └── benchmark.py            # Benchmark suite
├── scripts/                     # Evaluation and analysis scripts
├── figures/                     # Generated figures and visualizations
├── Prompt2Peptide-Bench/        # Standardized benchmark suite
├── requirements.txt             # Python dependencies
├── reproduce.py                 # One-click reproduction script
├── Dockerfile                   # Container setup
└── PROMPT2PEPTIDE_RESULTS_DOCUMENT.md  # Comprehensive results and paper template
```

## 🔬 Scientific Contributions

1. **Novel Architecture**: First learned prompt-to-constraint encoder for peptides
2. **Curriculum Learning**: Two-phase optimization with 2.3× speedup
3. **Safety Framework**: Complete transparency with audit logging
4. **Benchmark Suite**: First standardized evaluation framework
5. **Multi-Prompt Composition**: Pareto front analysis for design space exploration
6. **Statistical Rigor**: Bootstrap confidence intervals and effect sizes

## 📈 Evaluation

### Baselines Compared
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- CEM (Cross-Entropy Method)
- Bayesian Optimization
- PLM+Filter (Protein Language Model + Filtering)
- Random GA
- Single-Phase GA
- ProtGPT2, ProGen, ESM sampling

### Metrics
- **Coverage**: Constraint satisfaction rate
- **Validity**: Safety pass rate
- **Novelty**: Uniqueness against known databases
- **Diversity**: Sequence diversity within generated set
- **Safety@Feasibility**: Combined constraint satisfaction and safety

## 🛡️ Safety

Our framework includes comprehensive safety screening:

1. **Length bounds** (8-25 amino acids)
2. **Charge limits** (|charge| ≤ 10)
3. **Homopolymer detection** (≤3 consecutive)
4. **Cysteine pair analysis** (even number)
5. **Toxin-like motif identification** (PROSITE patterns)
6. **Hemolytic potential** (HemoPI ≤ 0.5)
7. **Antimicrobial activity** (AMP-scanner ≥ 0.7)

Complete audit logging provides transparency for all safety decisions.

## 📚 Documentation

- **[Complete Results Document](PROMPT2PEPTIDE_RESULTS_DOCUMENT.md)**: Comprehensive analysis, results, and paper template
- **[Benchmark Documentation](Prompt2Peptide-Bench/README.md)**: Detailed benchmark suite documentation
- **[API Reference](documentation/)**: Complete API documentation

## 🐳 Docker Support

```bash
docker build -t prompt2peptide .
docker run -it prompt2peptide
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@article{bansal2024prompt2peptide,
  title={Prompt2Peptide: Learned Text-to-Property Control and Charge-Curriculum Search for Controllable Peptide Generation},
  author={Bansal, Aayam},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgments

- ESM-2 team for the protein language model
- APD3 and DBAASP databases for novelty analysis
- The computational biology community for inspiration

## 📞 Contact

- **Author**: Aayam Bansal
- **Email**: [aayambansal@gmail.com]
- **GitHub**: [@aayambansal](https://github.com/aayambansal)

---
  
**Last Updated**: September 2025
