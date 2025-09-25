# Prompt2Peptide: Text-Conditioned Generation of Short Peptide Motifs

A fast pipeline that converts natural-language motif specifications into de novo peptide candidates using biophysical constraints and language model scoring.

## Overview

This project implements a two-stage peptide generation pipeline:

1. **Constraint-guided sampling**: Converts text prompts to biophysical constraints and uses local search to generate sequences
2. **Language model rescoring**: Uses ESM-2 (t6-8M) to score sequences for plausibility

## Features

- **Text-to-constraint mapping**: Natural language prompts → biophysical targets
- **Biophysical metrics**: GRAVY, net charge, hydrophobic moment
- **ESM-2 integration**: Language model plausibility scoring
- **Fast generation**: Optimized for short peptides (8-25 aa)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Generation

```bash
python main.py --prompt "cationic amphipathic helix, length 12–18"
```

### Advanced Options

```bash
python main.py --prompt "soluble acidic loop 10–14" --n 40 --iters 600 --out results.csv
```

### Evaluation

```bash
python evaluate.py
```

## Example Results

### Cationic Amphipathic Helix
```
1. KSGWRAWFIRILRL (μH=0.544, charge=4.00, GRAVY=0.057, ESM-LL=-4.024)
2. CYIKYRLYLHIFTK (μH=0.539, charge=3.04, GRAVY=0.129, ESM-LL=-4.171)
3. MFMIAHCRNLCGKK (μH=0.733, charge=3.04, GRAVY=0.164, ESM-LL=-4.216)
```

### Soluble Acidic Loop
```
1. DWVMVQPDPTK (μH=0.314, charge=-1.00, GRAVY=-0.809, ESM-LL=-4.152)
2. AAWQGVAEEEA (μH=0.399, charge=-3.00, GRAVY=-0.355, ESM-LL=-4.153)
3. FNDEHEWYLAA (μH=0.327, charge=-2.96, GRAVY=-0.836, ESM-LL=-4.479)
```

## Constraint Mapping

| Descriptor | Target Constraints |
|------------|-------------------|
| cationic | net charge +3 to +8 |
| amphipathic helix | μH ≥ 0.35 |
| hydrophobic | GRAVY ≥ 0.2 |
| soluble | GRAVY ≤ 0 |
| acidic | net charge -3 to 0 |

## Project Structure

```
prompt2peptide/
├── motifgen/
│   ├── __init__.py
│   ├── metrics.py          # Biophysical calculations
│   ├── esm_score.py        # ESM-2 integration
│   └── generate.py         # Core generation logic
├── main.py                 # CLI interface
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Dependencies

- `fair-esm>=2.0.0` - ESM-2 language model
- `torch>=1.9.0` - PyTorch backend
- `numpy>=1.21.0` - Numerical computations

## Performance

- **Generation speed**: ~10-20 sequences/minute
- **Model size**: ESM-2-t6-8M (8M parameters)
- **Memory usage**: <2GB RAM
- **Sequence length**: Optimized for 8-25 amino acids

## Evaluation Results

### Constraint Satisfaction
- **Cationic amphipathic**: 60% charge, 100% μH, 100% GRAVY
- **Soluble acidic**: 100% charge, 90% μH, 100% GRAVY

### Language Model Scores
- Generated sequences maintain competitive ESM-2 log-likelihoods
- Average ESM-LL: -4.2 to -4.4 (comparable to natural peptides)

## Limitations

- No wet-lab validation
- Simple heuristic constraints
- Limited to short peptide motifs
- No structural prediction

## Future Work

- Fine-tuned text encoders
- LoRA adaptation of ESM-2
- Structure-aware generation
- Wet-lab validation

## Citation

```bibtex
@article{prompt2peptide2024,
  title={Prompt2Peptide: Text-Conditioned Generation of Short Motifs with Biophysical Control},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
