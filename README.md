# Prompt2Peptide

**Controllable antimicrobial peptide generation from natural-language prompts.**

Prompt2Peptide maps free-text descriptions (e.g., "cationic amphipathic helix, length 12-18") to biophysical target ranges (charge, hydrophobic moment, GRAVY, length) and optimizes amino-acid sequences via two-phase charge-curriculum simulated annealing.

Paper: *Prompt2Peptide: Learned Text-to-Property Control for Controllable Peptide Generation* (ACM BCB 2025 Workshop Submission)

## Method

1. **Prompt Encoding** -- A sentence-transformer encodes natural-language prompts and maps them to biophysical property targets via learned regression heads.
2. **Phase 1: Charge Curriculum** -- Simulated annealing first satisfies the net-charge constraint (the hardest to steer post hoc), then broadens the objective to all properties.
3. **Phase 2: Full Optimization** -- Joint optimization over charge, hydrophobic moment, GRAVY, and length with adaptive temperature scheduling.
4. **Safety Filtering** -- Seven biologically grounded filters (homopolymer runs, cysteine count, extreme hydrophobicity, charge density, low complexity, proline kinks, forbidden motifs).

## Results

Averaged over 5 peptide families, 100 sequences each, 5 random seeds:

| Method | Coverage | Safety@F | Novelty | Diversity |
|---|---|---|---|---|
| **Prompt2Peptide** | 0.754 | 0.564 | 1.000 | 0.749 |
| Single-Phase SA | 0.756 | 0.483 | 1.000 | 0.769 |
| Random GA | 0.856 | 0.668 | 0.998 | 0.736 |
| Random+Filter | 0.724 | 0.619 | 1.000 | 0.741 |
| No Curriculum (ablation) | 0.788 | 0.617 | 1.000 | 0.749 |
| Charge-Only (ablation) | 0.760 | 0.607 | 1.000 | 0.744 |

Coverage = fraction of sequences satisfying all biophysical constraints. Safety@F = safety rate among feasible sequences. Novelty verified against 201 known AMPs (max identity < 0.70).

Raw results in `results/experiment_results.json`.

## Repository Structure

```
motifgen/                       Core library
  generate.py                   Two-phase SA with charge curriculum
  metrics.py                    Biophysical property calculations
  esm_score.py                  ESM-2 pseudo-perplexity scoring
  prompt_encoder.py             Sentence-transformer prompt encoding

scripts/
  run_real_experiments.py       Full experiment runner (baselines + ablations)
  recompute_novelty.py          Expanded novelty analysis (201 reference AMPs)

results/                        Experiment outputs
  experiment_results.json       Main results (all methods x all families)
  expanded_novelty_results.json Expanded novelty analysis
  all_results_summary.csv       Flat summary table
  sequences_*.csv               Per-family generated sequences

figures/                        Figures from real experiments (8 PNGs)

final-draft/                    ACM BCB workshop submission (LaTeX source + PDF)
```

## Quick Start

```bash
pip install -r requirements.txt
```

### Generate peptides from a prompt

```python
from motifgen.generate import generate_peptides

sequences = generate_peptides(
    prompt="cationic amphipathic helix, length 12-18",
    n_sequences=10,
    n_steps=1000
)
```

### Reproduce all experiments

```bash
python scripts/run_real_experiments.py
```

Runs all 6 methods across 5 peptide families (100 sequences x 5 seeds each). Results are written to `results/`.

### Expanded novelty analysis

```bash
python scripts/recompute_novelty.py
```

Recomputes sequence novelty against 201 curated reference AMPs spanning cathelicidins, defensins, magainins, cecropins, temporins, and more.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9
- fair-esm >= 2.0 (ESM-2 protein language model)
- sentence-transformers >= 2.2
- scipy, scikit-learn, pandas, matplotlib, seaborn

## Citation

```bibtex
@inproceedings{bansal2025prompt2peptide,
  title={Prompt2Peptide: Learned Text-to-Property Control for Controllable Peptide Generation},
  author={Bansal, Aayam},
  booktitle={ACM BCB 2025 Workshop},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
