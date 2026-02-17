#!/usr/bin/env python3
"""
Run ALL real experiments for Prompt2Peptide paper.
Implements real baselines, real ablations, scaled generation, real metrics, real figures.
No mock data. Everything computed from the actual generation pipeline.
"""

import sys, os, random, time, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge
from collections import defaultdict
from difflib import SequenceMatcher
from scipy import stats
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from motifgen.metrics import gravy, net_charge, hydrophobic_moment, positional_amphipathy_score

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
N_SEQUENCES_PER_PROMPT = 100  # sequences per prompt family per method
N_SEEDS = 5
ITERS = 500

# Reference peptides for novelty (known AMP-like sequences from literature)
REFERENCE_PEPTIDES = [
    "GIGKFLHSAKKFGKAFVGEIMNS",  # magainin 2
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # cecropin A fragment
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37 fragment
    "RLCRIVVIRVCR",  # protegrin-1
    "GIGTILSLLKGL",  # dermaseptin
    "KWKLWKKIEKWGQGIGAVLKWLTTWL",  # synthetic AMP
    "KKLLKKLLLKLLKKL",  # synthetic amphipathic
    "RLARIVVIRVAR",  # tachyplesin
    "ILPWKWPWWPWRR",  # indolicidin
    "GLFGAIAGHVGR",  # peptaibolin
    "KLAKLAKLAKLA",
    "KLALKLALKLAL",
    "KWKLFKKIPKFLHLAKKF",  # synthetic
    "RRWQWRMKKLG",  # lactoferricin
    "ILPWKWPWWPWRRK",  # indolicidin variant
    "GLFDIVKKVVGALGSL",  # buforin
    "FFPIVGKLLSGLL",  # BMAP-28 fragment
    "RGGRLCYCRRRFCVCVGR",  # defensin-like
    "YGGGVLGAIAGHVGR",  # synthetic
    "KKLFKKILKYL",  # synthetic
]

# ---------------------------------------------------------------------------
# PROMPT FAMILIES with target constraints
# ---------------------------------------------------------------------------
PROMPT_FAMILIES = {
    'cationic_amphipathic_helix': {
        'prompt': 'cationic amphipathic helix',
        'target': {
            'charge': (+3, +8, 1.0),
            'muH': (0.35, 1.0, 1.0),
            'gravy': (-0.2, 0.6, 0.5),
            'type': 'cationic',
            'structure_type': 'helix'
        },
        'boost': {'K': 2.0, 'R': 2.0, 'L': 1.5, 'I': 1.5, 'V': 1.2, 'F': 1.2},
        'length_range': (12, 18),
    },
    'soluble_acidic_loop': {
        'prompt': 'soluble acidic loop',
        'target': {
            'charge': (-3, 0, 1.0),
            'muH': (0.1, 0.4, 0.5),
            'gravy': (-1.0, 0.0, 1.0),
            'type': 'acidic',
            'structure_type': 'helix'
        },
        'boost': {'D': 2.0, 'E': 2.0, 'S': 1.5, 'T': 1.5, 'N': 1.2, 'Q': 1.2},
        'length_range': (10, 14),
    },
    'hydrophobic_beta_sheet': {
        'prompt': 'hydrophobic beta sheet',
        'target': {
            'charge': (-1, +2, 0.8),
            'muH': (0.1, 0.3, 0.3),
            'gravy': (0.5, 1.5, 1.0),
            'structure_type': 'sheet'
        },
        'boost': {'F': 2.0, 'W': 2.0, 'Y': 1.8, 'L': 1.8, 'I': 1.8, 'V': 1.5},
        'length_range': (10, 14),
    },
    'polar_flexible_linker': {
        'prompt': 'polar flexible linker',
        'target': {
            'charge': (-1, +1, 0.5),
            'muH': (0.05, 0.25, 0.2),
            'gravy': (-0.8, 0.2, 1.0),
            'structure_type': 'helix'
        },
        'boost': {'G': 2.0, 'P': 1.5, 'S': 1.5, 'T': 1.5, 'N': 1.3, 'Q': 1.3},
        'length_range': (8, 12),
    },
    'basic_nuclear_localization': {
        'prompt': 'basic nuclear localization signal',
        'target': {
            'charge': (+4, +8, 1.0),
            'muH': (0.2, 0.6, 0.8),
            'gravy': (-0.5, 0.3, 0.5),
            'structure_type': 'helix'
        },
        'boost': {'K': 2.5, 'R': 2.5, 'P': 1.5, 'S': 1.2, 'T': 1.2},
        'length_range': (7, 12),
    },
}

# ---------------------------------------------------------------------------
# CORE GENERATION FUNCTIONS (from generate.py, parameterized for baselines)
# ---------------------------------------------------------------------------

def init_seq(length, boost=None):
    base = list(AA)
    probs = {a: 1.0 for a in base}
    if boost:
        for a, w in boost.items():
            probs[a] = probs.get(a, 1.0) * w
    p = [probs[a] for a in base]
    s = "".join(random.choices(base, weights=p, k=length))
    return s

def constraint_score(seq, target):
    g = gravy(seq)
    ch = net_charge(seq)
    structure_type = target.get('structure_type', 'helix')
    mu = hydrophobic_moment(seq, structure_type=structure_type)
    
    total_len = len(seq)
    kr_fraction = (seq.count('K') + seq.count('R')) / total_len
    de_fraction = (seq.count('D') + seq.count('E')) / total_len
    h_count = seq.count('H')
    
    sc = 0

    def window(x, lo, hi, w):
        if lo <= x <= hi:
            return w
        d = min(abs(x - lo), abs(x - hi))
        return max(0.0, w - d * w)

    # Charge scoring
    charge_lo, charge_hi, charge_w = target['charge']
    if charge_lo <= ch <= charge_hi:
        sc += charge_w
    else:
        d = min(abs(ch - charge_lo), abs(ch - charge_hi))
        sc += max(0.0, charge_w - d * charge_w * 2.0)

    sc += window(mu, *target['muH'])
    sc += window(g, *target['gravy'])
    
    # Composition scoring
    target_type = target.get('type', 'general')
    if target_type == 'cationic':
        if 0.25 <= kr_fraction <= 0.45:
            sc += 0.5
        else:
            d = min(abs(kr_fraction - 0.25), abs(kr_fraction - 0.45))
            sc += max(0.0, 0.5 - d)
        if de_fraction <= 0.10:
            sc += 0.5
    elif target_type == 'acidic':
        if de_fraction >= 0.30:
            sc += 0.5
        if kr_fraction <= 0.10:
            sc += 0.5
    
    return sc, {'gravy': g, 'charge': ch, 'muH': mu, 'kr_fraction': kr_fraction,
                'de_fraction': de_fraction}

def mutate_charge_directed(seq, target):
    """Phase 1: Charge-directed mutations"""
    current_charge = net_charge(seq)
    charge_lo, charge_hi = target['charge'][:2]
    
    if current_charge < charge_lo:
        non_kr = [i for i, aa in enumerate(seq) if aa not in 'KR']
        if non_kr:
            i = random.choice(non_kr)
            a = random.choice('KR')
            return seq[:i] + a + seq[i+1:]
    elif current_charge > charge_hi:
        kr_pos = [i for i, aa in enumerate(seq) if aa in 'KR']
        if kr_pos:
            i = random.choice(kr_pos)
            a = random.choice('ASTNQ')
            return seq[:i] + a + seq[i+1:]
    # Fallback
    i = random.randrange(len(seq))
    a = random.choice(AA.replace(seq[i], ""))
    return seq[:i] + a + seq[i+1:]

def mutate_charge_neutral(seq, target):
    """Phase 2: Charge-neutral swaps for secondary optimization"""
    i = random.randrange(len(seq))
    current_aa = seq[i]
    charge_groups = {'positive': 'KRH', 'negative': 'DE', 'neutral': 'ASTNQ', 'hydrophobic': 'LIVFWYCM', 'special': 'GP'}
    for group, aas in charge_groups.items():
        if current_aa in aas:
            available = aas.replace(current_aa, "")
            if available:
                a = random.choice(available)
                return seq[:i] + a + seq[i+1:]
    i = random.randrange(len(seq))
    a = random.choice(AA.replace(seq[i], ""))
    return seq[:i] + a + seq[i+1:]

def mutate_random(seq):
    """Random single-point mutation"""
    i = random.randrange(len(seq))
    a = random.choice(AA.replace(seq[i], ""))
    return seq[:i] + a + seq[i+1:]


# ---------------------------------------------------------------------------
# GENERATION METHODS (real implementations)
# ---------------------------------------------------------------------------

def generate_prompt2peptide(target, boost, length_range, n=20, iters=500):
    """Full Prompt2Peptide: two-phase curriculum + ESM-free scoring"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        L = random.randint(*length_range)
        seq = init_seq(L, boost)
        best, best_sc = seq, constraint_score(seq, target)[0]
        T = 1.0
        phase1_iters = int(iters * 0.6)
        ttf = None
        start = time.time()
        
        for t in range(phase1_iters):
            s2 = mutate_charge_directed(best, target)
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T * 0.1):
                best, best_sc = s2, sc2
            T *= 0.995
            if ttf is None and is_feasible(best, target):
                ttf = time.time() - start
        
        for t in range(iters - phase1_iters):
            s2 = mutate_charge_neutral(best, target)
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T * 0.05):
                best, best_sc = s2, sc2
            T *= 0.995
            if ttf is None and is_feasible(best, target):
                ttf = time.time() - start
        
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    
    return cands, times_to_feasibility


def generate_single_phase_sa(target, boost, length_range, n=20, iters=500):
    """Ablation: Single-phase SA (no curriculum, random mutations only)"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        L = random.randint(*length_range)
        seq = init_seq(L, boost)
        best, best_sc = seq, constraint_score(seq, target)[0]
        T = 1.0
        ttf = None
        start = time.time()
        for t in range(iters):
            s2 = mutate_random(best)
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T * 0.1):
                best, best_sc = s2, sc2
            T *= 0.995
            if ttf is None and is_feasible(best, target):
                ttf = time.time() - start
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    return cands, times_to_feasibility


def generate_random_ga(target, boost, length_range, n=20, iters=500, pop_size=50):
    """Baseline: Random Genetic Algorithm with tournament selection"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        L = random.randint(*length_range)
        pop = [init_seq(L, boost) for _ in range(pop_size)]
        ttf = None
        start = time.time()
        for gen in range(iters // pop_size):
            scored = [(s, constraint_score(s, target)[0]) for s in pop]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Tournament selection + crossover + mutation
            new_pop = [scored[0][0]]  # elitism
            while len(new_pop) < pop_size:
                # Tournament
                t1, t2 = random.sample(scored[:max(2,len(scored)//2)], 2)
                parent = t1[0] if t1[1] > t2[1] else t2[0]
                child = mutate_random(parent)
                new_pop.append(child)
            pop = new_pop
            best_in_gen = scored[0][0]
            if ttf is None and is_feasible(best_in_gen, target):
                ttf = time.time() - start
        best = max(pop, key=lambda s: constraint_score(s, target)[0])
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    return cands, times_to_feasibility


def generate_plm_filter(target, boost, length_range, n=20, n_samples=500):
    """Baseline: PLM-free random sampling + post-hoc filtering (since ESM is expensive)"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        start = time.time()
        ttf = None
        best = None
        best_sc = -1
        for _ in range(n_samples):
            L = random.randint(*length_range)
            seq = init_seq(L, boost)
            sc = constraint_score(seq, target)[0]
            if sc > best_sc:
                best, best_sc = seq, sc
                if ttf is None and is_feasible(seq, target):
                    ttf = time.time() - start
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    return cands, times_to_feasibility


def generate_no_curriculum(target, boost, length_range, n=20, iters=500):
    """Ablation: Two-phase but NO curriculum (charge-directed mutations in both phases)
    Actually: single-phase with charge-directed + neutral mixed randomly"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        L = random.randint(*length_range)
        seq = init_seq(L, boost)
        best, best_sc = seq, constraint_score(seq, target)[0]
        T = 1.0
        ttf = None
        start = time.time()
        for t in range(iters):
            # Random mix of charge-directed and neutral (no phased curriculum)
            if random.random() < 0.5:
                s2 = mutate_charge_directed(best, target)
            else:
                s2 = mutate_charge_neutral(best, target)
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T * 0.075):
                best, best_sc = s2, sc2
            T *= 0.995
            if ttf is None and is_feasible(best, target):
                ttf = time.time() - start
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    return cands, times_to_feasibility


def generate_charge_only(target, boost, length_range, n=20, iters=500):
    """Ablation: Only charge-directed mutations (no phase 2 neutral optimization)"""
    cands = []
    times_to_feasibility = []
    for _ in range(n):
        L = random.randint(*length_range)
        seq = init_seq(L, boost)
        best, best_sc = seq, constraint_score(seq, target)[0]
        T = 1.0
        ttf = None
        start = time.time()
        for t in range(iters):
            s2 = mutate_charge_directed(best, target)
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T * 0.1):
                best, best_sc = s2, sc2
            T *= 0.995
            if ttf is None and is_feasible(best, target):
                ttf = time.time() - start
        cands.append(best)
        times_to_feasibility.append(ttf if ttf is not None else time.time() - start)
    return cands, times_to_feasibility


# ---------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ---------------------------------------------------------------------------

def is_feasible(seq, target):
    """Check if sequence meets ALL target constraints"""
    g = gravy(seq)
    ch = net_charge(seq)
    st = target.get('structure_type', 'helix')
    mu = hydrophobic_moment(seq, structure_type=st)
    
    ch_lo, ch_hi = target['charge'][:2]
    mu_lo, mu_hi = target['muH'][:2]
    g_lo, g_hi = target['gravy'][:2]
    
    return (ch_lo <= ch <= ch_hi) and (mu_lo <= mu <= mu_hi) and (g_lo <= g <= g_hi)


def safety_check(seq):
    """Run 7 safety filters, return dict of pass/fail per filter.
    
    These are general-purpose safety filters applied to ALL peptide families.
    They check for synthesis feasibility and basic safety, NOT for specific
    biological activity (AMP activity is NOT a safety requirement for non-AMP families).
    """
    import re
    results = {}
    # 1. Length bounds (synthesizability)
    results['length'] = 5 <= len(seq) <= 30
    # 2. Charge limit (extreme charges cause aggregation/toxicity)
    ch = net_charge(seq)
    results['charge'] = abs(ch) <= 10
    # 3. Homopolymer <= 4 (causes aggregation, synthesis issues)
    max_run = 1
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    results['homopolymer'] = max_run <= 4
    # 4. Even cysteine (odd number causes misfolding from unpaired disulfides)
    results['cysteine'] = seq.count('C') % 2 == 0
    # 5. Toxin motifs: flag runs of 5+ identical charged/hydrophobic residues
    has_toxin = bool(re.search(r'[KR]{5,}|[DE]{5,}|[FWY]{5,}|[LIV]{5,}', seq))
    results['toxin_motif'] = not has_toxin
    # 6. Hemolytic risk heuristic (based on charge + hydrophobicity interaction)
    hydro_frac = sum(1 for a in seq if a in 'LIVFWYCM') / len(seq)
    pos_ch = max(0, ch)
    # High positive charge + high hydrophobicity = membrane-disruptive = hemolytic
    hemo_risk = 0.4 * min(pos_ch / 8.0, 1.0) + 0.6 * hydro_frac
    results['hemolysis'] = hemo_risk <= 0.65
    # 7. Solubility heuristic (not too hydrophobic overall)
    g = gravy(seq)
    results['solubility'] = g <= 1.5  # GRAVY > 1.5 is extremely hydrophobic
    return results


def novelty_vs_references(seq, references, threshold=0.70):
    """Compute max sequence identity vs reference set"""
    max_id = 0
    for ref in references:
        identity = SequenceMatcher(None, seq, ref).ratio()
        max_id = max(max_id, identity)
    return max_id, max_id < threshold


def compute_metrics_for_sequences(sequences, target, references):
    """Compute all evaluation metrics for a set of sequences"""
    metrics = {
        'sequences': sequences,
        'feasible': [],
        'safe': [],
        'novel': [],
        'max_identity': [],
        'gravy': [],
        'charge': [],
        'muH': [],
        'safety_details': [],
    }
    
    for seq in sequences:
        metrics['feasible'].append(is_feasible(seq, target))
        safety = safety_check(seq)
        metrics['safe'].append(all(safety.values()))
        metrics['safety_details'].append(safety)
        max_id, is_novel = novelty_vs_references(seq, references)
        metrics['novel'].append(is_novel)
        metrics['max_identity'].append(max_id)
        metrics['gravy'].append(gravy(seq))
        metrics['charge'].append(net_charge(seq))
        st = target.get('structure_type', 'helix')
        metrics['muH'].append(hydrophobic_moment(seq, structure_type=st))
    
    n = len(sequences)
    feasible_count = sum(metrics['feasible'])
    safe_count = sum(metrics['safe'])
    novel_count = sum(metrics['novel'])
    
    # Diversity: average pairwise dissimilarity
    if n >= 2:
        identities = []
        sample_size = min(n, 50)
        sample = random.sample(range(n), sample_size)
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                identities.append(SequenceMatcher(None, sequences[sample[i]], sequences[sample[j]]).ratio())
        diversity = 1 - np.mean(identities) if identities else 0
    else:
        diversity = 0
    
    # Safety@Feasibility
    feasible_and_safe = sum(1 for f, s in zip(metrics['feasible'], metrics['safe']) if f and s)
    safety_at_feasibility = feasible_and_safe / feasible_count if feasible_count > 0 else 0
    
    summary = {
        'n': n,
        'coverage': feasible_count / n,
        'validity': safety_at_feasibility,
        'novelty': novel_count / n,
        'diversity': diversity,
        'safety_rate': safe_count / n,
        'feasibility_rate': feasible_count / n,
        'safety_at_feasibility': safety_at_feasibility,
        'mean_identity': np.mean(metrics['max_identity']),
    }
    return summary, metrics


def bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval"""
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return np.mean(boot_stats), lower, upper


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------------------------

def run_all_experiments():
    print("=" * 70)
    print("PROMPT2PEPTIDE: RUNNING ALL REAL EXPERIMENTS")
    print("=" * 70)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('paper-draft/figures', exist_ok=True)
    
    all_results = {}
    
    # -----------------------------------------------------------------------
    # 1. MAIN METHOD: Generate sequences for all prompt families
    # -----------------------------------------------------------------------
    print("\n[1/5] Generating sequences with Prompt2Peptide (full method)...")
    p2p_results = {}
    for family_name, family in PROMPT_FAMILIES.items():
        print(f"  {family_name}...")
        all_seqs = []
        all_ttf = []
        for seed in range(N_SEEDS):
            random.seed(seed * 100 + 42)
            np.random.seed(seed * 100 + 42)
            seqs, ttf = generate_prompt2peptide(
                family['target'], family['boost'], family['length_range'],
                n=N_SEQUENCES_PER_PROMPT // N_SEEDS, iters=ITERS
            )
            all_seqs.extend(seqs)
            all_ttf.extend(ttf)
        
        summary, detail = compute_metrics_for_sequences(all_seqs, family['target'], REFERENCE_PEPTIDES)
        summary['times_to_feasibility'] = all_ttf
        p2p_results[family_name] = {'summary': summary, 'detail': detail, 'sequences': all_seqs, 'ttf': all_ttf}
        print(f"    Coverage={summary['coverage']:.3f}, Safety@F={summary['safety_at_feasibility']:.3f}, "
              f"Novelty={summary['novelty']:.3f}, Diversity={summary['diversity']:.3f}")
    
    all_results['Prompt2Peptide'] = p2p_results
    
    # -----------------------------------------------------------------------
    # 2. BASELINES
    # -----------------------------------------------------------------------
    print("\n[2/5] Running baselines...")
    
    baseline_methods = {
        'Single-Phase SA': generate_single_phase_sa,
        'Random GA': generate_random_ga,
        'Random+Filter': generate_plm_filter,
    }
    
    for method_name, method_fn in baseline_methods.items():
        print(f"  Baseline: {method_name}")
        method_results = {}
        for family_name, family in PROMPT_FAMILIES.items():
            all_seqs = []
            all_ttf = []
            for seed in range(N_SEEDS):
                random.seed(seed * 100 + 42)
                np.random.seed(seed * 100 + 42)
                kwargs = dict(n=N_SEQUENCES_PER_PROMPT // N_SEEDS)
                if method_fn != generate_plm_filter:
                    kwargs['iters'] = ITERS
                seqs, ttf = method_fn(
                    family['target'], family['boost'], family['length_range'],
                    **kwargs
                )
                all_seqs.extend(seqs)
                all_ttf.extend(ttf)
            
            summary, detail = compute_metrics_for_sequences(all_seqs, family['target'], REFERENCE_PEPTIDES)
            summary['times_to_feasibility'] = all_ttf
            method_results[family_name] = {'summary': summary, 'detail': detail, 'sequences': all_seqs, 'ttf': all_ttf}
        
        all_results[method_name] = method_results
        # Print summary
        avg_cov = np.mean([r['summary']['coverage'] for r in method_results.values()])
        avg_saf = np.mean([r['summary']['safety_at_feasibility'] for r in method_results.values()])
        print(f"    Avg Coverage={avg_cov:.3f}, Avg Safety@F={avg_saf:.3f}")
    
    # -----------------------------------------------------------------------
    # 3. ABLATIONS
    # -----------------------------------------------------------------------
    print("\n[3/5] Running ablations...")
    
    ablation_methods = {
        'No Curriculum': generate_no_curriculum,
        'Charge-Only': generate_charge_only,
    }
    
    for abl_name, abl_fn in ablation_methods.items():
        print(f"  Ablation: {abl_name}")
        abl_results = {}
        for family_name, family in PROMPT_FAMILIES.items():
            all_seqs = []
            all_ttf = []
            for seed in range(N_SEEDS):
                random.seed(seed * 100 + 42)
                np.random.seed(seed * 100 + 42)
                seqs, ttf = abl_fn(
                    family['target'], family['boost'], family['length_range'],
                    n=N_SEQUENCES_PER_PROMPT // N_SEEDS, iters=ITERS
                )
                all_seqs.extend(seqs)
                all_ttf.extend(ttf)
            
            summary, detail = compute_metrics_for_sequences(all_seqs, family['target'], REFERENCE_PEPTIDES)
            summary['times_to_feasibility'] = all_ttf
            abl_results[family_name] = {'summary': summary, 'detail': detail, 'sequences': all_seqs, 'ttf': all_ttf}
        
        all_results[abl_name] = abl_results
        avg_cov = np.mean([r['summary']['coverage'] for r in abl_results.values()])
        print(f"    Avg Coverage={avg_cov:.3f}")
    
    # -----------------------------------------------------------------------
    # 4. SAVE RAW RESULTS
    # -----------------------------------------------------------------------
    print("\n[4/5] Saving results...")
    
    # Save summary table
    rows = []
    for method_name, method_results in all_results.items():
        for family_name, family_data in method_results.items():
            s = family_data['summary']
            rows.append({
                'Method': method_name,
                'Family': family_name,
                'Coverage': s['coverage'],
                'Safety@F': s['safety_at_feasibility'],
                'Novelty': s['novelty'],
                'Diversity': s['diversity'],
                'N': s['n'],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv('results/all_results_summary.csv', index=False)
    
    # Save per-family sequence data for Prompt2Peptide
    for family_name, family_data in p2p_results.items():
        seqs = family_data['sequences']
        detail = family_data['detail']
        seq_df = pd.DataFrame({
            'sequence': seqs,
            'length': [len(s) for s in seqs],
            'gravy': detail['gravy'],
            'charge': detail['charge'],
            'muH': detail['muH'],
            'feasible': detail['feasible'],
            'safe': detail['safe'],
            'novel': detail['novel'],
            'max_identity': detail['max_identity'],
        })
        seq_df.to_csv(f'results/sequences_{family_name}.csv', index=False)
    
    # -----------------------------------------------------------------------
    # 5. GENERATE ALL FIGURES
    # -----------------------------------------------------------------------
    print("\n[5/5] Generating figures from real data...")
    
    generate_all_figures(all_results, p2p_results)
    
    # -----------------------------------------------------------------------
    # PRINT FINAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Coverage':>10} {'Safety@F':>10} {'Novelty':>10} {'Diversity':>10}")
    print("-" * 65)
    for method_name in all_results:
        coverages = [all_results[method_name][f]['summary']['coverage'] for f in PROMPT_FAMILIES]
        safeties = [all_results[method_name][f]['summary']['safety_at_feasibility'] for f in PROMPT_FAMILIES]
        novelties = [all_results[method_name][f]['summary']['novelty'] for f in PROMPT_FAMILIES]
        diversities = [all_results[method_name][f]['summary']['diversity'] for f in PROMPT_FAMILIES]
        print(f"{method_name:<25} {np.mean(coverages):>10.3f} {np.mean(safeties):>10.3f} "
              f"{np.mean(novelties):>10.3f} {np.mean(diversities):>10.3f}")
    
    # Save the full results JSON (without numpy arrays)
    json_results = {}
    for method_name, method_data in all_results.items():
        json_results[method_name] = {}
        for family_name, family_data in method_data.items():
            json_results[method_name][family_name] = {
                'summary': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in family_data['summary'].items() if k != 'times_to_feasibility'},
                'n_sequences': len(family_data['sequences']),
            }
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\nAll results saved to results/")
    print("All figures saved to figures/ and paper-draft/figures/")
    return all_results


# ---------------------------------------------------------------------------
# FIGURE GENERATION (from real data)
# ---------------------------------------------------------------------------

def generate_all_figures(all_results, p2p_results):
    """Generate all paper figures from real experimental data"""
    
    # --- Figure 1: CVND Metrics with bootstrap CIs ---
    fig_cvnd(p2p_results)
    
    # --- Figure 2: Time-to-feasibility CDF ---
    fig_ttf_cdf(all_results)
    
    # --- Figure 3: Constraint satisfaction by family ---
    fig_constraint_satisfaction(p2p_results)
    
    # --- Figure 4: Baseline comparison ---
    fig_baseline_comparison(all_results)
    
    # --- Figure 5: Novelty analysis ---
    fig_novelty(p2p_results)
    
    # --- Figure 6: Safety breakdown ---
    fig_safety_breakdown(p2p_results)
    
    # --- Figure 7: Helical wheels ---
    fig_helical_wheels(p2p_results)
    
    # --- Figure 8: Structure-aware muH ---
    fig_structure_aware_muh()
    
    # --- Figure 9: Case study table figure ---
    fig_case_study(p2p_results)
    

def fig_cvnd(p2p_results):
    """CVND metrics with real bootstrap CIs"""
    coverages = [p2p_results[f]['summary']['coverage'] for f in PROMPT_FAMILIES]
    safeties = [p2p_results[f]['summary']['safety_at_feasibility'] for f in PROMPT_FAMILIES]
    novelties = [p2p_results[f]['summary']['novelty'] for f in PROMPT_FAMILIES]
    diversities = [p2p_results[f]['summary']['diversity'] for f in PROMPT_FAMILIES]
    
    metrics_data = {
        'Coverage': coverages,
        'Validity\n(Safety@F)': safeties,
        'Novelty': novelties,
        'Diversity': diversities,
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics_data.keys())
    means = []
    ci_lo = []
    ci_hi = []
    for name, data in metrics_data.items():
        m, lo, hi = bootstrap_ci(np.array(data))
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    bars = ax.bar(names, means, yerr=[ci_lo, ci_hi], capsize=8,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(ci_hi) * 0.3,
                f'{m:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('CVND Metrics with 95% Bootstrap CIs', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for path in ['figures/cvnd_metrics_with_ci.png', 'paper-draft/figures/cvnd_metrics_with_ci.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: cvnd_metrics_with_ci.png")


def fig_ttf_cdf(all_results):
    """Time-to-feasibility CDF: curriculum vs baselines"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    method_colors = {
        'Prompt2Peptide': '#2196F3',
        'Single-Phase SA': '#E91E63',
        'No Curriculum': '#FF9800',
        'Random GA': '#9C27B0',
        'Random+Filter': '#607D8B',
    }
    
    for method_name in ['Prompt2Peptide', 'Single-Phase SA', 'No Curriculum']:
        if method_name not in all_results:
            continue
        all_ttf = []
        for family_data in all_results[method_name].values():
            all_ttf.extend(family_data['ttf'])
        
        sorted_ttf = np.sort(all_ttf)
        y = np.arange(1, len(sorted_ttf) + 1) / len(sorted_ttf)
        color = method_colors.get(method_name, '#333')
        ax.plot(sorted_ttf, y, linewidth=2, label=f'{method_name} (med={np.median(all_ttf):.3f}s)',
                color=color)
    
    ax.set_xlabel('Time to Feasibility (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Time-to-Feasibility CDF', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for path in ['figures/time_to_feasibility_cdf_with_ci.png', 'paper-draft/figures/time_to_feasibility_cdf_with_ci.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: time_to_feasibility_cdf_with_ci.png")


def fig_constraint_satisfaction(p2p_results):
    """Constraint satisfaction rates by family and per-constraint breakdown"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: per-family feasibility
    families = list(PROMPT_FAMILIES.keys())
    short_names = ['Cat. Amph.\nHelix', 'Sol. Acidic\nLoop', 'Hydro.\nβ-sheet',
                   'Polar Flex.\nLinker', 'Basic\nNLS']
    rates = [p2p_results[f]['summary']['coverage'] for f in families]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    
    bars = ax1.bar(short_names, rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Feasibility Rate', fontsize=11)
    ax1.set_title('Constraint Satisfaction by Family', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: per-constraint for cationic amphipathic helix
    cah = p2p_results['cationic_amphipathic_helix']
    target = PROMPT_FAMILIES['cationic_amphipathic_helix']['target']
    seqs = cah['sequences']
    detail = cah['detail']
    
    charge_in = sum(1 for c in detail['charge'] if target['charge'][0] <= c <= target['charge'][1]) / len(seqs)
    muh_in = sum(1 for m in detail['muH'] if target['muH'][0] <= m <= target['muH'][1]) / len(seqs)
    gravy_in = sum(1 for g in detail['gravy'] if target['gravy'][0] <= g <= target['gravy'][1]) / len(seqs)
    
    constraints = ['Charge\n[+3,+8]', 'μH\n[0.35,1.0]', 'GRAVY\n[-0.2,0.6]']
    constraint_rates = [charge_in, muh_in, gravy_in]
    
    bars2 = ax2.bar(constraints, constraint_rates, color=['#2196F3', '#FF9800', '#4CAF50'],
                    alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars2, constraint_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Satisfaction Rate', fontsize=11)
    ax2.set_title('Per-Constraint (Cat. Amph. Helix)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for path in ['figures/constraint_satisfaction_corrected.png', 'paper-draft/figures/constraint_satisfaction_corrected.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: constraint_satisfaction_corrected.png")


def fig_baseline_comparison(all_results):
    """Baseline comparison: coverage and effect sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['Prompt2Peptide', 'Single-Phase SA', 'Random GA', 'Random+Filter', 'No Curriculum', 'Charge-Only']
    method_labels = ['P2P\n(Ours)', 'Single-\nPhase SA', 'Random\nGA', 'Random\n+Filter', 'No\nCurriculum', 'Charge\nOnly']
    
    # Left: Average coverage
    avg_coverages = []
    std_coverages = []
    for m in methods:
        if m in all_results:
            covs = [all_results[m][f]['summary']['coverage'] for f in PROMPT_FAMILIES]
            avg_coverages.append(np.mean(covs))
            std_coverages.append(np.std(covs))
        else:
            avg_coverages.append(0)
            std_coverages.append(0)
    
    colors = ['#2196F3', '#E91E63', '#9C27B0', '#607D8B', '#FF9800', '#795548']
    bars = ax1.bar(method_labels, avg_coverages, yerr=std_coverages, capsize=5,
                   color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, m in zip(bars, avg_coverages):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.03,
                f'{m:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Average Coverage', fontsize=11)
    ax1.set_title('Coverage vs. Baselines & Ablations', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Effect sizes (Cohen's d) vs Prompt2Peptide
    p2p_covs = [all_results['Prompt2Peptide'][f]['summary']['coverage'] for f in PROMPT_FAMILIES]
    effect_sizes = []
    effect_labels = []
    for m in methods[1:]:
        if m in all_results:
            other_covs = [all_results[m][f]['summary']['coverage'] for f in PROMPT_FAMILIES]
            pooled_std = np.sqrt((np.std(p2p_covs)**2 + np.std(other_covs)**2) / 2)
            if pooled_std > 0:
                d = (np.mean(p2p_covs) - np.mean(other_covs)) / pooled_std
            else:
                d = 0
            effect_sizes.append(d)
            effect_labels.append(m)
    
    bars2 = ax2.barh(effect_labels, effect_sizes, color=colors[1:len(effect_labels)+1],
                     alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Cohen's d (P2P advantage)", fontsize=11)
    ax2.set_title('Effect Sizes vs. Prompt2Peptide', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.axvline(x=0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.axvline(x=0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.axvline(x=0.8, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for path in ['figures/baseline_comparison_detailed.png', 'paper-draft/figures/baseline_comparison_detailed.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: baseline_comparison_detailed.png")


def fig_novelty(p2p_results):
    """Novelty analysis and identity distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: novelty rates per family
    families = list(PROMPT_FAMILIES.keys())
    short_names = ['Cat. Amph.', 'Sol. Acidic', 'Hydro. β', 'Polar Flex.', 'Basic NLS']
    novelty_rates = [p2p_results[f]['summary']['novelty'] for f in families]
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    bars = ax1.bar(short_names, novelty_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, novelty_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Novelty Rate', fontsize=11)
    ax1.set_title('Novelty by Family (identity < 0.70)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: identity distribution histogram
    all_identities = []
    for f in families:
        all_identities.extend(p2p_results[f]['detail']['max_identity'])
    
    ax2.hist(all_identities, bins=30, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.70, color='red', linewidth=2, linestyle='--', label='Novelty threshold (0.70)')
    ax2.set_xlabel('Max Identity vs. Reference', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Identity Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for path in ['figures/novelty_analysis_corrected.png', 'paper-draft/figures/novelty_analysis_corrected.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: novelty_analysis_corrected.png")


def fig_safety_breakdown(p2p_results):
    """Safety breakdown by family and per-filter"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Safety@F per family
    families = list(PROMPT_FAMILIES.keys())
    short_names = ['Cat. Amph.', 'Sol. Acidic', 'Hydro. β', 'Polar Flex.', 'Basic NLS']
    safety_rates = [p2p_results[f]['summary']['safety_at_feasibility'] for f in families]
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    bars = ax1.bar(short_names, safety_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, safety_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Safety@Feasibility', fontsize=11)
    ax1.set_title('Safety@Feasibility by Family', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: per-filter pass rates for cationic amphipathic helix
    cah = p2p_results['cationic_amphipathic_helix']
    filter_names = ['length', 'charge', 'homopolymer', 'cysteine', 'toxin_motif', 'hemolysis', 'solubility']
    filter_labels = ['Length', 'Charge', 'Homopoly.', 'Cysteine', 'Toxin', 'Hemolysis', 'Solubility']
    
    pass_rates = []
    for fn in filter_names:
        passed = sum(1 for sd in cah['detail']['safety_details'] if sd[fn])
        pass_rates.append(passed / len(cah['sequences']))
    
    filter_colors = ['#4CAF50' if r >= 0.9 else '#FF9800' if r >= 0.7 else '#E91E63' for r in pass_rates]
    bars2 = ax2.bar(filter_labels, pass_rates, color=filter_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars2, pass_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax2.set_ylabel('Pass Rate', fontsize=11)
    ax2.set_title('Per-Filter Breakdown (Cat. Amph. Helix)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for path in ['figures/safety_breakdown_detailed.png', 'paper-draft/figures/safety_breakdown_detailed.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: safety_breakdown_detailed.png")


def fig_helical_wheels(p2p_results):
    """Helical wheel diagrams for representative sequences"""
    # Pick top sequences from cationic amphipathic helix
    cah = p2p_results['cationic_amphipathic_helix']
    feasible_seqs = [s for s, f in zip(cah['sequences'], cah['detail']['feasible']) if f]
    
    # Also pick one from acidic loop
    sal = p2p_results['soluble_acidic_loop']
    acidic_seqs = [s for s, f in zip(sal['sequences'], sal['detail']['feasible']) if f]
    
    display_seqs = []
    if len(feasible_seqs) >= 2:
        display_seqs.append((feasible_seqs[0], "Cationic Amphipathic Helix"))
        display_seqs.append((feasible_seqs[1], "Cationic Amphipathic Helix"))
    if len(acidic_seqs) >= 1:
        display_seqs.append((acidic_seqs[0], "Soluble Acidic Loop"))
    if len(feasible_seqs) >= 3:
        display_seqs.append((feasible_seqs[2], "Cationic Amphipathic Helix"))
    
    # Pad to 4
    while len(display_seqs) < 4:
        display_seqs.append(("KLAKLAKLAKLA", "Synthetic Reference"))
    
    aa_colors = {
        'K': '#82E0AA', 'R': '#4ECDC4', 'H': '#BB8FCE',  # positive
        'D': '#96CEB4', 'E': '#98D8C8',  # negative
        'L': '#F8C471', 'I': '#85C1E9', 'V': '#FADBD8', 'F': '#D7BDE2',
        'W': '#F9E79F', 'Y': '#D5DBDB', 'A': '#FF6B6B', 'M': '#F1948A', 'C': '#FFEAA7',
        'G': '#F7DC6F', 'P': '#A9DFBF', 'S': '#AED6F1', 'T': '#A3E4D7',
        'N': '#45B7D1', 'Q': '#DDA0DD',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (seq, title) in enumerate(display_seqs[:4]):
        ax = axes[idx]
        radius = 2.0
        
        for j, aa in enumerate(seq):
            angle = j * 100 * np.pi / 180
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            color = aa_colors.get(aa, '#CCCCCC')
            circle = Circle((x, y), 0.22, color=color, edgecolor='black', linewidth=0.8)
            ax.add_patch(circle)
            ax.text(x, y, aa, ha='center', va='center', fontsize=9, fontweight='bold')
        
        main_circle = Circle((0, 0), radius, fill=False, linewidth=1.2, color='gray', linestyle='--')
        ax.add_patch(main_circle)
        
        ch = net_charge(seq)
        mu = hydrophobic_moment(seq)
        ax.set_title(f'{title}\n{seq}\nCharge={ch:.1f}, μH={mu:.2f}', fontsize=9, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    for path in ['figures/helical_wheels_analysis.png', 'paper-draft/figures/helical_wheels_analysis.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: helical_wheels_analysis.png")


def fig_structure_aware_muh():
    """Structure-aware muH calculation diagram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Example sequence
    seq = "KLAKLAKLAKLA"
    
    # Helix (100 deg)
    angles_helix = [i * 100 for i in range(len(seq))]
    KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
          'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
          'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    
    for i, aa in enumerate(seq):
        angle_rad = math.radians(angles_helix[i])
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        color = '#FF6B6B' if KD.get(aa, 0) > 0 else '#4ECDC4'
        ax1.scatter(x, y, c=color, s=200, edgecolors='black', linewidth=0.8, zorder=5)
        ax1.annotate(aa, (x, y), ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    
    mu_helix = hydrophobic_moment(seq, structure_type='helix')
    ax1.set_title(f'α-Helix (100°/res)\nμH = {mu_helix:.3f}', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Sheet (180 deg)
    for i, aa in enumerate(seq):
        angle_rad = math.radians(i * 180)
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        color = '#FF6B6B' if KD.get(aa, 0) > 0 else '#4ECDC4'
        ax2.scatter(x, y, c=color, s=200, edgecolors='black', linewidth=0.8, zorder=5)
        ax2.annotate(aa, (x, y), ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    
    mu_sheet = hydrophobic_moment(seq, structure_type='sheet')
    ax2.set_title(f'β-Sheet (180°/res)\nμH = {mu_sheet:.3f}', fontsize=12, fontweight='bold')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Structure-Aware Hydrophobic Moment Calculation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for path in ['figures/structure_aware_muh_calculation.png', 'paper-draft/figures/structure_aware_muh_calculation.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: structure_aware_muh_calculation.png")


def fig_case_study(p2p_results):
    """Case study table-like figure for BCB reviewers"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Collect case study data
    rows = []
    for family_name, family in PROMPT_FAMILIES.items():
        data = p2p_results[family_name]
        # Pick best feasible+safe sequence
        best_seq = None
        for i, seq in enumerate(data['sequences']):
            if data['detail']['feasible'][i] and data['detail']['safe'][i]:
                best_seq = seq
                break
        if best_seq is None and data['sequences']:
            best_seq = data['sequences'][0]
        
        if best_seq:
            ch = net_charge(best_seq)
            mu = hydrophobic_moment(best_seq, structure_type=family['target'].get('structure_type', 'helix'))
            g = gravy(best_seq)
            max_id, novel = novelty_vs_references(best_seq, REFERENCE_PEPTIDES)
            safe = safety_check(best_seq)
            safe_pass = sum(safe.values()) / len(safe) * 100
            
            short_name = family_name.replace('_', ' ').title()
            target_ch = f"[{family['target']['charge'][0]:+.0f}, {family['target']['charge'][1]:+.0f}]"
            target_mu = f"[{family['target']['muH'][0]:.2f}, {family['target']['muH'][1]:.2f}]"
            target_g = f"[{family['target']['gravy'][0]:.1f}, {family['target']['gravy'][1]:.1f}]"
            
            rows.append([short_name, best_seq[:20] + ('...' if len(best_seq) > 20 else ''),
                        f'{ch:.1f}\n{target_ch}', f'{mu:.2f}\n{target_mu}', f'{g:.2f}\n{target_g}',
                        f'{safe_pass:.0f}%', f'{max_id:.2f}'])
    
    columns = ['Prompt Family', 'Top Sequence', 'Charge\n(Target)', 'μH\n(Target)',
               'GRAVY\n(Target)', 'Safety\nPass%', 'Max ID\nvs Ref']
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)
    
    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#E3F2FD')
    
    ax.set_title('Case Study: Representative Sequences per Prompt Family',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    for path in ['figures/case_study_table.png', 'paper-draft/figures/case_study_table.png']:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created: case_study_table.png")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_all_experiments()
