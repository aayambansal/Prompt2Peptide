# --- robustness.py ---
import numpy as np
import pandas as pd
from .generate import generate
from .metrics import gravy, net_charge, hydrophobic_moment

def get_prompt_paraphrases():
    """Define paraphrases for each prompt type"""
    paraphrases = {
        "cationic_amphipathic": [
            "cationic amphipathic helix, length 12–18",
            "positively charged helical AMP",
            "K/R-rich helix with high μH",
            "basic amphipathic peptide helix",
            "cationic antimicrobial helix"
        ],
        "soluble_acidic": [
            "soluble acidic loop 10–14",
            "negatively charged soluble peptide",
            "D/E-rich flexible loop",
            "acidic hydrophilic sequence",
            "soluble anionic peptide"
        ],
        "hydrophobic_sheet": [
            "hydrophobic β-sheet, length 10–14",
            "membrane-spanning hydrophobic peptide",
            "F/W/Y-rich β-strand",
            "hydrophobic membrane peptide",
            "lipophilic β-sheet structure"
        ],
        "polar_flexible": [
            "polar flexible linker, length 8–12",
            "flexible polar peptide linker",
            "G/P-rich flexible sequence",
            "polar unstructured peptide",
            "flexible hydrophilic linker"
        ],
        "basic_nuclear": [
            "basic nuclear localization signal",
            "nuclear targeting peptide",
            "K/R-rich NLS sequence",
            "basic nuclear import signal",
            "positively charged NLS"
        ]
    }
    return paraphrases

def test_prompt_robustness(prompt_type, n=20, iters=300):
    """Test robustness across paraphrases of a prompt type"""
    paraphrases = get_prompt_paraphrases()
    
    if prompt_type not in paraphrases:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    results = {}
    
    for i, paraphrase in enumerate(paraphrases[prompt_type]):
        print(f"  Testing paraphrase {i+1}: '{paraphrase}'")
        
        # Generate sequences
        generated = generate(paraphrase, n=n, iters=iters)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'sequence': seq,
                'length': len(seq),
                'muH': met['muH'],
                'charge': met['charge'],
                'gravy': met['gravy'],
                'esm_ll': ll,
                'paraphrase': paraphrase
            }
            for seq, met, ll in generated
        ])
        
        results[paraphrase] = df
    
    return results

def analyze_robustness(results_dict, prompt_type):
    """Analyze robustness across paraphrases"""
    
    # Get target constraints based on prompt type
    if "cationic" in prompt_type:
        target_constraints = {
            'charge': (3, 8),
            'muH': (0.35, 1.0),
            'gravy': (-0.2, 0.6)
        }
    elif "acidic" in prompt_type:
        target_constraints = {
            'charge': (-3, 0),
            'muH': (0.1, 0.4),
            'gravy': (-1.0, 0.0)
        }
    elif "hydrophobic" in prompt_type:
        target_constraints = {
            'charge': (-1, 2),
            'muH': (0.1, 0.3),
            'gravy': (0.5, 1.5)
        }
    elif "polar" in prompt_type:
        target_constraints = {
            'charge': (-1, 1),
            'muH': (0.05, 0.25),
            'gravy': (-0.8, 0.2)
        }
    elif "basic" in prompt_type:
        target_constraints = {
            'charge': (4, 8),
            'muH': (0.2, 0.6),
            'gravy': (-0.5, 0.3)
        }
    else:
        target_constraints = {
            'charge': (0, 3),
            'muH': (0.15, 1.0),
            'gravy': (-0.5, 0.5)
        }
    
    def check_constraint_satisfaction(df, constraints):
        """Check constraint satisfaction for a DataFrame"""
        charge_ok = ((df['charge'] >= constraints['charge'][0]) & 
                     (df['charge'] <= constraints['charge'][1])).sum()
        muh_ok = ((df['muH'] >= constraints['muH'][0]) & 
                  (df['muH'] <= constraints['muH'][1])).sum()
        gravy_ok = ((df['gravy'] >= constraints['gravy'][0]) & 
                    (df['gravy'] <= constraints['gravy'][1])).sum()
        
        total = len(df)
        return {
            'charge_satisfaction': charge_ok / total * 100,
            'muh_satisfaction': muh_ok / total * 100,
            'gravy_satisfaction': gravy_ok / total * 100,
            'overall_satisfaction': (charge_ok + muh_ok + gravy_ok) / (3 * total) * 100
        }
    
    # Analyze each paraphrase
    paraphrase_analysis = {}
    for paraphrase, df in results_dict.items():
        satisfaction = check_constraint_satisfaction(df, target_constraints)
        
        paraphrase_analysis[paraphrase] = {
            'satisfaction': satisfaction,
            'mean_properties': {
                'muH': df['muH'].mean(),
                'charge': df['charge'].mean(),
                'gravy': df['gravy'].mean(),
                'esm_ll': df['esm_ll'].mean()
            },
            'std_properties': {
                'muH': df['muH'].std(),
                'charge': df['charge'].std(),
                'gravy': df['gravy'].std(),
                'esm_ll': df['esm_ll'].std()
            }
        }
    
    # Calculate robustness metrics
    satisfaction_rates = [analysis['satisfaction']['overall_satisfaction'] 
                         for analysis in paraphrase_analysis.values()]
    
    property_means = {}
    property_stds = {}
    for prop in ['muH', 'charge', 'gravy', 'esm_ll']:
        means = [analysis['mean_properties'][prop] for analysis in paraphrase_analysis.values()]
        stds = [analysis['std_properties'][prop] for analysis in paraphrase_analysis.values()]
        
        property_means[prop] = {
            'mean': np.mean(means),
            'std': np.std(means),
            'cv': np.std(means) / np.mean(means) if np.mean(means) != 0 else np.nan
        }
        
        property_stds[prop] = {
            'mean': np.mean(stds),
            'std': np.std(stds)
        }
    
    robustness_summary = {
        'prompt_type': prompt_type,
        'num_paraphrases': len(results_dict),
        'satisfaction_rates': satisfaction_rates,
        'mean_satisfaction': np.mean(satisfaction_rates),
        'std_satisfaction': np.std(satisfaction_rates),
        'min_satisfaction': np.min(satisfaction_rates),
        'robust': np.min(satisfaction_rates) >= 95.0,  # 95% threshold
        'property_consistency': property_means,
        'property_variability': property_stds,
        'detailed_analysis': paraphrase_analysis
    }
    
    return robustness_summary

def comprehensive_robustness_test():
    """Test robustness across all prompt types"""
    paraphrases = get_prompt_paraphrases()
    all_results = {}
    
    print("🛡️ COMPREHENSIVE PROMPT ROBUSTNESS TESTING")
    print("="*60)
    
    for prompt_type in paraphrases.keys():
        print(f"\n📋 Testing {prompt_type}...")
        
        # Test robustness
        results = test_prompt_robustness(prompt_type, n=15, iters=200)
        analysis = analyze_robustness(results, prompt_type)
        
        all_results[prompt_type] = {
            'results': results,
            'analysis': analysis
        }
        
        # Print summary
        print(f"  Mean satisfaction: {analysis['mean_satisfaction']:.1f}%")
        print(f"  Min satisfaction: {analysis['min_satisfaction']:.1f}%")
        print(f"  Robust: {'✅' if analysis['robust'] else '❌'}")
    
    return all_results
