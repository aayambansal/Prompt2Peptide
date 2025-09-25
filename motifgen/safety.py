# --- safety.py ---
import re
import numpy as np
import pandas as pd
from collections import Counter

def safety_screening(sequences):
    """Comprehensive safety screening for generated peptides"""
    
    results = []
    
    for seq in sequences:
        flags = []
        severity = "low"
        
        # Length bounds check
        if len(seq) < 5:
            flags.append("too_short")
            severity = "medium"
        elif len(seq) > 50:
            flags.append("too_long")
            severity = "medium"
        
        # Excessive positive charge
        from .metrics import net_charge
        charge = net_charge(seq)
        if charge > 10:
            flags.append("excessive_positive_charge")
            severity = "high"
        elif charge > 8:
            flags.append("high_positive_charge")
            severity = "medium"
        
        # Homopolymer runs (4+ consecutive identical residues)
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            pattern = aa * 4
            if pattern in seq:
                flags.append("homopolymer_run")
                severity = "medium"
        
        # Cysteine pairs (potential disulfide bonds)
        cys_count = seq.count('C')
        if cys_count >= 2:
            flags.append("cysteine_pairs")
            severity = "medium"
        
        # Known toxin-like motifs (simple regex patterns)
        toxin_patterns = [
            r'KKKKKKKK',  # Excessive positive charge
            r'RRRRRRRR',  # Excessive positive charge
            r'FFFFFFFF',  # Excessive hydrophobicity
            r'LLLLLLLL',  # Excessive hydrophobicity
            r'WWWWWWWW',  # Excessive hydrophobicity
            r'CCCCCCCC',  # Excessive cysteine
        ]
        
        for pattern in toxin_patterns:
            if re.search(pattern, seq):
                flags.append("toxin_like_motif")
                severity = "high"
                break
        
        # Check for known problematic sequences
        problematic_sequences = [
            "KKKKKKKKKKKK",  # Poly-lysine
            "RRRRRRRRRRRR",  # Poly-arginine
            "FFFFFFFFFFFF",  # Poly-phenylalanine
            "LLLLLLLLLLLL",  # Poly-leucine
            "CCCCCCCCCCCC",  # Poly-cysteine
        ]
        
        for prob_seq in problematic_sequences:
            if seq == prob_seq:
                flags.append("known_problematic")
                severity = "high"
                break
        
        # Calculate red-flag rate
        red_flag_rate = len(flags) / 10.0  # Normalize to 0-1 scale
        
        results.append({
            'sequence': seq,
            'length': len(seq),
            'flags': flags,
            'num_flags': len(flags),
            'severity': severity,
            'red_flag_rate': red_flag_rate,
            'is_safe': len(flags) == 0 and severity == "low"
        })
    
    return pd.DataFrame(results)

def safety_summary(safety_results):
    """Generate safety summary statistics"""
    
    total_sequences = len(safety_results)
    safe_sequences = safety_results['is_safe'].sum()
    unsafe_sequences = total_sequences - safe_sequences
    
    # Count by severity
    severity_counts = safety_results['severity'].value_counts()
    
    # Count by flag type
    all_flags = []
    for flags in safety_results['flags']:
        all_flags.extend(flags)
    
    flag_counts = Counter(all_flags)
    
    # Calculate overall red-flag rate
    overall_red_flag_rate = safety_results['red_flag_rate'].mean()
    
    summary = {
        'total_sequences': total_sequences,
        'safe_sequences': safe_sequences,
        'unsafe_sequences': unsafe_sequences,
        'safety_percentage': (safe_sequences / total_sequences) * 100,
        'overall_red_flag_rate': overall_red_flag_rate,
        'severity_distribution': severity_counts.to_dict(),
        'flag_distribution': dict(flag_counts),
        'sequences_with_flags': (safety_results['num_flags'] > 0).sum()
    }
    
    return summary

def comprehensive_safety_analysis(sequences_dict):
    """Analyze safety across all generated sequences"""
    
    print("🛡️ COMPREHENSIVE SAFETY SCREENING")
    print("="*50)
    
    all_results = {}
    overall_summary = {
        'total_sequences': 0,
        'total_safe': 0,
        'total_unsafe': 0,
        'overall_safety_rate': 0,
        'overall_red_flag_rate': 0
    }
    
    for prompt_type, sequences in sequences_dict.items():
        print(f"\n📋 Screening {prompt_type} ({len(sequences)} sequences)...")
        
        # Run safety screening
        safety_results = safety_screening(sequences)
        summary = safety_summary(safety_results)
        
        all_results[prompt_type] = {
            'results': safety_results,
            'summary': summary
        }
        
        # Print summary
        print(f"  Safe sequences: {summary['safe_sequences']}/{summary['total_sequences']} "
              f"({summary['safety_percentage']:.1f}%)")
        print(f"  Red-flag rate: {summary['overall_red_flag_rate']:.3f}")
        print(f"  Severity distribution: {summary['severity_distribution']}")
        
        # Update overall summary
        overall_summary['total_sequences'] += summary['total_sequences']
        overall_summary['total_safe'] += summary['safe_sequences']
        overall_summary['total_unsafe'] += summary['unsafe_sequences']
    
    # Calculate overall statistics
    overall_summary['overall_safety_rate'] = (
        overall_summary['total_safe'] / overall_summary['total_sequences'] * 100
        if overall_summary['total_sequences'] > 0 else 0
    )
    
    overall_summary['overall_red_flag_rate'] = np.mean([
        results['summary']['overall_red_flag_rate'] 
        for results in all_results.values()
    ])
    
    print(f"\n📊 OVERALL SAFETY SUMMARY:")
    print(f"  Total sequences: {overall_summary['total_sequences']}")
    print(f"  Safe sequences: {overall_summary['total_safe']} "
          f"({overall_summary['overall_safety_rate']:.1f}%)")
    print(f"  Overall red-flag rate: {overall_summary['overall_red_flag_rate']:.3f}")
    
    return all_results, overall_summary

def create_safety_report(sequences_dict):
    """Create a comprehensive safety report"""
    
    # Run safety analysis
    safety_results, overall_summary = comprehensive_safety_analysis(sequences_dict)
    
    # Create detailed report
    report = {
        'overall_summary': overall_summary,
        'detailed_results': safety_results,
        'recommendations': []
    }
    
    # Generate recommendations based on results
    if overall_summary['overall_safety_rate'] < 90:
        report['recommendations'].append(
            "Consider adding more stringent safety filters to improve safety rate"
        )
    
    if overall_summary['overall_red_flag_rate'] > 0.1:
        report['recommendations'].append(
            "High red-flag rate detected - review flagging criteria"
        )
    
    # Check for specific concerns
    for prompt_type, results in safety_results.items():
        summary = results['summary']
        
        if summary['safety_percentage'] < 80:
            report['recommendations'].append(
                f"Low safety rate for {prompt_type}: {summary['safety_percentage']:.1f}%"
            )
        
        if 'high' in summary['severity_distribution']:
            high_count = summary['severity_distribution'].get('high', 0)
            report['recommendations'].append(
                f"High severity flags detected in {prompt_type}: {high_count} sequences"
            )
    
    return report

def add_safety_columns_to_results(results_df, sequences):
    """Add safety columns to existing results DataFrame"""
    
    safety_results = safety_screening(sequences)
    
    # Add safety columns
    results_df['safety_flags'] = safety_results['flags'].tolist()
    results_df['num_safety_flags'] = safety_results['num_flags'].tolist()
    results_df['safety_severity'] = safety_results['severity'].tolist()
    results_df['red_flag_rate'] = safety_results['red_flag_rate'].tolist()
    results_df['is_safe'] = safety_results['is_safe'].tolist()
    
    return results_df
