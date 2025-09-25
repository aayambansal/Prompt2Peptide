# --- blast_novelty.py ---
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher
from .novelty import sequence_identity

# Simple BLAST-like scoring (without actual BLAST)
def simple_blast_score(query, subject, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    """Simple BLAST-like scoring without actual BLAST"""
    # Use sequence identity as a proxy for BLAST bitscore
    identity = sequence_identity(query, subject)
    
    # Convert identity to a bitscore-like metric
    # Higher identity = higher bitscore
    bitscore = identity * 100  # Scale to bitscore-like range
    
    return bitscore

def get_reference_databases():
    """Get reference peptide databases (simplified versions)"""
    
    # APD3-like antimicrobial peptides
    apd3_peptides = [
        "KKKKKKKKKKKK",  # Poly-lysine
        "RRRRRRRRRRRR",  # Poly-arginine
        "KLAKLAKLAKLA",  # Amphipathic
        "KLALKLALKLAL",  # Amphipathic variant
        "RGDGRGDGRGDG",  # RGD motif
        "DEEDDEEDDEED",  # Acidic
        "FFFFFFFFFFFF",  # Poly-phenylalanine
        "LLLLLLLLLLLL",  # Poly-leucine
        "GGGGGGGGGGGG",  # Poly-glycine
        "PPPPPPPPPPPP",  # Poly-proline
        "KLAKLAKLAKLAKLAKLA",  # Longer amphipathic
        "RRRRRRRRRRRRRRRR",  # Longer poly-arginine
        "KKKKKKKKKKKKKKKK",  # Longer poly-lysine
        "DEEDDEEDDEEDDEED",  # Longer acidic
        "FFFFFFFFFFFFFFFF",  # Longer hydrophobic
    ]
    
    # DBAASP-like peptides with activity
    dbaasp_peptides = [
        "KLAKLAKLAKLA",  # AMP
        "KLALKLALKLAL",  # AMP variant
        "RGDGRGDGRGDG",  # Cell penetrating
        "KKKKKKKKKKKK",  # CPP
        "RRRRRRRRRRRR",  # CPP
        "DEEDDEEDDEED",  # Acidic
        "FFFFFFFFFFFF",  # Hydrophobic
        "LLLLLLLLLLLL",  # Hydrophobic
        "GGGGGGGGGGGG",  # Flexible
        "PPPPPPPPPPPP",  # Flexible
    ]
    
    # Common protein motifs (PROSITE-style)
    common_motifs = [
        "RGD",  # Integrin binding
        "KKKK",  # Nuclear localization
        "RRRR",  # Nuclear localization
        "DEED",  # Acidic
        "KLAK",  # Amphipathic
        "KLAL",  # Amphipathic
        "FFFF",  # Hydrophobic
        "LLLL",  # Hydrophobic
        "GGGG",  # Flexible
        "PPPP",  # Flexible
    ]
    
    return {
        'apd3': apd3_peptides,
        'dbaasp': dbaasp_peptides,
        'motifs': common_motifs
    }

def check_motif_overlap(sequence, motifs):
    """Check for overlap with known motifs using regex patterns"""
    overlaps = []
    
    for motif in motifs:
        # Check if motif appears in sequence
        if motif in sequence:
            overlaps.append(motif)
        
        # Check reverse complement
        if motif[::-1] in sequence:
            overlaps.append(f"{motif}_rev")
    
    return overlaps

def blast_novelty_analysis(sequences, database_name="combined"):
    """Comprehensive BLAST-style novelty analysis"""
    
    # Get reference databases
    databases = get_reference_databases()
    
    if database_name == "combined":
        reference_sequences = []
        for db_name, db_seqs in databases.items():
            reference_sequences.extend(db_seqs)
    else:
        reference_sequences = databases.get(database_name, [])
        if not reference_sequences:  # Fallback to combined if not found
            reference_sequences = []
            for db_name, db_seqs in databases.items():
                reference_sequences.extend(db_seqs)
    
    results = []
    
    for seq in sequences:
        # Find best match in reference database
        best_bitscore = 0
        best_match = ""
        best_identity = 0
        
        for ref_seq in reference_sequences:
            bitscore = simple_blast_score(seq, ref_seq)
            identity = sequence_identity(seq, ref_seq)
            
            if bitscore > best_bitscore:
                best_bitscore = bitscore
                best_match = ref_seq
                best_identity = identity
        
        # Check motif overlaps
        motif_overlaps = check_motif_overlap(seq, databases['motifs'])
        
        results.append({
            'sequence': seq,
            'best_bitscore': best_bitscore,
            'best_identity': best_identity,
            'best_match': best_match,
            'motif_overlaps': motif_overlaps,
            'num_motif_overlaps': len(motif_overlaps),
            'is_novel': best_identity < 0.7  # 70% identity threshold
        })
    
    return pd.DataFrame(results)

def comprehensive_novelty_analysis(sequences_dict):
    """Analyze novelty across all generated sequences"""
    
    print("🔍 COMPREHENSIVE BLAST-STYLE NOVELTY ANALYSIS")
    print("="*60)
    
    all_results = {}
    
    for prompt_type, sequences in sequences_dict.items():
        print(f"\n📋 Analyzing {prompt_type} ({len(sequences)} sequences)...")
        
        # Analyze against different databases
        apd3_results = blast_novelty_analysis(sequences, "apd3")
        dbaasp_results = blast_novelty_analysis(sequences, "dbaasp")
        combined_results = blast_novelty_analysis(sequences, "combined")
        
        # Calculate summary statistics
        def calculate_stats(df):
            return {
                'mean_bitscore': df['best_bitscore'].mean(),
                'median_bitscore': df['best_bitscore'].median(),
                'min_bitscore': df['best_bitscore'].min(),
                'max_bitscore': df['best_bitscore'].max(),
                'mean_identity': df['best_identity'].mean(),
                'median_identity': df['best_identity'].median(),
                'novel_percentage': (df['is_novel'].sum() / len(df)) * 100,
                'mean_motif_overlaps': df['num_motif_overlaps'].mean(),
                'sequences_with_motifs': (df['num_motif_overlaps'] > 0).sum()
            }
        
        apd3_stats = calculate_stats(apd3_results)
        dbaasp_stats = calculate_stats(dbaasp_results)
        combined_stats = calculate_stats(combined_results)
        
        print(f"  APD3: {apd3_stats['novel_percentage']:.1f}% novel, "
              f"median bitscore: {apd3_stats['median_bitscore']:.1f}")
        print(f"  DBAASP: {dbaasp_stats['novel_percentage']:.1f}% novel, "
              f"median bitscore: {dbaasp_stats['median_bitscore']:.1f}")
        print(f"  Combined: {combined_stats['novel_percentage']:.1f}% novel, "
              f"median bitscore: {combined_stats['median_bitscore']:.1f}")
        
        all_results[prompt_type] = {
            'apd3': {'results': apd3_results, 'stats': apd3_stats},
            'dbaasp': {'results': dbaasp_results, 'stats': dbaasp_stats},
            'combined': {'results': combined_results, 'stats': combined_stats}
        }
    
    return all_results

def create_novelty_summary_table(analysis_results):
    """Create a summary table of novelty results"""
    
    summary_data = []
    
    for prompt_type, results in analysis_results.items():
        combined_stats = results['combined']['stats']
        
        summary_data.append({
            'Prompt Type': prompt_type.replace('_', ' ').title(),
            'Novel %': f"{combined_stats['novel_percentage']:.1f}%",
            'Median Bitscore': f"{combined_stats['median_bitscore']:.1f}",
            'Min Bitscore': f"{combined_stats['min_bitscore']:.1f}",
            'Mean Identity': f"{combined_stats['mean_identity']:.3f}",
            'Motif Overlaps': f"{combined_stats['mean_motif_overlaps']:.1f}",
            'Sequences w/ Motifs': f"{combined_stats['sequences_with_motifs']}/{len(results['combined']['results'])}"
        })
    
    return pd.DataFrame(summary_data)

def identify_potential_concerns(analysis_results, bitscore_threshold=50, identity_threshold=0.8):
    """Identify sequences that might be too similar to known peptides"""
    
    concerns = []
    
    for prompt_type, results in analysis_results.items():
        combined_results = results['combined']['results']
        
        # High bitscore concerns
        high_bitscore = combined_results[combined_results['best_bitscore'] > bitscore_threshold]
        if len(high_bitscore) > 0:
            concerns.append({
                'type': 'high_bitscore',
                'prompt': prompt_type,
                'count': len(high_bitscore),
                'sequences': high_bitscore['sequence'].tolist(),
                'bitscores': high_bitscore['best_bitscore'].tolist()
            })
        
        # High identity concerns
        high_identity = combined_results[combined_results['best_identity'] > identity_threshold]
        if len(high_identity) > 0:
            concerns.append({
                'type': 'high_identity',
                'prompt': prompt_type,
                'count': len(high_identity),
                'sequences': high_identity['sequence'].tolist(),
                'identities': high_identity['best_identity'].tolist()
            })
    
    return concerns
