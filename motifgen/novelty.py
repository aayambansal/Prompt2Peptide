# --- novelty.py ---
import numpy as np
from difflib import SequenceMatcher

def sequence_identity(seq1, seq2):
    """Calculate sequence identity between two sequences"""
    return SequenceMatcher(None, seq1, seq2).ratio()

def pairwise_identity_matrix(sequences):
    """Calculate pairwise identity matrix for a list of sequences"""
    n = len(sequences)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            identity = sequence_identity(sequences[i], sequences[j])
            matrix[i, j] = identity
            matrix[j, i] = identity
    return matrix

def novelty_score(sequences, reference_sequences=None, threshold=0.7):
    """Calculate novelty score for generated sequences"""
    if reference_sequences is None:
        # Use a small set of common peptide motifs as reference
        reference_sequences = [
            "KKKKKKKKKKKK",  # Poly-lysine
            "RRRRRRRRRRRR",  # Poly-arginine
            "LLLLLLLLLLLL",  # Poly-leucine
            "FFFFFFFFFFFF",  # Poly-phenylalanine
            "GGGGGGGGGGGG",  # Poly-glycine
            "PPPPPPPPPPPP",  # Poly-proline
            "KLAKLAKLAKLA",  # Amphipathic
            "KLALKLALKLAL",  # Amphipathic variant
            "RGDGRGDGRGDG",  # RGD motif
            "DEEDDEEDDEED",  # Acidic
        ]
    
    novel_count = 0
    max_identities = []
    
    for seq in sequences:
        max_identity = 0
        for ref_seq in reference_sequences:
            identity = sequence_identity(seq, ref_seq)
            max_identity = max(max_identity, identity)
        
        max_identities.append(max_identity)
        if max_identity < threshold:
            novel_count += 1
    
    return {
        'novel_count': novel_count,
        'total_count': len(sequences),
        'novelty_percentage': 100 * novel_count / len(sequences),
        'max_identities': max_identities,
        'avg_max_identity': np.mean(max_identities)
    }

def diversity_metrics(sequences):
    """Calculate diversity metrics for a set of sequences"""
    if len(sequences) < 2:
        return {'pairwise_diversity': 0, 'unique_sequences': len(set(sequences))}
    
    # Calculate pairwise identity matrix
    identity_matrix = pairwise_identity_matrix(sequences)
    
    # Get upper triangle (excluding diagonal)
    upper_triangle = identity_matrix[np.triu_indices_from(identity_matrix, k=1)]
    
    return {
        'pairwise_diversity': 1 - np.mean(upper_triangle),  # Average dissimilarity
        'unique_sequences': len(set(sequences)),
        'avg_pairwise_identity': np.mean(upper_triangle),
        'min_pairwise_identity': np.min(upper_triangle),
        'max_pairwise_identity': np.max(upper_triangle)
    }
