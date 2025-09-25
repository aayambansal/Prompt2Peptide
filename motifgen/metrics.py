# --- metrics.py ---
import math
import numpy as np

KD = { 'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
       'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
       'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2 }

def gravy(seq:str)->float:
    """Calculate GRAVY (Grand Average of Hydropathy) using Kyte-Doolittle scale"""
    return sum(KD[a] for a in seq)/len(seq)

# Net charge at pH 7.4 (quick heuristic; good enough for ranking small peptides)
def net_charge(seq:str, ph=7.4)->float:
    """Calculate net charge at given pH using approximate pKa values"""
    # pKa (approx): K 10.5, R 12.5, H 6.0; D 3.9, E 4.3
    from math import pow
    pka = {'K':10.5,'R':12.5,'H':6.0,'D':3.9,'E':4.3}
    pos = sum(1/(1+10**(ph-pka[r])) for r in seq if r in 'KRH')
    neg = sum(1/(1+10**(pka[r]-ph)) for r in seq if r in 'DE')
    return pos - neg

# Hydrophobic moment (Eisenberg) with structure-aware periodicity
def hydrophobic_moment(seq:str, deg_per_res=100, structure_type="helix")->float:
    """Calculate hydrophobic moment using Eisenberg method with structure-aware periodicity
    
    Args:
        seq: Amino acid sequence
        deg_per_res: Degrees per residue (100° for helix, ~180° for β-sheet)
        structure_type: "helix" (100°) or "sheet" (~180°) or "mixed" (100°)
    """
    if structure_type == "sheet":
        # β-sheet periodicity is approximately 180° per residue
        deg_per_res = 180
    elif structure_type == "helix":
        # α-helix periodicity is 100° per residue
        deg_per_res = 100
    # For mixed/unknown, default to helical periodicity
    
    rad = math.radians(deg_per_res)
    vx = vy = 0.0
    for i,a in enumerate(seq):
        angle = i*rad
        vx += KD[a]*math.cos(angle)
        vy += KD[a]*math.sin(angle)
    return math.sqrt(vx*vx+vy*vy)/len(seq)

def positional_amphipathy_score(seq:str, structure_type="helix")->float:
    """Calculate positional amphipathy score for helical-wheel segregation
    
    Args:
        seq: Amino acid sequence
        structure_type: "helix" (100°) or "sheet" (~180°) or "mixed" (100°)
    
    Returns:
        Amphipathy score (0-1, higher = better segregation)
    """
    if structure_type == "sheet":
        deg_per_res = 180
    elif structure_type == "helix":
        deg_per_res = 100
    else:
        deg_per_res = 100
    
    # Define hydrophobic and hydrophilic residues
    hydrophobic = set("LIVFWYCM")
    hydrophilic = set("STNQHREQD")
    
    # Calculate angular positions
    rad = math.radians(deg_per_res)
    hydrophobic_angles = []
    hydrophilic_angles = []
    
    for i, aa in enumerate(seq):
        angle = i * rad
        if aa in hydrophobic:
            hydrophobic_angles.append(angle)
        elif aa in hydrophilic:
            hydrophilic_angles.append(angle)
    
    if len(hydrophobic_angles) == 0 or len(hydrophilic_angles) == 0:
        return 0.0  # No segregation possible
    
    # Calculate mean angles for each group
    hydrophobic_mean = np.mean(hydrophobic_angles)
    hydrophilic_mean = np.mean(hydrophilic_angles)
    
    # Calculate angular separation (accounting for periodicity)
    angle_diff = abs(hydrophobic_mean - hydrophilic_mean)
    angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Handle periodicity
    
    # Normalize to 0-1 scale (perfect segregation = π radians = 1.0)
    segregation_score = angle_diff / math.pi
    
    # Weight by the balance of hydrophobic/hydrophilic residues
    balance_weight = 1.0 - abs(len(hydrophobic_angles) - len(hydrophilic_angles)) / len(seq)
    
    return segregation_score * balance_weight
