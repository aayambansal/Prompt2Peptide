# --- generate.py ---
import random
from .metrics import gravy, net_charge, hydrophobic_moment, positional_amphipathy_score
from .esm_score import esm_avg_loglik

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = set("LIVFWYCM")
POLAR = set("STNQHREQD")  # coarse

def init_seq(length:int, boost=None):
    """Initialize random sequence with optional amino acid boosting"""
    base = list(AA)
    probs = {a:1.0 for a in base}
    if boost:
        for a,w in boost.items():
            probs[a] = probs.get(a,1.0)*w
    p = [probs[a] for a in base]
    s = "".join(random.choices(base, weights=p, k=length))
    return s

def constraint_score(seq, target):
    """Enhanced scoring with composition constraints and improved charge handling"""
    # Calculate basic metrics
    g = gravy(seq)
    ch = net_charge(seq)
    
    # Use structure-aware hydrophobic moment calculation
    structure_type = target.get('structure_type', 'helix')
    mu = hydrophobic_moment(seq, structure_type=structure_type)
    
    # Calculate positional amphipathy score for helical-wheel segregation
    amphipathy_score = positional_amphipathy_score(seq, structure_type=structure_type)
    
    # Calculate composition metrics
    total_len = len(seq)
    kr_fraction = (seq.count('K') + seq.count('R')) / total_len
    de_fraction = (seq.count('D') + seq.count('E')) / total_len
    h_count = seq.count('H')
    
    sc = 0
    
    # Helper function for window scoring
    def window(x, lo, hi, w): 
        if lo <= x <= hi: return w
        d = min(abs(x-lo), abs(x-hi))
        return max(0.0, w - d*w)
    
    # Enhanced charge scoring with His penalty
    def enhanced_charge_score(charge, lo, hi, w):
        # Scale His contribution (partial protonation at pH 7.4)
        his_contribution = h_count * 0.15  # His contributes ~15% positive charge
        adjusted_charge = charge - (h_count * 0.85)  # Remove full His contribution, add back scaled
        
        if lo <= adjusted_charge <= hi:
            # Prefer center of range
            center = (lo + hi) / 2
            distance_from_center = abs(adjusted_charge - center)
            max_distance = (hi - lo) / 2
            center_bonus = (1 - distance_from_center / max_distance) * 0.3
            return w + center_bonus
        else:
            d = min(abs(adjusted_charge - lo), abs(adjusted_charge - hi))
            return max(0.0, w - d * w * 2.0)  # Even stronger penalty
    
    # Composition constraints
    def composition_score(kr_frac, de_frac, target_type):
        comp_score = 0
        if target_type == 'cationic':
            # K+R fraction should be 0.25-0.45
            if 0.25 <= kr_frac <= 0.45:
                comp_score += 1.0
            else:
                d = min(abs(kr_frac - 0.25), abs(kr_frac - 0.45))
                comp_score += max(0.0, 1.0 - d * 2.0)
            
            # D+E fraction should be ≤ 0.10
            if de_frac <= 0.10:
                comp_score += 1.0
            else:
                comp_score += max(0.0, 1.0 - (de_frac - 0.10) * 5.0)
        
        elif target_type == 'acidic':
            # D+E fraction should be high
            if de_frac >= 0.30:
                comp_score += 1.0
            else:
                comp_score += max(0.0, de_frac / 0.30)
            
            # K+R fraction should be low
            if kr_frac <= 0.10:
                comp_score += 1.0
            else:
                comp_score += max(0.0, 1.0 - (kr_frac - 0.10) * 2.0)
        
        return comp_score
    
    # Apply scoring
    sc += enhanced_charge_score(ch, *target['charge'])
    sc += window(mu, *target['muH'])
    sc += window(g, *target['gravy'])
    
    # Add composition scoring
    target_type = target.get('type', 'general')
    sc += composition_score(kr_fraction, de_fraction, target_type) * 0.5
    
    # Add positional amphipathy scoring for amphipathic sequences
    if target_type == 'cationic' and structure_type == 'helix':
        sc += amphipathy_score * 0.3  # Weight amphipathy scoring
    
    return sc, {
        'gravy': g, 
        'charge': ch, 
        'muH': mu,
        'kr_fraction': kr_fraction,
        'de_fraction': de_fraction,
        'h_count': h_count,
        'amphipathy_score': amphipathy_score
    }

def mutate(seq, target=None, phase="charge"):
    """Enhanced mutation with charge-directed moves and two-phase search"""
    if target is None or phase == "general":
        # Standard mutation
        i = random.randrange(len(seq))
        a = random.choice(AA.replace(seq[i], ""))
        return seq[:i]+a+seq[i+1:]
    
    # Get current charge
    current_charge = net_charge(seq)
    target_charge_lo, target_charge_hi = target['charge'][:2]
    target_charge_center = (target_charge_lo + target_charge_hi) / 2
    
    if phase == "charge":
        # Phase 1: Reach charge window
        if current_charge < target_charge_lo:
            # Need more positive charge - preferentially mutate non-KR to KR
            non_kr_positions = [i for i, aa in enumerate(seq) if aa not in 'KR']
            if non_kr_positions:
                i = random.choice(non_kr_positions)
                a = random.choice('KR')
                return seq[:i]+a+seq[i+1:]
        elif current_charge > target_charge_hi:
            # Need less positive charge - preferentially mutate KR to neutral
            kr_positions = [i for i, aa in enumerate(seq) if aa in 'KR']
            if kr_positions:
                i = random.choice(kr_positions)
                a = random.choice('ASTNQ')
                return seq[:i]+a+seq[i+1:]
    
    elif phase == "optimize":
        # Phase 2: Lock charge with swap-only mutations, optimize μH/GRAVY
        # Only allow mutations that don't change charge significantly
        i = random.randrange(len(seq))
        current_aa = seq[i]
        
        # Define charge-neutral swaps
        charge_groups = {
            'positive': 'KRH',
            'negative': 'DE',
            'neutral': 'ASTNQ',
            'hydrophobic': 'LIVFWYCM'
        }
        
        # Find current group
        current_group = None
        for group, aas in charge_groups.items():
            if current_aa in aas:
                current_group = group
                break
        
        if current_group:
            # Swap within same charge group
            available_aas = charge_groups[current_group].replace(current_aa, "")
            if available_aas:
                a = random.choice(available_aas)
                return seq[:i]+a+seq[i+1:]
    
    # Fallback to standard mutation
    i = random.randrange(len(seq))
    a = random.choice(AA.replace(seq[i], ""))
    return seq[:i]+a+seq[i+1:]

def generate(prompt:str, n=20, iters=500):
    """Generate peptides based on text prompt using constraint-guided sampling + LM rescoring"""
    # Enhanced parser with more prompt types
    prompt_lower = prompt.lower()
    
    if "cationic" in prompt_lower and "amphipathic" in prompt_lower:
        target = {
            'charge':(+3,+8, 1.0), 
            'muH':(0.35,1.0, 1.0), 
            'gravy':(-0.2,0.6, 0.5),
            'type': 'cationic'
        }
        boost = {'K':2.0,'R':2.0,'L':1.5,'I':1.5,'V':1.2,'F':1.2}
        L = random.randint(12,18)
    elif "soluble" in prompt_lower and "acidic" in prompt_lower:
        target = {
            'charge':(-3,0, 1.0), 
            'muH':(0.1,0.4, 0.5), 
            'gravy':(-1.0,0.0, 1.0),
            'type': 'acidic'
        }
        boost = {'D':2.0,'E':2.0,'S':1.5,'T':1.5,'N':1.2,'Q':1.2}
        L = random.randint(10,14)
    elif "hydrophobic" in prompt_lower and "sheet" in prompt_lower:
        target = {
            'charge':(-1,+2, 0.8), 
            'muH':(0.1,0.3, 0.3), 
            'gravy':(0.5,1.5, 1.0),
            'structure_type': 'sheet'  # Use β-sheet periodicity for μH
        }
        boost = {'F':2.0,'W':2.0,'Y':1.8,'L':1.8,'I':1.8,'V':1.5}
        L = random.randint(10,14)
    elif "polar" in prompt_lower and "flexible" in prompt_lower:
        target = {'charge':(-1,+1, 0.5), 'muH':(0.05,0.25, 0.2), 'gravy':(-0.8,0.2, 1.0)}
        boost = {'G':2.0,'P':1.5,'S':1.5,'T':1.5,'N':1.3,'Q':1.3}
        L = random.randint(8,12)
    elif "basic" in prompt_lower and "nuclear" in prompt_lower:
        target = {'charge':(+4,+8, 1.0), 'muH':(0.2,0.6, 0.8), 'gravy':(-0.5,0.3, 0.5)}
        boost = {'K':2.5,'R':2.5,'P':1.5,'S':1.2,'T':1.2}
        L = random.randint(7,12)
    else:
        target = {'charge':(0, +3, 1.0), 'muH':(0.15,1.0, 0.8), 'gravy':(-0.5,0.5, 0.5)}
        boost = {}
        L = random.randint(10,20)

    cands = []
    for _ in range(n):
        seq = init_seq(L, boost)
        best, best_sc = seq, constraint_score(seq, target)[0]
        T = 1.0
        
        # Two-phase search
        phase1_iters = int(iters * 0.6)  # 60% for charge phase
        phase2_iters = iters - phase1_iters  # 40% for optimization phase
        
        # Phase 1: Reach charge window
        for t in range(phase1_iters):
            s2 = mutate(best, target, phase="charge")
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T*0.1):
                best, best_sc = s2, sc2
            T *= 0.995
        
        # Phase 2: Lock charge, optimize μH/GRAVY
        for t in range(phase2_iters):
            s2 = mutate(best, target, phase="optimize")
            sc2 = constraint_score(s2, target)[0]
            if sc2 > best_sc or random.random() < (T*0.05):
                best, best_sc = s2, sc2
            T *= 0.995
        
        cands.append(best)

    # LM re-score and return top
    scored = [(seq, constraint_score(seq, target)[1], esm_avg_loglik(seq)) for seq in cands]
    scored.sort(key=lambda x: x[2], reverse=True)  # higher avg loglik = better
    return scored[:10]  # (seq, metrics, lm_score)
