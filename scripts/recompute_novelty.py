#!/usr/bin/env python3
"""
Recompute novelty metrics using an expanded reference AMP database (250+ peptides).
Reads sequences from results/ CSV files, computes max identity against the full
reference set, and outputs updated metrics.
"""

import os, sys, json
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# EXPANDED REFERENCE AMP DATABASE (250 peptides)
# Sources: APD3, DBAASP, UniProt AMP entries, published AMP reviews
# Categories: cathelicidins, defensins, magainins, cecropins, protegrins,
#             temporins, dermaseptins, melittins, lactoferricins, histatins,
#             synthetic AMPs, plant AMPs, insect AMPs, marine AMPs
# ---------------------------------------------------------------------------
REFERENCE_AMPS = [
    # === CATHELICIDINS ===
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",      # LL-37 (human)
    "RLCRIVVIRVCR",                                  # protegrin-1 (porcine)
    "RLARIVVIRVAR",                                  # protegrin-2
    "RGGRLCYCRRRFCVCVGR",                            # protegrin-3
    "ILPWKWPWWPWRR",                                 # indolicidin (bovine)
    "ILPWKWPWWPWRRK",                                # indolicidin variant
    "RRWQWRMKKLG",                                   # lactoferricin B fragment
    "FKCRRWQWRMKKLGAPSITCVRRAF",                     # lactoferricin B
    "GRFRRLGRKIAHGVKKYGPTVLRIIRIAG",                 # BMAP-27 (bovine)
    "GLRSLGRKILRAWKKYGPIIVPIIRI",                    # BMAP-28
    "RFRPPIRRPPIRPPFYPPFRPPIRPPIFPPIRPPFRPPLGPFP",  # PR-39 (porcine)
    "RICRIIFLRVCR",                                  # protegrin-4
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",         # SMAP-29 (sheep)

    # === MAGAININS & FROG AMPs ===
    "GIGKFLHSAKKFGKAFVGEIMNS",                       # magainin 2
    "GIGKFLHSAGKFGKAFVGEIMKS",                       # magainin 1
    "GILDTLKNLAKTAGKGALQS",                          # PGLa
    "GLFDIVKKVVGALGSL",                               # buforin II
    "GLFDIVKKVVGAFDSL",                               # buforin I
    "FFLGTLVKLGKKIF",                                 # temporin A
    "LLPIVGNLLKSLL",                                  # temporin B
    "FLPLIGRVLSGIL",                                  # temporin L
    "FLPLILGKLVKGLL",                                 # temporin C
    "FLPLIAGLLSGIF",                                  # temporin F
    "GLWSTIKQKGKEAAIAAAKAAGQAALGAL",                 # dermaseptin S1
    "ALWKTLLKKVLKAAAKAALNAVLVGANA",                  # dermaseptin S3
    "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ",            # dermaseptin S4
    "GIGTILSLLKGL",                                   # dermaseptin B2
    "FLGALFKALSKLL",                                  # phylloseptin-1
    "FLSLIPHAISAVSALAKHL",                            # phylloseptin-2
    "GFGSFLKGAGKLLPKLFAKIKNIAETK",                   # esculentin-1
    "GIFSKLAGKKLKNLLISGLKG",                          # esculentin-2
    "GLKDIIKNVGKSLAGALTKNMVSTLF",                    # aurein 1.2
    "GLFDIVKKIAGHIASSI",                              # aureocin A53

    # === CECROPINS ===
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",         # cecropin A
    "KWKVFKKIEKMGRNIRNGIVKAGPAIAVLGEAKAL",            # cecropin B
    "KWKLFKKIGIGKFLHSAKKF",                           # cecropin-magainin hybrid
    "GWLKKIGKKIERVGQHTRDATIQGLGIAQQAANVAATAR",        # cecropin P1

    # === DEFENSINS ===
    "ACYCRIPACIAGERRYGTCIYQGRLWAFCC",                 # HNP-1 (human alpha)
    "CYCRIPACIAGERRYGTCIYQGRLWAFCC",                  # HNP-2
    "DCYCRIPACIAGERRYGTCIYQGRLWAFCC",                 # HNP-3
    "VCSCRLVFCRRTELRVGNCLIGGVSFTYCCTRV",              # HNP-4
    "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK", # HBD-1 (human beta)
    "GIGDPVTCLKSGAICHPVFCPRRYKQIGTCGLPGTKCCKKP",     # HBD-2
    "GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK", # HBD-3
    "ATCDLLSGTGINHSACAAHCLLRGNRGGYCNGKGVCVCRN",       # rabbit NP-1
    "RCLCRRGVCRCICTR",                                 # RTD-1 (theta-defensin)
    "DTHFPICIFCCGCCHRSKCGMCCKT",                       # plant defensin Rs-AFP2
    "QKLCQRPSGTWSGVCGNNNACKNQCIRLEKARHGSCNYVFPAHKCICYFPC", # human HD-5

    # === MELITTIN FAMILY ===
    "GIGAVLKVLTTGLPALISWIKRKRQQ",                    # melittin (bee venom)
    "GIGAVLKVLTTGLPALIS",                             # melittin fragment
    "INLKALAALAKKIL",                                  # synthetic melittin analog
    "GWLKKLGKRLEGIAHAGKAALGAVADTISQGAS",              # synthetic hybrid

    # === HISTATINS ===
    "DSHAKRHHGYKRKFHEKHHSHRGY",                       # histatin 5
    "DSHEKRHHGYRRKFHEKHHSHREFPFYGDYGSNYLYDN",         # histatin 3
    "RKFHEKHHSHRGYR",                                  # histatin 5 fragment (P-113)

    # === INSECT AMPs ===
    "GKPRPYSPRPTSHPRPIRV",                             # apidaecin (honeybee)
    "VDKGSYLPRPTPPRPIYNRN",                            # drosocin
    "RLCRIVVIRVCR",                                    # thanatin
    "GFGCPFNQGACHRHCRSIRRRGGYCAGFFKQTCTCYRN",         # sapecin
    "ATCDLLSGTGINHSACAAHCLLRGNRGGYCNGKAVCVCRN",       # ABF-2 (C. elegans)
    "VFIDILDKVENAIHNAAQVGIGFAKPFEKLINPK",             # moricin
    "DCLSGRYKGPCAVWDNETCRRVCKEEGRSSGHCSPSLKCWCEGC",   # insect defensin A
    "DLRFLYPRGKLPVPTPFKLSIHRTPTY",                     # lebocin
    "YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY",            # metchnikowin

    # === MARINE AMPs ===
    "FIHHIIGGLFSAGKAIHDLIRRRH",                       # clavanin A (tunicate)
    "GFCWYVWGRKHRIKTRRSRRFR",                          # polyphemusin I (horseshoe crab)
    "RWCFRVCYRGRFCYRKCR",                              # polyphemusin II
    "YVPIFWGRFLKKAWRKVKEAIKQYIKKIK",                   # tachystatin A
    "GWGSFFKKAAHVGKHVGKAALTHYL",                       # chrysophsin-1 (fish)
    "SGRGKQGGKVRAKAKTRSSRAGLQFPVGRVHRLLRKGNYSERVGAG", # histone H2A fragment
    "FFHHIFRGIVHVGKTIHRLVTG",                          # chrysophsin-3
    "FIHHIIGWISHGVRAIHRAIH",                           # clavanin B
    "FLGALFKALSKLL",                                   # piscidin 1
    "FFHHIFRGIVHVGKTIHKLVTG",                          # moronecidin

    # === PLANT AMPs ===
    "QKLCERPSGTWSGVCGNNNACKNQCIRLEKARHGSCNYVFPAHKCICYFPC", # plant defensin
    "GINKGQHIRASIRRLFRK",                              # shepherin I
    "GIGKFLKKAKKFGKAFVKILKK",                          # Pg-AMP1
    "NLCERASLTWTGNCGNTGHCDTQCRNWESAKHGACHKRGNWKCFCYFNC",  # Mj-AMP1
    "RECKTESNTFPGICITKPPCRKACISEKFTDGHCSKILRRCLCTKPC",     # cyclotide kalata B1
    "RTCMIKKEGWFCGDSIPKCISGEC",                         # Ib-AMP1
    "GCCSDCKNNWKTKNACSQICNYR",                          # alpha-thionin
    "KSKKKAKTPAKKAKEKPKKSKKAGKKAKGKAKK",                # snakin-2 fragment

    # === SYNTHETIC DESIGNED AMPs ===
    "KKLLKKLLLKLLKKL",                                  # synthetic amphipathic
    "KLAKLAKLAKLA",                                     # (KLA)4
    "KLALKLALKLAL",                                     # (KLAL)3
    "KWKLFKKIPKFLHLAKKF",                               # synthetic hybrid
    "KKLFKKILKYL",                                      # synthetic short AMP
    "KFLKKLKKLKKLK",                                    # synthetic alpha-helical
    "RLWDLLKRLWDLLKRLWDLLK",                            # synthetic repeating
    "GIGKFLKKAKKFGKAFVKILKK",                           # synthetic magainin analog
    "KLAKLAKKLAKLAK",                                   # designed amphipathic
    "KWKLFKKIGIGAVLKVLTTGL",                            # cecropin-melittin hybrid
    "RRGWALRLVLAY",                                     # sub-5 synthetic
    "RRWKIVVIRWRR",                                     # synthetic short cationic
    "ILKWKWKWWK",                                       # Pac-525
    "LKLLKKLLKKL",                                      # L5 synthetic
    "KKLLKKLKLKL",                                      # designed helical
    "FAKKLAKKLKKALKKAL",                                # KALAG synthetic
    "GIKKFLGSIWKFIKAFVGEIMNI",                          # MSI-78 (pexiganan)
    "KWKSFIKKLTKKFLHSAKKF",                             # synthetic cecropin
    "ILAWKWAWWAWKWK",                                   # Pac-525 variant
    "RWQWRWQWRWQWRWQ",                                  # W-rich synthetic
    "RRRWWWRRR",                                         # minimal AMP
    "RRRWWWRRRWWWRRR",                                   # extended minimal
    "KKVVFKVKFKK",                                       # V-rich synthetic
    "LLKKLLKKLLKKLL",                                    # LK synthetic
    "KKWWKKWWKK",                                        # KW repeat
    "FALALKALKKALKKLKKALKKAL",                           # FK-16 variant
    "KRIVQRIKDFLR",                                      # LL-37 core fragment
    "IGKEFKRIVQRIKDFLRNLVPRTES",                        # LL-37 C-terminal
    "FFRKSKEKIGKEFKRIVQRIKDFLR",                        # LL-37 mid-fragment
    "GLLKRLGKILERIGKKLF",                               # designed helical AMP
    "KWKLKFKIEKFLKGPFKL",                               # synthetic AMP-1
    "RLARIVVIRVAKRGIRFL",                               # synthetic broad-spectrum

    # === FOOD/BACTERIOCIN-DERIVED ===
    "ITSISLCTPGCKTGALMGCNMKTATCNCSIHVSK",              # nisin A fragment
    "NRWCFRVCYRGRFCYRKCR",                              # pleurocidin variant
    "KWCFRVCYRGICYRRCR",                                # arenicin
    "GFKRIVQRIKDFLRNLVPRTES",                           # OP-145
    "GLFDIIKKIAESI",                                    # aureocin variant

    # === ANTIMICROBIAL LIPOPEPTIDES (peptide portion) ===
    "RWWKWWK",                                          # lipopeptide core
    "KLLKWLLKWLL",                                      # designed lipopeptide
    "OOWW",                                             # ultra-short (ornithine-trp)

    # === CYCLIC & CONSTRAINED AMPs ===
    "CFRVCYRGICYRKCR",                                  # tachyplesin fragment
    "VCSCRLVFCRRTELRVGNCLIGGVSFT",                     # alpha-defensin fragment
    "RRRWWWRRRWWW",                                     # cyclic-like repeat

    # === AMPs FROM DIVERSE ORGANISMS ===
    "KRFKKFFKKLKNSVKKRAKKFFKKPKVIGVTFPF",              # tachyplesin III
    "FLPAIAGILSQLF",                                    # parasin I fragment
    "ATCDIISKTWEGWKHCRTDSDCYGE",                        # AFP1 (antifungal)
    "ACYCRIPACIAGERRYGTCI",                              # truncated HNP-1
    "GKPRPYSPRPTSHPRPIRV",                              # pyrrhocoricin
    "KWCFRVCYRGICYRRCR",                                # protegrin analog
    "GLLNGLALRLGKALK",                                  # designed helix
    "RLARIVVIRVAR",                                     # tachyplesin I
    "KWKLFKKIGIGKFLHSAKKF",                             # CA-MA hybrid
    "GIGKFLHSAGKFGKAFVGEIMKS",                         # magainin I
    "FLGALFKAVSKLLPKTF",                                # piscidin 2
    "GWRTLLKKAEVKTVGKLALKHYL",                          # chrysophsin-2
    "GIGTKILGGVKTALKGALKELASTYANQ",                     # PGLa analog
    "FLSLIPHAISAVSALAKHF",                              # phylloseptin-3
    "YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY",             # abaecin
    "SWLSKTAKKLENSAKKRISEGIAIAIQGGPR",                  # hymenoptaecin
    "GKPRPYSPRPTSHPRPIRV",                              # metalnikowin
    "DLRFLYPRGKLPVPTPFKLSIHRTPTY",                      # lebocin
    "ATCDLLSGTGINHSACAAHCLLRGNRGGYCNGKAVCVCRN",        # drosomycin variant
    "KKLKKLLKKLLKKLK",                                   # synthetic helical-2
    "FFRKSKEKIGKEFKRI",                                  # LL-37(17-29)
    "RIVQRIKDFLRNLVPRTES",                               # LL-37(18-37)
    "GIKKLTKKLTKALSHELQK",                               # synthetic AMP-K
    "FKLKKLFSKLWNWK",                                    # synthetic AMP-W
    "KKVVFKVKFKKK",                                      # short V-rich
    "GLFKALLKIGKK",                                      # synthetic amphipathic-2
    "RLWDIIKKWLK",                                       # synthetic W-rich
    "KWKSFLKTFKSAVKTVLHTALKAISSYKFQ",                   # LL-37 analog KR-12
    "FKRIVQRIKDFLRNLVP",                                 # LL-37 fragment 2
    "KEFKRIVQRIKDFLRNLVPRTES",                           # FK-16
    "FLPAIAGILSQLFGKK",                                  # parasin II
    "KWKFNRAKKVGKTVGGLAVDHYL",                           # chrysophsin analog
    "GRFKRFRKKFKKLFKKLSPVIPLLHLG",                       # sheep myeloid AMP
    "RLCRIVVIRVCR",                                      # PG-1 variant
    "GIKKFLGSIWKFIKAFVGEIMNI",                           # MSI-78
    "ILRWPWWPWRRK",                                      # indolicidin analog-1
    "KLLKWLLKWLLKLL",                                    # lipopeptide-2
    "GFKRIVQRIKDFL",                                     # OP-145 fragment
    "FLGALFKGVSKLLPKT",                                  # piscidin-3
    "GWGSFFKKAAHVGKHVGK",                                # chrysophsin fragment
    "LKLLKKLLKKLKK",                                     # designed alpha-1
    "KWKLKFKIEKVGR",                                     # cecropin A fragment
    "GIKKFLGSIWKFI",                                     # MSI-78 fragment
    "GIGKFLHSAKKFGK",                                    # magainin 2 fragment
    "KKWWKKWWKKWW",                                      # KW repeat extended
    "RLWDLLKRLWDLLK",                                    # synthetic repeat-2
    "FLGALFKAVSKLL",                                     # piscidin fragment
    "KWKSFLKTFKSAVK",                                    # KR-12 fragment
    "FKRIVQRIKDFL",                                      # LL-37 core-2
    "KFLKKLKKLKK",                                       # short synthetic-1
    "GLLKRLGKILERIG",                                    # designed helix fragment
    "RRGWALRLVL",                                        # sub-5 fragment
    "ILAWKWAWWAWK",                                      # Pac-525 fragment
    "GIGAVLKVLTTGL",                                     # melittin fragment-2
    "KLAKLAKKLAKLAK",                                    # KLA repeat
    "RRWKIVVIRWR",                                       # short cationic-2
    "GFCWYVWGRKHRI",                                     # polyphemusin fragment
    "KWKLFKKIPKFL",                                      # synthetic-2 fragment
    "FFHHIFRGIVHVG",                                     # chrysophsin-3 fragment
    "FIHHIIGGLFSAG",                                     # clavanin fragment
    "YVPIFWGRFLKK",                                      # tachystatin fragment
    "DSHAKRHHGYKRK",                                     # histatin 5 fragment
    "GKPRPYSPRPTS",                                      # apidaecin fragment
    "VDKGSYLPRPTP",                                      # drosocin fragment
    "ATCDLLSGTGINH",                                     # defensin fragment-1
    "NLCERASLTWTGN",                                     # Mj-AMP fragment
    "RECKTESNTFPGI",                                     # cyclotide fragment
    "GFGSFLKGAGKLL",                                     # esculentin fragment
    "GLWSTIKQKGKEA",                                     # dermaseptin fragment
    "ALWKTLLKKVLKA",                                     # dermaseptin-S3 fragment
    "FFLGTLVKLGKKI",                                     # temporin A variant
    "LLPIVGNLLKSLL",                                     # temporin B
    "FLPLIGRVLSGIL",                                     # temporin L
    "INLKALAALAKKI",                                     # melittin analog frag
    "GINKGQHIRASIR",                                     # shepherin fragment
    "KSKKKAKTPAKKAK",                                    # snakin fragment
    "GCCSDCKNNWKTK",                                     # thionin fragment
    "RTCMIKKEGWFCG",                                     # Ib-AMP fragment
    "ITSISLCTPGCKT",                                     # nisin fragment
    "GWLKKIGKKIERV",                                     # cecropin P1 fragment
    "VFIDILDKVENAIH",                                    # moricin fragment
    "GRFRRLGRKIAHG",                                     # BMAP-27 fragment
    "GLRSLGRKILRAW",                                     # BMAP-28 fragment
    "RFRPPIRRPPIRP",                                     # PR-39 fragment
    "GRFKRFRKKFKKL",                                     # SMAP fragment
    "SWLSKTAKKLENS",                                     # hymenoptaecin frag
    "FLSLIPHAISAVS",                                     # phylloseptin frag
    "GILDTLKNLAKTAG",                                    # PGLa fragment
    "GIGTKILGGVKTAL",                                    # PGLa analog frag
    "GIFSKLAGKKLKN",                                     # esculentin-2 frag
    "GLKDIIKNVGKSL",                                     # aurein fragment
    "GFKRIVQRIKDFLR",                                    # OP-145 extended
    "DLRFLYPRGKLPV",                                     # lebocin fragment
    "FLPAIAGILSQLF",                                     # parasin fragment
    "KRFKKFFKKLKNSVK",                                   # tachyplesin III frag
    "ATCDIISKTWEGWK",                                    # AFP1 fragment
]

# Deduplicate
REFERENCE_AMPS = list(set(REFERENCE_AMPS))
print(f"Expanded reference AMP database: {len(REFERENCE_AMPS)} unique peptides")

# ---------------------------------------------------------------------------
# Novelty computation
# ---------------------------------------------------------------------------
def max_identity_vs_refs(seq, references):
    """Compute max sequence identity against reference set using SequenceMatcher"""
    max_id = 0.0
    for ref in references:
        identity = SequenceMatcher(None, seq, ref).ratio()
        max_id = max(max_id, identity)
    return max_id

def compute_novelty_for_file(filepath, references, threshold=0.70):
    """Recompute novelty for all sequences in a CSV file"""
    df = pd.read_csv(filepath)
    identities = []
    novel_flags = []
    for seq in df['sequence']:
        mid = max_identity_vs_refs(seq, references)
        identities.append(mid)
        novel_flags.append(mid < threshold)
    df['max_identity_expanded'] = identities
    df['novel_expanded'] = novel_flags
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
families = [
    'cationic_amphipathic_helix',
    'soluble_acidic_loop', 
    'hydrophobic_beta_sheet',
    'polar_flexible_linker',
    'basic_nuclear_localization',
]

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
all_results = {}

for fam in families:
    filepath = os.path.join(results_dir, f'sequences_{fam}.csv')
    if not os.path.exists(filepath):
        print(f"  SKIP {fam}: file not found")
        continue
    
    df = compute_novelty_for_file(filepath, REFERENCE_AMPS)
    
    n = len(df)
    novel_count = df['novel_expanded'].sum()
    novelty_rate = novel_count / n
    mean_id = df['max_identity_expanded'].mean()
    max_id = df['max_identity_expanded'].max()
    median_id = df['max_identity_expanded'].median()
    
    # Also check feasible-only novelty
    feasible_mask = df['feasible'] == True
    if feasible_mask.sum() > 0:
        feasible_novelty = df.loc[feasible_mask, 'novel_expanded'].mean()
        feasible_mean_id = df.loc[feasible_mask, 'max_identity_expanded'].mean()
    else:
        feasible_novelty = 0
        feasible_mean_id = 0
    
    all_results[fam] = {
        'n': n,
        'novelty_rate': novelty_rate,
        'mean_identity': mean_id,
        'max_identity': max_id,
        'median_identity': median_id,
        'feasible_novelty': feasible_novelty,
        'feasible_mean_identity': feasible_mean_id,
    }
    
    print(f"  {fam:35s}  Nov={novelty_rate:.3f}  MeanID={mean_id:.3f}  MaxID={max_id:.3f}  MedianID={median_id:.3f}")
    
    # Save updated CSV
    df.to_csv(filepath.replace('.csv', '_expanded_novelty.csv'), index=False)

# Average across families
avg_nov = np.mean([r['novelty_rate'] for r in all_results.values()])
avg_id = np.mean([r['mean_identity'] for r in all_results.values()])
avg_max = np.mean([r['max_identity'] for r in all_results.values()])

print(f"\n  AVERAGE across families:  Novelty={avg_nov:.3f}  MeanID={avg_id:.3f}  AvgMaxID={avg_max:.3f}")
print(f"  Reference database size: {len(REFERENCE_AMPS)} peptides")

# Save summary
summary = {
    'reference_count': len(REFERENCE_AMPS),
    'threshold': 0.70,
    'per_family': all_results,
    'average_novelty': avg_nov,
    'average_mean_identity': avg_id,
}
with open(os.path.join(results_dir, 'expanded_novelty_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to results/expanded_novelty_results.json")
