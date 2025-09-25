import csv, argparse
from motifgen.generate import generate

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", type=str, required=True)
ap.add_argument("--out", type=str, default="generated.csv")
ap.add_argument("--n", type=int, default=60, help="Number of candidates to generate")
ap.add_argument("--iters", type=int, default=800, help="Iterations for constraint optimization")
args = ap.parse_args()

print(f"Generating peptides for prompt: '{args.prompt}'")
print(f"Parameters: n={args.n}, iters={args.iters}")

rows = generate(args.prompt, n=args.n, iters=args.iters)
with open(args.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sequence","length","muH","charge","gravy","esm_ll"])
    for seq, met, ll in rows:
        w.writerow([seq, len(seq), met["muH"], met["charge"], met["gravy"], ll])
print(f"Wrote {len(rows)} sequences to {args.out}")

# Print top 5 sequences for quick inspection
print("\nTop 5 generated sequences:")
for i, (seq, met, ll) in enumerate(rows[:5]):
    print(f"{i+1}. {seq} (μH={met['muH']:.3f}, charge={met['charge']:.2f}, GRAVY={met['gravy']:.3f}, ESM-LL={ll:.3f})")
