"""
Aggregate exp2c dimension ablation (multi-seed)
Read exp2c_dim_ablation_results.json from each seed directory, compute mean±std.

Usage:
  python aggregate_exp2c.py --base results/bidir_comparison/bidir_30pct --seeds 42,137,271,314,577
"""

import argparse, json, os
import numpy as np

DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']


def load_seed(base, seed):
    path = os.path.join(base, f"results_v4_ph5_bidir30_s{seed}", "exp2c_dim_ablation_results.json")
    if not os.path.exists(path):
        print(f"⚠️ Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_diff_matrix(all_data, matrix_key):
    """Extract diff matrix from multiple seeds, compute mean±std"""
    n = len(all_data)
    
    # Build 13x13xn_seeds array
    vals = np.full((13, 13, n), np.nan)
    
    for si, data in enumerate(all_data):
        dm = data.get(matrix_key, {})
        for ai, abl in enumerate(DIM_NAMES):
            row = dm.get(abl, {})
            for ti, tgt in enumerate(DIM_NAMES):
                if ai == ti:
                    continue
                vals[ai, ti, si] = row.get(tgt, np.nan)
    
    mean = np.nanmean(vals, axis=2)
    std = np.nanstd(vals, axis=2)
    return mean, std


def print_matrix(mean, std, label):
    print(f"\n{'='*80}")
    print(f"  {label} · Mean (5 seeds)")
    print(f"{'='*80}")
    print(f"  {'ablated':<6s}", end='')
    for d in DIM_NAMES:
        print(f" {d:>8s}", end='')
    print()
    
    for ai, abl in enumerate(DIM_NAMES):
        print(f"  {abl:<6s}", end='')
        for ti, tgt in enumerate(DIM_NAMES):
            if ai == ti:
                print(f" {'---':>8s}", end='')
            else:
                print(f" {mean[ai,ti]:+.3f}", end='')
        print()
    
    print(f"\n  {'ablated':<6s}", end='')
    for d in DIM_NAMES:
        print(f" {d:>8s}", end='')
    print(f"  ← Std")
    
    for ai, abl in enumerate(DIM_NAMES):
        print(f"  {abl:<6s}", end='')
        for ti, tgt in enumerate(DIM_NAMES):
            if ai == ti:
                print(f" {'---':>8s}", end='')
            else:
                print(f" {std[ai,ti]:.3f}", end='')
        print()


def extract_significant(mean, std, threshold=0.10):
    """Extract dependency pairs where |mean| > threshold"""
    sig = []
    for ai, abl in enumerate(DIM_NAMES):
        for ti, tgt in enumerate(DIM_NAMES):
            if ai == ti:
                continue
            m = mean[ai, ti]
            s = std[ai, ti]
            if abs(m) > threshold:
                sig.append((abl, tgt, m, s))
    sig.sort(key=lambda x: -abs(x[2]))
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='bidir_30pct directory path')
    p.add_argument('--seeds', default='42,137,271,314,577')
    p.add_argument('--threshold', type=float, default=0.10,
                   help='significance threshold (default: 0.10)')
    args = p.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    all_data = []
    for s in seeds:
        d = load_seed(args.base, s)
        if d is not None:
            all_data.append(d)
    
    print(f"Loaded {len(all_data)}/{len(seeds)} seed results")
    
    if len(all_data) < 2:
        print("❌ Need at least 2 seeds")
        return
    
    # S2.1 Name format
    mean_n, std_n = aggregate_diff_matrix(all_data, 'diff_matrix_s2_name')
    print_matrix(mean_n, std_n, "S2.1 Name · Diff Matrix")
    
    sig_n = extract_significant(mean_n, std_n, args.threshold)
    print(f"\n  Significant dependencies (|effect| > {args.threshold}):")
    print(f"  {'Ablated':<6s} → {'Affected':<6s}  {'Mean':>8s} {'±Std':>8s}")
    for abl, tgt, m, s in sig_n:
        print(f"  {abl:<6s} → {tgt:<6s}  {m:+.3f}    ±{s:.3f}")
    
    # S2.4 Interact format
    mean_i, std_i = aggregate_diff_matrix(all_data, 'diff_matrix_s2_interact')
    print_matrix(mean_i, std_i, "S2.4 Interact · Diff Matrix")
    
    sig_i = extract_significant(mean_i, std_i, args.threshold)
    print(f"\n  Significant dependencies (|effect| > {args.threshold}):")
    print(f"  {'Ablated':<6s} → {'Affected':<6s}  {'Mean':>8s} {'±Std':>8s}")
    for abl, tgt, m, s in sig_i:
        print(f"  {abl:<6s} → {tgt:<6s}  {m:+.3f}    ±{s:.3f}")
    
    # Save summary
    out = {
        'n_seeds': len(all_data),
        'seeds': seeds[:len(all_data)],
        's2_name': {
            'mean': {DIM_NAMES[ai]: {DIM_NAMES[ti]: round(float(mean_n[ai,ti]), 4)
                     for ti in range(13) if ai != ti}
                     for ai in range(13)},
            'std': {DIM_NAMES[ai]: {DIM_NAMES[ti]: round(float(std_n[ai,ti]), 4)
                    for ti in range(13) if ai != ti}
                    for ai in range(13)},
            'significant': [{'ablated': a, 'affected': t, 'mean': round(m,4), 'std': round(s,4)}
                           for a, t, m, s in sig_n],
        },
        's2_interact': {
            'mean': {DIM_NAMES[ai]: {DIM_NAMES[ti]: round(float(mean_i[ai,ti]), 4)
                     for ti in range(13) if ai != ti}
                     for ai in range(13)},
            'std': {DIM_NAMES[ai]: {DIM_NAMES[ti]: round(float(std_i[ai,ti]), 4)
                    for ti in range(13) if ai != ti}
                    for ai in range(13)},
            'significant': [{'ablated': a, 'affected': t, 'mean': round(m,4), 'std': round(s,4)}
                           for a, t, m, s in sig_i],
        },
    }
    
    out_path = os.path.join(args.base, 'exp2c_aggregate.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved: {out_path}")


if __name__ == '__main__':
    main()
