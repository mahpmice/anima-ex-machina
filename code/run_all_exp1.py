"""
Batch exp1_reverse evaluation + aggregation
Directory structure:
  {base}/unidir_0pct/results_v4_ph5_s{seed}/model.pt
  {base}/bidir_15pct/results_v4_ph5_bidir15_s{seed}/model.pt
  {base}/bidir_30pct/results_v4_ph5_bidir30_s{seed}/model.pt

Usage (run in code directory):
  python3 run_all_exp1.py --base ../results/bidir_comparison --world swadesh_v6_world.md
  python3 run_all_exp1.py --base ../results/bidir_comparison --aggregate-only
"""

import os
import sys
import json
import argparse
import subprocess
import time
import numpy as np
from pathlib import Path


SEEDS = [42, 137, 271, 314, 577]

# Directory name -> (ratio label, subdir prefix)
RATIO_MAP = [
    (0.0,  'unidir_0pct',  'results_v4_ph5_s'),
    (0.15, 'bidir_15pct',  'results_v4_ph5_bidir15_s'),
    (0.30, 'bidir_30pct',  'results_v4_ph5_bidir30_s'),
]


def find_checkpoints(base):
    """Scan directory, return list of (ratio, seed, checkpoint_path, result_json_path)"""
    found = []
    for ratio, ratio_dir, prefix in RATIO_MAP:
        ratio_path = os.path.join(base, ratio_dir)
        if not os.path.isdir(ratio_path):
            print(f"  [SKIP] Directory does not exist: {ratio_path}")
            continue
        for seed in SEEDS:
            seed_dir = os.path.join(ratio_path, f"{prefix}{seed}")
            # Try two checkpoint naming conventions
            for ckpt_name in ['model_ph5.pt', 'model.pt']:
                ckpt = os.path.join(seed_dir, ckpt_name)
                if os.path.isfile(ckpt):
                    json_path = os.path.join(seed_dir, 'exp1_reverse_results.json')
                    found.append((ratio, seed, ckpt, json_path))
                    break
            else:
                print(f"  [MISS] No checkpoint: {seed_dir}/")
    return found


def run_eval(checkpoint, world, code_dir):
    """Run on single checkpointexp1_reverse.py"""
    cmd = [
        sys.executable,
        os.path.join(code_dir, 'exp1_reverse.py'),
        '--checkpoint', checkpoint,
        '--world', os.path.join(code_dir, world),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=code_dir)
    return result.returncode == 0, result.stderr[-300:] if result.stderr else ''


def aggregate(all_data):
    """
    all_data: list of (ratio, seed, results_dict)
    Return: Grouped by ratio mean±std
    """
    by_ratio = {}
    for ratio, seed, data in all_data:
        if data is None:
            continue
        by_ratio.setdefault(ratio, []).append((seed, data))

    summary = {}
    for ratio in sorted(by_ratio.keys()):
        runs = by_ratio[ratio]
        rkey = f"{int(ratio*100)}pct"
        entry = {
            'ratio': ratio,
            'n_seeds': len(runs),
            'seeds': [s for s, _ in runs],
            'metrics': {},
            'per_dim': {},
        }

        # ── Overall metrics ──
        metric_keys = [
            ('A_train',     'A_reverse'),
            ('B_novel',     'B_reverse'),
            ('C_mismatch',  'C_reverse'),
            ('D_fictional', 'D_reverse'),
            ('E_interact',  'E_reverse'),
            ('F_probes',    'F_reverse'),
            ('forward_A',   'A_forward'),
            ('forward_E',   'E_forward'),
        ]

        for json_key, label in metric_keys:
            vals = []
            for seed, data in runs:
                if json_key in data:
                    v = data[json_key]
                    if isinstance(v, dict):
                        v = v.get('overall', None)
                    if v is not None:
                        vals.append(float(v))
            if vals:
                arr = np.array(vals)
                entry['metrics'][label] = {
                    'mean': round(float(arr.mean()), 4),
                    'std':  round(float(arr.std()), 4),
                    'min':  round(float(arr.min()), 4),
                    'max':  round(float(arr.max()), 4),
                    'n':    len(vals),
                    'values': [round(v, 4) for v in vals],
                }

        # ── per-dim（Agroup andEGroup reverse） ──
        dims = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
        for group_key, group_label in [('A_train', 'A'), ('E_interact', 'E')]:
            dim_data = {d: [] for d in dims}
            for seed, data in runs:
                if group_key in data and isinstance(data[group_key], dict):
                    pd = data[group_key].get('per_dim', {})
                    for d in dims:
                        if d in pd and pd[d] is not None:
                            dim_data[d].append(float(pd[d]))
            dim_stats = {}
            for d in dims:
                if dim_data[d]:
                    arr = np.array(dim_data[d])
                    dim_stats[d] = {
                        'mean': round(float(arr.mean()), 4),
                        'std':  round(float(arr.std()), 4),
                    }
            if dim_stats:
                entry['per_dim'][f'{group_label}_reverse'] = dim_stats

        summary[rkey] = entry

    return summary


def print_table5(summary):
    """Print Table 5 format"""
    print(f"\n{'='*80}")
    print(f"  TABLE 5 · Resource Competition Across Reverse Ratios")
    print(f"{'='*80}")

    ratios = sorted(summary.keys(), key=lambda k: summary[k]['ratio'])

    # Header
    print(f"\n  {'Metric':<28s}", end='')
    for rk in ratios:
        r = summary[rk]
        print(f"  {int(r['ratio']*100):>2d}% (n={r['n_seeds']})     ", end='')
    print()
    print(f"  {'─'*76}")

    rows = [
        ('A_reverse',  'A reverse (name)'),
        ('B_reverse',  'B reverse (novel)'),
        ('C_reverse',  'C reverse (mismatch)'),
        ('D_reverse',  'D reverse (fictional)'),
        ('E_reverse',  'E reverse (interact)'),
        ('F_reverse',  'F reverse (probes)'),
        ('A_forward',  'A forward (name)'),
        ('E_forward',  'E forward (interact)'),
    ]

    for key, label in rows:
        print(f"  {label:<28s}", end='')
        for rk in ratios:
            m = summary[rk]['metrics'].get(key, None)
            if m:
                print(f"  {m['mean']:.3f} ± {m['std']:.3f}    ", end='')
            else:
                print(f"  {'—':>16s}    ", end='')
        print()

    print(f"  {'─'*76}")


def print_perdim(summary):
    """Print per-dim Details"""
    dims = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    ratios = sorted(summary.keys(), key=lambda k: summary[k]['ratio'])

    for group in ['A_reverse', 'E_reverse']:
        print(f"\n  Per-dim: {group}")
        print(f"  {'Dim':<6s}", end='')
        for rk in ratios:
            print(f"  {int(summary[rk]['ratio']*100):>2d}%           ", end='')
        print()
        print(f"  {'─'*60}")

        for d in dims:
            print(f"  {d:<6s}", end='')
            for rk in ratios:
                pd = summary[rk].get('per_dim', {}).get(group, {}).get(d, None)
                if pd:
                    print(f"  {pd['mean']:.3f} ± {pd['std']:.3f}  ", end='')
                else:
                    print(f"  {'—':>14s}  ", end='')
            print()


def main():
    p = argparse.ArgumentParser(description='Batchexp1evaluation+Summary')
    p.add_argument('--base', required=True,
                   help='results root dir (Include unidir_0pct/ bidir_15pct/ bidir_30pct/)')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--aggregate-only', action='store_true',
                   help='OnlySummaryexistingJSON，notrunevaluation')
    p.add_argument('--code-dir', default='.',
                   help='code directory（exp1_reverse.pyat position）')
    args = p.parse_args()

    base = os.path.abspath(args.base)
    code_dir = os.path.abspath(args.code_dir)

    print(f"{'='*60}")
    print(f"  Batch exp1_reverse evaluation")
    print(f"{'='*60}")
    print(f"  Base: {base}")
    print(f"  Code: {code_dir}")

    # ── scan ──
    entries = find_checkpoints(base)
    print(f"\n  Found {len(entries)} checkpoint")
    for ratio, seed, ckpt, _ in entries:
        print(f"    ratio={ratio:.2f}  seed={seed}  {os.path.basename(os.path.dirname(ckpt))}/")

    if not args.aggregate_only:
        # ── per-evaluation ──
        print(f"\n  Startevaluation...")
        t0 = time.time()
        for i, (ratio, seed, ckpt, json_path) in enumerate(entries):
            if os.path.exists(json_path):
                print(f"  [{i+1}/{len(entries)}] ratio={ratio:.2f} seed={seed} — existingresults，Skip")
                continue
            print(f"  [{i+1}/{len(entries)}] ratio={ratio:.2f} seed={seed} — evaluationin...")
            ok, err = run_eval(ckpt, args.world, code_dir)
            if ok:
                print(f"    ✓ Done")
            else:
                print(f"    ✗ Failed: {err}")
        elapsed = time.time() - t0
        print(f"\n  evaluationDone ({elapsed:.0f}s)")

    # ── collect results ──
    all_data = []
    for ratio, seed, ckpt, json_path in entries:
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            all_data.append((ratio, seed, data))
        else:
            print(f"  [MISS] no results: ratio={ratio:.2f} seed={seed}")
            all_data.append((ratio, seed, None))

    n_ok = sum(1 for _, _, d in all_data if d is not None)
    print(f"\n  Collected {n_ok}/{len(entries)} group results")

    # ── Summary ──
    summary = aggregate(all_data)
    print_table5(summary)
    print_perdim(summary)

    # ── Save ──
    out_path = os.path.join(base, 'table5_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {out_path}")

    # fullper-seeddata
    full_path = os.path.join(base, 'table5_full.json')
    full = [{'ratio': r, 'seed': s, 'results': d} for r, s, d in all_data]
    with open(full_path, 'w') as f:
        json.dump(full, f, indent=2, default=str)
    print(f"  Full data: {full_path}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
