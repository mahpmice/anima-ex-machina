"""
Multi-seed training + evaluation + aggregation


Usage:
  python3 run_multiseed.py --phase 5 --workers 4 --seeds 5

  Default config:
    reverse ratios: 0.0, 0.15, 0.30
    seeds: 42, 137, 271, 314, 577 (first N)
    workers: 4 (parallel workers)

  After running, autoSummary，output multiseed_summary.json
"""

import subprocess
import sys
import os
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SEED_POOL = [42, 137, 271, 314, 577]
RATIOS = [0.0, 0.15, 0.30]


def run_one(phase, world, bs, fp16, ratio, seed):
    """singletrain + evaluation。Return (ratio, seed, results_dict_or_None)"""
    ratio_tag = f"_bidir{int(ratio*100)}" if ratio > 0 else ""
    rd = f"results_v4_ph{phase}{ratio_tag}_s{seed}"
    ckpt = os.path.join(rd, f"model_ph{phase}.pt")

    # ── train ──
    cmd_train = [
        sys.executable, 'train_v6.py',
        '--phase', str(phase),
        '--world', world,
        '--bs', str(bs),
        '--reverse-ratio', str(ratio),
        '--seed', str(seed),
    ]
    if not fp16:
        cmd_train.append('--no-fp16')

    print(f"  [START] ratio={ratio} seed={seed}")
    t0 = time.time()

    result = subprocess.run(cmd_train, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] ratio={ratio} seed={seed} Training failed")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return (ratio, seed, None)

    # ── evaluation ──
    cmd_eval = [
        sys.executable, 'exp1_reverse.py',
        '--checkpoint', ckpt,
        '--world', world,
    ]
    result_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
    if result_eval.returncode != 0:
        print(f"  [FAIL] ratio={ratio} seed={seed} Evaluation failed")
        print(result_eval.stderr[-500:] if result_eval.stderr else "no stderr")
        return (ratio, seed, None)

    elapsed = time.time() - t0
    print(f"  [DONE] ratio={ratio} seed={seed} ({elapsed:.0f}s)")

    # ── Read results ──
    json_path = os.path.join(rd, 'exp1_reverse_results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return (ratio, seed, data)
    else:
        return (ratio, seed, None)


def aggregate(all_results):
    """Summarymulti-seedresults，Calculate mean±std"""
    import numpy as np

    # byratiogrouping
    by_ratio = {}
    for ratio, seed, data in all_results:
        if data is None:
            continue
        if ratio not in by_ratio:
            by_ratio[ratio] = []
        by_ratio[ratio].append((seed, data))

    summary = {}
    for ratio in sorted(by_ratio.keys()):
        runs = by_ratio[ratio]
        ratio_key = f"ratio_{int(ratio*100)}"
        summary[ratio_key] = {
            'n_seeds': len(runs),
            'seeds': [s for s, _ in runs],
        }

        # Collect key metrics
        metrics = {}
        for seed, data in runs:
            for group_key in ['A_train', 'B_novel', 'C_mismatch',
                              'D_fictional', 'E_interact', 'F_probes']:
                if group_key in data:
                    overall = data[group_key].get('overall', None)
                    if overall is not None:
                        if group_key not in metrics:
                            metrics[group_key] = []
                        metrics[group_key].append(overall)

            # Forward control
            for fwd_key in ['forward_A', 'forward_E']:
                if fwd_key in data:
                    val = data[fwd_key]
                    if val is not None:
                        if fwd_key not in metrics:
                            metrics[fwd_key] = []
                        metrics[fwd_key].append(val)

        # Calculate statistics
        stats = {}
        for key, vals in metrics.items():
            arr = np.array(vals)
            stats[key] = {
                'mean': round(float(arr.mean()), 4),
                'std': round(float(arr.std()), 4),
                'min': round(float(arr.min()), 4),
                'max': round(float(arr.max()), 4),
                'n': len(vals),
                'values': [round(v, 4) for v in vals],
            }
        summary[ratio_key]['metrics'] = stats

    return summary


def print_summary(summary):
    """Print paper-ready summary table"""
    print(f"\n{'='*80}")
    print(f"  MULTI-SEED SUMMARY")
    print(f"{'='*80}")

    # Collect all ratios
    ratios = sorted(summary.keys())

    # Print header
    header_ratios = [f"{r}" for r in ratios]
    print(f"\n  {'Metric':<25s}", end='')
    for r in ratios:
        n = summary[r]['n_seeds']
        print(f"  {r} (n={n})       ", end='')
    print()
    print(f"  {'─'*75}")

    # Metric rows
    all_keys = set()
    for r in ratios:
        all_keys.update(summary[r].get('metrics', {}).keys())

    display_order = [
        ('A_train', 'A · reverse (name)'),
        ('B_novel', 'B · reverse (novel)'),
        ('C_mismatch', 'C · reverse (mismatch)'),
        ('D_fictional', 'D · reverse (fictional)'),
        ('E_interact', 'E · reverse (interact)'),
        ('F_probes', 'F · reverse (probes)'),
        ('forward_A', 'A · forward (name)'),
        ('forward_E', 'E · forward (interact)'),
    ]

    for key, label in display_order:
        print(f"  {label:<25s}", end='')
        for r in ratios:
            m = summary[r].get('metrics', {}).get(key, None)
            if m:
                print(f"  {m['mean']:.3f} ± {m['std']:.3f}     ", end='')
            else:
                print(f"  {'—':>15s}     ", end='')
        print()

    print(f"  {'─'*75}")


def main():
    p = argparse.ArgumentParser(description='Multi-seedtrain+evaluation')
    p.add_argument('--phase', type=int, default=5, choices=[1, 2, 3, 4, 5, 6])
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--seeds', type=int, default=5,
                   help='useusefirst Nseed (from42,137,271,314,577in)')
    p.add_argument('--workers', type=int, default=4,
                   help='parallel workers')
    p.add_argument('--ratios', type=str, default='0.0,0.15,0.30',
                   help='comma-separatedreverse ratiolist')
    p.add_argument('--aggregate-only', action='store_true',
                   help='Only summarize existing results, no training')
    args = p.parse_args()

    seeds = SEED_POOL[:args.seeds]
    ratios = [float(x) for x in args.ratios.split(',')]
    fp16 = not args.no_fp16

    print(f"{'='*60}")
    print(f"  Multi-seed train + evaluation")
    print(f"{'='*60}")
    print(f"  Phase: {args.phase}")
    print(f"  Ratios: {ratios}")
    print(f"  Seeds: {seeds}")
    print(f"  Workers: {args.workers}")
    print(f"  Total runs: {len(ratios) * len(seeds)}")
    print(f"{'='*60}")

    if args.aggregate_only:
        # Read existing results only
        all_results = []
        for ratio in ratios:
            for seed in seeds:
                ratio_tag = f"_bidir{int(ratio*100)}" if ratio > 0 else ""
                rd = f"results_v4_ph{args.phase}{ratio_tag}_s{seed}"
                json_path = os.path.join(rd, 'exp1_reverse_results.json')
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                    all_results.append((ratio, seed, data))
                    print(f"  [FOUND] ratio={ratio} seed={seed}")
                else:
                    print(f"  [MISS]  ratio={ratio} seed={seed}")
                    all_results.append((ratio, seed, None))
    else:
        # Build task list
        tasks = []
        for ratio in ratios:
            for seed in seeds:
                tasks.append((args.phase, args.world, args.bs, fp16, ratio, seed))

        t0 = time.time()

        # Parallel execution
        all_results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, *task): task
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

        elapsed = time.time() - t0
        print(f"\n  Total elapsed: {elapsed:.0f}s")

    # ── Summary ──
    summary = aggregate(all_results)
    print_summary(summary)

    # Save
    out_path = 'multiseed_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {out_path}")

    # Also save fullper-seeddata
    full_path = 'multiseed_full.json'
    full_data = []
    for ratio, seed, data in all_results:
        full_data.append({
            'ratio': ratio, 'seed': seed,
            'results': data,
        })
    with open(full_path, 'w') as f:
        json.dump(full_data, f, indent=2, default=str)
    print(f"  Full dataexistingSave: {full_path}")


if __name__ == '__main__':
    main()
