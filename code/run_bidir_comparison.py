"""
Resource Competition Experiment · Table 5

Auto-scan all models under bidir_comparison/,
run forward (P→desc) and reverse (desc→P) evaluation for each model,
output Table 5: Resource competition across formats.

Usage:
  python run_bidir_comparison.py --dir path/to/bidir_comparison
"""

import torch
import numpy as np
import json, os, argparse, glob, re

from model import make_model, count_params
from v6_tool import parse_world, generate, build_tokens, FMT_META
from train_core import pretensorize, eval_desc_acc
from exp1_reverse import (
    build_test_group_A, build_test_group_E, build_test_group_F,
    build_test_group_B, build_test_group_D,
    eval_reverse_p,
)

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')


def find_models(base_dir):
    """
    Auto-discover all model.pt under base_dir.
    Returns list of (ratio_label, seed, model_path)
    """
    models = []
    for root, dirs, files in os.walk(base_dir):
        if 'model.pt' in files:
            path = os.path.join(root, 'model.pt')
            rel = os.path.relpath(root, base_dir)
            parts = rel.replace('\\', '/').split('/')

            # Extract ratio from parent directory name
            ratio_label = None
            for p in parts:
                p_low = p.lower()
                if 'unidir' in p_low or '_0pct' in p_low:
                    ratio_label = '0%'
                    break
                elif '_15pct' in p_low or 'bidir15' in p_low:
                    ratio_label = '15%'
                    break
                elif '_30pct' in p_low or 'bidir30' in p_low:
                    ratio_label = '30%'
                    break

            # Extract seed
            seed = None
            for p in parts:
                m = re.search(r'seed[_]?(\d+)', p.lower())
                if m:
                    seed = int(m.group(1))
            # If no seed subdirectory, try from filename
            if seed is None:
                m = re.search(r'seed[_]?(\d+)', rel.lower())
                if m:
                    seed = int(m.group(1))

            if ratio_label is None:
                ratio_label = rel  # fallback

            models.append((ratio_label, seed, path))

    return sorted(models)


def evaluate_one(model, cfg, t2id, pad_id, mask_id):
    """Run forward+reverse on one model, return 6 metrics for Table 5"""
    results = {}

    # ── Group A: trained entities, name format ──
    seqs_a, _ = build_test_group_A(cfg, t2id)
    tensor_a = pretensorize(seqs_a, pad_id)

    # A forward (mask desc, predict from P)
    ds_name = FMT_META['s2_name'][1]
    fwd_a, _ = eval_desc_acc(model, tensor_a, DEVICE, pad_id, mask_id, desc_start=ds_name)
    results['A_forward'] = round(fwd_a, 4)

    # A reverse (mask P, predict from desc)
    rev_a, _, _ = eval_reverse_p(model, tensor_a, DEVICE, pad_id, mask_id, p_start=3)
    results['A_reverse'] = round(rev_a, 4)

    # ── Group E: interaction results, interact format ──
    seqs_e, _ = build_test_group_E(cfg, t2id)
    tensor_e = pretensorize(seqs_e, pad_id)

    # E forward
    ds_inter = FMT_META['s2_interact'][1]
    fwd_e, _ = eval_desc_acc(model, tensor_e, DEVICE, pad_id, mask_id, desc_start=ds_inter)
    results['E_forward'] = round(fwd_e, 4)

    # E reverse
    rev_e, _, _ = eval_reverse_p(model, tensor_e, DEVICE, pad_id, mask_id, p_start=9)
    results['E_reverse'] = round(rev_e, 4)

    # ── Novel per-dim ──
    seqs_b, _ = build_test_group_B(cfg, t2id)
    if seqs_b:
        tensor_b = pretensorize(seqs_b, pad_id)
        fwd_b, fwd_b_dim = eval_desc_acc(model, tensor_b, DEVICE, pad_id, mask_id,
                                          desc_start=ds_name)
        novel_dims = sum(1 for v in fwd_b_dim.values() if v >= 0.99)
        results['novel_dims'] = novel_dims
        results['novel_overall'] = round(fwd_b, 4)

    return results


def run(base_dir, world_path, d_model=64):
    print(f"{'═'*70}")
    print(f"  Resource Competition · Table 5")
    print(f"  Dir: {base_dir}")
    print(f"  Device: {DEVICE}")
    print(f"{'═'*70}")

    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    cfg = data['cfg']
    tokens = build_tokens(cfg)
    t2id = {t: i for i, t in enumerate(tokens)}
    pad_id = t2id['PAD']
    mask_id = t2id['MASK']

    models = find_models(base_dir)
    if not models:
        print(f"  ⚠ No model.pt found")
        return

    print(f"\n  Found {len(models)} models:")
    for ratio, seed, path in models:
        rel = os.path.relpath(path, base_dir)
        print(f"    {ratio:6s}  seed={seed}  {rel}")

    all_results = []

    for ratio, seed, path in models:
        print(f"\n{'━'*50}")
        print(f"  {ratio} · seed={seed}")

        model = make_model(len(tokens), d=d_model, n_layers=4, n_heads=4,
                           max_len=48, dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()

        res = evaluate_one(model, cfg, t2id, pad_id, mask_id)
        res['ratio'] = ratio
        res['seed'] = seed
        all_results.append(res)

        print(f"    A fwd={res['A_forward']}  rev={res['A_reverse']}  "
              f"E fwd={res['E_forward']}  rev={res['E_reverse']}  "
              f"novel={res.get('novel_dims', '?')}/13")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Table 5 Summary ──
    print(f"\n{'═'*70}")
    print(f"  Table 5: Resource competition across formats")
    print(f"{'═'*70}\n")

    ratios = sorted(set(r['ratio'] for r in all_results),
                    key=lambda x: float(x.replace('%', '')) if '%' in x else 999)

    metrics = ['A_reverse', 'E_reverse', 'A_forward', 'E_forward', 'novel_dims']
    display = {
        'A_reverse': 'A reverse (name)',
        'E_reverse': 'E reverse (interact)',
        'A_forward': 'A forward (name)',
        'E_forward': 'E forward (interact)',
        'novel_dims': 'Novel (/13)',
    }

    print(f"  {'Metric':<25s}", end='')
    for ratio in ratios:
        print(f"  {ratio:>15s}", end='')
    print()
    print(f"  {'─'*70}")

    for metric in metrics:
        print(f"  {display[metric]:<25s}", end='')
        for ratio in ratios:
            vals = [r[metric] for r in all_results
                    if r['ratio'] == ratio and metric in r]
            if not vals:
                print(f"  {'—':>15s}", end='')
            elif len(vals) == 1:
                print(f"  {vals[0]:>15.3f}", end='')
            else:
                arr = np.array(vals)
                print(f"  {arr.mean():.3f}±{arr.std():.3f}".rjust(15), end='')
        print()

    # Save results
    out_path = os.path.join(base_dir, 'bidir_comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{'═'*70}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='anima_ex_machina/bidir_comparison')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--d', type=int, default=64)
    args = p.parse_args()
    run(args.dir, args.world, d_model=args.d)
