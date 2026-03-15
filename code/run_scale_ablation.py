"""
Model scale ablation


Four scales x Phase 5 forward-only（reverse_ratio=0）
Report S2.4 Interact、Probes、Novel per-dim

Usage:
  python run_scale_ablation.py                     # all four
  python run_scale_ablation.py --size xs           # Only runXS
  python run_scale_ablation.py --size xs s         # runXS and S
  python run_scale_ablation.py --seeds 42 123 456  # 3-seed（defaultsingleseed）
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, argparse, time

from model import make_model, count_params
from v6_tool import parse_world, generate, FMT_META
from train_core import (
    pretensorize, build_tagged_batches, train_epoch,
    eval_p_recon, eval_desc_acc, eval_full_mask, per_dimension_probe,
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

MAX_PHASE = 5
REVERSE_RATIO = 0.3  # Aligned with main model

# ── Scale definitions ──
SCALE_CONFIGS = {
    'xs': {'d': 32,  'n_layers': 2, 'n_heads': 4},
    's':  {'d': 64,  'n_layers': 4, 'n_heads': 4},
    'm':  {'d': 128, 'n_layers': 4, 'n_heads': 4},
    'l':  {'d': 256, 'n_layers': 4, 'n_heads': 4},
}


def train_one(seed, data, t, pad_id, mask_id, holdout_dims,
              d, n_layers, n_heads, bs=256, use_fp16=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model(data['vocab_size'], d=d, n_layers=n_layers,
                       n_heads=n_heads, max_len=48, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    params = count_params(model)
    print(f"    Model: d={d} layers={n_layers} heads={n_heads} → {params:,} params")

    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    mr = 0.30
    rng = np.random.default_rng(seed)
    total_step = 0

    for phase_num in range(1, MAX_PHASE + 1):
        pdef = phases[phase_num]
        max_steps = pdef['max_steps']
        stage_data = pdef['data']()

        step, epoch = 0, 0
        while step < max_steps:
            epoch += 1
            batches = build_tagged_batches(stage_data, bs, rng)
            steps_ep, avg_loss = train_epoch(
                model, opt, batches, DEVICE, pad_id, mask_id, mr,
                holdout_dims=holdout_dims, use_fp16=use_fp16,
                reverse_ratio=REVERSE_RATIO)
            step += steps_ep

            eval_acc = pdef['quick_eval'](model)

            if epoch % 10 == 0 or step >= max_steps:
                print(f"      ph{phase_num} ep{epoch:3d}  step={step:5d}  "
                      f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

            if avg_loss < 0.25 and eval_acc >= 0.95:
                print(f"    ✓ ph{phase_num} skip-step")
                break
            model.train()

        total_step += step

    return model, params, total_step


def evaluate(model, data, t, pad_id, mask_id):
    results = {}

    # S1
    p_acc, _ = eval_p_recon(model, t['s1_bare'][:200], DEVICE, pad_id, mask_id)
    results['s1_p_recon'] = round(p_acc, 4)

    # S2.1 name
    ds_n = FMT_META['s2_name'][1]
    n_acc, _ = eval_desc_acc(model, t['s2_name'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_n)
    results['s2_name'] = round(n_acc, 4)

    # S2.4 interact
    ds_i = FMT_META['s2_interact'][1]
    i_acc, i_dim = eval_desc_acc(model, t['s2_interact'][:200], DEVICE, pad_id, mask_id,
                                  desc_start=ds_i)
    results['s2_interact'] = round(i_acc, 4)
    results['s2_interact_per_dim'] = i_dim

    # Probes
    if t['probe_s2'].shape[0] > 0:
        pr_acc, _ = eval_desc_acc(model, t['probe_s2'], DEVICE, pad_id, mask_id,
                                   desc_start=ds_i)
        results['probes'] = round(pr_acc, 4)

    # Novel per-dim
    novel_data = data.get('novel', {})
    if novel_data:
        novel_scores = []
        for name, ndata in novel_data.items():
            ns2 = pretensorize([ndata['s2_name']], pad_id)
            na, nd = eval_desc_acc(model, ns2, DEVICE, pad_id, mask_id,
                                    desc_start=ds_n)
            correct_dims = sum(1 for v in nd.values() if v >= 0.99)
            novel_scores.append(correct_dims)
        results['novel_avg_dims'] = round(np.mean(novel_scores), 2)
        results['novel_per_entity'] = {name: sum(1 for v in nd.values() if v >= 0.99)
                                        for name, ndata in novel_data.items()
                                        for na, nd in [eval_desc_acc(
                                            model, pretensorize([ndata['s2_name']], pad_id),
                                            DEVICE, pad_id, mask_id, desc_start=ds_n)]}

    return results


def run(world_path, sizes, seeds, bs=256, use_fp16=True):
    t_start = time.time()

    print(f"{'═'*60}")
    print(f"  Model scale ablation")
    print(f"  Sizes: {sizes}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {DEVICE}  reverse={REVERSE_RATIO}")
    print(f"{'═'*60}")

    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    holdout_dims = data['holdout_dims']

    t = {
        's1_bare':       pretensorize(data['s1']['bare'], pad_id),
        's1_un':         pretensorize(data['s1']['un'], pad_id),
        's1_inter':      pretensorize(data['s1']['interact'], pad_id),
        's2_name':       pretensorize(data['s2']['name'], pad_id),
        's2_name_eidx':  torch.tensor(data['s2']['name_eidx'], dtype=torch.long),
        's2_gender':     pretensorize(data['s2']['gender'], pad_id),
        's2_gender_eidx':torch.tensor(data['s2']['gender_eidx'], dtype=torch.long),
        's2_un':         pretensorize(data['s2']['un'], pad_id),
        's2_il':         pretensorize(data['s2']['il'], pad_id),
        's2_un_held':    pretensorize(data['s2']['un_held'], pad_id),
        's2_interact':   pretensorize(data['s2']['interact'], pad_id),
        's2_auto':       pretensorize(data['s2']['auto'], pad_id),
        's3_entity':     pretensorize(data['s3']['entity'], pad_id),
        's3_entity_eidx':torch.tensor(data['s3']['entity_eidx'], dtype=torch.long),
        's3_interact':   pretensorize(data['s3']['interact'], pad_id),
        'probe_s2':      pretensorize(data['probes']['s2'], pad_id),
        'probe_s3':      pretensorize(data['probes']['s3'], pad_id),
        'anim_s3':       pretensorize(data['animacy_probes']['s3'], pad_id),
    }

    all_results = {}

    for size in sizes:
        cfg_m = SCALE_CONFIGS[size]
        size_results = {}

        for seed in seeds:
            print(f"\n{'━'*60}")
            print(f"  {size.upper()} · seed={seed}")
            print(f"{'━'*60}")

            seed_t0 = time.time()
            rd = f"results_scale/{size}_seed_{seed}"
            os.makedirs(rd, exist_ok=True)
            model_path = os.path.join(rd, 'model.pt')

            if os.path.exists(model_path):
                print(f"    Checkpoint exists，Skip training")
                model = make_model(data['vocab_size'], d=cfg_m['d'],
                                   n_layers=cfg_m['n_layers'],
                                   n_heads=cfg_m['n_heads'],
                                   max_len=48, dropout=0.0).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                                 weights_only=True))
                model.eval()
                params = count_params(model)
                total_steps = 0
            else:
                model, params, total_steps = train_one(
                    seed, data, t, pad_id, mask_id, holdout_dims,
                    d=cfg_m['d'], n_layers=cfg_m['n_layers'],
                    n_heads=cfg_m['n_heads'], bs=bs, use_fp16=use_fp16)
                torch.save(model.state_dict(), model_path)

            # evaluation
            eval_res = evaluate(model, data, t, pad_id, mask_id)
            print(f"    S1={eval_res['s1_p_recon']}  S2.1={eval_res['s2_name']}  "
                  f"S2.4={eval_res['s2_interact']}  Probe={eval_res.get('probes', 'N/A')}  "
                  f"Novel={eval_res.get('novel_avg_dims', 'N/A')}/13")

            seed_result = {
                'size': size, 'seed': seed, 'params': params,
                'd': cfg_m['d'], 'n_layers': cfg_m['n_layers'],
                'n_heads': cfg_m['n_heads'],
                'total_steps': total_steps,
                'elapsed_sec': round(time.time() - seed_t0, 1),
                'evaluation': eval_res,
            }
            with open(os.path.join(rd, 'results.json'), 'w') as f:
                json.dump(seed_result, f, indent=2, default=str)

            size_results[seed] = seed_result

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[size] = size_results

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")

    header = f"  {'Size':<6s}  {'Params':>10s}  {'S1':>7s}  {'S2.1':>7s}  {'S2.4':>7s}  {'Probe':>7s}  {'Novel':>7s}"
    print(header)
    print(f"  {'─'*62}")

    for size, size_res in all_results.items():
        s1_vals, s21_vals, s24_vals, pr_vals, nv_vals = [], [], [], [], []
        params = 0
        for seed, r in size_res.items():
            ev = r['evaluation']
            params = r['params']
            s1_vals.append(ev['s1_p_recon'])
            s21_vals.append(ev['s2_name'])
            s24_vals.append(ev['s2_interact'])
            pr_vals.append(ev.get('probes', 0))
            nv_vals.append(ev.get('novel_avg_dims', 0))

        def fmt(arr):
            a = np.array(arr)
            if len(a) == 1:
                return f"{a[0]:.3f}"
            return f"{a.mean():.3f}±{a.std():.3f}"

        print(f"  {size.upper():<6s}  {params:>10,d}  {fmt(s1_vals):>7s}  "
              f"{fmt(s21_vals):>7s}  {fmt(s24_vals):>7s}  {fmt(pr_vals):>7s}  "
              f"{fmt(nv_vals):>7s}")

    # Save
    os.makedirs('results_scale', exist_ok=True)
    with open('results_scale/summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Total: {time.time() - t_start:.0f}s")
    print(f"  Saved: results_scale/")
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Model scale ablation')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--size', nargs='+', default=None,
                   choices=['xs', 's', 'm', 'l'],
                   help='Which scales to run（default all）')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    args = p.parse_args()

    sizes = args.size if args.size else ['xs', 's', 'm', 'l']
    run(args.world, sizes, args.seeds, bs=args.bs, use_fp16=not args.no_fp16)
