"""
Experiment 2B Supplement · Shuffle Permutation Coverage
2026-03-15

Usage (parallel in 3 terminals):

Terminal 1:
  python3 exp2b_shuffle_supp.py --order 1,2,3,5,4 --seed 42
  python3 exp2b_shuffle_supp.py --order 1,2,3,5,4 --seed 123
  python3 exp2b_shuffle_supp.py --order 1,2,3,5,4 --seed 456

Terminal 2:
  python3 exp2b_shuffle_supp.py --order 3,1,2,4,5 --seed 42
  python3 exp2b_shuffle_supp.py --order 3,1,2,4,5 --seed 123
  python3 exp2b_shuffle_supp.py --order 3,1,2,4,5 --seed 456

Terminal 3:
  python3 exp2b_shuffle_supp.py --order 5,2,4,1,3 --seed 42
  python3 exp2b_shuffle_supp.py --order 5,2,4,1,3 --seed 123
  python3 exp2b_shuffle_supp.py --order 5,2,4,1,3 --seed 456

After completion:
  python3 exp2b_shuffle_summary.py
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, argparse, time

from model import make_model, count_params
from v6_tool import parse_world, generate, FMT_META
from train_core import (
    pretensorize, build_tagged_batches, train_epoch,
    eval_p_recon, eval_desc_acc, per_dimension_probe,
    repeat_tensor, repeat_eidx,
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

MAX_PHASE = 5
REVERSE_RATIO = 0.3


def run_shuffle(world_path, phase_order, seed=42, bs=256, use_fp16=True, d_model=64):
    t0 = time.time()
    order_str = ''.join(str(p) for p in phase_order)
    label = f"shuf_{order_str}_s{seed}"

    print(f"\n{'═'*60}")
    print(f"  Shuffle Supplement · {label}")
    print(f"  order={phase_order}  seed={seed}  reverse={REVERSE_RATIO}")
    print(f"  Device: {DEVICE}  bs={bs}  fp16={use_fp16}")
    print(f"{'═'*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    t2id = data['tokens2id']
    ee = data['cfg']['entity_encodings']
    train_names = data['cfg']['train_names']
    te = [(n, ee[n]) for n in train_names]
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

    model = make_model(data['vocab_size'], d=d_model, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    print(f"Model: {count_params(model):,} params")

    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    mr = 0.30
    rng = np.random.default_rng(seed)
    total_step = 0
    all_history = {}

    for phase_num in phase_order:
        pdef = phases[phase_num]
        pname = pdef['name']
        max_steps = pdef['max_steps']
        stage_data = pdef['data']()

        print(f"\n  Phase {phase_num}: {pname}")
        total_seqs = sum(td[1].shape[0] for td in stage_data)
        print(f"  Sequences: {total_seqs}  max_steps={max_steps}")

        step, history, epoch = 0, [], 0
        while step < max_steps:
            epoch += 1
            batches = build_tagged_batches(stage_data, bs, rng)
            steps_ep, avg_loss = train_epoch(
                model, opt, batches, DEVICE, pad_id, mask_id, mr,
                holdout_dims=holdout_dims, use_fp16=use_fp16,
                reverse_ratio=REVERSE_RATIO)
            step += steps_ep

            eval_acc = pdef['quick_eval'](model)
            history.append(dict(epoch=epoch, step=step,
                                loss=round(avg_loss, 5),
                                acc=round(eval_acc, 4)))

            if epoch % 5 == 0 or step >= max_steps:
                print(f"    ep{epoch:3d}  step={step:5d}  "
                      f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

            if avg_loss < 0.25 and eval_acc >= 0.95:
                print(f"  ✓ Early stop: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
                break
            model.train()

        total_step += step
        all_history[f'phase_{phase_num}'] = history

    # ═══ Evaluation ═══
    print(f"\n{'='*60}")
    print(f"EVAL · {label}")
    print(f"{'='*60}")

    eval_results = {}

    p_acc, _ = eval_p_recon(model, t['s1_bare'][:200], DEVICE, pad_id, mask_id)
    eval_results['s1_p_recon'] = {'overall': round(p_acc, 4)}
    print(f"  S1: {p_acc:.3f}")

    ds_n = FMT_META['s2_name'][1]
    n_acc, _ = eval_desc_acc(model, t['s2_name'][:200], DEVICE, pad_id, mask_id, desc_start=ds_n)
    eval_results['s2_name'] = {'overall': round(n_acc, 4)}
    print(f"  S2.1: {n_acc:.3f}")

    ds_g = FMT_META['s2_gender'][1]
    g_acc, _ = eval_desc_acc(model, t['s2_gender'][:200], DEVICE, pad_id, mask_id, desc_start=ds_g)
    eval_results['s2_gender'] = {'overall': round(g_acc, 4)}
    print(f"  S2.2: {g_acc:.3f}")

    ds_i = FMT_META['s2_interact'][1]
    i_acc, i_dim = eval_desc_acc(model, t['s2_interact'][:200], DEVICE, pad_id, mask_id, desc_start=ds_i)
    eval_results['s2_interact'] = {'overall': round(i_acc, 4), 'per_dim': i_dim}
    print(f"  S2.4: {i_acc:.3f}")

    ds_au = FMT_META['s2_auto'][1]
    au_acc, _ = eval_desc_acc(model, t['s2_auto'][:200], DEVICE, pad_id, mask_id, desc_start=ds_au)
    eval_results['s2_auto'] = {'overall': round(au_acc, 4)}
    print(f"  S2.5: {au_acc:.3f}")

    if t['probe_s2'].shape[0] > 0:
        pr_acc, _ = eval_desc_acc(model, t['probe_s2'], DEVICE, pad_id, mask_id, desc_start=ds_i)
        eval_results['probes'] = {'overall': round(pr_acc, 4)}
        print(f"  Probe: {pr_acc:.3f}")

    # ═══ Save ═══
    elapsed = time.time() - t0
    rd = f"results_2b_{label}"
    os.makedirs(rd, exist_ok=True)

    result = dict(
        label=label,
        phase_order=phase_order,
        seed=seed,
        reverse_ratio=REVERSE_RATIO,
        total_steps=total_step,
        elapsed_sec=round(elapsed, 1),
        history=all_history,
        evaluation=eval_results,
    )
    with open(os.path.join(rd, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    torch.save(model.state_dict(), os.path.join(rd, 'model.pt'))

    print(f"\n  Steps: {total_step}  Time: {elapsed:.0f}s")
    print(f"  Saved: {rd}/")
    print(f"{'='*60}")
    return eval_results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--order', required=True,
                   help='Phase order, comma-separated. e.g. 1,2,3,5,4')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--d', type=int, default=64)
    args = p.parse_args()

    order = [int(x) for x in args.order.split(',')]
    assert set(order).issubset({1,2,3,4,5}) and len(order) >= 1, f"Order must be subset of 1-5, got: {order}"

    run_shuffle(args.world, order, seed=args.seed, bs=args.bs,
                use_fp16=not args.no_fp16, d_model=args.d)
