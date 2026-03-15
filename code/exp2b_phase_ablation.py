"""
Exp 2B: Phase ablation


Train ablation models, each skipping one phase or shuffling order.
max_phase=5（excludingS3），reverse_ratio=0.3（aligned with bidir30）。

Usage:
  python exp2b_phase_ablation.py --skip 1 --world swadesh_v6_world.md
  python exp2b_phase_ablation.py --shuffle --world swadesh_v6_world.md
  python exp2b_phase_ablation.py --all --world swadesh_v6_world.md
  python exp2b_phase_ablation.py --minimal --world swadesh_v6_world.md

Parallel (3 terminals):
  terminal1: python exp2b_phase_ablation.py --skip 1
  terminal2: python exp2b_phase_ablation.py --skip 2
  terminal3: python exp2b_phase_ablation.py --skip 3
  （after completion）
  terminal1: python exp2b_phase_ablation.py --skip 4
  terminal2: python exp2b_phase_ablation.py --skip 5
  terminal3: python exp2b_phase_ablation.py --shuffle
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
from v6_rules import compute_result

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

MAX_PHASE = 5   # excludingS3
REVERSE_RATIO = 0.3  # aligned with bidir30


def run_ablation(world_path, skip_phase=None, shuffle_order=False,
                 bs=256, use_fp16=True, d_model=64):
    t0 = time.time()

    if skip_phase:
        label = f"DPh{skip_phase}"
    elif shuffle_order:
        label = "Shuffle"
    else:
        label = "Baseline"

    print(f"\n{'═'*60}")
    print(f"  Experiment 2B · {label}")
    print(f"  skip={skip_phase}  shuffle={shuffle_order}  reverse={REVERSE_RATIO}")
    print(f"  Device: {DEVICE}  bs={bs}  fp16={use_fp16}")
    print(f"{'═'*60}")

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
        # S3 tensorsstill need to construct（build_phasesinternally reference），but do not doS3evaluation
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

    # Determine training order（1-5，excluding6）
    phase_order = list(range(1, MAX_PHASE + 1))
    if skip_phase and skip_phase in phase_order:
        phase_order.remove(skip_phase)
        print(f"  Skip Phase {skip_phase}")
    if shuffle_order:
        rng_shuffle = np.random.default_rng(12345)
        rng_shuffle.shuffle(phase_order)
        print(f"  shuffle order: {phase_order}")

    mr = 0.30
    rng = np.random.default_rng(42)
    total_step = 0
    all_history = {}

    for phase_num in phase_order:
        pdef = phases[phase_num]
        pname = pdef['name']
        max_steps = pdef['max_steps']
        stage_data = pdef['data']()

        print(f"\n  Phase {phase_num}: {pname}")
        total_seqs = sum(td[1].shape[0] for td in stage_data)
        print(f"  Total sequences: {total_seqs}  max_steps={max_steps}")

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
                print(f"  ✓ skip-step: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
                break
            model.train()

        total_step += step
        all_history[f'phase_{phase_num}'] = history

    # ═══ evaluation（all S2 format）═══
    print(f"\n{'='*60}")
    print(f"EVALUATION · {label}")
    print(f"{'='*60}")

    eval_results = {}

    # S1 P-recon
    p_acc, p_dim = eval_p_recon(model, t['s1_bare'][:200], DEVICE, pad_id, mask_id)
    eval_results['s1_p_recon'] = {'overall': round(p_acc, 4), 'per_dim': p_dim}
    print(f"  S1 P-recon: {p_acc:.3f}")

    # S2.1 name
    ds_n = FMT_META['s2_name'][1]
    n_acc, n_dim = eval_desc_acc(model, t['s2_name'][:200], DEVICE, pad_id, mask_id,
                                  desc_start=ds_n)
    eval_results['s2_name'] = {'overall': round(n_acc, 4), 'per_dim': n_dim}
    print(f"  S2.1 name: {n_acc:.3f}")

    # S2.2 gender
    ds_g = FMT_META['s2_gender'][1]
    g_acc, _ = eval_desc_acc(model, t['s2_gender'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_g)
    eval_results['s2_gender'] = {'overall': round(g_acc, 4)}
    print(f"  S2.2 gender: {g_acc:.3f}")

    # S2.3 un/il
    ds_a = FMT_META['s2_article'][1]
    un_acc, _ = eval_desc_acc(model, t['s2_un'][:100], DEVICE, pad_id, mask_id,
                               desc_start=ds_a)
    il_acc, _ = eval_desc_acc(model, t['s2_il'][:100], DEVICE, pad_id, mask_id,
                               desc_start=ds_a)
    eval_results['s2_un'] = {'overall': round(un_acc, 4)}
    eval_results['s2_il'] = {'overall': round(il_acc, 4)}
    print(f"  S2.3 un: {un_acc:.3f}  il: {il_acc:.3f}")

    # S2.4 interact
    ds_i = FMT_META['s2_interact'][1]
    i_acc, i_dim = eval_desc_acc(model, t['s2_interact'][:200], DEVICE, pad_id, mask_id,
                                  desc_start=ds_i)
    eval_results['s2_interact'] = {'overall': round(i_acc, 4), 'per_dim': i_dim}
    print(f"  S2.4 interact: {i_acc:.3f}")

    # S2.5 auto
    ds_au = FMT_META['s2_auto'][1]
    au_acc, _ = eval_desc_acc(model, t['s2_auto'][:200], DEVICE, pad_id, mask_id,
                               desc_start=ds_au)
    eval_results['s2_auto'] = {'overall': round(au_acc, 4)}
    print(f"  S2.5 auto: {au_acc:.3f}")

    # Probes（S2format）
    if t['probe_s2'].shape[0] > 0:
        pr_acc, _ = eval_desc_acc(model, t['probe_s2'], DEVICE, pad_id, mask_id,
                                   desc_start=ds_i)
        eval_results['probes'] = {'overall': round(pr_acc, 4)}
        print(f"  Probes (held-out, S2): {pr_acc:.3f}")
        for idx, (a, b, rule, desc_txt, _) in enumerate(data['probes']['raw']):
            single = t['probe_s2'][idx:idx+1]
            sa, _ = eval_desc_acc(model, single, DEVICE, pad_id, mask_id,
                                   desc_start=ds_i)
            print(f"    [{rule}] {a}+{b}: {sa:.3f}")

    # Novelentities
    novel_data = data.get('novel', {})
    if novel_data:
        print(f"\n  ── Novel ──")
        for name, ndata in novel_data.items():
            ns2 = pretensorize([ndata['s2_name']], pad_id)
            na, nd = eval_desc_acc(model, ns2, DEVICE, pad_id, mask_id,
                                    desc_start=ds_n)
            correct_dims = sum(1 for v in nd.values() if v >= 0.99)
            print(f"    {name}: {na:.3f} ({correct_dims}/13)")
        eval_results['novel_tested'] = True

    # Embedding probe
    probe = per_dimension_probe(model, te, DEVICE, t2id)
    if probe:
        eval_results['embedding_probe'] = probe

    # ═══ Save ═══
    elapsed = time.time() - t0
    rd = f"results_2b_{label.lower()}"
    os.makedirs(rd, exist_ok=True)

    result = dict(
        label=label, skip_phase=skip_phase, shuffle_order=shuffle_order,
        phase_order=phase_order, reverse_ratio=REVERSE_RATIO,
        total_steps=total_step, elapsed_sec=round(elapsed, 1),
        history=all_history, evaluation=eval_results,
    )
    with open(os.path.join(rd, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    torch.save(model.state_dict(), os.path.join(rd, 'model.pt'))

    print(f"\n  Total steps: {total_step}  Elapsed: {elapsed:.0f}s")
    print(f"  Saved to {rd}/")
    print(f"{'='*60}")
    return eval_results


def run_all(world_path, bs=256, use_fp16=True, d_model=64):
    all_results = {}
    for skip in range(1, 6):
        r = run_ablation(world_path, skip_phase=skip, bs=bs,
                        use_fp16=use_fp16, d_model=d_model)
        all_results[f'DPh{skip}'] = r

    r = run_ablation(world_path, shuffle_order=True, bs=bs,
                    use_fp16=use_fp16, d_model=d_model)
    all_results['Shuffle'] = r

    # Summary
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    header = f"  {'Model':<12s}  {'S1':>6s}  {'S2.1':>6s}  {'S2.2':>6s}  {'S2.4':>6s}  {'S2.5':>6s}  {'Probe':>6s}"
    print(header)
    for label, r in all_results.items():
        s1 = r.get('s1_p_recon', {}).get('overall', 0)
        s21 = r.get('s2_name', {}).get('overall', 0)
        s22 = r.get('s2_gender', {}).get('overall', 0)
        s24 = r.get('s2_interact', {}).get('overall', 0)
        s25 = r.get('s2_auto', {}).get('overall', 0)
        pr = r.get('probes', {}).get('overall', 0)
        print(f"  {label:<12s}  {s1:>6.3f}  {s21:>6.3f}  {s22:>6.3f}  {s24:>6.3f}  {s25:>6.3f}  {pr:>6.3f}")

    with open('results_2b_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Summary: results_2b_summary.json")


def run_minimal(world_path, bs=256, use_fp16=True, d_model=64):
    all_results = {}
    r = run_ablation(world_path, skip_phase=1, bs=bs,
                    use_fp16=use_fp16, d_model=d_model)
    all_results['DPh1'] = r
    r = run_ablation(world_path, shuffle_order=True, bs=bs,
                    use_fp16=use_fp16, d_model=d_model)
    all_results['Shuffle'] = r
    with open('results_2b_minimal_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=None, choices=[1,2,3,4,5])
    p.add_argument('--shuffle', action='store_true')
    p.add_argument('--all', action='store_true')
    p.add_argument('--minimal', action='store_true')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--d', type=int, default=64)
    args = p.parse_args()

    if args.all:
        run_all(args.world, args.bs, not args.no_fp16, args.d)
    elif args.minimal:
        run_minimal(args.world, args.bs, not args.no_fp16, args.d)
    else:
        run_ablation(args.world, skip_phase=args.skip,
                    shuffle_order=args.shuffle, bs=args.bs,
                    use_fp16=not args.no_fp16, d_model=args.d)
