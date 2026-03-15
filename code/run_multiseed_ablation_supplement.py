"""
Multi-seed Phase Ablation (ΔPh5 + Shuffle)


3seed × 2ablation conditions (ΔPh5, Shuffle)
These are the hardest figures — ΔPh5=0 and Shuffle≈0. Multi-seed confirms not luck.

Usage:
  python run_multiseed_ablation.py                      # all
  python run_multiseed_ablation.py --condition dph5      # Only runΔPh5
  python run_multiseed_ablation.py --condition shuffle    # Only runShuffle
  python run_multiseed_ablation.py --seed 42             # Run only oneseed
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
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_SEEDS = [42, 123, 456]
REVERSE_RATIO = 0.3
MAX_PHASE = 5
D_MODEL = 64


def train_ablation(seed, data, t, pad_id, mask_id, holdout_dims,
                   skip_phase=None, shuffle_order=False,
                   bs=256, use_fp16=True):
    """trainoneablationModel"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    # Determine training order
    phase_order = list(range(1, MAX_PHASE + 1))
    if skip_phase and skip_phase in phase_order:
        phase_order.remove(skip_phase)
    if shuffle_order:
        rng_shuffle = np.random.default_rng(seed + 10000)  # Use different seedshuffle
        rng_shuffle.shuffle(phase_order)

    mr = 0.30
    rng = np.random.default_rng(seed)
    total_step = 0

    for phase_num in phase_order:
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
                break
            model.train()

        total_step += step

    return model, phase_order, total_step


def evaluate_ablation(model, data, t, pad_id, mask_id):
    """evaluationablationModel——Same asexp2bsame metrics"""
    results = {}

    # S1 P-recon
    p_acc, _ = eval_p_recon(model, t['s1_bare'][:200], DEVICE, pad_id, mask_id)
    results['s1_p_recon'] = round(p_acc, 4)

    # S2.1 name
    ds_n = FMT_META['s2_name'][1]
    n_acc, _ = eval_desc_acc(model, t['s2_name'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_n)
    results['s2_name'] = round(n_acc, 4)

    # S2.2 gender
    ds_g = FMT_META['s2_gender'][1]
    g_acc, _ = eval_desc_acc(model, t['s2_gender'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_g)
    results['s2_gender'] = round(g_acc, 4)

    # S2.4 interact
    ds_i = FMT_META['s2_interact'][1]
    i_acc, _ = eval_desc_acc(model, t['s2_interact'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_i)
    results['s2_interact'] = round(i_acc, 4)

    # S2.5 auto
    ds_au = FMT_META['s2_auto'][1]
    au_acc, _ = eval_desc_acc(model, t['s2_auto'][:200], DEVICE, pad_id, mask_id,
                               desc_start=ds_au)
    results['s2_auto'] = round(au_acc, 4)

    # Probes
    if t['probe_s2'].shape[0] > 0:
        pr_acc, _ = eval_desc_acc(model, t['probe_s2'], DEVICE, pad_id, mask_id,
                                   desc_start=ds_i)
        results['probes'] = round(pr_acc, 4)

    return results


def run(world_path, seeds, conditions, bs=256, use_fp16=True):
    t_start = time.time()

    print(f"{'═'*60}")
    print(f"  Multi-seedExperiment · Phaseablation")
    print(f"  Seeds: {seeds}")
    print(f"  Conditions: {conditions}")
    print(f"  Device: {DEVICE}")
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

    for cond in conditions:
        if cond.startswith('dph'):
            skip_phase = int(cond[3:])
            shuffle = False
            label = f'DPh{skip_phase}'
        elif cond == 'shuffle':
            skip_phase = None
            shuffle = True
            label = 'Shuffle'
        else:
            continue

        cond_results = {}

        for si, seed in enumerate(seeds):
            print(f"\n{'━'*60}")
            print(f"  {label} · Seed {seed} ({si+1}/{len(seeds)})")
            print(f"{'━'*60}")

            seed_t0 = time.time()
            rd = f"results_multiseed_ablation/{label.lower()}_seed_{seed}"
            os.makedirs(rd, exist_ok=True)
            model_path = os.path.join(rd, 'model.pt')

            if os.path.exists(model_path):
                print(f"    Checkpoint exists，Skip training")
                model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4,
                                   n_heads=4, max_len=48, dropout=0.0).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                                 weights_only=True))
                model.eval()
                phase_order = []
                total_steps = 0
            else:
                model, phase_order, total_steps = train_ablation(
                    seed, data, t, pad_id, mask_id, holdout_dims,
                    skip_phase=skip_phase, shuffle_order=shuffle,
                    bs=bs, use_fp16=use_fp16)
                torch.save(model.state_dict(), model_path)

            # evaluation
            eval_res = evaluate_ablation(model, data, t, pad_id, mask_id)
            print(f"    S1={eval_res['s1_p_recon']}  S2.1={eval_res['s2_name']}  "
                  f"S2.2={eval_res['s2_gender']}  S2.4={eval_res['s2_interact']}  "
                  f"S2.5={eval_res['s2_auto']}  Probe={eval_res.get('probes', 'N/A')}")

            seed_result = {
                'seed': seed, 'condition': label,
                'phase_order': phase_order,
                'total_steps': total_steps,
                'elapsed_sec': round(time.time() - seed_t0, 1),
                'evaluation': eval_res,
            }
            with open(os.path.join(rd, 'seed_results.json'), 'w') as f:
                json.dump(seed_result, f, indent=2, default=str)

            cond_results[seed] = seed_result
            print(f"    Done: {time.time() - seed_t0:.0f}s")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[label] = cond_results

    # ── SaveSummary ──
    os.makedirs('results_multiseed_ablation', exist_ok=True)
    with open('results_multiseed_ablation/all_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── PrintSummary ──
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")

    eval_keys = ['s1_p_recon', 's2_name', 's2_gender', 's2_interact', 's2_auto', 'probes']
    header = f"  {'Condition':<12s}"
    for k in eval_keys:
        header += f"  {k:>12s}"
    print(header)
    print(f"  {'─'*80}")

    for label, cond_res in all_results.items():
        vals = {k: [] for k in eval_keys}
        for seed, r in cond_res.items():
            ev = r['evaluation']
            for k in eval_keys:
                vals[k].append(ev.get(k, 0))

        line = f"  {label:<12s}"
        for k in eval_keys:
            arr = np.array(vals[k])
            line += f"  {arr.mean():.3f}±{arr.std():.3f}"
        print(line)

    total_elapsed = time.time() - t_start
    print(f"\n  Total: {total_elapsed:.0f}s")
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Multi-seedPhaseablation')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--condition', nargs='+',
                   choices=['dph1', 'dph2', 'dph3', 'dph4', 'dph5', 'shuffle'],
                   default=None, help='which to runCondition')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS

    if args.condition:
        conditions = args.condition
    else:
        conditions = ['dph1', 'dph2', 'dph3', 'dph4', 'dph5', 'shuffle']

    run(args.world, seeds, conditions, bs=args.bs, use_fp16=not args.no_fp16)
