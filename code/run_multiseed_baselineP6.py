"""
Multi-seed Baseline Phase 6 (bidir30)

4 seeds × (train bidir30 ph6 + exp1 + exp2a + exp2c)
Parallel execution with multiprocessing.

Usage:
  python run_multiseed_baselineP6.py                # run all 4 seeds in parallel
  python run_multiseed_baselineP6.py --seed 123     # run 1 seed only
  python run_multiseed_baselineP6.py --workers 2    # limit parallel workers
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, argparse, time, copy
from multiprocessing import Pool, cpu_count

from model import make_model, count_params
from v6_tool import parse_world, generate, FMT_META, seq_s2_name
from train_core import (
    pretensorize, build_tagged_batches, train_epoch,
    eval_p_recon, eval_desc_acc, eval_full_mask, per_dimension_probe,
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_SEEDS = [123, 456, 789, 1024]  # 4 seeds, excluding 42
REVERSE_RATIO = 0.3
MAX_PHASE = 6  # Phase 6 instead of 5
D_MODEL = 64
OUTPUT_DIR = 'results_multiseed_ph6'


def train_one_seed(seed, data, t, pad_id, mask_id, holdout_dims,
                   bs=256, use_fp16=True):
    """Train one bidir30 model to Phase 6"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    mr = 0.30
    rng = np.random.default_rng(seed)
    total_step = 0
    all_history = {}

    for phase_num in range(1, MAX_PHASE + 1):
        pdef = phases[phase_num]
        pname = pdef['name']
        max_steps = pdef['max_steps']
        stage_data = pdef['data']()

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

            if epoch % 10 == 0 or step >= max_steps:
                print(f"      [seed={seed}] ph{phase_num} ep{epoch:3d}  step={step:5d}  "
                      f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

            if avg_loss < 0.25 and eval_acc >= 0.95:
                print(f"    [seed={seed}] ✓ ph{phase_num} early stop: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
                break
            model.train()

        total_step += step
        all_history[f'phase_{phase_num}'] = history

    return model, all_history, total_step


def evaluate_baseline(model, data, t, pad_id, mask_id, holdout_dims):
    """Run full S2 evaluation"""
    results = {}
    t2id = data['tokens2id']
    ee = data['cfg']['entity_encodings']
    train_names = data['cfg']['train_names']
    te = [(n, ee[n]) for n in train_names]

    # S1 P-recon
    s1_acc, _ = eval_p_recon(model, t['s1_bare'], DEVICE, pad_id, mask_id)
    results['s1_p_recon'] = round(s1_acc, 4)

    # S2.1 name
    ds_name = FMT_META['s2_name'][1]
    s2n_acc, _ = eval_desc_acc(model, t['s2_name'], DEVICE, pad_id, mask_id, desc_start=ds_name)
    results['s2_name'] = round(s2n_acc, 4)

    # S2.2 gender
    ds_gender = FMT_META['s2_gender'][1]
    s2g_acc, _ = eval_desc_acc(model, t['s2_gender'], DEVICE, pad_id, mask_id, desc_start=ds_gender)
    results['s2_gender'] = round(s2g_acc, 4)

    # S2.4 interact
    ds_inter = FMT_META['s2_interact'][1]
    s2i_acc, _ = eval_desc_acc(model, t['s2_interact'], DEVICE, pad_id, mask_id, desc_start=ds_inter)
    results['s2_interact'] = round(s2i_acc, 4)

    # S2.5 auto
    ds_auto = FMT_META['s2_auto'][1]
    s2a_acc, _ = eval_desc_acc(model, t['s2_auto'], DEVICE, pad_id, mask_id, desc_start=ds_auto)
    results['s2_auto'] = round(s2a_acc, 4)

    # Probes
    if 'probes' in t and t['probes'] is not None and len(t['probes']) > 0:
        probe_acc, _ = eval_desc_acc(model, t['probes'], DEVICE, pad_id, mask_id, desc_start=ds_inter)
        results['probes'] = round(probe_acc, 4)

    # S3 (Phase 6)
    if 's3_entity' in t:
        ds_s3 = FMT_META.get('s3_entity', (None, 3))[1]
        s3e_acc, _ = eval_desc_acc(model, t['s3_entity'], DEVICE, pad_id, mask_id, desc_start=ds_s3)
        results['s3_entity'] = round(s3e_acc, 4)
    if 's3_interact' in t:
        ds_s3i = FMT_META.get('s3_interact', (None, 8))[1]
        s3i_acc, _ = eval_desc_acc(model, t['s3_interact'], DEVICE, pad_id, mask_id, desc_start=ds_s3i)
        results['s3_interact'] = round(s3i_acc, 4)

    # Novel entities
    novel_names = data['cfg'].get('novel_names', [])
    if novel_names:
        novel_seqs = []
        for n in novel_names:
            enc = ee[n]
            seq = seq_s2_name('touch', n, enc, data['cfg'], t2id)
            novel_seqs.append(seq)
        novel_t = pretensorize(novel_seqs, pad_id)

        # Per-dim
        fwd_novel, fwd_novel_dim = eval_desc_acc(model, novel_t, DEVICE, pad_id, mask_id, desc_start=ds_name)
        novel_dims_perdim = sum(1 for v in fwd_novel_dim.values() if v >= 0.99)
        results['novel_avg_dims_perdim'] = novel_dims_perdim
        results['novel_overall_perdim'] = round(fwd_novel, 4)

        # Full mask
        fm_novel, fm_novel_dim = eval_full_mask(model, novel_t, DEVICE, pad_id, mask_id, desc_start=ds_name)
        novel_dims_fullmask = sum(1 for v in fm_novel_dim.values() if v >= 0.99)
        results['novel_avg_dims_fullmask'] = novel_dims_fullmask
        results['novel_overall_fullmask'] = round(fm_novel, 4)

    return results


def run_exp1_reverse(model, data, t, pad_id, mask_id):
    """Run exp1 reverse inference"""
    from exp1_reverse import (
        build_test_group_A, build_test_group_B, build_test_group_D,
        build_test_group_E, build_test_group_F, eval_reverse_p
    )
    t2id = data['tokens2id']
    cfg = data['cfg']
    results = {}

    # A: trained entities
    seqs_a, _ = build_test_group_A(cfg, t2id)
    if seqs_a:
        tensor_a = pretensorize(seqs_a, pad_id)
        acc_a, _, _ = eval_reverse_p(model, tensor_a, DEVICE, pad_id, mask_id, p_start=3)
        results['A_train'] = round(acc_a, 4)

    # B: novel entities
    seqs_b, _ = build_test_group_B(cfg, t2id)
    if seqs_b:
        tensor_b = pretensorize(seqs_b, pad_id)
        acc_b, _, _ = eval_reverse_p(model, tensor_b, DEVICE, pad_id, mask_id, p_start=3)
        results['B_novel'] = round(acc_b, 4)

    # D: fictitious
    seqs_d, _, _ = build_test_group_D(cfg, t2id)
    if seqs_d:
        tensor_d = pretensorize(seqs_d, pad_id)
        acc_d, _, _ = eval_reverse_p(model, tensor_d, DEVICE, pad_id, mask_id, p_start=3)
        results['D_fictional'] = round(acc_d, 4)

    # E: interaction results
    seqs_e, _ = build_test_group_E(cfg, t2id)
    if seqs_e:
        tensor_e = pretensorize(seqs_e, pad_id)
        acc_e, _, _ = eval_reverse_p(model, tensor_e, DEVICE, pad_id, mask_id, p_start=9)
        results['E_interact'] = round(acc_e, 4)

    # F: held-out probes
    seqs_f, _ = build_test_group_F(cfg, t2id)
    if seqs_f:
        tensor_f = pretensorize(seqs_f, pad_id)
        acc_f, _, _ = eval_reverse_p(model, tensor_f, DEVICE, pad_id, mask_id, p_start=9)
        results['F_probes'] = round(acc_f, 4)

    return results


def run_exp2a_gender(model, data, t, pad_id, mask_id):
    """Run exp2a gender isolation"""
    from exp2a_gender import build_gender_ablation_s2gender, build_gender_ablation_s2interact
    t2id = data['tokens2id']
    cfg = data['cfg']
    results = {}

    ds_gender = FMT_META['s2_gender'][1]
    ds_inter = FMT_META['s2_interact'][1]

    # Gender format (returns dict, labels)
    conds, _ = build_gender_ablation_s2gender(cfg, t2id)
    for cname, tensor in conds.items():
        acc, _ = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds_gender)
        results[f'gender_{cname}'] = round(acc, 4)

    # Interact format (returns dict, labels)
    iconds, _ = build_gender_ablation_s2interact(cfg, t2id)
    for cname, tensor in iconds.items():
        acc, _ = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds_inter)
        results[f'interact_{cname}'] = round(acc, 4)

    return results


def run_exp2c_dim_ablation(model, data, t, pad_id, mask_id):
    """Run exp2c dimension ablation"""
    from exp2c_dim_ablation import ablate_and_eval, baseline_eval
    t2id = data['tokens2id']
    cfg = data['cfg']
    results = {}

    ds_name = FMT_META['s2_name'][1]
    ALL_DIMS = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']

    # Baseline
    base_name, _ = baseline_eval(model, t['s2_name'], DEVICE, pad_id, mask_id, ds_name, ALL_DIMS)
    results['baseline_name'] = base_name

    # Ablation matrix
    diff_matrix = {}
    for abl_dim in ALL_DIMS:
        diff_matrix[abl_dim] = {}
        abl_res = ablate_and_eval(model, t['s2_name'], DEVICE, pad_id, mask_id,
                                   ds_name, abl_dim, ALL_DIMS, t2id)
        for tgt_dim in ALL_DIMS:
            if tgt_dim != abl_dim:
                diff_matrix[abl_dim][tgt_dim] = round(abl_res.get(tgt_dim, 0) - base_name.get(tgt_dim, 0), 4)

    results['diff_matrix_name'] = diff_matrix
    return results


def run_single_seed(args_tuple):
    """Worker function for parallel execution"""
    seed, world_path, bs, use_fp16 = args_tuple
    
    print(f"\n{'='*60}")
    print(f"  Starting seed {seed}")
    print(f"{'='*60}")
    seed_t0 = time.time()

    # Load data (each process loads independently)
    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    cfg = data['cfg']
    tokens = data['tokens']
    t2id = data['tokens2id']
    pad_id = t2id['PAD']
    mask_id = t2id['MASK']
    holdout_dims = data['holdout_dims']

    # Pre-tensorize (full set for train_phases)
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
    if data['s2'].get('probes'):
        t['probes'] = pretensorize(data['s2']['probes'], pad_id)

    # Output directory
    rd = os.path.join(OUTPUT_DIR, f'seed_{seed}')
    os.makedirs(rd, exist_ok=True)
    model_path = os.path.join(rd, 'model.pt')

    # Check existing checkpoint
    if os.path.exists(model_path):
        print(f"  [seed={seed}] Loading existing checkpoint...")
        model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4, n_heads=4,
                           max_len=48, dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        history = {}
    else:
        print(f"  [seed={seed}] Training bidir30 Phase 6...")
        model, history, total_steps = train_one_seed(
            seed, data, t, pad_id, mask_id, holdout_dims, bs=bs, use_fp16=use_fp16)
        torch.save(model.state_dict(), model_path)
        print(f"  [seed={seed}] Saved: {model_path}")

    # Evaluation
    print(f"  [seed={seed}] Running baseline evaluation...")
    baseline_res = evaluate_baseline(model, data, t, pad_id, mask_id, holdout_dims)

    print(f"  [seed={seed}] Running exp1 reverse...")
    exp1_res = run_exp1_reverse(model, data, t, pad_id, mask_id)

    print(f"  [seed={seed}] Running exp2a gender...")
    exp2a_res = run_exp2a_gender(model, data, t, pad_id, mask_id)

    print(f"  [seed={seed}] Running exp2c dim ablation...")
    exp2c_res = run_exp2c_dim_ablation(model, data, t, pad_id, mask_id)

    # Save results
    seed_result = {
        'seed': seed,
        'max_phase': MAX_PHASE,
        'elapsed_sec': round(time.time() - seed_t0, 1),
        'baseline': baseline_res,
        'exp1_reverse': exp1_res,
        'exp2a_gender': exp2a_res,
        'exp2c_dim_ablation': exp2c_res,
        'history': history,
    }
    with open(os.path.join(rd, 'seed_results.json'), 'w') as f:
        json.dump(seed_result, f, indent=2, default=str)

    elapsed = time.time() - seed_t0
    print(f"  [seed={seed}] Done: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return seed, seed_result


def run(world_path, seeds, bs=256, use_fp16=True, workers=None):
    print(f"{'═'*60}")
    print(f"  Multi-seed Baseline Phase 6 (bidir30)")
    print(f"  Seeds: {seeds}")
    print(f"  Max Phase: {MAX_PHASE}")
    print(f"  Device: {DEVICE}")
    print(f"  Workers: {workers or 'all'}")
    print(f"{'═'*60}")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare args for workers
    args_list = [(seed, world_path, bs, use_fp16) for seed in seeds]

    # Parallel execution
    n_workers = workers or min(len(seeds), cpu_count())
    
    if n_workers == 1 or len(seeds) == 1:
        # Sequential
        all_results = {}
        for args in args_list:
            seed, result = run_single_seed(args)
            all_results[seed] = result
    else:
        # Parallel
        print(f"\n  Starting {len(seeds)} seeds with {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = pool.map(run_single_seed, args_list)
        all_results = {seed: result for seed, result in results}

    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'═'*60}")
    print(f"  Summary · {len(seeds)} seeds · Phase {MAX_PHASE}")
    print(f"{'═'*60}")

    metrics = {
        's1_p_recon': [], 's2_name': [], 's2_gender': [],
        's2_interact': [], 's2_auto': [], 'probes': [],
        's3_entity': [], 's3_interact': [],
        'novel_perdim': [], 'novel_fullmask': [],
        'exp1_A': [], 'exp1_E': [],
    }

    for seed, r in all_results.items():
        b = r['baseline']
        for k in ['s1_p_recon', 's2_name', 's2_gender', 's2_interact', 's2_auto', 
                  'probes', 's3_entity', 's3_interact']:
            if k in b:
                metrics[k].append(b[k])
        if 'novel_avg_dims_perdim' in b:
            metrics['novel_perdim'].append(b['novel_avg_dims_perdim'])
        if 'novel_avg_dims_fullmask' in b:
            metrics['novel_fullmask'].append(b['novel_avg_dims_fullmask'])

        e1 = r['exp1_reverse']
        if 'A_train' in e1:
            metrics['exp1_A'].append(e1['A_train'])
        if 'E_interact' in e1:
            metrics['exp1_E'].append(e1['E_interact'])

    print(f"\n  {'Metric':<20s}  {'Mean':>7s}  {'Std':>7s}")
    print(f"  {'─'*40}")
    for name, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            print(f"  {name:<20s}  {arr.mean():>7.4f}  {arr.std():>7.4f}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Multi-seed Baseline Phase 6')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--seed', type=int, default=None, help='Run single seed')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--workers', type=int, default=None, help='Parallel workers')
    p.add_argument('--no-fp16', action='store_true')
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS

    run(args.world, seeds, bs=args.bs, use_fp16=not args.no_fp16, workers=args.workers)
