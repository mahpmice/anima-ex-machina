"""
Multi-seed Baseline (bidir30)


5seed × (trainbidir30 ph5 + exp1 + exp2a + exp2c + exp2d)

Usage:
  python run_multiseed_baseline.py                    # run all 5 seeds
  python run_multiseed_baseline.py --seed 42          # run 1 seed only
  python run_multiseed_baseline.py --seeds 42 123     # Run specifiedseed
  python run_multiseed_baseline.py --skip-interp      # Skipexp2d（mostslow）
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, argparse, time, copy

from model import make_model, count_params
from v6_tool import parse_world, generate, FMT_META, seq_s2_name
from train_core import (
    pretensorize, build_tagged_batches, train_epoch,
    eval_p_recon, eval_desc_acc, eval_full_mask, per_dimension_probe,
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
REVERSE_RATIO = 0.3
MAX_PHASE = 5
D_MODEL = 64


def train_one_seed(seed, data, t, pad_id, mask_id, holdout_dims,
                   bs=256, use_fp16=True):
    """trainonebidir30Model，Returnmodel and trainhistory"""
    # Set global seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    mr = 0.30
    rng = np.random.default_rng(seed)  # useseedcontrolbatchorder
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
                print(f"      ph{phase_num} ep{epoch:3d}  step={step:5d}  "
                      f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

            if avg_loss < 0.25 and eval_acc >= 0.95:
                print(f"    ✓ ph{phase_num} skip-step: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
                break
            model.train()

        total_step += step
        all_history[f'phase_{phase_num}'] = history

    return model, all_history, total_step


def evaluate_baseline(model, data, t, pad_id, mask_id, holdout_dims):
    """runfull suiteS2evaluation，Returnresultsdict"""
    results = {}
    t2id = data['tokens2id']
    ee = data['cfg']['entity_encodings']
    train_names = data['cfg']['train_names']
    te = [(n, ee[n]) for n in train_names]

    # S1 P-recon
    p_acc, p_dim = eval_p_recon(model, t['s1_bare'][:200], DEVICE, pad_id, mask_id)
    results['s1_p_recon'] = round(p_acc, 4)

    # S2.1 name
    ds_n = FMT_META['s2_name'][1]
    n_acc, n_dim = eval_desc_acc(model, t['s2_name'][:200], DEVICE, pad_id, mask_id,
                                  desc_start=ds_n)
    results['s2_name'] = round(n_acc, 4)
    results['s2_name_per_dim'] = n_dim

    # S2.2 gender
    ds_g = FMT_META['s2_gender'][1]
    g_acc, _ = eval_desc_acc(model, t['s2_gender'][:200], DEVICE, pad_id, mask_id,
                              desc_start=ds_g)
    results['s2_gender'] = round(g_acc, 4)

    # S2.4 interact
    ds_i = FMT_META['s2_interact'][1]
    i_acc, i_dim = eval_desc_acc(model, t['s2_interact'][:200], DEVICE, pad_id, mask_id,
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

    # Novelentities
    novel_data = data.get('novel', {})
    if novel_data:
        # methodone：per-dimmask（eval_desc_acc）——eachtimesOnlymaskonedim，otherssee
        novel_scores_perdim = []
        for name, ndata in novel_data.items():
            ns2 = pretensorize([ndata['s2_name']], pad_id)
            na, nd = eval_desc_acc(model, ns2, DEVICE, pad_id, mask_id,
                                    desc_start=ds_n)
            correct_dims = sum(1 for v in nd.values() if v >= 0.99)
            novel_scores_perdim.append(correct_dims)
        results['novel_avg_dims_perdim'] = round(np.mean(novel_scores_perdim), 2)

        # Method 2：allmask（eval_full_mask）——alldescsimultaneouslymask，OnlywithParray
        # Same asblind_touch.pypairsaligned，is the strictest test
        novel_scores_fullmask = []
        for name, ndata in novel_data.items():
            ns2 = pretensorize([ndata['s2_name']], pad_id)
            fa, fd = eval_full_mask(model, ns2, DEVICE, pad_id, mask_id,
                                     desc_start=ds_n)
            correct_dims = sum(1 for v in fd.values() if v >= 0.99)
            novel_scores_fullmask.append(correct_dims)
        results['novel_avg_dims_fullmask'] = round(np.mean(novel_scores_fullmask), 2)

    return results


def run_exp1(model, data, checkpoint_path):
    """runexp1reverseinference，Returnkeynumbers"""
    from exp1_reverse import (
        build_test_group_A, build_test_group_B, build_test_group_C,
        build_test_group_D, build_test_group_E, build_test_group_F,
        eval_reverse_p,
    )
    from v6_tool import build_tokens

    cfg = data['cfg']
    tokens = build_tokens(cfg)
    t2id = {t: i for i, t in enumerate(tokens)}
    pad_id = t2id['PAD']
    mask_id = t2id['MASK']

    results = {}

    # A
    seqs_a, _ = build_test_group_A(cfg, t2id)
    tensor_a = pretensorize(seqs_a, pad_id)
    acc_a, dim_a, _ = eval_reverse_p(model, tensor_a, DEVICE, pad_id, mask_id, p_start=3)
    results['A_train'] = round(acc_a, 4)

    # B
    seqs_b, _ = build_test_group_B(cfg, t2id)
    if seqs_b:
        tensor_b = pretensorize(seqs_b, pad_id)
        acc_b, _, _ = eval_reverse_p(model, tensor_b, DEVICE, pad_id, mask_id, p_start=3)
        results['B_novel'] = round(acc_b, 4)

    # C
    seqs_c, _, _ = build_test_group_C(cfg, t2id)
    if seqs_c:
        tensor_c = pretensorize(seqs_c, pad_id)
        acc_c, _, _ = eval_reverse_p(model, tensor_c, DEVICE, pad_id, mask_id, p_start=3)
        results['C_mismatch'] = round(acc_c, 4)

    # D
    seqs_d, _, _ = build_test_group_D(cfg, t2id)
    tensor_d = pretensorize(seqs_d, pad_id)
    acc_d, _, _ = eval_reverse_p(model, tensor_d, DEVICE, pad_id, mask_id, p_start=3)
    results['D_fictional'] = round(acc_d, 4)

    # E
    seqs_e, _ = build_test_group_E(cfg, t2id)
    tensor_e = pretensorize(seqs_e, pad_id)
    acc_e, _, _ = eval_reverse_p(model, tensor_e, DEVICE, pad_id, mask_id, p_start=9)
    results['E_interact'] = round(acc_e, 4)

    # F
    seqs_f, _ = build_test_group_F(cfg, t2id)
    if seqs_f:
        tensor_f = pretensorize(seqs_f, pad_id)
        acc_f, _, _ = eval_reverse_p(model, tensor_f, DEVICE, pad_id, mask_id, p_start=9)
        results['F_probes'] = round(acc_f, 4)

    return results


def run_exp2a(model, data, model_path):
    """runexp2agender markerisolation，Returnkeynumbers"""
    from exp2a_gender import build_gender_ablation_s2gender, build_gender_ablation_s2interact

    cfg = data['cfg']
    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']

    results = {}

    # S2.2 Gender
    ds = FMT_META['s2_gender'][1]
    gender_seqs, _ = build_gender_ablation_s2gender(cfg, t2id)
    for cond_name, tensor in gender_seqs.items():
        acc, _ = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds)
        results[f'gender_{cond_name}'] = round(acc, 4)

    # S2.4 Interact
    ds_i = FMT_META['s2_interact'][1]
    interact_seqs, _ = build_gender_ablation_s2interact(cfg, t2id)
    for cond_name, tensor in interact_seqs.items():
        acc, _ = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds_i)
        results[f'interact_{cond_name}'] = round(acc, 4)

    return results


def run_exp2c(model, data, model_path):
    """runexp2cdimension ablation，Returndifferencematrix keystatistics"""
    from exp2c_dim_ablation import eval_baseline, eval_with_dim_ablated

    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']

    DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']

    s2_name = pretensorize(data['s2']['name'], pad_id)
    ds = FMT_META['s2_name'][1]
    p_start = ds - 13

    baseline = eval_baseline(model, s2_name, DEVICE, pad_id, mask_id, ds)

    # Builddifferencematrix
    diff_matrix = {}
    for abl_dim in range(13):
        abl_name = DIM_NAMES[abl_dim]
        dim_result = eval_with_dim_ablated(
            model, s2_name, DEVICE, pad_id, mask_id,
            desc_start=ds, ablated_dim=abl_dim, p_start=p_start)
        diffs = {}
        for tgt_name, acc in dim_result.items():
            diffs[tgt_name] = round(acc - baseline.get(tgt_name, 0), 3)
        diff_matrix[abl_name] = diffs

    # Extractkeystatistics：max negative dependency、max positive dependency
    all_diffs = []
    for abl in diff_matrix.values():
        for v in abl.values():
            all_diffs.append(v)

    return {
        'baseline': baseline,
        'diff_matrix': diff_matrix,
        'max_neg_diff': round(min(all_diffs), 3),
        'max_pos_diff': round(max(all_diffs), 3),
        'mean_abs_diff': round(np.mean([abs(d) for d in all_diffs]), 4),
    }


def run(world_path, seeds, bs=256, use_fp16=True, skip_interp=False):
    t_start = time.time()

    print(f"{'═'*60}")
    print(f"  Multi-seedExperiment · Baseline bidir30")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {DEVICE}  bs={bs}  fp16={use_fp16}")
    print(f"{'═'*60}")

    # dataOnlygenerateonetimes
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

    for si, seed in enumerate(seeds):
        print(f"\n{'━'*60}")
        print(f"  Seed {seed} ({si+1}/{len(seeds)})")
        print(f"{'━'*60}")

        seed_t0 = time.time()
        rd = f"results_multiseed/seed_{seed}"
        os.makedirs(rd, exist_ok=True)
        model_path = os.path.join(rd, 'model.pt')

        # ── train（Checkpoint existsthenSkip）──
        if os.path.exists(model_path):
            print(f"  [1/5] Checkpoint exists，Skip training，Load {model_path}")
            model = make_model(data['vocab_size'], d=D_MODEL, n_layers=4, n_heads=4,
                               max_len=48, dropout=0.0).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                             weights_only=True))
            model.eval()
            history = {}
            total_steps = 0
        else:
            print(f"  [1/5] train bidir30 ph5...")
            model, history, total_steps = train_one_seed(
                seed, data, t, pad_id, mask_id, holdout_dims,
                bs=bs, use_fp16=use_fp16)
            torch.save(model.state_dict(), model_path)

        # ── evaluation baseline ──
        print(f"  [2/5] Baselineevaluation...")
        baseline_res = evaluate_baseline(model, data, t, pad_id, mask_id, holdout_dims)
        print(f"    S1={baseline_res['s1_p_recon']}  S2.1={baseline_res['s2_name']}  "
              f"S2.4={baseline_res['s2_interact']}  Probe={baseline_res.get('probes', 'N/A')}  "
              f"Novel(fullmask)={baseline_res.get('novel_avg_dims_fullmask', 'N/A')}/13")

        # ── exp1 ──
        print(f"  [3/5] Exp1 reverseinference...")
        exp1_res = run_exp1(model, data, model_path)
        print(f"    A={exp1_res.get('A_train', 'N/A')}  B={exp1_res.get('B_novel', 'N/A')}  "
              f"D={exp1_res.get('D_fictional', 'N/A')}  E={exp1_res.get('E_interact', 'N/A')}  "
              f"F={exp1_res.get('F_probes', 'N/A')}")

        # ── exp2a ──
        print(f"  [4/5] Exp2a gender markerisolation...")
        exp2a_res = run_exp2a(model, data, model_path)
        print(f"    gender_Full={exp2a_res.get('gender_A_full', 'N/A')}  "
              f"gender_NoGender={exp2a_res.get('gender_B_no_gender', 'N/A')}  "
              f"interact_NoBoth={exp2a_res.get('interact_D_no_both', 'N/A')}")

        # ── exp2c ──
        print(f"  [5/5] Exp2c dimension ablation...")
        exp2c_res = run_exp2c(model, data, model_path)
        print(f"    max_neg={exp2c_res['max_neg_diff']}  max_pos={exp2c_res['max_pos_diff']}")

        # ── exp2d（select，mostslow）──
        exp2d_res = {}
        if not skip_interp:
            print(f"  [bonus] Exp2d interpretability...")
            try:
                from exp2d_interp import linear_probe_all, causal_intervention
                probe_res = linear_probe_all(model, data, data['cfg'], DEVICE)
                causal_res = causal_intervention(model, data, data['cfg'], DEVICE)
                exp2d_res = {'probes': probe_res, 'causal': causal_res}
            except Exception as e:
                print(f"    exp2dFailed: {e}")

        # ── Save ──
        seed_result = {
            'seed': seed,
            'total_steps': total_steps,
            'elapsed_sec': round(time.time() - seed_t0, 1),
            'baseline': baseline_res,
            'exp1_reverse': exp1_res,
            'exp2a_gender': exp2a_res,
            'exp2c_dim_ablation': exp2c_res,
            'exp2d_interp': exp2d_res,
            'history': history,
        }
        with open(os.path.join(rd, 'seed_results.json'), 'w') as f:
            json.dump(seed_result, f, indent=2, default=str)

        all_results[seed] = seed_result
        print(f"  Seed {seed} done: {time.time() - seed_t0:.0f}s")

        # Release GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── SaveSummary ──
    os.makedirs('results_multiseed', exist_ok=True)
    with open('results_multiseed/all_baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── PrintSummary ──
    print(f"\n{'═'*60}")
    print(f"  Summary · {len(seeds)} seeds")
    print(f"{'═'*60}")

    # Collect key metrics
    metrics = {
        's1_p_recon': [], 's2_name': [], 's2_gender': [],
        's2_interact': [], 's2_auto': [], 'probes': [],
        'novel_perdim': [], 'novel_fullmask': [],
        'exp1_A': [], 'exp1_B': [], 'exp1_D': [],
        'exp1_E': [], 'exp1_F': [],
        'gender_full': [], 'gender_no_gender': [],
        'interact_no_both': [],
    }

    for seed, r in all_results.items():
        b = r['baseline']
        metrics['s1_p_recon'].append(b['s1_p_recon'])
        metrics['s2_name'].append(b['s2_name'])
        metrics['s2_gender'].append(b['s2_gender'])
        metrics['s2_interact'].append(b['s2_interact'])
        metrics['s2_auto'].append(b['s2_auto'])
        metrics['probes'].append(b.get('probes', 0))
        if 'novel_avg_dims_perdim' in b:
            metrics['novel_perdim'].append(b['novel_avg_dims_perdim'])
        if 'novel_avg_dims_fullmask' in b:
            metrics['novel_fullmask'].append(b['novel_avg_dims_fullmask'])

        e1 = r['exp1_reverse']
        metrics['exp1_A'].append(e1.get('A_train', 0))
        metrics['exp1_B'].append(e1.get('B_novel', 0))
        metrics['exp1_D'].append(e1.get('D_fictional', 0))
        metrics['exp1_E'].append(e1.get('E_interact', 0))
        metrics['exp1_F'].append(e1.get('F_probes', 0))

        e2a = r['exp2a_gender']
        metrics['gender_full'].append(e2a.get('gender_A_full', 0))
        metrics['gender_no_gender'].append(e2a.get('gender_B_no_gender', 0))
        metrics['interact_no_both'].append(e2a.get('interact_D_no_both', 0))

    print(f"\n  {'Metric':<25s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
    print(f"  {'─'*55}")
    for name, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            print(f"  {name:<25s}  {arr.mean():>7.4f}  {arr.std():>7.4f}  "
                  f"{arr.min():>7.4f}  {arr.max():>7.4f}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Results: results_multiseed/")
    print(f"{'═'*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Multi-seedBaselineExperiment')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--seed', type=int, default=None, help='Run only oneseed')
    p.add_argument('--seeds', type=int, nargs='+', default=None, help='Run specifiedseeds')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--skip-interp', action='store_true', help='Skipexp2d')
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS

    run(args.world, seeds, bs=args.bs, use_fp16=not args.no_fp16,
        skip_interp=args.skip_interp)
