"""
Exp 2C: Dimension ablation (at test time)


Use bidir30 model, no retraining. All S2 format.
At test time, mask one dimension P value and desc pair, observe remaining 12 dims.

Output: 13x13 dependency matrix

Usage:
  python exp2c_dim_ablation.py --model results_v4_ph5_bidir30/model.pt --world swadesh_v6_world.md
"""

import torch
import numpy as np
import argparse, json, os

from model import make_model
from v6_tool import parse_world, generate, FMT_META
from train_core import pretensorize, N_DIMS, TOKENS_PER_DIM

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']


def eval_with_dim_ablated(model, tensor, device, pad_id, mask_id,
                          desc_start, ablated_dim, p_start,
                          eval_bs=128):
    """
    maskspecified dimP tokens and desc pair，evaluationremaining dimsdescaccuracy。
    """
    if tensor.shape[0] == 0:
        return {}

    B, L = tensor.shape
    tpd = TOKENS_PER_DIM

    # inablate target dim on input
    inp = tensor.clone()

    # mask Pregionpairstokens
    p_pos = p_start + ablated_dim
    if p_pos < L:
        inp[:, p_pos] = mask_id

    # mask descregionpairspair
    desc_pos0 = desc_start + ablated_dim * tpd
    desc_pos1 = desc_start + ablated_dim * tpd + 1
    if desc_pos1 < L:
        inp[:, desc_pos0] = mask_id
        inp[:, desc_pos1] = mask_id

    # pairsremaining dimsper-mask→pretest
    target_dims = [d for d in range(N_DIMS) if d != ablated_dim]
    per_dim_ok = {d: 0 for d in target_dims}
    per_dim_total = {d: 0 for d in target_dims}

    model.eval()
    with torch.no_grad():
        for d in target_dims:
            p0 = desc_start + d * tpd
            p1 = desc_start + d * tpd + 1
            if p1 >= L:
                continue

            test_inp = inp.clone()
            test_inp[:, p0] = mask_id
            test_inp[:, p1] = mask_id

            for i in range(0, B, eval_bs):
                j = min(i + eval_bs, B)
                batch = test_inp[i:j].to(device)
                logits = model(batch)
                gold0 = tensor[i:j, p0]
                gold1 = tensor[i:j, p1]
                valid = (gold0 != pad_id) & (gold1 != pad_id)
                pred0 = logits[:, p0].argmax(dim=-1).cpu()
                pred1 = logits[:, p1].argmax(dim=-1).cpu()
                both = (pred0 == gold0) & (pred1 == gold1) & valid
                per_dim_ok[d] += both.sum().item()
                per_dim_total[d] += valid.sum().item()

    result = {}
    for d in target_dims:
        if per_dim_total[d] > 0:
            result[DIM_NAMES[d]] = round(per_dim_ok[d] / per_dim_total[d], 3)
        else:
            result[DIM_NAMES[d]] = 0.0
    return result


def eval_baseline(model, tensor, device, pad_id, mask_id,
                  desc_start, eval_bs=128):
    """baseline：no ablation，per-dimensionsmask→pretest"""
    B, L = tensor.shape
    tpd = TOKENS_PER_DIM
    per_dim_ok = [0] * N_DIMS
    per_dim_total = [0] * N_DIMS

    model.eval()
    with torch.no_grad():
        for d in range(N_DIMS):
            p0 = desc_start + d * tpd
            p1 = desc_start + d * tpd + 1
            if p1 >= L:
                continue

            inp = tensor.clone()
            inp[:, p0] = mask_id
            inp[:, p1] = mask_id

            for i in range(0, B, eval_bs):
                j = min(i + eval_bs, B)
                batch = inp[i:j].to(device)
                logits = model(batch)
                gold0 = tensor[i:j, p0]
                gold1 = tensor[i:j, p1]
                valid = (gold0 != pad_id) & (gold1 != pad_id)
                pred0 = logits[:, p0].argmax(dim=-1).cpu()
                pred1 = logits[:, p1].argmax(dim=-1).cpu()
                both = (pred0 == gold0) & (pred1 == gold1) & valid
                per_dim_ok[d] += both.sum().item()
                per_dim_total[d] += valid.sum().item()

    return {DIM_NAMES[d]: round(per_dim_ok[d] / max(per_dim_total[d], 1), 3)
            for d in range(N_DIMS)}


def run(model_path, world_path, d_model=64):
    print(f"Experiment 2C · dimension ablation")
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print("="*60)

    cfg = parse_world(world_path)
    data = generate(cfg)
    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']

    model = make_model(data['vocab_size'], d=d_model, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    results = {}

    # ═══ S2.1 Nameformat ═══
    print("\n── S2.1 Name: dimension ablation ──")
    s2_name = pretensorize(data['s2']['name'], pad_id)
    ds = FMT_META['s2_name'][1]     # 16
    p_start = ds - 13               # 3

    baseline = eval_baseline(model, s2_name, DEVICE, pad_id, mask_id, ds)
    results['baseline_s2_name'] = baseline
    print(f"  Baseline:")
    for dim, acc in baseline.items():
        print(f"    {dim}: {acc:.3f}")

    dep_matrix = {}
    diff_matrix = {}

    for abl_dim in range(N_DIMS):
        abl_name = DIM_NAMES[abl_dim]
        dim_result = eval_with_dim_ablated(
            model, s2_name, DEVICE, pad_id, mask_id,
            desc_start=ds, ablated_dim=abl_dim, p_start=p_start)
        dep_matrix[abl_name] = dim_result

        diffs = {}
        for tgt_name, acc in dim_result.items():
            diffs[tgt_name] = round(acc - baseline.get(tgt_name, 0), 3)
        diff_matrix[abl_name] = diffs

    results['dep_matrix_s2_name'] = dep_matrix
    results['diff_matrix_s2_name'] = diff_matrix

    # Printdifferencematrix
    print(f"\n  ── differencematrix（after ablation - baseline）──")
    print(f"  {'ablated':<8s}", end='')
    for d in DIM_NAMES:
        print(f" {d:>6s}", end='')
    print()

    for abl_name in DIM_NAMES:
        print(f"  {abl_name:<8s}", end='')
        for tgt_name in DIM_NAMES:
            if abl_name == tgt_name:
                print(f" {'---':>6s}", end='')
            else:
                diff = diff_matrix[abl_name].get(tgt_name, 0)
                print(f" {diff:+.3f}", end='')
        print()

    # ═══ S2.4 Interactformat ═══
    print("\n── S2.4 Interact: dimension ablation ──")
    s2_inter = pretensorize(data['s2']['interact'], pad_id)
    ds_i = FMT_META['s2_interact'][1]  # 22
    p_start_i = ds_i - 13              # 9

    baseline_i = eval_baseline(model, s2_inter, DEVICE, pad_id, mask_id, ds_i)
    results['baseline_s2_interact'] = baseline_i

    dep_matrix_i = {}
    diff_matrix_i = {}

    for abl_dim in range(N_DIMS):
        abl_name = DIM_NAMES[abl_dim]
        dim_result = eval_with_dim_ablated(
            model, s2_inter, DEVICE, pad_id, mask_id,
            desc_start=ds_i, ablated_dim=abl_dim, p_start=p_start_i)
        dep_matrix_i[abl_name] = dim_result
        diffs = {}
        for tgt_name, acc in dim_result.items():
            diffs[tgt_name] = round(acc - baseline_i.get(tgt_name, 0), 3)
        diff_matrix_i[abl_name] = diffs

    results['dep_matrix_s2_interact'] = dep_matrix_i
    results['diff_matrix_s2_interact'] = diff_matrix_i

    print(f"\n  ── S2.4 differencematrix ──")
    print(f"  {'ablated':<8s}", end='')
    for d in DIM_NAMES:
        print(f" {d:>6s}", end='')
    print()
    for abl_name in DIM_NAMES:
        print(f"  {abl_name:<8s}", end='')
        for tgt_name in DIM_NAMES:
            if abl_name == tgt_name:
                print(f" {'---':>6s}", end='')
            else:
                diff = diff_matrix_i[abl_name].get(tgt_name, 0)
                print(f" {diff:+.3f}", end='')
        print()

    # ═══ Save ═══
    out_dir = os.path.dirname(model_path) or '.'
    out_path = os.path.join(out_dir, 'exp2c_dim_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSave: {out_path}")
    print("="*60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--d', type=int, default=64)
    args = p.parse_args()
    run(args.model, args.world, d_model=args.d)
