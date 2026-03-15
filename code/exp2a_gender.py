"""
Exp 2A: Category marker isolation contribution


Use trained bidir30 model (Phase 5), no retraining.
All tested on S2 format (with P array).

Usage:
  python exp2a_gender.py --model results_v4_ph5_bidir30/model.pt --world swadesh_v6_world.md
"""

import torch
import numpy as np
import argparse, json, os

from model import make_model
from v6_tool import (parse_world, generate, build_tokens, FMT_META,
                     get_gender, enc2tok, enc2desc, attenuate,
                     RULE_CASE_B, D_TAGS)
from train_core import pretensorize, eval_desc_acc

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']


def build_gender_ablation_s2gender(cfg, t2id):
    """
    S2.2 gender format ablation：[dtag, you, name, gender, P×13, desc×26] = 43
    genderat position3。desc_start=17, P at [4,17)

    ConditionA — Full:        standard sequence
    ConditionB — No Gender:   genderposition(pos 3)→MASK
    ConditionC — No Name:     nameposition(pos 2)→MASK
    ConditionD — Gender Swap: genderpositionreplace with opposite marker
    """
    ee = cfg['entity_encodings']
    train_names = cfg['train_names']
    mask_id = t2id['MASK']

    seqs_A, seqs_B, seqs_C, seqs_D = [], [], [], []
    labels = []

    for name in train_names:
        enc = ee[name]
        g = get_gender(enc, cfg)
        g_swap = '-in' if g == '-an' else '-an'

        for dtag in D_TAGS:
            att = attenuate(enc, dtag)
            p_tokens = enc2tok(att, t2id)
            desc_tokens = enc2desc(att, cfg, t2id)

            seq_a = [t2id[dtag], t2id['you'], t2id[name], t2id[g]] + p_tokens + desc_tokens
            seqs_A.append(seq_a)

            seq_b = [t2id[dtag], t2id['you'], t2id[name], mask_id] + p_tokens + desc_tokens
            seqs_B.append(seq_b)

            seq_c = [t2id[dtag], t2id['you'], mask_id, t2id[g]] + p_tokens + desc_tokens
            seqs_C.append(seq_c)

            seq_d = [t2id[dtag], t2id['you'], t2id[name], t2id[g_swap]] + p_tokens + desc_tokens
            seqs_D.append(seq_d)

            labels.append((name, dtag, g))

    pad_id = t2id['PAD']
    return {
        'A_full': pretensorize(seqs_A, pad_id),
        'B_no_gender': pretensorize(seqs_B, pad_id),
        'C_no_name': pretensorize(seqs_C, pad_id),
        'D_gender_swap': pretensorize(seqs_D, pad_id),
    }, labels


def build_gender_ablation_s2interact(cfg, t2id):
    """
    S2.4 interactformat ablation：[dtag, you, a, ga, -us, b, gb, case_b, rule, P×13, desc×26] = 48
    gaat position3, gbat position6。desc_start=22, P at [9,22)
    """
    ee = cfg['entity_encodings']
    mask_id = t2id['MASK']

    seqs_A, seqs_B, seqs_C, seqs_D = [], [], [], []
    labels = []

    for a, b, rule, enc_r in cfg['interactions']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')

        for dtag in D_TAGS:
            att = attenuate(enc_r, dtag)
            p_tokens = enc2tok(att, t2id)
            desc_tokens = enc2desc(att, cfg, t2id)

            prefix = [t2id[dtag], t2id['you'],
                      t2id[a], t2id[ga], t2id['-us'],
                      t2id[b], t2id[gb], t2id[cb], t2id[rule]]

            seqs_A.append(prefix + p_tokens + desc_tokens)

            p_b = list(prefix); p_b[3] = mask_id
            seqs_B.append(p_b + p_tokens + desc_tokens)

            p_c = list(prefix); p_c[6] = mask_id
            seqs_C.append(p_c + p_tokens + desc_tokens)

            p_d = list(prefix); p_d[3] = mask_id; p_d[6] = mask_id
            seqs_D.append(p_d + p_tokens + desc_tokens)

            labels.append((a, b, rule, dtag, ga, gb))

    pad_id = t2id['PAD']
    return {
        'A_full': pretensorize(seqs_A, pad_id),
        'B_no_ga': pretensorize(seqs_B, pad_id),
        'C_no_gb': pretensorize(seqs_C, pad_id),
        'D_no_both': pretensorize(seqs_D, pad_id),
    }, labels


def run(model_path, world_path, d_model=64):
    print(f"Exp 2A: Category marker isolation contribution")
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

    # ═══ Part 1: S2.2 Gender ═══
    print("\n── Part 1: S2.2 Gender ──")
    gender_seqs, gender_labels = build_gender_ablation_s2gender(cfg, t2id)
    ds = FMT_META['s2_gender'][1]  # 17

    for cond_name, tensor in gender_seqs.items():
        acc, dim_acc = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds)
        results[f'gender_{cond_name}'] = {'overall': round(acc, 4), 'per_dim': dim_acc}
        print(f"  {cond_name:<20s}  overall={acc:.4f}")
        for dim, a in dim_acc.items():
            print(f"    {dim}: {a:.3f}")

    print("\n  ── difference（vs Full）──")
    base_dims = results['gender_A_full']['per_dim']
    for cond in ['gender_B_no_gender', 'gender_C_no_name', 'gender_D_gender_swap']:
        cond_dims = results[cond]['per_dim']
        print(f"\n  {cond}:")
        for dim in DIM_NAMES:
            diff = cond_dims.get(dim, 0) - base_dims.get(dim, 0)
            arrow = '↓' if diff < -0.02 else ('↑' if diff > 0.02 else '=')
            print(f"    {dim}: {diff:+.3f} {arrow}")

    # ═══ Part 2: S2.4 Interact ═══
    print("\n── Part 2: S2.4 Interact ──")
    interact_seqs, interact_labels = build_gender_ablation_s2interact(cfg, t2id)
    ds_i = FMT_META['s2_interact'][1]  # 22

    for cond_name, tensor in interact_seqs.items():
        acc, dim_acc = eval_desc_acc(model, tensor, DEVICE, pad_id, mask_id, desc_start=ds_i)
        results[f'interact_{cond_name}'] = {'overall': round(acc, 4), 'per_dim': dim_acc}
        print(f"  {cond_name:<20s}  overall={acc:.4f}")

    print("\n  ── Interactdifference（vs Full）──")
    base_i = results['interact_A_full']['per_dim']
    for cond in ['interact_B_no_ga', 'interact_C_no_gb', 'interact_D_no_both']:
        cond_dims = results[cond]['per_dim']
        print(f"\n  {cond}:")
        for dim in DIM_NAMES:
            diff = cond_dims.get(dim, 0) - base_i.get(dim, 0)
            arrow = '↓' if diff < -0.02 else ('↑' if diff > 0.02 else '=')
            print(f"    {dim}: {diff:+.3f} {arrow}")

    # ═══ Part 3: byAnimacygrouping ═══
    print("\n── Part 3: Animacygrouping ──")
    ee = cfg['entity_encodings']
    ai = cfg['all_dims'].index('An')

    an1_idx, an0_idx = [], []
    for i, name in enumerate(cfg['train_names']):
        for d_idx in range(len(D_TAGS)):
            seq_idx = i * len(D_TAGS) + d_idx
            if ee[name][ai] == 1:
                an1_idx.append(seq_idx)
            else:
                an0_idx.append(seq_idx)

    for group_name, indices in [('An=1_animate', an1_idx), ('An=0_inanimate', an0_idx)]:
        if not indices:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long)
        print(f"\n  {group_name} (n={len(indices)}):")
        for cond_name, tensor in gender_seqs.items():
            sub = tensor[idx_t]
            acc, _ = eval_desc_acc(model, sub, DEVICE, pad_id, mask_id, desc_start=ds)
            print(f"    {cond_name}: {acc:.4f}")
            results[f'{group_name}_{cond_name}'] = {'overall': round(acc, 4)}

    # ═══ Save ═══
    out_dir = os.path.dirname(model_path) or '.'
    out_path = os.path.join(out_dir, 'exp2a_gender_results.json')
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
