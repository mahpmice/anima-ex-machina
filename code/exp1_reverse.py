"""
Exp 1: Language -> Perception reverse inference


Test if model can infer perceptual encoding (P-array) from language description (desc).
Model trained on P→desc (perception→language).
If model can do desc→P (language→perception), bidirectional bridge closes.

Method: Within MLM framework, same model, swap mask direction.
S2sequence contains bothParray and descdescription。trainmask desc，nowinmask P。

Six test groups：
  A. trained entities (S2 nameformat) — baseline
  B. novelentities — generalization
  C. name-desc mismatch — test if model uses desc or name
  D. fictitious encoding (never-seen combinations) — rule-level reverse inference
  E. interaction results (S2 interactformat) — reverse inference of causal results
  F. held-outinteractionpairs — unseen combinations

Usage：
  python exp1_reverse.py --checkpoint results_v4_ph6/model.pt --world swadesh_v6_world.md
  python exp1_reverse.py --checkpoint results_v4_ph6/model.pt --world swadesh_v6_world.md --d 128
"""

import torch
import numpy as np
import argparse, json, os

from model import make_model, count_params
from v6_tool import (
    parse_world, generate, build_tokens, FMT_META,
    enc2tok, enc2desc, val_to_degree, get_gender,
    seq_s2_name, seq_s2_interact, seq_s2_gender,
    D_TAGS, DEGREE_MARKS, DIM_BASES, TOKENS_PER_DIM,
    ANIMACY_PROBE_PAIRS, RULE_CASE_B,
)
from v6_rules import (
    ALL_DIMS, DIM, attenuate, generate_all_d_layers, compute_result,
)
from train_core import pretensorize

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════
# Core evaluation function：reverseinference (desc→P)
# ═══════════════════════════════════════════════════════

def eval_reverse_p(model, tensor, device, pad_id, mask_id,
                   p_start, p_len=13, eval_bs=128):
    """
    Reverse inference evaluation：maskall P positions，keep desc，predict P。
    
    Symmetric with forward eval (mask desc)。
    
    Return: (overall_acc, per_dim_dict, per_sample_details)
    per_sample_details: list of dicts, per-dim prediction result for each sequence
    """
    if tensor.shape[0] == 0:
        return 0.0, {}, []

    B, L = tensor.shape
    dim_names = ALL_DIMS  # ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    
    # maskall P positions
    inp = tensor.clone()
    for d in range(p_len):
        pos = p_start + d
        if pos < L:
            inp[:, pos] = mask_id

    per_dim_ok = [0] * p_len
    per_dim_total = [0] * p_len
    sample_details = []

    model.eval()
    with torch.no_grad():
        for i in range(0, B, eval_bs):
            j = min(i + eval_bs, B)
            batch = inp[i:j].to(device)
            logits = model(batch)

            for b_idx in range(j - i):
                detail = {}
                for d in range(p_len):
                    pos = p_start + d
                    if pos >= L:
                        break
                    gold = tensor[i + b_idx, pos].item()
                    if gold == pad_id:
                        continue
                    pred = logits[b_idx, pos].argmax(dim=-1).item()
                    correct = (pred == gold)

                    per_dim_ok[d] += int(correct)
                    per_dim_total[d] += 1

                    dim_name = dim_names[d] if d < len(dim_names) else f'D{d}'
                    detail[dim_name] = {
                        'gold': gold, 'pred': pred, 'correct': correct
                    }
                sample_details.append(detail)

    per_dim = {}
    total_ok, total_n = 0, 0
    for d in range(p_len):
        name = dim_names[d] if d < len(dim_names) else f'D{d}'
        if per_dim_total[d] > 0:
            per_dim[name] = round(per_dim_ok[d] / per_dim_total[d], 4)
            total_ok += per_dim_ok[d]
            total_n += per_dim_total[d]
        else:
            per_dim[name] = 0.0

    overall = total_ok / total_n if total_n > 0 else 0.0
    return overall, per_dim, sample_details


def eval_reverse_with_probs(model, tensor, device, pad_id, mask_id,
                            p_start, p_len=13, eval_bs=128):
    """
    Reverse eval with probability output。Output for each P positiontop-3 prediction and confidence。
    useshown in paperModelinreverseduring inferencecalibration characteristics。
    """
    if tensor.shape[0] == 0:
        return []

    B, L = tensor.shape
    dim_names = ALL_DIMS

    inp = tensor.clone()
    for d in range(p_len):
        pos = p_start + d
        if pos < L:
            inp[:, pos] = mask_id

    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, B, eval_bs):
            j = min(i + eval_bs, B)
            batch = inp[i:j].to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)

            for b_idx in range(j - i):
                sample = {}
                for d in range(p_len):
                    pos = p_start + d
                    if pos >= L:
                        break
                    gold = tensor[i + b_idx, pos].item()
                    if gold == pad_id:
                        continue
                    
                    p = probs[b_idx, pos]
                    top3_vals, top3_ids = p.topk(3)
                    
                    dim_name = dim_names[d] if d < len(dim_names) else f'D{d}'
                    sample[dim_name] = {
                        'gold': gold,
                        'gold_prob': p[gold].item(),
                        'pred': top3_ids[0].item(),
                        'pred_prob': top3_vals[0].item(),
                        'correct': top3_ids[0].item() == gold,
                        'top3': [(top3_ids[k].item(), round(top3_vals[k].item(), 4))
                                 for k in range(3)],
                    }
                results.append(sample)

    return results


# ═══════════════════════════════════════════════════════
# Fictitious encoding generation
# ═══════════════════════════════════════════════════════

def generate_fictional_encodings(existing_encs, n=10, seed=314):
    """
    Generate n fictitious encodings not overlapping with existing entities。
    perceptual dimensions(0-10): 0-4, Judgment dimensions(11-12): 0/1
    constraint: L=1 → An=1
    """
    rng = np.random.RandomState(seed)
    existing_set = set(tuple(e) for e in existing_encs)
    fictional = []

    attempts = 0
    while len(fictional) < n and attempts < n * 100:
        attempts += 1
        enc = [0] * 13
        # perceptual dimensions (T H M V Z A W R O S F): random0-4
        for i in range(11):
            enc[i] = rng.randint(0, 5)
        # L: random 0/1
        enc[11] = rng.randint(0, 2)
        # An: IfL=1thenAn=1，thenrandom
        if enc[11] == 1:
            enc[12] = 1
        else:
            enc[12] = rng.randint(0, 2)

        t = tuple(enc)
        if t not in existing_set:
            existing_set.add(t)
            fictional.append(enc)

    return fictional


# ═══════════════════════════════════════════════════════
# Test group construction
# ═══════════════════════════════════════════════════════

def build_test_group_A(cfg, t2id):
    """Agroup: trained entities，S2 nameformat，touchdistance"""
    seqs, labels = [], []
    ee = cfg['entity_encodings']
    for name in cfg['train_names']:
        enc = ee[name]
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        seqs.append(seq)
        labels.append(name)
    return seqs, labels


def build_test_group_B(cfg, t2id):
    """Bgroup: novelentities，S2 nameformat，touchdistance"""
    seqs, labels = [], []
    ee = cfg['entity_encodings']
    for name in cfg['novel_names']:
        enc = ee[name]
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        seqs.append(seq)
        labels.append(name)
    return seqs, labels


def build_test_group_C(cfg, t2id):
    """
    Cgroup: name-desc mismatch。
    give entitiesA's name, withentitiesB's encoding(P+desc)。
    IfModelusedescpredict P → outputB P → provesreversemapping based on languageRule
    IfModelusenamepredict P → outputA P → indicatesreversemapping relies on name memory
    """
    seqs, labels, expected = [], [], []
    ee = cfg['entity_encodings']
    tn = cfg['train_names']

    # select10different entity pairs cross-matched
    rng = np.random.RandomState(271)
    pairs = []
    used = set()
    attempts = 0
    while len(pairs) < min(10, len(tn) // 2) and attempts < 200:
        attempts += 1
        a_idx = rng.randint(0, len(tn))
        b_idx = rng.randint(0, len(tn))
        if a_idx == b_idx or (a_idx, b_idx) in used:
            continue
        name_a = tn[a_idx]
        name_b = tn[b_idx]
        # ensure encoding difference large enough（at least5dims different）
        enc_a = ee[name_a]
        enc_b = ee[name_b]
        diff = sum(1 for i in range(13) if enc_a[i] != enc_b[i])
        if diff >= 5:
            pairs.append((name_a, name_b))
            used.add((a_idx, b_idx))

    for name_a, name_b in pairs:
        enc_b = ee[name_b]
        # usename_a's name, withname_b's encoding
        seq = seq_s2_name('touch', name_a, enc_b, cfg, t2id)
        seqs.append(seq)
        labels.append(f"{name_a}(name)+{name_b}(enc)")
        expected.append({
            'name_entity': name_a,
            'enc_entity': name_b,
            'enc_p_tokens': enc2tok(enc_b, t2id),
            'name_p_tokens': enc2tok(ee[name_a], t2id),
        })

    return seqs, labels, expected


def build_test_group_D(cfg, t2id):
    """Dgroup: fictitious encoding，S2 nameformat"""
    ee = cfg['entity_encodings']
    all_encs = list(ee.values())
    fictionals = generate_fictional_encodings(all_encs, n=20, seed=314)

    seqs, labels, encs_out = [], [], []
    # usetrained entities's name cycled（name does not matter——if C proves model uses desc not name）
    tn = cfg['train_names']
    for i, enc in enumerate(fictionals):
        name = tn[i % len(tn)]  # borrow names in rotation
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        seqs.append(seq)
        labels.append(f"fictional_{i}({name})")
        encs_out.append(enc)

    return seqs, labels, encs_out


def build_test_group_E(cfg, t2id):
    """Egroup: traininteraction results，S2 interactformat，touchdistance"""
    seqs, labels = [], []
    ee = cfg['entity_encodings']
    for a, b, rule, enc_r in cfg['interactions']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        seq = seq_s2_interact('touch', a, ga, b, gb, rule, enc_r, cfg, t2id, case_b=cb)
        seqs.append(seq)
        labels.append(f"{a}+{b}({rule})")
    return seqs, labels


def build_test_group_F(cfg, t2id):
    """Fgroup: held-outinteractionpairs(probes)，S2 interactformat"""
    seqs, labels = [], []
    ee = cfg['entity_encodings']
    for a, b, rule, desc_txt, enc_r in cfg['probes']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        seq = seq_s2_interact('touch', a, ga, b, gb, rule, enc_r, cfg, t2id, case_b=cb)
        seqs.append(seq)
        labels.append(f"{a}+{b}({rule}) [probe]")
    return seqs, labels


# ═══════════════════════════════════════════════════════
# Print
# ═══════════════════════════════════════════════════════

def print_group_result(group_name, overall, per_dim, labels=None, details=None):
    """Print one test group result"""
    print(f"\n{'─'*60}")
    print(f"  {group_name}")
    print(f"{'─'*60}")
    print(f"  Overall: {overall:.4f} ({overall*100:.1f}%)")
    print(f"  Per-dim:")
    for dim, acc in per_dim.items():
        tag = '✓' if acc >= 0.8 else ('~' if acc >= 0.5 else '✗')
        print(f"    [{tag}] {dim:>3s}: {acc:.4f}")

    if labels and details and len(labels) == len(details):
        print(f"\n  Per-sample (first 20):")
        for idx in range(min(20, len(labels))):
            d = details[idx]
            n_correct = sum(1 for v in d.values() if v.get('correct', False))
            n_total = len(d)
            print(f"    {labels[idx]:<30s}  {n_correct}/{n_total}")


def print_mismatch_result(labels, expected, details, id2tokens):
    """CGroupspecialized：Printname-desc mismatch diagnosis"""
    print(f"\n  Disambiguation (name vs desc):")
    for idx in range(len(labels)):
        exp = expected[idx]
        det = details[idx]

        # pairsper dim，determine if prediction closer toname Pordesc P
        name_match, desc_match, neither = 0, 0, 0
        for d, dim_name in enumerate(ALL_DIMS):
            if dim_name not in det:
                continue
            pred = det[dim_name]['pred']
            gold_desc = exp['enc_p_tokens'][d]   # descpointing to P
            gold_name = exp['name_p_tokens'][d]  # namepointing to P

            if pred == gold_desc and pred != gold_name:
                desc_match += 1
            elif pred == gold_name and pred != gold_desc:
                name_match += 1
            elif pred == gold_desc and pred == gold_name:
                pass  # happen to be same，not counted
            else:
                neither += 1

        total_diag = name_match + desc_match + neither
        if total_diag > 0:
            print(f"    {labels[idx]:<40s}  "
                  f"desc={desc_match}  name={name_match}  neither={neither}")


# ═══════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════

def run(checkpoint, world_path, d_model=64, n_layers=4, n_heads=4, max_len=48):
    print(f"{'='*60}")
    print(f"  Exp 1: Language -> Perception reverse inference")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  World: {world_path}")
    print(f"  Device: {DEVICE}")
    print(f"  d_model={d_model}")

    # ── Load world ──
    cfg = parse_world(world_path)
    tokens = build_tokens(cfg)
    t2id = {t: i for i, t in enumerate(tokens)}
    id2tokens = {i: t for t, i in t2id.items()}
    pad_id = t2id['PAD']
    mask_id = t2id['MASK']

    # ── Load model ──
    model = make_model(len(tokens), d=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_len=max_len, dropout=0.0)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    print(f"  Model: {count_params(model):,} params")

    results = {}

    # ════════════════════════════════════════════════
    # Agroup: trained entities
    # ════════════════════════════════════════════════
    seqs_a, labels_a = build_test_group_A(cfg, t2id)
    tensor_a = pretensorize(seqs_a, pad_id)
    # S2 name: P starts at index 3
    acc_a, dim_a, det_a = eval_reverse_p(
        model, tensor_a, DEVICE, pad_id, mask_id, p_start=3)
    print_group_result("A · trained entities (S2 name, touch)", acc_a, dim_a, labels_a, det_a)
    results['A_train'] = {'overall': round(acc_a, 4), 'per_dim': dim_a}

    # ════════════════════════════════════════════════
    # Bgroup: novelentities
    # ════════════════════════════════════════════════
    seqs_b, labels_b = build_test_group_B(cfg, t2id)
    if seqs_b:
        tensor_b = pretensorize(seqs_b, pad_id)
        acc_b, dim_b, det_b = eval_reverse_p(
            model, tensor_b, DEVICE, pad_id, mask_id, p_start=3)
        print_group_result("B · novelentities (S2 name, touch)", acc_b, dim_b, labels_b, det_b)
        results['B_novel'] = {'overall': round(acc_b, 4), 'per_dim': dim_b}
    else:
        print("\n  Bgroup: no novelentities，Skip")

    # ════════════════════════════════════════════════
    # Cgroup: name-desc mismatch
    # ════════════════════════════════════════════════
    seqs_c, labels_c, expected_c = build_test_group_C(cfg, t2id)
    if seqs_c:
        tensor_c = pretensorize(seqs_c, pad_id)
        acc_c, dim_c, det_c = eval_reverse_p(
            model, tensor_c, DEVICE, pad_id, mask_id, p_start=3)
        print_group_result("C · name-desc mismatch", acc_c, dim_c, labels_c, det_c)
        print_mismatch_result(labels_c, expected_c, det_c, id2tokens)
        results['C_mismatch'] = {'overall': round(acc_c, 4), 'per_dim': dim_c}
    else:
        print("\n  Cgroup: no pairs，Skip")

    # ════════════════════════════════════════════════
    # Dgroup: fictitious encoding
    # ════════════════════════════════════════════════
    seqs_d, labels_d, encs_d = build_test_group_D(cfg, t2id)
    tensor_d = pretensorize(seqs_d, pad_id)
    acc_d, dim_d, det_d = eval_reverse_p(
        model, tensor_d, DEVICE, pad_id, mask_id, p_start=3)
    print_group_result("D · fictitious encoding (never-seen)", acc_d, dim_d, labels_d, det_d)
    results['D_fictional'] = {'overall': round(acc_d, 4), 'per_dim': dim_d}

    # detailed output with probabilities（fictitious encoding）
    prob_results = eval_reverse_with_probs(
        model, tensor_d, DEVICE, pad_id, mask_id, p_start=3)
    if prob_results:
        print(f"\n  DGroup confidenceDetails (first 5fictitious encoding):")
        for idx in range(min(5, len(prob_results))):
            pr = prob_results[idx]
            print(f"    {labels_d[idx]}:")
            for dim_name in ALL_DIMS:
                if dim_name not in pr:
                    continue
                info = pr[dim_name]
                gold_tok = id2tokens.get(info['gold'], '?')
                pred_tok = id2tokens.get(info['pred'], '?')
                mark = '✓' if info['correct'] else '✗'
                print(f"      [{mark}] {dim_name}: gold={gold_tok} "
                      f"pred={pred_tok} (p={info['pred_prob']:.3f})  "
                      f"gold_p={info['gold_prob']:.3f}")

    # ════════════════════════════════════════════════
    # Egroup: interaction results
    # ════════════════════════════════════════════════
    seqs_e, labels_e = build_test_group_E(cfg, t2id)
    tensor_e = pretensorize(seqs_e, pad_id)
    # S2 interact: P starts at index 9
    acc_e, dim_e, det_e = eval_reverse_p(
        model, tensor_e, DEVICE, pad_id, mask_id, p_start=9)
    print_group_result("E · interaction results (S2 interact, touch)", acc_e, dim_e, labels_e, det_e)
    results['E_interact'] = {'overall': round(acc_e, 4), 'per_dim': dim_e}

    # ════════════════════════════════════════════════
    # Fgroup: held-outinteractionpairs
    # ════════════════════════════════════════════════
    seqs_f, labels_f = build_test_group_F(cfg, t2id)
    if seqs_f:
        tensor_f = pretensorize(seqs_f, pad_id)
        acc_f, dim_f, det_f = eval_reverse_p(
            model, tensor_f, DEVICE, pad_id, mask_id, p_start=9)
        print_group_result("F · held-outinteractionpairs (probes)", acc_f, dim_f, labels_f, det_f)
        results['F_probes'] = {'overall': round(acc_f, 4), 'per_dim': dim_f}

        # probeper-pairoutput
        prob_f = eval_reverse_with_probs(
            model, tensor_f, DEVICE, pad_id, mask_id, p_start=9)
        print(f"\n  FGroupper-pairDetails:")
        for idx in range(len(labels_f)):
            det = det_f[idx]
            n_ok = sum(1 for v in det.values() if v.get('correct', False))
            print(f"    {labels_f[idx]:<35s}  {n_ok}/13")
    else:
        print("\n  Fgroup: no probes，Skip")

    # ════════════════════════════════════════════════
    # Forward comparison (Run forward on same data as control)
    # ════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  Control: Forward accuracy on same dataset (mask desc, predict from P)")
    print(f"{'─'*60}")

    from train_core import eval_desc_acc, eval_full_mask

    # AGroupforward
    ds_name = FMT_META['s2_name'][1]
    fwd_a, fwd_a_dim = eval_desc_acc(
        model, tensor_a, DEVICE, pad_id, mask_id, desc_start=ds_name)
    print(f"  A-forward (train, desc from P):   {fwd_a:.4f}")

    # EGroupforward
    ds_inter = FMT_META['s2_interact'][1]
    fwd_e, fwd_e_dim = eval_desc_acc(
        model, tensor_e, DEVICE, pad_id, mask_id, desc_start=ds_inter)
    print(f"  E-forward (interact, desc from P): {fwd_e:.4f}")

    results['forward_A'] = round(fwd_a, 4)
    results['forward_E'] = round(fwd_e, 4)

    # ════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Test group':<30s}  {'reverse(desc→P)':>12s}  {'forward(P→desc)':>12s}")
    print(f"  {'─'*56}")
    print(f"  {'A · trained entities':<30s}  {acc_a:>12.4f}  {fwd_a:>12.4f}")
    if seqs_b:
        print(f"  {'B · novelentities':<30s}  {acc_b:>12.4f}  {'—':>12s}")
    if seqs_c:
        print(f"  {'C · name-desc mismatch':<28s}  {acc_c:>12.4f}  {'—':>12s}")
    print(f"  {'D · fictitious encoding':<30s}  {acc_d:>12.4f}  {'—':>12s}")
    print(f"  {'E · interaction results':<30s}  {acc_e:>12.4f}  {fwd_e:>12.4f}")
    if seqs_f:
        print(f"  {'F · held-outinteractionpairs':<28s}  {acc_f:>12.4f}  {'—':>12s}")

    # ── Save ──
    out_dir = os.path.dirname(checkpoint) or '.'
    out_path = os.path.join(out_dir, 'exp1_reverse_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Experimentone：language→perceptionreverseinference')
    p.add_argument('--checkpoint', required=True,
                   help='Modelcheckpointpath (e.g. results_v4_ph6/model.pt)')
    p.add_argument('--world', default='swadesh_v6_world.md',
                   help='world config file path')
    p.add_argument('--d', type=int, default=64,
                   help='Modeld_model (default64，Ifused=128Modelchange to128)')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--max-len', type=int, default=48)
    args = p.parse_args()

    run(args.checkpoint, args.world,
        d_model=args.d, n_layers=args.layers,
        n_heads=args.heads, max_len=args.max_len)
