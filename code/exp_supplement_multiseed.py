"""
Supplementary: Multi-seed stability verification


Three supplements:
  1. CGroupdisambiguation count — 5 seeds → desc_match vs name_match mean±std
  2. Gender Swap per-dim — 5 seeds → per-dim breakdown, focus on L dimension
  3. 2Dinterpretability — 3 seeds → attention specificity, linear probing, causal intervention

alluseuseexistingMulti-seedModel，no re-newtrain。

Usage：
  # Run all（2DOnly run3seed）
  python exp_supplement_multiseed.py --world swadesh_v6_world.md --seed-dir results_multiseed

  # Only runCgroup andGender Swap（fast，5minutes）
  python exp_supplement_multiseed.py --world swadesh_v6_world.md --seed-dir results_multiseed --skip-interp

  # Only run2D（slow，containcausal intervention）
  python exp_supplement_multiseed.py --world swadesh_v6_world.md --seed-dir results_multiseed --only-interp --causal
"""

import torch
import numpy as np
import argparse, json, os, glob

from model import make_model
from v6_tool import (
    parse_world, generate, build_tokens, FMT_META,
    enc2tok, enc2desc, get_gender, seq_s2_name, seq_s2_interact,
    attenuate, D_TAGS, RULE_CASE_B,
)
from v6_rules import ALL_DIMS
from train_core import pretensorize, eval_desc_acc, N_DIMS, TOKENS_PER_DIM

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']


# ═══════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════

def find_seed_models(seed_dir):
    """Scan seed directory，Return [(seed_id, model_path), ...]"""
    models = []
    for d in sorted(glob.glob(os.path.join(seed_dir, 'seed_*'))):
        mp = os.path.join(d, 'model.pt')
        if os.path.exists(mp):
            seed_id = os.path.basename(d)
            models.append((seed_id, mp))
    return models


def load_model(model_path, vocab_size, d_model=64):
    model = make_model(vocab_size, d=d_model, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ═══════════════════════════════════════════════════════
# Supplement 1：CGroupdisambiguation count（fromexp1_reverseExtract）
# ═══════════════════════════════════════════════════════

def eval_reverse_p(model, tensor, device, pad_id, mask_id, p_start, p_len=13, eval_bs=128):
    """reverseinference：maskall P positions，keep desc，predict P"""
    if tensor.shape[0] == 0:
        return 0.0, {}, []
    B, L = tensor.shape
    dim_names = ALL_DIMS
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
                    detail[dim_name] = {'gold': gold, 'pred': pred, 'correct': correct}
                sample_details.append(detail)

    per_dim = {}
    total_ok, total_n = 0, 0
    for d in range(p_len):
        name = dim_names[d] if d < len(dim_names) else f'D{d}'
        if per_dim_total[d] > 0:
            per_dim[name] = round(per_dim_ok[d] / per_dim_total[d], 4)
            total_ok += per_dim_ok[d]
            total_n += per_dim_total[d]
    overall = total_ok / total_n if total_n > 0 else 0.0
    return overall, per_dim, sample_details


def build_and_run_C_disambiguation(model, cfg, t2id, pad_id, mask_id):
    """
    BuildCGroupsequences，runreverseinference，Returndisambiguation count。
    Return: (desc_match_total, name_match_total, neither_total, per_pair_details)
    """
    ee = cfg['entity_encodings']
    tn = cfg['train_names']

    # selectmatchpairs（Same asexp1_reverseone，seed=271）
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
        name_a, name_b = tn[a_idx], tn[b_idx]
        enc_a, enc_b = ee[name_a], ee[name_b]
        diff = sum(1 for i in range(13) if enc_a[i] != enc_b[i])
        if diff >= 5:
            pairs.append((name_a, name_b))
            used.add((a_idx, b_idx))

    seqs, labels, expected = [], [], []
    for name_a, name_b in pairs:
        enc_b = ee[name_b]
        seq = seq_s2_name('touch', name_a, enc_b, cfg, t2id)
        seqs.append(seq)
        labels.append(f"{name_a}(name)+{name_b}(enc)")
        expected.append({
            'enc_p_tokens': enc2tok(enc_b, t2id),
            'name_p_tokens': enc2tok(ee[name_a], t2id),
        })

    if not seqs:
        return 0, 0, 0, []

    tensor = pretensorize(seqs, pad_id)
    _, _, details = eval_reverse_p(model, tensor, DEVICE, pad_id, mask_id, p_start=3)

    total_desc, total_name, total_neither = 0, 0, 0
    per_pair = []
    for idx in range(len(labels)):
        exp = expected[idx]
        det = details[idx]
        desc_m, name_m, neith = 0, 0, 0
        for d, dim_name in enumerate(ALL_DIMS):
            if dim_name not in det:
                continue
            pred = det[dim_name]['pred']
            gold_desc = exp['enc_p_tokens'][d]
            gold_name = exp['name_p_tokens'][d]
            if pred == gold_desc and pred != gold_name:
                desc_m += 1
            elif pred == gold_name and pred != gold_desc:
                name_m += 1
            elif pred != gold_desc and pred != gold_name:
                neith += 1
        total_desc += desc_m
        total_name += name_m
        total_neither += neith
        per_pair.append({'label': labels[idx], 'desc': desc_m, 'name': name_m, 'neither': neith})

    return total_desc, total_name, total_neither, per_pair


# ═══════════════════════════════════════════════════════
# Supplement 2：Gender Swap per-dim（fromexp2aExtract）
# ═══════════════════════════════════════════════════════

def build_gender_seqs_s2gender(cfg, t2id):
    """BuildS2.2 genderformat Full and Gender Swapsequences"""
    ee = cfg['entity_encodings']
    train_names = cfg['train_names']
    mask_id = t2id['MASK']

    seqs_full, seqs_swap = [], []
    for name in train_names:
        enc = ee[name]
        g = get_gender(enc, cfg)
        g_swap = '-in' if g == '-an' else '-an'
        for dtag in D_TAGS:
            att = attenuate(enc, dtag)
            p_tokens = enc2tok(att, t2id)
            desc_tokens = enc2desc(att, cfg, t2id)

            seqs_full.append([t2id[dtag], t2id['you'], t2id[name], t2id[g]] + p_tokens + desc_tokens)
            seqs_swap.append([t2id[dtag], t2id['you'], t2id[name], t2id[g_swap]] + p_tokens + desc_tokens)

    pad_id = t2id['PAD']
    return pretensorize(seqs_full, pad_id), pretensorize(seqs_swap, pad_id)


def eval_gender_swap_perdim(model, full_tensor, swap_tensor, pad_id, mask_id, desc_start):
    """runFull and Swap per-dim，Returntwo groupsper_dim dict"""
    _, full_dim = eval_desc_acc(model, full_tensor, DEVICE, pad_id, mask_id, desc_start=desc_start)
    _, swap_dim = eval_desc_acc(model, swap_tensor, DEVICE, pad_id, mask_id, desc_start=desc_start)
    return full_dim, swap_dim


# ═══════════════════════════════════════════════════════
# Supplement 3：2Dinterpretability（fromexp2dExtractcore）
# ═══════════════════════════════════════════════════════

import math

def get_base(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def extract_attention(model, input_ids, device):
    model.eval()
    base = get_base(model)
    input_ids = input_ids.to(device)
    B, L = input_ids.shape
    attention_maps = []
    with torch.no_grad():
        pos = torch.arange(L, device=device).unsqueeze(0)
        h = base.drop(
            base.tokens_embed(input_ids) * math.sqrt(base.d_model)
            + base.pos_embed(pos)
        )
        for layer in base.encoder.layers:
            normed = layer.norm1(h)
            attn_output, attn_weights = layer.self_attn(
                normed, normed, normed,
                need_weights=True, average_attn_weights=False
            )
            attention_maps.append(attn_weights.cpu().numpy())
            h = h + attn_output
            h = h + layer.linear2(layer.dropout(layer.activation(
                layer.linear1(layer.norm2(h)))))
    return attention_maps


def compute_attention_specificity(attention_maps, desc_start, p_start, n_dims=13):
    """Return {(layer, head, dim): specificity} dict"""
    spec = {}
    for layer_idx, attn in enumerate(attention_maps):
        B, n_heads, L, _ = attn.shape
        for head_idx in range(n_heads):
            for d in range(n_dims):
                desc_pos0 = desc_start + d * 2
                p_pos = p_start + d
                if desc_pos0 >= L or p_pos >= L:
                    continue
                target_attn = attn[:, head_idx, desc_pos0, p_pos].mean()
                p_region_attn = attn[:, head_idx, desc_pos0, p_start:p_start+n_dims].mean()
                s = float(target_attn / max(p_region_attn, 1e-8))
                spec[(layer_idx, head_idx, DIM_NAMES[d])] = s
    return spec


def extract_hidden_states(model, input_ids, device):
    model.eval()
    base = get_base(model)
    input_ids = input_ids.to(device)
    B, L = input_ids.shape
    hidden_states = []
    with torch.no_grad():
        pos = torch.arange(L, device=device).unsqueeze(0)
        h = base.drop(
            base.tokens_embed(input_ids) * math.sqrt(base.d_model)
            + base.pos_embed(pos)
        )
        hidden_states.append(h.cpu().numpy())
        for layer in base.encoder.layers:
            h = layer(h)
            hidden_states.append(h.cpu().numpy())
    return hidden_states


def run_probe(X, y):
    """
    Linear probe：usetorchhand-written，independent of sklearn，completely eliminate matmul overflow。
    Leave-one-outCross-validation。
    """
    X = np.array(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e4, 1e4)
    # Normalize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    X = (X - mu) / sd
    y = np.array(y)

    n_classes = len(set(y))
    if n_classes < 2:
        return 0.0

    N, D = X.shape
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    # Leave-one-out
    correct = 0
    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        X_train, y_train = X_t[mask], y_t[mask]
        X_test, y_test = X_t[i:i+1], y_t[i:i+1]

        # Closed-form solution：approximate with ridge regression
        # W = (X^T X + λI)^{-1} X^T Y_onehot
        lam = 1.0
        Y_oh = torch.zeros(N - 1, n_classes, dtype=torch.float32)
        for j in range(N - 1):
            Y_oh[j, y_train[j]] = 1.0

        XtX = X_train.T @ X_train + lam * torch.eye(D, dtype=torch.float32)
        XtY = X_train.T @ Y_oh
        try:
            W = torch.linalg.solve(XtX, XtY)
        except Exception:
            continue

        pred = (X_test @ W).argmax(dim=-1).item()
        if pred == y_test.item():
            correct += 1

    return round(correct / N, 3) if N > 0 else 0.0


def run_interp_one_seed(model, cfg, data, do_causal=False):
    """
    Run on one model2Dinterpretabilityfull suite，Returnstructured results。
    """
    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    ee = cfg['entity_encodings']
    train_names = cfg['train_names']

    ds = FMT_META['s2_name'][1]   # 16
    p_start = ds - 13              # 3

    # ── Attention ──
    sample_names = [n for n in ['fire', 'water', 'tree', 'stone', 'iron'] if n in ee]
    sample_seqs = [seq_s2_name('touch', n, ee[n], cfg, t2id) for n in sample_names]
    sample_tensor = pretensorize(sample_seqs, pad_id)
    attn_maps = extract_attention(model, sample_tensor, DEVICE)
    attn_spec = compute_attention_specificity(attn_maps, ds, p_start)

    # ── Linear Probing ──
    seqs, encodings = [], []
    for name in train_names:
        enc = ee[name]
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        seqs.append(seq)
        encodings.append(enc)
    tensor = pretensorize(seqs, pad_id)
    n_ent = len(train_names)
    name_pos = 2

    hidden_states = extract_hidden_states(model, tensor, DEVICE)

    # maskedversion
    masked_tensor = tensor.clone()
    for d in range(N_DIMS):
        p0 = ds + d * TOKENS_PER_DIM
        p1 = ds + d * TOKENS_PER_DIM + 1
        if p1 < masked_tensor.shape[1]:
            masked_tensor[:, p0] = mask_id
            masked_tensor[:, p1] = mask_id
    hidden_states_masked = extract_hidden_states(model, masked_tensor, DEVICE)

    # (a) nameposition
    probe_name = {}
    for layer_idx, hs in enumerate(hidden_states):
        X = hs[:, name_pos, :]
        acc_vals = []
        for d in range(N_DIMS):
            y = np.array([encodings[i][d] for i in range(n_ent)])
            acc_vals.append(run_probe(X, y))
        probe_name[layer_idx] = float(np.mean(acc_vals))

    # (b) cross-dim (last layer)
    last_hs = hidden_states[-1]
    diag_vals, off_vals = [], []
    for src_d in range(N_DIMS):
        X = last_hs[:, p_start + src_d, :]
        for tgt_d in range(N_DIMS):
            y = np.array([encodings[i][tgt_d] for i in range(n_ent)])
            v = run_probe(X, y)
            if src_d == tgt_d:
                diag_vals.append(v)
            else:
                off_vals.append(v)
    cross_diag = float(np.mean(diag_vals))
    cross_off = float(np.mean(off_vals))

    # (c) masked descposition
    probe_masked = {}
    for layer_idx, hs in enumerate(hidden_states_masked):
        acc_vals = []
        for d in range(N_DIMS):
            desc_pos0 = ds + d * TOKENS_PER_DIM
            X = hs[:, desc_pos0, :]
            y = np.array([encodings[i][d] for i in range(n_ent)])
            acc_vals.append(run_probe(X, y))
        probe_masked[layer_idx] = float(np.mean(acc_vals))

    result = {
        'attn_spec': attn_spec,
        'probe_name': probe_name,
        'cross_diag': cross_diag,
        'cross_off': cross_off,
        'probe_masked': probe_masked,
    }

    # ── causal intervention ──
    if do_causal:
        base = get_base(model)
        n_layers = len(list(base.encoder.layers))
        causal = {}
        for patch_layer in range(n_layers):
            dim_rates = []
            for target_dim in range(11):
                high_ents = [(n, ee[n]) for n in train_names if ee[n][target_dim] >= 3]
                low_ents = [(n, ee[n]) for n in train_names if ee[n][target_dim] <= 1]
                if len(high_ents) < 3 or len(low_ents) < 3:
                    continue
                success, total = 0, 0
                desc_pos0 = ds + target_dim * 2
                for h_name, h_enc in high_ents[:5]:
                    for l_name, l_enc in low_ents[:5]:
                        h_seq = seq_s2_name('touch', h_name, h_enc, cfg, t2id)
                        l_seq = seq_s2_name('touch', l_name, l_enc, cfg, t2id)
                        h_tensor = pretensorize([h_seq], pad_id).to(DEVICE)
                        l_tensor = pretensorize([l_seq], pad_id).to(DEVICE)
                        with torch.no_grad():
                            Ln = h_tensor.shape[1]
                            pos = torch.arange(Ln, device=DEVICE).unsqueeze(0)
                            h_low = base.drop(
                                base.tokens_embed(l_tensor) * math.sqrt(base.d_model)
                                + base.pos_embed(pos))
                            low_p_hidden = []
                            for layer in base.encoder.layers:
                                h_low = layer(h_low)
                                low_p_hidden.append(h_low[:, p_start:p_start+13, :].clone())

                            h_high = base.drop(
                                base.tokens_embed(h_tensor) * math.sqrt(base.d_model)
                                + base.pos_embed(pos))
                            h_orig = h_high.clone()
                            for layer in base.encoder.layers:
                                h_orig = layer(h_orig)
                            h_orig = base.norm(h_orig)
                            orig_pred = base.head(h_orig)[0, desc_pos0].argmax().item()

                            h_patched = h_high.clone()
                            for li, layer in enumerate(base.encoder.layers):
                                h_patched = layer(h_patched)
                                if li == patch_layer:
                                    h_patched[:, p_start:p_start+13, :] = low_p_hidden[li]
                            h_patched = base.norm(h_patched)
                            patched_pred = base.head(h_patched)[0, desc_pos0].argmax().item()
                            if patched_pred != orig_pred:
                                success += 1
                            total += 1
                if total > 0:
                    dim_rates.append(success / total)
            causal[patch_layer] = float(np.mean(dim_rates)) if dim_rates else 0.0
        result['causal'] = causal

    return result


# ═══════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Supplementary experiment · Multi-seed stability verification')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--seed-dir', default='results_multiseed', help='Multi-seedModeldirectory')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--skip-interp', action='store_true', help='Skip2Dinterpretability（Only runCdisambiguation+Gender per-dim）')
    p.add_argument('--only-interp', action='store_true', help='Only run2Dinterpretability')
    p.add_argument('--causal', action='store_true', help='2DinIncludecausal intervention（slow）')
    p.add_argument('--interp-seeds', type=int, default=3, help='2Dhow manyseed（default3）')
    args = p.parse_args()

    print(f"{'='*60}")
    print(f"  Supplementary experiment · Multi-seed stability verification")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")

    # Load world
    cfg = parse_world(args.world)
    data = generate(cfg)
    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    vocab_size = data['vocab_size']

    # scanModel
    seed_models = find_seed_models(args.seed_dir)
    print(f"  Found {len(seed_models)} seed models: {[s[0] for s in seed_models]}")

    all_results = {}

    # ═══════════════════════════════════════════════════
    # Supplement 1 + Supplement 2：Cdisambiguation + Gender Swap per-dim
    # ═══════════════════════════════════════════════════

    if not args.only_interp:
        print(f"\n{'─'*60}")
        print(f"  Supplement 1：CGroupdisambiguation count（{len(seed_models)} seeds）")
        print(f"{'─'*60}")

        c_results = []
        for seed_id, mp in seed_models:
            model = load_model(mp, vocab_size, args.d)
            desc_m, name_m, neither, pairs = build_and_run_C_disambiguation(
                model, cfg, t2id, pad_id, mask_id)
            print(f"  {seed_id}: desc={desc_m}  name={name_m}  neither={neither}")
            c_results.append({
                'seed': seed_id, 'desc': desc_m, 'name': name_m,
                'neither': neither, 'pairs': pairs
            })
            del model
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

        desc_vals = [r['desc'] for r in c_results]
        name_vals = [r['name'] for r in c_results]
        dm, ds = mean_std(desc_vals)
        nm, ns = mean_std(name_vals)
        print(f"\n  Summary: desc={dm:.1f}±{ds:.1f}  name={nm:.1f}±{ns:.1f}")
        all_results['C_disambiguation'] = {
            'per_seed': c_results,
            'desc_mean': round(dm, 1), 'desc_std': round(ds, 1),
            'name_mean': round(nm, 1), 'name_std': round(ns, 1),
        }

        print(f"\n{'─'*60}")
        print(f"  Supplement 2：Gender Swap per-dim（{len(seed_models)} seeds）")
        print(f"{'─'*60}")

        ds_gender = FMT_META['s2_gender'][1]  # 17
        swap_results = {dim: {'full': [], 'swap': [], 'diff': []} for dim in DIM_NAMES}

        for seed_id, mp in seed_models:
            model = load_model(mp, vocab_size, args.d)
            full_t, swap_t = build_gender_seqs_s2gender(cfg, t2id)
            full_dim, swap_dim = eval_gender_swap_perdim(
                model, full_t, swap_t, pad_id, mask_id, ds_gender)
            print(f"\n  {seed_id}:")
            for dim in DIM_NAMES:
                fv = full_dim.get(dim, 0)
                sv = swap_dim.get(dim, 0)
                diff = sv - fv
                swap_results[dim]['full'].append(fv)
                swap_results[dim]['swap'].append(sv)
                swap_results[dim]['diff'].append(diff)
                arrow = '↓' if diff < -0.02 else ('↑' if diff > 0.02 else '=')
                print(f"    {dim}: full={fv:.3f}  swap={sv:.3f}  Δ={diff:+.3f} {arrow}")
            del model
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

        print(f"\n  ── Gender Swap per-dim Summary ──")
        print(f"  {'Dim':<5s} {'Full':>12s} {'Swap':>12s} {'Δ':>14s}")
        swap_summary = {}
        for dim in DIM_NAMES:
            fm, fs = mean_std(swap_results[dim]['full'])
            sm, ss = mean_std(swap_results[dim]['swap'])
            dm_v, ds_v = mean_std(swap_results[dim]['diff'])
            print(f"  {dim:<5s} {fm:.3f}±{fs:.3f}   {sm:.3f}±{ss:.3f}   {dm_v:+.3f}±{ds_v:.3f}")
            swap_summary[dim] = {
                'full_mean': round(fm, 4), 'full_std': round(fs, 4),
                'swap_mean': round(sm, 4), 'swap_std': round(ss, 4),
                'diff_mean': round(dm_v, 4), 'diff_std': round(ds_v, 4),
            }
        all_results['gender_swap_perdim'] = swap_summary

    # ═══════════════════════════════════════════════════
    # Supplement 3：2Dinterpretability
    # ═══════════════════════════════════════════════════

    if not args.skip_interp:
        n_interp = min(args.interp_seeds, len(seed_models))
        interp_models = seed_models[:n_interp]

        print(f"\n{'─'*60}")
        print(f"  Supplement 3：2Dinterpretability（{n_interp} seeds, causal={args.causal}）")
        print(f"{'─'*60}")

        interp_results = []
        for seed_id, mp in interp_models:
            print(f"\n  ── {seed_id} ──")
            model = load_model(mp, vocab_size, args.d)
            res = run_interp_one_seed(model, cfg, data, do_causal=args.causal)
            interp_results.append({'seed': seed_id, **res})
            del model
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

        # Summaryattention specificityhigh values
        print(f"\n  ── Attention Specificity Summary ──")
        # collect allwith(layer, head, dim) specificity
        all_keys = set()
        for r in interp_results:
            all_keys.update(r['attn_spec'].keys())

        high_spec = {}
        for key in sorted(all_keys):
            vals = [r['attn_spec'].get(key, 0) for r in interp_results]
            m, s = mean_std(vals)
            if m > 2.0:
                high_spec[str(key)] = {'mean': round(m, 2), 'std': round(s, 2)}
                layer, head, dim = key
                print(f"    L{layer}·H{head} → {dim}: {m:.2f}±{s:.2f}")

        # Summarylinear probing
        print(f"\n  ── Linear Probing Summary ──")
        print(f"  {'':>20s} {'Nameposition':>12s} {'Masked desc':>12s}")
        for layer_idx in range(5):
            name_vals = [r['probe_name'].get(layer_idx, 0) for r in interp_results]
            mask_vals = [r['probe_masked'].get(layer_idx, 0) for r in interp_results]
            nm, ns = mean_std(name_vals)
            mm, ms = mean_std(mask_vals)
            print(f"  layer_{layer_idx:>14d}  {nm:.3f}±{ns:.3f}  {mm:.3f}±{ms:.3f}")

        # cross-dim
        diag_vals = [r['cross_diag'] for r in interp_results]
        off_vals = [r['cross_off'] for r in interp_results]
        dm, ds_d = mean_std(diag_vals)
        om, os_o = mean_std(off_vals)
        print(f"\n  Cross-dim (last layer):")
        print(f"    diagonal: {dm:.3f}±{ds_d:.3f}")
        print(f"    off-diagonal: {om:.3f}±{os_o:.3f}")

        # causal intervention
        if args.causal:
            print(f"\n  ── causal intervention Summary ──")
            for layer_idx in range(4):
                vals = [r.get('causal', {}).get(layer_idx, 0) for r in interp_results]
                m, s = mean_std(vals)
                print(f"    Patch layer {layer_idx}: {m:.3f}±{s:.3f}")

        all_results['interpretability'] = {
            'n_seeds': n_interp,
            'high_specificity': high_spec,
            'cross_diag': {'mean': round(dm, 3), 'std': round(ds_d, 3)},
            'cross_off': {'mean': round(om, 3), 'std': round(os_o, 3)},
            'per_seed': [{k: v for k, v in r.items() if k != 'attn_spec'}
                         for r in interp_results],
        }

    # ═══ Save ═══
    out_path = os.path.join(args.seed_dir, 'exp_supplement_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"  All results saved: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
