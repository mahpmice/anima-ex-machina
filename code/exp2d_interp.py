"""
Exp 2D: Interpretability (fixed version)


Fixes:
  - Linear Probe: Pregiontokenspositiontrivially true（tokens=P0-P4directly encode value）
    Changed to three meaningful probes：
    (a) nameposition：Does model propagate dimension info to name tokens？
    (b) cross-dim：P[d1]positioncan decoded2(≠d1)？cross-dim info integration
    (c) masked descposition：MASKafterdescpositionhidden statecan decodedim value？
  - causal intervention：patchentire P region（not just onetokens），test each layer

Usage：
  python exp2d_interp.py --model results_v4_ph5_bidir30/model.pt --world swadesh_v6_world.md
  python exp2d_interp.py --model results_v4_ph5_bidir30/model.pt --causal
"""

import torch
import numpy as np
import argparse, json, os, math

from model import make_model, SwaModel
from v6_tool import (parse_world, generate, FMT_META,
                     get_gender, enc2tok, enc2desc, seq_s2_name, attenuate, D_TAGS)
from train_core import pretensorize, N_DIMS, TOKENS_PER_DIM

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

DIM_NAMES = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']


def get_base(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


# ═══════════════════════════════════════════════════
# Level 1：Attention
# ═══════════════════════════════════════════════════

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


def analyze_attention_alignment(attention_maps, desc_start, p_start, n_dims=13):
    results = {}
    for layer_idx, attn in enumerate(attention_maps):
        B, n_heads, L, _ = attn.shape
        layer_results = {}
        for head_idx in range(n_heads):
            head_results = {}
            for d in range(n_dims):
                desc_pos0 = desc_start + d * 2
                p_pos = p_start + d
                if desc_pos0 >= L or p_pos >= L:
                    continue
                target_attn = attn[:, head_idx, desc_pos0, p_pos].mean()
                p_region_attn = attn[:, head_idx, desc_pos0, p_start:p_start+n_dims].mean()
                all_attn = attn[:, head_idx, desc_pos0, :].mean()
                head_results[DIM_NAMES[d]] = {
                    'target_p': round(float(target_attn), 4),
                    'avg_p_region': round(float(p_region_attn), 4),
                    'avg_all': round(float(all_attn), 4),
                    'specificity': round(float(target_attn / max(p_region_attn, 1e-8)), 3),
                }
            layer_results[f'head_{head_idx}'] = head_results
        results[f'layer_{layer_idx}'] = layer_results
    return results


# ═══════════════════════════════════════════════════
# Level 2：Linear Probing（fixed version）
# ═══════════════════════════════════════════════════

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
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
    except ImportError:
        return -1.0

    if len(set(y)) < 2:
        return 0.0

    n_min = min(np.bincount(y)[np.bincount(y) > 0])
    if n_min < 2:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=min(5, n_min), shuffle=True, random_state=42)

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    try:
        scores = cross_val_score(clf, X, y, cv=cv)
        return round(scores.mean(), 3)
    except Exception:
        return 0.0


def linear_probe_all(model, data, cfg, device):
    """
    Three probes：
    (a) nameposition(pos 2)hidden state of → each dim value
    (b) P[d1]position(last layer) → d2value（13×13 cross-dimmatrix）
    (c) masked descpositionhidden state of → dim value（info encoding before model prediction）
    """
    import warnings; warnings.filterwarnings('ignore')

    t2id = data['tokens2id']
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    ee = cfg['entity_encodings']
    train_names = cfg['train_names']

    seqs, encodings = [], []
    for name in train_names:
        enc = ee[name]
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        seqs.append(seq)
        encodings.append(enc)

    tensor = pretensorize(seqs, pad_id)
    ds = FMT_META['s2_name'][1]   # 16
    p_start = ds - 13              # 3
    name_pos = 2
    n_ent = len(train_names)

    hidden_states = extract_hidden_states(model, tensor, device)

    # maskedversion
    masked_tensor = tensor.clone()
    for d in range(N_DIMS):
        p0 = ds + d * TOKENS_PER_DIM
        p1 = ds + d * TOKENS_PER_DIM + 1
        if p1 < masked_tensor.shape[1]:
            masked_tensor[:, p0] = mask_id
            masked_tensor[:, p1] = mask_id
    hidden_states_masked = extract_hidden_states(model, masked_tensor, device)

    results = {}

    # ── (a) Nameposition ──
    print("\n  (a) Nameposition: name tokenscan decodeeach dim value？")
    probe_a = {}
    for layer_idx, hs in enumerate(hidden_states):
        X = hs[:, name_pos, :]
        layer_res = {}
        for d in range(N_DIMS):
            y = np.array([encodings[i][d] for i in range(n_ent)])
            layer_res[DIM_NAMES[d]] = run_probe(X, y)
        probe_a[f'layer_{layer_idx}'] = layer_res
        avg = np.mean(list(layer_res.values()))
        print(f"    layer_{layer_idx}: avg={avg:.3f}")
    results['probe_name_position'] = probe_a

    # ── (b) Cross-dim ──
    print("\n  (b) Cross-dim (last layer): P[d1]can decoded2？")
    last_hs = hidden_states[-1]
    cross_matrix = {}
    for source_dim in range(N_DIMS):
        source_pos = p_start + source_dim
        X = last_hs[:, source_pos, :]
        row = {}
        for target_dim in range(N_DIMS):
            y = np.array([encodings[i][target_dim] for i in range(n_ent)])
            row[DIM_NAMES[target_dim]] = run_probe(X, y)
        cross_matrix[DIM_NAMES[source_dim]] = row
    results['probe_cross_dim'] = cross_matrix

    diag = [cross_matrix[DIM_NAMES[d]][DIM_NAMES[d]] for d in range(N_DIMS)]
    off = [cross_matrix[DIM_NAMES[d1]][DIM_NAMES[d2]]
           for d1 in range(N_DIMS) for d2 in range(N_DIMS) if d1 != d2]
    print(f"    diagonal avg: {np.mean(diag):.3f}")
    print(f"    off-diagonal avg: {np.mean(off):.3f}")

    header = 'src\\tgt'
    print(f"\n    {header:<8s}", end='')
    for d in DIM_NAMES:
        print(f" {d:>5s}", end='')
    print()
    for src in DIM_NAMES:
        print(f"    {src:<8s}", end='')
        for tgt in DIM_NAMES:
            v = cross_matrix[src][tgt]
            print(f" {v:.2f}{'*' if src == tgt else ' '}", end='')
        print()

    # ── (c) Masked descposition ──
    print("\n  (c) Masked descposition: MASKpositionhidden statecan decodedim value？")
    probe_c = {}
    for layer_idx, hs in enumerate(hidden_states_masked):
        layer_res = {}
        for d in range(N_DIMS):
            desc_pos0 = ds + d * TOKENS_PER_DIM
            X = hs[:, desc_pos0, :]
            y = np.array([encodings[i][d] for i in range(n_ent)])
            layer_res[DIM_NAMES[d]] = run_probe(X, y)
        probe_c[f'layer_{layer_idx}'] = layer_res
        avg = np.mean(list(layer_res.values()))
        print(f"    layer_{layer_idx}: avg={avg:.3f}")
    results['probe_masked_desc'] = probe_c

    return results


# ═══════════════════════════════════════════════════
# Level 3：causal intervention（improved version）
# ═══════════════════════════════════════════════════

def causal_intervention(model, data, cfg, device):
    """patchentire P region，do at each layer separately"""
    t2id = data['tokens2id']
    pad_id = data['pad_id']
    ee = cfg['entity_encodings']
    train_names = cfg['train_names']

    base = get_base(model)
    model.eval()

    ds = FMT_META['s2_name'][1]
    p_start = ds - 13
    n_layers = len(list(base.encoder.layers))

    results_by_layer = {}

    for patch_layer in range(n_layers):
        print(f"\n    Patch after layer {patch_layer}:")
        dim_results = {}

        for target_dim in range(11):
            dim_name = DIM_NAMES[target_dim]
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

                    h_tensor = pretensorize([h_seq], pad_id).to(device)
                    l_tensor = pretensorize([l_seq], pad_id).to(device)

                    with torch.no_grad():
                        L = h_tensor.shape[1]
                        pos = torch.arange(L, device=device).unsqueeze(0)

                        h_low = base.drop(
                            base.tokens_embed(l_tensor) * math.sqrt(base.d_model)
                            + base.pos_embed(pos))
                        low_p_hidden = []
                        for layer in base.encoder.layers:
                            h_low = layer(h_low)
                            low_p_hidden.append(
                                h_low[:, p_start:p_start+13, :].clone())

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
                rate = round(success / total, 3)
                dim_results[dim_name] = {
                    'success_rate': rate,
                    'n_total': total, 'n_success': success,
                }
                print(f"      {dim_name}: {success}/{total} = {rate:.3f}")

        results_by_layer[f'patch_layer_{patch_layer}'] = dim_results

    print(f"\n    ── Summary ──")
    for layer_key, dim_res in results_by_layer.items():
        if dim_res:
            rates = [v['success_rate'] for v in dim_res.values()]
            print(f"    {layer_key}: avg={np.mean(rates):.3f}  "
                  f"max={max(rates):.3f}")

    return results_by_layer


def run(model_path, world_path, d_model=64, do_causal=False):
    print(f"Experiment 2D · interpretability（fixed version）")
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print("="*60)

    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    t2id = data['tokens2id']
    pad_id = data['pad_id']

    model = make_model(data['vocab_size'], d=d_model, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    results = {}

    # ═══ Level 1 ═══
    print("\n── Level 1: Attention ──")
    ee = data['cfg']['entity_encodings']
    sample_names = [n for n in ['fire', 'water', 'tree', 'stone', 'iron'] if n in ee]
    sample_seqs = [seq_s2_name('touch', n, ee[n], data['cfg'], t2id) for n in sample_names]
    sample_tensor = pretensorize(sample_seqs, pad_id)
    attn_maps = extract_attention(model, sample_tensor, DEVICE)

    ds = FMT_META['s2_name'][1]
    p_start = ds - 13
    alignment = analyze_attention_alignment(attn_maps, ds, p_start)
    results['attention_alignment'] = alignment

    print("\n  High specificity (>2.0):")
    for lk, ld in alignment.items():
        for hk, hd in ld.items():
            for dim, vals in hd.items():
                s = vals.get('specificity', 0)
                if s > 2.0:
                    print(f"    {lk} {hk} → {dim}: {s:.2f}")

    # ═══ Level 2 ═══
    print("\n── Level 2: Linear Probing ──")
    probe_results = linear_probe_all(model, data, data['cfg'], DEVICE)
    results.update(probe_results)

    # ═══ Level 3 ═══
    if do_causal:
        print("\n── Level 3: causal intervention ──")
        causal_results = causal_intervention(model, data, data['cfg'], DEVICE)
        results['causal_intervention'] = causal_results

    # ═══ Save ═══
    out_dir = os.path.dirname(model_path) or '.'
    out_path = os.path.join(out_dir, 'exp2d_interp_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    attn_path = os.path.join(out_dir, 'exp2d_attention_maps.npz')
    np.savez(attn_path, **{f'layer_{i}': am for i, am in enumerate(attn_maps)},
             sample_names=sample_names)

    print(f"\nSave: {out_path}")
    print("="*60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--causal', action='store_true')
    args = p.parse_args()
    run(args.model, args.world, d_model=args.d, do_causal=args.causal)
