"""
Experiment 3: Per-dimension Confidence Calibration
═══════════════════════════════════════════════
Purpose: Fill paper §3.1.3 confidence calibration
Config: 26-token main system (bidir30), 5 seeds, pre-trained models

Run forward evaluation on 5 novel entities, export softmax probability at each DESC position.
Core relation: high confidence when correct=1 vs low confidence when correct=0
         → model knows what it is uncertain about

Usage:
  python exp3_confidence.py <bidir_30pct_directory>

Example:
  python exp3_confidence.py /Users/liuzhiwei/Downloads/Anima\ ex\ Machina/results/bidir_comparison/bidir_30pct

Output:
  exp3_confidence.csv  — Each row is one (seed × entity × dimension) complete record
  exp3_summary.csv     — Summary statistics
"""

import torch
import torch.nn.functional as F
import csv, os, sys, json, shutil

# ── Auto-patch: v6_tool.py uses 'with' as variable name (Python reserved word) ──
_v6_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6_tool.py')
with open(_v6_path, 'r', encoding='utf-8') as _f:
    _src = _f.read()
if 'for d, v, with, need in gaps:' in _src:
    _patched = _src.replace(
        'for d, v, with, need in gaps:',
        'for d, v, cnt, need in gaps:'
    ).replace(
        'print(f"  {d}={v}: with{with}，need{need}")',
        'print(f"  {d}={v}: cnt={cnt}，need={need}")'
    )
    _bak = _v6_path + '.bak'
    if not os.path.exists(_bak):
        shutil.copy2(_v6_path, _bak)
    with open(_v6_path, 'w', encoding='utf-8') as _f:
        _f.write(_patched)
    print("  [auto-patch] Fixed 'with' keyword in v6_tool.py (backup: v6_tool.py.bak)")

from model import make_model
from v6_tool import parse_world, generate, FMT_META, enc2desc

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

SEEDS = [42, 137, 271, 314, 577]
NOVEL_ENTITIES = ['bird', 'shell', 'claw', 'cave', 'honey']
DIM_NAMES = ['T', 'H', 'M', 'V', 'Z', 'A', 'W', 'R', 'O', 'S', 'F', 'L', 'An']
DEVICE = 'cpu'


def load_model(ckpt_path, vocab_size):
    """Load model, handle torch.compile key prefix and key renames."""
    model = make_model(vocab_size, d=64, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    clean = {}
    for k, v in state.items():
        k = k.replace('_orig_mod.', '')
        # handle checkpoint key name differences
        k = k.replace('token_embed.', 'tokens_embed.')
        clean[k] = v
    model.load_state_dict(clean)
    model.eval()
    return model


def run_confidence(base_dir, world_path='swadesh_v6_world.md'):
    # ── Generate data ──
    cfg = parse_world(world_path)
    data = generate(cfg)
    t2id = data['tokens2id']
    id2t = data['id2tokens']
    mask_id = data['mask_id']
    ee = cfg['entity_encodings']

    desc_start = FMT_META['s2_name'][1]  # 16

    # P-token IDs (for reference)
    p_token_ids = set(t2id[f'P{i}'] for i in range(5))

    rows = []

    for seed in SEEDS:
        ckpt_path = os.path.join(base_dir,
                                 f'results_v4_ph5_bidir30_s{seed}', 'model.pt')
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Missing: {ckpt_path}")
            continue

        model = load_model(ckpt_path, data['vocab_size'])
        print(f"  ✓ Seed {seed} loaded from {ckpt_path}")

        for entity_name in NOVEL_ENTITIES:
            # Novel entity's s2_name sequence (touch distance, P-array as context)
            novel_seq = data['novel'][entity_name]['s2_name']
            seq_tensor = torch.tensor([novel_seq], dtype=torch.long)

            # gold DESC tokens
            gold_enc = ee[entity_name]
            gold_desc = enc2desc(gold_enc, cfg, t2id)  # 26 token IDs

            # Mask all 26 DESC positions
            inp = seq_tensor.clone()
            for pos in range(desc_start, desc_start + 26):
                inp[0, pos] = mask_id

            # forward pass
            with torch.no_grad():
                logits = model(inp)                        # [1, 42, vocab]
                probs = F.softmax(logits[0], dim=-1)       # [42, vocab]

            # Extract each dimension
            for d in range(13):
                dim_name = DIM_NAMES[d]
                pos_deg  = desc_start + d * 2       # degree marker
                pos_base = desc_start + d * 2 + 1   # base word

                gold_deg_id  = gold_desc[d * 2]
                gold_base_id = gold_desc[d * 2 + 1]

                pred_deg_id  = probs[pos_deg].argmax().item()
                pred_base_id = probs[pos_base].argmax().item()

                conf_deg  = probs[pos_deg, pred_deg_id].item()
                conf_base = probs[pos_base, pred_base_id].item()

                # Gold token probability (regardless of whether selected)
                gold_prob_deg  = probs[pos_deg, gold_deg_id].item()
                gold_prob_base = probs[pos_base, gold_base_id].item()

                correct = int(pred_deg_id == gold_deg_id
                              and pred_base_id == gold_base_id)

                # Dimension-level confidence: geometric mean of two positions
                conf_dim = (conf_deg * conf_base) ** 0.5
                gold_prob_dim = (gold_prob_deg * gold_prob_base) ** 0.5

                # Detect if output is P-token (regression flag)
                pred_is_p_deg  = int(pred_deg_id in p_token_ids)
                pred_is_p_base = int(pred_base_id in p_token_ids)

                rows.append({
                    'seed': seed,
                    'entity': entity_name,
                    'dimension': dim_name,
                    'gold_degree': id2t[gold_deg_id],
                    'gold_base': id2t[gold_base_id],
                    'pred_degree': id2t[pred_deg_id],
                    'pred_base': id2t[pred_base_id],
                    'conf_degree': round(conf_deg, 6),
                    'conf_base': round(conf_base, 6),
                    'conf_dim': round(conf_dim, 6),
                    'gold_prob_degree': round(gold_prob_deg, 6),
                    'gold_prob_base': round(gold_prob_base, 6),
                    'gold_prob_dim': round(gold_prob_dim, 6),
                    'correct': correct,
                    'pred_is_p_token': int(pred_is_p_deg or pred_is_p_base),
                })

    # ── Write raw CSV ──
    fields = [
        'seed', 'entity', 'dimension',
        'gold_degree', 'gold_base', 'pred_degree', 'pred_base',
        'conf_degree', 'conf_base', 'conf_dim',
        'gold_prob_degree', 'gold_prob_base', 'gold_prob_dim',
        'correct', 'pred_is_p_token',
    ]
    with open('exp3_confidence.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n✓ Raw data: {len(rows)} rows → exp3_confidence.csv")

    # ── Summary statistics ──
    correct_confs = [r['conf_dim'] for r in rows if r['correct'] == 1]
    wrong_confs   = [r['conf_dim'] for r in rows if r['correct'] == 0]

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    n_total = len(rows)
    n_correct = len(correct_confs)
    n_wrong = len(wrong_confs)
    print(f"  Total predictions: {n_total}")
    print(f"  Correct: {n_correct} ({n_correct/n_total*100:.1f}%)")
    print(f"  Wrong:   {n_wrong} ({n_wrong/n_total*100:.1f}%)")

    if correct_confs:
        mean_c = sum(correct_confs) / len(correct_confs)
        min_c = min(correct_confs)
        max_c = max(correct_confs)
        print(f"\n  Confidence when CORRECT:")
        print(f"    mean={mean_c:.4f}  min={min_c:.4f}  max={max_c:.4f}")

    if wrong_confs:
        mean_w = sum(wrong_confs) / len(wrong_confs)
        min_w = min(wrong_confs)
        max_w = max(wrong_confs)
        print(f"\n  Confidence when WRONG:")
        print(f"    mean={mean_w:.4f}  min={min_w:.4f}  max={max_w:.4f}")

    if correct_confs and wrong_confs:
        gap = (sum(correct_confs)/len(correct_confs)
               - sum(wrong_confs)/len(wrong_confs))
        print(f"\n  Calibration gap (correct − wrong): {gap:+.4f}")
        if gap > 0.1:
            print(f"  → Model has significantly higher confidence on correct dimensions ✓")
        else:
            print(f"  → Calibration signal weak")

    # ── Per-dimension breakdown ──
    print(f"\n── Per-dimension breakdown ──")
    for dim in DIM_NAMES:
        dr = [r for r in rows if r['dimension'] == dim]
        n_c = sum(1 for r in dr if r['correct'] == 1)
        n_w = sum(1 for r in dr if r['correct'] == 0)
        mc = (sum(r['conf_dim'] for r in dr if r['correct'] == 1) / n_c
              if n_c > 0 else 0)
        mw = (sum(r['conf_dim'] for r in dr if r['correct'] == 0) / n_w
              if n_w > 0 else 0)
        acc = n_c / len(dr) if dr else 0
        print(f"  {dim:>3s}: acc={acc:.2f}  "
              f"conf_correct={mc:.3f}  conf_wrong={mw:.3f}  "
              f"(n_correct={n_c}, n_wrong={n_w})")

    # ── Per-entity breakdown ──
    print(f"\n── Per-entity breakdown ──")
    for ent in NOVEL_ENTITIES:
        er = [r for r in rows if r['entity'] == ent]
        n_c = sum(1 for r in er if r['correct'] == 1)
        acc = n_c / len(er) if er else 0
        mc = (sum(r['conf_dim'] for r in er if r['correct'] == 1) / n_c
              if n_c > 0 else 0)
        mw_list = [r['conf_dim'] for r in er if r['correct'] == 0]
        mw = sum(mw_list) / len(mw_list) if mw_list else 0
        print(f"  {ent:>8s}: acc={acc:.2f}  "
              f"conf_correct={mc:.3f}  conf_wrong={mw:.3f}  "
              f"({n_c}/13 dims correct, avg across seeds)")

    # ── Write summary CSV ──
    summary_rows = []
    for dim in DIM_NAMES:
        for ent in NOVEL_ENTITIES:
            dr = [r for r in rows if r['dimension'] == dim and r['entity'] == ent]
            n_c = sum(1 for r in dr if r['correct'] == 1)
            n_w = sum(1 for r in dr if r['correct'] == 0)
            mc = (sum(r['conf_dim'] for r in dr if r['correct'] == 1) / n_c
                  if n_c > 0 else 0)
            mw = (sum(r['conf_dim'] for r in dr if r['correct'] == 0) / n_w
                  if n_w > 0 else 0)
            summary_rows.append({
                'dimension': dim,
                'entity': ent,
                'n_seeds_correct': n_c,
                'n_seeds_wrong': n_w,
                'mean_conf_correct': round(mc, 6),
                'mean_conf_wrong': round(mw, 6),
            })
    with open('exp3_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n✓ Summary → exp3_summary.csv")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python exp3_confidence.py <bidir_30pct_directory>")
        print("Example: python exp3_confidence.py bidir_30pct")
        sys.exit(1)
    run_confidence(sys.argv[1])
