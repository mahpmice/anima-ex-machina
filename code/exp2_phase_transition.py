"""
Experiment 2: Phase 2 Transition Tracking
═══════════════════════════════════════════════
Purpose: Fill paper §3.1.2 phase transition figure
Config: 26-token main system (bidir30), 5 seeds

During Phase 2 training, pause every 50 steps, run forward pass on 5 entities,
record token type at DESC positions (P-token vs correct DESC-token).

Core curves: P-token ratio decreases with step, DESC accuracy increases with step.
Crossover point = phase transition: the moment perception generates language.

Usage:
  python exp2_phase_transition.py [options]

Options:
  --world PATH       World config file (default: swadesh_v6_world.md)
  --bs INT           batch size (default: 256)
  --track-every INT  Run tracking eval every N steps (default: 50)
  --ph1-dir PATH     If existing Phase 1 checkpoint, load from here to skip Phase 1 training
                     Format: directory should contain results_v4_ph5_bidir30_s{seed}/model_ph1.pt
                     If not provided, train Phase 1 from scratch

Example (train from scratch):
  python exp2_phase_transition.py

Example (load existing Phase 1 checkpoint):
  python exp2_phase_transition.py --ph1-dir bidir_30pct

Output:
  exp2_phase_transition.csv  — Each row is one (seed × step × entity) tracking record
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv, os, sys, argparse, time, shutil

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

from model import make_model, count_params
from v6_tool import (parse_world, generate, FMT_META, seq_s2_name,
                     enc2desc, enc2tok, build_tokens)
from v6_rules import attenuate
from train_core import (
    pretensorize, repeat_tensor, repeat_eidx,
    build_tagged_batches, mask_p_region, mask_desc_region, mask_p_in_s2,
    eval_p_recon, eval_desc_acc, N_DIMS, TOKENS_PER_DIM,

)
from train_phases import build_phases

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

SEEDS = [42, 137, 271, 314, 577]
TRACK_ENTITIES = ['fire', 'water', 'stone', 'bird', 'snake']
DIM_NAMES = ['T', 'H', 'M', 'V', 'Z', 'A', 'W', 'R', 'O', 'S', 'F', 'L', 'An']
REVERSE_RATIO = 0.3   # bidir30

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')


def load_model_state(model, ckpt_path):
    """Load checkpoint, handle torch.compile prefix and key renames."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    clean = {}
    for k, v in state.items():
        k = k.replace('_orig_mod.', '')
        k = k.replace('token_embed.', 'tokens_embed.')
        clean[k] = v
    model.load_state_dict(clean)


# ═══════════════════════════════════════════════════
# Tracking eval: measure P-token ratio and DESC accuracy
# ═══════════════════════════════════════════════════

def eval_tracking(model, tracking_seqs, gold_descs, p_token_ids,
                  device, mask_id, desc_start=16):
    """
    For each tracking entity, mask all 26 DESC positions,
    forward pass, count P-tokens and correct DESC tokens.

    Returns: list of dicts, one per entity
    """
    model.eval()
    results = []

    for entity_name, seq, gold in zip(
            TRACK_ENTITIES, tracking_seqs, gold_descs):
        inp = torch.tensor([seq], dtype=torch.long)

        # mask all 26 DESC positions
        for pos in range(desc_start, desc_start + 26):
            inp[0, pos] = mask_id

        with torch.no_grad():
            logits = model(inp.to(device))
            preds = logits[0].argmax(dim=-1).cpu()  # [seq_len]

        p_count = 0
        correct_count = 0

        for i in range(26):
            pos = desc_start + i
            pred_id = preds[pos].item()
            gold_id = gold[i]

            if pred_id in p_token_ids:
                p_count += 1
            if pred_id == gold_id:
                correct_count += 1

        results.append({
            'entity': entity_name,
            'p_ratio': round(p_count / 26, 4),
            'correct_ratio': round(correct_count / 26, 4),
            'p_count': p_count,
            'correct_count': correct_count,
        })

    model.train()
    return results


# ═══════════════════════════════════════════════════
# Single-batch training step
# ═══════════════════════════════════════════════════

def train_one_batch(model, opt, fmt_tag, batch, eidx, device,
                    pad_id, mask_id, mr, holdout_dims, reverse_ratio):
    """Train on one batch. Returns loss value."""
    batch = batch.to(device)

    if fmt_tag == 's1_bare':
        masked, targets = mask_p_region(batch, mr, pad_id, mask_id,
                                        p_start=2, p_len=13)
    else:
        meta = FMT_META.get(fmt_tag)
        if meta is None:
            return 0.0
        _, desc_start, has_p = meta
        if desc_start is None:
            return 0.0

        do_reverse = (has_p and reverse_ratio > 0
                      and torch.rand(1).item() < reverse_ratio)

        if do_reverse:
            p_start = desc_start - 13
            masked, targets = mask_p_in_s2(
                batch, mr, pad_id, mask_id, p_start=p_start)
        else:
            masked, targets = mask_desc_region(
                batch, mr, pad_id, mask_id, desc_start,
                eidx=eidx, holdout_dims=holdout_dims)

    logits = model(masked)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    return loss.item()


# ═══════════════════════════════════════════════════
# Phase 1 standard training (no tracking needed)
# ═══════════════════════════════════════════════════

def train_phase1(model, opt, phases, bs, rng, device, pad_id, mask_id,
                 holdout_dims):
    """Train Phase 1 to convergence. Standard loop."""
    from train_core import train_epoch

    pdef = phases[1]
    max_steps = pdef['max_steps']
    stage_data = pdef['data']()
    step, epoch = 0, 0
    mr = 0.30

    print(f"\n  Phase 1: {pdef['name']}")
    total_seqs = sum(td[1].shape[0] for td in stage_data)
    print(f"  Sequences: {total_seqs}  max_steps={max_steps}")

    while step < max_steps:
        epoch += 1
        batches = build_tagged_batches(stage_data, bs, rng)
        steps_ep, avg_loss = train_epoch(
            model, opt, batches, device, pad_id, mask_id, mr,
            holdout_dims=holdout_dims, use_fp16=False,
            reverse_ratio=REVERSE_RATIO)
        step += steps_ep

        eval_acc = pdef['quick_eval'](model)
        if epoch % 10 == 0:
            print(f"    ep{epoch:3d}  step={step:5d}  "
                  f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

        if avg_loss < 0.25 and eval_acc >= 0.95:
            print(f"  ✓ Phase 1 converged: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
            break

    return step


# ═══════════════════════════════════════════════════
# Phase 2 training WITH tracking
# ═══════════════════════════════════════════════════

def train_phase2_with_tracking(model, opt, phases, bs, rng, device,
                                pad_id, mask_id, holdout_dims,
                                tracking_seqs, gold_descs, p_token_ids,
                                track_every=50):
    """
    Train Phase 2 one batch at a time.
    Every track_every steps, run tracking eval.
    Returns: list of tracking records
    """
    pdef = phases[2]
    max_steps = pdef['max_steps']
    stage_data = pdef['data']()
    mr = 0.30
    desc_start = FMT_META['s2_name'][1]

    print(f"\n  Phase 2: {pdef['name']} (with tracking every {track_every} steps)")
    total_seqs = sum(td[1].shape[0] for td in stage_data)
    print(f"  Sequences: {total_seqs}  max_steps={max_steps}")

    tracking_records = []
    step = 0

    # Step 0 eval (before any training)
    results = eval_tracking(model, tracking_seqs, gold_descs,
                            p_token_ids, device, mask_id, desc_start)
    for r in results:
        r['step'] = 0
        tracking_records.append(r)
    model.train()

    while step < max_steps:
        batches = build_tagged_batches(stage_data, bs, rng)

        for fmt_tag, batch, eidx in batches:
            train_one_batch(model, opt, fmt_tag, batch, eidx, device,
                            pad_id, mask_id, mr, holdout_dims,
                            REVERSE_RATIO)
            step += 1

            if step % track_every == 0:
                results = eval_tracking(
                    model, tracking_seqs, gold_descs,
                    p_token_ids, device, mask_id, desc_start)
                for r in results:
                    r['step'] = step
                    tracking_records.append(r)
                model.train()

                if step % (track_every * 10) == 0:
                    # Print progress
                    avg_p = np.mean([r['p_ratio'] for r in results])
                    avg_c = np.mean([r['correct_ratio'] for r in results])
                    print(f"    step={step:5d}  "
                          f"avg_p_ratio={avg_p:.3f}  "
                          f"avg_correct={avg_c:.3f}")

            if step >= max_steps:
                break

        # Check convergence
        eval_acc = pdef['quick_eval'](model)
        if eval_acc >= 0.95:
            # Run final tracking
            results = eval_tracking(
                model, tracking_seqs, gold_descs,
                p_token_ids, device, mask_id, desc_start)
            for r in results:
                r['step'] = step
                tracking_records.append(r)
            print(f"  ✓ Phase 2 converged at step {step}, acc={eval_acc:.3f}")
            break

    return tracking_records


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def run(world_path='swadesh_v6_world.md', bs=256, track_every=50,
        ph1_dir=None):
    t0 = time.time()
    print(f"{'═'*60}")
    print(f"  Experiment 2: Phase 2 Transition Tracking")
    print(f"{'═'*60}")
    print(f"  Device: {DEVICE}  bs={bs}  reverse={REVERSE_RATIO}")
    print(f"  Track every {track_every} steps")
    print(f"  Seeds: {SEEDS}")
    print(f"  Entities: {TRACK_ENTITIES}")

    # ── Generate data ──
    cfg = parse_world(world_path)
    data = generate(cfg)
    t2id = data['tokens2id']
    id2t = data['id2tokens']
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    ee = cfg['entity_encodings']
    holdout_dims = data['holdout_dims']

    # P-token IDs
    p_token_ids = set(t2id[f'P{i}'] for i in range(5))

    # ── Build eval sequences for tracking entities ──
    tracking_seqs = []
    gold_descs = []
    for name in TRACK_ENTITIES:
        enc = ee[name]
        seq = seq_s2_name('touch', name, enc, cfg, t2id)
        gold = enc2desc(enc, cfg, t2id)
        tracking_seqs.append(seq)
        gold_descs.append(gold)

    # ── Pre-tensorize (for phase definitions) ──
    from train_core import pretensorize
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

    all_tracking = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"{'─'*60}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Model ──
        model = make_model(data['vocab_size'], d=64, n_layers=4, n_heads=4,
                           max_len=48, dropout=0.1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        # ── Phase definitions ──
        phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

        # ── Phase 1 ──
        ph1_loaded = False
        if ph1_dir:
            ph1_path = os.path.join(
                ph1_dir, f'results_v4_ph5_bidir30_s{seed}', 'model_ph1.pt')
            if os.path.exists(ph1_path):
                load_model_state(model, ph1_path)
                print(f"  ✓ Phase 1 loaded from {ph1_path}")
                ph1_loaded = True
            else:
                # Also try results_v4_ph2 format
                ph1_path2 = os.path.join(
                    ph1_dir, f'results_v4_ph2_bidir30_s{seed}', 'model_ph1.pt')
                if os.path.exists(ph1_path2):
                    load_model_state(model, ph1_path2)
                    print(f"  ✓ Phase 1 loaded from {ph1_path2}")
                    ph1_loaded = True

        if not ph1_loaded:
            print(f"  Training Phase 1 from scratch...")
            train_phase1(model, opt, phases, bs, rng, DEVICE,
                         pad_id, mask_id, holdout_dims)

        # ── Phase 2 with tracking ──
        records = train_phase2_with_tracking(
            model, opt, phases, bs, rng, DEVICE,
            pad_id, mask_id, holdout_dims,
            tracking_seqs, gold_descs, p_token_ids,
            track_every=track_every)

        for r in records:
            r['seed'] = seed
        all_tracking.extend(records)

    # ── Write CSV ──
    fields = ['seed', 'step', 'entity', 'p_ratio', 'correct_ratio',
              'p_count', 'correct_count']
    with open('exp2_phase_transition.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_tracking)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  ✓ Done. {len(all_tracking)} records → exp2_phase_transition.csv")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'═'*60}")

    # ── Quick stats: find phase transition point ──
    print(f"\n── Phase transition estimate ──")
    steps_in_data = sorted(set(r['step'] for r in all_tracking))
    for step_val in steps_in_data:
        step_rows = [r for r in all_tracking if r['step'] == step_val]
        avg_p = np.mean([r['p_ratio'] for r in step_rows])
        avg_c = np.mean([r['correct_ratio'] for r in step_rows])
        # Find near crossover point
        if abs(avg_p - avg_c) < 0.15 and step_val > 0:
            print(f"  ★ Crossover near step {step_val}: "
                  f"p_ratio={avg_p:.3f}, correct={avg_c:.3f}")
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Exp2: Phase transition tracking')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--track-every', type=int, default=50)
    p.add_argument('--ph1-dir', default=None,
                   help='Directory with existing Phase 1 checkpoints')
    args = p.parse_args()
    run(args.world, args.bs, args.track_every, args.ph1_dir)
