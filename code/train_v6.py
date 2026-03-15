"""
Swadesh v4 Training Entry
Thin shell. Data->Model->Phases->Run.

Usage:
  python train_v6.py --phase 2       # S1 + S2.1
  python train_v6.py --phase 6       # all
  python train_v6.py --phase 2 --bs 512 --no-fp16
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, argparse, time

from model import make_model, count_params
from v6_tool import parse_world, generate
from train_core import (
    pretensorize, build_tagged_batches, train_epoch,
    eval_p_recon, eval_desc_acc, eval_full_mask, per_dimension_probe,
)
from train_phases import build_phases

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')


def run(max_phase=2, world_path='swadesh_v6_world.md', bs=256, use_fp16=True,
        reverse_ratio=0.0, seed=42):
    t0 = time.time()
    print(f"Max phase: {max_phase}")
    print(f"World: {world_path}")
    print(f"Device: {DEVICE}  bs={bs}  fp16={use_fp16}  reverse={reverse_ratio}  seed={seed}")

    # Fix randomness
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_w = parse_world(world_path)
    data = generate(cfg_w)
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    t2id = data['tokens2id']
    id2tokens = data['id2tokens']
    ee = data['cfg']['entity_encodings']
    train_names = data['cfg']['train_names']
    te = [(n, ee[n]) for n in train_names]
    holdout_dims = data['holdout_dims']

    # ── pre tensor ize ──
    print("pretensorize...")
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

    # ── Model ──
    model = make_model(data['vocab_size'], d=64, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.1).to(DEVICE)

    compiled = False
    if hasattr(torch, 'compile') and DEVICE != 'mps':
        try:
            model = torch.compile(model)
            compiled = True
        except Exception:
            pass

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    print(f"Model: {count_params(model):,} params  compiled={compiled}")

    # ── Phase Definitions ──
    phases = build_phases(data, t, DEVICE, pad_id, mask_id, holdout_dims)

    mr = 0.30
    rng = np.random.default_rng(seed)
    total_step = 0
    ratio_tag = f"_bidir{int(reverse_ratio*100)}" if reverse_ratio > 0 else ""
    rd = f"results_v4_ph{max_phase}{ratio_tag}_s{seed}"
    all_history = {}

    # ── Training loop ──
    for phase_num in range(1, max_phase + 1):
        pdef = phases[phase_num]
        pname = pdef['name']
        max_steps = pdef['max_steps']
        stage_data = pdef['data']()

        print(f"\n{'═'*60}")
        print(f"  Phase {phase_num}: {pname}")
        total_seqs = sum(td[1].shape[0] for td in stage_data)
        print(f"  Total sequences: {total_seqs}  max_steps={max_steps}")
        print(f"{'═'*60}")

        step, history = 0, []
        epoch = 0

        while step < max_steps:
            epoch += 1
            batches = build_tagged_batches(stage_data, bs, rng)
            steps_ep, avg_loss = train_epoch(
                model, opt, batches, DEVICE, pad_id, mask_id, mr,
                holdout_dims=holdout_dims, use_fp16=use_fp16,
                reverse_ratio=reverse_ratio)
            step += steps_ep

            eval_acc = pdef['quick_eval'](model)

            history.append(dict(epoch=epoch, step=step,
                                loss=round(avg_loss, 5),
                                acc=round(eval_acc, 4)))

            if epoch % 5 == 0 or step >= max_steps:
                print(f"    ep{epoch:3d}  step={step:5d}  "
                      f"loss={avg_loss:.4f}  acc={eval_acc:.3f}")

            if avg_loss < 0.25 and eval_acc >= 0.95:
                print(f"  ✓ skip-step: loss={avg_loss:.4f}, acc={eval_acc:.3f}")
                break

            model.train()

        total_step += step
        all_history[f'phase_{phase_num}'] = history

        print(f"\n  ── Phase {phase_num} evaluation ──")
        pdef['end_eval'](model)

        # eachPhaseSavefastsnapshot
        os.makedirs(rd, exist_ok=True)
        ckpt_path = os.path.join(rd, f'model_ph{phase_num}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ✓ fastsnapshot: {ckpt_path}")

    # ═══════════════════════════════════════════════════
    # FINAL ANALYSIS
    # ═══════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"FINAL ANALYSIS  (Phase 1-{max_phase})")
    print(f"{'='*60}")

    # Embedding probe
    probe = per_dimension_probe(model, te, DEVICE, t2id)
    if probe:
        print(f"\n── Per-Dimension Embedding Probe ──")
        for dim, acc in probe.items():
            tag = '✓' if acc > 0.4 else '✗'
            print(f"  [{tag}] {dim}: {acc:.3f}")

    # ── Save ──
    elapsed = time.time() - t0
    os.makedirs(rd, exist_ok=True)
    result = dict(
        max_phase=max_phase, total_steps=total_step,
        elapsed_sec=round(elapsed, 1),
        history=all_history,
    )
    with open(os.path.join(rd, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    torch.save(model.state_dict(), os.path.join(rd, 'model.pt'))

    print(f"\n  Total steps: {total_step}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Saved to {rd}/")
    print(f"{'='*60}")
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Swadesh v4 train')
    p.add_argument('--phase', type=int, default=2, choices=[1,2,3,4,5,6],
                   help='traintowhich phase（1=S1, 2=S1+S2.1, ..., 6=all）')
    p.add_argument('--world', default='swadesh_v6_world.md')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--reverse-ratio', type=float, default=0.0,
                   help='S2reversemaskratio (0.0=forward-only, 0.3=30%%reverse)')
    p.add_argument('--seed', type=int, default=42,
                   help='randomseed (default42)')
    args = p.parse_args()
    run(args.phase, args.world, args.bs, use_fp16=not args.no_fp16,
        reverse_ratio=args.reverse_ratio, seed=args.seed)
