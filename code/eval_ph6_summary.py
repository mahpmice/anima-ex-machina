#!/usr/bin/env python3
"""
Evaluate Phase 6 models and summarize key metrics:
- S3 entity (full mask)
- S3 interact (full mask)
- Probe held-out pairs
- Animacy test

Usage:
  python eval_ph6_summary.py
"""

import torch
import numpy as np
import json, os, glob

from model import make_model
from v6_tool import parse_world, generate
from train_core import pretensorize, eval_full_mask, eval_desc_acc
from train_phases import FMT_META

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 64

# Model paths
MODEL_PATHS = {
    42: 'results_v4_ph6_s42/model_ph6.pt',
    123: 'results_multiseed_ph6/seed_123/model.pt',
    456: 'results_multiseed_ph6/seed_456/model.pt',
    789: 'results_multiseed_ph6/seed_789/model.pt',
    1024: 'results_multiseed_ph6/seed_1024/model.pt',
}


def load_model(path, vocab_size):
    model = make_model(vocab_size, d=D_MODEL, n_layers=4, n_heads=4,
                       max_len=48, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def evaluate_s3(model, t, pad_id, mask_id):
    """Evaluate S3 metrics"""
    results = {}
    
    # S3 entity (full mask)
    ds_s3 = FMT_META.get('s3_entity', (None, 3))[1]
    if t['s3_entity'].shape[0] > 0:
        e_acc, e_dim = eval_full_mask(model, t['s3_entity'], DEVICE, pad_id, mask_id, desc_start=ds_s3)
        results['s3_entity_fullmask'] = round(e_acc, 4)
        results['s3_entity_dims_99'] = sum(1 for v in e_dim.values() if v >= 0.99)
    
    # S3 interact (full mask)
    ds_s3i = FMT_META.get('s3_interact', (None, 8))[1]
    if t['s3_interact'].shape[0] > 0:
        i_acc, i_dim = eval_full_mask(model, t['s3_interact'][:100], DEVICE, pad_id, mask_id, desc_start=ds_s3i)
        results['s3_interact_fullmask'] = round(i_acc, 4)
        results['s3_interact_dims_99'] = sum(1 for v in i_dim.values() if v >= 0.99)
    
    # Probe held-out pairs
    if t['probe_s3'].shape[0] > 0:
        p_acc, _ = eval_full_mask(model, t['probe_s3'], DEVICE, pad_id, mask_id, desc_start=ds_s3i)
        results['probe_heldout'] = round(p_acc, 4)
    
    # Animacy test
    if t['anim_s3'].shape[0] > 0:
        a_acc, _ = eval_full_mask(model, t['anim_s3'], DEVICE, pad_id, mask_id, desc_start=ds_s3i)
        results['animacy_test'] = round(a_acc, 4)
    
    return results


def main():
    print(f"{'═'*60}")
    print(f"  Phase 6 Evaluation Summary")
    print(f"  Device: {DEVICE}")
    print(f"{'═'*60}")
    
    # Load data
    cfg_w = parse_world('swadesh_v6_world.md')
    data = generate(cfg_w)
    pad_id = data['pad_id']
    mask_id = data['mask_id']
    vocab_size = data['vocab_size']
    
    # Prepare tensors
    t = {
        's3_entity': pretensorize(data['s3']['entity'], pad_id),
        's3_interact': pretensorize(data['s3']['interact'], pad_id),
        'probe_s3': pretensorize(data['probes']['s3'], pad_id),
        'anim_s3': pretensorize(data['animacy_probes']['s3'], pad_id),
    }
    
    print(f"\n  S3 entity samples: {t['s3_entity'].shape[0]}")
    print(f"  S3 interact samples: {t['s3_interact'].shape[0]}")
    print(f"  Probe held-out: {t['probe_s3'].shape[0]}")
    print(f"  Animacy test: {t['anim_s3'].shape[0]}")
    
    # Evaluate each seed
    all_results = {}
    
    for seed, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"\n  [seed={seed}] Model not found: {path}")
            continue
        
        print(f"\n  [seed={seed}] Evaluating...")
        model = load_model(path, vocab_size)
        results = evaluate_s3(model, t, pad_id, mask_id)
        all_results[seed] = results
        
        print(f"    S3 entity:  {results.get('s3_entity_fullmask', 'N/A')}")
        print(f"    S3 interact: {results.get('s3_interact_fullmask', 'N/A')}")
        print(f"    Probe held: {results.get('probe_heldout', 'N/A')}")
        print(f"    Animacy:    {results.get('animacy_test', 'N/A')}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary statistics
    if all_results:
        print(f"\n{'═'*60}")
        print(f"  Summary ({len(all_results)} seeds)")
        print(f"{'═'*60}")
        
        metrics = ['s3_entity_fullmask', 's3_interact_fullmask', 'probe_heldout', 'animacy_test']
        
        print(f"\n  {'Metric':<22s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
        print(f"  {'─'*55}")
        
        summary = {}
        for m in metrics:
            vals = [r[m] for r in all_results.values() if m in r]
            if vals:
                arr = np.array(vals)
                summary[m] = {
                    'mean': round(arr.mean(), 4),
                    'std': round(arr.std(), 4),
                    'min': round(arr.min(), 4),
                    'max': round(arr.max(), 4),
                }
                print(f"  {m:<22s}  {arr.mean():>7.4f}  {arr.std():>7.4f}  {arr.min():>7.4f}  {arr.max():>7.4f}")
        
        # Save results
        output = {
            'seeds': list(all_results.keys()),
            'per_seed': all_results,
            'summary': summary,
        }
        
        with open('ph6_summary.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n  Saved: ph6_summary.json")
    
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
