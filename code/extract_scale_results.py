"""
Extract scale comparison results
Usage:python extract_scale_results.py
      python extract_scale_results.py --dir results_scale
"""

import json, os, argparse, glob


def run(base_dir='results_scale'):
    print(f"{'═'*70}")
    print(f"  Model Scale Comparison · Results")
    print(f"{'═'*70}")

    rows = []
    for d in sorted(os.listdir(base_dir)):
        fp = os.path.join(base_dir, d, 'results.json')
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            r = json.load(f)
        ev = r['evaluation']
        rows.append({
            'label': r.get('size', d).upper(),
            'seed': r.get('seed', '?'),
            'd': r.get('d', '?'),
            'layers': r.get('n_layers', '?'),
            'heads': r.get('n_heads', '?'),
            'd_ff': r.get('d', 0) * 4,
            'params': r.get('params', 0),
            's1': ev.get('s1_p_recon', '—'),
            's2_1': ev.get('s2_name', '—'),
            's2_4': ev.get('s2_interact', '—'),
            'probe': ev.get('probes', '—'),
            'novel': ev.get('novel_avg_dims', '—'),
        })

    if not rows:
        print(f"  ⚠ {base_dir}/ no results under")
        return

    # Architecture confirmation
    print(f"\n── Architecture ──\n")
    print(f"  {'Model':<6s}  {'d':>4s}  {'layers':>6s}  {'heads':>5s}  {'d_ff':>5s}  {'Params':>10s}")
    print(f"  {'─'*45}")
    seen = set()
    for r in rows:
        key = r['label']
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['label']:<6s}  {r['d']:>4}  {r['layers']:>6}  {r['heads']:>5}  "
              f"{r['d_ff']:>5}  {r['params']:>10,}")

    # Core metrics
    print(f"\n── Core metrics ──\n")
    print(f"  {'Model':<6s}  {'seed':>5s}  {'S2.4':>7s}  {'Probe':>7s}  {'Novel':>7s}  {'S1':>7s}  {'S2.1':>7s}")
    print(f"  {'─'*55}")
    for r in rows:
        s24 = f"{r['s2_4']:.3f}" if isinstance(r['s2_4'], float) else str(r['s2_4'])
        pr = f"{r['probe']:.3f}" if isinstance(r['probe'], float) else str(r['probe'])
        nv = f"{r['novel']}/13" if isinstance(r['novel'], (int, float)) else str(r['novel'])
        s1 = f"{r['s1']:.3f}" if isinstance(r['s1'], float) else str(r['s1'])
        s21 = f"{r['s2_1']:.3f}" if isinstance(r['s2_1'], float) else str(r['s2_1'])
        print(f"  {r['label']:<6s}  {r['seed']:>5}  {s24:>7s}  {pr:>7s}  {nv:>7s}  {s1:>7s}  {s21:>7s}")

    print(f"\n{'═'*70}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='results_scale')
    args = p.parse_args()
    run(args.dir)
