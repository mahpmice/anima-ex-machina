"""
Aggregate multi-seed experiment results


Read all results from results_multiseed/ and results_multiseed_ablation/
Output paper-ready mean±std tables.

Usage:
  python aggregate_results.py
  python aggregate_results.py --latex    # also output LaTeX tables
"""

import json, os, argparse
import numpy as np


def load_baseline_results():
    """Always scan individual seed directories, no reliance on summary files"""
    results = {}
    base = 'results_multiseed'
    if not os.path.exists(base):
        return {}
    for d in sorted(os.listdir(base)):
        if d.startswith('seed_'):
            fp = os.path.join(base, d, 'seed_results.json')
            if os.path.exists(fp):
                with open(fp) as f:
                    r = json.load(f)
                results[r['seed']] = r
    return results


def load_ablation_results():
    """Always scan individual directories"""
    results = {}
    base = 'results_multiseed_ablation'
    if not os.path.exists(base):
        return {}
    for d in sorted(os.listdir(base)):
        fp = os.path.join(base, d, 'seed_results.json')
        if os.path.exists(fp):
            with open(fp) as f:
                r = json.load(f)
            cond = r.get('condition', 'unknown')
            if cond not in results:
                results[cond] = {}
            results[cond][r['seed']] = r
    return results


def fmt(arr):
    """Format mean±std"""
    if len(arr) == 0:
        return '—'
    if len(arr) == 1:
        return f"{arr[0]:.3f}"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.3f}±{s:.3f}"


def fmt_latex(arr):
    """LaTeX format"""
    if len(arr) == 0:
        return '—'
    if len(arr) == 1:
        return f"{arr[0]:.3f}"
    m, s = np.mean(arr), np.std(arr)
    return f"${m:.3f} \\pm {s:.3f}$"


def run(do_latex=False):
    print("═" * 70)
    print("  Multi-seed Experiment Results Summary")
    print("═" * 70)

    baseline = load_baseline_results()
    ablation = load_ablation_results()

    if not baseline:
        print("  ⚠ No baseline results found")
    if not ablation:
        print("  ⚠ No ablation results found")

    # ═══════════════════════════════════════════
    # Table 1: Baseline S2 evaluation
    # ═══════════════════════════════════════════
    if baseline:
        n_seeds = len(baseline)
        print(f"\n── Table 1: Baseline bidir30 ({n_seeds} seeds) ──\n")

        metrics = {
            'S1 P-recon': 's1_p_recon',
            'S2.1 Name': 's2_name',
            'S2.2 Gender': 's2_gender',
            'S2.4 Interact': 's2_interact',
            'S2.5 Auto': 's2_auto',
            'Probes': 'probes',
            'Novel dims': 'novel_avg_dims',
        }

        print(f"  {'Metric':<20s}  {'Mean±Std':>15s}  {'Min':>7s}  {'Max':>7s}  Seeds")
        print(f"  {'─'*65}")

        for display_name, key in metrics.items():
            vals = []
            for seed, r in baseline.items():
                v = r.get('baseline', {}).get(key)
                if v is not None:
                    vals.append(float(v))
            arr = np.array(vals) if vals else np.array([])
            if len(arr) > 0:
                print(f"  {display_name:<20s}  {fmt(arr):>15s}  "
                      f"{arr.min():>7.3f}  {arr.max():>7.3f}  {[str(s) for s in baseline.keys()]}")

    # ═══════════════════════════════════════════
    # Table 2: Exp1 reverse inference
    # ═══════════════════════════════════════════
    if baseline:
        print(f"\n── Table 2: Exp1 reverse inference ({n_seeds} seeds) ──\n")

        exp1_metrics = {
            'A Train': 'A_train',
            'B Novel': 'B_novel',
            'C Mismatch': 'C_mismatch',
            'D Fictional': 'D_fictional',
            'E Interact': 'E_interact',
            'F Probes': 'F_probes',
        }

        print(f"  {'Group':<15s}  {'Mean±Std':>15s}  {'Min':>7s}  {'Max':>7s}")
        print(f"  {'─'*50}")

        for display_name, key in exp1_metrics.items():
            vals = []
            for seed, r in baseline.items():
                v = r.get('exp1_reverse', {}).get(key)
                if v is not None:
                    vals.append(float(v))
            arr = np.array(vals) if vals else np.array([])
            if len(arr) > 0:
                print(f"  {display_name:<15s}  {fmt(arr):>15s}  "
                      f"{arr.min():>7.3f}  {arr.max():>7.3f}")

    # ═══════════════════════════════════════════
    # Table 3: Exp2a gender marker isolation
    # ═══════════════════════════════════════════
    if baseline:
        print(f"\n── Table 3: Exp2a gender marker isolation ({n_seeds} seeds) ──\n")

        e2a_metrics = {
            'Gender Full': 'gender_A_full',
            'Gender NoGender': 'gender_B_no_gender',
            'Gender NoName': 'gender_C_no_name',
            'Gender Swap': 'gender_D_gender_swap',
            'Interact Full': 'interact_A_full',
            'Interact NoGA': 'interact_B_no_ga',
            'Interact NoGB': 'interact_C_no_gb',
            'Interact NoBoth': 'interact_D_no_both',
        }

        print(f"  {'Condition':<20s}  {'Mean±Std':>15s}  {'Min':>7s}  {'Max':>7s}")
        print(f"  {'─'*55}")

        for display_name, key in e2a_metrics.items():
            vals = []
            for seed, r in baseline.items():
                v = r.get('exp2a_gender', {}).get(key)
                if v is not None:
                    vals.append(float(v))
            arr = np.array(vals) if vals else np.array([])
            if len(arr) > 0:
                print(f"  {display_name:<20s}  {fmt(arr):>15s}  "
                      f"{arr.min():>7.3f}  {arr.max():>7.3f}")

    # ═══════════════════════════════════════════
    # Table 4: Phase ablation
    # ═══════════════════════════════════════════
    if ablation:
        print(f"\n── Table 4: Phase ablation ──\n")

        eval_keys = ['s1_p_recon', 's2_name', 's2_gender', 's2_interact', 's2_auto', 'probes']
        header = f"  {'Condition':<12s}"
        for k in eval_keys:
            header += f"  {k:>14s}"
        print(header)
        print(f"  {'─'*100}")

        # Baseline row
        if baseline:
            line = f"  {'Baseline':<12s}"
            for k in eval_keys:
                vals = [float(r['baseline'].get(k, 0)) for r in baseline.values()
                        if r.get('baseline', {}).get(k) is not None]
                line += f"  {fmt(np.array(vals)):>14s}"
            print(line)

        # Ablation rows
        for label, cond_res in ablation.items():
            line = f"  {label:<12s}"
            for k in eval_keys:
                vals = [float(r['evaluation'].get(k, 0)) for r in cond_res.values()
                        if r.get('evaluation', {}).get(k) is not None]
                line += f"  {fmt(np.array(vals)):>14s}"
            print(line)

    # ═══════════════════════════════════════════
    # LaTeX output
    # ═══════════════════════════════════════════
    if do_latex and baseline:
        print(f"\n── LaTeX Tables ──\n")

        # Table 1 LaTeX
        print("% Table 1: Baseline")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Metric & Accuracy \\\\")
        print("\\midrule")
        metrics_latex = {
            'S1 P-recon': 's1_p_recon',
            'S2.1 Name': 's2_name',
            'S2.2 Gender': 's2_gender',
            'S2.4 Interact': 's2_interact',
            'S2.5 Auto': 's2_auto',
            'Probes (held-out)': 'probes',
        }
        for display_name, key in metrics_latex.items():
            vals = [float(r['baseline'][key]) for r in baseline.values()
                    if r.get('baseline', {}).get(key) is not None]
            if vals:
                print(f"{display_name} & {fmt_latex(np.array(vals))} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}\n")

        # Table 2 LaTeX: Exp1
        print("% Table 2: Reverse inference")
        print("\\begin{tabular}{lc}")
        print("\\toprule")
        print("Test Group & Reverse (desc$\\to$P) \\\\")
        print("\\midrule")
        for display_name, key in exp1_metrics.items():
            vals = [float(r['exp1_reverse'][key]) for r in baseline.values()
                    if r.get('exp1_reverse', {}).get(key) is not None]
            if vals:
                print(f"{display_name} & {fmt_latex(np.array(vals))} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}\n")

    # ═══════════════════════════════════════════
    # Save summary JSON
    # ═══════════════════════════════════════════
    summary = {'baseline_seeds': len(baseline) if baseline else 0}

    if baseline:
        for key in ['s1_p_recon', 's2_name', 's2_gender', 's2_interact',
                     's2_auto', 'probes']:
            vals = [float(r['baseline'].get(key, 0)) for r in baseline.values()
                    if r.get('baseline', {}).get(key) is not None]
            if vals:
                arr = np.array(vals)
                summary[f'baseline_{key}'] = {
                    'mean': round(float(arr.mean()), 4),
                    'std': round(float(arr.std()), 4),
                    'min': round(float(arr.min()), 4),
                    'max': round(float(arr.max()), 4),
                    'n': len(vals),
                }

        for key in ['A_train', 'B_novel', 'C_mismatch', 'D_fictional',
                     'E_interact', 'F_probes']:
            vals = [float(r['exp1_reverse'].get(key, 0)) for r in baseline.values()
                    if r.get('exp1_reverse', {}).get(key) is not None]
            if vals:
                arr = np.array(vals)
                summary[f'exp1_{key}'] = {
                    'mean': round(float(arr.mean()), 4),
                    'std': round(float(arr.std()), 4),
                    'n': len(vals),
                }

    with open('results_multiseed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: results_multiseed_summary.json")
    print("═" * 70)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--latex', action='store_true')
    args = p.parse_args()
    run(do_latex=args.latex)
