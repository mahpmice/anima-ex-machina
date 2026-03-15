"""


python exp2b_shuffle_summary.py
"""
import json, glob

print(f"{'Label':<25s} {'S1':>6s} {'S2.1':>6s} {'S2.2':>6s} {'S2.4':>6s} {'S2.5':>6s} {'Probe':>6s}  order  seed")
print('-'*95)

for path in sorted(glob.glob('results_2b_*/results.json')):
    with open(path) as f:
        d = json.load(f)
    e = d.get('evaluation', {})
    label = d.get('label', path.split('/')[0])
    order = d.get('phase_order', [])
    seed = d.get('seed', '?')
    s1 = e.get('s1_p_recon',{}).get('overall',0)
    s21 = e.get('s2_name',{}).get('overall',0)
    s22 = e.get('s2_gender',{}).get('overall',0)
    s24 = e.get('s2_interact',{}).get('overall',0)
    s25 = e.get('s2_auto',{}).get('overall',0)
    pr = e.get('probes',{}).get('overall',0)
    print(f"{label:<25s} {s1:>6.3f} {s21:>6.3f} {s22:>6.3f} {s24:>6.3f} {s25:>6.3f} {pr:>6.3f}  {order}  {seed}")
