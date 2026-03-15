"""
Swadesh v4 Phase Definitions
Data config and eval config for each phase.

Change phase design -> edit this file
Change mask/eval logic -> edit train_core.py
Change run params -> edit train_v6.py

each phase isone dict:
  name:       display name
  max_steps:  max trainsteps
  data:       callable → list of (fmt_tag, tensor, eidx_or_None)
  quick_eval: callable(model) → float  (traininfastfastevaluation，useskip-step)
  end_eval:   callable(model) → None   (end of phasePrintdetailedevaluation)

V4changes：evalfunction auto-handlesplaces2-tokens/dimensions，callusesignature unchanged。
V4-Dlayers：s2_name/gender/s3_entityall3layersDize(×3)，repeatproportionally shrink1/3。
"""

from v6_tool import FMT_META
from train_core import (
    pretensorize, repeat_tensor, repeat_eidx,
    eval_p_recon, eval_desc_acc, eval_desc_holdout, eval_full_mask,
)


def build_phases(data, tensors, device, pad_id, mask_id, holdout_dims):
    """
    Build all 6  phase definitions。

    Parameters:
      data:          generate() Return Full data dict
      tensors:       pre tensor izedata dict (see train_v6.py)
      device:        trainDevice
      pad_id, mask_id: tokens ids
      holdout_dims:  per-entity held-out dimensionslist

    Return:
      dict[int → phase_def]
    """
    t = tensors  # short alias

    phases = {}

    # ──────────────────────────────────────────────
    # Phase 1: S1 · pure perception
    # ──────────────────────────────────────────────
    phases[1] = dict(
        name='S1 · pure perception',
        max_steps=5000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 10), None),
            ('s1_bare', repeat_tensor(t['s1_un'], 5), None),
            ('s1_bare', repeat_tensor(t['s1_inter'], 5), None),
        ],
        quick_eval=lambda model: eval_p_recon(
            model, t['s1_bare'][:100], device, pad_id, mask_id)[0],
        end_eval=lambda model: _eval_phase1(model, t, device, pad_id, mask_id),
    )

    # ──────────────────────────────────────────────
    # Phase 2: S1 + S2.1 · naming
    # s2_name: 294 base (was 98), repeat 15→5
    # ──────────────────────────────────────────────
    phases[2] = dict(
        name='S2.1 · naming',
        max_steps=5000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 5), None),
            ('s1_bare', repeat_tensor(t['s1_un'], 3), None),
            ('s2_name', repeat_tensor(t['s2_name'], 5),
             repeat_eidx(t['s2_name_eidx'], 5)),
        ],
        quick_eval=lambda model: eval_desc_acc(
            model, t['s2_name'][:50], device, pad_id, mask_id,
            desc_start=FMT_META['s2_name'][1])[0],
        end_eval=lambda model: _eval_phase2(
            model, data, t, device, pad_id, mask_id, holdout_dims),
    )

    # ──────────────────────────────────────────────
    # Phase 3: S1 + S2.1-2.2 · gender marker
    # s2_name: repeat 8→3; s2_gender: repeat 10→3
    # ──────────────────────────────────────────────
    phases[3] = dict(
        name='S2.2 · gender marker',
        max_steps=5000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 3), None),
            ('s2_name', repeat_tensor(t['s2_name'], 3),
             repeat_eidx(t['s2_name_eidx'], 3)),
            ('s2_gender', repeat_tensor(t['s2_gender'], 3),
             repeat_eidx(t['s2_gender_eidx'], 3)),
        ],
        quick_eval=lambda model: eval_desc_acc(
            model, t['s2_gender'][:50], device, pad_id, mask_id,
            desc_start=FMT_META['s2_gender'][1])[0],
        end_eval=lambda model: _eval_phase3(
            model, t, device, pad_id, mask_id),
    )

    # ──────────────────────────────────────────────
    # Phase 4: S1 + S2.1-2.3 · un/il
    # s2_name/gender: repeat 5→2
    # ──────────────────────────────────────────────
    phases[4] = dict(
        name='S2.3 · un/il',
        max_steps=5000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 3), None),
            ('s2_name', repeat_tensor(t['s2_name'], 2),
             repeat_eidx(t['s2_name_eidx'], 2)),
            ('s2_gender', repeat_tensor(t['s2_gender'], 2),
             repeat_eidx(t['s2_gender_eidx'], 2)),
            ('s2_article', repeat_tensor(t['s2_un'], 5), None),
            ('s2_article', repeat_tensor(t['s2_il'], 5), None),
        ],
        quick_eval=lambda model: eval_desc_acc(
            model, t['s2_un'][:50], device, pad_id, mask_id,
            desc_start=FMT_META['s2_article'][1])[0],
        end_eval=lambda model: _eval_phase4(
            model, t, device, pad_id, mask_id),
    )

    # ──────────────────────────────────────────────
    # Phase 5: S1 + S2.1-2.5 · interaction naming
    # s2_name/gender: repeat 3→1
    # ──────────────────────────────────────────────
    phases[5] = dict(
        name='S2.4-5 · interaction naming',
        max_steps=10000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 3), None),
            ('s2_name', repeat_tensor(t['s2_name'], 1),
             repeat_eidx(t['s2_name_eidx'], 1)),
            ('s2_gender', repeat_tensor(t['s2_gender'], 1),
             repeat_eidx(t['s2_gender_eidx'], 1)),
            ('s2_article', repeat_tensor(t['s2_un'], 3), None),
            ('s2_article', repeat_tensor(t['s2_il'], 3), None),
            ('s2_interact', repeat_tensor(t['s2_interact'], 10), None),
            ('s2_auto', repeat_tensor(t['s2_auto'], 10), None),
        ],
        quick_eval=lambda model: eval_desc_acc(
            model, t['s2_interact'][:50], device, pad_id, mask_id,
            desc_start=FMT_META['s2_interact'][1])[0],
        end_eval=lambda model: _eval_phase5(
            model, t, device, pad_id, mask_id),
    )

    # ──────────────────────────────────────────────
    # Phase 6: S1 + S2 + S3 · dialogue
    # s2_name/gender: repeat 3→1; s3_entity: repeat 15→5
    # ──────────────────────────────────────────────
    phases[6] = dict(
        name='S3 · dialogue',
        max_steps=20000,
        data=lambda: [
            ('s1_bare', repeat_tensor(t['s1_bare'], 3), None),
            ('s2_name', repeat_tensor(t['s2_name'], 1),
             repeat_eidx(t['s2_name_eidx'], 1)),
            ('s2_gender', repeat_tensor(t['s2_gender'], 1),
             repeat_eidx(t['s2_gender_eidx'], 1)),
            ('s2_interact', repeat_tensor(t['s2_interact'], 3), None),
            ('s2_auto', repeat_tensor(t['s2_auto'], 3), None),
            ('s3_entity', repeat_tensor(t['s3_entity'], 5),
             repeat_eidx(t['s3_entity_eidx'], 5)),
            ('s3_interact', repeat_tensor(t['s3_interact'], 15), None),
        ],
        quick_eval=lambda model: eval_full_mask(
            model, t['s3_entity'][:50], device, pad_id, mask_id,
            desc_start=FMT_META['s3_entity'][1])[0],
        end_eval=lambda model: _eval_phase6(
            model, data, t, device, pad_id, mask_id),
    )

    return phases


# ═══════════════════════════════════════════════════════
# Detailed evaluation at end of each phase
# ═══════════════════════════════════════════════════════

def _eval_phase1(model, t, device, pad_id, mask_id):
    p_acc, p_dim = eval_p_recon(model, t['s1_bare'][:200],
                                 device, pad_id, mask_id)
    print(f"    S1 P-recon:  {p_acc:.3f}")
    for dim, acc in p_dim.items():
        tag = '✓' if acc > 0.4 else '✗'
        print(f"      [{tag}] {dim}: {acc:.3f}")


def _eval_phase2(model, data, t, device, pad_id, mask_id, holdout_dims):
    ds = FMT_META['s2_name'][1]
    d_acc, d_dim = eval_desc_acc(model, t['s2_name'],
                                 device, pad_id, mask_id, desc_start=ds)
    h_acc = eval_desc_holdout(model, t['s2_name'],
                              device, pad_id, mask_id, desc_start=ds,
                              eidx_list=data['s2']['name_eidx'],
                              holdout_dims=holdout_dims)
    p_acc, _ = eval_p_recon(model, t['s1_bare'][:200],
                             device, pad_id, mask_id)
    print(f"    S1 P-recon:  {p_acc:.3f}")
    print(f"    S2.1 desc:   {d_acc:.3f}")
    print(f"    S2.1 held:   {h_acc:.3f}")
    for dim, acc in d_dim.items():
        tag = '✓' if acc > 0.4 else '✗'
        print(f"      [{tag}] {dim}: {acc:.3f}")


def _eval_phase3(model, t, device, pad_id, mask_id):
    ds = FMT_META['s2_gender'][1]
    g_acc, _ = eval_desc_acc(model, t['s2_gender'],
                             device, pad_id, mask_id, desc_start=ds)
    ds_n = FMT_META['s2_name'][1]
    n_acc, _ = eval_desc_acc(model, t['s2_name'][:50],
                             device, pad_id, mask_id, desc_start=ds_n)
    print(f"    S2.1 retain: {n_acc:.3f}")
    print(f"    S2.2 desc:   {g_acc:.3f}")


def _eval_phase4(model, t, device, pad_id, mask_id):
    ds = FMT_META['s2_article'][1]
    un_acc, _ = eval_desc_acc(model, t['s2_un'][:100],
                              device, pad_id, mask_id, desc_start=ds)
    il_acc, _ = eval_desc_acc(model, t['s2_il'][:100],
                              device, pad_id, mask_id, desc_start=ds)
    unh_acc, _ = eval_desc_acc(model, t['s2_un_held'][:100],
                               device, pad_id, mask_id, desc_start=ds)
    print(f"    S2.3 un:     {un_acc:.3f}")
    print(f"    S2.3 il:     {il_acc:.3f}")
    print(f"    S2.3 un-hld: {unh_acc:.3f}")


def _eval_phase5(model, t, device, pad_id, mask_id):
    ds_i = FMT_META['s2_interact'][1]
    ds_a = FMT_META['s2_auto'][1]
    i_acc, _ = eval_desc_acc(model, t['s2_interact'][:100],
                             device, pad_id, mask_id, desc_start=ds_i)
    a_acc, _ = eval_desc_acc(model, t['s2_auto'][:100],
                             device, pad_id, mask_id, desc_start=ds_a)
    print(f"    S2.4 inter:  {i_acc:.3f}")
    print(f"    S2.5 auto:   {a_acc:.3f}")


def _eval_phase6(model, data, t, device, pad_id, mask_id):
    ds_e = FMT_META['s3_entity'][1]
    ds_i = FMT_META['s3_interact'][1]

    e_acc, e_dim = eval_full_mask(model, t['s3_entity'],
                                  device, pad_id, mask_id, desc_start=ds_e)
    i_acc, i_dim = eval_full_mask(model, t['s3_interact'][:100],
                                  device, pad_id, mask_id, desc_start=ds_i)
    print(f"    S3 entity (full mask):   {e_acc:.3f}")
    print(f"    S3 interact (full mask): {i_acc:.3f}")

    if e_dim:
        print(f"    S3 entity per-dim:")
        for dim, acc in e_dim.items():
            tag = '✓' if acc > 0.4 else '✗'
            print(f"      [{tag}] {dim}: {acc:.3f}")

    # Probes
    if t['probe_s3'].shape[0] > 0:
        p_acc, _ = eval_full_mask(model, t['probe_s3'],
                                  device, pad_id, mask_id, desc_start=ds_i)
        print(f"    Probe (held-out pairs): {p_acc:.3f}")
        for idx, (a, b, rule, desc_txt, _) in enumerate(data['probes']['raw']):
            single = t['probe_s3'][idx:idx+1]
            sa, _ = eval_full_mask(model, single,
                                   device, pad_id, mask_id, desc_start=ds_i)
            print(f"      [{rule}] {a}+{b}: {sa:.3f}")

    # Animacy
    if t['anim_s3'].shape[0] > 0:
        a_acc, _ = eval_full_mask(model, t['anim_s3'],
                                  device, pad_id, mask_id, desc_start=ds_i)
        print(f"    Animacy (-in -us):       {a_acc:.3f}")
        for idx, (a, b, rule, _) in enumerate(data['animacy_probes']['raw']):
            single = t['anim_s3'][idx:idx+1]
            sa, _ = eval_full_mask(model, single,
                                   device, pad_id, mask_id, desc_start=ds_i)
            print(f"      [{rule}] {a}(-in -us)+{b}: {sa:.3f}")
