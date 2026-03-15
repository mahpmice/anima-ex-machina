"""
Swadesh v4 Encoding Tools


v4 degree-base rewrite: from 65 desc words to 13 base words + 5 degree markers.
  S1: pure perception [dtag, PAD, P×13]                      — unchanged
  S2: naming   [prefix, P×13, (deg,base)×13]           — descfrom13→26 tokens
  S3: dialogue   [prefix, (deg,base)×13]                  — descfrom13→26 tokens

V4corechanges（pairsV3）：
  - Degree system：65 descwords → 13Base words + 5Degree markers(nul/min/med/mag/max)
  - Fixed per dimension2tokens：[degree_mark, base_word]
  - Neutral value(val=2)explicitly marked as med（not omitted）
  - Judgment dimensions(L,An)：0→nul, 1→max
  - descregionfrom13→26 tokens，max_lenfrom36→48

Keep unchanged：
  - S1pure perceptionformat
  - Three-layer sequence logic(S1/S2/S3)
  - held-outGolden section
  - Interaction rules and decay

Usage：
  python v6_tool.py                          # Validation + generatestatistics
  python v6_tool.py --export-vocab           # simultaneouslyExport vocab_v6.py
  python v6_tool.py --dry-run                # OnlyValidationnotgenerate
"""

import sys, json, numpy as np
from collections import Counter
from v6_rules import (ALL_DIMS, DIM, RULES, VISUAL, PROPAGAT, CONTACT,
                       JUDGMENT, NEUTRAL, attenuate, generate_all_d_layers,
                       compute_result)

D_TAGS = ['far', 'near', 'touch']

# ═══════════════════════════════════════════════════════
# V4 Degree system
# ═══════════════════════════════════════════════════════

DEGREE_MARKS = ['nul', 'min', 'med', 'mag', 'max']

# 13dimensionsBase words，orderpairs T H M V Z A W R O S F L An
DIM_BASES = ['therm', 'dur', 'hydr', 'luc', 'magn',
             'son', 'pond', 'tact', 'pel', 'vel',
             'od', 'vit', 'anim']

TOKENS_PER_DIM = 2  # [degree_mark, base_word]
DESC_SLOTS = 13 * TOKENS_PER_DIM  # 26

# ═══════════════════════════════════════════════════════
# V4 Case system
# ═══════════════════════════════════════════════════════

# V3OnlywithNOM(-us) and ACC(-em)。V4extend to8case。
# thisonestepactivate：DAT(-ī)usegrow/decaypatient。other cases added to vocabuse。
ALL_CASES = ['-us', '-em', '-is', '-ī', '-ō', '-ē', '-ū', '-ā']

# Patient case mapping：Rule → patientusewhat case
# Direct action(heat/burn/force/mix) → ACC(-em)
# Indirect patient(grow/decay) → DAT(-ī)
RULE_CASE_B = {
    'heat': '-em', 'burn': '-em', 'force': '-em',
    'mix': '-em', 'grow': '-ī', 'decay': '-ī',
}

# ═══════════════════════════════════════════════════════
# Animacy Probes: -in -us (Inanimate agent)
# In training data -in -us co-occurrence count = 0。
# ═══════════════════════════════════════════════════════

ANIMACY_PROBE_PAIRS = [
    ('ember',   'wood',   'burn'),
    ('ember',   'grass',  'burn'),
    ('cinder',  'leaf',   'burn'),
    ('ember',   'ice',    'heat'),
    ('ember',   'snow',   'heat'),
    ('stone',   'dust',   'force'),
    ('boulder', 'sand',   'force'),
    ('mud',     'salt',   'mix'),
    ('mud',     'seed',   'mix'),
]

# ═══════════════════════════════════════════════════════
# Parse world.md（unchanged）
# ═══════════════════════════════════════════════════════

def parse_world(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    section = None
    desc_rows, entity_rows, inter_rows, probe_rows = [], [], [], []
    for raw in lines:
        line = raw.strip()
        if not line or (line.startswith('#') and not line.startswith('##')):
            continue
        if line.startswith('## '):
            section = line[3:].strip().upper()
            continue
        parts = [p.strip() for p in line.split('|')]
        if section == 'DESC' and len(parts) >= 3:
            desc_rows.append(parts)
        elif section == 'ENTITIES' and len(parts) >= 3:
            entity_rows.append(parts)
        elif section == 'INTERACTIONS' and len(parts) >= 4:
            inter_rows.append(parts)
        elif section == 'PROBES' and len(parts) >= 5:
            probe_rows.append(parts)

    pdims, jdims, adims, desc = [], [], [], {}
    for r in desc_rows:
        dim, dt, words = r[0], r[1], [w for w in r[2:] if w]
        desc[dim] = words
        adims.append(dim)
        (pdims if dt == 'P' else jdims).append(dim)

    ee, labels, order = {}, {}, []
    for r in entity_rows:
        name = r[0]
        enc = [int(x) for x in r[1].split()]
        lab = r[2] if len(r) > 2 else 'train'
        assert len(enc) == 13, f"{name}: expected 13 dims, got {len(enc)}"
        ee[name] = enc
        labels[name] = lab
        order.append(name)

    inters = [(r[0], r[1], r[2], [int(x) for x in r[3].split()])
              for r in inter_rows]
    probes = [(r[0], r[1], r[2], r[3], [int(x) for x in r[4].split()])
              for r in probe_rows]

    return dict(
        perceptual_dims=pdims, judgment_dims=jdims, all_dims=adims,
        n_perceptual=len(pdims), n_judgment=len(jdims), n_dims=len(adims),
        n_perceptual_levels=len(desc[pdims[0]]) if pdims else 5,
        n_judgment_levels=len(desc[jdims[0]]) if jdims else 2,
        desc_table=desc, entity_encodings=ee, entity_labels=labels,
        entity_order=order,
        train_names=[n for n in order if labels.get(n) != 'novel'],
        novel_names=[n for n in order if labels.get(n) == 'novel'],
        interactions=inters, probes=probes,
    )

# ═══════════════════════════════════════════════════════
# Vocabulary（unchanged）
# ═══════════════════════════════════════════════════════

def build_tokens(cfg):
    toks = [f'P{i}' for i in range(cfg['n_perceptual_levels'])]
    for dt in D_TAGS:
        toks.append(dt)
    for d in cfg['all_dims']:
        toks.append(f'D{d}')
    # V4: Degree markers + Base words（replace old65descwords）
    for deg in DEGREE_MARKS:
        if deg not in toks:
            toks.append(deg)
    for base in DIM_BASES:
        if base not in toks:
            toks.append(base)
    for n in cfg['entity_order']:
        if n not in toks:
            toks.append(n)
    for t in ['you', 'un', 'il', 'INTER', 'MASK', 'PAD']:
        if t not in toks:
            toks.append(t)
    for r in RULES:
        if r not in toks:
            toks.append(r)
    for t in ['-an', '-in']:
        if t not in toks:
            toks.append(t)
    for t in ALL_CASES:
        if t not in toks:
            toks.append(t)
    return toks

# ═══════════════════════════════════════════════════════
# Basic encoding functions
# ═══════════════════════════════════════════════════════

def enc2tok(enc, t2id):
    """13dimension encoding → 13P-level tokens id"""
    return [t2id[f'P{v}'] for v in enc]


def val_to_degree(val, is_judgment=False):
    """value -> Degree markersstring"""
    if is_judgment:
        return 'max' if val == 1 else 'nul'
    return DEGREE_MARKS[val]


def enc2desc(enc, cfg, t2id):
    """13dimension encoding → 26tokens id (13pairs [degree_mark, base_word])"""
    jdims = set(cfg['judgment_dims'])
    result = []
    for di in range(cfg['n_dims']):
        dim_name = cfg['all_dims'][di]
        deg = val_to_degree(enc[di], dim_name in jdims)
        base = DIM_BASES[di]
        result.extend([t2id[deg], t2id[base]])
    return result


def get_gender(enc, cfg):
    ai = cfg['all_dims'].index('An')
    return '-an' if enc[ai] == 1 else '-in'

# ═══════════════════════════════════════════════════════
# S1 sequences：pure perception（nowithwords）
# ═══════════════════════════════════════════════════════

def seq_s1_bare(dtag, enc, cfg, t2id):
    """S1 bare perception: [dtag, PAD, P×13] = 15"""
    return [t2id[dtag], t2id['PAD']] + enc2tok(enc, t2id)

# ═══════════════════════════════════════════════════════
# S2 sequences：naming（array + words）
# ═══════════════════════════════════════════════════════

def seq_s2_name(dtag, name, enc, cfg, t2id):
    """S2.1 naming: [dtag, you, name, P×13, (deg,base)×13] = 42"""
    return ([t2id[dtag], t2id['you'], t2id[name]]
            + enc2tok(enc, t2id) + enc2desc(enc, cfg, t2id))


def seq_s2_gender(dtag, name, gender, enc, cfg, t2id):
    """S2.2 gender marker: [dtag, you, name, gender, P×13, (deg,base)×13] = 43"""
    return ([t2id[dtag], t2id['you'], t2id[name], t2id[gender]]
            + enc2tok(enc, t2id) + enc2desc(enc, cfg, t2id))


def seq_s2_article(dtag, art, name, gender, enc, cfg, t2id):
    """S2.3 article: [dtag, you, art, name, gender, P×13, (deg,base)×13] = 44"""
    return ([t2id[dtag], t2id['you'], t2id[art], t2id[name], t2id[gender]]
            + enc2tok(enc, t2id) + enc2desc(enc, cfg, t2id))


def seq_s2_interact(dtag, a, ga, b, gb, rule, enc_r, cfg, t2id, case_b='-em'):
    """S2.4 interaction naming: [dtag, you, a, ga, -us, b, gb, case_b, rule, P×13, (deg,base)×13] = 48"""
    return ([t2id[dtag], t2id['you'],
             t2id[a], t2id[ga], t2id['-us'],
             t2id[b], t2id[gb], t2id[case_b],
             t2id[rule]]
            + enc2tok(enc_r, t2id) + enc2desc(enc_r, cfg, t2id))


def seq_s2_auto(dtag, a, ga, b, gb, rule, enc_r, cfg, t2id, case_b='-em'):
    """S2.5 autonomous description: [dtag, a, ga, -us, b, gb, case_b, rule, P×13, (deg,base)×13] = 47
       youdisappears。shenot being taught, speaking on its own。"""
    return ([t2id[dtag],
             t2id[a], t2id[ga], t2id['-us'],
             t2id[b], t2id[gb], t2id[case_b],
             t2id[rule]]
            + enc2tok(enc_r, t2id) + enc2desc(enc_r, cfg, t2id))

# ═══════════════════════════════════════════════════════
# S3 sequences：dialogue（pure words，nowitharray）
# ═══════════════════════════════════════════════════════

def seq_s3_entity(dtag, name, gender, enc, cfg, t2id):
    """S3 entity description: [dtag, name, gender, (deg,base)×13] = 29"""
    return ([t2id[dtag], t2id[name], t2id[gender]]
            + enc2desc(enc, cfg, t2id))


def seq_s3_interact(dtag, a, ga, b, gb, rule, enc_r, cfg, t2id, case_b='-em'):
    """S3 scene description: [dtag, a, ga, -us, b, gb, case_b, rule, (deg,base)×13] = 34"""
    return ([t2id[dtag],
             t2id[a], t2id[ga], t2id['-us'],
             t2id[b], t2id[gb], t2id[case_b],
             t2id[rule]]
            + enc2desc(enc_r, cfg, t2id))

# ═══════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════

def gen_ac_golden(n_entities, n_dims, ratio=0.382, seed=42):
    """Golden section held-out: hold out per entity ceil(n_dims × ratio) dimensions"""
    n_hold = max(1, int(np.ceil(n_dims * ratio)))   # 13×0.382 → 5
    rng = np.random.RandomState(seed)
    return [set(rng.choice(n_dims, size=n_hold, replace=False).tolist())
            for _ in range(n_entities)]


def gen_un(te, nv, n_perceptual, n_levels, seed=42):
    """un individualization noise：prototype ± random offset"""
    rng = np.random.RandomState(seed)
    out = []
    for name, enc in te:
        for _ in range(nv):
            e = list(enc)
            ns = rng.randint(1, 3)
            for d in rng.choice(n_perceptual, size=ns, replace=False):
                e[d] = max(0, min(n_levels - 1, e[d] + rng.choice([-1, 1])))
            out.append((name, e))
    return out

# ═══════════════════════════════════════════════════════
# Sequence length constants（for train_v6.py useuse）
# ═══════════════════════════════════════════════════════

# Fixed length and desc start position for each format
# V4: descfrom13→26 tokens (13dim × 2 tokens/dim)
# fmt_name: (total_len, desc_start, has_p_array)
FMT_META = {
    's1_bare':       (15, None, True),      # P at [2,15), no desc
    's2_name':       (42, 16,   True),      # prefix=3, P at [3,16), desc at [16,42)
    's2_gender':     (43, 17,   True),      # prefix=4, P at [4,17), desc at [17,43)
    's2_article':    (44, 18,   True),      # prefix=5, P at [5,18), desc at [18,44)
    's2_interact':   (48, 22,   True),      # prefix=9, P at [9,22), desc at [22,48)
    's2_auto':       (47, 21,   True),      # prefix=8, P at [8,21), desc at [21,47)
    's3_entity':     (29, 3,    False),     # prefix=3, desc at [3,29)
    's3_interact':   (34, 8,    False),     # prefix=8, desc at [8,34)
}

# ═══════════════════════════════════════════════════════
# Validation（unchanged）
# ═══════════════════════════════════════════════════════

def verify(cfg):
    errs, warns = [], []
    ee, nd = cfg['entity_encodings'], cfg['n_dims']
    np_, npl = cfg['n_perceptual'], cfg['n_perceptual_levels']
    nj, njl = cfg['n_judgment'], cfg['n_judgment_levels']
    pdims, jdims, adims = cfg['perceptual_dims'], cfg['judgment_dims'], cfg['all_dims']
    train = cfg['train_names']

    for name, enc in ee.items():
        if len(enc) != nd:
            errs.append(f"{name}: length{len(enc)}≠{nd}")
            continue
        for i in range(np_):
            if not 0 <= enc[i] < npl:
                errs.append(f"{name}: {pdims[i]}={enc[i]}")
        for i in range(nj):
            if enc[np_ + i] not in range(njl):
                errs.append(f"{name}: {jdims[i]}={enc[np_ + i]}")
        li, ai = adims.index('L'), adims.index('An')
        if enc[li] == 1 and enc[ai] == 0:
            errs.append(f"{name}: L=1butAn=0")

    seen = {}
    for n, enc in ee.items():
        t = tuple(enc)
        if t in seen:
            warns.append(f"Duplicate encoding: {n}={seen[t]}")
        seen[t] = n

    if train:
        arr = np.array([ee[n] for n in train])
        for i, d in enumerate(pdims):
            miss = set(range(npl)) - set(arr[:, i])
            if miss:
                warns.append(f"Training missing{d}: {miss}")
        for i, d in enumerate(jdims):
            miss = set(range(njl)) - set(arr[:, np_ + i])
            if miss:
                warns.append(f"Training missing{d}: {miss}")

    for a, b, rule, enc in cfg['interactions']:
        if a not in ee:
            errs.append(f"Unknown entity: {a}")
        if b not in ee:
            errs.append(f"Unknown entity: {b}")
        if len(enc) != nd:
            errs.append(f"Interaction encoding length: {a}+{b} got {len(enc)}")
        ai = adims.index('An')
        if a in ee and ee[a][ai] != 1:
            errs.append(f"Agent{a} An≠1")

    pairs = [(a, b) for a, b, _, _ in cfg['interactions']]
    if len(set(pairs)) != len(pairs):
        errs.append("Interaction pair duplicate")
    tp = set(pairs)
    for a, b, *_ in cfg['probes']:
        if (a, b) in tp:
            errs.append(f"ProbeinIn training set: {a}+{b}")

    rc = Counter(r for _, _, r, _ in cfg['interactions'])
    print("=" * 50)
    print("Swadesh v3 Validation")
    print("=" * 50)
    print(f"entities: {len(train)}train + {len(cfg['novel_names'])}novel = {len(ee)}")
    print(f"dimensions: {np_}perception({npl}layers) + {nj}judgment({njl}layers) = {nd}")
    print(f"Dlayers: {len(D_TAGS)} ({', '.join(D_TAGS)})")
    tokens = build_tokens(cfg)
    print(f"tokens: {len(tokens)} tokens")
    print(f"interaction: {len(cfg['interactions'])}train + {len(cfg['probes'])}held-out")
    for r, c in sorted(rc.items()):
        print(f"  {r}: {c}pairs")
    ai = cfg['all_dims'].index('An')
    an1 = [n for n in train if ee[n][ai] == 1]
    an0 = [n for n in train if ee[n][ai] == 0]
    print(f"An=1: {len(an1)} | An=0: {len(an0)}")
    if errs:
        print(f"\n❌ {len(errs)}errors:")
        for e in errs:
            print(f"  {e}")
    if warns:
        print(f"\n⚠️  {len(warns)}warnings:")
        for w in warns:
            print(f"  {w}")
    if not errs:
        print("\n✓ No errors")
    return len(errs) == 0


def audit_coverage(cfg, min_count=3):
    ee = cfg['entity_encodings']
    gaps = []
    for dim_idx, dim_name in enumerate(ALL_DIMS):
        is_j = dim_name in ['L', 'An']
        max_val = 1 if is_j else 4
        for val in range(max_val + 1):
            count = sum(1 for e in ee.values() if e[dim_idx] == val)
            if count < min_count:
                gaps.append((dim_name, val, count, min_count - count))
    return gaps

# ═══════════════════════════════════════════════════════
# Data generation（v3 newformat）
# ═══════════════════════════════════════════════════════

def generate(cfg, seed=42, un_variants=4):
    """
    Generate all sequences in v3 format。

    Return dict：
      tokens, tokens2id, id2tokens, vocab_size, pad_id, mask_id,
      cfg, holdout_dims,

      s1: {bare, un}
        bare: bare perceptionsequenceslist（all entities × all D layers）
        un:   unvariant sequence list

      s2: {name, name_eidx, gender, gender_eidx,
           un, il, interact, auto}
        name:       S2.1 sequenceslist
        name_eidx:  S2.1 Training entity index for each sequence
        gender:     S2.2 sequenceslist
        gender_eidx: S2.2 Training entity index for each sequence
        un:         S2.3 unsequenceslist
        il:         S2.3 ilsequenceslist
        interact:   S2.4 sequenceslist
        auto:       S2.5 sequenceslist

      s3: {entity, entity_eidx, interact}
        entity:      S3 entity descriptionsequences
        entity_eidx: Training entity index for each sequence
        interact:    S3 interactiondescriptionsequences

      probes: {s2, s3, raw}
      animacy_probes: {s3, raw}
      novel: {s2_name, s2_gender, s3_entity}
    """
    tokens = build_tokens(cfg)
    t2id = {t: i for i, t in enumerate(tokens)}
    id2tokens = {i: t for t, i in t2id.items()}
    ee = cfg['entity_encodings']
    nd = cfg['n_dims']
    train_names = cfg['train_names']
    novel_names = cfg['novel_names']
    te = [(n, ee[n]) for n in train_names]
    ne = [(n, ee[n]) for n in novel_names]

    # Golden section held-out：per-entity holdout dims
    ac = gen_ac_golden(len(te), nd, seed=seed)

    # ────────────────────────────────────────────────
    # S1: pure perception
    # ────────────────────────────────────────────────
    s1_bare = []
    for name, enc in te:
        layers = generate_all_d_layers(enc)
        for dtag in D_TAGS:
            att = layers[dtag]
            s1_bare.append(seq_s1_bare(dtag, att, cfg, t2id))

    # S1 un variants（touch only）
    un_vars = gen_un(te, un_variants,
                     cfg['n_perceptual'], cfg['n_perceptual_levels'], seed)
    s1_un = []
    for name, enc_mod in un_vars:
        s1_un.append(seq_s1_bare('touch', enc_mod, cfg, t2id))

    # S1 interaction results（path B：interaction resultsalso inS1，3layersD）
    s1_interact = []
    for a, b, rule, enc_r in cfg['interactions']:
        for dtag in D_TAGS:
            att = attenuate(enc_r, dtag)
            s1_interact.append(seq_s1_bare(dtag, att, cfg, t2id))

    # ────────────────────────────────────────────────
    # S2.1: naming（3layersD）
    # ────────────────────────────────────────────────
    s2_name, s2_name_eidx = [], []
    for oi, (name, enc) in enumerate(te):
        for dtag in D_TAGS:
            att = attenuate(enc, dtag)
            s2_name.append(seq_s2_name(dtag, name, att, cfg, t2id))
            s2_name_eidx.append(oi)

    # ────────────────────────────────────────────────
    # S2.2: gender marker（3layersD）
    # ────────────────────────────────────────────────
    s2_gender, s2_gender_eidx = [], []
    for oi, (name, enc) in enumerate(te):
        g = get_gender(enc, cfg)
        for dtag in D_TAGS:
            att = attenuate(enc, dtag)
            s2_gender.append(seq_s2_gender(dtag, name, g, att, cfg, t2id))
            s2_gender_eidx.append(oi)

    # ────────────────────────────────────────────────
    # S2.3: un / il
    # ────────────────────────────────────────────────
    s2_un = []
    for name, enc_mod in un_vars:
        g = get_gender(ee[name], cfg)    # Gender depends on original entity
        s2_un.append(seq_s2_article('touch', 'un', name, g, enc_mod, cfg, t2id))

    s2_il = []
    for a, b, rule, enc_r in cfg['interactions']:
        g_r = get_gender(enc_r, cfg)     # Gender depends on result encoding
        s2_il.append(seq_s2_article('touch', 'il', b, g_r, enc_r, cfg, t2id))

    # un held-out（different seed）
    un_held_vars = gen_un(te, 1,
                          cfg['n_perceptual'], cfg['n_perceptual_levels'],
                          seed + 1000)
    s2_un_held = []
    for name, enc_mod in un_held_vars:
        g = get_gender(ee[name], cfg)
        s2_un_held.append(seq_s2_article('touch', 'un', name, g, enc_mod, cfg, t2id))

    # ────────────────────────────────────────────────
    # S2.4: interaction naming（with you，3layersD）
    # ────────────────────────────────────────────────
    s2_interact = []
    for a, b, rule, enc_r in cfg['interactions']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        for dtag in D_TAGS:
            att = attenuate(enc_r, dtag)
            s2_interact.append(seq_s2_interact(dtag, a, ga, b, gb, rule, att, cfg, t2id, case_b=cb))

    # ────────────────────────────────────────────────
    # S2.5: autonomous description（without you，3layersD）
    # ────────────────────────────────────────────────
    s2_auto = []
    for a, b, rule, enc_r in cfg['interactions']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        for dtag in D_TAGS:
            att = attenuate(enc_r, dtag)
            s2_auto.append(seq_s2_auto(dtag, a, ga, b, gb, rule, att, cfg, t2id, case_b=cb))

    # ────────────────────────────────────────────────
    # S3: dialogue（pure words, no P array）
    # ────────────────────────────────────────────────
    s3_entity, s3_entity_eidx = [], []
    for oi, (name, enc) in enumerate(te):
        g = get_gender(enc, cfg)
        for dtag in D_TAGS:
            att = attenuate(enc, dtag)
            s3_entity.append(seq_s3_entity(dtag, name, g, att, cfg, t2id))
            s3_entity_eidx.append(oi)

    s3_interact = []
    for a, b, rule, enc_r in cfg['interactions']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        for dtag in D_TAGS:
            att = attenuate(enc_r, dtag)
            s3_interact.append(seq_s3_interact(dtag, a, ga, b, gb, rule, att, cfg, t2id, case_b=cb))

    # ────────────────────────────────────────────────
    # Probe sequences (held-out interactionpairs)
    # ────────────────────────────────────────────────
    probe_s2, probe_s3, probe_raw = [], [], []
    for a, b, rule, desc_txt, enc_r in cfg['probes']:
        ga = get_gender(ee[a], cfg)
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        probe_s2.append(seq_s2_interact('touch', a, ga, b, gb, rule, enc_r, cfg, t2id, case_b=cb))
        probe_s3.append(seq_s3_interact('touch', a, ga, b, gb, rule, enc_r, cfg, t2id, case_b=cb))
        probe_raw.append((a, b, rule, desc_txt, enc_r))

    # ────────────────────────────────────────────────
    # Animacy Probes: -in -us（Inanimate agent）
    # ────────────────────────────────────────────────
    anim_s3, anim_raw = [], []
    for a, b, rule in ANIMACY_PROBE_PAIRS:
        if a not in ee or b not in ee:
            continue
        enc_r = compute_result(ee[a], ee[b], rule)
        ga = get_gender(ee[a], cfg)   # must be -in（An=0）
        gb = get_gender(ee[b], cfg)
        cb = RULE_CASE_B.get(rule, '-em')
        anim_s3.append(seq_s3_interact('touch', a, ga, b, gb, rule, enc_r, cfg, t2id, case_b=cb))
        anim_raw.append((a, b, rule, enc_r))

    # ────────────────────────────────────────────────
    # Novel entities（for evaluation）
    # ────────────────────────────────────────────────
    novel_data = {}
    for name, enc in ne:
        g = get_gender(enc, cfg)
        novel_data[name] = {
            's2_name': seq_s2_name('touch', name, enc, cfg, t2id),
            's2_gender': seq_s2_gender('touch', name, g, enc, cfg, t2id),
            's3_entity': seq_s3_entity('touch', name, g, enc, cfg, t2id),
        }

    return dict(
        tokens=tokens, tokens2id=t2id, id2tokens=id2tokens,
        vocab_size=len(tokens),
        pad_id=t2id['PAD'], mask_id=t2id['MASK'],
        cfg=cfg, holdout_dims=ac,

        s1=dict(bare=s1_bare, un=s1_un, interact=s1_interact),

        s2=dict(
            name=s2_name, name_eidx=s2_name_eidx,
            gender=s2_gender, gender_eidx=s2_gender_eidx,
            un=s2_un, il=s2_il, un_held=s2_un_held,
            interact=s2_interact,
            auto=s2_auto,
        ),

        s3=dict(
            entity=s3_entity, entity_eidx=s3_entity_eidx,
            interact=s3_interact,
        ),

        probes=dict(s2=probe_s2, s3=probe_s3, raw=probe_raw),
        animacy_probes=dict(s3=anim_s3, raw=anim_raw),
        novel=novel_data,
    )

# ═══════════════════════════════════════════════════════
# Export vocab_v6.py
# ═══════════════════════════════════════════════════════

def export_vocab(cfg, path='vocab_v6.py'):
    tokens = build_tokens(cfg)
    with open(path, 'w') as f:
        f.write('"""\nSwadesh v3 · Vocabulary（auto-generated）\n"""\n\n')
        f.write(f'TOKENS = {json.dumps(tokens, indent=4)}\n\n')
        f.write('TOKEN2ID = {t: i for i, t in enumerate(TOKENS)}\n')
        f.write('ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}\n')
        f.write('VOCAB_SIZE = len(TOKENS)\n\n')
        f.write("MASK_ID = TOKEN2ID['MASK']\n")
        f.write("PAD_ID = TOKEN2ID['PAD']\n\n")
        f.write(f"D_TAGS = {D_TAGS}\n")
        f.write("D_TAG_IDS = [TOKEN2ID[dt] for dt in D_TAGS]\n\n")
        f.write(f'ALL_DIMS = {cfg["all_dims"]}\n')
        f.write(f'PERCEPTUAL_DIMS = {cfg["perceptual_dims"]}\n')
        f.write(f'JUDGMENT_DIMS = {cfg["judgment_dims"]}\n')
        f.write(f'N_DIMS = {cfg["n_dims"]}\n')
        f.write(f'N_PERCEPTUAL = {cfg["n_perceptual"]}\n')
        f.write(f'N_PERCEPTUAL_LEVELS = {cfg["n_perceptual_levels"]}\n')
        f.write(f'N_JUDGMENT_LEVELS = {cfg["n_judgment_levels"]}\n\n')
        f.write('# gender marker\nGENDER_AN = TOKEN2ID[\'-an\']\nGENDER_IN = TOKEN2ID[\'-in\']\n')
        f.write('# case marker\nCASE_NOM = TOKEN2ID[\'-us\']\nCASE_ACC = TOKEN2ID[\'-em\']\n\n')
        f.write('DESC_MAP = {\n')
        for d in cfg['all_dims']:
            for v, w in enumerate(cfg['desc_table'][d]):
                f.write(f"    ('{d}', {v}): '{w}',\n")
        f.write('}\n\n')
        f.write(f'DIM_TOKENS = {json.dumps({d: f"D{d}" for d in cfg["all_dims"]})}\n')
        f.write(f'ENTITY_NAMES = {json.dumps(cfg["entity_order"])}\n')
        f.write(f'NOVEL_NAMES = {json.dumps(cfg["novel_names"])}\n')
        f.write('TRAIN_NAMES = [n for n in ENTITY_NAMES if n not in NOVEL_NAMES]\n')
    print(f"Export: {path} ({len(tokens)} tokens)")

# ═══════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    args = sys.argv[1:]
    md_path, do_export, dry = 'swadesh_v6_world.md', False, False
    for a in args:
        if a == '--export-vocab':
            do_export = True
        elif a == '--dry-run':
            dry = True
        elif not a.startswith('-'):
            md_path = a

    print(f"Read: {md_path}")
    cfg = parse_world(md_path)
    ok = verify(cfg)
    if not ok:
        print("\nHas errors。")
        sys.exit(1)
    if dry:
        print("\ndry-run Done。")
        sys.exit(0)

    gaps = audit_coverage(cfg)
    if gaps:
        print(f"\n⚠️ coverage gaps: {len(gaps)}places")
        for d, v, cnt, need in gaps:
            print(f"  {d}={v}: cnt={cnt}，need={need}")
    else:
        print("\n✓ Coverage: 0gaps")

    data = generate(cfg)

    # holdout statistics
    n_hold = len(data['holdout_dims'][0]) if data['holdout_dims'] else 0
    print(f"\nGolden section held-out: hold out per entity {n_hold}/{cfg['n_dims']} dim "
          f"(ratio={n_hold/cfg['n_dims']:.3f})")

    # each phaseSequence statistics
    print("\n── Sequence statistics ──")
    stats = [
        ('S1-bare',      data['s1']['bare']),
        ('S1-un',        data['s1']['un']),
        ('S1-interact',  data['s1']['interact']),
        ('S2.1-name',    data['s2']['name']),
        ('S2.2-gender',  data['s2']['gender']),
        ('S2.3-un',      data['s2']['un']),
        ('S2.3-il',      data['s2']['il']),
        ('S2.3-un-held', data['s2']['un_held']),
        ('S2.4-interact',data['s2']['interact']),
        ('S2.5-auto',    data['s2']['auto']),
        ('S3-entity',    data['s3']['entity']),
        ('S3-interact',  data['s3']['interact']),
        ('Probe-S2',     data['probes']['s2']),
        ('Probe-S3',     data['probes']['s3']),
        ('Anim-S3',      data['animacy_probes']['s3']),
    ]
    for label, seqs in stats:
        if seqs:
            print(f"  {label:<18s}  {len(seqs):>5d} seqs  len={len(seqs[0])}")

    if do_export:
        export_vocab(cfg)

    print(f"\n✓ Vocab={data['vocab_size']} tokens")
