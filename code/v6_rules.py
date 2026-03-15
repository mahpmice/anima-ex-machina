"""
Swadesh v6.1 · Rule functions + Decay functions
13-dim encoding: T H M V Z A W R O S F L An (D moved out as sequence condition)

Decay functions: Generate perceptual encodings at far/near/touch from touch encoding.
Rule functions: Compute interaction results from two entities' touch encodings.
"""

import numpy as np

# ═══════════════════════════════════════════════════
# Dimension indices (13 dims, no D)
# ═══════════════════════════════════════════════════

ALL_DIMS = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
DIM = {d: i for i, d in enumerate(ALL_DIMS)}

# Perceptual channel categories
VISUAL   = [DIM['V'], DIM['Z'], DIM['S']]        # visible from afar
PROPAGAT = [DIM['A'], DIM['F']]                    # gradual propagation
CONTACT  = [DIM['T'], DIM['H'], DIM['M'],          # contact required
            DIM['R'], DIM['W'], DIM['O']]
JUDGMENT = [DIM['L'], DIM['An']]                   # no decay

NEUTRAL = 2  # neutral state = perceiver itself = reference point

# ═══════════════════════════════════════════════════
# Decay functions
# ═══════════════════════════════════════════════════

def attenuate(touch_enc, d_tag):
    """
    Generate perceptual encoding at specified D-layer from touch encoding (full 13-dim).
    
    d_tag: 'far', 'near', 'touch'
    Return: 13dimension encoding
    
    Rule：
      Visual channels (V, Z, S)       — keep original value at all D-layers
      Propagation channels (A, F)           — gradual decay：
                                   touch → original value
                                   near  → max(0, original value - 1)
                                   far   → max(0, original value - 2)
      Contact channels (T, H, M, R, W, O) — far/near → 2(neutral), touch → original value
      Judgment dimensions (L, An)          — keep original value at all D-layers
    """
    r = list(touch_enc)
    
    if d_tag == 'touch':
        return r
    
    # Contact channels：far and nearallreturn to neutral
    for i in CONTACT:
        r[i] = NEUTRAL
    
    # Propagation channels：gradual decay
    if d_tag == 'near':
        for i in PROPAGAT:
            r[i] = max(0, touch_enc[i] - 1)
    elif d_tag == 'far':
        for i in PROPAGAT:
            r[i] = max(0, touch_enc[i] - 2)
    else:
        raise ValueError(f"Unknown D tag: {d_tag}")
    
    return r


def generate_all_d_layers(touch_enc):
    """One entity's touch encoding → 3D-layer encodings"""
    return {
        'far':   attenuate(touch_enc, 'far'),
        'near':  attenuate(touch_enc, 'near'),
        'touch': attenuate(touch_enc, 'touch'),
    }


# ═══════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════

def clamp(v, lo=0, hi=4): return max(lo, min(hi, int(round(v))))
def clampj(v): return max(0, min(1, int(round(v))))
def mid(a, b): return clamp((a + b) / 2)

# ═══════════════════════════════════════════════════
# Interaction rules（13dims, no D）
# ═══════════════════════════════════════════════════

def rule_heat(enc_a, enc_b):
    """Heat transfer：high-temp source heats low-temp object。"""
    r = list(enc_b)
    r[DIM['T']] = clamp(enc_b[DIM['T']] + (enc_a[DIM['T']] - enc_b[DIM['T']]) * 0.6)
    if enc_b[DIM['H']] >= 3 and enc_a[DIM['T']] >= 3:
        r[DIM['H']] = clamp(enc_b[DIM['H']] - 2)
    elif enc_b[DIM['H']] >= 2 and enc_a[DIM['T']] >= 3:
        r[DIM['H']] = clamp(enc_b[DIM['H']] - 1)
    if enc_b[DIM['T']] <= 1 and enc_b[DIM['H']] >= 2 and enc_a[DIM['T']] >= 3:
        r[DIM['M']] = clamp(enc_b[DIM['M']] + 2)
    if enc_a[DIM['V']] >= 3:
        r[DIM['V']] = clamp(max(enc_b[DIM['V']], enc_a[DIM['V']] - 1))
    if enc_a[DIM['T']] >= 3:
        r[DIM['S']] = clamp(max(enc_b[DIM['S']], 1))
    if enc_a[DIM['T']] >= 3 and enc_b[DIM['M']] >= 2:
        r[DIM['A']] = clamp(max(enc_b[DIM['A']], 1))
    return r

def rule_force(enc_a, enc_b):
    """Force transfer：agent pushes patient to move。"""
    r = list(enc_b)
    r[DIM['S']] = clamp(max(enc_b[DIM['S']], enc_a[DIM['S']] - 1))
    r[DIM['A']] = clamp(max(enc_b[DIM['A']], enc_a[DIM['A']] - 1))
    if enc_b[DIM['Z']] <= 1 and enc_a[DIM['S']] >= 3:
        r[DIM['Z']] = clamp(enc_b[DIM['Z']] + 1)
    if enc_b[DIM['O']] >= 2:
        r[DIM['V']] = clamp(enc_b[DIM['V']] - 1)
    return r

def rule_burn(enc_a, enc_b):
    """Combustion：fire/lightning + combustible -> destruction。"""
    r = list(enc_b)
    r[DIM['T']] = clamp(max(enc_b[DIM['T']], enc_a[DIM['T']] - 1))
    r[DIM['V']] = clamp(max(enc_b[DIM['V']], 2))
    r[DIM['A']] = clamp(max(enc_b[DIM['A']], 1))
    r[DIM['F']] = clamp(max(enc_b[DIM['F']], 3))
    r[DIM['M']] = 0
    if enc_b[DIM['H']] <= 1 and enc_b[DIM['L']] == 0:
        r[DIM['H']] = 3
    elif enc_b[DIM['H']] >= 3:
        r[DIM['H']] = clamp(enc_b[DIM['H']] - 1)
    else:
        r[DIM['H']] = clamp(enc_b[DIM['H']] - 1)
    r[DIM['L']] = 0
    r[DIM['An']] = 0
    r[DIM['S']] = clamp(max(enc_b[DIM['S']], 1))
    if enc_b[DIM['L']] == 1:
        r[DIM['O']] = clamp(max(enc_b[DIM['O']], 2))
    return r

def rule_mix(enc_a, enc_b):
    """Mixing：liquid+solid/powder->dims tend toward liquid，Mincrease。"""
    r = [0]*13
    for i in range(11):  # perceptual dims tend to neutral
        r[i] = mid(enc_a[i], enc_b[i])
    r[DIM['M']] = clamp(max(mid(enc_a[DIM['M']], enc_b[DIM['M']]), enc_a[DIM['M']] - 1))
    r[DIM['O']] = clamp(max(mid(enc_a[DIM['O']], enc_b[DIM['O']]), 3))
    r[DIM['S']] = clamp(min(enc_a[DIM['S']], enc_b[DIM['S']]))
    r[DIM['L']] = 0
    r[DIM['An']] = 0
    return r

def rule_grow(enc_a, enc_b):
    """Growth：water/rain + seed/plant -> life enhanced。"""
    r = list(enc_b)
    is_sun = enc_a[DIM['T']] >= 3 and enc_a[DIM['V']] >= 4
    if is_sun:
        r[DIM['M']] = clamp(enc_b[DIM['M']] - 2)
        r[DIM['T']] = clamp(enc_b[DIM['T']] + 1)
        if enc_b[DIM['L']] == 1:
            r[DIM['L']] = 1
        r[DIM['An']] = 1
    else:
        r[DIM['L']] = 1
        r[DIM['An']] = 1
        r[DIM['M']] = clamp(max(enc_b[DIM['M']], 2))
        r[DIM['Z']] = clamp(enc_b[DIM['Z']] + 1)
        r[DIM['W']] = clamp(enc_b[DIM['W']] + 1)
    return r

def rule_decay(enc_a, enc_b):
    """Decay：water + dead matter -> decomposition。"""
    r = list(enc_b)
    r[DIM['H']] = clamp(enc_b[DIM['H']] - 1)
    r[DIM['F']] = clamp(enc_b[DIM['F']] + 2)
    r[DIM['M']] = clamp(max(enc_b[DIM['M']], enc_a[DIM['M']] - 1))
    r[DIM['V']] = clamp(enc_b[DIM['V']] - 1)
    r[DIM['O']] = clamp(max(enc_b[DIM['O']], 3))
    r[DIM['L']] = 0
    r[DIM['An']] = 0
    r[DIM['S']] = 0
    r[DIM['A']] = clamp(max(enc_b[DIM['A']], enc_a[DIM['A']]))
    return r


RULES = {
    'heat': rule_heat,
    'force': rule_force,
    'burn': rule_burn,
    'mix': rule_mix,
    'grow': rule_grow,
    'decay': rule_decay,
}

def compute_result(enc_a, enc_b, rule_name):
    """Given two entity encodings and rule name, compute result encoding"""
    fn = RULES.get(rule_name)
    if fn is None:
        raise ValueError(f"Unknown rule: {rule_name}")
    return fn(enc_a, enc_b)
