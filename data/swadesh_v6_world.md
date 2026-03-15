# Swadesh v6 · World Configuration
# Draft 2026-03-13
#
# This file is the sole data source. v6_tool.py reads this to generate all training data.
# To add entities, modify encodings, or adjust rules, edit only this file.
#
# v6.1 change: D (distance) moved out of entity encoding, becomes sequence condition.
# Entity encoding reduced from 14-dim to 13-dim: T H M V Z A W R O S F L An
# D has 3 values: far / near / touch, as sequence prefix.
# Decay function auto-masks dimensions based on D (returns 2=no info).
#
# Format conventions:
#   - Tables use | separator
#   - Section titles starting with ## are tool anchors
#   - Lines starting with # are comments, skipped by tool
#   - Empty lines skipped

## DESC
# Description vocab. Format: dim | type | val0 | val1 | val2 | val3 | val4
# Type: P=perception(5 levels) J=judgment(2 levels)
# Perception dims have 5 words (0-4), judgment dims 2 (0-1), rest empty
T  | P | gel   | frig  | tep   | cal   | ferv
H  | P | flux  | mol   | firm  | dur   | rig
M  | P | arid  | sic   | ror   | hum   | mad
V  | P | nox   | dim   | lux   | clar  | fulg
Z  | P | min   | parv  | med   | mag   | max
A  | P | mut   | sus   | aud   | ton   | rug
W  | P | van   | lev   | libr  | grav  | dens
R  | P | lis   | sub   | plan  | asp   | scab
O  | P | vit   | pel   | tran  | turb  | opac
S  | P | quie  | lent  | mod   | cel   | rap
F  | P | pur   | ten   | odo   | acr   | putr
L  | J | mort  | viv
An | J | inan  | anim

## DTAGS
# Distance condition tags. D is not entity property, but observation condition.
# Decay rules:
#   Visual channels (V, Z, S)    — keep original at all D-layers
#   Propagation (A, F)           — gradual decay: touch→orig, near→orig-1, far→orig-2 (min 0)
#   Contact (T, H, M, R, W, O)   — far/near→2(no info), touch→original
# L, An (judgment dims) — keep original at all D-layers
far   | V Z S original | A F original-2 | T H M R W O to 2
near  | V Z S original | A F original-1 | T H M R W O to 2
touch | all original

## ENTITIES
# Entity encoding table. Format: name | T H M V Z A W R O S F L An | label
# Perception dims range 0-4, judgment dims 0/1
# Constraint: L=1 → An=1
# Encoding is full perception at touch distance (all dims active)
#
# Entity encoding table. Format: name | T H M V Z D A W R O S F L An | label
# Perception dims range 0-4, judgment dims 0/1
# Constraint: L=1 → An=1
# Label: train / novel
#
# ═══ Animals (6) ═══
fish      | 1 1 4 2 1 0 1 1 3 3 2 1 1 | train
bird      | 2 1 0 3 1 4 0 1 4 4 0 1 1 | novel
snake     | 1 1 1 1 1 1 1 1 4 3 1 1 1 | train
worm      | 1 0 3 0 0 0 0 1 3 1 1 1 1 | train
ant       | 1 2 0 0 0 0 0 2 4 2 0 1 1 | train
louse     | 1 1 0 0 0 0 0 1 4 1 0 1 1 | train
#
# ═══ Natural (12) ═══
sun       | 3 2 0 4 4 0 2 2 0 0 0 0 1 | train
moon      | 1 2 0 3 3 0 2 2 0 0 0 0 1 | train
star      | 1 2 0 3 0 0 0 2 0 0 0 0 1 | train
fire      | 4 0 0 4 1 3 0 2 1 3 4 0 1 | train
water     | 1 0 4 2 3 1 2 1 1 2 0 0 1 | train
stone     | 1 4 0 1 2 0 3 3 4 0 0 0 0 | train
ice       | 0 4 1 3 2 0 2 0 1 0 0 0 0 | train
wind      | 1 0 0 0 4 3 0 2 0 4 0 0 1 | train
rain      | 1 0 4 1 1 2 1 1 1 3 0 0 1 | train
snow      | 0 1 2 3 1 0 0 1 3 1 0 0 0 | train
cloud     | 1 0 3 2 4 0 0 2 3 2 0 0 1 | train
lightning | 4 0 0 4 1 4 0 2 0 4 4 0 1 | train
#
# ═══ Products of Change (5) ═══
smoke     | 2 0 0 1 3 0 0 2 3 3 4 0 1 | train
steam     | 3 0 4 2 2 1 0 2 3 3 1 0 1 | train
ash       | 1 1 0 1 0 0 0 2 4 0 1 0 0 | train
mud       | 1 1 4 1 2 0 3 3 4 0 2 0 0 | train
dust      | 2 1 0 1 0 0 0 2 2 2 0 0 0 | train
#
# ═══ Plants (4) ═══
tree      | 2 3 2 2 4 1 4 3 4 0 1 1 1 | train
grass     | 2 1 2 2 0 1 0 1 3 0 1 1 1 | train
leaf      | 2 0 2 2 0 1 0 1 2 1 1 1 1 | train
seed      | 2 2 0 1 0 0 0 2 4 0 0 1 1 | train
#
# ═══ Materials (6) ═══
wood      | 2 3 0 2 2 1 3 3 4 0 1 0 0 | train
salt      | 1 3 0 3 0 0 1 3 2 0 0 0 0 | train
iron      | 1 4 0 2 1 1 4 2 4 0 1 0 0 | train
sand      | 2 1 0 2 0 0 1 3 4 0 0 0 0 | train
earth     | 1 2 2 1 4 0 4 3 4 0 1 0 0 | train
path      | 2 3 0 2 3 0 4 3 4 0 0 0 0 | train
#
# ═══ v4 additions (23 train + 2 novel only) ═══
rope      | 2 1 0 2 1 0 0 4 4 1 0 0 0 | train
clay      | 1 1 3 1 1 0 3 2 4 0 0 0 0 | train
root      | 1 2 2 1 0 0 1 3 4 0 1 1 1 | train
bark      | 2 3 0 1 1 0 1 4 4 0 1 0 0 | train
moss      | 2 0 4 1 0 0 0 1 4 0 1 1 1 | train
coal      | 1 3 0 0 0 0 2 2 4 0 3 0 0 | train
fog       | 1 0 4 1 4 0 0 2 4 1 0 0 0 | train
dew       | 1 0 4 3 0 0 0 1 0 0 0 0 0 | train
frost     | 0 2 1 3 0 0 0 2 1 0 0 0 0 | train
wave      | 1 0 4 2 3 3 3 1 2 4 0 0 1 | train
ember     | 3 1 0 3 0 0 0 2 3 0 2 0 0 | train
rust      | 1 2 0 1 0 0 2 4 4 0 1 0 0 | train
flame     | 4 0 0 4 0 3 0 2 0 4 4 0 1 | train
soot      | 1 1 0 0 0 0 0 3 4 0 3 0 0 | train
thorn     | 2 4 0 1 0 0 0 4 4 0 0 1 1 | train
flower    | 2 0 2 4 0 0 0 1 3 0 3 1 1 | train
fruit     | 2 1 3 3 1 0 1 1 3 0 2 1 1 | train
feather   | 2 0 0 2 0 0 0 1 4 3 0 0 0 | train
shell     | 1 4 0 2 0 1 1 3 4 0 0 0 0 | novel
claw      | 1 4 0 1 0 0 0 4 4 0 0 0 0 | novel
river     | 1 0 4 2 4 2 4 1 2 3 0 0 1 | train
lake      | 1 0 4 3 4 0 4 1 1 0 0 0 1 | train
sea       | 1 0 4 2 4 4 4 1 1 3 1 0 1 | train
mountain  | 1 4 0 2 4 0 4 4 4 0 0 0 0 | train
cave      | 1 4 2 0 3 1 4 3 4 0 1 0 0 | novel
honey     | 2 0 3 3 0 0 1 1 1 1 3 0 0 | novel
#
# ═══ Natural forces (new) (3) ═══
thunder   | 1 0 0 3 4 4 0 2 0 0 0 0 1 | train
spring    | 1 0 4 2 1 2 1 1 1 2 0 0 1 | train
tide      | 1 0 4 2 4 2 4 1 1 2 0 0 1 | train
#
# ═══ Water bodies (new) (3) ═══
pond      | 1 0 4 2 2 0 3 1 1 0 0 0 1 | train
marsh     | 1 0 4 2 3 1 3 3 3 0 2 0 1 | train
brook     | 1 0 4 3 1 2 1 1 1 3 0 0 1 | train
#
# ═══ Minerals (new) (6) ═══
boulder   | 1 4 0 1 3 0 4 4 4 0 0 0 0 | train
pebble    | 1 3 0 1 0 0 1 1 4 0 0 0 0 | train
copper    | 1 3 0 3 1 0 4 2 4 0 1 0 0 | train
chalk     | 1 2 0 3 1 0 1 2 4 0 0 0 0 | train
flint     | 1 4 0 1 1 0 2 0 4 0 0 0 0 | train
obsidian  | 1 4 0 1 1 0 2 0 1 0 0 0 0 | train
#
# ═══ Plants (new) (6) ═══
vine      | 2 0 3 2 2 0 1 2 3 0 1 1 1 | train
reed      | 2 1 3 2 2 1 0 1 3 1 1 1 1 | train
fern      | 2 0 2 2 1 0 0 1 3 0 1 1 1 | train
lichen    | 2 1 1 1 0 0 0 2 4 0 0 1 1 | train
mushroom  | 2 0 3 1 0 0 0 1 3 0 3 1 1 | train
algae     | 2 0 4 1 0 0 0 0 2 0 2 1 1 | train
#
# ═══ Animals (new) (10) ═══
horse     | 2 2 1 2 3 2 4 2 4 4 1 1 1 | train
bear      | 2 2 1 2 3 3 4 3 4 2 2 1 1 | train
ox        | 2 2 1 2 4 2 4 2 4 1 1 1 1 | train
wolf      | 2 1 1 2 2 3 3 2 4 4 2 1 1 | train
deer      | 2 1 1 2 3 1 3 1 4 4 1 1 1 | train
dog       | 2 1 1 2 2 3 2 2 4 3 1 1 1 | train
frog      | 2 0 3 2 0 2 0 0 4 3 1 1 1 | train
bee       | 2 1 0 2 0 2 0 1 4 4 0 1 1 | train
spider    | 2 1 1 1 0 0 0 1 4 3 0 1 1 | train
turtle    | 2 4 1 2 1 0 3 3 4 1 1 1 1 | train
#
# ═══ Products (new) (6) ═══
charcoal  | 2 3 0 0 0 0 2 3 4 0 3 0 0 | train
tar       | 2 1 2 0 0 0 2 1 4 0 4 0 0 | train
glass     | 1 4 0 3 1 0 2 0 1 0 1 0 0 | train
slag      | 2 3 0 1 0 0 3 4 4 0 1 0 0 | train
foam      | 1 0 3 3 2 0 0 0 3 1 1 0 0 | train
cinder    | 3 2 0 2 0 0 0 2 4 0 2 0 0 | train
#
# ═══ Terrain (new) (5) ═══
cliff     | 1 4 0 2 3 0 4 4 4 0 0 0 0 | train
hill      | 1 2 1 2 4 0 4 2 4 0 0 0 0 | train
valley    | 1 1 2 2 4 0 4 2 4 0 0 0 0 | train
island    | 1 2 2 3 4 1 4 2 4 0 0 0 0 | train
shore     | 1 1 3 3 3 2 3 1 4 0 1 0 0 | train
#
# ═══ Materials (new) (5) ═══
bone      | 2 4 0 3 1 0 1 1 4 0 0 0 0 | train
hide      | 2 1 1 1 1 0 1 2 4 0 2 0 0 | train
wool      | 2 0 0 2 1 0 0 3 4 0 1 0 0 | train
pitch     | 2 0 1 0 1 0 2 1 4 0 4 0 0 | train
amber     | 2 3 0 3 1 0 1 0 1 0 1 0 0 | train


## INTERACTIONS
# Interaction pairs. Format: obj_a | obj_b | rule | T H M V Z A W R O S F L An
# Interaction at touch distance. Result encoding is full perception at touch.
# obj_a must be entity with An=1
#
# Interaction pairs. Format: obj_a | obj_b | rule | T H M V Z D A W R O S F L An
# Result encoding auto-computed by rule functions (v6_rules.py)
# obj_a must be entity with An=1
#
# ── heat (14) ──
fire          | ice           | heat  | 2 2 3 3 2 0 2 0 1 1 0 0 0
fire          | snow          | heat  | 2 1 2 3 1 1 0 1 3 1 0 0 0
fire          | water         | heat  | 3 0 4 3 3 1 2 1 1 2 0 0 1
sun           | ice           | heat  | 2 2 3 3 2 0 2 0 1 1 0 0 0
sun           | snow          | heat  | 2 1 2 3 1 1 0 1 3 1 0 0 0
sun           | water         | heat  | 2 0 4 3 3 1 2 1 1 2 0 0 1
fire          | iron          | heat  | 3 2 2 3 1 1 4 2 4 1 1 0 0
sun           | earth         | heat  | 2 1 4 3 4 1 4 3 4 1 1 0 0
sun           | stone         | heat  | 2 2 2 3 2 0 3 3 4 1 0 0 0
fire          | copper        | heat  | 3 1 2 3 1 0 4 2 4 1 1 0 0
fire          | charcoal      | heat  | 3 1 0 3 0 0 2 3 4 1 3 0 0
fire          | stone         | heat  | 3 2 2 3 2 0 3 3 4 1 0 0 0
sun           | hide          | heat  | 3 1 1 3 1 0 1 2 4 1 2 0 0
fire          | glass         | heat  | 3 2 2 3 1 0 2 0 1 1 1 0 0
# ── force (17) ──
wind          | sand          | force | 2 1 0 1 1 2 1 3 4 3 0 0 0
wind          | dust          | force | 2 1 0 0 1 2 0 2 2 3 0 0 0
wind          | water         | force | 1 0 4 2 3 2 2 1 1 3 0 0 1
wind          | leaf          | force | 2 0 2 1 1 2 0 1 2 3 1 1 1
wind          | snow          | force | 0 1 2 2 2 2 0 1 3 3 0 0 0
wind          | ash           | force | 1 1 0 0 1 2 0 2 4 3 1 0 0
water         | stone         | force | 1 4 0 0 2 0 3 3 4 1 0 0 0
wind          | feather       | force | 2 0 0 1 1 2 0 1 4 3 0 0 0
sea           | stone         | force | 1 4 2 1 2 4 3 2 4 2 1 0 0
wind          | smoke         | force | 2 0 0 0 3 2 0 2 3 3 4 0 1
wind          | reed          | force | 2 1 3 1 2 2 0 1 3 3 1 1 1
wave          | shore         | force | 1 1 3 2 3 2 3 1 4 3 1 0 0
river         | pebble        | force | 1 3 0 0 1 1 1 1 4 2 0 0 0
tide          | sand          | force | 2 1 0 1 0 1 1 3 4 1 0 0 0
wind          | flame         | force | 4 0 0 4 1 3 0 2 0 4 4 0 1
river         | sand          | force | 2 1 0 1 1 1 1 3 4 2 0 0 0
wave          | sand          | force | 2 1 0 1 1 2 1 3 4 3 0 0 0
# ── R3: Phase change —— deferred to v7 (requires "no-agent" syntax) ──
# Phase change has no agent: water freezes when cold, who did it? No one.
# Two-entity format presumes agent/patient, unsuitable for phase change.
# ── burn (18) ──
fire          | tree          | burn  | 3 2 0 2 4 1 4 3 4 1 3 0 0
fire          | grass         | burn  | 3 0 0 2 0 1 0 1 3 1 3 0 0
fire          | wood          | burn  | 3 2 0 2 2 1 3 3 4 1 3 0 0
fire          | leaf          | burn  | 3 0 0 2 0 1 0 1 2 1 3 0 0
lightning     | tree          | burn  | 3 2 0 2 4 1 4 3 4 1 3 0 0
lightning     | grass         | burn  | 3 0 0 2 0 1 0 1 3 1 3 0 0
fire          | sand          | burn  | 3 3 0 2 0 1 1 3 4 1 3 0 0
fire          | clay          | burn  | 3 3 0 2 1 1 3 2 4 1 3 0 0
fire          | rope          | burn  | 3 3 0 2 1 1 0 4 4 1 3 0 0
fire          | vine          | burn  | 3 0 0 2 2 1 1 2 3 1 3 0 0
fire          | reed          | burn  | 3 0 0 2 2 1 0 1 3 1 3 0 0
fire          | moss          | burn  | 3 0 0 2 0 1 0 1 4 1 3 0 0
fire          | bark          | burn  | 3 2 0 2 1 1 1 4 4 1 3 0 0
fire          | coal          | burn  | 3 2 0 2 0 1 2 2 4 1 3 0 0
fire          | hide          | burn  | 3 3 0 2 1 1 1 2 4 1 3 0 0
fire          | wool          | burn  | 3 3 0 2 1 1 0 3 4 1 3 0 0
fire          | bone          | burn  | 3 3 0 3 1 1 1 1 4 1 3 0 0
lightning     | wood          | burn  | 3 2 0 2 2 1 3 3 4 1 3 0 0
# ── mix (13) ──
water         | earth         | mix   | 1 1 3 2 4 0 3 4 3 0 0 0 0
water         | sand          | mix   | 2 0 3 2 2 0 2 2 3 0 0 0 0
water         | dust          | mix   | 2 0 3 2 2 0 1 2 3 2 0 0 0
water         | ash           | mix   | 1 0 3 2 2 0 1 2 3 0 0 0 0
rain          | earth         | mix   | 1 1 3 1 2 1 2 4 3 0 0 0 0
rain          | sand          | mix   | 2 0 3 2 0 1 1 2 3 0 0 0 0
rain          | dust          | mix   | 2 0 3 1 0 1 0 2 3 2 0 0 0
water         | clay          | mix   | 1 0 4 2 2 0 2 2 3 0 0 0 0
water         | salt          | mix   | 1 2 3 2 2 0 2 2 3 0 0 0 0
water         | chalk         | mix   | 1 1 3 2 2 0 2 2 3 0 0 0 0
rain          | ash           | mix   | 1 0 3 1 0 1 0 2 3 0 0 0 0
water         | mud           | mix   | 1 0 4 2 2 0 2 2 3 0 1 0 0
water         | copper        | mix   | 1 2 3 2 2 0 3 2 3 0 0 0 0
# ── grow (14) ──
water         | seed          | grow  | 2 2 2 1 1 0 1 2 4 0 0 1 1
rain          | seed          | grow  | 2 2 2 1 1 0 1 2 4 0 0 1 1
water         | grass         | grow  | 2 1 2 2 1 1 1 1 3 0 1 1 1
rain          | tree          | grow  | 2 3 2 2 4 1 4 3 4 0 1 1 1
sun           | seed          | grow  | 3 2 0 1 0 0 0 2 4 0 0 1 1
sun           | grass         | grow  | 3 1 0 2 0 1 0 1 3 0 1 1 1
rain          | moss          | grow  | 2 0 4 1 1 0 1 1 4 0 1 1 1
water         | algae         | grow  | 2 0 4 1 1 0 1 0 2 0 2 1 1
rain          | vine          | grow  | 2 0 3 2 3 0 2 2 3 0 1 1 1
rain          | mushroom      | grow  | 2 0 3 1 1 0 1 1 3 0 3 1 1
rain          | fern          | grow  | 2 0 2 2 2 0 1 1 3 0 1 1 1
rain          | flower        | grow  | 2 0 2 4 1 0 1 1 3 0 3 1 1
water         | reed          | grow  | 2 1 3 2 3 1 1 1 3 1 1 1 1
sun           | vine          | grow  | 3 0 1 2 2 0 1 2 3 0 1 1 1
# ── decay (12) ──
water         | wood          | decay | 2 2 3 1 2 1 3 3 4 0 3 0 0
water         | leaf          | decay | 2 0 3 1 0 1 0 1 3 0 3 0 0
rain          | wood          | decay | 2 2 3 1 2 2 3 3 4 0 3 0 0
water         | bark          | decay | 2 2 3 0 1 1 1 4 4 0 3 0 0
rain          | bark          | decay | 2 2 3 0 1 2 1 4 4 0 3 0 0
water         | root          | decay | 1 1 3 0 0 1 1 3 4 0 3 0 0
water         | fruit         | decay | 2 0 3 2 1 1 1 1 3 0 4 0 0
rain          | hide          | decay | 2 0 3 0 1 2 1 2 4 0 4 0 0
water         | rope          | decay | 2 0 3 1 1 1 0 4 4 0 2 0 0
rain          | bone          | decay | 2 3 3 2 1 2 1 1 4 0 2 0 0
water         | wool          | decay | 2 0 3 1 1 1 0 3 4 0 3 0 0
rain          | reed          | decay | 2 0 3 1 2 2 0 1 3 0 3 0 0


## PROBES
# Held-out interaction pairs. Results auto-computed by rule functions.
# Held-out interaction pairs. Results auto-computed by rule functions.
fire          | frost         | heat  | like fire+ice       | 2 1 3 3 0 0 0 2 1 1 0 0 0
sun           | mud           | heat  | drying mud          | 2 1 4 3 2 1 3 3 4 1 2 0 0
wind          | rain          | force | wind blows rain     | 1 0 4 1 2 2 1 1 1 3 0 0 1
wind          | fire          | force | fire spreads        | 4 0 0 4 2 3 0 2 1 3 4 0 1
lightning     | iron          | heat  | molten iron         | 3 2 2 3 1 1 4 2 4 1 1 0 0
rain          | leaf          | decay | rain rots leaf      | 2 0 3 1 0 2 0 1 3 0 3 0 0
sun           | tree          | grow  | sun alone stress    | 3 3 0 2 4 1 4 3 4 0 1 1 1
wind          | cloud         | force | wind blows cloud    | 1 0 3 1 4 2 0 2 3 3 0 0 1
water         | iron          | decay | rust                | 1 3 3 1 1 1 4 2 4 0 3 0 0

