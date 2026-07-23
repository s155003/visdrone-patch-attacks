"""
Overnight batch: body-panel recovery + shapes + orientations.
ALL patches stay ON the car body, OFF the glass band, INSIDE the car boundary.
Target: 20-30% ASR while physically realistic.

Group 1 - body-panel ASR recovery:
  R38 both-ends (hood+trunk), narrow glass band
  R39 bigger single-end patch
  R40 narrow glass band only
Group 2 - grad student's shapes (single, on-body, off-glass):
  R41 circle   R42 triangle   R43 hexagon   R44 lshape   R45 cross
Group 3 - multi-shape stacking + orientation:
  R46 shapes stacked (overlapping) on body
  R47 shapes apart (separated) on body
  R48 tilted footprint (R13-style) + shape, on body
"""
import importlib, patch_attack_shapes as SH

# (tag, dict of overrides)
VARIANTS = [
    # Group 1: body-panel recovery (square shape)
    ("run38_bothends",   dict(SHAPE="square", BOTH_ENDS=True,  GLASS_BAND=0.30, FIXED_FRAC=0.50)),
    ("run39_biggerend",  dict(SHAPE="square", BOTH_ENDS=False, GLASS_BAND=0.40, FIXED_FRAC=0.60)),
    ("run40_narrowglass",dict(SHAPE="square", BOTH_ENDS=False, GLASS_BAND=0.28, FIXED_FRAC=0.55)),
    # Group 2: single shapes on a body panel
    ("run41_circle",     dict(SHAPE="circle",   BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55)),
    ("run42_triangle",   dict(SHAPE="triangle", BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55)),
    ("run43_hexagon",    dict(SHAPE="hexagon",  BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55)),
    ("run44_lshape",     dict(SHAPE="lshape",   BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55)),
    ("run45_cross",      dict(SHAPE="cross",    BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55)),
    # Group 3: multi-shape + orientation
    ("run46_stacked",    dict(SHAPE="circle", BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.50, MULTI_SHAPE="stacked", MULTI_N=3)),
    ("run47_apart",      dict(SHAPE="circle", BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.50, MULTI_SHAPE="apart",   MULTI_N=3)),
    ("run48_tilted",     dict(SHAPE="hexagon", BOTH_ENDS=True, GLASS_BAND=0.30, FIXED_FRAC=0.55, TILTED=True)),
]

for tag, cfg in VARIANTS:
    importlib.reload(SH)
    # reset all knobs, then apply this variant's overrides
    SH.SHAPE="square"; SH.BOTH_ENDS=False; SH.TILTED=False; SH.MULTI_SHAPE=""; SH.MULTI_N=3
    SH.GLASS_BAND=0.40; SH.FIXED_FRAC=0.45; SH.END_OFFSET=0.35
    for k,v in cfg.items(): setattr(SH, k, v)
    SH.SAVE_DIR = f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70)
    SH.run()
