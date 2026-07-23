"""
Combined paint-detection attack: detect car paint using BOTH brightness + dom-color,
combined 4 ways. Hardened so paint stays strictly on the car (erosion + largest-blob +
tight inscribed rect). Adaptive routes light cars->brightness, dark cars->dom-color.
car->van, 100 images, GPU.
  R58 adaptive | R59 union | R60 intersection | R61 vote
"""
import importlib, patch_attack_paintcombo as PC

VARIANTS = [
    ("run58_adaptive",     dict(COMBINE_METHOD="adaptive")),
    ("run59_union",        dict(COMBINE_METHOD="union")),
    ("run60_intersection", dict(COMBINE_METHOD="intersection")),
    ("run61_vote",         dict(COMBINE_METHOD="vote")),
]
for tag, cfg in VARIANTS:
    importlib.reload(PC)
    PC.USE_PAINT = True
    PC.PAINT_LO_PCT = 35
    for k,v in cfg.items(): setattr(PC, k, v)
    PC.SAVE_DIR = f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70)
    PC.run()
