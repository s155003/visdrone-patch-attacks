"""
Paint-detection attack: detect the car's actual PAINT pixels (excluding glass/wheels/ground)
from image values, then apply adversarial pattern ONLY on detected paint.
6 combos: {brightness, domcolor} detection x {stricter, medium, looser} thresholds.
All conform to real paint, never glass, never off-car. car->van, 100 images, GPU.
"""
import importlib, patch_attack_paintdetect as PD

VARIANTS = [
    ("run53_bright_strict", dict(PAINT_METHOD="brightness", PAINT_LO_PCT=45)),  # exclude more (stricter)
    ("run54_bright_med",    dict(PAINT_METHOD="brightness", PAINT_LO_PCT=35)),  # medium
    ("run55_bright_loose",  dict(PAINT_METHOD="brightness", PAINT_LO_PCT=25)),  # exclude less (more paint)
    ("run56_domcolor",      dict(PAINT_METHOD="domcolor")),                     # dominant-color cluster
    ("run57_domcolor_v2",   dict(PAINT_METHOD="domcolor")),                     # (same, variance check)
]
for tag, cfg in VARIANTS:
    importlib.reload(PD)
    PD.USE_PAINT = True
    PD.PAINT_METHOD = "brightness"; PD.PAINT_LO_PCT = 35
    for k,v in cfg.items(): setattr(PD, k, v)
    PD.SAVE_DIR = f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70)
    PD.run()
