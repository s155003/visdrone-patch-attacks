"""
CHAINED master run: per-box paint detection -> HSV color space -> tiles-in-paint.
Runs all three experiment families back-to-back. car->van, 100 images, GPU.

BATCH 1 - per-box paint detection (stats over car pixels only):
  R67 perbox bright loose | R68 perbox bright med | R69 perbox adaptive | R70 perbox union
BATCH 2 - HSV vs RGB color space:
  R71 RGB baseline | R72 HSV
BATCH 3 - 49 independent tiles filling the detected paint area:
  R73 paint-tiles-49
"""
import importlib

# ---- BATCH 1: per-box ----
import patch_attack_perbox as PB
for tag,cfg in [("run67_perbox_bright_loose",dict(COMBINE_METHOD="adaptive",PAINT_LO_PCT=25)),
                ("run68_perbox_bright_med",  dict(COMBINE_METHOD="adaptive",PAINT_LO_PCT=35)),
                ("run69_perbox_adaptive",    dict(COMBINE_METHOD="adaptive",PAINT_LO_PCT=30)),
                ("run70_perbox_union",       dict(COMBINE_METHOD="union",   PAINT_LO_PCT=25))]:
    importlib.reload(PB); PB.USE_PAINT=True; PB.COMBINE_METHOD="adaptive"; PB.PAINT_LO_PCT=30
    for k,v in cfg.items(): setattr(PB,k,v)
    PB.SAVE_DIR=f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70); PB.run()

# ---- BATCH 2: HSV vs RGB ----
import patch_attack_hsv as HV
for tag,cfg in [("run71_rgb_baseline",dict(COLOR_SPACE="rgb")),
                ("run72_hsv",         dict(COLOR_SPACE="hsv"))]:
    importlib.reload(HV); HV.USE_PAINT=True; HV.COMBINE_METHOD="adaptive"; HV.PAINT_LO_PCT=25
    for k,v in cfg.items(): setattr(HV,k,v)
    HV.SAVE_DIR=f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70); HV.run()

# ---- BATCH 3: tiles-in-paint ----
import patch_attack_painttiles as PT
for tag,cfg in [("run73_painttiles49",dict(PAINT_TILES=49,PAINT_TILE_INDEP=True))]:
    importlib.reload(PT); PT.USE_PAINT=True; PT.USE_PAINTTILES=True
    PT.COMBINE_METHOD="adaptive"; PT.PAINT_LO_PCT=25
    for k,v in cfg.items(): setattr(PT,k,v)
    PT.SAVE_DIR=f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70); PT.run()
