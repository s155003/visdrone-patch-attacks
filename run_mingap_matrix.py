"""
Minimal-gap tile sweep: tiles fill their cell MINUS a 2px hairline gap, so they
are PROVABLY non-merging but as large/contiguous-looking as possible.
Tests whether hairline separation preserves R26's ~47% while guaranteeing distinctness.
All independent patterns, 100 images, GPU. Sweep tile counts.
  R33: 25 tiles   R34: 36 tiles   R35: 49 tiles   R36: 64 tiles   R37: 81 tiles
Baselines: R26 (49 merged) 47.5%; R28 (49 big-gap separated) 16.7%; R32b (81 separated) 26.5%
"""
import importlib, patch_attack_mingap as MG

VARIANTS = [
    ("run33_25mingap", 25),
    ("run34_36mingap", 36),
    ("run35_49mingap", 49),
    ("run36_64mingap", 64),
    ("run37_81mingap", 81),
]
for tag, n in VARIANTS:
    importlib.reload(MG)
    MG.N_TILES = n
    MG.SEP_MODE = "mingap"
    MG.MIN_GAP_PX = 2
    MG.PATTERN_MODE = "independent"
    MG.SAVE_DIR = f"patch_examples_{tag}"
    print("\n" + "#"*70 + f"\n#  {tag.upper()}: {n} tiles, 2px hairline gap, independent -> {MG.SAVE_DIR}\n" + "#"*70)
    MG.run()
