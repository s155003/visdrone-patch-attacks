"""
Test MORE tiles (same tile size) to see if more coverage rescues the scatter approach.
All independent-pattern, grid placement, 100 images, on GPU.
Run 24: 25 tiles
Run 25: 36 tiles
Run 26: 49 tiles (heavily overlapping -> near-contiguous; tests if coverage/contiguity matters)
Baseline: contiguous patch (R22) = 23.4%; 9-tile scatter (R16-19) = 3.3-5.2%
"""
import importlib, patch_attack_scatter_multi as SC

VARIANTS = [
    ("run24_25tiles", 25),
    ("run25_36tiles", 36),
    ("run26_49tiles", 49),
]
for tag, n in VARIANTS:
    importlib.reload(SC)
    SC.N_TILES  = n
    SC.PLACEMENT = "grid"          # grid so more tiles fill coverage evenly
    SC.PATTERN_MODE = "independent"
    SC.SAVE_DIR = f"patch_examples_{tag}"
    print("\n" + "#"*70 + f"\n#  {tag.upper()}: {n} tiles, independent, grid -> {SC.SAVE_DIR}\n" + "#"*70)
    SC.run()
