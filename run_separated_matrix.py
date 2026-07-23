"""
Separated-tiles matrix (all INDEPENDENT patterns, provably non-merging).
Tests whether the capacity effect survives when tiles are clearly separated.
  R28: 49 tiles, spaced (gaps every tile)
  R29: 49 tiles, shrunk
  R30: 49 tiles, both (smaller + spaced)
  R31: 7 bars, guaranteed gaps between bars (anti-amalgamation)
  R32a: 64 tiles, spaced
  R32b: 81 tiles, spaced
  R32c: 100 tiles, spaced
Baselines: R26 (49 merged/independent) 47.5%; R27 (49 shared) 2.7%
"""
import importlib, patch_attack_separated as SP

VARIANTS = [
    ("run28_49spaced", 49, "spaced"),
    ("run29_49shrunk", 49, "shrunk"),
    ("run30_49both",   49, "both"),
    ("run31_7bars",     7, "bars"),
    ("run32a_64tiles", 64, "spaced"),
    ("run32b_81tiles", 81, "spaced"),
    ("run32c_100tiles",100,"spaced"),
]
for tag, n, mode in VARIANTS:
    importlib.reload(SP)
    SP.N_TILES = n
    SP.SEP_MODE = mode
    SP.PATTERN_MODE = "independent"
    SP.SAVE_DIR = f"patch_examples_{tag}"
    print("\n" + "#"*70 + f"\n#  {tag.upper()}: n={n}, mode={mode}, independent -> {SP.SAVE_DIR}\n" + "#"*70)
    SP.run()
