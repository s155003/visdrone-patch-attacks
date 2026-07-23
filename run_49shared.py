"""Control R27: 49 tiles, SHARED pattern (all tiles identical), grid, 100 img.
Same coverage/tiling as R26 (49 independent); ONLY the pattern capacity differs.
Isolates whether independent per-region capacity is the driver of R26's 47.5%."""
import patch_attack_scatter_multi as SC

SC.N_TILES      = 49
SC.PLACEMENT    = "grid"
SC.PATTERN_MODE = "shared"     # <-- the only change vs R26 (which was "independent")
SC.SAVE_DIR     = "patch_examples_run27_49shared"

print("\n" + "#" * 70 + "\n#  RUN27: 49 tiles, SHARED, grid -> patch_examples_run27_49shared\n" + "#" * 70)
SC.run()
