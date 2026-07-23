"""Re-run ONLY the scale-up variants R21-R23 on GPU (R20 already done: 0.190, kept).
Mirrors run_ceiling_matrix.py's scale-up loop exactly, but skips the per-image R20."""
import importlib, patch_attack_scaleup as SU

SCALEUPS = [
    ("run21_occlusion", 0.90),   # bigger/occlusion
    ("run22_scaleup07", 0.70),   # honest scale-up 0.7
    ("run23_scaleup045", 0.45),  # honest scale-up 0.45
]
for tag, frac in SCALEUPS:
    importlib.reload(SU)             # reset module state (same as orchestrator)
    SU.SAVE_DIR   = f"patch_examples_{tag}"
    SU.FIXED_FRAC = frac
    print("\n" + "#" * 70 + f"\n#  {tag.upper()}: fixed frac={frac}, 100 img, 80 ep\n" + "#" * 70)
    SU.run()
