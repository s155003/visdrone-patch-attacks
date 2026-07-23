"""
Run all 4 ceiling/scale-up experiments back-to-back:
  Run 20: PER-IMAGE (separate patch per car) - absolute ceiling
  Run 21: bigger/occlusion patch, FIXED 0.9, 100 img, 80 ep
  Run 22: honest scale-up, FIXED 0.7, 100 img, 80 ep
  Run 23: honest scale-up, adaptive ~0.45, 100 img, 80 ep
"""
# --- Run 20: per-image (separate module) ---
print("\n" + "#"*70 + "\n#  RUN 20: PER-IMAGE (absolute ceiling)\n" + "#"*70)
import patch_attack_perimage as PI
PI.SAVE_DIR = "patch_examples_run20_perimage"
PI.run()

# --- Runs 21-23: scale-ups (shared module, different FIXED_FRAC) ---
import importlib, patch_attack_scaleup as SU
SCALEUPS = [
    ("run21_occlusion", 0.90),   # bigger/occlusion
    ("run22_scaleup07", 0.70),   # honest scale-up 0.7
    ("run23_scaleup045", 0.45),  # honest scale-up 0.45 (fixed for clean comparison)
]
for tag, frac in SCALEUPS:
    importlib.reload(SU)   # reset module state
    SU.SAVE_DIR   = f"patch_examples_{tag}"
    SU.FIXED_FRAC = frac
    print("\n" + "#"*70 + f"\n#  {tag.upper()}: fixed frac={frac}, 100 img, 80 ep\n" + "#"*70)
    SU.run()
