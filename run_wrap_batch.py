"""
Car-wrap attack: patch covers ONLY the wrappable body panels (hood + roof + trunk),
skipping the front and rear windshields, never off the car. Like an auto-shop vinyl wrap.
  R51 (wrap tiles):      49 independent tiles distributed across hood/roof/trunk
  R52 (wrap contiguous): one contiguous patch per zone (hood, roof, trunk)
All independent patterns, car→van, 100 images, GPU.
"""
import importlib, patch_attack_wrap as WR

VARIANTS = [
    ("run51_wrap_tiles",      dict(WRAP_MODE="tiles",      WRAP_TILES=49)),
    ("run52_wrap_contiguous", dict(WRAP_MODE="contiguous")),
]
for tag, cfg in VARIANTS:
    importlib.reload(WR)
    WR.USE_WRAP = True
    WR.WRAP_ZONES = [(0.00,0.22),(0.36,0.60),(0.74,1.00)]  # hood, roof, trunk
    # independent patterns: give it 49 tiles worth of capacity
    for k,v in cfg.items(): setattr(WR, k, v)
    WR.SAVE_DIR = f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}  (hood+roof+trunk, skip windshields)\n"+"#"*70)
    WR.run()
