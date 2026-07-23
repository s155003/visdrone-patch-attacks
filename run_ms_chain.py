"""
MULTISCALE chain - built on R63's winner (multiscale x3 = 27.5%, the only run to break
the ~13% off-glass ceiling). All runs use a COVERAGE FLOOR so every car gets painted
(fixes the R67-R70 bug where some cars got no patch at all). All on-paint/off-glass.
car->van, 100 images, GPU. ORDERED MOST-PROMISING FIRST.

R81 ms x3 + union mask + floor   <- reproduce/improve R63 with the FIXED ellipse mask
R82 ms x3 + 49 tiles + floor     <- stack the two winners (frequency + tile capacity)
R83 ms x4 scales                 <- more frequency diversity
R84 ms x3, gentler ratio (1.5)   <- finer frequency spacing
R85 ms x5 scales                 <- push scales further
R86 ms x3 + floor 0.55           <- more guaranteed coverage
R87 ms x3 + adaptive mask + floor<- per-box adaptive with floor
"""
import importlib, patch_attack_ms as MS

VARIANTS = [
    # most promising first
    ("run84_ms3_ratio15",      dict(USE_MS=True, MS_SCALES=3, MS_RATIO=1.5, COMBINE_METHOD="union",    COVER_FLOOR=0.40)),
    ("run85_ms5_union",        dict(USE_MS=True, MS_SCALES=5, MS_RATIO=2.0, COMBINE_METHOD="union",    COVER_FLOOR=0.40)),
    ("run86_ms3_floor55",      dict(USE_MS=True, MS_SCALES=3, MS_RATIO=2.0, COMBINE_METHOD="union",    COVER_FLOOR=0.55)),
    ("run87_ms3_adaptive",     dict(USE_MS=True, MS_SCALES=3, MS_RATIO=2.0, COMBINE_METHOD="adaptive", COVER_FLOOR=0.40)),
]
for tag, cfg in VARIANTS:
    importlib.reload(MS)
    MS.USE_PAINT=True; MS.USE_MS=False; MS.USE_PAINTTILES=False
    MS.PAINT_LO_PCT=25; MS.COMBINE_METHOD="union"; MS.COVER_FLOOR=0.40
    MS.MS_SCALES=3; MS.MS_RATIO=2.0; MS.PAINT_TILES=49
    for k,v in cfg.items(): setattr(MS,k,v)
    MS.SAVE_DIR=f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70)
    MS.run()
