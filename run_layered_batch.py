"""
Layered paint attack: stack N adversarial layers on the same on-car paint area to pile
up more adversarial capacity (more params optimizing the same pixels). Tests whether
layering breaks past the ~13% off-glass ceiling. Uses the best paint mask (union).
Also throws in stronger training (more epochs + higher margin).
car->van, 100 images, GPU.
  R62 additive x3   R63 multiscale x3   R64 blend x3   R65 additive x5 (more layers)
  R66 additive x3 + stronger (150 ep, margin 8)
"""
import importlib, patch_attack_layered as LY

VARIANTS = [
    ("run62_additive3",   dict(LAYER_MODE="additive",   N_LAYERS=3)),
    ("run63_multiscale3", dict(LAYER_MODE="multiscale", N_LAYERS=3)),
    ("run64_blend3",      dict(LAYER_MODE="blend",      N_LAYERS=3)),
    ("run65_additive5",   dict(LAYER_MODE="additive",   N_LAYERS=5)),
    ("run66_additive3_strong", dict(LAYER_MODE="additive", N_LAYERS=3, EPOCHS=150, MARGIN=8.0)),
]
for tag, cfg in VARIANTS:
    importlib.reload(LY)
    LY.USE_PAINT = True
    LY.COMBINE_METHOD = "union"   # largest usable paint area (best from R58-61)
    LY.PAINT_LO_PCT = 25          # loose = most paint (best from R53-55)
    LY.EPOCHS = 80; LY.MARGIN = 5.0
    for k,v in cfg.items(): setattr(LY, k, v)
    LY.SAVE_DIR = f"patch_examples_{tag}"
    print("\n"+"#"*70+f"\n#  {tag.upper()}: {cfg}\n"+"#"*70)
    LY.run()
