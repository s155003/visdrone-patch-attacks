"""
SHAPE / OPACITY study — 6 scenarios, sequential, one GPU.

  B0   no patch                       -- must score 0.000 or the metric is broken
  S1   square   opacity 1.0   10%     -- reference
  S2   circle   opacity 1.0   10%     -- shape
  S3   ellipse  opacity 1.0   10%     -- shape (box-matched)
  S4   triangle opacity 1.0   10%     -- shape
  S5   square   opacity 0.5   10%     -- opacity

Every row changes exactly ONE thing from S1.

All shapes are AREA-MATCHED: each shape's canvas fill fraction is divided out,
so a circle and a square cover the same pixel count. Verified numerically —
all four land on exactly 320 px^2 on an 80x40 box at frac=0.10.

At 10% the overflow cap never fires (checked on 2:1 and near-square boxes), so
NO_OVERFLOW is inert here and the shape comparison is uncontaminated by
capping. It stays on as a guarantee, not a correction.

Each run: trains on 100 val images, scores on ALL test images, writes its own
patch_examples_<RUN_NAME>/ with results.json, universal_patch.png, and a
.boxes.json per image.

B0 runs FIRST. If it isn't 0.000, stop — nothing after it means anything.
"""

import os, subprocess

PY      = os.path.expanduser("~/attack_env/bin/python")
WORKDIR = os.path.expanduser("~/attack-test")
SCRIPT  = "patch_attack_shapes.py"
os.chdir(WORKDIR)

RUNS = [
    dict(RUN_NAME="sh_b0_nopatch",  MODE="none",        SHAPE="square",   OPACITY="1.0"),
    dict(RUN_NAME="sh_s1_square",   MODE="adversarial", SHAPE="square",   OPACITY="1.0"),
    dict(RUN_NAME="sh_s2_circle",   MODE="adversarial", SHAPE="circle",   OPACITY="1.0"),
    dict(RUN_NAME="sh_s3_ellipse",  MODE="adversarial", SHAPE="ellipse",  OPACITY="1.0"),
    dict(RUN_NAME="sh_s4_triangle", MODE="adversarial", SHAPE="triangle", OPACITY="1.0"),
    dict(RUN_NAME="sh_s5_opacity",  MODE="adversarial", SHAPE="square",   OPACITY="0.5"),
]

BASE = dict(
    AREA_FRAC="0.10",
    NO_OVERFLOW="1",
    IMG_SIZE="640",
    ROT_DEG="20",
    LR="0.03",
    EPOCHS="80",
    SEED="0",
    NUM_TRAIN="100",
    NUM_EVAL="0",              # 0 = ALL test images
    MIN_AREA_FRAC="0.001",
    TRAIN_IMAGES="datasets/VisDrone/images/val",
    TRAIN_LABELS="datasets/VisDrone/labels/val",
    EVAL_IMAGES="datasets/VisDrone/images/test",
    EVAL_LABELS="datasets/VisDrone/labels/test",
)

for cfg in RUNS:
    env = dict(os.environ); env.update(BASE); env.update(cfg)
    print(f"\n{'='*70}\n=== {cfg['RUN_NAME']}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u", SCRIPT], env=env)
    if r.returncode != 0:
        print(f"!!! {cfg['RUN_NAME']} FAILED rc={r.returncode} — continuing",
              flush=True)

print("\n=== SHAPE STUDY COMPLETE ===")
