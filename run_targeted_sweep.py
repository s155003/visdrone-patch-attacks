"""
CAR -> VAN patch size sweep. Sequential, one GPU.

Train on VisDrone val, score on VisDrone test. Zero overlap; the model was
fine-tuned on train so it has seen neither.

  B0   no patch          -- clean detector. The floor. Should be ASR ~0.
  B1   normal patch 20%  -- reference attack (== S2, so it's also a data point)
  S1   10%
  S3   30%
  S4   40%

The one variable is AREA_FRAC. Everything else is fixed.

B0 and the gray/random baselines need no training and finish in minutes.
Run those FIRST — if B0's ASR isn't ~0, the metric is broken and nothing
after it means anything.
"""

import os, subprocess

PY      = os.path.expanduser("~/attack_env/bin/python")
WORKDIR = os.path.expanduser("~/attack-test")
SCRIPT  = "patch_attack_targeted.py"
os.chdir(WORKDIR)

RUNS = [
    # --- baselines, no training, minutes each ---
    dict(RUN_NAME="b0_nopatch",     MODE="none",        AREA_FRAC="0.20"),
    dict(RUN_NAME="b1_gray_20",     MODE="gray",        AREA_FRAC="0.20"),
    dict(RUN_NAME="b2_random_20",   MODE="random",      AREA_FRAC="0.20"),
    # --- the sweep ---
    dict(RUN_NAME="s1_adv_10",      MODE="adversarial", AREA_FRAC="0.10"),
    dict(RUN_NAME="s2_adv_20",      MODE="adversarial", AREA_FRAC="0.20"),
    dict(RUN_NAME="s3_adv_30",      MODE="adversarial", AREA_FRAC="0.30"),
    dict(RUN_NAME="s4_adv_40",      MODE="adversarial", AREA_FRAC="0.40"),
]

BASE = dict(
    IMG_SIZE="640",
    MIN_AREA_FRAC="0.001",     # raise this if cars are too small to be plausible
    EPOCHS="80",
    NUM_TRAIN="100",
    NUM_EVAL="0",              # all test images
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

print("\n=== SWEEP COMPLETE ===")
