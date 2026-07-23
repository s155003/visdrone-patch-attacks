"""
Hiding-attack chain. Runs sequentially in ONE process on ONE GPU.

Ordered so the UNTRAINED baselines land first — they need no optimization and
validate compositing / area-sizing / the ASR metric against Pathak's Table I
(gray 53%, random 55%) BEFORE any training run is trusted.

  H0a  gray patch   @640   -- no training, ~minutes.  Target ~0.53
  H0b  random patch @640   -- no training, ~minutes.  Target ~0.55
  H1   adversarial  @640, NPS ON   -- the Table I comparison number (~0.84?)
  H2   adversarial  @640, NPS OFF  -- printability ablation
  H3   adversarial  @1280, NPS ON  -- comparability with the car->van runs

If H0a comes back near 0.05 instead of 0.53, STOP: the pipeline is wrong and
no training run will mean anything. That is the entire reason it runs first.
"""

import os, sys, subprocess

PY      = os.path.expanduser("~/attack_env/bin/python")
WORKDIR = os.path.expanduser("~/attack-test")
SCRIPT  = "patch_attack_hide.py"          # relative — script uses relative
                                          # data/model paths, same as
                                          # patch_attack_ms.py. Must cd first.
os.chdir(WORKDIR)

RUNS = [
    dict(RUN_NAME="h0a_gray_640",    MODE="gray",        IMG_SIZE="640",  NPS_WEIGHT="0.0"),
    dict(RUN_NAME="h0b_random_640",  MODE="random",      IMG_SIZE="640",  NPS_WEIGHT="0.0"),
    dict(RUN_NAME="h1_adv_640_nps",  MODE="adversarial", IMG_SIZE="640",  NPS_WEIGHT="0.01"),
    dict(RUN_NAME="h2_adv_640_nonps",MODE="adversarial", IMG_SIZE="640",  NPS_WEIGHT="0.0"),
    dict(RUN_NAME="h3_adv_1280_nps", MODE="adversarial", IMG_SIZE="1280", NPS_WEIGHT="0.01"),
]

for cfg in RUNS:
    env = dict(os.environ)
    env.update(cfg)
    print(f"\n{'='*70}\n=== {cfg['RUN_NAME']}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u", SCRIPT], env=env)
    if r.returncode != 0:
        print(f"!!! {cfg['RUN_NAME']} FAILED rc={r.returncode} — continuing",
              flush=True)

print("\n=== CHAIN COMPLETE ===")
