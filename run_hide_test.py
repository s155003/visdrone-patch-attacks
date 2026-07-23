"""Pathak-matched hiding eval: train patch on val (100 imgs), SCORE on full
VisDrone TEST split (1610 imgs). Zero train/eval overlap. Separate runner —
run_hide_batch.py is untouched. One run, sequential on one GPU."""
import os, subprocess
PY, WORKDIR, SCRIPT = os.path.expanduser("~/attack_env/bin/python"), os.path.expanduser("~/attack-test"), "patch_attack_hide.py"
os.chdir(WORKDIR)
RUN = dict(
    RUN_NAME="h1_adv_640_nps_test", MODE="adversarial", IMG_SIZE="640", NPS_WEIGHT="0.01",
    TRAIN_IMAGES="datasets/VisDrone/images/val",     TRAIN_LABELS="datasets/VisDrone/labels/val",
    EVAL_IMAGES_DIR="datasets/VisDrone/images/test", EVAL_LABELS_DIR="datasets/VisDrone/labels/test",
    NUM_IMAGES="100", EVAL_IMAGES="0",
)
env = dict(os.environ); env.update(RUN)
print(f"\n{'='*70}\n=== {RUN['RUN_NAME']}\n{'='*70}", flush=True)
r = subprocess.run([PY, "-u", SCRIPT], env=env)
print(f"\n=== TEST RUN COMPLETE rc={r.returncode} ===")
