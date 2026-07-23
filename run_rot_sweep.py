"""Rotation sweep + clean control. Sequential, one GPU, fresh subprocess per run.
Everything identical to s2b (AREA_FRAC=0.20, center, SEED=0, EPOCHS=80) except the
swept ROT_DEG; b0_clean is MODE=none (no patch, no training) as an eval sanity check."""
import os, subprocess
PY, WORKDIR, SCRIPT = os.path.expanduser("~/attack_env/bin/python"), os.path.expanduser("~/attack-test"), "patch_attack_targeted.py"
os.chdir(WORKDIR)
RUNS = [
    dict(RUN_NAME="b0_clean", MODE="none",        AREA_FRAC="0.20", PLACE_MODE="center", SEED="0", EPOCHS="80"),
    dict(RUN_NAME="f_rot00",  MODE="adversarial", AREA_FRAC="0.20", PLACE_MODE="center", SEED="0", EPOCHS="80", ROT_DEG="0"),
    dict(RUN_NAME="f_rot45",  MODE="adversarial", AREA_FRAC="0.20", PLACE_MODE="center", SEED="0", EPOCHS="80", ROT_DEG="45"),
]
for cfg in RUNS:
    env = dict(os.environ); env.update(cfg)
    log = os.path.join(WORKDIR, f"targeted_{cfg['RUN_NAME']}.log")
    print(f"=== START {cfg['RUN_NAME']} -> {log}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([PY, "-u", SCRIPT], env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"=== DONE  {cfg['RUN_NAME']} rc={r.returncode}", flush=True)
print("=== ROT SWEEP COMPLETE ===", flush=True)
