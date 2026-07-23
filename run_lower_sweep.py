"""Lower-placement exploratory sweep. Sequential, one GPU, fresh subprocess per run.
All PLACE_MODE=lower, SEED=0, EPOCHS=80, ROT_DEG=20, LR=0.03 unless the run sets LR.
exp_* namespace only. exp_lower_20 (LOWER_TALL=0.60) already exists, not re-run."""
import os, subprocess
PY, WORKDIR, SCRIPT = os.path.expanduser("~/attack_env/bin/python"), os.path.expanduser("~/attack-test"), "patch_attack_targeted.py"
os.chdir(WORKDIR)
COMMON = dict(PLACE_MODE="lower", SEED="0", EPOCHS="80", ROT_DEG="20")
RUNS = [
    dict(RUN_NAME="exp_lower_20_t50",  AREA_FRAC="0.20", LOWER_TALL="0.50"),
    dict(RUN_NAME="exp_lower_20_t70",  AREA_FRAC="0.20", LOWER_TALL="0.70"),
    dict(RUN_NAME="exp_lower_20_t80",  AREA_FRAC="0.20", LOWER_TALL="0.80"),
    dict(RUN_NAME="exp_lower_10",      AREA_FRAC="0.10", LOWER_TALL="0.60"),
    dict(RUN_NAME="exp_lower_10_t50",  AREA_FRAC="0.10", LOWER_TALL="0.50"),
    dict(RUN_NAME="exp_lower_10_t70",  AREA_FRAC="0.10", LOWER_TALL="0.70"),
    dict(RUN_NAME="exp_lower_10_t90",  AREA_FRAC="0.10", LOWER_TALL="0.90"),
    dict(RUN_NAME="exp_lower_07",      AREA_FRAC="0.07", LOWER_TALL="0.60"),
    dict(RUN_NAME="exp_lower_05",      AREA_FRAC="0.05", LOWER_TALL="0.60"),
    dict(RUN_NAME="exp_lower_05_lr12", AREA_FRAC="0.05", LOWER_TALL="0.60", LR="0.12"),
    dict(RUN_NAME="exp_lower_03_lr30", AREA_FRAC="0.03", LOWER_TALL="0.60", LR="0.30"),
]
for cfg in RUNS:
    env = dict(os.environ); env.update(COMMON); env.update(cfg)
    log = os.path.join(WORKDIR, f"targeted_{cfg['RUN_NAME']}.log")
    print(f"=== START {cfg['RUN_NAME']} -> {log}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([PY, "-u", SCRIPT], env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"=== DONE  {cfg['RUN_NAME']} rc={r.returncode}", flush=True)
print("=== LOWER SWEEP COMPLETE ===", flush=True)
