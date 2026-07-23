"""Center-placement patch-size sweep. Sequential, one GPU, fresh subprocess per run.
Each run writes its own targeted_<RUN_NAME>.log. Everything identical to s2i
(center placement, SEED=0) except the swept AREA_FRAC (and EPOCHS for s2n)."""
import os, subprocess

PY      = os.path.expanduser("~/attack_env/bin/python")
WORKDIR = os.path.expanduser("~/attack-test")
SCRIPT  = "patch_attack_targeted.py"
os.chdir(WORKDIR)

RUNS = [
    dict(RUN_NAME="s2j_center_04",      AREA_FRAC="0.04", PLACE_MODE="center", SEED="0"),
    dict(RUN_NAME="s2k_center_07",      AREA_FRAC="0.07", PLACE_MODE="center", SEED="0"),
    dict(RUN_NAME="s2m_center_15",      AREA_FRAC="0.15", PLACE_MODE="center", SEED="0"),
    dict(RUN_NAME="s2l_center_03",      AREA_FRAC="0.03", PLACE_MODE="center", SEED="0"),
    dict(RUN_NAME="s2n_center_05_e160", AREA_FRAC="0.05", PLACE_MODE="center", SEED="0", EPOCHS="160"),
]

for cfg in RUNS:
    env = dict(os.environ); env.update(cfg)
    log = os.path.join(WORKDIR, f"targeted_{cfg['RUN_NAME']}.log")
    print(f"=== START {cfg['RUN_NAME']} -> {log}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([PY, "-u", SCRIPT], env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"=== DONE  {cfg['RUN_NAME']} rc={r.returncode}", flush=True)

print("=== CENTER SWEEP COMPLETE ===", flush=True)
