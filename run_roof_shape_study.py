"""Roof-placed SHAPE study: square/circle/ellipse/triangle + opacity(0.5), all
constrained to the central roof rectangle (PLACE_MODE=roof, ROOF_LEN=0.30) -> provably
off glass. Sequential, one GPU, fresh subprocess each. 160 epochs (roof patches small)."""
import os, subprocess
PY, WORKDIR, SCRIPT = os.path.expanduser("~/attack_env/bin/python"), os.path.expanduser("~/attack-test"), "patch_attack_shapes.py"
os.chdir(WORKDIR)
BASE = dict(PLACE_MODE="roof", ROOF_LEN="0.30", ROOF_WID="0.70", AREA_FRAC="0.20",
            SEED="0", EPOCHS="160", ROT_DEG="20")
RUNS = [
    dict(RUN_NAME="sh_roof_square",   MODE="adversarial", SHAPE="square",   OPACITY="1.0"),
    dict(RUN_NAME="sh_roof_circle",   MODE="adversarial", SHAPE="circle",   OPACITY="1.0"),
    dict(RUN_NAME="sh_roof_ellipse",  MODE="adversarial", SHAPE="ellipse",  OPACITY="1.0"),
    dict(RUN_NAME="sh_roof_triangle", MODE="adversarial", SHAPE="triangle", OPACITY="1.0"),
    dict(RUN_NAME="sh_roof_opacity",  MODE="adversarial", SHAPE="square",   OPACITY="0.5"),
]
for cfg in RUNS:
    env = dict(os.environ); env.update(BASE); env.update(cfg)
    log = os.path.join(WORKDIR, f"roofshape_{cfg['RUN_NAME']}.log")
    print(f"=== START {cfg['RUN_NAME']} -> {log}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([PY, "-u", SCRIPT], env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"=== DONE  {cfg['RUN_NAME']} rc={r.returncode}", flush=True)
print("=== ROOF SHAPE STUDY COMPLETE ===", flush=True)
