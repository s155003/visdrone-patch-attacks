"""Render b0_clean (no-patch) baseline figures for specific stems.

b0_clean has no full figure set, so when you build a per-image comparison folder
you need its baseline panel rendered on demand. Format matches
make_figures_targeted.py exactly (same header / colours / caption / legend), so
baseline panels sit alongside the attack panels without looking different.

b0_clean is mode="none": patched == clean, so every eligible car scores 'car'
(green) and flip/lost are 0 -- that is the correct control.

usage:  PYTHONPATH=~/attack-test python tools_session/render_baseline.py <stem> [<stem> ...]
"""
import os, sys, json
import numpy as np, cv2
import make_figures_targeted as M
import patch_attack_targeted as T

IMG_DIR = "datasets/VisDrone/images/test"
LBL_DIR = "datasets/VisDrone/labels/test"
RUN = "patch_examples_b0_clean"

STEMS = sys.argv[1:] or ["0000006_00159_d_0000001"]

rj = json.load(open(os.path.join(RUN, "results.json")))
img_size = int(rj.get("img_size", 640))
out_dir = os.path.join(RUN, "annotated")
os.makedirs(out_dir, exist_ok=True)

for stem in STEMS:
    bj = os.path.join(RUN, stem + ".boxes.json")
    if not os.path.exists(bj):
        print(f"[skip] no detections for {stem}"); continue
    d = json.load(open(bj))
    orig = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
    if orig is None:
        print(f"[skip] no image for {stem}"); continue
    img = cv2.resize(orig, (img_size, img_size))
    boxes, _ = T.load_car_boxes(os.path.join(LBL_DIR, stem + ".txt"))
    pbgr = img.copy()                       # mode=none: patched == clean

    left = img.copy(); ncar = 0
    for det in d["clean"]:
        if int(det[0]) != T.SRC_IDX:
            continue
        cv2.rectangle(left, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), (0, 255, 0), 2)
        ncar += 1
    M.header(left, f"CLEAN  ({ncar} cars detected)")

    nf = nc = nl = na = 0
    for (c, xc, yc, w, h) in boxes:
        bpx = ((xc-w/2)*img_size, (yc-h/2)*img_size, (xc+w/2)*img_size, (yc+h/2)*img_size)
        o = M.outcome(bpx, d["clean"], d["patched"])
        if o is None:
            continue
        cv2.rectangle(pbgr, (int(bpx[0]), int(bpx[1])), (int(bpx[2]), int(bpx[3])), M.COL[o], 2)
        nf += o == "flip"; nc += o == "car"; nl += o == "lost"; na += o == "ambig"
    M.header(pbgr, f"PATCHED  flip={nf}  car={nc}  lost={nl}  ambig={na}")

    combo = np.hstack([left, pbgr])
    cap = np.zeros((86, combo.shape[1], 3), np.uint8)
    cv2.putText(cap, "FULL EVAL SET  (CLEAN BASELINE, no patch)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (200, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(cap, f"{os.path.basename(RUN)}   |   no patch   |   strict ASR 0.000",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    x = 10
    for col, txt in [((0, 255, 0), "still car (fail)"), ((255, 0, 0), "-> van (ASR success)"),
                     ((0, 0, 255), "vanished (fail)"), ((0, 255, 255), "ambiguous (excl)")]:
        cv2.rectangle(cap, (x, 62), (x + 16, 76), col, -1)
        cv2.putText(cap, txt, (x + 22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += 22 + len(txt) * 9 + 20
    combo = np.vstack([combo, cap])

    outp = os.path.join(out_dir, f"{stem}_full.jpg")
    cv2.imwrite(outp, combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[baseline] {outp}  (cars={ncar}, still={nc})")
