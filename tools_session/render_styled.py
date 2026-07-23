"""Presentation-styled before/after figures (white/red theme, PPT-friendly).

Legend uses the paper's metric names, framed as car->X transitions so the red
(patch destroyed a real detection) vs gray (there was no clean car to attack)
distinction is unambiguous:

    ASR      car detected -> now reads VAN
    RAcc     car detected -> still a car
    VR       car detected -> erased by the patch
    excluded no clean car: already van, or never detected

Outcome boxes are drawn from the SAVED boxes.json (authoritative, so counts are
exact); the patch pixels are reconstructed only for display. Handles both the
targeted engine and the shapes/roof engine.

usage:  PYTHONPATH=~/attack-test python tools_session/render_styled.py all
        PYTHONPATH=~/attack-test python tools_session/render_styled.py center_square_20pct_rot20
"""
import os, sys, json, random
import numpy as np, cv2, torch
import patch_attack_targeted as T
import patch_attack_shapes as S

S.DEVICE = "cpu"
IMG_DIR = "datasets/VisDrone/images/test"
LBL_DIR = "datasets/VisDrone/labels/test"
STEM = "0000006_00159_d_0000001"          # change to restyle a different scene
OUT_DIR = "final_picks/" + STEM
os.makedirs(OUT_DIR, exist_ok=True)

# run_key -> (run_dir, engine 't'=targeted / 's'=shapes, descriptor)
RUNS = {
    "baseline_clean_nopatch":   ("patch_examples_b0_clean",         "t", "clean baseline  |  no patch"),
    "center_square_20pct_rot00":("patch_examples_f_rot00",          "t", "center  |  square  |  20%  |  0 deg"),
    "center_square_20pct_rot20":("patch_examples_s2b_adv_20_hinge", "t", "center  |  square  |  20%  |  +/-20 deg"),
    "center_square_20pct_rot45":("patch_examples_f_rot45",          "t", "center  |  square  |  20%  |  +/-45 deg"),
    "center_square_10pct_rot20":("patch_examples_s2h_center_10",    "t", "center  |  square  |  10%  |  +/-20 deg"),
    "offset_square_20pct_rot20":("patch_examples_s2d_offset_20",    "t", "offset  |  square  |  20%  |  +/-20 deg"),
    "roof_square_11pct_rot20":  ("patch_examples_sh_roof_square",   "s", "roof  |  square  |  ~11%  |  +/-20 deg"),
    "roof_circle_11pct_rot20":  ("patch_examples_sh_roof_circle",   "s", "roof  |  circle  |  ~11%  |  +/-20 deg"),
}

# ---- palette (BGR) ----
WHITE, INK   = (255, 255, 255), (54, 46, 42)
REDACC       = (48, 44, 200)          # presentation red accent
C_VAN        = (196, 92, 28)          # blue  -> van        (ASR)
C_STILL      = (64, 158, 58)          # green -> still car  (RAcc)
C_VANISH     = (44, 44, 200)          # red   -> erased by patch (VR)
C_EXCL       = (150, 150, 150)        # gray  -> excluded
C_EXCLT      = (110, 110, 110)        # darker gray for text on white
COL = {"flip": C_VAN, "car": C_STILL, "lost": C_VANISH, "excl": C_EXCL}
LEGEND = [(C_VAN,    "ASR",      "car detected  ->  now reads VAN"),
          (C_STILL,  "RAcc",     "car detected  ->  still a car"),
          (C_VANISH, "VR",       "car detected  ->  erased by the patch"),
          (C_EXCL,   "excluded", "no clean car: already van, or never detected")]
FONT = cv2.FONT_HERSHEY_DUPLEX


def outcome(bpx, clean, patched):
    """'excl' covers BOTH already-van-in-clean (ambiguous) AND never-detected."""
    car_c = T.detected_as(clean, T.SRC_IDX, bpx)
    van_c = T.detected_as(clean, T.TGT_IDX, bpx)
    if (not car_c) or van_c:
        return "excl"
    if T.detected_as(patched, T.TGT_IDX, bpx):
        return "flip"
    if not T.detected_as(patched, T.SRC_IDX, bpx):
        return "lost"
    return "car"


def bar(img, y0, y1, txts):
    cv2.rectangle(img, (0, y0), (img.shape[1], y1), WHITE, -1)
    cv2.rectangle(img, (0, y0), (img.shape[1], y0 + 3), REDACC, -1)
    for txt, x, col, sc, th in txts:
        cv2.putText(img, txt, (x, y0 + 27), FONT, sc, col, th, cv2.LINE_AA)


def reconstruct(run_dir, engine, img):
    rj = json.load(open(os.path.join(run_dir, "results.json")))
    if rj.get("mode") == "none":
        return img.copy()
    patch = torch.load(os.path.join(run_dir, "universal_patch.pt"), map_location="cpu").float()
    x0 = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    if engine == "t":
        T.PLACE_MODE = rj.get("place_mode", "center")
        T.LOWER_TALL = float(rj.get("lower_tall", 0.60)); T.LOWER_WIDE = float(rj.get("lower_wide", 0.75))
        T.ROOF_LEN = float(rj.get("roof_len", 0.30)); T.ROOF_WID = float(rj.get("roof_wid", 0.70))
        T.ROT_DEG = float(rj.get("rot_deg", 20.0))
        boxes, _ = T.load_car_boxes(os.path.join(LBL_DIR, STEM + ".txt"))
        with torch.no_grad():
            pimg = T.apply_patch(x0, patch, boxes, float(rj.get("area_frac", 0.2)),
                                 np.random.RandomState(T.SEED)).clamp(0, 1)
    else:
        S.SHAPE = rj.get("shape", "square"); S.OPACITY = float(rj.get("opacity", 1.0))
        S.AREA_FRAC = float(rj.get("area_frac", 0.2)); S.NO_OVERFLOW = int(rj.get("no_overflow", 1))
        S.ROT_DEG = float(rj.get("rot_deg", 20.0)); S.PLACE_MODE = rj.get("place_mode", "center")
        S.ROOF_LEN = float(rj.get("roof_len", 0.30)); S.ROOF_WID = float(rj.get("roof_wid", 0.70))
        S.PATCH_PX = 64
        r = S.load_car_boxes(os.path.join(LBL_DIR, STEM + ".txt"))
        boxes = r[0] if isinstance(r, tuple) else r
        with torch.no_grad():
            pimg = S.apply_patch(x0, patch, boxes, random.Random(S.SEED)).clamp(0, 1)
    return (pimg[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()


def render(run_key):
    run_dir, engine, desc = RUNS[run_key]
    rj = json.load(open(os.path.join(run_dir, "results.json")))
    sz = int(rj.get("img_size", 640))
    d = json.load(open(os.path.join(run_dir, STEM + ".boxes.json")))
    img = cv2.resize(cv2.imread(os.path.join(IMG_DIR, STEM + ".jpg")), (sz, sz))
    pbgr = reconstruct(run_dir, engine, img)
    boxes, _ = T.load_car_boxes(os.path.join(LBL_DIR, STEM + ".txt"))

    left = img.copy(); ncar = 0
    for det in d["clean"]:
        if int(det[0]) == T.SRC_IDX:
            cv2.rectangle(left, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), C_STILL, 2); ncar += 1

    nf = nc = nl = na = 0
    for (c, xc, yc, w, h) in boxes:
        bpx = ((xc-w/2)*sz, (yc-h/2)*sz, (xc+w/2)*sz, (yc+h/2)*sz)
        o = outcome(bpx, d["clean"], d["patched"])
        cv2.rectangle(pbgr, (int(bpx[0]), int(bpx[1])), (int(bpx[2]), int(bpx[3])), COL[o], 2)
        nf += o == "flip"; nc += o == "car"; nl += o == "lost"; na += o == "excl"

    gap = 8
    panels = np.hstack([left, np.full((sz, gap, 3), 255, np.uint8), pbgr])
    W = panels.shape[1]
    body = np.vstack([np.full((38, W, 3), 255, np.uint8), panels])
    bar(body, 0, 38, [(f"CLEAN   {ncar} cars", 10, INK, 0.66, 1),
                      ("PATCHED", sz + gap + 10, INK, 0.66, 1)])
    # per-image counts, each colour-coded to its outcome
    x = sz + gap + 150
    for txt, col in [(f"-> van {nf}", C_VAN), (f"vanished {nl}", C_VANISH),
                     (f"still {nc}", C_STILL), (f"excl {na}", C_EXCLT)]:
        cv2.putText(body, txt, (x, 27), FONT, 0.58, col, 1, cv2.LINE_AA)
        x += cv2.getTextSize(txt, FONT, 0.58, 1)[0][0] + 26

    # 2x2 legend; no run descriptor (captions live in the slide deck)
    caph = 94
    cap = np.full((caph, W, 3), 255, np.uint8)
    cv2.rectangle(cap, (0, 0), (W, 3), REDACC, -1)
    col_x, row_y = [16, W // 2 + 16], [10, 52]
    for i, (col, metric, meaning) in enumerate(LEGEND):
        cx, ry = col_x[i % 2], row_y[i // 2]
        cv2.rectangle(cap, (cx, ry), (cx + 20, ry + 20), col, -1)
        cv2.rectangle(cap, (cx, ry), (cx + 20, ry + 20), INK, 1)
        cv2.putText(cap, metric,  (cx + 30, ry + 16), FONT, 0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(cap, meaning, (cx + 30, ry + 36), FONT, 0.47, INK, 1, cv2.LINE_AA)

    base = os.path.basename(run_dir).replace("patch_examples_", "")
    p = os.path.join(OUT_DIR, base + ".jpg")
    cv2.imwrite(p, np.vstack([body, cap]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[styled] {p}  (van {nf}, vanished {nl}, still {nc}, excl {na})")
    return p


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "center_square_20pct_rot20"
    for k in (list(RUNS) if arg == "all" else [arg]):
        render(k)
