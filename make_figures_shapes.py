"""Shape/opacity study figures. Reconstructs the patched image with
patch_attack_shapes.py's OWN apply_patch (shape mask + opacity blend) — NOT the
square compositing — so circles/triangles/opacity draw truthfully. Four-outcome
coloring, strict_asr caption read from each run's results.json. CPU only, no model.

Usage: python make_figures_shapes.py <run_dir> [n] [full|random]
"""
import os, sys, json, glob, random
import numpy as np
import cv2
import torch
import patch_attack_shapes as S          # import-safe; reuse its apply_patch

S.DEVICE = "cpu"                          # force CPU (shape_mask builds on DEVICE)
IMG_DIR = "datasets/VisDrone/images/test"
LBL_DIR = "datasets/VisDrone/labels/test"
SAMPLE_SEED = 1234
UNDER = {"sh_s3_ellipse", "sh_s4_triangle", "sh_s5_opacity",
         "sh_roof_ellipse", "sh_roof_triangle", "sh_roof_opacity"}   # under-converged runs (hinge Δ > -300)
COL = {"car": (0, 255, 0), "flip": (255, 0, 0), "lost": (0, 0, 255), "ambig": (0, 255, 255)}
LEGEND = [((0, 255, 0), "still car (fail)"), ((255, 0, 0), "-> van (ASR success)"),
          ((0, 0, 255), "vanished (fail)"), ((0, 255, 255), "ambiguous (excl)")]


def outcome(bpx, clean, patched):
    if not S.detected_as(clean, S.SRC_IDX, bpx):
        return None
    if S.detected_as(clean, S.TGT_IDX, bpx):
        return "ambig"
    if S.detected_as(patched, S.TGT_IDX, bpx):
        return "flip"
    if not S.detected_as(patched, S.SRC_IDX, bpx):
        return "lost"
    return "car"


def header(img, title):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def load_boxes(stem):
    r = S.load_car_boxes(os.path.join(LBL_DIR, stem + ".txt"))
    return r[0] if isinstance(r, tuple) else r


def main(run_dir, n=10, mode="full"):
    run_dir = run_dir.rstrip("/")
    base = os.path.basename(run_dir).replace("patch_examples_", "")
    rj = json.load(open(os.path.join(run_dir, "results.json")))
    # restore this run's shape/opacity config so apply_patch composites correctly
    S.SHAPE = rj.get("shape", "square")
    S.OPACITY = float(rj.get("opacity", 1.0))
    S.AREA_FRAC = float(rj.get("area_frac", 0.10))
    S.NO_OVERFLOW = int(rj.get("no_overflow", 1))
    S.ROT_DEG = float(rj.get("rot_deg", 20.0))
    S.PLACE_MODE = rj.get("place_mode", "center")     # restore roof placement for reconstruction
    S.ROOF_LEN = float(rj.get("roof_len", 0.30))
    S.ROOF_WID = float(rj.get("roof_wid", 0.70))
    S.PATCH_PX = 64
    if not hasattr(S, "_COV") or S._COV is None:
        S._COV = {}
    S._COV.update(cap=0, frac_sum=0.0, n=0)
    run_mode = rj.get("mode", "adversarial")
    strict = float(rj.get("strict_asr", 0.0))
    img_size = int(rj.get("img_size", S.IMG_SIZE)) if "img_size" in rj else S.IMG_SIZE
    under = base in UNDER
    patch = torch.load(os.path.join(run_dir, "universal_patch.pt"), map_location="cpu").float()

    # score every image (four outcomes per GT car)
    scored = []
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf)); boxes = load_boxes(stem)
        outs, flips = [], 0
        for (c, xc, yc, w, h) in boxes:
            bpx = ((xc-w/2)*img_size, (yc-h/2)*img_size, (xc+w/2)*img_size, (yc+h/2)*img_size)
            o = outcome(bpx, d["clean"], d["patched"]); outs.append((bpx, o)); flips += (o == "flip")
        scored.append((flips, stem, d, outs))
    if not scored:
        print(f"[fig] no .boxes.json in {run_dir}"); return

    if mode == "random":
        by = sorted(scored, key=lambda t: t[1])
        rr = np.random.RandomState(SAMPLE_SEED)
        idx = sorted(rr.choice(len(by), min(n, len(by)), replace=False).tolist())
        sel = [by[i] for i in idx]; suffix = "_random"
        tag_sel = f"RANDOM SAMPLE (seed {SAMPLE_SEED})"
        print(f"[fig] {base}: {len(scored)} imgs | RANDOM {len(sel)}")
    else:
        sel = scored; suffix = "_full"; tag_sel = "FULL EVAL SET"
        print(f"[fig] {base}: {len(scored)} imgs | FULL {len(sel)} -> _full.jpg q92")

    out_dir = os.path.join(run_dir, "annotated"); os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(S.SEED)
    for flips, stem, d, outs in sel:
        ip = os.path.join(IMG_DIR, stem + ".jpg"); orig = cv2.imread(ip)
        if orig is None:
            continue
        img = cv2.resize(orig, (img_size, img_size))
        if run_mode == "none":
            pbgr = img.copy()                       # no patch was applied
        else:
            x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
                  .float().unsqueeze(0) / 255.0)
            boxes = load_boxes(stem)
            with torch.no_grad():
                pimg = S.apply_patch(x0, patch, boxes, rng).clamp(0, 1)   # SHAPE + OPACITY
            pbgr = (pimg[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()

        left = img.copy(); ncar = 0
        for det in d["clean"]:
            if int(det[0]) != S.SRC_IDX:
                continue
            cv2.rectangle(left, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), (0, 255, 0), 2); ncar += 1
        header(left, f"CLEAN  ({ncar} cars detected)")

        nf = nc = nl = na = 0
        for bpx, o in outs:
            if o is None:
                continue
            cv2.rectangle(pbgr, (int(bpx[0]), int(bpx[1])), (int(bpx[2]), int(bpx[3])), COL[o], 2)
            nf += (o == "flip"); nc += (o == "car"); nl += (o == "lost"); na += (o == "ambig")
        header(pbgr, f"PATCHED  flip={nf} car={nc} lost={nl} ambig={na}")

        combo = np.hstack([left, pbgr])
        lines = []
        if under:
            lines.append(("UNDER-CONVERGED -- patch did not train", (0, 0, 255), 0.66, 2))
        lines.append((tag_sel, ((0, 220, 255) if mode != "random" else (200, 255, 200)), 0.60, 2))
        area_lbl = (f"roof-fit (~11% of bbox, req {int(round(S.AREA_FRAC*100))}%)"
                    if S.PLACE_MODE == "roof" else f"{int(round(S.AREA_FRAC*100))}% of bbox")
        lines.append((f"{base}  |  {S.SHAPE} op{S.OPACITY:.1f}  |  {area_lbl}"
                      f"  |  strict ASR {strict:.3f}", (255, 255, 255), 0.58, 1))
        caph = 22 * len(lines) + 26
        cap = np.zeros((caph, combo.shape[1], 3), np.uint8)
        y = 20
        for txt, col, sc, th in lines:
            cv2.putText(cap, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA); y += 22
        x = 10
        for col, txt in LEGEND:
            cv2.rectangle(cap, (x, y - 12), (x + 16, y + 2), col, -1)
            cv2.putText(cap, txt, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            x += 22 + len(txt) * 9 + 20
        combo = np.vstack([combo, cap])

        if mode == "full":
            cv2.imwrite(os.path.join(out_dir, f"{stem}_full.jpg"), combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_random.png"), combo)
    print(f"[fig] done -> {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python make_figures_shapes.py <run_dir> [n] [full|random]"); sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10,
         sys.argv[3] if len(sys.argv) > 3 else "full")
