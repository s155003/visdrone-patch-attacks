"""make_figures.py — before/after poster comparisons for a hiding run.

READ-ONLY w.r.t. the detector: it reuses the SAVED <stem>.boxes.json detections
and NEVER runs YOLO, so it never loads the model and never touches CUDA. It
reuses patch_attack_hide's functions by import (no rewrites). The only tensor
work is reconstructing the patched image on the CPU via apply_patch.

Usage:  ~/attack_env/bin/python make_figures.py <run_dir> [n_examples]
        e.g.  make_figures.py patch_examples_h1_adv_640_nps 10

Caveat (cosmetic only): the eval saved detections, not the patched pixels, so the
patched image is re-composited with a fresh seeded RNG. Patch SIZE is fixed
(EVAL_AREA_FRAC), only the random rotation may differ slightly from the exact
evaluated frame. The drawn boxes come straight from the saved detections, so the
green/red boxes and the disappeared-count are exact.
"""
import os, sys, json, glob
import numpy as np
import cv2
import torch
import patch_attack_hide as H          # import-safe: no CUDA runs at import time

H.MODE = "figure"                      # any non-"adversarial" value -> apply_patch
                                       # skips test-time jitter (clean canonical patch)

SPLITS = {
    "val":  ("datasets/VisDrone/images/val",  "datasets/VisDrone/labels/val"),
    "test": ("datasets/VisDrone/images/test", "datasets/VisDrone/labels/test"),
}


def find_source(stem, prefer_test):
    order = ["test", "val"] if prefer_test else ["val", "test"]
    for k in order:
        img_dir, lbl_dir = SPLITS[k]
        p = os.path.join(img_dir, stem + ".jpg")
        if os.path.isfile(p):
            return p, os.path.join(lbl_dir, stem + ".txt")
    return None, None


def n_vehicles(dets):
    return sum(1 for d in dets if int(d[0]) in H.TARGET_CLASSES)


def load_all_gt(lbl_path, img_size):
    """EVERY ground-truth box in the label file (nothing filtered out), each tagged
    kept/excluded. kept = scored vehicle (car/van/truck/bus) with area >= 0.1% of
    the image; excluded = motor (class 4) OR below the 0.1% filter."""
    gt = []
    if not os.path.isfile(lbl_path):
        return gt
    for line in open(lbl_path):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0]); xc, yc, w, h = map(float, p[1:5])
        kept = (c in H.TARGET_CLASSES) and (w * h >= 0.001)
        x1, y1 = (xc - w / 2) * img_size, (yc - h / 2) * img_size
        x2, y2 = (xc + w / 2) * img_size, (yc + h / 2) * img_size
        gt.append((c, x1, y1, x2, y2, kept))
    return gt


def draw_panel(img_bgr, gt, dets, det_color, title):
    out = img_bgr.copy()
    # ground truth UNDERNEATH: yellow = scored vehicle, gray = excluded (motor / <0.1%)
    for (c, x1, y1, x2, y2, kept) in gt:
        col = (0, 255, 255) if kept else (150, 150, 150)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 1)
    # detections ON TOP: green (clean) / red (patched)
    for d in dets:
        c = int(d[0])
        if c not in H.TARGET_CLASSES:
            continue
        x1, y1, x2, y2, conf = d[1], d[2], d[3], d[4], d[5]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, det_color, 2)
        cv2.putText(out, f"{H.CLS_NAMES[c]} {conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(run_dir, n_examples=10):
    run_dir = run_dir.rstrip("/")
    prefer_test = run_dir.endswith("_test")

    # img_size + class-mean ASR from results.json (h3 is 1280)
    img_size = 640
    cls_mean = None
    rj = os.path.join(run_dir, "results.json")
    if os.path.isfile(rj):
        _rj = json.load(open(rj))
        img_size = int(_rj.get("img_size", 640))
        cls_mean = _rj.get("hiding_asr_class_mean")

    patch = torch.load(os.path.join(run_dir, "universal_patch.pt"),
                       map_location="cpu").float()

    # rank every image by vehicles-disappeared (clean_veh - patched_veh), from json
    scored = []
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf))
        disappeared = n_vehicles(d["clean"]) - n_vehicles(d["patched"])
        scored.append((disappeared, stem, d))
    if not scored:
        print(f"[fig] no .boxes.json in {run_dir}"); return
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n_examples]
    print(f"[fig] {run_dir}: {len(scored)} imgs | img_size={img_size} | "
          f"top {len(top)} by vehicles-disappeared (range {top[0][0]}..{top[-1][0]})")

    out_dir = os.path.join(run_dir, "annotated")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(H.SEED)
    made = 0
    for disappeared, stem, d in top:
        ip, lbl = find_source(stem, prefer_test)
        if ip is None:
            print(f"  [skip] source image not found for {stem}"); continue
        orig = cv2.imread(ip)
        if orig is None:
            print(f"  [skip] cv2 could not read {ip}"); continue
        img = cv2.resize(orig, (img_size, img_size))
        x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
              .float().unsqueeze(0) / 255.0)               # CPU tensor, RGB
        boxes, _ = H.load_boxes_norm(lbl)
        with torch.no_grad():
            pimg = H.apply_patch(x0, patch, boxes, H.EVAL_AREA_FRAC, rng).clamp(0, 1)
        patched_bgr = (pimg[0].permute(1, 2, 0).cpu().numpy() * 255
                       ).astype(np.uint8)[:, :, ::-1].copy()

        gt = load_all_gt(lbl, img_size)
        n_gt_kept = sum(1 for g in gt if g[5])
        left = draw_panel(img, gt, d["clean"], (0, 255, 0),
                          f"CLEAN  (GT {n_gt_kept} veh / {n_vehicles(d['clean'])} detected)")
        right = draw_panel(patched_bgr, gt, d["patched"], (0, 0, 255),
                           f"PATCHED  (GT {n_gt_kept} veh / {n_vehicles(d['patched'])} detected)")
        combo = np.hstack([left, right])
        # caption + legend bar (self-labeling figure)
        cap = np.zeros((64, combo.shape[1], 3), np.uint8)
        cm = f"{cls_mean:.3f}" if cls_mean is not None else "n/a"
        cv2.putText(cap, f"{os.path.basename(run_dir)}   |   class-mean ASR {cm}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        legend = [((0, 255, 255), "GT vehicle"),
                  ((150, 150, 150), "GT excluded (motor / <0.1%)"),
                  ((0, 255, 0), "clean detection"),
                  ((0, 0, 255), "patched detection")]
        x = 10
        for col, txt in legend:
            cv2.rectangle(cap, (x, 42), (x + 18, 56), col, -1)
            cv2.putText(cap, txt, (x + 24, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            x += 24 + len(txt) * 9 + 22
        combo = np.vstack([combo, cap])
        outp = os.path.join(out_dir, f"{stem}_compare.png")
        cv2.imwrite(outp, combo)
        made += 1
        print(f"  [save] {outp}  (-{disappeared} vehicles)")
    print(f"[fig] wrote {made} comparison figures to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python make_figures.py <run_dir> [n_examples]"); sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
