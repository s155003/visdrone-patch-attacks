"""Before/after figures for the TARGETED car->van runs (three-outcome coloring).

READ-ONLY w.r.t. the detector: reuses the SAVED <stem>.boxes.json detections,
never runs YOLO, never touches CUDA. Reuses patch_attack_targeted's functions.

Per ground-truth car, the RIGHT (patched) panel colors the box by outcome, exactly
as evaluate() scores it:
    green = still detected as car  (attack failed)
    blue  = now detected as van    (attack succeeded -- this is the ASR)
    red   = no detection at all     (vanished; a failure, not a success)
LEFT (clean) panel = clean image + its car detections (green).

Modes:
    top    -> the n images with the MOST car->van flips (best-case; overstates)
    random -> n images picked with a FIXED seed (SAMPLE_SEED), representative;
              same seed + same test set => same images across runs, comparable.

Usage:  ~/attack_env/bin/python make_figures_targeted.py <run_dir> [n] [top|random]
"""
import os, sys, json, glob
import numpy as np
import cv2
import torch
import patch_attack_targeted as T          # import-safe: no CUDA at import

IMG_DIR = "datasets/VisDrone/images/test"  # targeted runs eval on the test split
LBL_DIR = "datasets/VisDrone/labels/test"
SAMPLE_SEED = 1234                          # fixed; identical across runs

COL = {"car": (0, 255, 0), "flip": (255, 0, 0), "lost": (0, 0, 255),
       "ambig": (0, 255, 255)}   # BGR; ambig = yellow


def outcome(bpx, clean, patched):
    """None = not eligible (car not detected in clean). Else, matching the STRICT
    metric:
      ambig = already van@box in CLEAN -> ill-posed, NOT a patch success (4th colour)
      flip  = van@box in patched AND NOT in clean -> patch-induced (the ASR)
      lost  = no detection at all
      car   = still car"""
    if not T.detected_as(clean, T.SRC_IDX, bpx):
        return None
    if T.detected_as(clean, T.TGT_IDX, bpx):
        return "ambig"                       # pre-existing van; excluded from ASR
    if T.detected_as(patched, T.TGT_IDX, bpx):
        return "flip"                        # van in patched, not in clean
    if not T.detected_as(patched, T.SRC_IDX, bpx):
        return "lost"
    return "car"


def header(img, title):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def main(run_dir, n=10, mode="top"):
    run_dir = run_dir.rstrip("/")
    rj = json.load(open(os.path.join(run_dir, "results.json")))
    img_size = int(rj.get("img_size", 640))
    area_frac = float(rj.get("area_frac", 0.20))
    run_mode = rj.get("mode", "adversarial")
    # restore the run's ACTUAL placement so the reconstructed patch lands where it did
    T.PLACE_MODE = rj.get("place_mode", "center")
    T.LOWER_TALL = float(rj.get("lower_tall", 0.60))
    T.LOWER_WIDE = float(rj.get("lower_wide", 0.75))
    T.ROOF_LEN = float(rj.get("roof_len", 0.30))
    T.ROOF_WID = float(rj.get("roof_wid", 0.70))
    T.ROT_DEG = float(rj.get("rot_deg", 20.0))
    patch = torch.load(os.path.join(run_dir, "universal_patch.pt"),
                       map_location="cpu").float()

    # score every image (outcomes per GT car) and accumulate full-set totals
    scored = []
    agg = {"flip": 0, "car": 0, "lost": 0, "ambig": 0}
    S_elig = S_amb = S_newvan = 0   # strict-ASR accumulators (patch-induced flips only)
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf))
        boxes, _ = T.load_car_boxes(os.path.join(LBL_DIR, stem + ".txt"))
        outs, flips = [], 0
        for (c, xc, yc, w, h) in boxes:
            bpx = ((xc - w/2)*img_size, (yc - h/2)*img_size,
                   (xc + w/2)*img_size, (yc + h/2)*img_size)
            o = outcome(bpx, d["clean"], d["patched"])
            outs.append((bpx, o))
            if o is not None:
                agg[o] += 1
                flips += (o == "flip")
                S_elig += 1
                vc = T.detected_as(d["clean"], T.TGT_IDX, bpx)
                vp = T.detected_as(d["patched"], T.TGT_IDX, bpx)
                if vc: S_amb += 1
                if vp and not vc: S_newvan += 1
        scored.append((flips, stem, d, outs))
    if not scored:
        print(f"[fig] no .boxes.json in {run_dir}"); return
    strict_asr = S_newvan / (S_elig - S_amb) if (S_elig - S_amb) else 0.0

    # selection
    if mode == "random":
        by_stem = sorted(scored, key=lambda t: t[1])          # deterministic order
        sel_rng = np.random.RandomState(SAMPLE_SEED)          # same seed both runs
        idx = sorted(sel_rng.choice(len(by_stem), min(n, len(by_stem)),
                                    replace=False).tolist())
        sel = [by_stem[i] for i in idx]
        suffix, sel_label = "_random", f"RAND-{len(sel)}"
        tag = f"RANDOM SAMPLE (seed {SAMPLE_SEED})"
        print(f"[fig] {run_dir}: {len(scored)} imgs | RANDOM {len(sel)} (seed {SAMPLE_SEED})")
    elif mode == "full":
        sel = scored                                          # every image
        suffix, sel_label = "_full", f"FULL-{len(sel)}"
        tag = "FULL EVAL SET"
        print(f"[fig] {run_dir}: {len(scored)} imgs | FULL {len(sel)} -> _full.jpg (q92)")
    else:
        scored.sort(key=lambda t: t[0], reverse=True)
        sel = scored[:n]
        suffix, sel_label = "_targeted", f"TOP-{len(sel)}"
        tag = "SELECTED HIGH-FLIP EXAMPLE -- NOT TYPICAL"
        print(f"[fig] {run_dir}: {len(scored)} imgs | TOP {len(sel)} by flips "
              f"(range {sel[0][0]}..{sel[-1][0]})")

    # representativeness: selected set vs full eval set
    sel_agg = {"flip": 0, "car": 0, "lost": 0, "ambig": 0}
    for _, _, _, outs in sel:
        for _, o in outs:
            if o is not None:
                sel_agg[o] += 1

    def summarize(a):
        t = a["flip"] + a["car"] + a["lost"]
        return t, (100 * a["flip"] / t if t else 0.0)
    ft, fbp = summarize(agg)
    st, sbp = summarize(sel_agg)
    print(f"[repr] FULL eval : eligible {ft:5d} | blue(->van) {agg['flip']:5d} ({fbp:4.1f}%)  "
          f"green(car) {agg['car']:5d}  red(lost) {agg['lost']:5d}")
    print(f"[repr] {sel_label:8s}: eligible {st:5d} | blue(->van) {sel_agg['flip']:5d} ({sbp:4.1f}%)  "
          f"green(car) {sel_agg['car']:5d}  red(lost) {sel_agg['lost']:5d}")
    if mode == "top":
        v = "representative" if sbp <= fbp + 10 else f"OVERSTATES ({sbp:.1f}% vs {fbp:.1f}%)"
        print(f"[repr] top blue {sbp:.1f}% vs full {fbp:.1f}%  ->  {v}")
    else:
        print(f"[repr] random blue {sbp:.1f}% vs full {fbp:.1f}%  (gap {sbp-fbp:+.1f} pp)")
    print(f"[repr] selected stems: {[s[1] for s in sel]}")

    out_dir = os.path.join(run_dir, "annotated")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(T.SEED)
    for flips, stem, d, outs in sel:
        ip = os.path.join(IMG_DIR, stem + ".jpg")
        orig = cv2.imread(ip)
        if orig is None:
            print(f"  [skip] cannot read {ip}"); continue
        img = cv2.resize(orig, (img_size, img_size))
        x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
              .float().unsqueeze(0) / 255.0)
        boxes, _ = T.load_car_boxes(os.path.join(LBL_DIR, stem + ".txt"))
        if run_mode == "none":
            pbgr = img.copy()   # this run applied no patch; patched == clean
        else:
            with torch.no_grad():
                pimg = T.apply_patch(x0, patch, boxes, area_frac, rng).clamp(0, 1)
            pbgr = (pimg[0].permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)[:, :, ::-1].copy()

        left = img.copy()
        n_clean_car = 0
        for det in d["clean"]:
            if int(det[0]) != T.SRC_IDX:
                continue
            x1, y1, x2, y2 = int(det[1]), int(det[2]), int(det[3]), int(det[4])
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n_clean_car += 1
        header(left, f"CLEAN  ({n_clean_car} cars detected)")

        nf = nc = nl = na = 0
        for bpx, o in outs:
            if o is None:
                continue
            x1, y1, x2, y2 = int(bpx[0]), int(bpx[1]), int(bpx[2]), int(bpx[3])
            cv2.rectangle(pbgr, (x1, y1), (x2, y2), COL[o], 2)
            nf += (o == "flip"); nc += (o == "car"); nl += (o == "lost"); na += (o == "ambig")
        header(pbgr, f"PATCHED  flip={nf}  car={nc}  lost={nl}  ambig={na}")

        combo = np.hstack([left, pbgr])
        cap = np.zeros((86, combo.shape[1], 3), np.uint8)
        tag_col = (0, 220, 255) if mode == "top" else (200, 255, 200)
        cv2.putText(cap, tag, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.64, tag_col, 2, cv2.LINE_AA)
        cv2.putText(cap, f"{os.path.basename(run_dir)}   |   patch {int(round(area_frac*100))}% of bbox"
                    f"   |   strict ASR {strict_asr:.3f}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        legend = [((0, 255, 0), "still car (fail)"),
                  ((255, 0, 0), "-> van (ASR success)"),
                  ((0, 0, 255), "vanished (fail)"),
                  ((0, 255, 255), "ambiguous (excl)")]
        x = 10
        for col, txt in legend:
            cv2.rectangle(cap, (x, 62), (x + 16, 76), col, -1)
            cv2.putText(cap, txt, (x + 22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            x += 22 + len(txt) * 9 + 20
        combo = np.vstack([combo, cap])
        if mode == "full":
            outp = os.path.join(out_dir, f"{stem}_full.jpg")
            cv2.imwrite(outp, combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            outp = os.path.join(out_dir, f"{stem}{suffix}.png")
            cv2.imwrite(outp, combo)
            print(f"  [save] {outp}  ({nf} flips)")
    print(f"[fig] done -> {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python make_figures_targeted.py <run_dir> [n] [top|random]"); sys.exit(1)
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    md = sys.argv[3] if len(sys.argv) > 3 else "top"
    main(sys.argv[1], nn, md)
