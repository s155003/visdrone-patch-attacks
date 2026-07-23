"""Read-only metrics from saved detections. No model, no GPU, no re-run.

(1) strict ASR / lost / still on the strict denominator (eligible - ambiguous),
    as an exclusive partition so the three rates sum to exactly 1.0.
(2) AP@0.5 (VOC all-points) per class + overall, from the PATCHED detections
    vs VisDrone GT, using the same 0.1% object-area filter.

Detections in <stem>.boxes.json are [cls, x1, y1, x2, y2, conf] in 640px coords.
GT is YOLO-normalised in the custom 5-class remap: 0=car 1=van 2=truck 3=bus 4=motor.

NOTE: saved detections were thresholded at conf 0.25 with NMS already applied, so
this is AP at a FIXED OPERATING POINT, not a full PR sweep. Comparable across
these runs; NOT directly comparable to published VisDrone tables.

usage:  PYTHONPATH=~/attack-test python tools_session/ap_and_rates.py
"""
import glob, os, json
import numpy as np
import patch_attack_targeted as T          # reuse load_car_boxes / detected_as

IMG = 640
LBL = "datasets/VisDrone/labels/test"
MIN_AREA = 0.001                            # 0.1% object filter (== MIN_AREA_FRAC)
NAMES = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

RUNS = ["b0_clean", "s2b_adv_20_hinge", "s2d_offset_20",
        "s2h_center_10", "f_rot00", "f_rot45",
        "sh_roof_square", "sh_roof_circle"]


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(stem):
    """All-class GT in 640px xyxy, dropping boxes below the 0.1% area filter."""
    p = os.path.join(LBL, stem + ".txt")
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p):
        q = ln.split()
        if len(q) < 5:
            continue
        c = int(q[0]); xc, yc, w, h = map(float, q[1:5])
        if w * h < MIN_AREA:
            continue
        out.append((c, ((xc-w/2)*IMG, (yc-h/2)*IMG, (xc+w/2)*IMG, (yc+h/2)*IMG)))
    return out


# ---------- (1) strict ASR / lost / still, exclusive partition ----------
def rates(run_dir):
    elig = amb = flip = still = lost = 0
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf))
        clean, patched = d["clean"], d["patched"]
        lp = os.path.join(LBL, stem + ".txt")
        if not os.path.exists(lp):
            continue
        gt, _ = T.load_car_boxes(lp)                 # eligible cars, 0.1% filtered
        for (c, xc, yc, w, h) in gt:
            b = ((xc-w/2)*IMG, (yc-h/2)*IMG, (xc+w/2)*IMG, (yc+h/2)*IMG)
            if not T.detected_as(clean, T.SRC_IDX, b):
                continue                              # not eligible
            elig += 1
            if T.detected_as(clean, T.TGT_IDX, b):
                amb += 1                              # already van in clean -> excluded
                continue
            # exclusive partition on the non-ambiguous eligible cars
            if T.detected_as(patched, T.TGT_IDX, b):
                flip += 1                             # -> van   (ASR)
            elif T.detected_as(patched, T.SRC_IDX, b):
                still += 1                            # -> car   (RAcc)
            else:
                lost += 1                             # -> gone  (VR)
    den = elig - amb
    return dict(elig=elig, amb=amb, den=den,
                asr=flip/den, lost=lost/den, still=still/den)


# ---------- (2) AP@0.5, VOC all-points ----------
def ap_voc(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_per_class(run_dir):
    dets = {c: [] for c in NAMES}
    gts, ngt = {}, {c: 0 for c in NAMES}
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf))
        gts[stem] = {c: [] for c in NAMES}
        for (c, box) in load_gt(stem):
            if c in NAMES:
                gts[stem][c].append(box); ngt[c] += 1
        for row in d["patched"]:
            c = int(row[0])
            if c in NAMES:
                dets[c].append((row[5] if len(row) > 5 else 1.0, stem,
                                (row[1], row[2], row[3], row[4])))
    aps = {}
    for c in NAMES:
        dl = sorted(dets[c], key=lambda x: -x[0])         # rank by confidence
        matched = {s: np.zeros(len(gts[s][c]), bool) for s in gts}
        tp = np.zeros(len(dl)); fp = np.zeros(len(dl))
        for i, (conf, stem, box) in enumerate(dl):
            best, bj = 0.5, -1                            # IoU >= 0.5 to count
            for j, gb in enumerate(gts[stem][c]):
                if matched[stem][j]:
                    continue
                v = iou_xyxy(box, gb)
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                tp[i] = 1; matched[stem][bj] = True
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / ngt[c] if ngt[c] else np.zeros_like(ctp)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        aps[c] = ap_voc(rec, prec) if ngt[c] else float("nan")
    return aps, ngt


if __name__ == "__main__":
    print("### strict ASR / lost / still on strict denominator (elig - amb) ###")
    print(f"{'run':20} {'elig':>5} {'amb':>4} {'den':>5} {'ASR':>7} {'lost':>7} {'still':>7} {'sum':>6}")
    for r in RUNS:
        d = "patch_examples_" + r
        if not os.path.isdir(d):
            print(f"{r:20} (missing)"); continue
        x = rates(d)
        print(f"{r:20} {x['elig']:>5} {x['amb']:>4} {x['den']:>5} "
              f"{x['asr']:>7.4f} {x['lost']:>7.4f} {x['still']:>7.4f} "
              f"{x['asr']+x['lost']+x['still']:>6.4f}")

    print("\n### AP@0.5 on PATCHED detections (conf 0.25 operating point) ###")
    print(f"{'run':20}" + "".join(f"{NAMES[c]:>8}" for c in NAMES) + f"{'mAP5':>8}{'mAP4':>8}")
    ngt_ref = None
    for r in RUNS:
        d = "patch_examples_" + r
        if not os.path.isdir(d):
            continue
        aps, ngt_ref = ap_per_class(d)
        m5 = np.nanmean([aps[c] for c in NAMES])
        m4 = np.nanmean([aps[c] for c in [0, 1, 2, 3]])
        print(f"{r:20}" + "".join(f"{aps[c]:>8.3f}" for c in NAMES) + f"{m5:>8.3f}{m4:>8.3f}")
    if ngt_ref:
        print(f"{'GT count/class:':20}" + "".join(f"{ngt_ref[c]:>8}" for c in NAMES))
