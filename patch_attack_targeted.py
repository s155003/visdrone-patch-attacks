"""
TARGETED adversarial patch attack: CAR -> VAN.

Patch geometry is IDENTICAL to patch_attack_hide.py (the verbatim Pathak/Thys
replication): a plain square sized by % of bounding-box area, centred on the
box, rotated +-20 deg, composited straight onto the vehicle.

  NO paint detection. NO ellipse footprint. NO coverage floor.
  Those belonged to the autobody-wrap approach and are gone.

What changed vs the hiding script: the LOSS and the METRIC.
  hiding   -> minimize max class score      -> vehicle disappears
  targeted -> minimize (car - van) margin   -> car is called a van

USAGE: cd ~/attack-test && RUN_NAME=t20 AREA_FRAC=0.20 \
       ~/attack_env/bin/python -u patch_attack_targeted.py

MAIN VARIABLE: AREA_FRAC — the patch's area as a fraction of the car's
bounding box. This is the sweep. 0.10 / 0.20 / 0.30 / 0.40.
"""

import os, glob, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "yolo11l_visdrone_pretrained.pt"
DEVICE     = "cuda"

RUN_NAME   = os.environ.get("RUN_NAME", "t_test")
MODE       = os.environ.get("MODE", "adversarial")   # adversarial|gray|random|none
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "640"))
AREA_FRAC  = float(os.environ.get("AREA_FRAC", "0.20"))   # <-- THE SWEEP
TV_WEIGHT  = float(os.environ.get("TV_WEIGHT", "2.5"))
EPOCHS     = int(os.environ.get("EPOCHS", "80"))

# train on val, score on test. Model was fine-tuned on train, so it has seen
# neither. Zero overlap between the two sets below.
TRAIN_IMAGES = os.environ.get("TRAIN_IMAGES", "datasets/VisDrone/images/val")
TRAIN_LABELS = os.environ.get("TRAIN_LABELS", "datasets/VisDrone/labels/val")
EVAL_IMAGES  = os.environ.get("EVAL_IMAGES",  "datasets/VisDrone/images/test")
EVAL_LABELS  = os.environ.get("EVAL_LABELS",  "datasets/VisDrone/labels/test")

NUM_TRAIN  = int(os.environ.get("NUM_TRAIN", "100"))
NUM_EVAL   = int(os.environ.get("NUM_EVAL", "0"))     # 0 = all

SAVE_DIR   = f"patch_examples_{RUN_NAME}"
PATCH_PX   = 64
LR         = float(os.environ.get("LR", "0.03"))
CONF_THRESH = 0.25
IOU_THRESH  = 0.5
SEED       = int(os.environ.get("SEED", "0"))

SRC_IDX, TGT_IDX = 0, 1        # car -> van
MARGIN     = 5.0               # from patch_attack_ms.py's targeted_loss
CLS_NAMES  = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

# Object size filter, as a fraction of ORIGINAL image area. Labels are
# normalized, so w*h IS that fraction — resize-invariant.
# Pathak uses 0.001 (0.1%). Raise it to keep only cars big enough that a
# patch on them is a physically plausible printed sign.
MIN_AREA_FRAC = float(os.environ.get("MIN_AREA_FRAC", "0.001"))

# Keep the patch off the windshield by clamping placement to the middle of the
# box (mostly roof). 1.0 = centred exactly, no clamp needed since the patch is
# centred anyway; PLACE_JITTER lets it wander within the middle fraction.
PLACE_FRAC = float(os.environ.get("PLACE_FRAC", "0.0"))   # 0 = dead centre
PLACE_MODE  = os.environ.get("PLACE_MODE", "center")   # center | offset
OFFSET_FRAC = float(os.environ.get("OFFSET_FRAC", "1.0"))
LOWER_TALL = float(os.environ.get("LOWER_TALL", "0.60"))   # frac down from top, taller-than-wide
LOWER_WIDE = float(os.environ.get("LOWER_WIDE", "0.75"))   # frac down from top, wider-than-tall
ROOF_LEN = float(os.environ.get("ROOF_LEN", "0.30"))   # roof = central ROOF_LEN of car LENGTH
ROOF_WID = float(os.environ.get("ROOF_WID", "0.70"))   #        x ROOF_WID of car WIDTH
_ROOF = {"fit": 0, "n": 0}   # ROOF metric accumulator (eval pass)
_LOWER = {"clamped": 0, "oncar_sum": 0.0, "n": 0}   # LOWER metric accumulator (eval pass)
_ROOFP = {"area_sum": 0.0, "shrunk": 0, "n": 0}   # ROOF-placement metric (eval pass)

ROT_DEG    = float(os.environ.get("ROT_DEG", "20.0"))
# ==================================================


def load_car_boxes(label_path):
    """[(cls, xc, yc, w, h)] normalized. Cars only — they're the attack target."""
    out, dropped = [], 0
    if not os.path.exists(label_path):
        return out, dropped
    for line in open(label_path).read().strip().splitlines():
        if not line:
            continue
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        if c != SRC_IDX:
            continue
        xc, yc, w, h = map(float, p[1:5])
        if w * h < MIN_AREA_FRAC:
            dropped += 1
            continue
        out.append((c, xc, yc, w, h))
    return out, dropped


def tv_loss(patch):
    """Thys 3: sum sqrt((dx)^2 + (dy)^2). Keeps the patch printable-smooth."""
    dh = (patch[:, :-1, :-1] - patch[:, 1:, :-1]) ** 2
    dw = (patch[:, :-1, :-1] - patch[:, :-1, 1:]) ** 2
    return torch.sqrt(dh + dw + 1e-6).sum() / patch.numel()


def apply_patch(img, patch, boxes, frac, rng):
    """
    IDENTICAL geometry to the Pathak replication: square patch, area = frac *
    bbox area, centred on the box, rotated +-20 deg. No mask, no ellipse.
    One affine grid_sample per car; gradients reach the patch.
    """
    _, _, H, W = img.shape
    out = img
    for (_, xc, yc, bw, bh) in boxes:
        cx, cy = xc * W, yc * H
        bwp, bhp = bw * W, bh * H
        if PLACE_FRAC > 0:      # optional wander within the middle of the box
            cx += rng.uniform(-1, 1) * PLACE_FRAC * bwp / 2
            cy += rng.uniform(-1, 1) * PLACE_FRAC * bhp / 2
        s = math.sqrt(frac * bwp * bhp)
        if s < 4:
            continue
        th = math.radians(rng.uniform(-ROT_DEG, ROT_DEG))
        ct, st = math.cos(th), math.sin(th)
        # NOTE: th moved above the placement branches (needed for PLACE_MODE=lower's clamp).
        # This reorders the RNG vs the old code, so OFFSET runs no longer reproduce
        # bit-for-bit (s2d already computed; flagged so nobody's confused later).
        if PLACE_MODE == "offset":
            if bhp >= bwp:                      # box taller than wide -> car runs vertically
                room = max(0.0, bhp/2 - s/2)
                cy += rng.choice([-1, 1]) * room * OFFSET_FRAC
            else:                               # box wider than tall -> car runs horizontally
                room = max(0.0, bwp/2 - s/2)
                cx += rng.choice([-1, 1]) * room * OFFSET_FRAC
        elif PLACE_MODE == "lower":
            # aspect decides which way the car runs; place low along the box's vertical axis
            ext = s * (abs(ct) + abs(st))       # rotated footprint extent
            y_top, y_bot = yc*H - bhp/2, yc*H + bhp/2
            cy_new = y_top + (LOWER_TALL if bhp >= bwp else LOWER_WIDE) * bhp
            lo, hi = y_top + ext/2, y_bot - ext/2
            if lo > hi:                          # patch too big to fit -> centre it
                cy = yc*H; _clamp = True
            else:
                cyc = min(max(cy_new, lo), hi)
                _clamp = abs(cyc - cy_new) > 1e-6; cy = cyc
            # on_car_frac: fraction of rotated patch footprint inside the inscribed ellipse
            _g = np.linspace(-0.5, 0.5, 9) * s
            _gx, _gy = np.meshgrid(_g, _g)
            _rx = cx + _gx*ct - _gy*st; _ry = cy + _gx*st + _gy*ct
            _oncar = float((((_rx - xc*W)/(bwp/2))**2 + ((_ry - yc*H)/(bhp/2))**2 <= 1.0).mean())
            _LOWER["clamped"] += int(_clamp); _LOWER["oncar_sum"] += _oncar; _LOWER["n"] += 1
        elif PLACE_MODE == "roof":
            # roof = central ROOF_LEN of car LENGTH x ROOF_WID of car WIDTH, box-centred.
            # Shrink the square so its rotated footprint fits STRICTLY inside that rectangle
            # -> can never reach windshield / rear glass / doors. Guaranteed by construction:
            # s capped so s*(|cos|+|sin|) <= min(roof extents). Patch only shrinks, never grows.
            _Lc = max(bwp, bhp); _Wc = min(bwp, bhp)
            _ext = abs(ct) + abs(st)
            _s_fit = min(ROOF_LEN * _Lc, ROOF_WID * _Wc) / max(_ext, 1e-6)
            _req_s = s
            s = min(s, _s_fit)
            if s < 4:
                continue                          # roof too small on this car -> skip
            _ROOFP["area_sum"] += (s*s) / (bwp*bhp)
            _ROOFP["shrunk"] += int(s < _req_s - 1e-6)
            _ROOFP["n"] += 1
        # ROOF metric: does the patch (worst-case +-ROT_DEG extent) fit entirely inside the
        # roof = central 15% of car LENGTH x 70% of car WIDTH (long axis = box long axis)?
        _rc = math.radians(ROT_DEG)
        _he = (s/2) * (abs(math.cos(_rc)) + abs(math.sin(_rc)))
        if bhp >= bwp:                          # long axis vertical
            _dl, _ds, _Ln, _Wd = abs(cy - yc*H), abs(cx - xc*W), bhp, bwp
        else:                                   # long axis horizontal
            _dl, _ds, _Ln, _Wd = abs(cx - xc*W), abs(cy - yc*H), bwp, bhp
        _ROOF["fit"] += int((_dl + _he <= 0.075*_Ln) and (_ds + _he <= 0.35*_Wd))
        _ROOF["n"] += 1
        a11, a12 = W * ct / s,  H * st / s
        a13 = (2.0 / s) * (ct * (W/2 - cx) + st * (H/2 - cy))
        a21, a22 = -W * st / s, H * ct / s
        a23 = (2.0 / s) * (-st * (W/2 - cx) + ct * (H/2 - cy))
        theta = torch.tensor([[[a11, a12, a13], [a21, a22, a23]]],
                             device=img.device, dtype=img.dtype)
        grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
        warped = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False,
                               padding_mode="zeros")
        mask = F.grid_sample(torch.ones_like(patch[:1]).unsqueeze(0), grid,
                             align_corners=False, padding_mode="zeros")
        out = out * (1 - mask) + warped * mask
    return out


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def targeted_loss(raw_out, boxes):
    """
    Margin loss from patch_attack_ms.py: push VAN above CAR.

      clamp(car_logit - van_logit + MARGIN, min=0), summed over in-box anchors

    LOGIT SPACE, not probability space. cls comes out of the head already
    sigmoided; at p=0.95 the gradient is p(1-p)=0.05, so minimizing the
    probability pushes on the flat tail and the loss goes inert. This is the
    single most important detail in the whole script.
    """
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    pred = pred[0]                              # [9, N]
    cx = pred[0, :] / IMG_SIZE
    cy = pred[1, :] / IMG_SIZE
    cls = pred[4:, :]                           # [5, N] sigmoided
    src_l = _logit(cls[SRC_IDX, :])
    tgt_l = _logit(cls[TGT_IDX, :])
    TAU = math.log(CONF_THRESH / (1 - CONF_THRESH))   # logit of the detection threshold

    margin_t, hinge_t, n = None, None, 0
    for (_, xc, yc, w, h) in boxes:
        m = ((cx >= xc - w/2) & (cx <= xc + w/2) &
             (cy >= yc - h/2) & (cy <= yc + h/2))
        if m.sum() == 0:
            continue
        vm = torch.clamp(src_l[m] - tgt_l[m] + MARGIN, min=0).sum()   # van above car
        vh = torch.clamp(TAU - tgt_l[m], min=0).sum()                 # van above threshold
        margin_t = vm if margin_t is None else margin_t + vm
        hinge_t  = vh if hinge_t  is None else hinge_t  + vh
        n += 1
    if n == 0:
        z = cls.sum() * 0.0
        return z, z, 0
    return margin_t, hinge_t, n


def detect(yolo, timg):
    """From patch_attack_ms.py. Returns [(cls, x1, y1, x2, y2, conf)]."""
    arr = (timg[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:, :, ::-1].copy(), verbose=False,
                           conf=CONF_THRESH, device=DEVICE)[0]
    return [(int(b.cls[0]), *b.xyxy[0].tolist(), float(b.conf[0]))
            for b in res.boxes]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detected_as(dets, cls_id, box_px):
    for c, x1, y1, x2, y2, _ in dets:
        if c == cls_id and iou(box_px, (x1, y1, x2, y2)) >= IOU_THRESH:
            return True
    return False


def evaluate(yolo, data, patch, rng):
    """
    Targeted ASR, Pathak-style denominator:
      eligible = cars CORRECTLY DETECTED AS CAR in the clean image
      flipped  = those same cars now detected as VAN in the patched image
      ASR      = flipped / eligible

    Also reports 'lost' — cars that vanished entirely rather than flipping.
    That is a hiding side-effect, NOT a targeted success. Kept separate.
    """
    elig = amb = flipped = newvan = lost = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    _ROOF["fit"] = 0; _ROOF["n"] = 0    # count roof-fit over the eval pass only
    _LOWER["clamped"] = 0; _LOWER["oncar_sum"] = 0.0; _LOWER["n"] = 0
    _ROOFP["area_sum"] = 0.0; _ROOFP["shrunk"] = 0; _ROOFP["n"] = 0
    for stem, x0, boxes in data:
        clean = detect(yolo, x0)
        if MODE == "none":
            patched = clean
        else:
            with torch.no_grad():
                pimg = apply_patch(x0, patch, boxes, AREA_FRAC, rng).clamp(0, 1)
            patched = detect(yolo, pimg)
        for (c, xc, yc, w, h) in boxes:
            bpx = ((xc-w/2)*IMG_SIZE, (yc-h/2)*IMG_SIZE,
                   (xc+w/2)*IMG_SIZE, (yc+h/2)*IMG_SIZE)
            if not detected_as(clean, SRC_IDX, bpx):
                continue
            elig += 1
            van_c = detected_as(clean,   TGT_IDX, bpx)   # ambiguous: already van in clean
            van_p = detected_as(patched, TGT_IDX, bpx)
            if van_c: amb += 1
            if van_p: flipped += 1                        # original (aggregate, inflated)
            if van_p and not van_c: newvan += 1           # patch-induced flip
            if (not van_p) and (not detected_as(patched, SRC_IDX, bpx)): lost += 1
        with open(os.path.join(SAVE_DIR, stem + ".boxes.json"), "w") as f:
            json.dump({"clean": clean, "patched": patched}, f)
    if _ROOF["n"] > 0:
        print(f"ROOF: roof_frac={100*_ROOF['fit']/_ROOF['n']:.1f}% n={_ROOF['n']}", flush=True)
    if _LOWER["n"] > 0:
        print(f"LOWER: clamped={100*_LOWER['clamped']/_LOWER['n']:.1f}% "
              f"mean_on_car_frac={_LOWER['oncar_sum']/_LOWER['n']:.3f} n={_LOWER['n']}", flush=True)
    if _ROOFP["n"] > 0:
        print(f"ROOFFIT: mean_realized_area={_ROOFP['area_sum']/_ROOFP['n']:.3f} "
              f"shrunk={100*_ROOFP['shrunk']/_ROOFP['n']:.1f}% n={_ROOFP['n']}", flush=True)
    strict_den = elig - amb
    return dict(elig=elig, amb=amb, flipped=flipped, newvan=newvan, lost=lost,
                asr_strict=(newvan/strict_den if strict_den else 0.0),
                asr_orig=(flipped/elig if elig else 0.0),
                asr_newvan=(newvan/elig if elig else 0.0))


def build(img_dir, lbl_dir, limit):
    out, dropped, kept = [], 0, 0
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        lbl = os.path.join(lbl_dir,
                           os.path.splitext(os.path.basename(ip))[0] + ".txt")
        boxes, d = load_car_boxes(lbl)
        dropped += d
        if not boxes:
            continue
        orig = cv2.imread(ip)
        if orig is None:
            continue
        img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
              .float().unsqueeze(0) / 255.0).to(DEVICE)
        out.append((os.path.splitext(os.path.basename(ip))[0], x0, boxes))
        kept += len(boxes)
        if limit > 0 and len(out) >= limit:
            break
    return out, kept, dropped


def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = random.Random(SEED)

    print(f"=== {RUN_NAME} | car->van | MODE={MODE} | AREA_FRAC={AREA_FRAC} | "
          f"IMG_SIZE={IMG_SIZE} | ROT_DEG={ROT_DEG} | LR={LR} | "
          f"LOWER_TALL={LOWER_TALL} | LOWER_WIDE={LOWER_WIDE} | "
          f"ROOF_LEN={ROOF_LEN} | ROOF_WID={ROOF_WID} | MIN_AREA_FRAC={MIN_AREA_FRAC} ===", flush=True)

    yolo = YOLO(MODEL_PATH)
    raw = YOLO(MODEL_PATH).model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters():
        p.requires_grad_(False)

    eval_set, ev_kept, ev_drop = build(EVAL_IMAGES, EVAL_LABELS, NUM_EVAL)
    print(f"[eval ] {EVAL_IMAGES}: {len(eval_set)} images | {ev_kept} cars | "
          f"{ev_drop} dropped below {MIN_AREA_FRAC*100:.2f}% area", flush=True)

    if MODE == "adversarial":
        train_set, tr_kept, tr_drop = build(TRAIN_IMAGES, TRAIN_LABELS, NUM_TRAIN)
        print(f"[train] {TRAIN_IMAGES}: {len(train_set)} images | "
              f"{tr_kept} cars", flush=True)

    if MODE == "gray":
        patch = torch.full((3, PATCH_PX, PATCH_PX), 0.5, device=DEVICE)
    elif MODE in ("random", "none"):
        patch = torch.rand((3, PATCH_PX, PATCH_PX), device=DEVICE)
    else:
        patch = torch.rand((3, PATCH_PX, PATCH_PX), device=DEVICE,
                           requires_grad=True)

    if MODE == "adversarial":
        opt = torch.optim.Adam([patch], lr=LR)
        t0 = time.time()
        for ep in range(EPOCHS):
            tot = mar_s = hin_s = tv_s = 0.0
            nb = 0
            for stem, x0, boxes in train_set:
                pimg = apply_patch(x0, patch.clamp(0, 1), boxes,
                                   AREA_FRAC, rng).clamp(0, 1)
                lmar, lhinge, n = targeted_loss(raw(pimg), boxes)
                if n == 0:
                    continue
                ltv = torch.clamp(TV_WEIGHT * tv_loss(patch.clamp(0, 1)),
                                  min=0.1)
                loss = lmar + lhinge + ltv
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
                tot += float(loss.detach().cpu())
                mar_s += float(lmar.detach().cpu())
                hin_s += float(lhinge.detach().cpu())
                tv_s += float(ltv.detach().cpu())
                nb += 1
            print(f"  epoch {ep+1}/{EPOCHS}  total {tot/max(1,nb):8.3f}  "
                  f"margin {mar_s/max(1,nb):8.3f}  hinge {hin_s/max(1,nb):8.3f}  "
                  f"tv {tv_s/max(1,nb):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # SAVE BEFORE EVAL — a crash at the save step after training has burned a
    # run before. .cpu() on everything.
    final = patch.detach().clamp(0, 1)
    torch.save(final.cpu(), os.path.join(SAVE_DIR, "universal_patch.pt"))
    cv2.imwrite(os.path.join(SAVE_DIR, "universal_patch.png"),
                (final.cpu().permute(1, 2, 0).numpy() * 255)
                .astype(np.uint8)[:, :, ::-1])
    print(f"[save] {SAVE_DIR}/universal_patch.png", flush=True)

    print(f"\nEvaluating on {len(eval_set)} images...", flush=True)
    r = evaluate(yolo, eval_set, final, rng)
    elig, amb, flipped, newvan, lost = r["elig"], r["amb"], r["flipped"], r["newvan"], r["lost"]

    res = dict(run=RUN_NAME, mode=MODE, area_frac=AREA_FRAC,
               img_size=IMG_SIZE, min_area_frac=MIN_AREA_FRAC, rot_deg=ROT_DEG,
               place_mode=PLACE_MODE, lower_tall=LOWER_TALL, lower_wide=LOWER_WIDE,
               roof_len=ROOF_LEN, roof_wid=ROOF_WID,
               epochs=EPOCHS, eval_images=len(eval_set),
               cars_kept=ev_kept, cars_dropped=ev_drop,
               eligible=elig, ambiguous=amb, flipped=flipped, new_van=newvan, lost=lost,
               targeted_asr=r["asr_strict"],            # STRICT is the primary metric now
               targeted_asr_strict=r["asr_strict"],
               targeted_asr_newvan=r["asr_newvan"],
               targeted_asr_orig=r["asr_orig"],
               lost_rate=(lost/elig if elig else 0.0))
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n==== CAR->VAN — {RUN_NAME} (patch = {AREA_FRAC*100:.0f}% of bbox) ====")
    print(f"eval images        : {len(eval_set)}")
    print(f"eligible (car clean): {elig}   ambiguous (van@box in clean): {amb}")
    print(f"flipped to van      : {flipped}  (new-van, patch-induced: {newvan})")
    print(f"lost entirely       : {lost}  (hiding side-effect, NOT a flip)")
    print(f"\ntargeted ASR (STRICT, excl {amb} ambiguous): {r['asr_strict']:.3f}   <- HEADLINE")
    print(f"targeted ASR (new-van / all eligible)      : {r['asr_newvan']:.3f}")
    print(f"targeted ASR (original, inflated)          : {r['asr_orig']:.3f}")
    print(f"lost rate   : {lost/elig if elig else 0:.3f}")


if __name__ == "__main__":
    run()
