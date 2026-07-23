"""
SHAPE / OPACITY study — targeted CAR -> VAN adversarial patch.

Six scenarios (one factor changed per row from S1):

  B0   no patch                          -- the floor, must score 0.000
  S1   square   opacity 1.0   10%        -- reference
  S2   circle   opacity 1.0   10%        -- shape
  S3   ellipse  opacity 1.0   10%        -- shape (box-matched)
  S4   triangle opacity 1.0   10%        -- shape
  S5   square   opacity 0.5   10%        -- opacity

All shapes request the SAME AREA. Only the outline changes. Each shape's
canvas fill-fraction is divided out so a circle and a square really do cover
the same number of pixels -- otherwise "shape" would secretly be "size".

  square 1.000 | circle pi/4 | ellipse pi/4 | triangle 0.500

NO_OVERFLOW (default ON): the patch's ROTATED footprint is capped per box so
it fits inside the box's inscribed ellipse. Box corners are road on any
diagonal car, so the ellipse is the proxy for "on the vehicle". This is a
per-box cap, not a global size choice -- a 2:1 box fits a bigger patch than a
square one, so a single "conservative" size would either spill on the tight
boxes or waste area on the roomy ones.

  CAVEAT: the inscribed ellipse is a GUESS at the car's extent. Cars are
  blobby from above so it's a reasonable one, but nobody has verified it
  against pixels. "Zero overflow" means zero outside the ellipse, not zero
  outside the car.

METRIC: strict targeted ASR.
  eligible  = cars correctly detected as CAR in the clean image
  ambiguous = of those, ones that ALSO had a van box in the clean image.
              EXCLUDED -- the detector was already confused, so "did the
              patch flip it?" is ill-posed. This exclusion exists because the
              B0 control scored 0.049 instead of 0.000 without it.
  flipped   = van at the box in patched AND NOT van in clean
  ASR       = flipped / (eligible - ambiguous)

  lost = car erased entirely. A FAILURE, never counted as success.

USAGE: cd ~/attack-test && RUN_NAME=sh_s1 SHAPE=square OPACITY=1.0 \
       AREA_FRAC=0.10 ~/attack_env/bin/python -u patch_attack_shapes.py
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

RUN_NAME    = os.environ.get("RUN_NAME", "sh_test")
MODE        = os.environ.get("MODE", "adversarial")      # adversarial | none
SHAPE       = os.environ.get("SHAPE", "square")          # square|circle|ellipse|triangle
OPACITY     = float(os.environ.get("OPACITY", "1.0"))    # 1.0 = solid
AREA_FRAC   = float(os.environ.get("AREA_FRAC", "0.10"))
NO_OVERFLOW = int(os.environ.get("NO_OVERFLOW", "1"))     # 1 = cap per box
PLACE_MODE  = os.environ.get("PLACE_MODE", "center")     # center | roof
ROOF_LEN    = float(os.environ.get("ROOF_LEN", "0.30"))  # roof = central ROOF_LEN of car length
ROOF_WID    = float(os.environ.get("ROOF_WID", "0.70"))  #        x ROOF_WID of car width
IMG_SIZE    = int(os.environ.get("IMG_SIZE", "640"))
ROT_DEG     = float(os.environ.get("ROT_DEG", "20.0"))
LR          = float(os.environ.get("LR", "0.03"))
TV_WEIGHT   = float(os.environ.get("TV_WEIGHT", "2.5"))
EPOCHS      = int(os.environ.get("EPOCHS", "80"))
SEED        = int(os.environ.get("SEED", "0"))

TRAIN_IMAGES = os.environ.get("TRAIN_IMAGES", "datasets/VisDrone/images/val")
TRAIN_LABELS = os.environ.get("TRAIN_LABELS", "datasets/VisDrone/labels/val")
EVAL_IMAGES  = os.environ.get("EVAL_IMAGES",  "datasets/VisDrone/images/test")
EVAL_LABELS  = os.environ.get("EVAL_LABELS",  "datasets/VisDrone/labels/test")
NUM_TRAIN    = int(os.environ.get("NUM_TRAIN", "100"))
NUM_EVAL     = int(os.environ.get("NUM_EVAL", "0"))       # 0 = ALL test images

SAVE_DIR    = f"patch_examples_{RUN_NAME}"
PATCH_PX    = 64
CONF_THRESH = 0.25
IOU_THRESH  = 0.5
MIN_AREA_FRAC = float(os.environ.get("MIN_AREA_FRAC", "0.001"))   # Pathak III.C

SRC_IDX, TGT_IDX = 0, 1        # car -> van
MARGIN = float(os.environ.get("MARGIN", "5.0"))
TAU    = math.log(CONF_THRESH / (1 - CONF_THRESH))   # detection-threshold logit
CLS_NAMES = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

# canvas fill fraction per shape -- divided out so every shape has equal AREA
FILL = {"square": 1.0, "circle": math.pi/4, "ellipse": math.pi/4, "triangle": 0.5}

_COV = {"cap": 0, "frac_sum": 0.0, "n": 0}
# ==================================================


def shape_mask(shape, px):
    """Boolean mask in the patch canvas, coords u,v in [-1,1]."""
    g = torch.linspace(-1, 1, px, device=DEVICE)
    v, u = torch.meshgrid(g, g, indexing="ij")
    if shape == "square":
        m = torch.ones_like(u)
    elif shape in ("circle", "ellipse"):
        # circular in CANVAS space. For 'ellipse' the footprint itself is
        # stretched to the box's aspect, so a circular canvas mask lands as a
        # true ellipse in image space.
        m = ((u**2 + v**2) <= 1.0).float()
    elif shape == "triangle":
        # apex up: vertices (0,-1), (-1,1), (1,1)
        m = ((v <= 1.0) & (2*u + v >= -1.0) & (-2*u + v >= -1.0)).float()
    else:
        raise ValueError(f"unknown SHAPE={shape}")
    return m.unsqueeze(0)          # [1, px, px]


def shape_boundary(shape, n=64):
    """Extreme points in canvas coords -- used for the overflow cap."""
    if shape == "square":
        return [(-1,-1), (1,-1), (1,1), (-1,1)]
    if shape == "triangle":
        return [(0,-1), (-1,1), (1,1)]
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return list(zip(np.cos(t), np.sin(t)))     # circle / ellipse


def footprint(shape, frac, bwp, bhp):
    """
    Footprint size (pw, ph) in pixels such that the SHAPE's area equals
    frac * box_area. Fill fraction is divided out so shapes are area-matched.
    """
    target = frac * bwp * bhp / FILL[shape]
    if shape == "ellipse":
        # stretch to the box's aspect: pw/ph = bwp/bhp
        pw = math.sqrt(target * bwp / bhp)
        ph = pw * bhp / bwp
    else:
        pw = ph = math.sqrt(target)
    return pw, ph


def cap_to_ellipse(shape, pw, ph, th, bwp, bhp):
    """
    Shrink (pw, ph) so the ROTATED shape fits inside the box's inscribed
    ellipse (semi-axes bwp/2, bhp/2). Returns (pw, ph, was_capped).
    Exact: transform the shape's boundary, find the worst ellipse value,
    scale by 1/sqrt(max) if it exceeds 1.
    """
    a, b = bwp/2.0, bhp/2.0
    ct, st = math.cos(th), math.sin(th)
    worst = 0.0
    for (u, v) in shape_boundary(shape):
        dx = (pw/2)*u*ct - (ph/2)*v*st
        dy = (pw/2)*u*st + (ph/2)*v*ct
        worst = max(worst, (dx/a)**2 + (dy/b)**2)
    if worst <= 1.0:
        return pw, ph, False
    k = 1.0 / math.sqrt(worst)
    return pw*k, ph*k, True


def cap_to_roof(shape, pw, ph, th, bwp, bhp, roof_len, roof_wid):
    """Shrink (pw,ph) so the ROTATED shape fits STRICTLY inside the central roof
    RECTANGLE (roof_len of car length x roof_wid of car width, box-centred). Off the
    windshield/rear-glass/doors by construction. Returns (pw, ph, was_capped)."""
    if bhp >= bwp:                       # long axis vertical
        hx, hy = roof_wid*bwp/2.0, roof_len*bhp/2.0
    else:                                # long axis horizontal
        hx, hy = roof_len*bwp/2.0, roof_wid*bhp/2.0
    ct, st = math.cos(th), math.sin(th)
    worst = 0.0
    for (u, v) in shape_boundary(shape):
        dx = (pw/2)*u*ct - (ph/2)*v*st
        dy = (pw/2)*u*st + (ph/2)*v*ct
        worst = max(worst, abs(dx)/hx, abs(dy)/hy)
    if worst <= 1.0:
        return pw, ph, False
    k = 1.0 / worst
    return pw*k, ph*k, True


def apply_patch(img, patch, boxes, rng):
    """
    Composite the patch onto every car. The AFFINE handles placement, size and
    rotation; the SHAPE rides along as a mask on the patch canvas. OPACITY is
    folded into the mask -- out = img*(1-m*a) + warped*(m*a) is exactly
    alpha-blending inside the shape and untouched outside it.
    """
    _, _, H, W = img.shape
    out = img
    smask = shape_mask(SHAPE, PATCH_PX)
    for (_, xc, yc, bw, bh) in boxes:
        cx, cy = xc*W, yc*H
        bwp, bhp = bw*W, bh*H
        th = math.radians(rng.uniform(-ROT_DEG, ROT_DEG))
        pw, ph = footprint(SHAPE, AREA_FRAC, bwp, bhp)
        capped = False
        if PLACE_MODE == "roof":
            pw, ph, capped = cap_to_roof(SHAPE, pw, ph, th, bwp, bhp, ROOF_LEN, ROOF_WID)
        elif NO_OVERFLOW:
            pw, ph, capped = cap_to_ellipse(SHAPE, pw, ph, th, bwp, bhp)
        if pw < 4 or ph < 4:
            continue
        _COV["cap"] += int(capped)
        _COV["frac_sum"] += (pw*ph*FILL[SHAPE]) / (bwp*bhp)
        _COV["n"] += 1

        ct, st = math.cos(th), math.sin(th)
        # affine_grid maps OUTPUT(image) normalized coords -> INPUT(canvas) coords
        a11, a12 = W*ct/pw,  H*st/pw
        a13 = (2.0/pw) * (ct*(W/2 - cx) + st*(H/2 - cy))
        a21, a22 = -W*st/ph, H*ct/ph
        a23 = (2.0/ph) * (-st*(W/2 - cx) + ct*(H/2 - cy))
        theta = torch.tensor([[[a11, a12, a13], [a21, a22, a23]]],
                             device=img.device, dtype=img.dtype)
        grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
        warped = F.grid_sample(patch.unsqueeze(0), grid, align_corners=False,
                               padding_mode="zeros")
        m = F.grid_sample(smask.unsqueeze(0), grid, align_corners=False,
                          padding_mode="zeros") * OPACITY
        out = out * (1 - m) + warped * m
    return out


def load_car_boxes(label_path):
    out, dropped = [], 0
    if not os.path.exists(label_path):
        return out, dropped
    for line in open(label_path).read().strip().splitlines():
        p = line.split()
        if len(p) < 5 or int(p[0]) != SRC_IDX:
            continue
        xc, yc, w, h = map(float, p[1:5])
        if w*h < MIN_AREA_FRAC:        # Pathak III.C: labels are normalized,
            dropped += 1               # so w*h IS the image-area fraction
            continue
        out.append((SRC_IDX, xc, yc, w, h))
    return out, dropped


def tv_loss(patch):
    dh = (patch[:, :-1, :-1] - patch[:, 1:, :-1]) ** 2
    dw = (patch[:, :-1, :-1] - patch[:, :-1, 1:]) ** 2
    return torch.sqrt(dh + dw + 1e-6).sum() / patch.numel()


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def targeted_loss(raw_out, boxes):
    """
    margin: clamp(car_l - van_l + MARGIN, 0)   -> van beats car
    hinge : clamp(TAU - van_l, 0)              -> van clears the DETECTION bar

    The hinge exists because margin alone is satisfied by annihilating both
    logits: car=-15, van=-8 scores 0 but van's probability is 0.0003 so
    nothing is detected. Destroying a car is easier than building a van, so
    the optimizer takes that road. Adding the hinge doubled flips, 12% -> 22%.

    LOGIT space: cls comes out already sigmoided, and at p=0.95 the gradient
    is p(1-p)=0.05 -- minimizing the probability pushes on the flat tail.
    """
    pred = (raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out)[0]
    cx, cy = pred[0, :] / IMG_SIZE, pred[1, :] / IMG_SIZE
    cls = pred[4:, :]
    src_l, tgt_l = _logit(cls[SRC_IDX, :]), _logit(cls[TGT_IDX, :])
    mar, hin, n = None, None, 0
    for (_, xc, yc, w, h) in boxes:
        m = ((cx >= xc-w/2) & (cx <= xc+w/2) & (cy >= yc-h/2) & (cy <= yc+h/2))
        if m.sum() == 0:
            continue
        mv = torch.clamp(src_l[m] - tgt_l[m] + MARGIN, min=0).sum()
        hv = torch.clamp(TAU - tgt_l[m], min=0).sum()
        mar = mv if mar is None else mar + mv
        hin = hv if hin is None else hin + hv
        n += 1
    if n == 0:
        z = cls.sum() * 0.0
        return z, z, 0
    return mar, hin, n


def detect(yolo, timg):
    arr = (timg[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:, :, ::-1].copy(), verbose=False,
                           conf=CONF_THRESH, device=DEVICE)[0]
    return [(int(b.cls[0]), *b.xyxy[0].tolist(), float(b.conf[0]))
            for b in res.boxes]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def detected_as(dets, cls_id, box):
    return any(c == cls_id and iou(box, (x1, y1, x2, y2)) >= IOU_THRESH
               for c, x1, y1, x2, y2, _ in dets)


def evaluate(yolo, data, patch, rng):
    _COV["cap"] = 0; _COV["frac_sum"] = 0.0; _COV["n"] = 0
    elig = amb = flipped = lost = still = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    for stem, x0, boxes in data:
        clean = detect(yolo, x0)
        if MODE == "none":
            patched = clean
        else:
            with torch.no_grad():
                patched = detect(yolo, apply_patch(x0, patch, boxes, rng).clamp(0, 1))
        for (c, xc, yc, w, h) in boxes:
            bpx = ((xc-w/2)*IMG_SIZE, (yc-h/2)*IMG_SIZE,
                   (xc+w/2)*IMG_SIZE, (yc+h/2)*IMG_SIZE)
            if not detected_as(clean, SRC_IDX, bpx):
                continue
            elig += 1
            if detected_as(clean, TGT_IDX, bpx):    # already van when clean
                amb += 1                             # -> ill-posed, exclude
                continue
            if detected_as(patched, TGT_IDX, bpx):
                flipped += 1
            elif not detected_as(patched, SRC_IDX, bpx):
                lost += 1
            else:
                still += 1
        with open(os.path.join(SAVE_DIR, stem + ".boxes.json"), "w") as f:
            json.dump({"clean": clean, "patched": patched}, f)
    den = elig - amb
    return (flipped/den if den else 0.0), elig, amb, flipped, lost, still


def build(img_dir, lbl_dir, limit):
    out, kept, dropped = [], 0, 0
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        lbl = os.path.join(lbl_dir, os.path.splitext(os.path.basename(ip))[0] + ".txt")
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

    print(f"=== {RUN_NAME} | car->van | SHAPE={SHAPE} | OPACITY={OPACITY} | "
          f"AREA_FRAC={AREA_FRAC} | NO_OVERFLOW={NO_OVERFLOW} | ROT={ROT_DEG} | "
          f"PLACE={PLACE_MODE} | ROOF_LEN={ROOF_LEN} | ROOF_WID={ROOF_WID} | "
          f"LR={LR} | MODE={MODE} ===", flush=True)

    yolo = YOLO(MODEL_PATH)
    raw = YOLO(MODEL_PATH).model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters():
        p.requires_grad_(False)

    eval_set, ev_kept, ev_drop = build(EVAL_IMAGES, EVAL_LABELS, NUM_EVAL)
    print(f"[eval ] {EVAL_IMAGES}: {len(eval_set)} images | {ev_kept} cars | "
          f"{ev_drop} dropped below {MIN_AREA_FRAC*100:.2f}% area", flush=True)

    patch = torch.rand((3, PATCH_PX, PATCH_PX), device=DEVICE,
                       requires_grad=(MODE == "adversarial"))

    if MODE == "adversarial":
        train_set, tr_kept, _ = build(TRAIN_IMAGES, TRAIN_LABELS, NUM_TRAIN)
        print(f"[train] {TRAIN_IMAGES}: {len(train_set)} images | {tr_kept} cars",
              flush=True)
        opt = torch.optim.Adam([patch], lr=LR)
        t0 = time.time()
        for ep in range(EPOCHS):
            tot = ms = hs = ts = 0.0; nb = 0
            for stem, x0, boxes in train_set:
                pimg = apply_patch(x0, patch.clamp(0, 1), boxes, rng).clamp(0, 1)
                lmar, lhin, n = targeted_loss(raw(pimg), boxes)
                if n == 0:
                    continue
                ltv = torch.clamp(TV_WEIGHT*tv_loss(patch.clamp(0, 1)), min=0.1)
                loss = lmar + lhin + ltv
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
                tot += float(loss.detach().cpu()); ms += float(lmar.detach().cpu())
                hs += float(lhin.detach().cpu()); ts += float(ltv.detach().cpu()); nb += 1
            # hinge should START HIGH and FALL. Flat = the patch never trained;
            # anything shallower than about -300 over 80 epochs is suspect.
            print(f"  epoch {ep+1}/{EPOCHS}  total {tot/max(1,nb):8.2f}  "
                  f"margin {ms/max(1,nb):8.2f}  hinge {hs/max(1,nb):8.2f}  "
                  f"tv {ts/max(1,nb):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # SAVE BEFORE EVAL -- a crash at the save step after training has burned a
    # run before. .cpu() on everything.
    final = patch.detach().clamp(0, 1)
    torch.save(final.cpu(), os.path.join(SAVE_DIR, "universal_patch.pt"))
    cv2.imwrite(os.path.join(SAVE_DIR, "universal_patch.png"),
                (final.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)[:, :, ::-1])
    print(f"[save] {SAVE_DIR}/universal_patch.png", flush=True)

    print(f"\nEvaluating on {len(eval_set)} images...", flush=True)
    asr, elig, amb, flipped, lost, still = evaluate(yolo, eval_set, final, rng)
    den = elig - amb

    cov = dict(capped_pct=100*_COV["cap"]/max(1, _COV["n"]),
               mean_realized_frac=_COV["frac_sum"]/max(1, _COV["n"]),
               n=_COV["n"])
    res = dict(run=RUN_NAME, shape=SHAPE, opacity=OPACITY, area_frac=AREA_FRAC,
               no_overflow=NO_OVERFLOW, rot_deg=ROT_DEG, lr=LR, mode=MODE,
               place_mode=PLACE_MODE, roof_len=ROOF_LEN, roof_wid=ROOF_WID,
               epochs=EPOCHS, eval_images=len(eval_set),
               cars_kept=ev_kept, cars_dropped=ev_drop,
               eligible=elig, ambiguous=amb, denominator=den,
               flipped=flipped, lost=lost, still_car=still,
               strict_asr=asr, lost_rate=(lost/den if den else 0.0),
               still_rate=(still/den if den else 0.0), overflow=cov)
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n==== {RUN_NAME} | {SHAPE} | opacity {OPACITY} | {AREA_FRAC*100:.0f}% ====")
    print(f"eval images : {len(eval_set)}   cars kept {ev_kept}  dropped {ev_drop}")
    print(f"eligible {elig}  ambiguous(excl) {amb}  ->  denominator {den}")
    print(f"flipped {flipped}   lost {lost}   still-car {still}")
    print(f"OVERFLOW: capped={cov['capped_pct']:.1f}% "
          f"mean_realized_frac={cov['mean_realized_frac']:.4f} n={cov['n']}")
    print(f"\nstrict ASR : {asr:.3f}   <- HEADLINE")
    print(f"lost rate  : {lost/den if den else 0:.3f}")


if __name__ == "__main__":
    run()
