"""
HIDING adversarial patch attack — replication of Pathak et al. (IROS 2024)
Sec. III.C, which follows Thys et al. (CVPRW 2019) Sec. 3.

TASK: make vehicles UNDETECTABLE (hiding / concealment).
      NOT the car->van targeted class-flip of patch_attack_ms.py.
      Different loss, different sizing, different metric, different denominator.

Interfaces below (model path, data paths, detect(), iou(), tensor layout,
normalized-coord matcher) are COPIED FROM patch_attack_ms.py, not guessed.

USAGE: cd ~/attack-test && RUN_NAME=h0a_gray_640 MODE=gray \
       ~/attack_env/bin/python -u patch_attack_hide.py

WHAT MATCHES THE PAPER
  - 64x64 patch canvas, random init                        (Pathak III.C)
  - L = alpha*L_nps + beta*L_tv + L_obj                     (Thys 3)
  - Adam, network frozen, patch pixels are the variables    (Thys 3)
  - Train sizing: patch area ~ U(15%,35%) of bbox area      (Pathak III.C)
  - Eval  sizing: patch area = 20% of bbox area, fixed      (Pathak III.C)
  - Patch centred on the bbox                               (Thys 3.2)
  - flip H/V, hue +-0.08, contrast [0.5,1.5], saturation
    [0.5,1.5], brightness +-0.3, noise +-0.1, rot +-20 deg  (Pathak III.C)
  - 640x640 input                                           (Pathak III.A)
  - objects < 0.1% of ORIGINAL image area dropped           (Pathak III.C)
  - gray / random patch baselines, same size+scale+rotation (Pathak III.D)
  - 4 classes: car, van, truck, bus (motor excluded)        (Pathak III.A)

DEVIATIONS — state these in the writeup
  1. NO OBJECTNESS CHANNEL. Thys minimizes max(p_obj); YOLOv2 emits objectness
     as its own scalar. YOLO11 is anchor-free: the raw tensor is [1,9,N] =
     4 box coords + 5 class scores, no objectness. A detection exists iff some
     class clears threshold, so we minimize the max class score. All 5 classes
     are vehicles, so there is no escape class to flip into — class suppression
     IS hiding here. Forced by the architecture; the single biggest deviation.
  2. Thys takes max over the WHOLE IMAGE (Inria: 1-2 persons). VisDrone has
     ~15 vehicles/image, so a global max trains against one object and ignores
     the rest. We take per-GT-box max, summed over boxes.
  3. Stock YOLO11-L keeps its coarsest feature level; Pathak discards it and
     adds a high-res level on every detector. Not replicated.
  4. ASR is single-threshold (see hiding_asr). Pathak averages ASR over all
     recall thresholds per class, then across classes — an AP-style
     integration. Ours is a strict subset. Report the difference.
  5. TV uses Thys's exact formula, sqrt((dx)^2+(dy)^2), NOT the L1-mean tv_loss
     in patch_attack_ms.py. Different scale => TV_WEIGHT here is NOT comparable
     to that script's 0.05.

HARD RULES honored: DEVICE="cuda"; every .numpy() preceded by .cpu().
"""

import os, glob, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
# paths/model VERIFIED against patch_attack_ms.py — run from ~/attack-test
MODEL_PATH = "yolo11l_visdrone_pretrained.pt"
VAL_IMAGES = "datasets/VisDrone/images/val"
VAL_LABELS = "datasets/VisDrone/labels/val"
TRAIN_IMAGES    = os.environ.get("TRAIN_IMAGES",    "datasets/VisDrone/images/val")
TRAIN_LABELS    = os.environ.get("TRAIN_LABELS",    "datasets/VisDrone/labels/val")
EVAL_IMAGES_DIR = os.environ.get("EVAL_IMAGES_DIR", "datasets/VisDrone/images/val")
EVAL_LABELS_DIR = os.environ.get("EVAL_LABELS_DIR", "datasets/VisDrone/labels/val")
PRINT_FILE = "30values.txt"          # from gitlab.com/EAVISE/adversarial-yolo
DEVICE     = "cuda"

RUN_NAME   = os.environ.get("RUN_NAME", "hide_test")
MODE       = os.environ.get("MODE", "adversarial")     # adversarial|gray|random
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "640"))
NPS_WEIGHT = float(os.environ.get("NPS_WEIGHT", "0.01"))   # alpha  (0 = ablate)
TV_WEIGHT  = float(os.environ.get("TV_WEIGHT", "2.5"))     # beta
EPOCHS     = int(os.environ.get("EPOCHS", "80"))
# TRAIN and EVAL image counts are INDEPENDENT. The patch is universal, so 100
# images trains it fine; eval is inference-only and costs no backprop. h0a
# showed why this matters: on 100 images bus had n=3, and Pathak's metric
# weights bus equally with 883 cars, so one bus swung the headline by 0.083.
# EVAL_IMAGES=0 -> use every val image (proper sample for the rare classes).
NUM_IMAGES  = int(os.environ.get("NUM_IMAGES", "100"))
EVAL_IMAGES = int(os.environ.get("EVAL_IMAGES", "0"))
SAVE_DIR   = f"patch_examples_{RUN_NAME}"

PATCH_PX    = 64                 # Pathak III.C
LR          = 0.03
CONF_THRESH = 0.25
IOU_THRESH  = 0.5
SEED        = 0

# 0=car 1=van 2=truck 3=bus 4=motor. Pathak uses 4 classes, motor excluded.
TARGET_CLASSES = [0, 1, 2, 3]
CLS_NAMES      = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

MIN_AREA_FRAC   = 0.001          # Pathak III.C: <0.1% of ORIGINAL image area.
                                 # labels are normalized, so bw*bh IS that
                                 # fraction — resize-invariant, no need for
                                 # original pixel dims.
TRAIN_AREA_FRAC = (0.15, 0.35)   # fraction of BBOX AREA, resampled per object
EVAL_AREA_FRAC  = 0.20           # fixed
# NOTE: area-fraction sizing. patch_attack_ms.py sizes by paint-detection +
# COVER_FLOOR, which is not convertible to this without the aspect ratio.

HUE_DELTA  = 0.08                # fraction of a full 2pi rotation -> +-28.8 deg
CONTRAST   = (0.5, 1.5)
SATURATION = (0.5, 1.5)
BRIGHTNESS = 0.3
NOISE      = 0.1
ROT_DEG    = 20.0
# ==================================================


# ---------------------------------------------------------------- data

def load_boxes_norm(label_path):
    """[(cls, xc, yc, w, h)] normalized, target classes only, 0.1% filter."""
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
        if c not in TARGET_CLASSES:
            continue
        xc, yc, w, h = map(float, p[1:5])
        if w * h < MIN_AREA_FRAC:        # Pathak III.C
            dropped += 1
            continue
        out.append((c, xc, yc, w, h))
    return out, dropped


# ---------------------------------------------------------------- NPS / TV

def load_printability(path, side):
    """Thys 30values.txt: one 'r,g,b' per line, floats in [0,1] -> [K,3,s,s]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"printability file not found: {path}\n"
            f"  fetch non_printability/30values.txt from\n"
            f"  https://gitlab.com/EAVISE/adversarial-yolo\n"
            f"  or set NPS_WEIGHT=0 to run the ablation without it.")
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r, g, b = [float(v) for v in line.replace(" ", "").split(",")]
        rows.append(np.stack([np.full((side, side), r),
                              np.full((side, side), g),
                              np.full((side, side), b)]))
    arr = np.asarray(rows, dtype=np.float32)
    print(f"[nps] {arr.shape[0]} printable colours from {path}")
    return torch.tensor(arr, device=DEVICE)


def nps_loss(patch, palette):
    """L_nps = sum_p min_c ||p - c||  (Thys 3 / Sharif). Normalized by numel."""
    d = (patch.unsqueeze(0) - palette + 1e-6) ** 2       # [K,3,H,W]
    d = torch.sqrt(d.sum(1) + 1e-6)                      # [K,H,W]
    return d.min(0)[0].sum() / patch.numel()


def tv_loss_thys(patch):
    """L_tv = sum sqrt((p_ij-p_i+1,j)^2 + (p_ij-p_i,j+1)^2)   (Thys 3)."""
    dh = (patch[:, :-1, :-1] - patch[:, 1:, :-1]) ** 2
    dw = (patch[:, :-1, :-1] - patch[:, :-1, 1:]) ** 2
    return torch.sqrt(dh + dw + 1e-6).sum() / patch.numel()


# ---------------------------------------------------------------- transforms

def rgb_hue_rotate(x, theta):
    """Hue rotation as a linear YIQ-space rotation — differentiable.
    An RGB->HSV->RGB round trip is not (it branches on argmax)."""
    c, s = math.cos(theta), math.sin(theta)
    m = torch.tensor([
        [0.299 + 0.701*c + 0.168*s, 0.587 - 0.587*c + 0.330*s,
         0.114 - 0.114*c - 0.497*s],
        [0.299 - 0.299*c - 0.328*s, 0.587 + 0.413*c + 0.035*s,
         0.114 - 0.114*c + 0.292*s],
        [0.299 - 0.300*c + 1.250*s, 0.587 - 0.588*c - 1.050*s,
         0.114 + 0.886*c - 0.203*s],
    ], device=x.device, dtype=x.dtype)
    return torch.einsum("ij,jhw->ihw", m, x)


def jitter(patch, rng):
    """Pathak III.C stack, in the order the paper lists them."""
    x = patch
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[1])
    x = rgb_hue_rotate(x, rng.uniform(-HUE_DELTA, HUE_DELTA) * 2 * math.pi)
    c = rng.uniform(*CONTRAST)
    x = (x - x.mean()) * c + x.mean()
    s = rng.uniform(*SATURATION)
    g = (0.299*x[0] + 0.587*x[1] + 0.114*x[2]).unsqueeze(0)
    x = g + (x - g) * s
    x = x + rng.uniform(-BRIGHTNESS, BRIGHTNESS)
    x = x + torch.empty_like(x).uniform_(-NOISE, NOISE)   # noise: const wrt patch
    return x.clamp(0, 1)


def apply_patch(img, patch, boxes_norm, area_frac, rng):
    """
    Paste `patch` on every box, sized by FRACTION OF BOX AREA, box-centred,
    rotated +-ROT_DEG. One affine grid_sample per object; gradients reach the
    patch. Rotating the MASK with the texture tilts the FOOTPRINT, not just the
    pattern (the R12->R13 lesson).

    img        : [1,3,H,W] in [0,1]
    boxes_norm : [(cls, xc, yc, w, h)] normalized
    """
    _, _, H, W = img.shape
    out = img
    for (_, xc, yc, bw, bh) in boxes_norm:
        cx, cy = xc * W, yc * H
        bwp, bhp = bw * W, bh * H
        frac = (rng.uniform(*area_frac) if isinstance(area_frac, tuple)
                else area_frac)
        s = math.sqrt(frac * bwp * bhp)          # square patch: s^2 = frac*area
        if s < 4:
            continue
        p = jitter(patch, rng) if MODE == "adversarial" else patch
        th = math.radians(rng.uniform(-ROT_DEG, ROT_DEG))
        ct, st = math.cos(th), math.sin(th)
        # affine_grid maps OUTPUT(image) normalized coords -> INPUT(patch) coords.
        #   X = cx + (s/2)(u*cos - v*sin);  Y = cy + (s/2)(u*sin + v*cos)
        # inverted, with x_n = 2X/W - 1:
        a11, a12 = W * ct / s,  H * st / s
        a13 = (2.0 / s) * (ct * (W/2 - cx) + st * (H/2 - cy))
        a21, a22 = -W * st / s, H * ct / s
        a23 = (2.0 / s) * (-st * (W/2 - cx) + ct * (H/2 - cy))
        theta = torch.tensor([[[a11, a12, a13], [a21, a22, a23]]],
                             device=img.device, dtype=img.dtype)
        grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
        warped = F.grid_sample(p.unsqueeze(0), grid, align_corners=False,
                               padding_mode="zeros")
        mask = F.grid_sample(torch.ones_like(p[:1]).unsqueeze(0), grid,
                             align_corners=False, padding_mode="zeros")
        out = out * (1 - mask) + warped * mask
    return out


# ---------------------------------------------------------------- model/loss

def _logit(p):
    e = 1e-6
    p = p.clamp(e, 1 - e)
    return torch.log(p) - torch.log1p(-p)


def hide_loss(raw_out, boxes_norm):
    """
    L_obj adapted. raw_out[0] -> [1,9,N]; pred[:4]=cx,cy,w,h in IMG_SIZE px;
    pred[4:] = 5 class scores, ALREADY SIGMOIDED.

    Per GT box: anchors whose predicted centre lies inside the box (the same
    normalized-coord matcher targeted_loss uses), take max class score over
    classes, then max over those anchors — the single most confident detection
    of that object — and minimize it IN LOGIT SPACE.

    Logit space matters: at p=0.95 the sigmoid gradient is p(1-p)=0.05, so
    minimizing the probability directly pushes on the flat tail. Same trap that
    made the car->van margin loss inert in probability space.
    """
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    pred = pred[0]                                  # [9, N]
    cx = pred[0, :] / IMG_SIZE
    cy = pred[1, :] / IMG_SIZE
    cls = pred[4:, :]                               # [5, N] sigmoided
    conf_logit = _logit(cls.max(0)[0])              # [N]

    total, n = None, 0
    for (_, xc, yc, w, h) in boxes_norm:
        m = ((cx >= xc - w/2) & (cx <= xc + w/2) &
             (cy >= yc - h/2) & (cy <= yc + h/2))
        if m.sum() == 0:
            continue
        v = conf_logit[m].max()                     # per-box max
        total = v if total is None else total + v   # summed over boxes
        n += 1
    if n == 0:
        return cls.sum() * 0.0, 0
    return total, n


def detect(yolo, timg):
    """Copied from patch_attack_ms.py. Returns [(cls, x1, y1, x2, y2, conf)]."""
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


def correctly_detected(dets, cls_id, box_px):
    """Right class AND IoU >= 0.5. dets are (cls,x1,y1,x2,y2,conf)."""
    for c, x1, y1, x2, y2, _ in dets:
        if c == cls_id and iou(box_px, (x1, y1, x2, y2)) >= IOU_THRESH:
            return True
    return False


# ---------------------------------------------------------------- eval

def hiding_asr(yolo, data, patch, rng):
    """
    Pathak III.D: ASR = ratio of correctly detected objects lost to the patch.
      eligible = objects correctly detected in the CLEAN image
      hidden   = eligible objects NOT correctly detected in the PATCHED image
      ASR      = hidden / eligible

    NOTE THE DENOMINATOR: detected-clean, NOT all GT. The car->van runs use
    GT-eligible cars. The two ASRs are NOT interchangeable.
    """
    elig = hidden = 0
    per_cls = {c: [0, 0] for c in TARGET_CLASSES}
    os.makedirs(os.path.join(SAVE_DIR, "annotated"), exist_ok=True)
    for stem, x0, boxes in data:
        clean = detect(yolo, x0)
        with torch.no_grad():
            pimg = apply_patch(x0, patch, boxes, EVAL_AREA_FRAC, rng).clamp(0, 1)
        patched = detect(yolo, pimg)
        for (c, xc, yc, w, h) in boxes:
            bpx = ((xc-w/2)*IMG_SIZE, (yc-h/2)*IMG_SIZE,
                   (xc+w/2)*IMG_SIZE, (yc+h/2)*IMG_SIZE)
            if not correctly_detected(clean, c, bpx):
                continue
            elig += 1
            per_cls[c][1] += 1
            if not correctly_detected(patched, c, bpx):
                hidden += 1
                per_cls[c][0] += 1
        with open(os.path.join(SAVE_DIR, stem + ".boxes.json"), "w") as f:
            json.dump({"clean": clean, "patched": patched}, f)
    pooled = hidden / elig if elig else 0.0
    # Pathak III.D: "we report the ASR as the MEAN of average ASR ACROSS ALL
    # CLASSES" — equal weight per class, NOT pooled. VisDrone is ~88% car and
    # car is the hardest class to hide (their Fig 2b: the detector is biased
    # toward the majority class), so pooling collapses to the car number and
    # systematically understates. Classes with zero eligible objects are
    # excluded rather than counted as 0.
    present = [c for c in TARGET_CLASSES if per_cls[c][1] > 0]
    cls_mean = (sum(per_cls[c][0] / per_cls[c][1] for c in present) /
                len(present)) if present else 0.0
    return pooled, cls_mean, elig, hidden, per_cls


# ---------------------------------------------------------------- main

def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = random.Random(SEED)

    print(f"=== {RUN_NAME} | MODE={MODE} | IMG_SIZE={IMG_SIZE} | "
          f"NPS={NPS_WEIGHT} | TV={TV_WEIGHT} | EPOCHS={EPOCHS} ===", flush=True)

    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)
    raw = raw_yolo.model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters():
        p.requires_grad_(False)          # freeze the network (Thys 3)

    def collect(img_dir, lbl_dir, limit):
        imgs, dropped, kept = [], 0, 0
        for ip in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            lbl = os.path.join(lbl_dir,
                               os.path.splitext(os.path.basename(ip))[0] + ".txt")
            boxes, d = load_boxes_norm(lbl)
            dropped += d
            if not boxes:
                continue
            imgs.append((ip, boxes))
            kept += len(boxes)
            if limit > 0 and len(imgs) >= limit:
                break
        return imgs, dropped, kept

    train_imgs, tr_drop, tr_kept = collect(TRAIN_IMAGES,    TRAIN_LABELS,    NUM_IMAGES)
    eval_imgs,  ev_drop, ev_kept = collect(EVAL_IMAGES_DIR, EVAL_LABELS_DIR, EVAL_IMAGES)

    def preload(subset):
        out = []
        for ip, boxes in subset:
            orig = cv2.imread(ip)
            if orig is None:
                continue
            img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
            x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
                  .float().unsqueeze(0) / 255.0).to(DEVICE)
            out.append((os.path.splitext(os.path.basename(ip))[0], x0, boxes))
        return out

    train_set = preload(train_imgs)
    eval_set  = preload(eval_imgs)
    n_by_cls = {c: 0 for c in TARGET_CLASSES}
    for _, _, bs in eval_set:
        for (c, *_r) in bs:
            n_by_cls[c] += 1
    print(f"[data] TRAIN split: {TRAIN_IMAGES} | {len(train_set)} imgs "
          f"(NUM_IMAGES={NUM_IMAGES}) | kept {tr_kept} dropped {tr_drop}", flush=True)
    print(f"[data] EVAL  split: {EVAL_IMAGES_DIR} | {len(eval_set)} imgs "
          f"(EVAL_IMAGES={EVAL_IMAGES}) | kept {ev_kept} dropped {ev_drop} "
          f"below {MIN_AREA_FRAC*100:.1f}% area", flush=True)
    print("[data] eval objects per class: " +
          "  ".join(f"{CLS_NAMES[c]}={n_by_cls[c]}" for c in TARGET_CLASSES) +
          "   <-- class-mean ASR is only as stable as the smallest of these",
          flush=True)
    data = train_set

    if MODE == "gray":
        patch = torch.full((3, PATCH_PX, PATCH_PX), 0.5, device=DEVICE)
    elif MODE == "random":
        patch = torch.rand((3, PATCH_PX, PATCH_PX), device=DEVICE)
    else:
        patch = torch.rand((3, PATCH_PX, PATCH_PX), device=DEVICE,
                           requires_grad=True)

    if MODE == "adversarial":
        palette = load_printability(PRINT_FILE, PATCH_PX) if NPS_WEIGHT > 0 else None
        opt = torch.optim.Adam([patch], lr=LR)          # Thys 3
        t0 = time.time()
        for ep in range(EPOCHS):
            tot, det_a, nps_a, tv_a, nb = 0.0, 0.0, 0.0, 0.0, 0
            for stem, x0, boxes in data:
                pimg = apply_patch(x0, patch.clamp(0, 1), boxes,
                                   TRAIN_AREA_FRAC, rng).clamp(0, 1)
                lobj, n = hide_loss(raw(pimg), boxes)
                if n == 0:
                    continue
                loss = lobj
                nps_term = torch.zeros((), device=patch.device)
                tv_term  = torch.zeros((), device=patch.device)
                if NPS_WEIGHT > 0:
                    nps_term = NPS_WEIGHT * nps_loss(patch.clamp(0, 1), palette)
                    loss = loss + nps_term
                if TV_WEIGHT > 0:
                    # Thys's real loss floors the TV term: max(2.5*tv, 0.1).
                    # Past that floor TV stops producing gradient, so it does
                    # not keep flattening the patch and fighting the attack.
                    tv_term = torch.clamp(
                        TV_WEIGHT * tv_loss_thys(patch.clamp(0, 1)), min=0.1)
                    loss = loss + tv_term
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
                tot   += float(loss.detach().cpu())
                det_a += float(lobj.detach().cpu())
                nps_a += float(nps_term.detach().cpu())
                tv_a  += float(tv_term.detach().cpu())
                nb += 1
            d = max(1, nb)
            print(f"  epoch {ep+1}/{EPOCHS}  total {tot/d:.3f}  det {det_a/d:.3f}  "
                  f"nps {nps_a/d:.4f}  tv {tv_a/d:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- SAVE BEFORE EVAL: a crash at the save step after training completed
    # ---- has burned a run before. .cpu() on everything.
    final = patch.detach().clamp(0, 1)
    torch.save(final.cpu(), os.path.join(SAVE_DIR, "universal_patch.pt"))
    pimg = (final.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(os.path.join(SAVE_DIR, "universal_patch.png"), pimg)
    print(f"[save] patch written to {SAVE_DIR}/", flush=True)

    print(f"\nEvaluating on {len(eval_set)} images (clean vs patched)...",
          flush=True)
    pooled, cls_mean, elig, hidden, per_cls = hiding_asr(yolo, eval_set,
                                                         final, rng)

    res = dict(run=RUN_NAME, mode=MODE, img_size=IMG_SIZE,
               nps_weight=NPS_WEIGHT, tv_weight=TV_WEIGHT, epochs=EPOCHS,
               eval_area_frac=EVAL_AREA_FRAC,
               train_images=len(train_set), eval_images=len(eval_set),
               objects_kept=ev_kept, objects_dropped=ev_drop,
               eligible_detected_clean=elig, hidden=hidden,
               hiding_asr_class_mean=cls_mean,   # <- Pathak's definition
               hiding_asr_pooled=pooled,         # <- car-dominated
               per_class={CLS_NAMES[k]: dict(hidden=v[0], eligible=v[1],
                          asr=(v[0]/v[1] if v[1] else 0.0))
                          for k, v in per_cls.items()})
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n==== HIDING PATCH RESULTS — {RUN_NAME} ====")
    print(f"train images: {len(train_set)}   eval images: {len(eval_set)}")
    print(f"objects kept: {ev_kept}   dropped by 0.1% filter: {ev_drop}")
    print(f"eligible (correctly detected CLEAN): {elig}   hidden: {hidden}")
    for k, v in per_cls.items():
        flag = "  <-- small n, unstable" if 0 < v[1] < 30 else ""
        print(f"   {CLS_NAMES[k]:6s} {v[0]:5d}/{v[1]:5d} = "
              f"{(v[0]/v[1] if v[1] else 0):.3f}{flag}")
    print(f"\nhiding ASR (class-mean, Pathak III.D): {cls_mean:.3f}   <- HEADLINE")
    print(f"hiding ASR (pooled, car-dominated)   : {pooled:.3f}")
    print(f"\nSaved to {SAVE_DIR}/")


if __name__ == "__main__":
    run()
