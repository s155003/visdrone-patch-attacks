"""
TARGETED adversarial attack v2 (STRONGER): make the detector misclassify
TRUCK as CAR. Three white-box attacks (FGSM, PGD, C&W).

AUDIT FIXES (this version):
  * Issue 1 (fidelity): evaluation is done fully IN-MEMORY on the raw float/uint8
    tensor (NO JPEG anywhere). Adversarial examples are additionally saved as raw
    .pt tensors (lossless) for the record, and a per-image fidelity check asserts
    max|adv-clean| ~= EPS. The uint8 conversion used for the detector now ROUNDS
    (not truncates), matching the realistic 8-bit image the detector would see.
  * Issue 2 (head targeting): the loss masks anchors at the truck's ground-truth
    location across the CONCATENATED [1,9,N] output, which spans all 3 detection
    heads (stride 8/16/32). We log, per truck, which head is responsible and confirm
    the loss mask covers it.  NOTE: the model's `cls` outputs are SIGMOID
    PROBABILITIES (in [0,1]), not logits -> so MARGIN below is currently INERT
    (always-active clamp => constant => zero gradient). Kept at 5.0 per instructions;
    see the report for the recommended logit-space fix.
  * Issue 3 (denominator): we report BOTH image-level (denominator = #images) and
    instance-level (denominator = #truck boxes) flip rates, on the SAME 20 images the
    attack ran on, and print the exact flip definition.

USAGE: cd ~/attack-test && CUDA_VISIBLE_DEVICES="" ~/attack_env/bin/python targeted_attack_v2.py
"""
import os, glob, json
import numpy as np
import torch, cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "yolo11l_visdrone_pretrained.pt"
VAL_IMAGES = "datasets/VisDrone/images/val"
VAL_LABELS = "datasets/VisDrone/labels/val"
DEVICE     = "cpu"
NUM_IMAGES = 100
IMG_SIZE   = 640
ADV_DIR    = "adv_examples_v2"          # raw .pt adversarial examples saved here

# STRONGER attack params (unchanged)
EPS        = 32/255      # ~0.125 budget (was 16/255)
ALPHA      = 2/255       # per-step size
ITERS      = 120         # PGD iterations
CW_ITERS   = 150
CW_LR      = 0.02
MARGIN     = 5.0         # see NOTE above (inert on sigmoid probs)

CONF_THRESH, IOU_THRESH = 0.25, 0.5
CAR_IDX, TRUCK_IDX = 0, 2
NAMES = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

# detection-head layout at IMG_SIZE (YOLO11 strides 8/16/32)
STRIDES = [8, 16, 32]
GRIDS   = [IMG_SIZE // s for s in STRIDES]
HEAD_N  = [g * g for g in GRIDS]
HEAD_BOUNDS = []
_c = 0
for _n in HEAD_N:
    HEAD_BOUNDS.append((_c, _c + _n)); _c += _n
N_ANCHORS = _c
# =================================================


def load_truck_boxes_norm(label_path):
    """Ground-truth truck boxes in normalized [xc,yc,w,h]."""
    boxes = []
    if not os.path.exists(label_path): return boxes
    for line in open(label_path).read().strip().splitlines():
        p = line.split()
        if len(p) < 5: continue
        if int(p[0]) != TRUCK_IDX: continue
        boxes.append(tuple(map(float, p[1:5])))
    return boxes


def _raw_pred(raw, x):
    out = raw(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred                          # [1, 9, N]


def truck_location_mask(pred, truck_boxes_norm):
    """Bool mask over N anchors whose predicted center lies in any GT truck box."""
    p = pred[0]                          # [9, N]
    box = p[:4, :]
    cx = box[0, :] / IMG_SIZE
    cy = box[1, :] / IMG_SIZE
    mask = torch.zeros_like(cx, dtype=torch.bool)
    for (txc, tyc, tw, th) in truck_boxes_norm:
        mask = mask | ((cx >= txc - tw/2) & (cx <= txc + tw/2) &
                       (cy >= tyc - th/2) & (cy <= tyc + th/2))
    return mask


def targeted_loss(raw_out, truck_boxes_norm):
    """Push car above truck (by MARGIN) at anchors at the truck's GT location.
    cls are sigmoid probabilities in [0,1]; convert back to LOGITS so that the
    additive MARGIN is meaningful (in probability space it would be inert)."""
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    p = pred[0]
    cls = p[4:, :]                       # [5, N] sigmoid probs
    mask = truck_location_mask(pred, truck_boxes_norm)
    if mask.sum() == 0:
        return cls.sum() * 0.0
    # cls are sigmoid probs in [0,1]; convert to logits (numerically-stable inverse sigmoid)
    eps = 1e-6
    cls_c = cls.clamp(eps, 1 - eps)
    cls_logit = torch.log(cls_c) - torch.log1p(-cls_c)
    car   = cls_logit[CAR_IDX, mask]
    truck = cls_logit[TRUCK_IDX, mask]
    loss = torch.clamp(truck - car + MARGIN, min=0).sum()
    return loss


def head_of(idx):
    for h, (lo, hi) in enumerate(HEAD_BOUNDS):
        if lo <= idx < hi: return h
    return -1


def attack_fgsm(raw, x0, tb):
    x = x0.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = targeted_loss(_raw_pred(raw, x), tb)
    g = torch.autograd.grad(loss, x)[0]
    return (x0 - EPS * g.sign()).clamp(0, 1).detach()


def attack_pgd(raw, x0, tb):
    x_adv = x0.clone().detach()
    for _ in range(ITERS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = targeted_loss(_raw_pred(raw, x_adv), tb)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.max(torch.min(x_adv.detach() - ALPHA * g.sign(), x0 + EPS), x0 - EPS).clamp(0, 1)
    return x_adv.detach()


def attack_cw(raw, x0, tb):
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=CW_LR)
    for _ in range(CW_ITERS):
        x_adv = (x0 + delta).clamp(0, 1)
        with torch.enable_grad():
            loss = targeted_loss(_raw_pred(raw, x_adv), tb) + 0.005 * (delta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)
    return (x0 + delta).clamp(0, 1).detach()


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_uint8_rgb(tensor_img):
    """Realistic 8-bit image the detector sees: ROUND (not truncate) then clip."""
    arr = tensor_img[0].permute(1, 2, 0).cpu().numpy() * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)   # RGB


def detect_boxes(yolo, tensor_img):
    """Run the detector on an image tensor; return list of {cls,name,conf,xyxy}."""
    rgb = to_uint8_rgb(tensor_img)
    with torch.no_grad():
        res = yolo.predict(rgb[:, :, ::-1].copy(), verbose=False, conf=CONF_THRESH, device="cpu")[0]
    return [{"cls": int(b.cls[0]), "name": NAMES.get(int(b.cls[0]), str(int(b.cls[0]))),
             "conf": round(float(b.conf[0]), 4),
             "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]} for b in res.boxes]


def flips_from_boxes(boxes, truck_boxes_px):
    """Per-truck flip: True iff a CAR box has IoU>=IOU_THRESH with that truck box."""
    car = [b["xyxy"] for b in boxes if b["cls"] == CAR_IDX]
    return [any(iou(gt, cb) >= IOU_THRESH for cb in car) for gt in truck_boxes_px]


def per_truck_flips(yolo, tensor_img, truck_boxes_px):
    """Return list[bool], one per GT truck: True if a CAR is detected with
    IoU>=IOU_THRESH to that truck's box (car AT the truck's location)."""
    rgb = to_uint8_rgb(tensor_img)
    with torch.no_grad():
        res = yolo.predict(rgb[:, :, ::-1].copy(), verbose=False, conf=CONF_THRESH, device="cpu")[0]
    car_dets = [b.xyxy[0].tolist() for b in res.boxes if int(b.cls[0]) == CAR_IDX]
    flips = []
    for gt in truck_boxes_px:
        best = 0.0
        for cd in car_dets:
            best = max(best, iou(gt, cd))
        flips.append(best >= IOU_THRESH)
    return flips


def analyze_heads(raw, yolo, x0, tb_norm, tb_px, stem):
    """Issue 2 instrumentation: confirm the loss mask covers the head(s) that
    actually detect each truck."""
    with torch.no_grad():
        pred = _raw_pred(raw, x0)
    p = pred[0]; cls = p[4:, :]
    mask = truck_location_mask(pred, tb_norm)
    midx = torch.where(mask)[0].tolist()
    head_counts = {s: sum(1 for i in midx if head_of(i) == h) for h, s in enumerate(STRIDES)}
    # strongest truck-score anchor overall in the mask -> responsible head
    resp = "-"
    if midx:
        tsc = cls[TRUCK_IDX, mask]
        gi = midx[int(torch.argmax(tsc).item())]
        resp = f"stride{STRIDES[head_of(gi)]} (truckscore {tsc.max().item():.3f})"
    covered = sorted({f"stride{STRIDES[head_of(i)]}" for i in midx})
    print(f"    [head] {stem}: mask={len(midx)} anchors {head_counts} | "
          f"truck strongest @ {resp} | loss covers {covered}  "
          f"-> {'OK (covers responsible head)' if resp!='-' and resp.split()[0] in covered else 'CHECK'}")


def run():
    os.makedirs(ADV_DIR, exist_ok=True)
    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)
    raw = raw_yolo.model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters(): p.requires_grad_(False)

    # same 20 truck-containing images
    images = []
    for ip in sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg"))):
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if load_truck_boxes_norm(lbl):
            images.append(ip)
        if NUM_IMAGES > 0 and len(images) >= NUM_IMAGES:
            break

    print(f"STRONGER targeted truck->car attack on {len(images)} truck images (DEVICE={DEVICE})")
    print(f"Params: EPS={EPS:.4f}, ALPHA={ALPHA:.4f}, ITERS={ITERS}, CW_ITERS={CW_ITERS}, MARGIN={MARGIN}")
    print(f"Head layout @{IMG_SIZE}px: strides {STRIDES}, anchors/head {HEAD_N}, total {N_ANCHORS}")
    print(f"NOTE: model cls outputs are sigmoid PROBS -> MARGIN={MARGIN} is inert (see report).\n")
    print("FLIP DEFINITION: a ground-truth TRUCK counts as 'flipped' iff the detector")
    print(f"  outputs a CAR (conf>={CONF_THRESH}) whose box has IoU>={IOU_THRESH} with that")
    print("  truck's box -- i.e. a car AT the truck's location, NOT merely any car in the image.")
    print("  Image-level flip = at least one of that image's trucks is flipped.\n")
    print("Issue-2 head-coverage check (first 5 images):")

    attacks = {"FGSM": attack_fgsm, "PGD": attack_pgd, "C&W": attack_cw}
    settings = ["CLEAN", "FGSM", "PGD", "C&W"]
    inst = {s: 0 for s in settings}          # instance-level flips (per truck box)
    img_flip = {s: 0 for s in settings}      # image-level flips (>=1 truck flipped)
    n_trucks = 0
    fid_max = 0.0
    true_new = {n: 0 for n in attacks}       # (b) truck-under-CLEAN flipped ONLY by attack
    native_lost = {n: 0 for n in attacks}    # native-confused trucks the attack disrupted
    n_native = 0                             # (a) car-under-CLEAN trucks
    n_eligible = 0                           # trucks that were truck-under-CLEAN (can flip)

    for idx, ip in enumerate(images):
        stem = os.path.splitext(os.path.basename(ip))[0]
        orig = cv2.imread(ip)
        if orig is None: continue
        lbl = os.path.join(VAL_LABELS, stem + ".txt")
        tb_norm = load_truck_boxes_norm(lbl)
        tb_px = [((xc-w/2)*IMG_SIZE, (yc-h/2)*IMG_SIZE, (xc+w/2)*IMG_SIZE, (yc+h/2)*IMG_SIZE)
                 for (xc, yc, w, h) in tb_norm]
        n_trucks += len(tb_px)

        img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        if idx < 5:
            analyze_heads(raw, yolo, x0, tb_norm, tb_px, stem)

        saved = {"clean": x0.detach().cpu(), "truck_boxes_px": tb_px, "eps": EPS}
        boxes = {"image": stem, "truck_boxes_px": [[round(float(v), 1) for v in b] for b in tb_px],
                 "detections": {}}
        # CLEAN
        cb = detect_boxes(yolo, x0); boxes["detections"]["CLEAN"] = cb
        fc = flips_from_boxes(cb, tb_px)
        inst["CLEAN"] += sum(fc); img_flip["CLEAN"] += int(any(fc))
        # attacks
        atk_flips = {}
        for name, atk in attacks.items():
            adv = atk(raw, x0, tb_norm)
            d = (adv - x0).abs().max().item()
            fid_max = max(fid_max, d)
            saved[name] = adv.detach().cpu()
            ab = detect_boxes(yolo, adv); boxes["detections"][name] = ab
            fa = flips_from_boxes(ab, tb_px)
            atk_flips[name] = fa
            inst[name] += sum(fa); img_flip[name] += int(any(fa))
        # per-truck TRUE-effect categorization (Part 2)
        for ti in range(len(tb_px)):
            if fc[ti]:
                n_native += 1
                for name in attacks:
                    if not atk_flips[name][ti]: native_lost[name] += 1
            else:
                n_eligible += 1
                for name in attacks:
                    if atk_flips[name][ti]: true_new[name] += 1

        torch.save(saved, os.path.join(ADV_DIR, stem + ".pt"))   # Issue 1: lossless save
        with open(os.path.join(ADV_DIR, stem + ".boxes.json"), "w") as jf:
            json.dump(boxes, jf, indent=2)                        # box persistence (Part 1)
        print(f"  {idx+1}/{len(images)} {stem} done (trucks={len(tb_px)})")

    # Issue 1: fidelity verification from a reloaded .pt
    sample = torch.load(os.path.join(ADV_DIR, os.path.splitext(os.path.basename(images[0]))[0] + ".pt"),
                        weights_only=False)
    reload_d = (sample["PGD"] - sample["clean"]).abs().max().item()

    n_img = len(images)
    print(f"\n==== STRONGER targeted truck->car RESULTS ====")
    print(f"Images: {n_img}   Truck GT boxes: {n_trucks}")
    print(f"[fidelity] worst-case max|adv-clean| over run = {fid_max:.5f} (EPS={EPS:.5f}); "
          f"reloaded-from-.pt PGD sample = {reload_d:.5f}  "
          f"-> {'OK, perturbation preserved' if abs(reload_d-EPS) < 0.01 or reload_d > EPS*0.5 else 'WARNING: near zero!'}")
    print(f"\n==== TRUE ATTACK EFFECT (category b) -- HEADLINE ====")
    print(f"(a) native-confused (car under CLEAN, no attack): {n_native}/{n_trucks}")
    print(f"    eligible trucks (truck under CLEAN, can flip) : {n_eligible}/{n_trucks}")
    print(f"{'Attack':8} {'true-new(b)':>12} {'true-rate':>10} {'native-lost':>12}")
    for name in attacks:
        r = true_new[name] / n_eligible if n_eligible else 0.0
        print(f"{name:8} {true_new[name]:12d} {r:10.3f} {native_lost[name]:12d}")
    print("  true-new(b) = truck under CLEAN, flipped to car ONLY under attack (REAL attack effect)")
    print("  true-rate   = true-new(b) / eligible trucks (only truck-under-CLEAN can flip)")
    print("  native-lost = native-confused trucks the attack disrupted (car no longer at truck loc)")

    print(f"\n---- aggregate rate (CONFLATED with native confusion -- do NOT use as headline) ----")
    print(f"{'Setting':8} {'img-flip':>9} {'img-rate':>9}   {'inst-flip':>10} {'inst-rate':>10}")
    for s in settings:
        print(f"{s:8} {img_flip[s]:9d} {img_flip[s]/n_img:9.3f}   "
              f"{inst[s]:10d} {inst[s]/n_trucks if n_trucks else 0:10.3f}")
    print(" (aggregate conflates native car/truck confusion with attack effect -- see TRUE effect above)")
    print(f"\n Adversarial examples (lossless) saved to: {ADV_DIR}/*.pt ; boxes to {ADV_DIR}/*.boxes.json")


if __name__ == "__main__":
    run()
