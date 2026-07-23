"""
TARGETED adversarial attack: make the detector misclassify CAR as BUS.

Three white-box attacks (FGSM, PGD, C&W), all targeted:
  - push DOWN the car-class score (SOURCE)
  - push UP the bus-class score (TARGET)
at the car's ground-truth location, so the label flips car->bus.

Improvements carried over from the truck->car v2 work:
  - Location-based targeting (attack the SOURCE object's GT location, not just
    current-argmax anchors) so the attack keeps its grip as it works.
  - Margin loss in LOGIT space (inverse-sigmoid), so MARGIN is meaningful.
  - True-effect categorization: separates real attack flips from the model's
    native car/bus confusion.
  - Saves adversarial images (.pt) + per-image detection boxes (.boxes.json).

Model output (5-class): [x,y,w,h, car(0),van(1),truck(2),bus(3),motor(4)], shape [1,9,N].

USAGE: cd ~/attack-test && CUDA_VISIBLE_DEVICES="" ~/attack_env/bin/python targeted_attack_car2bus.py
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
NUM_IMAGES = 30          # cars are common; start modest, raise once tuned
IMG_SIZE   = 640
SAVE_DIR   = "adv_examples_car2bus"

# ---- tunable attack params (fresh for car->bus; re-tune as needed) ----
EPS        = 16/255      # perturbation budget (raise toward 24-32/255 if weak)
ALPHA      = 2/255       # PGD per-step size
ITERS      = 120         # PGD iterations
CW_ITERS   = 150
CW_LR      = 0.02
MARGIN     = 5.0         # push target logit above source logit by this margin

CONF_THRESH, IOU_THRESH = 0.25, 0.5

# SOURCE = what we attack, TARGET = what we want it to become
SRC_IDX    = 0    # car  (source: the class we're attacking)
TGT_IDX    = 3    # bus  (target: what we want cars to become)
SRC_NAME, TGT_NAME = "car", "bus"
# =================================================

def load_src_boxes_norm(label_path):
    """Ground-truth SOURCE-class (car) boxes in normalized [xc,yc,w,h]."""
    boxes = []
    if not os.path.exists(label_path): return boxes
    for line in open(label_path).read().strip().splitlines():
        if not line: continue
        p = line.split()
        if len(p) < 5: continue
        if int(p[0]) != SRC_IDX: continue
        boxes.append(tuple(map(float, p[1:5])))
    return boxes

def targeted_loss(raw_out, src_boxes_norm):
    """
    Push TARGET(bus) logit ABOVE SOURCE(car) logit at anchors whose predicted
    box-center falls inside a ground-truth car box. Operates in LOGIT space so
    the margin is meaningful.
    raw_out: [1, 9, N] = [x,y,w,h (pixels), 5 class scores(sigmoid probs)]
    """
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    pred = pred[0]                       # [9, N]
    box = pred[:4, :]                    # [4, N]  x,y,w,h (pixel coords)
    cls = pred[4:, :]                    # [5, N]  sigmoid probs
    cx = box[0, :] / IMG_SIZE
    cy = box[1, :] / IMG_SIZE

    mask = torch.zeros_like(cx, dtype=torch.bool)
    for (sxc, syc, sw, sh) in src_boxes_norm:
        x1, x2 = sxc - sw/2, sxc + sw/2
        y1, y2 = syc - sh/2, syc + sh/2
        mask = mask | ((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))
    if mask.sum() == 0:
        return (cls.sum() * 0.0)

    # convert sigmoid probs -> logits (numerically stable inverse sigmoid)
    e = 1e-6
    cls_logit = torch.log(cls.clamp(e, 1 - e)) - torch.log1p(-cls.clamp(e, 1 - e))
    tgt = cls_logit[TGT_IDX, mask]       # bus: want HIGH
    src = cls_logit[SRC_IDX, mask]       # car: want LOW
    # margin loss: penalize while src + MARGIN > tgt (push target above source)
    loss = torch.clamp(src - tgt + MARGIN, min=0).sum()
    return loss

def attack_fgsm(raw, x0, sb):
    x = x0.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = targeted_loss(raw(x), sb)
    g = torch.autograd.grad(loss, x)[0]
    return (x0 - EPS * g.sign()).clamp(0, 1).detach()

def attack_pgd(raw, x0, sb):
    x_adv = x0.clone().detach()
    for _ in range(ITERS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = targeted_loss(raw(x_adv), sb)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.max(torch.min(x_adv.detach() - ALPHA * g.sign(), x0 + EPS), x0 - EPS).clamp(0, 1)
    return x_adv.detach()

def attack_cw(raw, x0, sb):
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=CW_LR)
    for _ in range(CW_ITERS):
        x_adv = (x0 + delta).clamp(0, 1)
        with torch.enable_grad():
            loss = targeted_loss(raw(x_adv), sb) + 0.005 * (delta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)
    return (x0 + delta).clamp(0, 1).detach()

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def detect(yolo, tensor_img):
    """Return list of (cls, x1,y1,x2,y2, conf) for a tensor image."""
    arr = (tensor_img[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:,:,::-1].copy(), verbose=False, conf=CONF_THRESH, device="cpu")[0]
    return [(int(b.cls[0]), *b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]

def is_target_at(dets, src_box_px):
    """Does a TARGET(bus) detection overlap this source(car) GT box (IoU>=thr)?"""
    best = 0.0
    for cls, x1, y1, x2, y2, conf in dets:
        if cls == TGT_IDX:
            i = iou(src_box_px, (x1,y1,x2,y2))
            if i > best: best = i
    return best >= IOU_THRESH

def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)
    raw = raw_yolo.model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters(): p.requires_grad_(False)

    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    images = []
    for ip in all_images:
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        if load_src_boxes_norm(lbl):
            images.append(ip)
        if NUM_IMAGES > 0 and len(images) >= NUM_IMAGES:
            break
    print(f"TARGETED {SRC_NAME}->{TGT_NAME} attack on {len(images)} {SRC_NAME} images (DEVICE={DEVICE})")
    print(f"Params: EPS={EPS:.4f}, ITERS={ITERS}, MARGIN={MARGIN}\n")

    attacks = {"FGSM": attack_fgsm, "PGD": attack_pgd, "C&W": attack_cw}
    # per-category counters for the TRUE-effect analysis
    n_src = 0                                   # total source(car) GT boxes
    native_tgt = 0                              # (a) already bus under CLEAN
    true_new = {"FGSM":0, "PGD":0, "C&W":0}     # (b) flipped ONLY by attack
    agg = {"CLEAN":0, "FGSM":0, "PGD":0, "C&W":0}  # aggregate (conflated)
    worst_delta = 0.0

    for idx, ip in enumerate(images):
        orig = cv2.imread(ip)
        if orig is None: continue
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        sb_norm = load_src_boxes_norm(lbl)
        sb_px = [((xc-w/2)*IMG_SIZE,(yc-h/2)*IMG_SIZE,(xc+w/2)*IMG_SIZE,(yc+h/2)*IMG_SIZE)
                 for (xc,yc,w,h) in sb_norm]
        n_src += len(sb_px)

        img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0

        # clean detections + per-source native status
        clean_dets = detect(yolo, x0)
        clean_is_tgt = [is_target_at(clean_dets, b) for b in sb_px]   # already-bus?
        native_tgt += sum(clean_is_tgt)
        agg["CLEAN"] += sum(clean_is_tgt)

        stem = os.path.splitext(os.path.basename(ip))[0]
        save = {"clean": x0.cpu(), "src_boxes_px": sb_px, "eps": EPS}
        boxes_rec = {"clean": clean_dets}

        for name, atk in attacks.items():
            adv = atk(raw, x0, sb_norm)
            worst_delta = max(worst_delta, (adv - x0).abs().max().item())
            adv_dets = detect(yolo, adv)
            adv_is_tgt = [is_target_at(adv_dets, b) for b in sb_px]
            agg[name] += sum(adv_is_tgt)
            # (b) true-new: was NOT bus under clean, IS bus under attack
            for c_clean, c_adv in zip(clean_is_tgt, adv_is_tgt):
                if (not c_clean) and c_adv:
                    true_new[name] += 1
            save[name] = adv.cpu()
            boxes_rec[name] = adv_dets

        torch.save(save, os.path.join(SAVE_DIR, stem + ".pt"))
        with open(os.path.join(SAVE_DIR, stem + ".boxes.json"), "w") as f:
            json.dump(boxes_rec, f)
        print(f"  {idx+1}/{len(images)} {stem} done ({SRC_NAME}s={len(sb_px)})")

    eligible = n_src - native_tgt   # source boxes actually eligible to be flipped
    print(f"\n==== TARGETED {SRC_NAME}->{TGT_NAME} RESULTS ====")
    print(f"Images: {len(images)}   {SRC_NAME} GT boxes: {n_src}")
    print(f"[fidelity] worst-case max|adv-clean| = {worst_delta:.5f} (EPS={EPS:.5f})")
    print(f"(a) already-{TGT_NAME} under CLEAN (native): {native_tgt}")
    print(f"    eligible {SRC_NAME}s (correctly not-{TGT_NAME} under CLEAN): {eligible}\n")
    print(f"{'Setting':8} {'agg':>6} {'true-new(b)':>12} {'true-rate':>10}")
    print(f"{'CLEAN':8} {agg['CLEAN']:6d} {'--':>12} {'--':>10}   <- native confusion baseline")
    for name in ["FGSM","PGD","C&W"]:
        tr = true_new[name]/eligible if eligible else 0.0
        print(f"{name:8} {agg[name]:6d} {true_new[name]:12d} {tr:10.3f}")
    print(f"\ntrue-new(b) = {SRC_NAME}s that were NOT {TGT_NAME} under CLEAN but became {TGT_NAME} under attack.")
    print(f"true-rate = true-new / eligible {SRC_NAME}s.  This is the HEADLINE (real attack effect).")
    print(f"'agg' = raw count detected as {TGT_NAME} (conflates native confusion - do NOT use as headline).")

if __name__ == "__main__":
    run()
