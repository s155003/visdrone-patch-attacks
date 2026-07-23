"""
TARGETED adversarial attack: make the detector misclassify TRUCK as CAR.

Three white-box attacks (FGSM, PGD, C&W), all targeted:
  - push DOWN the truck-class score
  - push UP the car-class score
for locations where the model detects a truck, so the label flips truck->car.

Model output (5-class): [x, y, w, h, car(0), van(1), truck(2), bus(3), motor(4)]
Raw model output shape: [1, 9, N]  (4 box coords + 5 class scores, N anchors)

USAGE: set SETTINGS, then:  python targeted_attack.py
"""
import os, glob
import numpy as np
import torch, cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "yolo11l_visdrone_pretrained.pt"
VAL_IMAGES = "datasets/VisDrone/images/val"
VAL_LABELS = "datasets/VisDrone/labels/val"    # already 5-class YOLO (0-4)
DEVICE     = "cpu"                # cpu so it doesn't fight GPU training
NUM_IMAGES = 20
IMG_SIZE   = 640

EPS, ALPHA, ITERS = 8/255, 2/255, 40
CW_ITERS, CW_LR = 60, 0.01
CONF_THRESH, IOU_THRESH = 0.25, 0.5

CAR_IDX   = 0    # target class (what we want trucks to become)
TRUCK_IDX = 2    # source class (what we're attacking)
# =================================================

def targeted_loss(raw_out):
    """
    For the raw model output [1, 9, N], rows 4: are the 5 class scores.
    We want to MINIMIZE:  truck_score - car_score  (summed over anchors that
    currently look like trucks), which pushes truck down and car up.
    Focus on anchors where truck is currently a strong prediction.
    """
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    # pred: [1, 9, N] -> class scores are rows 4..8
    cls_scores = pred[:, 4:, :]              # [1, 5, N]
    car_score   = cls_scores[:, CAR_IDX, :]   # [1, N]
    truck_score = cls_scores[:, TRUCK_IDX, :] # [1, N]
    # Only care about anchors where truck is the argmax among classes
    # (i.e. the model currently thinks "truck" here)
    is_truck = (cls_scores.argmax(dim=1) == TRUCK_IDX)  # [1, N] bool
    if is_truck.sum() == 0:
        # no truck-like anchors; return a tiny differentiable zero
        return (truck_score - car_score).sum() * 0.0
    # loss = truck - car over truck anchors -> minimizing flips them toward car
    diff = (truck_score - car_score)         # [1, N]
    return diff[is_truck].sum()

def attack_fgsm(raw, x0):
    x = x0.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = targeted_loss(raw(x))
    g = torch.autograd.grad(loss, x)[0]
    # minimize loss -> step DOWN the gradient
    return (x0 - EPS * g.sign()).clamp(0, 1).detach()

def attack_pgd(raw, x0):
    x_adv = x0.clone().detach()
    for _ in range(ITERS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = targeted_loss(raw(x_adv))
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.max(torch.min(x_adv.detach() - ALPHA * g.sign(), x0 + EPS), x0 - EPS).clamp(0, 1)
    return x_adv.detach()

def attack_cw(raw, x0):
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=CW_LR)
    for _ in range(CW_ITERS):
        x_adv = (x0 + delta).clamp(0, 1)
        with torch.enable_grad():
            loss = targeted_loss(raw(x_adv)) + 0.01 * (delta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)
    return (x0 + delta).clamp(0, 1).detach()

def load_truck_gt(label_path, sx, sy):
    """Return ground-truth TRUCK boxes only, in IMG_SIZE space."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path).read().strip().splitlines():
        if not line: continue
        p = line.split()
        if len(p) < 5: continue
        if int(p[0]) != TRUCK_IDX: continue
        xc, yc, w, h = map(float, p[1:5])
        x1 = (xc - w/2) * IMG_SIZE; y1 = (yc - h/2) * IMG_SIZE
        x2 = (xc + w/2) * IMG_SIZE; y2 = (yc + h/2) * IMG_SIZE
        boxes.append((x1, y1, x2, y2))
    return boxes

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def count_truck_as_car(yolo, tensor_img, truck_gts):
    """How many ground-truth trucks are now detected as CAR at their location?"""
    arr = (tensor_img[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:,:,::-1].copy(), verbose=False, conf=CONF_THRESH)[0]
    dets = [(int(b.cls[0]), *b.xyxy[0].tolist()) for b in res.boxes]
    flipped = 0      # truck GT now detected as car
    still_truck = 0  # truck GT still detected as truck
    for gt in truck_gts:
        best_car = 0.0; best_truck = 0.0
        for cls, x1, y1, x2, y2 in dets:
            i = iou(gt, (x1,y1,x2,y2))
            if cls == CAR_IDX and i > best_car:   best_car = i
            if cls == TRUCK_IDX and i > best_truck: best_truck = i
        if best_car >= IOU_THRESH:   flipped += 1
        elif best_truck >= IOU_THRESH: still_truck += 1
    return flipped, still_truck

def run():
    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)
    raw = raw_yolo.model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters(): p.requires_grad_(False)

    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    # Keep only images that actually contain a truck (class TRUCK_IDX), so
    # NUM_IMAGES counts real truck images, not blank ones.
    images = []
    for ip in all_images:
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        if load_truck_gt(lbl, 1.0, 1.0):    # any truck box present
            images.append(ip)
        if NUM_IMAGES > 0 and len(images) >= NUM_IMAGES:
            break
    print(f"Targeted truck->car attack on {len(images)} truck-containing images (DEVICE={DEVICE})...\n")

    attacks = {"FGSM": attack_fgsm, "PGD": attack_pgd, "C&W": attack_cw}
    # stats: total trucks, and for each attack how many flipped to car
    totals = {"total_trucks": 0,
              "CLEAN_flipped": 0, "CLEAN_truck": 0,
              "FGSM_flipped": 0, "PGD_flipped": 0, "C&W_flipped": 0}

    for idx, ip in enumerate(images):
        orig = cv2.imread(ip)
        if orig is None: continue
        H, W = orig.shape[:2]; sx, sy = IMG_SIZE/W, IMG_SIZE/H
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        truck_gts = load_truck_gt(lbl, sx, sy)
        if not truck_gts:
            continue   # skip images with no trucks
        totals["total_trucks"] += len(truck_gts)

        img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0

        # clean baseline
        f, t = count_truck_as_car(yolo, x0, truck_gts)
        totals["CLEAN_flipped"] += f; totals["CLEAN_truck"] += t

        for name, atk in attacks.items():
            adv = atk(raw, x0)
            f, t = count_truck_as_car(yolo, adv, truck_gts)
            totals[f"{name}_flipped"] += f

        if (idx+1) % 5 == 0:
            print(f"  processed {idx+1}/{len(images)} images...")

    tt = totals["total_trucks"]
    print(f"\n==== TARGETED truck->car RESULTS ====")
    print(f"Total ground-truth trucks tested: {tt}\n")
    if tt == 0:
        print("No trucks found in the sampled images - increase NUM_IMAGES.")
        return
    print(f"{'Setting':8} {'truck->car':>12} {'flip rate':>10}")
    print(f"{'CLEAN':8} {totals['CLEAN_flipped']:12d} {totals['CLEAN_flipped']/tt:10.3f}")
    for name in ["FGSM","PGD","C&W"]:
        fl = totals[f"{name}_flipped"]
        print(f"{name:8} {fl:12d} {fl/tt:10.3f}")
    print("\n'flip rate' = fraction of real trucks the model now labels as car.")
    print("Higher = more successful targeted attack. CLEAN shows the baseline (should be ~0).")

if __name__ == "__main__":
    run()
