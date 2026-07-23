"""
Evaluate adversarial attacks (FGSM, PGD, C&W) on a fine-tuned VisDrone YOLO model.

Attack Success Rate (ASR) = 1 - Accuracy,  where  Accuracy = TP / (TP + FP + FN),
measured against VisDrone ground-truth annotations.

USAGE:  set the paths in SETTINGS, then:  python attack_eval.py
"""

import os
import glob
import numpy as np
import torch
import cv2
from ultralytics import YOLO

# ==================== SETTINGS (edit these) ====================
MODEL_PATH = "best.pt"
VAL_IMAGES = "/home/aarav/Downloads/VisDrone2019-DET-val/images"
VAL_ANNOTS = "/home/aarav/Downloads/VisDrone2019-DET-val/annotations"
DEVICE     = "cuda"          # "cuda" if GPU available, else "cpu"
NUM_IMAGES = -1              # number of val images to use (-1 = all 548)
IMG_SIZE   = 640

EPS      = 8.0 / 255.0       # perturbation budget (FGSM & PGD)
ALPHA    = 2.0 / 255.0       # PGD step size
ITERS    = 30                # PGD iterations
CW_ITERS = 30                # C&W iterations
CW_LR    = 0.01              # C&W learning rate
CW_C     = 0.01              # C&W perturbation-size penalty weight

IOU_THRESH  = 0.5            # IoU to count a detection as a True Positive
CONF_THRESH = 0.25          # detection confidence threshold
# ==============================================================


def load_ground_truth(annot_path, scale_x, scale_y):
    """VisDrone annotation -> list of (class, x1,y1,x2,y2) in 640-space.
    model_class = visdrone_category - 1; skip ignored(0)/others(11)/score0."""
    gts = []
    if not os.path.exists(annot_path):
        return gts
    with open(annot_path) as f:
        for line in f.read().strip().splitlines():
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            x, y, w, h = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            score, cat = int(p[4]), int(p[5])
            if score == 0 or cat == 0 or cat == 11:
                continue
            cls = cat - 1
            gts.append((cls, x * scale_x, y * scale_y,
                        (x + w) * scale_x, (y + h) * scale_y))
    return gts


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_counts(preds, gts):
    """preds: [(cls,x1,y1,x2,y2,conf)], gts: [(cls,x1,y1,x2,y2)] -> TP, FP, FN."""
    matched = set()
    tp = 0
    for pred in sorted(preds, key=lambda z: -z[5]):
        pcls = pred[0]
        best_i, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched or gt[0] != pcls:
                continue
            i = iou(pred[1:5], gt[1:5])
            if i > best_i:
                best_i, best_j = i, j
        if best_i >= IOU_THRESH and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gts) - len(matched)
    return tp, fp, fn


def raw_confidence(raw_out):
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    return pred[:, 4:, :].max(dim=1)[0].sum()


# ---------------- Attacks (use RAW model, gradient-based) ----------------
def attack_fgsm(raw, x0):
    x = x0.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = raw_confidence(raw(x))
    g = torch.autograd.grad(loss, x)[0]
    return (x0 - EPS * g.sign()).clamp(0, 1).detach()

def attack_pgd(raw, x0):
    x_adv = x0.clone().detach()
    for _ in range(ITERS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = raw_confidence(raw(x_adv))
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.max(torch.min(x_adv.detach() - ALPHA * g.sign(),
                                    x0 + EPS), x0 - EPS).clamp(0, 1)
    return x_adv.detach()

def attack_cw(raw, x0):
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=CW_LR)
    for _ in range(CW_ITERS):
        x_adv = (x0 + delta).clamp(0, 1)
        with torch.enable_grad():
            loss = raw_confidence(raw(x_adv)) + CW_C * (delta ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)
    return (x0 + delta).clamp(0, 1).detach()


def predict_boxes(yolo, tensor_img):
    arr = (tensor_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:, :, ::-1].copy(), verbose=False, conf=CONF_THRESH)[0]
    return [(int(b.cls[0]), *b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]


def evaluate():
    yolo = YOLO(MODEL_PATH)                     # for detection/counting
    raw_yolo = YOLO(MODEL_PATH)                 # SEPARATE instance for attacks (avoids tensor sharing)
    raw = raw_yolo.model.float().to(DEVICE)     # for gradient attacks
    raw.train(False)
    for p in raw.parameters():
        p.requires_grad_(False)

    images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    if NUM_IMAGES > 0:
        images = images[:NUM_IMAGES]
    print(f"Evaluating on {len(images)} images...\n")

    attacks = {"FGSM": attack_fgsm, "PGD": attack_pgd, "C&W": attack_cw}
    stats = {k: [0, 0, 0] for k in ["CLEAN", "FGSM", "PGD", "C&W"]}

    for idx, img_path in enumerate(images):
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        H, W = orig.shape[:2]
        sx, sy = IMG_SIZE / W, IMG_SIZE / H
        annot = os.path.join(VAL_ANNOTS,
                             os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        gts = load_ground_truth(annot, sx, sy)

        img640 = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = torch.from_numpy(img640[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x0 = x0.to(DEVICE)

        tp, fp, fn = match_counts(predict_boxes(yolo, x0), gts)
        stats["CLEAN"][0] += tp; stats["CLEAN"][1] += fp; stats["CLEAN"][2] += fn

        for name, atk in attacks.items():
            x_adv = atk(raw, x0)
            tp, fp, fn = match_counts(predict_boxes(yolo, x_adv), gts)
            stats[name][0] += tp; stats[name][1] += fp; stats[name][2] += fn

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(images)} done")

    print("\n==== RESULTS (ASR = 1 - Accuracy) ====")
    print(f"{'Setting':8} {'TP':>7} {'FP':>7} {'FN':>7} {'Accuracy':>9} {'ASR':>7}")
    for name in ["CLEAN", "FGSM", "PGD", "C&W"]:
        tp, fp, fn = stats[name]
        d = tp + fp + fn
        acc = tp / d if d > 0 else 0.0
        print(f"{name:8} {tp:7d} {fp:7d} {fn:7d} {acc:9.3f} {1-acc:7.3f}")


if __name__ == "__main__":
    evaluate()
