"""
Evaluate adversarial attacks (FGSM, PGD, C&W) on a fine-tuned VisDrone YOLO model,
restricted to 5 VEHICLE classes: car, van, truck, bus, motor.

ASR = 1 - Accuracy,  Accuracy = TP / (TP + FP + FN), vs VisDrone ground truth.
Also saves a bar chart of ASR per attack.

USAGE: set the SETTINGS, then:  python attack_eval5.py
"""

import os, glob
import numpy as np
import torch, cv2
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")            # no display needed; save to file
import matplotlib.pyplot as plt

# ==================== SETTINGS ====================
MODEL_PATH = "best.pt"
VAL_IMAGES = "/home/aarav/Downloads/VisDrone2019-DET-val/images"
VAL_ANNOTS = "/home/aarav/Downloads/VisDrone2019-DET-val/annotations"
DEVICE     = "cuda"
NUM_IMAGES = 50                  # -1 for all 548
IMG_SIZE   = 640

EPS, ALPHA, ITERS = 8/255, 2/255, 30
CW_ITERS, CW_LR, CW_C = 30, 0.01, 0.01
IOU_THRESH, CONF_THRESH = 0.5, 0.25

# Only evaluate these 5 vehicle classes (model class IDs):
#   car=3, van=4, truck=5, bus=8, motor=9
VEHICLE_CLASSES = {3, 4, 5, 8, 9}
# =================================================


def load_ground_truth(annot_path, sx, sy):
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
            if cls not in VEHICLE_CLASSES:      # keep only the 5 vehicle classes
                continue
            gts.append((cls, x*sx, y*sy, (x+w)*sx, (y+h)*sy))
    return gts


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def match_counts(preds, gts):
    matched, tp = set(), 0
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
            tp += 1; matched.add(best_j)
    return tp, len(preds)-tp, len(gts)-len(matched)


def raw_confidence(out):
    pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred[:, 4:, :].max(dim=1)[0].sum()


def attack_fgsm(raw, x0):
    x = x0.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = raw_confidence(raw(x))
    g = torch.autograd.grad(loss, x)[0]
    return (x0 - EPS*g.sign()).clamp(0, 1).detach()

def attack_pgd(raw, x0):
    x_adv = x0.clone().detach()
    for _ in range(ITERS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = raw_confidence(raw(x_adv))
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.max(torch.min(x_adv.detach()-ALPHA*g.sign(), x0+EPS), x0-EPS).clamp(0, 1)
    return x_adv.detach()

def attack_cw(raw, x0):
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=CW_LR)
    for _ in range(CW_ITERS):
        x_adv = (x0+delta).clamp(0, 1)
        with torch.enable_grad():
            loss = raw_confidence(raw(x_adv)) + CW_C*(delta**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)
    return (x0+delta).clamp(0, 1).detach()


def predict_boxes(yolo, tensor_img):
    arr = (tensor_img[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:, :, ::-1].copy(), verbose=False, conf=CONF_THRESH)[0]
    out = []
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls not in VEHICLE_CLASSES:          # keep only the 5 vehicle classes
            continue
        out.append((cls, *b.xyxy[0].tolist(), float(b.conf[0])))
    return out


def evaluate():
    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)                 # separate instance for attacks
    raw = raw_yolo.model.float().to(DEVICE)
    raw.train(False)
    for p in raw.parameters():
        p.requires_grad_(False)

    images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    if NUM_IMAGES > 0:
        images = images[:NUM_IMAGES]
    print(f"Evaluating on {len(images)} images (5 vehicle classes only)...\n")

    attacks = {"FGSM": attack_fgsm, "PGD": attack_pgd, "C&W": attack_cw}
    stats = {k: [0, 0, 0] for k in ["CLEAN", "FGSM", "PGD", "C&W"]}

    for idx, img_path in enumerate(images):
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        H, W = orig.shape[:2]
        sx, sy = IMG_SIZE/W, IMG_SIZE/H
        annot = os.path.join(VAL_ANNOTS, os.path.splitext(os.path.basename(img_path))[0]+".txt")
        gts = load_ground_truth(annot, sx, sy)

        img640 = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        x0 = torch.from_numpy(img640[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0)/255.0
        x0 = x0.to(DEVICE)

        tp, fp, fn = match_counts(predict_boxes(yolo, x0), gts)
        stats["CLEAN"][0] += tp; stats["CLEAN"][1] += fp; stats["CLEAN"][2] += fn
        for name, atk in attacks.items():
            tp, fp, fn = match_counts(predict_boxes(yolo, atk(raw, x0)), gts)
            stats[name][0] += tp; stats[name][1] += fp; stats[name][2] += fn

        if (idx+1) % 10 == 0:
            print(f"  {idx+1}/{len(images)} done")

    # ---- table ----
    print("\n==== RESULTS (5 vehicle classes | ASR = 1 - Accuracy) ====")
    print(f"{'Setting':8} {'TP':>7} {'FP':>7} {'FN':>7} {'Accuracy':>9} {'ASR':>7}")
    asr_values = {}
    for name in ["CLEAN", "FGSM", "PGD", "C&W"]:
        tp, fp, fn = stats[name]
        d = tp+fp+fn
        acc = tp/d if d > 0 else 0.0
        asr_values[name] = 1-acc
        print(f"{name:8} {tp:7d} {fp:7d} {fn:7d} {acc:9.3f} {1-acc:7.3f}")

    # ---- bar chart ----
    labels = ["CLEAN", "FGSM", "PGD", "C&W"]
    vals = [asr_values[l] for l in labels]
    colors = ["gray", "#4C9", "#39C", "#C36"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel("Attack Success Rate (1 - Accuracy)")
    plt.title("Adversarial Attack Success Rate\n(VisDrone, 5 vehicle classes)")
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("asr_results.png", dpi=150)
    print("\nSaved bar chart: asr_results.png")


if __name__ == "__main__":
    evaluate()
