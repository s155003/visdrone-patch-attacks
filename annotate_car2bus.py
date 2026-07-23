"""
Annotate car->bus adversarial examples: draw 4-way side-by-side PNGs
(CLEAN | FGSM | PGD | C&W) from the saved .pt images + .boxes.json detections.

Boxes drawn:
  yellow = ground-truth CAR box (source)
  red    = BUS detections   (target - what we want to appear)
  green  = CAR detections   (source - what we're suppressing)
  gray   = other detections

Prioritizes cars that FLIPPED to bus (true-new) so the compelling cases come first.

USAGE: cd ~/attack-test && ~/attack_env/bin/python annotate_car2bus.py
"""
import os, glob, json
import numpy as np
import torch, cv2

SAVE_DIR = "adv_examples_car2bus"
OUT_DIR  = os.path.join(SAVE_DIR, "annotated")
IMG_SIZE = 640
IOU_THRESH = 0.5
SRC_IDX, TGT_IDX = 0, 3          # car (source), bus (target)
PANELS = ["clean", "FGSM", "PGD", "C&W"]

COLORS = {  # BGR for cv2
    "gt":   (0, 255, 255),   # yellow  - GT car box
    "bus":  (0, 0, 255),     # red     - target detections
    "car":  (0, 255, 0),     # green   - source detections
    "other":(150,150,150),   # gray
}

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def tensor_to_bgr(t):
    """[1,3,H,W] float 0-1 -> HxWx3 uint8 BGR for cv2."""
    arr = (t[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)  # RGB
    return arr[:, :, ::-1].copy()                                      # -> BGR

def draw_panel(img_bgr, dets, gt_boxes, label):
    im = img_bgr.copy()
    # GT car boxes (yellow)
    for (x1,y1,x2,y2) in gt_boxes:
        cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), COLORS["gt"], 2)
    # detections
    for d in dets:
        cls, x1, y1, x2, y2, conf = d
        if   cls == TGT_IDX: color, tag = COLORS["bus"], f"bus {conf:.2f}"
        elif cls == SRC_IDX: color, tag = COLORS["car"], f"car {conf:.2f}"
        else:                color, tag = COLORS["other"], None
        cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        if tag:
            cv2.putText(im, tag, (int(x1), max(12,int(y1)-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    # panel label bar
    cv2.rectangle(im, (0,0), (IMG_SIZE, 26), (30,30,30), -1)
    cv2.putText(im, label, (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return im

def flipped_to_bus(dets, gt_boxes):
    """Does any GT car box have a BUS detection overlapping it (IoU>=thr)?"""
    for gb in gt_boxes:
        for cls, x1,y1,x2,y2,conf in dets:
            if cls == TGT_IDX and iou(gb,(x1,y1,x2,y2)) >= IOU_THRESH:
                return True
    return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pts = sorted(glob.glob(os.path.join(SAVE_DIR, "*.pt")))
    if not pts:
        print(f"No .pt files in {SAVE_DIR}/ - has the attack saved any images yet?")
        return
    print(f"Found {len(pts)} saved examples. Annotating...\n")

    made, flipped_first = [], []
    for pt in pts:
        stem = os.path.splitext(os.path.basename(pt))[0]
        data = torch.load(pt)
        gt = data["src_boxes_px"]
        jf = os.path.join(SAVE_DIR, stem + ".boxes.json")
        if not os.path.exists(jf):
            continue
        boxes = json.load(open(jf))

        panels = []
        any_flip = False
        for key in PANELS:
            if key not in data or key not in boxes:
                continue
            img = tensor_to_bgr(data[key])
            dets = boxes[key]
            if key != "clean" and flipped_to_bus(dets, gt):
                any_flip = True
            panels.append(draw_panel(img, dets, gt, key.upper()))
        if not panels:
            continue
        strip = np.hstack(panels)
        out = os.path.join(OUT_DIR, ("FLIP_" if any_flip else "") + stem + ".png")
        cv2.imwrite(out, strip)
        made.append(out)
        if any_flip: flipped_first.append(out)

    print(f"Wrote {len(made)} annotated PNGs to {OUT_DIR}/")
    print(f"  of which {len(flipped_first)} show at least one car->bus flip (prefixed FLIP_).")
    if flipped_first:
        print("\nFlip examples (look at these first):")
        for f in flipped_first[:10]:
            print("  " + f)

if __name__ == "__main__":
    main()
