"""
DIGITAL ADVERSARIAL PATCH attack: make the detector misclassify CAR as BUS.

Patch-based (Thys et al. / Pathak et al. style), adapted for DIGITAL-only and
for TARGETED class-flipping (car->bus) instead of hiding:
  - A square patch is optimized and pasted ON each ground-truth car.
  - Patch is scaled to a fraction of each car's bounding box.
  - Optimized with Adam to push the BUS logit above the CAR logit (margin loss),
    at the car's location.
  - Digital-only simplifications vs the papers: NO non-printability (NPS) loss,
    NO physical transforms (rotation/brightness jitter). Keeps a light Total-
    Variation (TV) loss so the patch stays smooth/patch-like, not pure noise.

Unlike FGSM/PGD/C&W (whole-image perturbation), here ONLY the patch pixels
change; the rest of the image is untouched. The SAME patch is trained across
images (a single universal car->bus patch), matching the paper's approach.

Model output (5-class): [x,y,w,h, car(0),van(1),truck(2),bus(3),motor(4)], [1,9,N].

USAGE: cd ~/attack-test && CUDA_VISIBLE_DEVICES="" ~/attack_env/bin/python patch_attack_car2bus.py
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
NUM_IMAGES = 30
IMG_SIZE   = 640
SAVE_DIR   = "patch_examples_car2van_30"

# ---- patch / optimization params ----
PATCH_FRAC = 0.7       # patch side length as a fraction of sqrt(box area) -> ~ covers part of the car
EPOCHS     = 40        # optimization passes over the training images
LR         = 0.03      # Adam learning rate for the patch pixels
TV_WEIGHT  = 0.05      # total-variation weight (keep patch smooth; 0 = allow noisy)
MARGIN     = 5.0       # push bus logit above car logit by this margin (logit space)

CONF_THRESH, IOU_THRESH = 0.25, 0.5
SRC_IDX, TGT_IDX = 0, 1      # car (source), van (target)
SRC_NAME, TGT_NAME = "car", "van"
PATCH_PX = 80                # patch is stored at this resolution, resized per object
# =================================================

def load_src_boxes_norm(label_path):
    boxes = []
    if not os.path.exists(label_path): return boxes
    for line in open(label_path).read().strip().splitlines():
        if not line: continue
        p = line.split()
        if len(p) < 5: continue
        if int(p[0]) != SRC_IDX: continue
        boxes.append(tuple(map(float, p[1:5])))   # xc,yc,w,h normalized
    return boxes

def apply_patch(img, patch, boxes_norm):
    """
    Paste the (differentiable) patch onto each car box in img.
    img:   [1,3,H,W] tensor (0-1)
    patch: [3,PATCH_PX,PATCH_PX] tensor (0-1), the thing we optimize
    Returns a NEW image tensor with the patch composited in (grad flows to patch).
    """
    out = img.clone()
    _, _, H, W = img.shape
    for (xc, yc, bw, bh) in boxes_norm:
        # patch side = PATCH_FRAC * sqrt(box area), in pixels
        side = int(PATCH_FRAC * ((bw*W) * (bh*H)) ** 0.5)
        side = max(8, min(side, H, W))
        # center it on the car
        cx, cy = int(xc*W), int(yc*H)
        x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
        x2 = min(W, x1 + side);    y2 = min(H, y1 + side)
        pw, ph = x2 - x1, y2 - y1
        if pw < 4 or ph < 4: continue
        # resize patch to (ph, pw) differentiably
        p = torch.nn.functional.interpolate(patch.unsqueeze(0), size=(ph, pw),
                                            mode="bilinear", align_corners=False)[0]
        out[0, :, y1:y2, x1:x2] = p
    return out

def targeted_loss(raw_out, src_boxes_norm):
    """Push bus logit above car logit at anchors inside the car boxes (logit space)."""
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    pred = pred[0]                       # [9, N]
    box = pred[:4, :]; cls = pred[4:, :]
    cx = box[0, :] / IMG_SIZE; cy = box[1, :] / IMG_SIZE
    mask = torch.zeros_like(cx, dtype=torch.bool)
    for (sxc, syc, sw, sh) in src_boxes_norm:
        x1, x2 = sxc - sw/2, sxc + sw/2
        y1, y2 = syc - sh/2, syc + sh/2
        mask = mask | ((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))
    if mask.sum() == 0:
        return cls.sum() * 0.0
    e = 1e-6
    cls_logit = torch.log(cls.clamp(e,1-e)) - torch.log1p(-cls.clamp(e,1-e))
    tgt = cls_logit[TGT_IDX, mask]; src = cls_logit[SRC_IDX, mask]
    return torch.clamp(src - tgt + MARGIN, min=0).sum()

def tv_loss(patch):
    """Total variation: keep the patch smooth."""
    dh = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
    dw = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
    return dh + dw

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def detect(yolo, timg):
    arr = (timg[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    with torch.no_grad():
        res = yolo.predict(arr[:,:,::-1].copy(), verbose=False, conf=CONF_THRESH, device="cpu")[0]
    return [(int(b.cls[0]), *b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]

def is_bus_at(dets, box_px):
    for cls,x1,y1,x2,y2,conf in dets:
        if cls == TGT_IDX and iou(box_px,(x1,y1,x2,y2)) >= IOU_THRESH:
            return True
    return False

def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    yolo = YOLO(MODEL_PATH)
    raw_yolo = YOLO(MODEL_PATH)
    raw = raw_yolo.model.float().to(DEVICE); raw.train(False)
    for p in raw.parameters(): p.requires_grad_(False)

    # gather car images
    imgs = []
    for ip in sorted(glob.glob(os.path.join(VAL_IMAGES,"*.jpg"))):
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        if load_src_boxes_norm(lbl): imgs.append(ip)
        if NUM_IMAGES>0 and len(imgs)>=NUM_IMAGES: break

    # preload images as tensors + their car boxes
    data = []
    for ip in imgs:
        orig = cv2.imread(ip)
        if orig is None: continue
        lbl = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(ip))[0]+".txt")
        img = cv2.resize(orig,(IMG_SIZE,IMG_SIZE))
        x0 = torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).float().unsqueeze(0)/255.0
        data.append((os.path.splitext(os.path.basename(ip))[0], x0, load_src_boxes_norm(lbl)))
    print(f"DIGITAL PATCH {SRC_NAME}->{TGT_NAME}: training ONE universal patch on {len(data)} images")
    print(f"Params: PATCH_FRAC={PATCH_FRAC}, EPOCHS={EPOCHS}, LR={LR}, TV={TV_WEIGHT}, MARGIN={MARGIN}\n")

    # the patch we optimize (start from random)
    patch = torch.rand(3, PATCH_PX, PATCH_PX, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=LR)

    for ep in range(EPOCHS):
        tot = 0.0
        for stem, x0, boxes in data:
            if not boxes: continue
            patched = apply_patch(x0, patch.clamp(0,1), boxes)
            loss = targeted_loss(raw(patched), boxes) + TV_WEIGHT * tv_loss(patch)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): patch.clamp_(0,1)
            tot += float(loss)
        print(f"  epoch {ep+1}/{EPOCHS}  avg loss {tot/max(1,len(data)):.3f}")

    # ---- evaluate the trained patch ----
    print("\nEvaluating trained patch (clean vs patched)...")
    n_src=0; native_bus=0; true_new=0; agg_clean=0; agg_patch=0
    final_patch = patch.detach().clamp(0,1)
    torch.save(final_patch, os.path.join(SAVE_DIR,"universal_patch.pt"))
    # also save a viewable PNG of the patch itself
    pimg = (final_patch.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]
    cv2.imwrite(os.path.join(SAVE_DIR,"universal_patch.png"), pimg)

    for stem, x0, boxes in data:
        if not boxes: continue
        boxes_px = [((xc-w/2)*IMG_SIZE,(yc-h/2)*IMG_SIZE,(xc+w/2)*IMG_SIZE,(yc+h/2)*IMG_SIZE)
                    for (xc,yc,w,h) in boxes]
        n_src += len(boxes_px)
        clean_dets = detect(yolo, x0)
        patched = apply_patch(x0, final_patch, boxes)
        patch_dets = detect(yolo, patched)
        for bpx in boxes_px:
            c_clean = is_bus_at(clean_dets, bpx)
            c_patch = is_bus_at(patch_dets, bpx)
            native_bus += int(c_clean)
            agg_clean  += int(c_clean)
            agg_patch  += int(c_patch)
            if (not c_clean) and c_patch: true_new += 1
        # save example + boxes
        torch.save({"clean":x0, "patched":patched.detach(), "boxes_px":boxes_px},
                   os.path.join(SAVE_DIR, stem+".pt"))
        with open(os.path.join(SAVE_DIR, stem+".boxes.json"),"w") as f:
            json.dump({"clean":clean_dets,"patched":patch_dets}, f)

    eligible = n_src - native_bus
    print(f"\n==== DIGITAL PATCH {SRC_NAME}->{TGT_NAME} RESULTS ====")
    print(f"Images: {len(data)}   {SRC_NAME} GT boxes: {n_src}")
    print(f"(a) already-{TGT_NAME} under CLEAN (native): {native_bus}")
    print(f"    eligible {SRC_NAME}s: {eligible}")
    print(f"aggregate detected as {TGT_NAME}: CLEAN={agg_clean}  PATCHED={agg_patch}")
    tr = true_new/eligible if eligible else 0.0
    print(f"\nTRUE-NEW {SRC_NAME}->{TGT_NAME} flips (patch only): {true_new}")
    print(f"TRUE-RATE (true-new / eligible): {tr:.3f}   <- HEADLINE")
    print(f"\nSaved universal patch + per-image examples to {SAVE_DIR}/")

if __name__ == "__main__":
    run()
