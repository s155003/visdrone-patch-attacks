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
SAVE_DIR   = "patch_examples_oriented"

# ---- patch / optimization params ----
PATCH_FRAC = 0.45      # patch W/H as a fraction of the car's ACTUAL width/height (kept small + centered to stay ON the car, off background corners)
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

def estimate_car_angle(img, xc, yc, bw, bh, H, W):
    """
    Estimate a car's orientation angle (radians) from its pixels in the box,
    using PCA on the gradient/intensity distribution. Falls back to the box
    aspect ratio for tiny or ambiguous cars.
    Returns angle in radians of the car's LONG axis relative to horizontal.
    """
    import numpy as _np
    x1 = int((xc - bw/2) * W); x2 = int((xc + bw/2) * W)
    y1 = int((yc - bh/2) * H); y2 = int((yc + bh/2) * H)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = img[0, :, y1:y2, x1:x2].mean(0).cpu().numpy()   # grayscale car crop
    # edge magnitude via simple gradients
    gx = _np.abs(_np.diff(crop, axis=1, prepend=crop[:, :1]))
    gy = _np.abs(_np.diff(crop, axis=0, prepend=crop[:1, :]))
    edge = gx + gy
    ys, xs = _np.nonzero(edge > edge.mean())
    if len(xs) < 8:
        # fallback: aspect ratio -> horizontal if wider, vertical if taller
        return 0.0 if (bw * W) >= (bh * H) else _np.pi / 2
    # PCA: principal axis of the edge points = car's long axis
    pts = _np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(_np.float32)
    cov = pts.T @ pts / max(1, len(pts))
    evals, evecs = _np.linalg.eigh(cov)
    major = evecs[:, _np.argmax(evals)]     # eigenvector of largest eigenvalue
    angle = _np.arctan2(major[1], major[0])
    return float(angle)

def _rotate_patch(patch_chw, angle, out_h, out_w):
    """Differentiably rotate+resize a [3,ph,pw] patch to [3,out_h,out_w] at `angle` (rad)."""
    import torch as _t
    p = patch_chw.unsqueeze(0)  # [1,3,ph,pw]
    # first resize to target size, then rotate via affine grid
    p = _t.nn.functional.interpolate(p, size=(out_h, out_w), mode="bilinear", align_corners=False)
    cos, sin = _t.cos(_t.tensor(angle)), _t.sin(_t.tensor(angle))
    # affine matrix for rotation (grid_sample samples input; use inverse rotation)
    theta = _t.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=p.dtype).unsqueeze(0)
    grid = _t.nn.functional.affine_grid(theta, p.size(), align_corners=False)
    p = _t.nn.functional.grid_sample(p, grid, align_corners=False, padding_mode="border")
    return p[0]  # [3,out_h,out_w]

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
        # patch sized to the car's actual width/height (proportional, small,
        # centered) AND rotated to match the car's estimated orientation so its
        # edges follow the car's body instead of sticking out over the sides.
        box_w_px = bw * W; box_h_px = bh * H
        pw = int(PATCH_FRAC * box_w_px)
        ph = int(PATCH_FRAC * box_h_px)
        pw = max(4, min(pw, W)); ph = max(4, min(ph, H))
        cx, cy = int(xc*W), int(yc*H)
        # estimate car orientation and rotate the patch to align with it
        angle = estimate_car_angle(img, xc, yc, bw, bh, H, W)
        # use a square-ish canvas big enough to hold the rotated patch without clipping
        side = max(pw, ph)
        x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
        x2 = min(W, x1 + side);    y2 = min(H, y1 + side)
        rh, rw = y2 - y1, x2 - x1
        if rh < 4 or rw < 4: continue
        p = _rotate_patch(patch, angle, rh, rw)   # [3,rh,rw], differentiable
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

def _draw_annotated(x0, patched, clean_dets, patch_dets, boxes_px, save_path):
    """Draw a CLEAN | PATCHED side-by-side with boxes, save as PNG.
    yellow = GT source box, red = target-class det, green = source-class det, gray = other."""
    def to_bgr(t):
        a = (t[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
        return a[:, :, ::-1].copy()
    def panel(img_bgr, dets, label):
        im = img_bgr.copy()
        for (bx1,by1,bx2,by2) in boxes_px:
            cv2.rectangle(im,(int(bx1),int(by1)),(int(bx2),int(by2)),(0,255,255),2)  # yellow GT
        for d in dets:
            cls,x1,y1,x2,y2,conf = d
            if   cls == TGT_IDX: color,tag = (0,0,255), f"{TGT_NAME} {conf:.2f}"     # red target
            elif cls == SRC_IDX: color,tag = (0,255,0), f"{SRC_NAME} {conf:.2f}"     # green source
            else:                color,tag = (150,150,150), None
            cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            if tag: cv2.putText(im,tag,(int(x1),max(12,int(y1)-4)),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
        cv2.rectangle(im,(0,0),(im.shape[1],26),(30,30,30),-1)
        cv2.putText(im,label,(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
        return im
    left  = panel(to_bgr(x0), clean_dets, "CLEAN")
    right = panel(to_bgr(patched), patch_dets, "PATCHED")
    cv2.imwrite(save_path, np.hstack([left, right]))

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
        # AUTO-ANNOTATE: draw CLEAN|PATCHED PNG for this image (FLIP_ prefix if any flip)
        img_flip = any(is_bus_at(patch_dets, bpx) and not is_bus_at(clean_dets, bpx) for bpx in boxes_px)
        os.makedirs(os.path.join(SAVE_DIR, "annotated"), exist_ok=True)
        png_name = ("FLIP_" if img_flip else "") + stem + ".png"
        _draw_annotated(x0, patched, clean_dets, patch_dets, boxes_px,
                        os.path.join(SAVE_DIR, "annotated", png_name))

    eligible = n_src - native_bus
    print(f"\n==== DIGITAL PATCH {SRC_NAME}->{TGT_NAME} RESULTS ====")
    print(f"Images: {len(data)}   {SRC_NAME} GT boxes: {n_src}")
    print(f"(a) already-{TGT_NAME} under CLEAN (native): {native_bus}")
    print(f"    eligible {SRC_NAME}s: {eligible}")
    print(f"aggregate detected as {TGT_NAME}: CLEAN={agg_clean}  PATCHED={agg_patch}")
    tr = true_new/eligible if eligible else 0.0
    print(f"\nTRUE-NEW {SRC_NAME}->{TGT_NAME} flips (patch only): {true_new}")
    print(f"TRUE-RATE (true-new / eligible): {tr:.3f}   <- HEADLINE")
    print(f"\nSaved universal patch + per-image examples + annotated PNGs to {SAVE_DIR}/")
    print(f"    Annotated CLEAN|PATCHED images in {SAVE_DIR}/annotated/ (FLIP_ prefix = has a flip)")

if __name__ == "__main__":
    run()
