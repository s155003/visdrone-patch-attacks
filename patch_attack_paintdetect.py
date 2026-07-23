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
DEVICE     = "cuda"
NUM_IMAGES = 100
IMG_SIZE   = 640
SAVE_DIR   = "patch_examples_paintdetect"   # overridden

# ---- patch / optimization params ----
PATCH_FRAC = 0.45      # (unused when FIXED_FRAC>0)
FIXED_FRAC = 0.45      # patch size (fits on an end body-panel with clearance)
END_OFFSET = 0.35      # shift toward an end (fraction of car length); larger = further onto the panel
GLASS_BAND = 0.40      # middle fraction reserved as glass (patch NEVER enters this band)
SHAPE       = "square" # square|circle|triangle|hexagon|lshape|cross
BOTH_ENDS   = False    # if True, place a patch on BOTH body-panel ends (hood + trunk)
TILTED      = False    # if True, use R13-style tilted footprint (rotate the shape mask too)
MULTI_SHAPE = ""       # "" = single; "stacked" = overlapping shapes; "apart" = separated shapes on panel
MULTI_N     = 3        # number of shapes when MULTI_SHAPE is set
WRAP_MODE   = "tiles"  # "tiles" (49 indep tiles in wrap zones) or "contiguous" (one patch per zone)
WRAP_TILES  = 49       # tiles for WRAP_MODE="tiles"
# Car-wrap allowed zones along the LONG axis (0=one end, 1=other end). Skip windshields.
WRAP_ZONES  = [(0.00,0.22),(0.36,0.60),(0.74,1.00)]   # hood, roof, trunk
# (the two gaps 0.22-0.36 and 0.60-0.74 are the windshields, never touched)
EPOCHS     = 80        # optimization passes over the training images
LR         = 0.03      # Adam learning rate for the patch pixels
TV_WEIGHT  = 0.05      # total-variation weight (keep patch smooth; 0 = allow noisy)
MARGIN     = 5.0       # push bus logit above car logit by this margin (logit space)
CONF_WEIGHT = 0.0      # DISABLED: the confidence term hurt (Run 11) - clean margin loss is better
                       #        (so flipped detections are strong enough to survive NMS/conf-threshold)
MOMENTUM    = 0.0      # DISABLED: momentum hurt (Run 11) - plain Adam is better

CONF_THRESH, IOU_THRESH = 0.25, 0.5
SRC_IDX, TGT_IDX = 0, 1      # car (source), van (target)
SRC_NAME, TGT_NAME = "car", "van"
PATCH_PX = 160
PAINT_METHOD = "brightness"  # "brightness" or "domcolor" - how to detect car paint
PAINT_LO_PCT = 35            # brightness percentile below which = glass/wheel/shadow (excluded)
PAINT_FILL   = "tiles"       # "tiles" or "patch" over the detected paint
PAINT_TILE_PX = 10           # tile size in pixels for "tiles" mode                # patch is stored at this resolution, resized per object
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
    ROBUST orientation estimate (radians) of the car's LONG axis.
    Improvements over the naive version (which fell back to 0/90 too often on
    tiny cars): (1) UPSCALE the crop so tiny cars have enough pixels, (2) use a
    weighted gradient-STRUCTURE-TENSOR (every pixel votes by its gradient
    magnitude) instead of a hard edge threshold, which is far more stable on
    small/blurry aerial cars. Aspect-ratio fallback only if truly featureless.
    """
    import numpy as _np
    import cv2 as _cv2
    x1 = int((xc - bw/2) * W); x2 = int((xc + bw/2) * W)
    y1 = int((yc - bh/2) * H); y2 = int((yc + bh/2) * H)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = img[0, :, y1:y2, x1:x2].mean(0).cpu().numpy().astype(_np.float32)
    # (1) UPSCALE small crops so there are enough pixels for a stable estimate
    ch, cw = crop.shape
    scale = max(1, int(64 / max(1, min(ch, cw))))   # bring short side up to ~64px
    if scale > 1:
        crop = _cv2.resize(crop, (cw*scale, ch*scale), interpolation=_cv2.INTER_LINEAR)
    # smooth a touch to reduce pixel noise dominating the gradient
    crop = _cv2.GaussianBlur(crop, (3,3), 0)
    # (2) STRUCTURE TENSOR: weight every pixel by gradient magnitude (no hard threshold)
    gx = _cv2.Sobel(crop, _cv2.CV_32F, 1, 0, ksize=3)
    gy = _cv2.Sobel(crop, _cv2.CV_32F, 0, 1, ksize=3)
    Jxx = float((gx*gx).sum()); Jyy = float((gy*gy).sum()); Jxy = float((gx*gy).sum())
    # if there's basically no gradient signal, fall back to aspect ratio
    if (Jxx + Jyy) < 1e-6:
        return 0.0 if (bw * W) >= (bh * H) else _np.pi / 2
    # the DOMINANT EDGE direction is the eigenvector of the structure tensor with
    # the LARGER eigenvalue; the car's LONG axis is PERPENDICULAR to that (edges
    # run across the car's width), so we take the smaller-eigenvalue direction.
    # standard structure-tensor dominant EDGE orientation; the car's LONG axis is
    # perpendicular to it. Negate for image y-axis (points down). Validated to
    # recover tilted angles within ~1 degree.
    edge_dir = 0.5 * _np.arctan2(2*Jxy, Jxx - Jyy)
    angle = -(edge_dir + _np.pi/2)
    return float(angle)

def _rotate_patch(patch_chw, angle, out_h, out_w):
    """Differentiably rotate+resize a [3,ph,pw] patch to [3,out_h,out_w] at `angle` (rad)."""
    import torch as _t
    p = patch_chw.unsqueeze(0)  # [1,3,ph,pw]
    # first resize to target size, then rotate via affine grid
    p = _t.nn.functional.interpolate(p, size=(out_h, out_w), mode="bilinear", align_corners=False)
    cos, sin = _t.cos(_t.tensor(angle)), _t.sin(_t.tensor(angle))
    # affine matrix for rotation (grid_sample samples input; use inverse rotation)
    theta = _t.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=p.dtype, device=p.device).unsqueeze(0)
    grid = _t.nn.functional.affine_grid(theta, p.size(), align_corners=False)
    p = _t.nn.functional.grid_sample(p, grid, align_corners=False, padding_mode="border")
    return p[0]  # [3,out_h,out_w]

def make_shape_mask(shape, size, device):
    """HxW float mask (1 inside shape, 0 outside). Built on CPU then moved to device."""
    import numpy as _np, cv2 as _cv2, torch as _t
    m = _np.zeros((size, size), _np.float32); c = size//2
    if shape == "square": m[:] = 1.0
    elif shape == "circle": _cv2.circle(m,(c,c),max(1,c-1),1.0,-1)
    elif shape == "triangle":
        _cv2.fillPoly(m,[_np.array([[c,1],[1,size-2],[size-2,size-2]],_np.int32)],1.0)
    elif shape == "hexagon":
        pts=[[int(c+(c-1)*_np.cos(_np.pi/6+i*_np.pi/3)),int(c+(c-1)*_np.sin(_np.pi/6+i*_np.pi/3))] for i in range(6)]
        _cv2.fillPoly(m,[_np.array(pts,_np.int32)],1.0)
    elif shape == "lshape": m[:,:size//2]=1.0; m[size//2:,:]=1.0
    elif shape == "cross":
        w=max(1,size//3); m[:,c-w//2:c+w//2+1]=1.0; m[c-w//2:c+w//2+1,:]=1.0
    return _t.from_numpy(m).to(device)

def _car_axis_frames(xc, yc, bw, bh, W, H):
    """Return car center (px), long-axis unit vec, perp unit vec, length, width (px)."""
    import math as _m
    cx = xc*W; cy = yc*H
    box_w_px = bw*W; box_h_px = bh*H
    car_len = max(box_w_px, box_h_px)
    car_wid = min(box_w_px, box_h_px)
    return cx, cy, car_len, car_wid

def _paint_body_mask(u, v):
    """u in [0,1] along length, v in [-0.5,0.5] across width. 1=painted body, 0=glass."""
    import numpy as _np
    m = _np.ones_like(u, dtype='float32')
    # exclude ONLY the front and rear windshields (thin strips); everything else painted
    m[(u>=0.24)&(u<0.34)] = 0.0                         # front windshield (thin)
    m[(u>=0.62)&(u<0.72)] = 0.0                         # rear windshield (thin)
    return m

def _detect_paint_mask(crop_bgr):
    """Return HxW float mask (1=car paint, 0=glass/wheel/ground) from a car crop."""
    import numpy as _np, cv2 as _cv2
    if crop_bgr.shape[0] < 3 or crop_bgr.shape[1] < 3:
        return _np.ones(crop_bgr.shape[:2], _np.float32)
    if PAINT_METHOD == "domcolor":
        small = crop_bgr.reshape(-1,3).astype(_np.float32)
        K=4
        crit=(_cv2.TERM_CRITERIA_EPS+_cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        try:
            _,labels,centers=_cv2.kmeans(small,K,None,crit,3,_cv2.KMEANS_PP_CENTERS)
        except Exception:
            return _np.ones(crop_bgr.shape[:2], _np.float32)
        labels=labels.reshape(crop_bgr.shape[:2])
        bright=[centers[k].mean() for k in range(K)]; sizes=[(labels==k).sum() for k in range(K)]
        order=_np.argsort(bright); bc=order[K//2:]
        paint_k=bc[_np.argmax([sizes[k] for k in bc])]
        return (labels==paint_k).astype(_np.float32)
    else:  # brightness
        g=_cv2.cvtColor(crop_bgr,_cv2.COLOR_BGR2GRAY).astype(_np.float32)
        lo=_np.percentile(g, PAINT_LO_PCT)
        paint=(g>lo).astype(_np.uint8)
        k=_np.ones((3,3),_np.uint8)
        paint=_cv2.morphologyEx(paint,_cv2.MORPH_OPEN,k)
        paint=_cv2.morphologyEx(paint,_cv2.MORPH_CLOSE,k)
        return paint.astype(_np.float32)

def paint_apply(out, img, patch, xc, yc, bw, bh, angle):
    """Paint ONE continuous adversarial pattern over the whole car body silhouette,
    excluding all glass (windshields + side windows). Oriented + clamped to the car."""
    import math as _m, numpy as _np, torch as _t
    _,_,H,W = out.shape
    cx=xc*W; cy=yc*H
    box_w_px=bw*W; box_h_px=bh*H
    dx=_m.cos(angle); dy=_m.sin(angle)
    # VisDrone boxes are AXIS-ALIGNED but cars are DIAGONAL -> the box corners are ground.
    # Compute the largest oriented car-rectangle that fits INSIDE the axis-aligned box.
    # A rect of half-extents (L/2 along axis, Wd/2 across) rotated by angle spans
    #   box_w = L*|cos| + Wd*|sin|,  box_h = L*|sin| + Wd*|cos|.
    # Solve for L, Wd given the box (box_w_px, box_h_px):
    ca=abs(dx); sa=abs(dy); det=ca*ca - sa*sa
    if abs(det) < 1e-3:
        # ~45 deg: fall back to a conservative inscribed square
        L = Wd = 0.7*min(box_w_px,box_h_px)
    else:
        L  = (box_w_px*ca - box_h_px*sa)/det
        Wd = (box_h_px*ca - box_w_px*sa)/det
        L=abs(L); Wd=abs(Wd)
    # clamp to sane positive values and apply a safety margin so we never touch the edge
    L = max(4.0, min(L, max(box_w_px,box_h_px)))*0.92
    Wd= max(4.0, min(Wd, max(box_w_px,box_h_px)))*0.92
    car_len=L; car_wid=Wd
    # bounding region of the (rotated) car in image space: use the diagonal
    diag=int(((car_len**2+car_wid**2)**0.5))+2
    x1=max(0,int(cx-diag/2)); y1=max(0,int(cy-diag/2))
    x2=min(W,x1+diag); y2=min(H,y1+diag)
    rh,rw=y2-y1,x2-x1
    if rh<4 or rw<4: return
    # resize the single patch to cover this region
    psrc = patch if patch.dim()==3 else patch[0]
    p = _rotate_patch(psrc, 0.0, rh, rw)   # fill region; orientation handled by mask coords
    # build the body-minus-glass mask in image space, in the car's oriented frame
    # DETECT car paint from the actual image pixels in this region
    crop_bgr = (img[0,:,y1:y2,x1:x2].permute(1,2,0).cpu().numpy()[:,:,::-1]*255).astype('uint8').copy()
    paint = _detect_paint_mask(crop_bgr)   # 1=paint, 0=glass/wheel/ground
    # ALSO restrict to the inscribed oriented car-rect so we never paint ground corners
    ys,xs=_np.meshgrid(_np.arange(y1,y2),_np.arange(x1,x2),indexing='ij')
    uu=((xs-cx)*dx+(ys-cy)*dy)/max(1.0,car_len)+0.5
    vv=(-(xs-cx)*dy+(ys-cy)*dx)/max(1.0,car_wid)
    inside=((uu>=0.0)&(uu<=1.0)&(_np.abs(vv)<=0.5)).astype('float32')
    mask = paint * inside
    cmask=_t.from_numpy(mask).to(out.device).unsqueeze(0)
    region=out[0,:,y1:y2,x1:x2]
    out[0,:,y1:y2,x1:x2]=cmask*p+(1-cmask)*region
    return out

def wrap_apply(out, img, patch, xc, yc, bw, bh, angle):
    """Place patch content ONLY in the car-wrap allowed zones (hood/roof/trunk),
    skipping windshields, never off the car. Differentiable."""
    import math as _m, torch as _t
    _,_,H,W = out.shape
    cx = xc*W; cy = yc*H
    box_w_px = bw*W; box_h_px = bh*H
    car_len = max(box_w_px, box_h_px)
    car_wid = min(box_w_px, box_h_px)
    dx = _m.cos(angle); dy = _m.sin(angle)          # long axis unit vec
    px_, py_ = -dy, dx                               # perp (width) unit vec
    # width kept slightly inside the car so we never spill over the sides
    half_w = 0.42 * car_wid

    def stamp_axis_rect(t0, t1, patch_src, ntiles_along=1):
        # place along the axis from fraction t0..t1 of the car length, full allowed width
        seg_len = (t1 - t0) * car_len
        if seg_len < 3: return
        # center of this segment along the axis (relative to car center: -0.5..0.5)
        # split segment into ntiles_along cells if tiling
        for i in range(ntiles_along):
            f0 = t0 + (i/ntiles_along)*(t1-t0)
            f1 = t0 + ((i+1)/ntiles_along)*(t1-t0)
            fc = ((f0+f1)/2.0) - 0.5     # -0.5..0.5 along axis from center
            seg_h = (f1-f0)*car_len
            along = fc * car_len
            ccx = cx + along*dx; ccy = cy + along*dy
            side_along = max(3, int(seg_h*0.9))
            side_perp  = max(3, int(2*half_w))
            side = min(side_along, side_perp)  # square-ish tile fits inside zone+width
            x1 = int(ccx - side/2); y1 = int(ccy - side/2)
            x2 = x1+side; y2 = y1+side
            x1c=max(0,x1); y1c=max(0,y1); x2c=min(W,x2); y2c=min(H,y2)
            rh,rw = y2c-y1c, x2c-x1c
            if rh<3 or rw<3: continue
            p = _rotate_patch(patch_src, angle, rh, rw)
            region = out[0,:,y1c:y2c,x1c:x2c]
            # CAR-BOUNDARY MASK: only paint pixels that fall inside the car's
            # (rotated) footprint, so tiles never spill over the car's edges.
            import numpy as _np, torch as _t
            yy, xx = _np.meshgrid(_np.arange(y1c,y2c), _np.arange(x1c,x2c), indexing='ij')
            # transform pixel coords into car-aligned frame (u along length, v across width)
            u = (xx-cx)*dx + (yy-cy)*dy
            v = -(xx-cx)*dy + (yy-cy)*dx   # note: perp
            inside = (_np.abs(u) <= 0.5*car_len) & (_np.abs(v) <= 0.5*car_wid)
            cmask = _t.from_numpy(inside.astype('float32')).to(out.device).unsqueeze(0)
            out[0,:,y1c:y2c,x1c:x2c] = cmask*p + (1-cmask)*region

    if WRAP_MODE == "contiguous":
        # one contiguous patch per allowed zone
        pi = 0
        for (a,b) in WRAP_ZONES:
            src = patch[pi % patch.shape[0]] if patch.dim()==4 else patch
            stamp_axis_rect(a, b, src, ntiles_along=1); pi += 1
    else:
        # distribute WRAP_TILES across zones proportional to zone length
        total = sum(b-a for a,b in WRAP_ZONES)
        assigned = 0
        for zi,(a,b) in enumerate(WRAP_ZONES):
            n = max(1, round(WRAP_TILES * (b-a)/total))
            for i in range(n):
                src = patch[assigned % patch.shape[0]] if patch.dim()==4 else patch
                f0 = a + (i/n)*(b-a); f1 = a + ((i+1)/n)*(b-a)
                stamp_axis_rect(f0, f1, src, ntiles_along=1)
                assigned += 1
    return out

def apply_patch(img, patch, boxes_norm):
    """
    Paste the (differentiable) patch onto each car box in img.
    img:   [1,3,H,W] tensor (0-1)
    patch: [3,PATCH_PX,PATCH_PX] tensor (0-1), the thing we optimize
    Returns a NEW image tensor with the patch composited in (grad flows to patch).
    """
    import math as _m
    out = img.clone()
    _, _, H, W = img.shape

    # WRAP MODE: place only in hood/roof/trunk zones, skip windshields
    if globals().get('USE_PAINT', False):
        for (xc, yc, bw, bh) in boxes_norm:
            angle = estimate_car_angle(img, xc, yc, bw, bh, H, W)
            paint_apply(out, img, patch, xc, yc, bw, bh, angle)
        return out

    def _stamp(cx, cy, side, angle, patch_src):
        """Composite one shaped, oriented patch centered at (cx,cy) with clearance-safe side."""
        x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
        x2 = min(W, x1 + side);    y2 = min(H, y1 + side)
        rh, rw = y2 - y1, x2 - x1
        if rh < 4 or rw < 4: return
        p = _rotate_patch(patch_src, angle, rh, rw)          # [3,rh,rw] differentiable
        # shape mask (rotated too if TILTED) so only the shape's pixels are placed
        smask = make_shape_mask(SHAPE, max(rh, rw), out.device)[:rh, :rw]  # [rh,rw]
        if TILTED:
            # rotate the mask to match the car angle (footprint follows the car)
            mm = smask.unsqueeze(0).unsqueeze(0)
            cos, sin = _m.cos(angle), _m.sin(angle)
            import torch as _t
            theta = _t.tensor([[cos,-sin,0.0],[sin,cos,0.0]], dtype=mm.dtype, device=mm.device).unsqueeze(0)
            grid = _t.nn.functional.affine_grid(theta, mm.size(), align_corners=False)
            smask = _t.nn.functional.grid_sample(mm, grid, align_corners=False, padding_mode="zeros")[0,0]
        smask = smask.unsqueeze(0)                            # [1,rh,rw]
        region = out[0, :, y1:y2, x1:x2]
        out[0, :, y1:y2, x1:x2] = smask * p + (1 - smask) * region

    for (xc, yc, bw, bh) in boxes_norm:
        box_w_px = bw * W; box_h_px = bh * H
        short_side = min(box_w_px, box_h_px)
        adapt_frac = FIXED_FRAC if FIXED_FRAC > 0 else (0.30 + max(0.0,min(1.0,(short_side-12.0)/28.0))*0.25)
        pw = max(4, min(int(adapt_frac*box_w_px), W))
        ph = max(4, min(int(adapt_frac*box_h_px), H))
        angle = estimate_car_angle(img, xc, yc, bw, bh, H, W)
        car_len = max(box_w_px, box_h_px)
        dx = _m.cos(angle); dy = _m.sin(angle)
        glass_half = (GLASS_BAND/2.0) * car_len
        patch_center_off = END_OFFSET * car_len
        max_half_glass = max(2.0, patch_center_off - glass_half - 1.0)
        max_half_end   = max(2.0, (0.5*car_len) - patch_center_off)
        max_half = min(max_half_glass, max_half_end)
        side = max(4, int(min(max(pw, ph), 2*max_half)))

        # which ends to place on: one end, or both (hood + trunk)
        ends = [+1, -1] if BOTH_ENDS else [+1]
        for sgn in ends:
            shift = sgn * END_OFFSET * car_len
            cx = int(xc*W + shift*dx); cy = int(yc*H + shift*dy)
            if MULTI_SHAPE:
                # place MULTI_N shapes near this end: stacked (overlapping, small jitter)
                # or apart (spread along the car WIDTH axis, still off glass)
                perp_dx, perp_dy = -dy, dx     # perpendicular = car width direction
                for k in range(MULTI_N):
                    if MULTI_SHAPE == "stacked":
                        # overlapping: tiny offsets around the end point
                        off = (k - (MULTI_N-1)/2.0) * (side*0.18)
                        mx = int(cx + off*perp_dx); my = int(cy + off*perp_dy)
                        _stamp(mx, my, side, angle, patch)
                    else:  # "apart": spread across the width, non-overlapping, STAY IN CAR
                        # clamp spread so shapes never exceed the car's half-width
                        car_wid = min(box_w_px, box_h_px)
                        small = max(4, int(side*0.55))
                        # available half-width for centers (leave room for the shape)
                        avail = max(0.0, 0.5*car_wid - small/2.0 - 1.0)
                        step = (2*avail)/(MULTI_N-1) if MULTI_N > 1 else 0
                        off = (k - (MULTI_N-1)/2.0) * step
                        mx = int(cx + off*perp_dx); my = int(cy + off*perp_dy)
                        _stamp(mx, my, small, angle, patch)
            else:
                _stamp(cx, cy, side, angle, patch)
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
    # margin term: push target(van) above source(car) by MARGIN
    margin_term = torch.clamp(src - tgt + MARGIN, min=0).sum()
    # POWER (confidence boost): also drive the target class's absolute PROBABILITY up,
    # so the flipped detection is confident enough to actually register (survive
    # the conf-threshold and NMS). Penalize (1 - van_prob) at the car locations.
    tgt_prob = cls[TGT_IDX, mask]
    conf_term = (1.0 - tgt_prob).sum()
    return margin_term + CONF_WEIGHT * conf_term

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
        x0 = (torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).float().unsqueeze(0)/255.0).to(DEVICE)
        data.append((os.path.splitext(os.path.basename(ip))[0], x0, load_src_boxes_norm(lbl)))
    print(f"DIGITAL PATCH {SRC_NAME}->{TGT_NAME}: training ONE universal patch on {len(data)} images")
    print(f"Params: PATCH_FRAC={PATCH_FRAC}, EPOCHS={EPOCHS}, LR={LR}, TV={TV_WEIGHT}, MARGIN={MARGIN}\n")

    # the patch we optimize (start from random)
    # WRAP: independent patterns. tiles mode -> WRAP_TILES patterns; contiguous -> one per zone (3)
    patch = torch.rand(3, PATCH_PX, PATCH_PX, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=LR)
    grad_momentum = torch.zeros_like(patch)   # POWER: momentum buffer (MI-style)

    for ep in range(EPOCHS):
        tot = 0.0
        for stem, x0, boxes in data:
            if not boxes: continue
            patched = apply_patch(x0, patch.clamp(0,1), boxes)
            loss = targeted_loss(raw(patched), boxes) + TV_WEIGHT * tv_loss(patch)
            opt.zero_grad(); loss.backward()
            # Optional momentum (disabled by default: MOMENTUM=0 -> plain Adam, which
            # beat the momentum version in Run 11). Only engages if MOMENTUM > 0.
            if MOMENTUM > 0:
                with torch.no_grad():
                    if patch.grad is not None:
                        g = patch.grad
                        gnorm = g.abs().mean() + 1e-12
                        grad_momentum.mul_(MOMENTUM).add_(g / gnorm)
                        patch.grad.copy_(grad_momentum)
            opt.step()
            with torch.no_grad(): patch.clamp_(0,1)
            tot += float(loss)
        print(f"  epoch {ep+1}/{EPOCHS}  avg loss {tot/max(1,len(data)):.3f}")

    # ---- evaluate the trained patch ----
    print("\nEvaluating trained patch (clean vs patched)...")
    n_src=0; native_bus=0; true_new=0; agg_clean=0; agg_patch=0
    final_patch = patch.detach().clamp(0,1)
    torch.save(final_patch, os.path.join(SAVE_DIR,"universal_patch.pt"))
    # also save a viewable PNG of the patch itself
    _pp = final_patch if final_patch.dim()==3 else final_patch[0]   # first tile if independent
    pimg = (_pp.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]
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
        torch.save({"clean":x0.cpu(), "patched":patched.detach().cpu(), "boxes_px":boxes_px},
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
