"""THROWAWAY: eyeball where the targeted patch lands (AREA_FRAC=0.20), CPU-only.
Imports patch_attack_targeted (no CUDA at import), composites an UNTRAINED random
patch onto 3 test images that have cars, draws the GT car boxes (yellow), saves PNGs.
No model is loaded; no GPU is touched."""
import os, glob
import numpy as np, cv2, torch
import patch_attack_targeted as T          # import-safe: no CUDA at import

OUT = "targeted_preview"
os.makedirs(OUT, exist_ok=True)
IMG = T.IMG_SIZE                            # 640
FRAC = 0.20
img_dir = T.EVAL_IMAGES                     # datasets/VisDrone/images/test
lbl_dir = getattr(T, "EVAL_LABELS", img_dir.replace("/images/", "/labels/"))

rng = np.random.RandomState(T.SEED)
patch = torch.rand((3, T.PATCH_PX, T.PATCH_PX))    # untrained random, CPU

picked = 0
for ip in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
    stem = os.path.splitext(os.path.basename(ip))[0]
    boxes, _drop = T.load_car_boxes(os.path.join(lbl_dir, stem + ".txt"))
    if not boxes:
        continue
    orig = cv2.imread(ip)
    if orig is None:
        continue
    img = cv2.resize(orig, (IMG, IMG))
    x0 = (torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
          .float().unsqueeze(0) / 255.0)              # CPU tensor, RGB
    patched = T.apply_patch(x0, patch, boxes, FRAC, rng).clamp(0, 1)
    pbgr = (patched[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
    # draw GT car boxes (yellow) so placement is judgeable vs the car body
    for (_, xc, yc, bw, bh) in boxes:
        x1, y1 = int((xc - bw / 2) * IMG), int((yc - bh / 2) * IMG)
        x2, y2 = int((xc + bw / 2) * IMG), int((yc + bh / 2) * IMG)
        cv2.rectangle(pbgr, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.rectangle(pbgr, (0, 0), (pbgr.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(pbgr, f"{stem}  AREA_FRAC={FRAC}  {len(boxes)} cars  yellow=GT car box",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    outp = os.path.join(OUT, f"place_{picked+1}_{stem}.png")
    cv2.imwrite(outp, pbgr)
    print(f"[preview] {outp}  ({len(boxes)} cars)")
    picked += 1
    if picked >= 3:
        break
print(f"[preview] wrote {picked} images to {OUT}/")
