"""
Adversarial PGD attack on a YOLO object detector (ultralytics / YOLOv8-style).
Works with both a fine-tuned model (e.g. VisDrone best.pt) and stock yolov5su/yolov8.

WHAT IT DOES:
  1. Runs the detector on a clean image  -> saves 'result_before.jpg' (green boxes)
  2. Crafts a small PGD perturbation that suppresses detection confidence
  3. Runs the detector on the perturbed image -> saves 'result_after.jpg' (red boxes)
  4. Prints how many objects were detected before vs after

HOW TO USE:
  - Set MODEL_PATH to your weights  (e.g. "best.pt"  or  "yolov5su.pt")
  - Set IMAGE_PATH to your image
  - Set DEVICE to "cuda" if you have a GPU, else "cpu"
  - Run:  python attack.py
"""

import numpy as np
import torch
import cv2
from ultralytics import YOLO

# ------------------ SETTINGS (edit these) ------------------
MODEL_PATH = "best.pt"        # your fine-tuned model, or "yolov5su.pt" for stock
IMAGE_PATH = "trial_2.png" # the image to attack
DEVICE     = "cuda"           # "cuda" if you have a GPU, otherwise "cpu"

EPS   = 8.0 / 255.0   # perturbation budget (bigger = stronger + more visible)
ALPHA = 2.0 / 255.0   # step size per iteration
ITERS = 30            # number of PGD steps
CONF  = 0.25          # confidence threshold for counting/drawing detections
# -----------------------------------------------------------


def detection_confidence(raw_out):
    """Pull the per-candidate 'best class score' out of the raw model output.
    Output shape is [1, 4+num_classes, num_candidates]; rows 4: are class scores.
    This works regardless of how many classes the model has (10 or 80)."""
    pred = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    class_scores = pred[:, 4:, :]           # [1, num_classes, num_candidates]
    best_conf, _ = class_scores.max(dim=1)  # [1, num_candidates]
    return best_conf


def main():
    # --- Load model ---
    yolo = YOLO(MODEL_PATH)
    model = yolo.model.to(DEVICE).eval()

    # --- Load + prep image to 640x640, normalized to [0,1] ---
    orig = cv2.imread(IMAGE_PATH)
    if orig is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")
    img640 = cv2.resize(orig, (640, 640))
    # BGR->RGB, HWC->CHW, add batch dim, scale to 0-1
    x0 = torch.from_numpy(img640[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x0 = x0.to(DEVICE)

    # --- PGD attack: push detection confidence toward zero ---
    x_adv = x0.clone().detach()
    for i in range(ITERS):
        x_adv.requires_grad_(True)
        out = model(x_adv)
        loss = detection_confidence(out).sum()   # total confidence; we minimize it
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - ALPHA * grad.sign()          # step to lower confidence
        x_adv = torch.max(torch.min(x_adv, x0 + EPS), x0 - EPS)  # project into eps-ball
        x_adv = torch.clamp(x_adv, 0, 1)                       # keep valid pixel range

    # --- Count + draw detections for a given image tensor ---
    def count_and_draw(tensor_img, out_name, color):
        arr = (tensor_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        arr_bgr = arr[:, :, ::-1].copy()
        res = yolo.predict(arr_bgr, verbose=False, conf=CONF)[0]
        n = len(res.boxes)
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_name = yolo.names[int(b.cls[0])]
            score = float(b.conf[0])
            cv2.rectangle(arr_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(arr_bgr, f"{cls_name} {score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(out_name, arr_bgr)
        return n

    n_before = count_and_draw(x0,    "result_before.jpg", (0, 255, 0))  # green
    n_after  = count_and_draw(x_adv, "result_after.jpg",  (0, 0, 255))  # red

    # --- Also save a side-by-side comparison ---
    before_img = cv2.imread("result_before.jpg")
    after_img  = cv2.imread("result_after.jpg")
    side_by_side = np.hstack([before_img, after_img])
    cv2.imwrite("result_comparison.jpg", side_by_side)

    max_change = (x_adv - x0).abs().max().item() * 255
    print(f"Model:      {MODEL_PATH}")
    print(f"Image:      {IMAGE_PATH}")
    print(f"Detections BEFORE attack: {n_before}")
    print(f"Detections AFTER  attack: {n_after}")
    print(f"Max pixel change: {max_change:.1f}/255 (budget was {EPS*255:.0f})")
    print("Saved: result_before.jpg, result_after.jpg, result_comparison.jpg")


if __name__ == "__main__":
    main()
