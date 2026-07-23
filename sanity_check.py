"""
Sanity check: is best.pt actually detecting well?
Runs the model on a few images, prints detection counts + confidences,
and saves annotated images so you can SEE how well it draws boxes.

USAGE: set the SETTINGS, then:  python3 sanity_check.py
"""

import os, glob
import cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "best.pt"
# Folder of images to check (uses your val images by default):
IMAGE_DIR  = "/home/aarav/Downloads/VisDrone2019-DET-val/images"
NUM_IMAGES = 3            # how many images to check
CONF       = 0.25         # confidence threshold
DEVICE     = "cuda"       # "cuda" or "cpu"
# =================================================

model = YOLO(MODEL_PATH)

# Show what the model thinks its classes are (confirms it's the right model)
print("Model classes:", model.names)
print()

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))[:NUM_IMAGES]
if not images:
    print(f"No .jpg images found in {IMAGE_DIR}")
    raise SystemExit

for img_path in images:
    res = model.predict(img_path, verbose=False, conf=CONF, device=DEVICE)[0]
    n = len(res.boxes)
    # tally detections per class
    counts = {}
    confs = []
    for b in res.boxes:
        name = model.names[int(b.cls[0])]
        counts[name] = counts.get(name, 0) + 1
        confs.append(float(b.conf[0]))
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    base = os.path.basename(img_path)
    print(f"{base}")
    print(f"   detections: {n}")
    print(f"   by class:   {counts}")
    print(f"   avg confidence: {avg_conf:.3f}")

    # save an annotated copy so you can LOOK at it
    annotated = res.plot()          # ultralytics draws the boxes for us
    out_name = f"check_{base}"
    cv2.imwrite(out_name, annotated)
    print(f"   saved annotated image: {out_name}")
    print()

print("Done. Open the check_*.jpg images to see how well the model detects.")
print("Good sign: many boxes on obvious vehicles with confidence > 0.5.")
print("Bad sign:  few/no boxes, or boxes in wrong places -> weights may be wrong.")
