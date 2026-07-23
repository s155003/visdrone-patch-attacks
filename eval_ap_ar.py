"""
Evaluate AP and AR (paper methodology) for a model on VisDrone test-dev.
Uses ultralytics' built-in .val() -> per-class AP@0.5 and recall, IoU 0.5, 640px.

Paper reports 4 classes: car, van, truck, bus (NOT motor). We compute all 5,
then read off the 4 the paper uses.

USAGE: edit MODEL_PATH, then: cd ~/attack-test && ~/attack_env/bin/python eval_ap_ar.py
"""
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "yolo11l_visdrone_pretrained.pt"   # change per model
DATA_YAML  = "visdrone_testdev.yaml"
IMG_SIZE   = 640          # matches paper
IOU        = 0.5          # AP@0.5, matches paper
DEVICE     = 0            # GPU (free now)
# =================================================

model = YOLO(MODEL_PATH)
print(f"\nEvaluating {MODEL_PATH} on VisDrone test-dev (AP/AR @ IoU 0.5, {IMG_SIZE}px)\n")

metrics = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    iou=IOU,
    conf=0.001,           # low conf so AP integrates the full PR curve (standard)
    device=DEVICE,
    verbose=True,
    split="val",
)

names = model.names
print("\n==== PER-CLASS AP@0.5 and RECALL ====")
print(f"{'class':10} {'AP@0.5':>8} {'Recall':>8}")
# metrics.box has per-class arrays
ap50 = metrics.box.ap50      # per-class AP@0.5
recall = metrics.box.r       # per-class recall
maps_idx = metrics.box.ap_class_index  # which class each row is
paper_classes = {0:'car', 1:'van', 2:'truck', 3:'bus'}
paper_ap, paper_r = [], []
for i, c in enumerate(maps_idx):
    nm = names[int(c)]
    a = float(ap50[i]); r = float(recall[i])
    print(f"{nm:10} {a:8.3f} {r:8.3f}")
    if int(c) in paper_classes:
        paper_ap.append(a); paper_r.append(r)

print("\n==== PAPER-STYLE 4-CLASS SUMMARY (car, van, truck, bus) ====")
if paper_ap:
    print(f"mean AP@0.5 (4-class): {sum(paper_ap)/len(paper_ap):.3f}")
    print(f"mean Recall (4-class): {sum(paper_r)/len(paper_r):.3f}")
print(f"\n(All-5-class mAP50 from ultralytics: {metrics.box.map50:.3f})")
