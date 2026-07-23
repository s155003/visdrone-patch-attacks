"""
Evaluate AP and AR on VisDrone test-dev at BOTH 640 (paper-comparable) and
1280 (model's native training resolution). Reports per-class + 4-class summary
for each resolution, side by side.

USAGE: edit MODEL_PATH, then: cd ~/attack-test && ~/attack_env/bin/python eval_ap_ar_both.py
"""
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "yolo11l_combined_scratch.pt"   # change per model
DATA_YAML  = "visdrone_testdev.yaml"
IOU        = 0.5
DEVICE     = 0
RESOLUTIONS = [640, 1280]      # run both
PAPER_CLASSES = {0:'car', 1:'van', 2:'truck', 3:'bus'}  # excludes motor(4)
# =================================================

def run_at(model, imgsz):
    metrics = model.val(data=DATA_YAML, imgsz=imgsz, iou=IOU,
                        conf=0.001, device=DEVICE, verbose=False, split="val")
    names = model.names
    ap50 = metrics.box.ap50
    recall = metrics.box.r
    idx = metrics.box.ap_class_index
    rows = []
    paper_ap, paper_r = [], []
    for i, c in enumerate(idx):
        c = int(c)
        rows.append((names[c], float(ap50[i]), float(recall[i])))
        if c in PAPER_CLASSES:
            paper_ap.append(float(ap50[i])); paper_r.append(float(recall[i]))
    mean_ap = sum(paper_ap)/len(paper_ap) if paper_ap else 0
    mean_r  = sum(paper_r)/len(paper_r) if paper_r else 0
    return rows, mean_ap, mean_r, float(metrics.box.map50)

model = YOLO(MODEL_PATH)
print(f"\n=== {MODEL_PATH} on VisDrone test-dev ===\n")

results = {}
for res in RESOLUTIONS:
    print(f"Running at {res}px ...")
    results[res] = run_at(model, res)

# print side by side
print(f"\n{'':10} " + "  ".join(f"{'@'+str(r)+'px AP':>12} {'Rec':>6}" for r in RESOLUTIONS))
# gather class names (same across res)
classes = [row[0] for row in results[RESOLUTIONS[0]][0]]
for ci, cname in enumerate(classes):
    line = f"{cname:10} "
    for r in RESOLUTIONS:
        rows = results[r][0]
        ap = rows[ci][1]; rec = rows[ci][2]
        line += f"{ap:12.3f} {rec:6.3f}  "
    print(line)

print(f"\n==== 4-CLASS SUMMARY (car, van, truck, bus) ====")
print(f"{'resolution':12} {'mean AP@0.5':>12} {'mean Recall':>12} {'(5-cls mAP50)':>14}")
for r in RESOLUTIONS:
    _, mean_ap, mean_r, map50 = results[r]
    tag = "paper-match" if r==640 else "native-res"
    print(f"{str(r)+'px':12} {mean_ap:12.3f} {mean_r:12.3f} {map50:14.3f}   <- {tag}")
