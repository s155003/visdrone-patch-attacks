"""
AP/AR eval for the 10-CLASS YOLOv5-small model on VisDrone test-dev.
Same methodology (IoU 0.5, dual-resolution) but reads the 10-class indices.

The YOLOv5 model outputs 10 VisDrone classes:
  0=pedestrian,1=people,2=bicycle,3=car,4=van,5=truck,6=tricycle,
  7=awning-tricycle,8=bus,9=motor
Paper's 4 vehicle classes in THIS model's indexing: car=3, van=4, truck=5, bus=8.

IMPORTANT: the test-dev labels we converted are 5-class (0-4). For the 10-class
model, ultralytics .val() needs labels matching ITS class scheme. So this script
uses a SEPARATE 10-class test-dev label set + yaml (see convert note).

USAGE: cd ~/attack-test && ~/attack_env/bin/python eval_ap_ar_yolov5.py
"""
from ultralytics import YOLO

MODEL_PATH = "best.pt"                       # the YOLOv5-small 10-class model
DATA_YAML  = "visdrone_testdev_10class.yaml" # 10-class labels version
IOU        = 0.5
DEVICE     = 0
RESOLUTIONS = [640, 1280]
# paper's 4 classes in 10-class indexing:
PAPER_CLASSES = {3:'car', 4:'van', 5:'truck', 8:'bus'}

def run_at(model, imgsz):
    m = model.val(data=DATA_YAML, imgsz=imgsz, iou=IOU, conf=0.001,
                  device=DEVICE, verbose=False, split="val")
    names = model.names
    ap50=m.box.ap50; recall=m.box.r; idx=m.box.ap_class_index
    rows=[]; pap=[]; prc=[]
    for i,c in enumerate(idx):
        c=int(c)
        rows.append((names[c], float(ap50[i]), float(recall[i])))
        if c in PAPER_CLASSES:
            pap.append(float(ap50[i])); prc.append(float(recall[i]))
    mean_ap=sum(pap)/len(pap) if pap else 0
    mean_r=sum(prc)/len(prc) if prc else 0
    return rows, mean_ap, mean_r, float(m.box.map50)

model=YOLO(MODEL_PATH)
print(f"\n=== {MODEL_PATH} (10-class YOLOv5) on VisDrone test-dev ===\n")
results={}
for res in RESOLUTIONS:
    print(f"Running at {res}px ...")
    results[res]=run_at(model,res)

print(f"\n4 paper classes = car(3), van(4), truck(5), bus(8)\n")
print(f"{'resolution':12} {'mean AP@0.5':>12} {'mean Recall':>12}")
for r in RESOLUTIONS:
    _,ap,rc,_=results[r]
    print(f"{str(r)+'px':12} {ap:12.3f} {rc:12.3f}")

# also show the 4 relevant per-class rows
print(f"\nPer-class (paper's 4) at each resolution:")
for r in RESOLUTIONS:
    print(f"  @{r}px:")
    for nm,ap,rc in results[r][0]:
        if nm in ['car','van','truck','bus']:
            print(f"    {nm:8} AP {ap:.3f}  Rec {rc:.3f}")
