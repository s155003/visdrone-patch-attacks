import cv2, numpy as np
from ultralytics import YOLO

MODEL = "yolo11l_visdrone_pretrained.pt"
IMG   = "test_image.png"
OUT   = "test_image_detected.png"
NAMES = {0:"car",1:"van",2:"truck",3:"bus",4:"motor"}
COLORS = {0:(0,255,0),1:(255,128,0),2:(0,0,255),3:(0,255,255),4:(255,0,255)}  # BGR

model = YOLO(MODEL)
res = model.predict(IMG, verbose=False, conf=0.25, device="cpu")[0]
img = cv2.imread(IMG)
n = 0
for b in res.boxes:
    cls = int(b.cls[0]); conf = float(b.conf[0])
    x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
    color = COLORS.get(cls,(200,200,200)); name = NAMES.get(cls,str(cls))
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    cv2.putText(img,f"{name} {conf:.2f}",(x1,max(12,y1-4)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    n += 1
cv2.imwrite(OUT, img)
# print a per-class summary
from collections import Counter
counts = Counter(int(b.cls[0]) for b in res.boxes)
print(f"Total detections: {n}")
for c,ct in sorted(counts.items()):
    print(f"  {NAMES.get(c,c)}: {ct}")
print(f"Saved annotated image to {OUT}")
