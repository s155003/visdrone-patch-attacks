"""
Report the model's performance under SEVERAL different 'accuracy' definitions,
so you can see which metric gives which number. Run on your val set.

USAGE: set SETTINGS, then:  python3 metrics_report.py
"""
import os, glob
import numpy as np
import torch, cv2
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = "best.pt"
VAL_IMAGES = "/home/aarav/Downloads/VisDrone2019-DET-val/images"
VAL_ANNOTS = "/home/aarav/Downloads/VisDrone2019-DET-val/annotations"
DEVICE     = "cuda"
NUM_IMAGES = 50
IMG_SIZE   = 640
CONF       = 0.25
VEHICLE_CLASSES = {3, 4, 5, 8, 9}   # car, van, truck, bus, motor
# =================================================

def load_gt(ap, sx, sy, classes=None):
    gts = []
    if not os.path.exists(ap): return gts
    for line in open(ap).read().strip().splitlines():
        if not line: continue
        p = line.split(",")
        if len(p) < 6: continue
        x,y,w,h = int(p[0]),int(p[1]),int(p[2]),int(p[3])
        score,cat = int(p[4]),int(p[5])
        if score==0 or cat==0 or cat==11: continue
        cls = cat-1
        if classes and cls not in classes: continue
        gts.append((cls, x*sx, y*sy, (x+w)*sx, (y+h)*sy))
    return gts

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def counts(preds, gts, iou_t):
    matched,tp,correct_cls=set(),0,0
    for pr in sorted(preds,key=lambda z:-z[5]):
        best_i,best_j=0.0,-1
        for j,g in enumerate(gts):
            if j in matched: continue
            i=iou(pr[1:5],g[1:5])
            if i>best_i: best_i,best_j=i,j
        if best_i>=iou_t and best_j>=0:
            # localization matched; check class
            if gts[best_j][0]==pr[0]:
                tp+=1; correct_cls+=1
            matched.add(best_j)
    fp=len(preds)-tp; fn=len(gts)-len(matched)
    return tp,fp,fn

def predict(yolo,t,classes=None):
    arr=(t[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    with torch.no_grad():
        res=yolo.predict(arr[:,:,::-1].copy(),verbose=False,conf=CONF)[0]
    out=[]
    for b in res.boxes:
        c=int(b.cls[0])
        if classes and c not in classes: continue
        out.append((c,*b.xyxy[0].tolist(),float(b.conf[0])))
    return out

def run():
    yolo=YOLO(MODEL_PATH)
    imgs=sorted(glob.glob(os.path.join(VAL_IMAGES,"*.jpg")))[:NUM_IMAGES]
    print(f"Evaluating {len(imgs)} images under several metric definitions...\n")

    # accumulate for various settings
    S = {k:[0,0,0] for k in ["10cls@0.5","10cls@0.3","5veh@0.5","5veh@0.3","cars@0.5"]}

    for ip in imgs:
        orig=cv2.imread(ip)
        if orig is None: continue
        H,W=orig.shape[:2]; sx,sy=IMG_SIZE/W,IMG_SIZE/H
        ap=os.path.join(VAL_ANNOTS,os.path.splitext(os.path.basename(ip))[0]+".txt")
        img=cv2.resize(orig,(IMG_SIZE,IMG_SIZE))
        x0=torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).float().unsqueeze(0)/255.0

        p_all=predict(yolo,x0)
        p_veh=[d for d in p_all if d[0] in VEHICLE_CLASSES]
        p_car=[d for d in p_all if d[0]==3]

        for key,(preds,classes,iou_t) in {
            "10cls@0.5":(p_all,None,0.5),
            "10cls@0.3":(p_all,None,0.3),
            "5veh@0.5":(p_veh,VEHICLE_CLASSES,0.5),
            "5veh@0.3":(p_veh,VEHICLE_CLASSES,0.3),
            "cars@0.5":(p_car,{3},0.5),
        }.items():
            g=load_gt(ap,sx,sy,classes)
            tp,fp,fn=counts(preds,g,iou_t)
            S[key][0]+=tp; S[key][1]+=fp; S[key][2]+=fn

    print(f"{'Metric':14} {'TP':>6} {'FP':>6} {'FN':>6} {'Accuracy':>9} {'Precision':>10} {'Recall':>8}")
    for k,(tp,fp,fn) in S.items():
        acc=tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0
        prec=tp/(tp+fp) if (tp+fp)>0 else 0
        rec=tp/(tp+fn) if (tp+fn)>0 else 0
        print(f"{k:14} {tp:6d} {fp:6d} {fn:6d} {acc:9.3f} {prec:10.3f} {rec:8.3f}")
    print("\nNote: 'Accuracy' = TP/(TP+FP+FN). 'Precision' = TP/(TP+FP) (of what I detected, how much was right).")
    print("Precision is often what people mean by 'accuracy' colloquially - and it's much higher.")

if __name__=="__main__":
    run()
