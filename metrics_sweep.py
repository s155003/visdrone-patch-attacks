"""
Sweep several confidence thresholds and report Accuracy / Precision / Recall
at each, for a few metric definitions. Lets you SEE which threshold maximizes
each metric, so you can pick the best operating point.

USAGE: set SETTINGS, then:  python3 metrics_sweep.py
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
NUM_IMAGES = -1                       # -1 for all 548
IMG_SIZE   = 640
THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]   # values to try
VEHICLE_CLASSES = {3, 4, 5, 8, 9}
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

def counts(preds, gts, iou_t=0.5):
    matched,tp=set(),0
    for pr in sorted(preds,key=lambda z:-z[5]):
        best_i,best_j=0.0,-1
        for j,g in enumerate(gts):
            if j in matched or g[0]!=pr[0]: continue
            i=iou(pr[1:5],g[1:5])
            if i>best_i: best_i,best_j=i,j
        if best_i>=iou_t and best_j>=0:
            tp+=1; matched.add(best_j)
    return tp, len(preds)-tp, len(gts)-len(matched)

def run():
    yolo=YOLO(MODEL_PATH)
    imgs=sorted(glob.glob(os.path.join(VAL_IMAGES,"*.jpg")))
    if NUM_IMAGES>0: imgs=imgs[:NUM_IMAGES]
    print(f"Sweeping thresholds on {len(imgs)} images...\n")

    # Pre-run predictions at the LOWEST threshold once, then filter per-threshold
    # (predicting once at low conf and filtering is faster than re-predicting)
    min_conf=min(THRESHOLDS)
    per_image=[]   # (gts_all, gts_veh, gts_car, all_preds_low)
    for ip in imgs:
        orig=cv2.imread(ip)
        if orig is None: continue
        H,W=orig.shape[:2]; sx,sy=IMG_SIZE/W,IMG_SIZE/H
        ap=os.path.join(VAL_ANNOTS,os.path.splitext(os.path.basename(ip))[0]+".txt")
        img=cv2.resize(orig,(IMG_SIZE,IMG_SIZE))
        arr=img  # already BGR uint8
        with torch.no_grad():
            res=yolo.predict(arr[:,:,::-1].copy(),verbose=False,conf=min_conf,device=DEVICE)[0]
        preds=[(int(b.cls[0]),*b.xyxy[0].tolist(),float(b.conf[0])) for b in res.boxes]
        per_image.append((
            load_gt(ap,sx,sy,None),
            load_gt(ap,sx,sy,VEHICLE_CLASSES),
            load_gt(ap,sx,sy,{3}),
            preds
        ))

    for scope_name, scope_classes, gt_idx in [
        ("10-class", None, 0),
        ("5-vehicle", VEHICLE_CLASSES, 1),
        ("cars-only", {3}, 2),
    ]:
        print(f"=== {scope_name} (IoU 0.5) ===")
        print(f"{'CONF':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Accuracy':>9} {'Precision':>10} {'Recall':>8}")
        for th in THRESHOLDS:
            TP=FP=FN=0
            for gts_all, gts_veh, gts_car, preds in per_image:
                gts=[gts_all,gts_veh,gts_car][gt_idx]
                fp_preds=[p for p in preds if p[5]>=th and (scope_classes is None or p[0] in scope_classes)]
                tp,fp,fn=counts(fp_preds,gts)
                TP+=tp; FP+=fp; FN+=fn
            acc=TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0
            prec=TP/(TP+FP) if (TP+FP)>0 else 0
            rec=TP/(TP+FN) if (TP+FN)>0 else 0
            print(f"{th:6.2f} {TP:6d} {FP:6d} {FN:6d} {acc:9.3f} {prec:10.3f} {rec:8.3f}")
        print()

if __name__=="__main__":
    run()
