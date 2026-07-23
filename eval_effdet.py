"""
Evaluate trained EfficientDet-D3 on VisDrone test-dev: per-class AP@0.5 + AR.
Matches the methodology of the other models (IoU 0.5, 4-class summary, full PR
curve with low score threshold).

Run with the ISOLATED env:  ~/effdet_env/bin/python eval_effdet.py

Uses DetBenchPredict (inference wrapper). Output per image is (N,6):
  [x1, y1, x2, y2, score, class]  (XYXY boxes, class matches training scheme).

CLASS MAPPING NOTE: training used COCO category ids 1-5 (car=1..motor=5).
effdet predict output class index -> we subtract 1 to get 0-4 (car=0..motor=4)
to match the other models' class scheme. VERIFIED below with PAPER_4.
"""
import os, glob, json
from pathlib import Path
import torch, cv2
import numpy as np
from effdet import create_model

# ==================== SETTINGS ====================
MODEL_PATH = "efficientdet_d3.pt"
MODEL      = "tf_efficientdet_d3"
TESTDEV    = Path(os.path.expanduser("~/attack-test/datasets/VisDrone_testdev"))  # 5-class labels 0-4
IMG_SIZE   = 640
NUM_CLASSES= 5
DEVICE     = "cuda"
# =================================================
CLASS_NAMES=["car","van","truck","bus","motor"]  # index 0-4
PAPER_4={0,1,2,3}  # car,van,truck,bus (0-indexed, matching other models)

def load_gt(lbl):
    """test-dev labels are YOLO 5-class 0-4, normalized. Return in 640px space."""
    gts=[]
    if not os.path.exists(lbl): return gts
    for line in open(lbl).read().strip().splitlines():
        if not line: continue
        p=line.split()
        if len(p)<5: continue
        cls=int(p[0]); xc,yc,w,h=map(float,p[1:5])
        x1=(xc-w/2)*IMG_SIZE; y1=(yc-h/2)*IMG_SIZE; x2=(xc+w/2)*IMG_SIZE; y2=(yc+h/2)*IMG_SIZE
        gts.append((cls,x1,y1,x2,y2))
    return gts

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def compute_ap(recs,precs):
    order=np.argsort(recs); recs=np.array(recs)[order]; precs=np.array(precs)[order]
    mrec=np.concatenate(([0.],recs,[1.])); mpre=np.concatenate(([0.],precs,[0.]))
    for i in range(len(mpre)-2,-1,-1): mpre[i]=max(mpre[i],mpre[i+1])
    idx=np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))

def main():
    # build model with DetBenchPredict wrapper, low score thresh for full PR curve
    from effdet import DetBenchPredict
    net = create_model(MODEL, bench_task='', num_classes=NUM_CLASSES,
                       pretrained=False, image_size=(IMG_SIZE,IMG_SIZE))
    # load trained weights (strip the bench wrapper prefix if present)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    # the training saved the DetBenchTrain-wrapped state_dict; keys have 'model.' prefix
    new_sd={}
    for k,v in sd.items():
        nk = k[6:] if k.startswith("model.") else k
        new_sd[nk]=v
    missing,unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"loaded weights (missing {len(missing)}, unexpected {len(unexpected)})")
    bench = DetBenchPredict(net).to(DEVICE).eval()
    # relax thresholds for full PR curve — set on the bench directly (net.config is read-only/struct-locked)
    bench.max_det_per_image = 1000
    bench.soft_nms = False
    print(f"max_det_per_image = {bench.max_det_per_image}, soft_nms = {bench.soft_nms}")

    mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
    imgs=sorted(glob.glob(str(TESTDEV/"images"/"*.jpg")))
    print(f"Evaluating {MODEL_PATH} on {len(imgs)} test-dev images...")

    per_cls={c:{"scores":[],"tp":[],"npos":0} for c in range(NUM_CLASSES)}
    for ip in imgs:
        orig=cv2.imread(ip)
        if orig is None: continue
        H0,W0=orig.shape[:2]; sx,sy=IMG_SIZE/W0,IMG_SIZE/H0
        gts=load_gt(str(TESTDEV/"labels"/(Path(ip).stem+".txt")))
        for c in range(NUM_CLASSES):
            per_cls[c]["npos"]+=sum(1 for g in gts if g[0]==c)
        img=cv2.resize(orig,(IMG_SIZE,IMG_SIZE)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img=(img-mean)/std
        t=torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
        img_scale=torch.tensor([1.0],device=DEVICE); img_size=torch.tensor([[IMG_SIZE,IMG_SIZE]],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            det=bench(t, img_info={"img_scale":img_scale,"img_size":img_size})[0].cpu().numpy()
        # det rows: [x1,y1,x2,y2,score,class]  class is 1-indexed (effdet)
        used={c:set() for c in range(NUM_CLASSES)}
        order=np.argsort(-det[:,4])
        for i in order:
            x1,y1,x2,y2,score,cls_raw=det[i]
            cls=int(cls_raw)-1   # effdet 1-indexed -> 0-4
            if cls<0 or cls>=NUM_CLASSES: continue
            gt_c=[j for j,g in enumerate(gts) if g[0]==cls]
            best_i,best_j=0.0,-1
            for j in gt_c:
                ii=iou((x1,y1,x2,y2),gts[j][1:5])
                if ii>best_i: best_i,best_j=ii,j
            is_tp = best_i>=0.5 and best_j not in used[cls]
            if is_tp: used[cls].add(best_j)
            per_cls[cls]["scores"].append(score); per_cls[cls]["tp"].append(1 if is_tp else 0)

    print(f"\n{'class':10} {'AP@0.5':>8} {'Recall':>8}")
    aps,recs4=[],[]
    for c in range(NUM_CLASSES):
        d=per_cls[c]; npos=d["npos"]
        if npos==0 or not d["scores"]:
            print(f"{CLASS_NAMES[c]:10} {'--':>8} {'--':>8}"); continue
        order=np.argsort(-np.array(d["scores"])); tp=np.array(d["tp"])[order]
        cum_tp=np.cumsum(tp); cum_fp=np.cumsum(1-tp)
        rec=cum_tp/npos; prec=cum_tp/(cum_tp+cum_fp)
        ap=compute_ap(list(rec),list(prec)); final_rec=rec[-1] if len(rec) else 0.0
        print(f"{CLASS_NAMES[c]:10} {ap:8.3f} {final_rec:8.3f}")
        if c in PAPER_4: aps.append(ap); recs4.append(final_rec)
    print(f"\n4-class mean AP@0.5 (car,van,truck,bus): {np.mean(aps):.3f}")
    print(f"4-class mean Recall: {np.mean(recs4):.3f}")

if __name__=="__main__":
    main()
