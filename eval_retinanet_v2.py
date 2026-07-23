import os, glob, json
from pathlib import Path
from collections import OrderedDict
import torch, cv2
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7

MODEL_PATH = "retinanet_v2_custom.pt"
FPN_MODE   = "custom"
TESTDEV    = Path(os.path.expanduser("~/attack-test/datasets/VisDrone_testdev"))
IMG_SIZE   = 640
DEVICE     = "cuda"
NUM_CLASSES= 5
CLASS_NAMES=["car","van","truck","bus","motor"]
PAPER_4={0,1,2,3}

class ResNetV2FPN(nn.Module):
    def __init__(self, custom_fpn=True):
        super().__init__()
        out_indices=(1,2,3,4) if custom_fpn else (2,3,4)
        self.body=timm.create_model('resnetv2_50', pretrained=False,
                                     features_only=True, out_indices=out_indices)
        in_ch=self.body.feature_info.channels()
        extra = LastLevelMaxPool() if custom_fpn else LastLevelP6P7(256,256)
        self.fpn=FeaturePyramidNetwork(in_channels_list=in_ch, out_channels=256,
                                       extra_blocks=extra)
        self.out_channels=256
    def forward(self,x):
        feats=self.body(x)
        d=OrderedDict((str(i),f) for i,f in enumerate(feats))
        return self.fpn(d)

def build_model(mode):
    custom=(mode=="custom")
    backbone=ResNetV2FPN(custom_fpn=custom)
    if custom: anchor_sizes=((16,),(32,),(64,),(128,),(256,))
    else:      anchor_sizes=((32,),(64,),(128,),(256,),(512,))
    ag=AnchorGenerator(anchor_sizes,((0.5,1.0,2.0),)*len(anchor_sizes))
    return RetinaNet(backbone,num_classes=NUM_CLASSES+1,anchor_generator=ag)

def load_gt(lbl):
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
    model=build_model(FPN_MODE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.score_thresh=0.001; model.detections_per_img=1000
    model.to(DEVICE).eval()
    imgs=sorted(glob.glob(str(TESTDEV/"images"/"*.jpg")))
    print(f"Evaluating {MODEL_PATH} ({FPN_MODE}, timm ResNet50-v2) on {len(imgs)} test-dev images...")
    per_cls={c:{"scores":[],"tp":[],"npos":0} for c in range(NUM_CLASSES)}
    for ip in imgs:
        orig=cv2.imread(ip)
        if orig is None: continue
        gts=load_gt(str(TESTDEV/"labels"/(Path(ip).stem+".txt")))
        for c in range(NUM_CLASSES): per_cls[c]["npos"]+=sum(1 for g in gts if g[0]==c)
        img=cv2.resize(orig,(IMG_SIZE,IMG_SIZE)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        t=torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0
        with torch.no_grad(): out=model(t)[0]
        boxes=out["boxes"].cpu().numpy(); scores=out["scores"].cpu().numpy(); labels=out["labels"].cpu().numpy()
        used={c:set() for c in range(NUM_CLASSES)}
        for i in np.argsort(-scores):
            cls=int(labels[i])-1
            if cls<0 or cls>=NUM_CLASSES: continue
            gt_c=[j for j,g in enumerate(gts) if g[0]==cls]
            best_i,best_j=0.0,-1
            for j in gt_c:
                ii=iou(boxes[i],gts[j][1:5])
                if ii>best_i: best_i,best_j=ii,j
            is_tp=best_i>=0.5 and best_j not in used[cls]
            if is_tp: used[cls].add(best_j)
            per_cls[cls]["scores"].append(scores[i]); per_cls[cls]["tp"].append(1 if is_tp else 0)
    print(f"\n{'class':10} {'AP@0.5':>8} {'Recall':>8}")
    aps,recs4=[],[]
    for c in range(NUM_CLASSES):
        d=per_cls[c]; npos=d["npos"]
        if npos==0 or not d["scores"]:
            print(f"{CLASS_NAMES[c]:10} {'--':>8} {'--':>8}"); continue
        order=np.argsort(-np.array(d["scores"])); tp=np.array(d["tp"])[order]
        cum_tp=np.cumsum(tp); cum_fp=np.cumsum(1-tp)
        rec=cum_tp/npos; prec=cum_tp/(cum_tp+cum_fp)
        ap=compute_ap(list(rec),list(prec)); fr=rec[-1] if len(rec) else 0.0
        print(f"{CLASS_NAMES[c]:10} {ap:8.3f} {fr:8.3f}")
        if c in PAPER_4: aps.append(ap); recs4.append(fr)
    print(f"\n4-class mean AP@0.5 (car,van,truck,bus): {np.mean(aps):.3f}")
    print(f"4-class mean Recall: {np.mean(recs4):.3f}")

if __name__=="__main__": main()
