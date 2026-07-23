"""
Train RetinaNet with a TRUE ResNet50-v2 backbone (via timm) + custom small-object
FPN, matching the paper's "Resnet50v2-RetinaNet" with the FPN alteration.

Uses timm's resnetv2_50 (genuine pre-activation ResNet-v2), NOT torchvision's
mislabeled retinanet..._v2. FPN alteration: adds the high-res stride-4 (160x160)
level and uses small-biased anchors, per the paper's small-object modification.

Run: cd ~/attack-test && ~/attack_env/bin/python train_retinanet_v2.py
(Uses attack_env — timm is available there via ultralytics; if not, agent installs timm)

Data: reuses the COCO-format annotations from the RetinaNet work.
"""
import os, json
from pathlib import Path
from collections import OrderedDict
import torch, cv2
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7

# ==================== SETTINGS ====================
BASE      = Path(os.path.expanduser("~/attack-test/datasets/VisDrone"))
FPN_MODE  = "custom"       # "custom" (small-object) or "standard"
IMG_SIZE  = 640
NUM_CLASSES = 5
EPOCHS    = 30
BATCH     = 8
LR        = 0.001          # same stable value as the v1 RetinaNet
WARMUP_ITERS = 500
DEVICE    = "cuda"
OUT_NAME  = f"retinanet_v2_{FPN_MODE}"
# =================================================
CLASS_NAMES=["car","van","truck","bus","motor"]

class ResNetV2FPN(nn.Module):
    def __init__(self, custom_fpn=True):
        super().__init__()
        out_indices=(1,2,3,4) if custom_fpn else (2,3,4)
        self.body=timm.create_model('resnetv2_50', pretrained=True,
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
    custom = (mode=="custom")
    backbone=ResNetV2FPN(custom_fpn=custom)
    if custom:
        anchor_sizes=((16,),(32,),(64,),(128,),(256,))
    else:
        anchor_sizes=((32,),(64,),(128,),(256,),(512,))
    ag=AnchorGenerator(anchor_sizes,((0.5,1.0,2.0),)*len(anchor_sizes))
    return RetinaNet(backbone,num_classes=NUM_CLASSES+1,anchor_generator=ag)

class CocoVisDrone(Dataset):
    def __init__(self, split):
        self.img_dir=BASE/"images"/split
        ann=json.load(open(BASE/"annotations"/f"instances_{split}.json"))
        self.images={im["id"]:im for im in ann["images"]}
        self.by_img={}
        for a in ann["annotations"]:
            self.by_img.setdefault(a["image_id"],[]).append(a)
        self.ids=list(self.images.keys())
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        iid=self.ids[i]; im=self.images[iid]
        img=cv2.imread(str(self.img_dir/im["file_name"]))
        H0,W0=img.shape[:2]
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=torch.from_numpy(img).permute(2,0,1).float()/255.0
        sx,sy=IMG_SIZE/W0,IMG_SIZE/H0
        boxes,labels=[],[]
        for a in self.by_img.get(iid,[]):
            x,y,w,h=a["bbox"]
            x1,y1,x2,y2=x*sx,y*sy,(x+w)*sx,(y+h)*sy
            if x2<=x1 or y2<=y1: continue
            boxes.append([x1,y1,x2,y2]); labels.append(a["category_id"])
        if boxes:
            boxes=torch.tensor(boxes,dtype=torch.float32); labels=torch.tensor(labels,dtype=torch.int64)
        else:
            boxes=torch.zeros((0,4),dtype=torch.float32); labels=torch.zeros((0,),dtype=torch.int64)
        return img,{"boxes":boxes,"labels":labels}

def collate(b): return tuple(zip(*b))

def main():
    print(f"RetinaNet ResNet50-v2 (timm) FPN={FPN_MODE}, {IMG_SIZE}px, {EPOCHS} epochs")
    ds=CocoVisDrone("train")
    dl=DataLoader(ds,batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=4)
    model=build_model(FPN_MODE).to(DEVICE)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                        lr=LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    gstep=0
    for ep in range(EPOCHS):
        model.train(); running=0.0
        for bi,(imgs,tgts) in enumerate(dl):
            if gstep<WARMUP_ITERS:
                for g in opt.param_groups: g['lr']=LR*(gstep+1)/WARMUP_ITERS
            imgs=[im.to(DEVICE) for im in imgs]
            tgts=[{k:v.to(DEVICE) for k,v in t.items()} for t in tgts]
            ld=model(imgs,tgts); loss=sum(ld.values())
            if not torch.isfinite(loss):
                print(f"  ep{ep+1} it{bi} non-finite, skip"); opt.zero_grad(); gstep+=1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
            opt.step(); running+=float(loss.detach()); gstep+=1
            if bi%50==0:
                print(f"  ep{ep+1}/{EPOCHS} it{bi}/{len(dl)} loss {float(loss.detach()):.3f} lr {opt.param_groups[0]['lr']:.5f}")
        if gstep>=WARMUP_ITERS: sched.step()
        print(f"Epoch {ep+1} done, avg loss {running/len(dl):.3f}")
        torch.save(model.state_dict(),f"{OUT_NAME}.pt")
    print(f"Saved {OUT_NAME}.pt")

if __name__=="__main__":
    main()
