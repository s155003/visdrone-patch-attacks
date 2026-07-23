"""
Train RetinaNet-ResNet50 on VisDrone 5-class, at 640px (paper resolution).
Supports STANDARD FPN or the paper-style CUSTOM small-object FPN via FPN_MODE.

Uses torchvision detection. Data in COCO format (from 1_yolo_to_coco.py).

USAGE: edit SETTINGS, then: cd ~/attack-test && ~/attack_env/bin/python 2_train_retinanet.py
"""
import os, math
from pathlib import Path
import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7
import json

# ==================== SETTINGS ====================
BASE       = Path(os.path.expanduser("~/attack-test/datasets/VisDrone"))
FPN_MODE   = "custom"       # "standard" or "custom"  <-- change per run
IMG_SIZE   = 640              # paper resolution
NUM_CLASSES= 5                # 5 vehicle classes (+ background handled internally)
EPOCHS     = 30
BATCH      = 8
LR         = 0.001            # lowered from 0.005 (NaN fix)
DEVICE     = "cuda"
OUT_NAME   = f"retinanet_{FPN_MODE}"   # weights saved as OUT_NAME.pt
# =================================================

CLASS_NAMES=["car","van","truck","bus","motor"]

class CocoVisDrone(Dataset):
    def __init__(self, split):
        self.img_dir = BASE/"images"/split
        ann = json.load(open(BASE/"annotations"/f"instances_{split}.json"))
        self.images = {im["id"]: im for im in ann["images"]}
        self.by_img = {}
        for a in ann["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a)
        self.ids = list(self.images.keys())
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]
        im = self.images[iid]
        path = self.img_dir/im["file_name"]
        img = cv2.imread(str(path))
        H0,W0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0
        sx, sy = IMG_SIZE/W0, IMG_SIZE/H0
        boxes, labels = [], []
        for a in self.by_img.get(iid, []):
            x,y,w,h = a["bbox"]
            x1,y1,x2,y2 = x*sx, y*sy, (x+w)*sx, (y+h)*sy
            if x2<=x1 or y2<=y1: continue
            boxes.append([x1,y1,x2,y2]); labels.append(a["category_id"])
        if boxes:
            boxes=torch.tensor(boxes,dtype=torch.float32)
            labels=torch.tensor(labels,dtype=torch.int64)
        else:
            boxes=torch.zeros((0,4),dtype=torch.float32)
            labels=torch.zeros((0,),dtype=torch.int64)
        return img, {"boxes":boxes,"labels":labels}

def collate(batch): return tuple(zip(*batch))

def build_model(mode):
    if mode=="standard":
        # true torchvision default: C3,C4,C5 + P6,P7 = 5 levels (finest 80x80)
        backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT",
                                       returned_layers=[2,3,4],
                                       extra_blocks=LastLevelP6P7(256,256))
        anchor_sizes=((32,),(64,),(128,),(256,),(512,))   # 5 levels
    else:  # custom (paper-style): add high-res C2 (160x160), drop coarse top -> 5 levels
        backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT",
                                       returned_layers=[1,2,3,4],
                                       extra_blocks=LastLevelMaxPool())
        anchor_sizes=((16,),(32,),(64,),(128,),(256,))    # 5 levels, small-biased
    aspect=((0.5,1.0,2.0),)*len(anchor_sizes)
    ag=AnchorGenerator(anchor_sizes,aspect)
    # num_classes+1 for background
    return RetinaNet(backbone,num_classes=NUM_CLASSES+1,anchor_generator=ag)

def main():
    print(f"Training RetinaNet-ResNet50, FPN_MODE={FPN_MODE}, {IMG_SIZE}px, {EPOCHS} epochs")
    ds=CocoVisDrone("train")
    dl=DataLoader(ds,batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=4)
    model=build_model(FPN_MODE).to(DEVICE)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                        lr=LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)

    WARMUP_ITERS = 500   # linear LR warmup over first 500 iters (NaN-prevention)
    global_step = 0
    for ep in range(EPOCHS):
        model.train()
        running=0.0
        for bi,(imgs,tgts) in enumerate(dl):
            # LR warmup: ramp from ~0 to LR over WARMUP_ITERS
            if global_step < WARMUP_ITERS:
                warmup_factor = (global_step + 1) / WARMUP_ITERS
                for g in opt.param_groups:
                    g['lr'] = LR * warmup_factor
            imgs=[im.to(DEVICE) for im in imgs]
            tgts=[{k:v.to(DEVICE) for k,v in t.items()} for t in tgts]
            loss_dict=model(imgs,tgts)
            loss=sum(loss_dict.values())
            # skip any non-finite loss instead of letting it poison the model
            if not torch.isfinite(loss):
                print(f"  ep{ep+1}/{EPOCHS} it{bi}/{len(dl)} WARN non-finite loss, skipping batch")
                opt.zero_grad(); global_step+=1; continue
            opt.zero_grad()
            loss.backward()
            # gradient clipping: cap gradient norm to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            running+=float(loss.detach())
            global_step+=1
            if bi%50==0:
                cur_lr = opt.param_groups[0]['lr']
                print(f"  ep{ep+1}/{EPOCHS} it{bi}/{len(dl)} loss {float(loss.detach()):.3f} lr {cur_lr:.5f}")
        # only step cosine schedule after warmup is done
        if global_step >= WARMUP_ITERS:
            sched.step()
        print(f"Epoch {ep+1} done, avg loss {running/len(dl):.3f}")
        torch.save(model.state_dict(), f"{OUT_NAME}.pt")
    print(f"Saved {OUT_NAME}.pt")

if __name__=="__main__":
    main()
