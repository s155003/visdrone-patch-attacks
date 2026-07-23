"""
Train EfficientDet-D3 on VisDrone 5-class vehicles, using the effdet library.
Reuses the COCO-format annotations already generated for RetinaNet.

Run with the ISOLATED env:  ~/effdet_env/bin/python train_effdet.py
(effdet_env has torch 2.13; do NOT use attack_env)

Data expected (already exists from RetinaNet work):
  ~/attack-test/datasets/VisDrone/images/{train,val}/
  ~/attack-test/datasets/VisDrone/annotations/instances_{train,val}.json
"""
import os, math, json
from pathlib import Path
import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from effdet import create_model

# ==================== SETTINGS ====================
BASE      = Path(os.path.expanduser("~/attack-test/datasets/VisDrone"))
MODEL     = "tf_efficientdet_d3"
IMG_SIZE  = 640            # D3's native is 896, but we use 640 to match the paper/other models
NUM_CLASSES = 5
EPOCHS    = 30
BATCH     = 18
LR        = 0.002          # scaled up from 0.001 for batch 18
WARMUP_ITERS = 500
DEVICE    = "cuda"
OUT_NAME  = "efficientdet_d3"
# =================================================

class CocoDS(Dataset):
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
        iid = self.ids[i]; im = self.images[iid]
        img = cv2.imread(str(self.img_dir/im["file_name"]))
        H0,W0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        # normalize (ImageNet mean/std, as effdet expects)
        mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
        img=(img-mean)/std
        img=torch.from_numpy(img).permute(2,0,1).float()
        sx,sy=IMG_SIZE/W0, IMG_SIZE/H0
        boxes,labels=[],[]
        for a in self.by_img.get(iid,[]):
            x,y,w,h=a["bbox"]
            # effdet wants boxes as [ymin,xmin,ymax,xmax]
            y1,x1,y2,x2 = y*sy,(x)*sx,(y+h)*sy,(x+w)*sx
            if x2<=x1 or y2<=y1: continue
            boxes.append([y1,x1,y2,x2]); labels.append(a["category_id"])
        if not boxes:
            boxes=[[0,0,1,1]]; labels=[0]  # effdet needs at least one; dummy bg
        target={
            "bbox": torch.tensor(boxes,dtype=torch.float32),
            "cls":  torch.tensor(labels,dtype=torch.int64),
            "img_size": torch.tensor([IMG_SIZE,IMG_SIZE],dtype=torch.float32),
            "img_scale": torch.tensor([1.0],dtype=torch.float32),
        }
        return img, target

def collate(batch):
    imgs=torch.stack([b[0] for b in batch])
    # effdet bench expects target dict with padded tensors; build per-batch
    max_n=max(b[1]["bbox"].shape[0] for b in batch)
    bboxes=torch.zeros(len(batch),max_n,4)
    clss=torch.ones(len(batch),max_n,dtype=torch.int64)*-1  # -1 = ignore/pad
    for i,b in enumerate(batch):
        n=b[1]["bbox"].shape[0]
        bboxes[i,:n]=b[1]["bbox"]; clss[i,:n]=b[1]["cls"]
    img_size=torch.stack([b[1]["img_size"] for b in batch])
    img_scale=torch.cat([b[1]["img_scale"] for b in batch])
    target={"bbox":bboxes,"cls":clss,"img_size":img_size,"img_scale":img_scale}
    return imgs, target

def main():
    print(f"EfficientDet-D3 training | {IMG_SIZE}px | batch {BATCH} | {EPOCHS} epochs")
    ds=CocoDS("train")
    dl=DataLoader(ds,batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=4)
    # bench_task='train' wraps the model with the loss computation
    model=create_model(MODEL, bench_task='train', num_classes=NUM_CLASSES,
                       pretrained=True, image_size=(IMG_SIZE,IMG_SIZE),
                       bench_labeler=True).to(DEVICE)
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)

    gstep=0
    for ep in range(EPOCHS):
        model.train(); running=0.0
        for bi,(imgs,tgt) in enumerate(dl):
            if gstep<WARMUP_ITERS:
                for g in opt.param_groups: g['lr']=LR*(gstep+1)/WARMUP_ITERS
            imgs=imgs.to(DEVICE)
            tgt={k:v.to(DEVICE) for k,v in tgt.items()}
            out=model(imgs,tgt)
            loss=out['loss']
            if not torch.isfinite(loss):
                print(f"  ep{ep+1} it{bi} WARN non-finite loss, skip"); opt.zero_grad(); gstep+=1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
            opt.step(); running+=float(loss.detach()); gstep+=1
            if bi%50==0:
                print(f"  ep{ep+1}/{EPOCHS} it{bi}/{len(dl)} loss {float(loss.detach()):.3f} lr {opt.param_groups[0]['lr']:.5f}")
        if gstep>=WARMUP_ITERS: sched.step()
        print(f"Epoch {ep+1} done, avg loss {running/len(dl):.3f}")
        torch.save(model.state_dict(), f"{OUT_NAME}.pt")
    print(f"Saved {OUT_NAME}.pt")

if __name__=="__main__":
    main()
