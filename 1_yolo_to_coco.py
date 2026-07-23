"""
Convert VisDrone 5-class YOLO-format labels -> COCO JSON format,
which torchvision detection training expects.

Reads: images/{train,val} + labels/{train,val} (YOLO: class xc yc w h normalized)
Writes: annotations/instances_{train,val}.json (COCO format)

5 classes: 0=car,1=van,2=truck,3=bus,4=motor (COCO category ids 1-5)
"""
import os, glob, json
from pathlib import Path
import cv2

# point at the existing 5-class VisDrone data
BASE = Path(os.path.expanduser("~/attack-test/datasets/VisDrone"))
CLASS_NAMES = ["car","van","truck","bus","motor"]

def convert_split(split):
    img_dir = BASE/"images"/split
    lbl_dir = BASE/"labels"/split
    images, annotations = [], []
    ann_id = 1
    img_id = 1
    for img_path in sorted(glob.glob(str(img_dir/"*.jpg"))):
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        fname = os.path.basename(img_path)
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})
        lbl = lbl_dir/(Path(fname).stem + ".txt")
        if lbl.exists():
            for line in open(lbl):
                line=line.strip()
                if not line: continue
                p=line.split()
                if len(p)<5: continue
                cls=int(p[0]); xc,yc,w,h=map(float,p[1:5])
                # YOLO normalized -> COCO absolute [x,y,w,h] (top-left)
                bw=w*W; bh=h*H
                bx=(xc*W)-bw/2; by=(yc*H)-bh/2
                if bw<=0 or bh<=0: continue
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls+1,          # COCO cats are 1-indexed
                    "bbox": [bx, by, bw, bh],
                    "area": bw*bh, "iscrowd": 0,
                })
                ann_id+=1
        img_id+=1
    categories=[{"id":i+1,"name":n} for i,n in enumerate(CLASS_NAMES)]
    coco={"images":images,"annotations":annotations,"categories":categories}
    out_dir=BASE/"annotations"; out_dir.mkdir(exist_ok=True)
    out=out_dir/f"instances_{split}.json"
    with open(out,"w") as f: json.dump(coco,f)
    print(f"{split}: {len(images)} images, {len(annotations)} boxes -> {out}")

if __name__=="__main__":
    for split in ["train","val"]:
        convert_split(split)
    print("Done. COCO annotations written.")
