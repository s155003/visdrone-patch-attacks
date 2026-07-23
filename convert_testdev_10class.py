"""
Convert VisDrone test-dev raw DET annotations -> YOLO format with ALL 10 classes
(for evaluating the 10-class YOLOv5 model).

Raw category is 1-indexed: 1=pedestrian..10=motor, 0=ignored, 11=others.
YOLO 10-class index = category - 1 (so category 1->0 ... 10->9). Skip 0 and 11.
"""
import os, glob, shutil
from pathlib import Path
import cv2

SRC = Path(os.path.expanduser("~/Downloads/VisDrone2019-DET-test-dev"))
OUT = Path(os.path.expanduser("~/attack-test/datasets/VisDrone_testdev_10class"))

def convert():
    ann_dir=SRC/"annotations"; img_dir=SRC/"images"
    out_img=OUT/"images"; out_lbl=OUT/"labels"
    out_img.mkdir(parents=True,exist_ok=True); out_lbl.mkdir(parents=True,exist_ok=True)
    n_img,n_box=0,0
    for ann in glob.glob(str(ann_dir/"*.txt")):
        stem=Path(ann).stem
        img_path=None
        for ext in [".jpg",".jpeg",".png"]:
            p=img_dir/f"{stem}{ext}"
            if p.exists(): img_path=p; break
        if img_path is None: continue
        img=cv2.imread(str(img_path))
        if img is None: continue
        H,W=img.shape[:2]
        lines=[]
        for line in open(ann):
            line=line.strip().rstrip(",")
            if not line: continue
            parts=line.split(",")
            if len(parts)<6: continue
            try:
                x,y,w,h=int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3])
                score,cat=int(parts[4]),int(parts[5])
            except ValueError: continue
            if score==0: continue
            if cat==0 or cat==11: continue   # ignored / others
            cls=cat-1                          # 1..10 -> 0..9
            xc=(x+w/2)/W; yc=(y+h/2)/H; nw=w/W; nh=h/H
            if nw<=0 or nh<=0: continue
            xc,yc=min(max(xc,0),1),min(max(yc,0),1); nw,nh=min(max(nw,0),1),min(max(nh,0),1)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        shutil.copy(img_path,out_img/img_path.name)
        with open(out_lbl/f"{stem}.txt","w") as f:
            f.write("\n".join(lines)+("\n" if lines else ""))
        n_img+=1; n_box+=len(lines)
    print(f"Converted {n_img} images, {n_box} boxes (10-class)")
    print(f"Output: {OUT}")

if __name__=="__main__": convert()
