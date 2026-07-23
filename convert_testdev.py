"""
Convert VisDrone test-dev raw DET annotations -> YOLO format (5-class vehicles),
so ultralytics can run .val() on it. Produces a labels/ folder alongside images.

Raw line: x,y,w,h,score,category,trunc,occ  (category 1-indexed, pixels)
VisDrone category -> 5-class index:
  4(car)->0, 5(van)->1, 6(truck)->2, 9(bus)->3, 10(motor)->4
(We keep all 5 vehicle classes; the 4-class paper metric just reads car/van/truck/bus.)
"""
import os, glob, shutil
from pathlib import Path
import cv2

SRC = Path(os.path.expanduser("~/Downloads/VisDrone2019-DET-test-dev"))
# Build a YOLO-format dataset dir ultralytics can read
OUT = Path(os.path.expanduser("~/attack-test/datasets/VisDrone_testdev"))

CAT_TO_CLASS = {4:0, 5:1, 6:2, 9:3, 10:4}

def convert():
    ann_dir = SRC / "annotations"
    img_dir = SRC / "images"
    out_img = OUT / "images"
    out_lbl = OUT / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_img, n_box = 0, 0
    for ann in glob.glob(str(ann_dir / "*.txt")):
        stem = Path(ann).stem
        # find image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p; break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        lines = []
        for line in open(ann):
            line = line.strip().rstrip(",")
            if not line: continue
            parts = line.split(",")
            if len(parts) < 6: continue
            try:
                x,y,w,h = int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3])
                score,cat = int(parts[4]),int(parts[5])
            except ValueError:
                continue
            if score == 0: continue
            if cat not in CAT_TO_CLASS: continue
            cls = CAT_TO_CLASS[cat]
            xc=(x+w/2)/W; yc=(y+h/2)/H; nw=w/W; nh=h/H
            if nw<=0 or nh<=0: continue
            xc,yc=min(max(xc,0),1),min(max(yc,0),1)
            nw,nh=min(max(nw,0),1),min(max(nh,0),1)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # copy image, write label (even if empty -> valid negative)
        shutil.copy(img_path, out_img / img_path.name)
        with open(out_lbl / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1; n_box += len(lines)

    print(f"Converted {n_img} images, {n_box} vehicle boxes")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    convert()
