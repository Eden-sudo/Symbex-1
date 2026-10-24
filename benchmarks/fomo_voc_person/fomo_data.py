"""
fomo_data.py -- the data of the fomo_voc_person benchmark: person detection, FOMO-style, from Pascal VOC.

    python fomo_data.py                 # downloads VOC 2007 + 2012 once (~2 GB) and writes data/fomo_*.npz

FOMO (Edge Impulse's "Faster Objects, More Objects") does not draw boxes: it splits the image into a grid and says,
for every cell, whether the CENTER of an object falls in it. That is a classification per cell -- the task binary
networks do well -- and it locates every person in the image.
    images     96 x 96 RGB uint8 (the whole image resized: no crop)
    grid       12 x 12 cells of 8 x 8 pixels: class 1 where the center of a person box falls, 0 elsewhere
    centers    the person centers themselves (in 96 x 96 pixels), for the F1 score of the detections
Splits, as the VOC standard: training = VOC 2007 trainval + VOC 2012 trainval; test = VOC 2007 test. Every image is
kept, with or without persons (the network must also learn to say "nobody here"); boxes marked "difficult" are left
out of the targets, as in the VOC evaluation.
Writes data/fomo_train.npz and data/fomo_test.npz (x, y, centers, image_of_center).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOC_DIR = os.path.join(HERE, "..", "data", "voc")
SIZE, CELL = 96, 8
GRID = SIZE // CELL


def voc(year, split):
    import torchvision
    try:
        return torchvision.datasets.VOCDetection(VOC_DIR, year=year, image_set=split, download=True)
    except Exception as e:                       # the official server is sometimes down
        sys.exit(f"[X] could not download VOC {year} {split} ({e}).\n    Download the tar files by hand (a mirror such as "
                 f"pjreddie.com/projects/pascal-voc-dataset-mirror) into {VOC_DIR} and run again.")


def convert(datasets, name):
    from PIL import Image
    xs, ys, centers, owner = [], [], [], []
    n_person = 0
    for ds in datasets:
        for i in range(len(ds)):
            img, ann = ds[i]
            w0, h0 = img.size
            xs.append(np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
            grid = np.zeros((GRID, GRID), dtype=np.uint8)
            objs = ann["annotation"].get("object", [])
            if isinstance(objs, dict):
                objs = [objs]
            for o in objs:
                if o["name"] != "person" or o.get("difficult", "0") == "1":
                    continue
                b = o["bndbox"]
                cx = (float(b["xmin"]) + float(b["xmax"])) / 2 * SIZE / w0
                cy = (float(b["ymin"]) + float(b["ymax"])) / 2 * SIZE / h0
                gx, gy = min(int(cx // CELL), GRID - 1), min(int(cy // CELL), GRID - 1)
                grid[gy, gx] = 1
                centers.append((cx, cy))
                owner.append(len(xs) - 1)
                n_person += 1
            ys.append(grid)
            if (len(xs)) % 2000 == 0:
                print(f"      {name}: {len(xs)} images")
    x, y = np.stack(xs), np.stack(ys)
    path = os.path.join(HERE, "data", f"fomo_{name}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=x, y=y, centers=np.array(centers, dtype=np.float32).reshape(-1, 2),
                        image_of_center=np.array(owner, dtype=np.int32))
    with_p = int((y.reshape(len(y), -1).max(1) > 0).sum())
    print(f"[*] {name}: {len(x)} images ({with_p} with persons, {len(x) - with_p} without), {n_person} persons, "
          f"{100 * y.mean():.2f}% of the cells positive -> {path}  ({os.path.getsize(path) / 2**20:.0f} MB)")


def main():
    os.chdir(HERE)
    print("[*] Pascal VOC (downloaded once into ../data/voc) ...")
    convert([voc("2007", "trainval"), voc("2012", "trainval")], "train")
    convert([voc("2007", "test")], "test")


if __name__ == "__main__":
    main()
