#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:27:32 2025

@author: tanjintoma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing script for DOTA v1.0 dataset
- Reads polygon labels from DOTA
- Converts polygons → bounding boxes
- Splits train set into train/val
- Saves annotations in COCO JSON format
- Also saves YOLO TXT format labels
"""

import os
import json
import random
import shutil
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image

# ============================
# CONFIG
# ============================
DATASET_ROOT = "/content/drive/MyDrive/Datasets/DOTA"
OUTPUT_DIR   = "/content/drive/MyDrive/Datasets/DOTA_coco"

CATEGORIES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool"
]
CATEGORY2ID = {cat: i+1 for i, cat in enumerate(CATEGORIES)}  # COCO ids start at 1

# ============================
# HELPERS
# ============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_dota_label(label_file):
    """Reads one DOTA label .txt file"""
    objects = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 9:  # skip headers
                continue
            x_coords = list(map(float, parts[0:8:2]))
            y_coords = list(map(float, parts[1:8:2]))
            xmin, ymin = min(x_coords), min(y_coords)
            xmax, ymax = max(x_coords), max(y_coords)
            category = parts[8]
            if category not in CATEGORY2ID:
                continue
            objects.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "category": category
            })
    return objects

# ============================
# STEP 1. COCO Builder 
# ============================
def build_coco_dict(image_label_pairs, output_img_dir,
                    start_ann_id=0, start_img_id=0):
    images, annotations = [], []
    ann_id = start_ann_id
    img_id = start_img_id

    for img_path, lbl_path in tqdm(image_label_pairs):
        img = Image.open(img_path)
        w, h = img.size
        img_info = {
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        }
        images.append(img_info)

        objects = parse_dota_label(lbl_path)
        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox"]
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY2ID[obj["category"]],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

        # Copy original image
        ensure_dir(output_img_dir)
        shutil.copy(img_path, os.path.join(output_img_dir, os.path.basename(img_path)))
        img_id += 1

    categories = [{"id": i+1, "name": cat} for i, cat in enumerate(CATEGORIES)]
    return {"images": images, "annotations": annotations, "categories": categories}

# ============================
# STEP 2. Split train/val
# ============================
def split_train_val(images, labels, split_ratio=0.8):
    data = list(zip(images, labels))
    random.shuffle(data)
    split = int(len(data) * split_ratio)
    return data[:split], data[split:]

# ============================
# STEP 3. YOLO Writer
# ============================
def save_yolo_labels(annotations, images, output_label_dir):
    """Writes YOLO format labels from COCO-style annotations"""
    ensure_dir(output_label_dir)

    # Map image_id -> image info
    img_dict = {img["id"]: img for img in images}

    # Group annotations by image
    anns_by_img = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img in images:
        w, h = img["width"], img["height"]
        fname = os.path.splitext(img["file_name"])[0] + ".txt"
        outpath = os.path.join(output_label_dir, fname)

        anns = anns_by_img.get(img["id"], [])
        lines = []
        for ann in anns:
            xmin, ymin, bw, bh = ann["bbox"]
            xc = xmin + bw / 2
            yc = ymin + bh / 2
            # Normalize
            xc /= w
            yc /= h
            bw /= w
            bh /= h
            cls_id = ann["category_id"] - 1
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(outpath, "w") as f:
            f.write("\n".join(lines))

# ============================
# MAIN
# ============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect dataset
    train_images = sorted(glob(os.path.join(DATASET_ROOT, "train/images/*.png")))
    train_labels = sorted(glob(os.path.join(DATASET_ROOT, "train/labels/*.txt")))
    val_images   = sorted(glob(os.path.join(DATASET_ROOT, "val/images/*.png")))
    val_labels   = sorted(glob(os.path.join(DATASET_ROOT, "val/labels/*.txt")))

    # Split train → train/val
    train_data, val_split = split_train_val(train_images, train_labels, split_ratio=0.8)

    val_imgs, val_lbls     = zip(*val_split)
    train_imgs, train_lbls = zip(*train_data)
    test_imgs, test_lbls   = val_images, val_labels  # official val → test

    # Build COCO for each split
    coco_train = build_coco_dict(list(zip(train_imgs, train_lbls)),
                                 os.path.join(OUTPUT_DIR, "train/images"))
    coco_val = build_coco_dict(list(zip(val_imgs, val_lbls)),
                               os.path.join(OUTPUT_DIR, "val/images"))
    coco_test = build_coco_dict(list(zip(test_imgs, test_lbls)),
                                os.path.join(OUTPUT_DIR, "test/images"))

    # Save JSONs
    with open(os.path.join(OUTPUT_DIR, "train/instances_train.json"), "w") as f:
        json.dump(coco_train, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val/instances_val.json"), "w") as f:
        json.dump(coco_val, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "test/instances_test.json"), "w") as f:
        json.dump(coco_test, f, indent=2)

    # Save YOLO labels
    save_yolo_labels(coco_train["annotations"], coco_train["images"],
                     os.path.join(OUTPUT_DIR, "train/labels"))
    save_yolo_labels(coco_val["annotations"], coco_val["images"],
                     os.path.join(OUTPUT_DIR, "val/labels"))
    save_yolo_labels(coco_test["annotations"], coco_test["images"],
                     os.path.join(OUTPUT_DIR, "test/labels"))

    print(f"✅ Preprocessing complete. Dataset saved at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
