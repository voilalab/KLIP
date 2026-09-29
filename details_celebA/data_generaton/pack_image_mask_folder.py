#!/usr/bin/env python3
"""
Pack an MVTec-AD style folder of face images + ground-truth masks into the .npy
format read by the notebook / run_klip_celeba.py.

    <images>/<N>.png            face image (any size, RGB)
    <masks>/<N>_mask.png        binary mask (white = anomaly)

Output: {'imgs': (K,256,256,3) uint8, 'labels': (K,256,256,1) uint8 0/255, 'names': (K,)}
Images are resized bicubic, masks nearest-neighbour. Files are sorted numerically.

This is how celeba_faces_random28.npy was made from
mvtec_ad_moreface/faces/test/random + ground_truth/random.

Usage:
  python pack_image_mask_folder.py --images ../data/mvtec_ad_moreface/faces/test/random \
      --masks ../data/mvtec_ad_moreface/faces/ground_truth/random --out ../data/celeba_faces_random28.npy
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True)
ap.add_argument('--masks', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--prefix', default='random_', help='name prefix stored in names[]')
a = ap.parse_args()


def num(p):
    m = re.match(r'(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 1 << 30


imgs, labels, names = [], [], []
for f in sorted(glob.glob(os.path.join(a.images, '*.png')), key=num):
    stem = os.path.splitext(os.path.basename(f))[0]
    mp = os.path.join(a.masks, f'{stem}_mask.png')
    if not os.path.exists(mp):
        print('[skip] no mask for', f)
        continue
    im = np.asarray(Image.open(f).convert('RGB').resize((256, 256), Image.BICUBIC), np.uint8)
    m = np.asarray(Image.open(mp).convert('L').resize((256, 256), Image.NEAREST)) > 127
    imgs.append(im); labels.append((m * 255).astype(np.uint8)[..., None]); names.append(f'{a.prefix}{stem}')

np.save(a.out, {'imgs': np.stack(imgs), 'labels': np.stack(labels), 'names': np.array(names),
                'source': f'{a.images} + {a.masks}'}, allow_pickle=True)
print(f'saved {a.out}: {len(imgs)} images')
