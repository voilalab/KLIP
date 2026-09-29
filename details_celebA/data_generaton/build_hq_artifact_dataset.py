#!/usr/bin/env python3
"""
Build the CelebA-HQ face + artifact test set (celeba_hq_artifacts.npy).

Each face gets every selected artifact, alpha-composited at the fixed forehead
location:  out = alpha * colour + (1 - alpha) * face,  mask = artifact mask.

Artifacts: one file per artifact in assets/artifacts/ (dict alpha, color, mask; the .png/_mask.png
next to each is the same artifact for viewing):
  1_small_mark.npy      -> '_2'       small mark (pngegg stroke)         recovered, 52 mask px
  2_stitched_scar.npy   -> '_scar'    red stitched scar                  recovered, 96 px
  3_lightning_bolt.npy  -> '_bolt'    Harry Potter lightning bolt        recovered, 167 px
  4_large_scar.npy      -> '_bigscar' large scar                         recovered, 227 px
  5_synthetic_scar.npy  -> '_synscar' long stitched synthetic scar       make_synthetic_scar.py, 157 px

Paper set: the 20 faces below x 5 artifacts = 100 images, index = 5*face + variant,
variants in the order small, scar, bolt, bigscar, synscar.

Usage:
  python build_hq_artifact_dataset.py --faces-dir ../data/celebA_HQ_real \
      --out-npy ../data/celeba_hq_artifacts.npy [--out-dir ../data/celeba_hq_artifacts_png]
"""
import argparse
import os

import numpy as np
from PIL import Image

PAPER_FACES = ['hq_001', 'hq_042', 'hq_071', 'hq_004', 'hq_005', 'hq_160', 'hq_007', 'hq_008', 'hq_009', 'hq_010',
               'hq_011', 'hq_012', 'hq_114', 'hq_014', 'hq_015', 'hq_018', 'hq_019', 'hq_021', 'hq_180', 'hq_023']
VARIANTS = [('_2', '1_small_mark'), ('_scar', '2_stitched_scar'), ('_bolt', '3_lightning_bolt'),
            ('_bigscar', '4_large_scar'), ('_synscar', '5_synthetic_scar')]

ap = argparse.ArgumentParser()
ap.add_argument('--faces-dir', default='../data/celebA_HQ_real')
ap.add_argument('--faces', default=','.join(PAPER_FACES), help='comma list of face stems')
ap.add_argument('--variants', default=','.join(v[0] for v in VARIANTS), help='comma list of suffixes')
ap.add_argument('--artifacts-dir', default='../assets/artifacts')
ap.add_argument('--out-npy', default='../data/celeba_hq_artifacts.npy')
ap.add_argument('--out-dir', default='', help='optional: also write <stem><suffix>.png + _mask.png pairs')
a = ap.parse_args()

art = {f: np.load(os.path.join(a.artifacts_dir, f'{f}.npy'), allow_pickle=True).item() for _, f in VARIANTS}
use = [v for v in VARIANTS if v[0] in a.variants.split(',')]
if a.out_dir:
    os.makedirs(a.out_dir, exist_ok=True)

imgs, labels, names = [], [], []
for stem in a.faces.split(','):
    face = np.asarray(Image.open(os.path.join(a.faces_dir, f'{stem}.png')).convert('RGB')
                      .resize((256, 256), Image.BICUBIC), np.float32) / 255
    for suf, f in use:
        r = art[f]
        al = r['alpha'][..., None]
        out8 = (np.clip(al * r['color'] + (1 - al) * face, 0, 1) * 255).round().astype(np.uint8)
        m = r['mask'].astype(np.uint8)
        imgs.append(out8); labels.append(m[..., None]); names.append(f'{stem}{suf}')
        if a.out_dir:
            Image.fromarray(out8).save(os.path.join(a.out_dir, f'{stem}{suf}.png'))
            Image.fromarray(m, 'L').save(os.path.join(a.out_dir, f'{stem}{suf}_mask.png'))

imgs, labels = np.stack(imgs), np.stack(labels)
np.save(a.out_npy, {'imgs': imgs, 'labels': labels, 'names': np.array(names),
                    'layout': f'index = {len(use)}*face + variant; variants = {[v[0] for v in use]}'},
        allow_pickle=True)
print(f'saved {a.out_npy}: imgs {imgs.shape}, labels {labels.shape}')
