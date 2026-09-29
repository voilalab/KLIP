#!/usr/bin/env python3
"""
Download real CelebA-HQ 256x256 faces (the data google/ddpm-celebahq-256 was trained on)
from the Hugging Face dataset `korexyz/celeba-hq-256x256` and save the first N as PNGs.

The paper set uses the first 200 images of the validation parquet shard, saved as
hq_000.png ... hq_199.png.

Usage:
  pip install huggingface_hub pyarrow pillow
  python download_celebahq.py --out ../data/celebA_HQ_real --n 200
"""
import argparse
import io
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='../data/celebA_HQ_real')
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--cache', default='../data/hf_cache')
a = ap.parse_args()

path = hf_hub_download('korexyz/celeba-hq-256x256', 'data/validation-00000-of-00001.parquet',
                       repo_type='dataset', local_dir=a.cache)
col = pq.read_table(path).column('image')
os.makedirs(a.out, exist_ok=True)
for i in range(a.n):
    im = Image.open(io.BytesIO(col[i].as_py()['bytes'])).convert('RGB')
    assert im.size == (256, 256), im.size
    im.save(os.path.join(a.out, f'hq_{i:03d}.png'))
print(f'saved {a.n} faces to {a.out}/ (hq_000.png ...)')
