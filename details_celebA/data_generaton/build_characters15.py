#!/usr/bin/env python3
"""
Build the 15-image film/TV character set (celeba_characters15.npy).

Sources:
  celeba_characters34.npy   34 character / celebrity faces with hand-drawn masks
                            (imgs (34,256,256,3) uint8, labels (34,256,256,1) uint8 0/255)
  celeba_faces_random28.npy 28 character faces packed by pack_image_mask_folder.py

Selection = 14 images from the 34-image set + random/1.png (Voldemort) at position 5.
Each entry is (source, index, name):
"""
import argparse

import numpy as np

SELECTION = [
    ('c34', 0, 'Klingon (Worf)'), ('c34', 1, 'Borg implants (woman)'), ('c34', 2, 'Seven of Nine'),
    ('c34', 33, 'Ferengi'), ('c34', 9, 'Eye patch (bald)'), ('r28', 1, 'Voldemort'),
    ('c34', 11, 'Geordi visor'), ('c34', 30, 'Tyrion Lannister'), ('c34', 17, 'Cheek scar (blonde)'),
    ('c34', 19, 'Harry Potter (colour)'), ('c34', 20, 'Eye patch (glasses)'), ('c34', 23, 'Terminator'),
    ('c34', 24, 'Spider-Man'), ('c34', 31, 'Mad-Eye Moody'), ('c34', 29, 'Harry Potter (grey)'),
]

ap = argparse.ArgumentParser()
ap.add_argument('--characters34', default='../data/celeba_characters34.npy')
ap.add_argument('--random28', default='../data/celeba_faces_random28.npy')
ap.add_argument('--out', default='../data/celeba_characters15.npy')
a = ap.parse_args()

src = {'c34': np.load(a.characters34, allow_pickle=True).item(),
       'r28': np.load(a.random28, allow_pickle=True).item()}
imgs = np.stack([src[s]['imgs'][i] for s, i, _ in SELECTION])
labels = np.stack([src[s]['labels'][i] for s, i, _ in SELECTION])
np.save(a.out, {'imgs': imgs, 'labels': labels,
                'names': np.array([f'character_{k:02d}' for k in range(len(SELECTION))]),
                'character': np.array([n for _, _, n in SELECTION]),
                'source_index': np.array([f'{s}[{i}]' for s, i, _ in SELECTION])}, allow_pickle=True)
print(f'saved {a.out}: imgs {imgs.shape}')
