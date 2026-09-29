#!/usr/bin/env python3
"""
Procedural "synthetic scar" artifacts (the 5th artifact of the CelebA-HQ set).

Draws a slightly curved stroke with optional cross stitches at 8x resolution,
downsamples, roughens and feathers the edge, and stores per-pixel alpha, colour
and a binary mask in the same format as assets/artifact_variants_recovered.npy:

    {mask_px: {'alpha': (256,256) float32 [0,1],
               'color': (256,256,3) float32 [0,1],
               'mask':  (256,256) uint8 0/255,
               'example': str}}

The scar is placed at the same forehead location as the real artifacts
(centre (x=126, y=73) on aligned 256x256 CelebA-HQ faces). The dict key is the
mask size in pixels; the paper set uses key 157 ("long_stitched").

Usage:
  python make_synthetic_scar.py --out ../assets/artifact_variants_scar_synth.npy [--preview preview.png]
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


def scar_artifact(length=22, angle_deg=-40, curvature=0.18, width=1.7, n_stitches=4, stitch_len=7,
                  stitch_width=1.1, center=(126, 73), edge_sigma=0.45, alpha_peak=0.97,
                  colour=(0.58, 0.33, 0.33), colour_jitter=0.05, seed=0):
    r = np.random.default_rng(seed)
    S = 8  # supersampling factor
    im = Image.new('L', (256 * S, 256 * S), 0)
    dr = ImageDraw.Draw(im)

    # main stroke: quadratic curve sampled as a polyline, slightly thicker in the middle
    t = np.linspace(-0.5, 0.5, 40)
    th = np.deg2rad(angle_deg)
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    pts = (np.array(center)[None, :] + (t * length)[:, None] * u[None, :]
           + (curvature * length * (t ** 2 - 0.25) * 2)[:, None] * v[None, :])
    w = width * (1 + 0.3 * np.sin(np.linspace(0, np.pi, 40)))
    for i in range(39):
        dr.line([tuple(pts[i] * S), tuple(pts[i + 1] * S)], fill=255, width=int(round(w[i] * S)))

    # cross stitches perpendicular to the stroke
    for k in np.linspace(0.15, 0.85, n_stitches) if n_stitches else []:
        j = int(k * 39)
        d = pts[min(j + 1, 39)] - pts[max(j - 1, 0)]
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        L = stitch_len * (0.85 + 0.3 * r.random())
        a, b = pts[j] - n * L / 2, pts[j] + n * L / 2
        dr.line([tuple(a * S), tuple(b * S)], fill=255, width=int(round(stitch_width * S)))

    hard = np.asarray(im.resize((256, 256), Image.LANCZOS), np.float32) / 255

    # roughen the boundary a little, then soften
    noise = gaussian_filter(r.standard_normal((256, 256)), 1.2)
    noise /= np.abs(noise).max()
    rough = np.clip(hard + 0.35 * noise * hard * (1 - hard) * 4, 0, 1)
    alpha = np.clip(gaussian_filter(rough, edge_sigma) * alpha_peak, 0, 1).astype(np.float32)
    mask = ((alpha > 0.25) * 255).astype(np.uint8)

    # stroke core a bit darker/redder than the feathered edge, like the real scars
    core = gaussian_filter(hard, 0.6)[..., None]
    col = np.tile(np.array(colour, np.float32), (256, 256, 1)) * (1 - 0.15 * core) + np.array([0.08, 0, 0], np.float32) * core
    col += (gaussian_filter(r.standard_normal((256, 256)), 1.5)[..., None] * colour_jitter).astype(np.float32)
    return {'alpha': alpha, 'color': np.clip(col, 0, 1).astype(np.float32), 'mask': mask,
            'example': 'synthetic scar (procedural), generated 2026-09-28'}


SPECS = {
    'short_plain':   dict(length=17, angle_deg=-35, curvature=0.10, n_stitches=0, width=1.5, seed=0),
    'stitched':      dict(length=22, angle_deg=-40, curvature=0.18, n_stitches=4, width=1.7, seed=1),
    'long_stitched': dict(length=34, angle_deg=-25, curvature=0.12, n_stitches=6, width=2.1, stitch_len=8, seed=2),  # key 157, used in the paper set
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../assets/artifact_variants_scar_synth.npy')
    ap.add_argument('--preview', default='', help='optional PNG: scars composited on flat skin + alpha maps')
    a = ap.parse_args()

    variants = {}
    for name, sp in SPECS.items():
        r = scar_artifact(**sp)
        k = int((r['mask'] > 0).sum())
        r['example'] += f' [{name}]'
        variants[k] = r
        print(f'{name:14s} -> key {k} (mask px)')
    np.save(a.out, variants, allow_pickle=True)
    print('saved', a.out, 'keys', list(variants))

    if a.preview:
        skin = np.tile(np.array([0.87, 0.72, 0.62], np.float32), (256, 256, 1))
        crop = (slice(43, 103), slice(96, 156))
        comp = [(np.clip(r['alpha'][..., None] * r['color'] + (1 - r['alpha'][..., None]) * skin, 0, 1) * 255).astype(np.uint8)[crop]
                for r in variants.values()]
        alph = [(np.stack([r['alpha']] * 3, -1) * 255).astype(np.uint8)[crop] for r in variants.values()]
        big = np.concatenate([np.concatenate(comp, 1), np.concatenate(alph, 1)], 0)
        Image.fromarray(big).resize((big.shape[1] * 4, big.shape[0] * 4), Image.NEAREST).save(a.preview)
        print('preview', a.preview)


if __name__ == '__main__':
    main()
