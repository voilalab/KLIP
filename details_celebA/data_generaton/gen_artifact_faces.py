#!/usr/bin/env python3
"""
Batch-generate synthetic "face + forehead artifact" images and binary masks.

Reuses the exact compositing pipeline from celebA_11_npy.ipynb (same artifact
PNG, width fraction, opacity, rotation, forehead placement, mask threshold).

Usage:
  python gen_artifact_faces.py                       # all ../data/celebA/*.png -> ../data/test_artifact_added_celebA/
  python gen_artifact_faces.py --src celebA --out DIR --limit 50
  python gen_artifact_faces.py --jitter --seed 0     # random per-image scale/rotation/offset/opacity
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image

ARTIFACT_WIDTH_FRAC = 0.18
OPACITY             = 0.85
ROT_DEG             = -12

def _gaussian_blur_mask(mask, sigma=1.2):
    if sigma <= 0: return mask
    radius = int(max(1, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma**2)); k /= k.sum()
    m = np.pad(mask.astype(np.float32), ((0, 0), (radius, radius)), mode="reflect")
    mh = np.array([np.convolve(row, k, mode="valid") for row in m])
    mh = np.pad(mh, ((radius, radius), (0, 0)), mode="reflect")
    return np.array([np.convolve(col, k, mode="valid") for col in mh.T]).T

def load_artifact_rgba(path, opacity=OPACITY):
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3]
    if alpha.mean() < 0.01 or alpha.std() < 0.01:
        raise ValueError("Artifact PNG has no usable alpha channel.")
    a = alpha.copy()
    for _ in range(2):
        a2 = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                a2 = np.maximum(a2, np.roll(np.roll(a, dy, 0), dx, 1))
        a = a2
    a = _gaussian_blur_mask(a, sigma=1.0)
    rgba[..., 3] = np.clip(a * opacity, 0, 1)
    return Image.fromarray((np.clip(rgba, 0, 1) * 255).round().astype(np.uint8), mode="RGBA")

def detect_face_bbox(face_rgb01):
    """mediapipe -> face_recognition -> None (caller uses fixed fallback box)."""
    H, W = face_rgb01.shape[:2]
    img_u8 = (np.clip(face_rgb01, 0, 1) * 255).astype(np.uint8)
    try:
        import mediapipe as mp
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            res = fd.process(img_u8)
            if res.detections:
                best, best_area = None, -1
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x0, y0, ww, hh = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
                    if ww*hh > best_area: best_area, best = ww*hh, (x0, y0, ww, hh)
                return best
    except Exception:
        pass
    try:
        import face_recognition
        locs = face_recognition.face_locations(img_u8, model="hog")
        if locs:
            t, r, b, le = max(locs, key=lambda l: (l[1]-l[3])*(l[2]-l[0]))
            return (le, t, r-le, b-t)
    except Exception:
        pass
    return None

def composite(face_path, art_rgba_pil, out_image_path, out_mask_path,
              width_frac=ARTIFACT_WIDTH_FRAC, rot_deg=ROT_DEG, dx_frac=0.0, dy_frac=0.0):
    face_pil = Image.open(face_path).convert("RGB")
    face_rgb01 = np.asarray(face_pil, dtype=np.float32) / 255.0
    H_f, W_f = face_rgb01.shape[:2]

    bbox = detect_face_bbox(face_rgb01)
    x, y, w, h = bbox if bbox else (int(0.15*W_f), int(0.15*H_f), int(0.70*W_f), int(0.70*H_f))
    forehead_cx = int(x + (0.50 + dx_frac) * w)
    forehead_cy = int(y + (0.20 + dy_frac) * h)

    target_w = int(max(32, width_frac * w))
    scale = target_w / art_rgba_pil.size[0]
    art = art_rgba_pil.resize((target_w, max(1, int(art_rgba_pil.size[1] * scale))), resample=Image.LANCZOS)
    art = art.rotate(rot_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))

    base = face_pil.convert("RGBA")
    ow, oh = art.size
    x0, y0 = int(forehead_cx - ow/2), int(forehead_cy - oh/2)
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    tmp.paste(art, (x0, y0), art)
    Image.alpha_composite(base, tmp).convert("RGB").save(out_image_path)

    art_alpha = np.array(art.split()[-1], dtype=np.float32) / 255.0
    mask_full = np.zeros((H_f, W_f), dtype=np.float32)
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(W_f, x0+ow), min(H_f, y0+oh)
    if x1 < x2 and y1 < y2:
        mask_full[y1:y2, x1:x2] = art_alpha[y1-y0:y2-y0, x1-x0:x2-x0]
    mask_binary = (mask_full > 0.3).astype(np.uint8) * 255
    Image.fromarray(mask_binary, mode="L").save(out_mask_path)
    return int((mask_binary > 0).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/celebA", help="dir of face images (png/jpg)")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--artifact", default="../assets/pngegg.png")
    ap.add_argument("--out", default="../data/test_artifact_added_celebA")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="_2", help="output name suffix, matches existing naming")
    ap.add_argument("--jitter", action="store_true", help="randomise scale/rotation/offset/opacity per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, a.pattern)))
    if a.limit: files = files[:a.limit]
    if not files: sys.exit(f"no files matched {a.src}/{a.pattern}")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    art_default = load_artifact_rgba(a.artifact)

    n_done = n_skip = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        oi = os.path.join(a.out, f"{stem}{a.suffix}.png")
        om = os.path.join(a.out, f"{stem}{a.suffix}_mask.png")
        if not a.overwrite and os.path.exists(oi) and os.path.exists(om):
            n_skip += 1; continue
        if a.jitter:
            art = load_artifact_rgba(a.artifact, opacity=float(rng.uniform(0.7, 0.95)))
            px = composite(f, art, oi, om,
                           width_frac=float(rng.uniform(0.14, 0.24)),
                           rot_deg=float(rng.uniform(-30, 30)),
                           dx_frac=float(rng.uniform(-0.12, 0.12)),
                           dy_frac=float(rng.uniform(-0.05, 0.08)))
        else:
            px = composite(f, art_default, oi, om)
        n_done += 1
        print(f"[{n_done}/{len(files)}] {oi}  mask_px={px}")
    print(f"done: {n_done} generated, {n_skip} skipped (already existed) -> {a.out}/")

if __name__ == "__main__":
    main()
