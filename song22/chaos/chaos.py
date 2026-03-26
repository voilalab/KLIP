# chaos.py
import os
import numpy as np
import tensorflow_datasets as tfds
import pydicom

_DESCRIPTION = """
CHAOS CT (local DICOM): HU -> attenuation coefficients (relative to water),
background zeroed, 3D volumes sliced into 2D images. Variable native resolution.
"""

def _is_dicom(path):
  try:
    _ = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    return True
  except Exception:
    return False

def _series_key(ds):
  # Robustly fetch SeriesInstanceUID as the grouping key
  return getattr(ds, "SeriesInstanceUID", None)

def _slice_sort_key(ds):
  # Prefer geometric sorting, then fall back to InstanceNumber
  pos = getattr(ds, "ImagePositionPatient", None)
  if pos is not None and len(pos) == 3:
    return float(pos[2])
  inst = getattr(ds, "InstanceNumber", None)
  return float(inst) if inst is not None else 0.0

def _load_series(file_list):
  """Load a DICOM series into a 3D HU volume (z,y,x)."""
  headers = []
  for f in file_list:
    try:
      headers.append(pydicom.dcmread(f, force=True))
    except Exception:
      # Skip unreadable
      continue
  if not headers:
    return None

  # Sort slices along the acquisition axis
  headers.sort(key=_slice_sort_key)

  # Read pixel arrays, apply rescale to HU
  slices = []
  for ds in headers:
    arr = ds.pixel_array.astype(np.float32)

    # Convert to HU using DICOM metadata if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    slices.append(hu)

  vol_hu = np.stack(slices, axis=0)  # (z, y, x)
  return vol_hu

def _hu_to_mu_rel(vol_hu, bg_hu_threshold=-900.0):
  """HU -> relative attenuation (water=1.0), zero background."""
  mu_rel = (vol_hu / 1000.0) + 1.0       # relative to water
  mu_rel[vol_hu <= bg_hu_threshold] = 0.0 # background -> 0
  # Normalize per-volume to [0,1] for uint8 export
  vmax = mu_rel.max()
  if vmax > 0:
    mu_rel = mu_rel / vmax
  return mu_rel

def _slice_3d_img(vol, view_plane='ax'):
  """Return list of 2D slices from a 3D volume (z,y,x)."""
  imgs = []
  if view_plane == 'cor':   # (y, z, x)
    for idx in range(vol.shape[1]):
      imgs.append(vol[:, idx, :])
  elif view_plane == 'sag': # (x, z, y)
    for idx in range(vol.shape[2]):
      imgs.append(vol[:, :, idx])
  else:                     # 'ax' (default): (z, y, x)
    for idx in range(vol.shape[0]):
      imgs.append(vol[idx, :, :])
  return imgs

class Chaos(tfds.core.GeneratorBasedBuilder):
  """TFDS builder for local CHAOS CT DICOM series -> 2D slices."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        # Variable native size; your existing pipeline resizes later.
        'image': tfds.features.Image(shape=(None, None, 1)),  # uint8 PNG
      }),
      supervised_keys=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    root = "/data/jhong392/datasets/CHAOS/CTs/train_CTs_dark"
    return {'train': self._generate_examples(root)}

  def _iter_dicom_paths(self, root):
    for dirpath, _, filenames in os.walk(root):
      for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        if _is_dicom(fpath):
          yield fpath

  def _group_by_series(self, root):
    """Return dict: SeriesInstanceUID -> [file paths]."""
    series_map = {}
    for f in self._iter_dicom_paths(root):
      try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
      except Exception:
        continue
      key = _series_key(ds)
      if key is None:
        # Skip non-standard files without SeriesInstanceUID
        continue
      series_map.setdefault(key, []).append(f)
    return series_map

  def _generate_examples(self, root):
    """Yield (key, example) pairs."""
    series_map = self._group_by_series(root)
    count = -1
    for _, files in series_map.items():
      vol_hu = _load_series(files)
      if vol_hu is None:
        continue
      vol_mu = _hu_to_mu_rel(vol_hu, bg_hu_threshold=-900.0)
      # Slice; choose axial by default (match common CT orientation)
      slices = _slice_3d_img(vol_mu, view_plane='ax')
      for sl in slices:
        count += 1
        # Scale to 0..255 uint8 with channel dim
        img = (np.clip(sl, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None]
        yield count, {'image': img}
