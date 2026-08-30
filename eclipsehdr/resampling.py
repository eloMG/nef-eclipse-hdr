"""Shared translation-only resampling for alignment export and HDR merging."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .errors import EclipseHDRError


def translated_coordinate_grid(
    y0: int, y1: int, width: int, dy: float, dx: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Build source coordinates for a translation applied in output space."""

    source_y = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(dy)
    source_x = np.arange(width, dtype=np.float32)[None, :] - np.float32(dx)
    coordinates = np.empty((2, y1 - y0, width), dtype=np.float32)
    coordinates[0, :, :] = source_y
    coordinates[1, :, :] = source_x
    return coordinates, (source_y, source_x)


def sample_translated_rgb_chunk(
    frame: np.ndarray,
    y0: int,
    y1: int,
    dy: float,
    dx: float,
    order: int,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate one RGB row chunk, sampling source at ``output - shift``.

    The returned coordinate array is reused by the HDR merge for saturation
    masks.  It is empty for an exact zero shift, where a direct slice is used.
    """

    height, width, _ = frame.shape
    scale = np.float32(1.0 / 65535.0) if normalize else np.float32(1.0)
    if dy == 0.0 and dx == 0.0:
        rgb = np.asarray(frame[y0:y1, :, :], dtype=np.float32)
        if normalize:
            rgb *= scale
        valid = np.ones((y1 - y0, width), dtype=bool)
        coordinates = np.empty((0,), dtype=np.float32)
        return rgb, valid, coordinates

    coordinates, components = translated_coordinate_grid(y0, y1, width, dy, dx)
    source_y, source_x = components
    valid = (
        (source_y >= 0.0)
        & (source_y <= height - 1.0)
        & (source_x >= 0.0)
        & (source_x <= width - 1.0)
    )
    rgb = np.empty((y1 - y0, width, 3), dtype=np.float32)
    for channel in range(3):
        ndimage.map_coordinates(
            frame[..., channel],
            coordinates,
            output=rgb[..., channel],
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    if normalize:
        rgb *= scale
    return rgb, valid, coordinates


def translate_linear_u16(
    frame: np.ndarray,
    dy: float,
    dx: float,
    *,
    chunk_rows: int,
    interpolation_order: int,
    output: np.ndarray,
) -> np.ndarray:
    """Translate one developed uint16 RGB frame into a same-sized output."""

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint16:
        raise EclipseHDRError(
            f"Aligned source must be HxWx3 uint16, got {frame.shape} {frame.dtype}"
        )
    if output.shape != frame.shape or output.dtype != np.uint16:
        raise EclipseHDRError(
            f"Aligned output must be {frame.shape} uint16, got {output.shape} {output.dtype}"
        )
    if chunk_rows < 1:
        raise EclipseHDRError("Aligned export chunk_rows must be positive")
    if interpolation_order not in (0, 1):
        raise EclipseHDRError("Only nearest or bilinear translation interpolation is supported")
    if not math.isfinite(dy) or not math.isfinite(dx):
        raise EclipseHDRError("Aligned export offsets must be finite")

    if dy == 0.0 and dx == 0.0:
        output[...] = frame
        return output

    height = frame.shape[0]
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        sampled, _, _ = sample_translated_rgb_chunk(
            frame,
            y0,
            y1,
            dy,
            dx,
            interpolation_order,
            normalize=False,
        )
        np.clip(sampled, 0.0, 65535.0, out=sampled)
        np.rint(sampled, out=sampled)
        output[y0:y1] = sampled.astype(np.uint16)
    return output
