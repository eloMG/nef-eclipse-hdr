"""Relative exposure-normalized HDR merge in linear floating-point space."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from .errors import EclipseHDRError
from .models import MergeConfig, MergeStatistics
from .resampling import sample_translated_rgb_chunk

LOGGER = logging.getLogger(__name__)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if not edge0 < edge1:
        raise EclipseHDRError(f"Invalid smoothstep interval [{edge0}, {edge1}]")
    normalized = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _validate_config(config: MergeConfig) -> None:
    levels = (
        config.black_start,
        config.black_full_weight,
        config.highlight_falloff,
        config.saturation_cutoff,
    )
    if not (0 <= levels[0] < levels[1] < levels[2] < levels[3] <= 1):
        raise EclipseHDRError(
            "Merge thresholds must satisfy 0 <= black_start < black_full_weight < "
            "highlight_falloff < saturation_cutoff <= 1"
        )
    if config.chunk_rows < 1:
        raise EclipseHDRError("chunk_rows must be positive")
    if config.interpolation_order not in (0, 1):
        raise EclipseHDRError("Only nearest or bilinear translation interpolation is supported")


def _sample_mask_chunk(
    mask: Optional[np.ndarray],
    y0: int,
    y1: int,
    width: int,
    coordinates: np.ndarray,
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if coordinates.size == 0:
        return np.asarray(mask[y0:y1, :], dtype=bool)
    sampled = ndimage.map_coordinates(
        mask,
        coordinates,
        order=0,
        mode="constant",
        cval=1,
        prefilter=False,
    )
    if sampled.shape != (y1 - y0, width):
        raise EclipseHDRError("Internal saturation-mask sampling shape mismatch")
    return sampled.astype(bool, copy=False)


def _sample_postprocessed_clip_chunk(
    frame: np.ndarray,
    y0: int,
    y1: int,
    dy: float,
    dx: float,
    cutoff: float,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Reject any clipped source neighbor before interpolation can hide it.

    Only the source rows needed by this output chunk (plus a one-pixel halo) are
    materialized, avoiding a full-resolution boolean allocation for every frame.
    """

    height, width, _ = frame.shape
    source_first = float(y0) - dy
    source_last = float(y1 - 1) - dy
    requested_y0 = int(math.floor(min(source_first, source_last))) - 2
    requested_y1 = int(math.ceil(max(source_first, source_last))) + 3
    if requested_y1 <= 0 or requested_y0 >= height:
        return np.ones((y1 - y0, width), dtype=bool)
    source_y0 = max(0, requested_y0)
    source_y1 = min(height, requested_y1)
    source = np.asarray(frame[source_y0:source_y1, :, :])
    threshold_code = int(math.ceil(cutoff * 65535.0))
    clipped = np.any(source >= threshold_code, axis=2)
    clipped = ndimage.binary_dilation(clipped, iterations=1)

    if coordinates.size == 0:
        local_y0 = y0 - source_y0
        return clipped[local_y0 : local_y0 + (y1 - y0), :]
    local_coordinates = coordinates.copy()
    local_coordinates[0] -= np.float32(source_y0)
    sampled = ndimage.map_coordinates(
        clipped.astype(np.uint8, copy=False),
        local_coordinates,
        order=0,
        mode="constant",
        cval=1,
        prefilter=False,
    )
    if sampled.shape != (y1 - y0, width):
        raise EclipseHDRError("Internal postprocessed-clipping mask shape mismatch")
    return sampled.astype(bool, copy=False)


def merge_linear_hdr(
    frames: Sequence[np.ndarray],
    offsets_yx: Sequence[tuple[float, float]],
    exposure_values: Sequence[float],
    reference_index: int,
    config: MergeConfig,
    *,
    saturation_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    output: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, MergeStatistics]:
    """Merge aligned linear RGB into relative reference-exposure units.

    For frame ``i``, ``q_i = shutter * ISO / aperture**2`` and the normalized
    estimate is ``rgb_i / (q_i / q_reference)``. Weights are computed from the
    original developed measurement, never from the exposure-normalized estimate.
    One scalar weight per RGB triplet prevents channel-dependent color seams.
    """

    _validate_config(config)
    if not frames:
        raise EclipseHDRError("Cannot merge an empty bracket")
    count = len(frames)
    if len(offsets_yx) != count or len(exposure_values) != count:
        raise EclipseHDRError("Frame, offset, and exposure counts must match")
    if not 0 <= reference_index < count:
        raise EclipseHDRError(f"Reference index {reference_index} is outside the bracket")
    if saturation_masks is None:
        masks: Sequence[Optional[np.ndarray]] = [None] * count
    elif len(saturation_masks) != count:
        raise EclipseHDRError("Saturation-mask count must match frame count")
    else:
        masks = saturation_masks

    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise EclipseHDRError(f"Developed frame dimensions differ: {sorted(shapes)}")
    height, width, channels = frames[0].shape
    if channels != 3:
        raise EclipseHDRError(f"Expected RGB frames, got shape {frames[0].shape}")
    for index, frame in enumerate(frames):
        if frame.dtype != np.uint16:
            raise EclipseHDRError(
                f"Developed frame {index} must be uint16 linear RGB, got {frame.dtype}"
            )
    offsets = np.asarray(offsets_yx, dtype=np.float64)
    if offsets.shape != (count, 2) or not np.isfinite(offsets).all():
        raise EclipseHDRError("All translation offsets must be finite (dy, dx) pairs")
    for index, mask in enumerate(masks):
        if mask is not None and tuple(mask.shape) != (height, width):
            raise EclipseHDRError(
                f"Saturation mask {index} shape {mask.shape} does not match {(height, width)}"
            )

    exposures = np.asarray(exposure_values, dtype=np.float64)
    if np.any(~np.isfinite(exposures)) or np.any(exposures <= 0):
        raise EclipseHDRError("All exposure factors must be finite and positive")
    relative = exposures / exposures[reference_index]
    if np.any(relative < np.finfo(np.float32).tiny):
        raise EclipseHDRError("Relative exposure range is too large for float processing")

    if output is None:
        hdr = np.empty((height, width, 3), dtype=np.float32)
    else:
        if output.shape != (height, width, 3) or output.dtype != np.float32:
            raise EclipseHDRError(
                f"Output must be {(height, width, 3)} float32, got {output.shape} {output.dtype}"
            )
        hdr = output

    weighted_count = 0
    fallback_count = 0
    all_saturated_count = 0
    all_dark_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    relative_scale = max(float(np.max(relative)), 1.0)

    for y0 in range(0, height, config.chunk_rows):
        y1 = min(height, y0 + config.chunk_rows)
        rows = y1 - y0
        weighted_sum = np.zeros((rows, width, 3), dtype=np.float64)
        weight_sum = np.zeros((rows, width), dtype=np.float64)
        fallback_metric = np.full((rows, width), np.inf, dtype=np.float32)
        fallback_rgb = np.zeros((rows, width, 3), dtype=np.float32)
        saturated_ratio = np.full((rows, width), np.inf, dtype=np.float64)
        saturated_rgb = np.zeros((rows, width, 3), dtype=np.float32)
        dark_ratio = np.full((rows, width), -np.inf, dtype=np.float64)
        dark_rgb = np.zeros((rows, width, 3), dtype=np.float32)
        any_spatial = np.zeros((rows, width), dtype=bool)
        all_saturated = np.ones((rows, width), dtype=bool)
        all_dark = np.ones((rows, width), dtype=bool)

        for frame, mask, (dy, dx), exposure_ratio in zip(
            frames, masks, offsets_yx, relative
        ):
            rgb, spatial, coordinates = sample_translated_rgb_chunk(
                frame, y0, y1, float(dy), float(dx), config.interpolation_order
            )
            source_saturated = _sample_mask_chunk(mask, y0, y1, width, coordinates)
            postprocessed_saturated = _sample_postprocessed_clip_chunk(
                frame,
                y0,
                y1,
                float(dy),
                float(dx),
                config.saturation_cutoff,
                coordinates,
            )
            level = np.max(rgb, axis=2)
            saturated = (level >= config.saturation_cutoff) | postprocessed_saturated
            if source_saturated is not None:
                saturated |= source_saturated
            dark = level <= config.black_start

            any_spatial |= spatial
            all_saturated &= (~spatial) | saturated
            all_dark &= (~spatial) | dark

            low_weight = _smoothstep(
                config.black_start, config.black_full_weight, level
            )
            high_weight = 1.0 - _smoothstep(
                config.highlight_falloff, config.saturation_cutoff, level
            )
            # The signal factor favors the highest-SNR measurement that remains
            # safely below clipping.  All operations remain in linear code space.
            weight = low_weight * high_weight * level
            weight *= spatial & ~saturated
            radiance = rgb / np.float32(exposure_ratio)
            weighted_sum += weight[..., None] * radiance
            weight_sum += weight

            # Generic closest-to-midscale fallback, with deterministic exposure
            # tie-breaking for exactly clipped or exactly black samples.
            tie = np.where(
                level > 0.5,
                exposure_ratio / relative_scale,
                -exposure_ratio / relative_scale,
            )
            metric = np.abs(level - 0.5) + np.float32(1e-4) * tie.astype(np.float32)
            replace = spatial & (metric < fallback_metric)
            fallback_metric[replace] = metric[replace]
            fallback_rgb[replace] = radiance[replace]

            replace_saturated = spatial & saturated & (exposure_ratio < saturated_ratio)
            saturated_ratio[replace_saturated] = exposure_ratio
            saturated_rgb[replace_saturated] = radiance[replace_saturated]
            replace_dark = spatial & dark & (exposure_ratio > dark_ratio)
            dark_ratio[replace_dark] = exposure_ratio
            dark_rgb[replace_dark] = radiance[replace_dark]

        if not np.all(any_spatial):
            # The zero-shift reference should cover the full output by construction.
            raise EclipseHDRError("Some output pixels have no spatially valid source frame")
        usable = weight_sum > np.finfo(np.float32).eps
        chunk = np.empty((rows, width, 3), dtype=np.float32)
        chunk[usable] = (weighted_sum[usable] / weight_sum[usable, None]).astype(np.float32)
        fallback = ~usable
        chunk[fallback] = fallback_rgb[fallback]
        saturated_fallback = fallback & all_saturated
        dark_fallback = fallback & all_dark & ~all_saturated
        chunk[saturated_fallback] = saturated_rgb[saturated_fallback]
        chunk[dark_fallback] = dark_rgb[dark_fallback]
        if not np.isfinite(chunk).all():
            raise EclipseHDRError(f"Non-finite radiance produced in output rows {y0}:{y1}")

        hdr[y0:y1] = chunk
        weighted_count += int(np.count_nonzero(usable))
        fallback_count += int(np.count_nonzero(fallback))
        all_saturated_count += int(np.count_nonzero(saturated_fallback))
        all_dark_count += int(np.count_nonzero(dark_fallback))
        global_min = min(global_min, float(np.min(chunk)))
        global_max = max(global_max, float(np.max(chunk)))
        LOGGER.debug("Merged rows %d:%d of %d", y0, y1, height)

    if hasattr(hdr, "flush"):
        hdr.flush()  # type: ignore[attr-defined]
    stats = MergeStatistics(
        pixel_count=height * width,
        weighted_pixel_count=weighted_count,
        fallback_pixel_count=fallback_count,
        all_saturated_pixel_count=all_saturated_count,
        all_dark_pixel_count=all_dark_count,
        min_value=global_min,
        max_value=global_max,
        exposure_factors=tuple(float(value) for value in exposures),
        relative_exposures=tuple(float(value) for value in relative),
    )
    return hdr, stats


def create_float32_memmap(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Create a valid .npy-backed output array for a memory-bounded group merge."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
