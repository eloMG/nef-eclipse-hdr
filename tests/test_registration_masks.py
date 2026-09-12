from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from eclipsehdr.errors import RegistrationError
from eclipsehdr.models import ExposureMetadata, RegistrationConfig
from eclipsehdr.registration import (
    _downsample_mask_any,
    _registration_valid_mask,
    estimate_group_alignment,
    estimate_pair_translation,
)


def _metadata(count: int) -> list[ExposureMetadata]:
    start = datetime(2024, 4, 8, 18, 0, 0)
    return [
        ExposureMetadata(
            Path(f"frame_{index}.nef"),
            shutter_s=0.001,
            iso=100,
            aperture=8,
            timestamp=start + timedelta(seconds=0.3 * index),
        )
        for index in range(count)
    ]


def _partial_limb_scene(size: int) -> np.ndarray:
    y, x = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = 0.49 * size, 0.52 * size
    radius = np.hypot(y - cy, x - cx)
    angle = np.arctan2(y - cy, x - cx)
    limb = 0.18 * np.exp(-0.5 * ((radius - 0.25 * size) / 1.35) ** 2)
    # Retain an asymmetric arc instead of a complete, rotationally symmetric ring.
    limb *= (angle > -2.65) & (angle < 1.25)
    corona = 0.035 * np.exp(-radius / (0.22 * size))
    prominence = 0.11 * np.exp(
        -((x - (cx + 0.22 * size)) ** 2 + (y - (cy - 0.08 * size)) ** 2) / 13.0
    )
    luminance = limb + corona + prominence
    return np.stack([luminance, 0.88 * luminance, 0.68 * luminance], axis=2).astype(
        np.float32
    )


def _add_clipped_blob(
    rgb: np.ndarray, center_yx: tuple[float, float], radius: float
) -> np.ndarray:
    y, x = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
    mask = (y - center_yx[0]) ** 2 + (x - center_yx[1]) ** 2 <= radius**2
    rgb[mask] = 1.0
    return mask


def test_postprocessed_clipping_is_used_without_a_sensor_mask() -> None:
    rgb = np.zeros((64, 72, 3), dtype=np.float32)
    rgb[23:27, 31:36] = 1.0

    valid = _registration_valid_mask(rgb, None, dilation_px=0)

    assert valid is not None
    assert not np.any(valid[23:27, 31:36])
    assert valid[10, 10]


def test_thumbnail_mask_reduction_preserves_sites_between_stride_samples() -> None:
    mask = np.zeros((20, 22), dtype=bool)
    mask[3, 4] = True
    mask[19, 21] = True

    reduced = _downsample_mask_any(mask, step=5)

    assert reduced.shape == (4, 5)
    assert reduced[0, 0]
    assert reduced[-1, -1]


def test_group_rejects_saturation_mask_count_and_shape_mismatches() -> None:
    frames = [np.zeros((96, 112, 3), dtype=np.uint16) for _ in range(3)]
    config = RegistrationConfig(manual_roi_xywh=(0, 0, 96, 96))

    with pytest.raises(RegistrationError, match="count must match frame count"):
        estimate_group_alignment(
            frames,
            _metadata(3),
            1,
            config,
            saturation_masks=[None, None],
        )

    masks = [
        np.zeros((96, 112), dtype=np.uint8),
        np.zeros((95, 112), dtype=np.uint8),
        None,
    ]
    with pytest.raises(RegistrationError, match=r"mask 1 shape .* does not match"):
        estimate_group_alignment(
            frames,
            _metadata(3),
            1,
            config,
            saturation_masks=masks,
        )


def test_group_crops_full_resolution_masks_with_nonzero_manual_roi() -> None:
    roi_size = 128
    x0, y0 = 41, 27
    height, width = 190, 224
    displacement_yx = (2.25, -3.4)
    reference_roi = _partial_limb_scene(roi_size)
    moving_roi = ndimage.shift(
        reference_roi,
        shift=(displacement_yx[0], displacement_yx[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    reference_mask_roi = _add_clipped_blob(reference_roi, (64.0, 91.0), 7.0)
    moving_mask_roi = _add_clipped_blob(moving_roi, (67.0, 86.0), 7.0)

    frames = [
        np.zeros((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 3), dtype=np.float32),
    ]
    masks = [
        np.zeros((height, width), dtype=np.uint8),
        np.zeros((height, width), dtype=np.uint8),
    ]
    for frame, roi in zip(frames, (reference_roi, moving_roi)):
        frame[y0 : y0 + roi_size, x0 : x0 + roi_size] = roi
    for mask, roi_mask in zip(masks, (reference_mask_roi, moving_mask_roi)):
        mask[y0 : y0 + roi_size, x0 : x0 + roi_size] = roi_mask

    result = estimate_group_alignment(
        frames,
        _metadata(2),
        0,
        RegistrationConfig(
            max_shift_px=8,
            crop_size_px=roi_size,
            manual_roi_xywh=(x0, y0, roi_size, roi_size),
            upsample_factor=20,
        ),
        saturation_masks=masks,
    )

    assert (result.crop.x0, result.crop.y0, result.crop.shape) == (
        x0,
        y0,
        (roi_size, roi_size),
    )
    assert abs(result.frames[1].dy + displacement_yx[0]) < 0.3
    assert abs(result.frames[1].dx + displacement_yx[1]) < 0.3


def test_masks_preserve_translation_with_partial_limb_and_clipped_blobs() -> None:
    size = 160
    displacement_yx = (2.35, -3.6)
    reference = _partial_limb_scene(size)
    moving = ndimage.shift(
        reference,
        shift=(displacement_yx[0], displacement_yx[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    # Exposure-dependent breakout covers different sections of the useful arc.
    # Each clipped area is small enough that substantial common support remains.
    reference_mask = _add_clipped_blob(reference, (62.0, 111.0), 10.0)
    moving_mask = _add_clipped_blob(moving, (70.0, 105.0), 12.0)
    candidates = estimate_pair_translation(
        reference,
        moving,
        RegistrationConfig(max_shift_px=8, crop_size_px=size, upsample_factor=20),
        reference_saturation_mask=reference_mask,
        moving_saturation_mask=moving_mask,
    )

    assert abs(candidates[0].dy + displacement_yx[0]) < 0.15
    assert abs(candidates[0].dx + displacement_yx[1]) < 0.15


def test_tight_roi_keeps_mask_when_clipping_exceeds_five_percent() -> None:
    size = 160
    displacement_yx = (2.35, -3.6)
    reference = _partial_limb_scene(size)
    moving = ndimage.shift(
        reference,
        shift=(displacement_yx[0], displacement_yx[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    reference_mask = _add_clipped_blob(reference, (62.0, 111.0), 21.0)
    moving_mask = _add_clipped_blob(moving, (70.0, 105.0), 21.0)

    assert np.mean(reference_mask) == pytest.approx(0.0536328125)
    assert _registration_valid_mask(reference, reference_mask) is not None
    candidates = estimate_pair_translation(
        reference,
        moving,
        RegistrationConfig(max_shift_px=8, crop_size_px=size, upsample_factor=20),
        reference_saturation_mask=reference_mask,
        moving_saturation_mask=moving_mask,
    )

    error = np.hypot(
        candidates[0].dy + displacement_yx[0],
        candidates[0].dx + displacement_yx[1],
    )
    assert error < 0.5
