from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage

from eclipsehdr.models import (
    ExposureMetadata,
    FrameAlignment,
    RegistrationCandidate,
    RegistrationConfig,
)
from eclipsehdr.registration import (
    _has_independent_support,
    _method_disagreement,
    estimate_group_alignment,
    estimate_pair_translation,
    physical_sanity_checks,
)


def _solar_scene(size: int = 256) -> np.ndarray:
    y, x = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size * 0.49, size * 0.52
    radius = np.hypot(y - cy, x - cx)
    disc = 0.55 * (radius <= 35)
    limb = 0.32 * np.exp(-0.5 * ((radius - 35) / 1.4) ** 2)
    corona = 0.08 * np.exp(-radius / 38.0)
    # Asymmetric prominence keeps the translation peak unique.
    prominence = 0.22 * np.exp(-((x - (cx + 31)) ** 2 + (y - (cy - 18)) ** 2) / 18.0)
    luminance = np.clip(disc + limb + corona + prominence, 0, None)
    rgb = np.stack([luminance, luminance * 0.88, luminance * 0.62], axis=2)
    return rgb.astype(np.float32)


def _u16_shifted(scene: np.ndarray, displacement_yx: tuple[float, float], exposure: float) -> np.ndarray:
    measured = np.clip(scene * exposure, 0.0, 1.0)
    shifted = ndimage.shift(
        measured,
        shift=(displacement_yx[0], displacement_yx[1], 0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.rint(shifted * 65535.0).astype(np.uint16)


def _metadata(count: int, reference: int) -> list[ExposureMetadata]:
    start = datetime(2024, 4, 8, 18, 0, 0)
    return [
        ExposureMetadata(
            Path(f"frame_{i}.nef"),
            shutter_s=1 / 1000,
            iso=100,
            aperture=8,
            timestamp=start + timedelta(seconds=0.3 * i),
        )
        for i in range(count)
    ]


def test_pair_shift_sign_and_axis_convention() -> None:
    scene = _solar_scene()
    reference = _u16_shifted(scene, (0.0, 0.0), 1.0)
    moving = _u16_shifted(scene, (2.35, -3.6), 0.45)
    config = RegistrationConfig(max_shift_px=8, crop_size_px=128, upsample_factor=20)
    best = estimate_pair_translation(reference, moving, config)[0]
    # The returned shift is what must be applied, hence the opposite displacement.
    assert abs(best.dy - (-2.35)) < 0.25
    assert abs(best.dx - 3.6) < 0.25


def test_five_frame_exposure_variant_alignment_and_linear_motion() -> None:
    scene = _solar_scene()
    displacements = [(-3.2, 1.6), (-1.6, 0.8), (0.0, 0.0), (1.6, -0.8), (3.2, -1.6)]
    exposures = [0.3, 0.6, 1.0, 1.8, 3.2]
    frames = [_u16_shifted(scene, shift, exposure) for shift, exposure in zip(displacements, exposures)]
    config = RegistrationConfig(
        max_shift_px=8,
        crop_size_px=192,
        upsample_factor=20,
        manual_roi_xywh=(0, 0, 256, 256),
        max_linear_residual_px=0.5,
        max_acceleration_px=0.75,
    )
    result = estimate_group_alignment(frames, _metadata(5, 2), 2, config)
    expected = [(-dy, -dx) for dy, dx in displacements]
    for actual, (expected_dy, expected_dx) in zip(result.frames, expected):
        assert abs(actual.dy - expected_dy) < 0.3
        assert abs(actual.dx - expected_dx) < 0.3
    assert not result.suspicious, result.issues


def test_physical_check_flags_discontinuous_outlier() -> None:
    offsets = [
        FrameAlignment(0, 2.0, -2.0, "test", 1.0),
        FrameAlignment(1, 1.0, -1.0, "test", 1.0),
        FrameAlignment(2, 0.0, 0.0, "reference", 1.0),
        FrameAlignment(3, 8.0, 8.0, "test", 1.0),
        FrameAlignment(4, -2.0, 2.0, "test", 1.0),
    ]
    issues, _, _, residual = physical_sanity_checks(
        offsets,
        _metadata(5, 2),
        2,
        RegistrationConfig(max_shift_px=20, max_linear_residual_px=1, max_acceleration_px=2),
    )
    assert residual > 1
    assert any("constant linear motion" in issue for issue in issues)
    assert any("discontinuity" in issue for issue in issues)


def test_consensus_ignores_garbage_support_and_flags_good_disagreement() -> None:
    best = RegistrationCandidate(0.0, 0.0, "bounded-ZNCC/log-gradient/phase", 0.90, 0.8, 3.0, 0.1)
    garbage_same_shift = RegistrationCandidate(
        0.0, 0.0, "bounded-ZNCC/mtb/phase", -1.0, -1.0, 0.1, 1.0
    )
    good_disagreement = RegistrationCandidate(
        0.0, 10.0, "bounded-ZNCC/log-highpass/phase", 0.89, 0.8, 3.0, 0.1
    )
    frame = FrameAlignment(
        0,
        dy=best.dy,
        dx=best.dx,
        method=best.method,
        score=best.score,
        candidates=(best, good_disagreement, garbage_same_shift),
    )
    assert not _has_independent_support(frame, threshold=1.5, min_psr=1.25, min_score=0.05)
    assert _method_disagreement(frame, threshold=1.5, min_psr=1.25, min_score=0.05)


def test_irregular_timestamps_do_not_flag_exact_constant_velocity() -> None:
    start = datetime(2024, 4, 8, 18, 0, 0)
    seconds = [0.0, 0.1, 0.7, 0.8, 1.4]
    reference = 2
    metadata = [
        ExposureMetadata(
            Path(f"frame_{i}.nef"),
            0.001,
            100,
            8,
            start + timedelta(seconds=seconds[i]),
        )
        for i in range(5)
    ]
    relative_times = np.asarray(seconds) - seconds[reference]
    offsets = [
        FrameAlignment(i, dy=2.0 * time, dx=3.0 * time, method="test", score=1.0)
        for i, time in enumerate(relative_times)
    ]
    issues, basis, _, residual = physical_sanity_checks(
        offsets,
        metadata,
        reference,
        RegistrationConfig(max_shift_px=20, max_linear_residual_px=0.01, max_acceleration_px=0.01),
    )
    assert basis == "capture_timestamp"
    assert residual < 1e-9
    assert not any("discontinuity" in issue for issue in issues)


def test_auto_roi_rejects_fixed_horizon_and_tracks_dim_sun() -> None:
    height, width = 384, 512
    y, x = np.mgrid[:height, :width].astype(np.float32)
    displacements = [(-4.0, 2.0), (-2.0, 1.0), (0.0, 0.0), (2.0, -1.0), (4.0, -2.0)]
    frames = []
    for dy, dx in displacements:
        scene = np.zeros((height, width), dtype=np.float32)
        scene[300:316, 16:496] = 0.9  # stationary elongated horizon/cloud band
        sun = 0.38 * np.exp(-((x - (420 + dx)) ** 2 + (y - (105 + dy)) ** 2) / 180.0)
        scene = np.maximum(scene, sun)
        rgb = np.stack([scene, 0.9 * scene, 0.7 * scene], axis=2)
        frames.append(np.rint(rgb * 65535).astype(np.uint16))
    result = estimate_group_alignment(
        frames,
        _metadata(5, 2),
        2,
        RegistrationConfig(
            max_shift_px=8,
            crop_size_px=128,
            preview_max_dim=512,
            max_linear_residual_px=0.5,
            max_acceleration_px=0.75,
        ),
    )
    assert result.crop.x0 > 300
    assert result.crop.y1 < 220
    expected = [(-dy, -dx) for dy, dx in displacements]
    for actual, (expected_dy, expected_dx) in zip(result.frames, expected):
        assert abs(actual.dy - expected_dy) < 0.3
        assert abs(actual.dx - expected_dx) < 0.3


def test_totality_diamond_ring_regime_remains_translation_only() -> None:
    size = 256
    y, x = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = 126.0, 132.0
    radius = np.hypot(y - cy, x - cx)
    corona = 0.12 * np.exp(-np.maximum(radius - 30.0, 0.0) / 35.0) * (radius >= 27.0)
    ring = 0.12 * np.exp(-0.5 * ((radius - 30.0) / 1.5) ** 2)
    diamond = 0.75 * np.exp(-((x - (cx + 28)) ** 2 + (y - (cy - 8)) ** 2) / 14.0)
    luminance = corona + ring + diamond
    scene = np.stack([luminance, 0.9 * luminance, 0.75 * luminance], axis=2)
    displacements = [(-3.2, 1.6), (-1.6, 0.8), (0.0, 0.0), (1.6, -0.8), (3.2, -1.6)]
    exposures = [0.1, 0.3, 1.0, 3.0, 8.0]
    frames = [_u16_shifted(scene, shift, exposure) for shift, exposure in zip(displacements, exposures)]
    result = estimate_group_alignment(
        frames,
        _metadata(5, 2),
        2,
        RegistrationConfig(
            max_shift_px=8,
            crop_size_px=192,
            manual_roi_xywh=(0, 0, size, size),
            max_linear_residual_px=0.5,
            max_acceleration_px=0.75,
        ),
    )
    assert not result.suspicious, result.issues
    expected = [(-dy, -dx) for dy, dx in displacements]
    for actual, (expected_dy, expected_dx) in zip(result.frames, expected):
        assert abs(actual.dy - expected_dy) < 0.3
        assert abs(actual.dx - expected_dx) < 0.3
