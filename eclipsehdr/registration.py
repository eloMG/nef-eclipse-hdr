"""Exposure-tolerant, strictly translation-only registration."""

from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal.windows import tukey
from skimage.feature import match_template
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

from .errors import RegistrationError
from .metadata import usable_time_axis
from .models import (
    AlignmentResult,
    CropRegion,
    ExposureMetadata,
    FrameAlignment,
    RegistrationCandidate,
    RegistrationConfig,
)

LOGGER = logging.getLogger(__name__)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Linear-light luminance from uint16 or normalized RGB."""

    scale = np.float32(1.0 / 65535.0) if rgb.dtype == np.uint16 else np.float32(1.0)
    values = rgb.astype(np.float32, copy=False)
    return scale * (
        np.float32(0.2126) * values[..., 0]
        + np.float32(0.7152) * values[..., 1]
        + np.float32(0.0722) * values[..., 2]
    )


def _normalized_signal(luminance: np.ndarray) -> np.ndarray:
    finite = np.isfinite(luminance)
    if not np.any(finite):
        return np.zeros_like(luminance, dtype=np.float32)
    values = luminance[finite]
    background = float(np.percentile(values, 25.0))
    positive = np.maximum(luminance.astype(np.float32, copy=False) - background, 0.0)
    nonzero = positive[positive > 0]
    if nonzero.size < 16:
        return np.zeros_like(positive)
    scale = float(np.percentile(nonzero, 99.8))
    if not math.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        return np.zeros_like(positive)
    # asinh is approximately linear near black and logarithmic in highlights.
    compressed = np.arcsinh(positive / np.float32(max(scale * 0.03, 1e-8)))
    top = float(np.percentile(compressed[finite], 99.8))
    if top <= 0:
        return np.zeros_like(positive)
    return np.clip(compressed / np.float32(top), 0.0, 1.0).astype(np.float32)


def _clamped_bounds(center: float, length: int, limit: int) -> tuple[int, int]:
    length = min(max(1, length), limit)
    start = int(round(center - length / 2.0))
    start = max(0, min(start, limit - length))
    return start, start + length


def _validate_manual_roi(
    roi_xywh: tuple[int, int, int, int], shape: tuple[int, int]
) -> CropRegion:
    x, y, width, height = roi_xywh
    image_height, image_width = shape
    if width < 64 or height < 64:
        raise RegistrationError("Manual ROI must be at least 64 x 64 pixels")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise RegistrationError(
            f"Manual ROI {roi_xywh} lies outside the {image_width} x {image_height} image"
        )
    return CropRegion(y, y + height, x, x + width, method="manual")


def find_registration_crop(
    frames: Sequence[np.ndarray], config: RegistrationConfig
) -> CropRegion:
    """Find one shared crop around the dominant compact signal in thumbnails."""

    if not frames:
        raise RegistrationError("Cannot find a registration crop in an empty bracket")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise RegistrationError(f"Developed frame dimensions differ: {sorted(shapes)}")
    height, width, channels = frames[0].shape
    if channels != 3:
        raise RegistrationError(f"Expected RGB frames, got shape {frames[0].shape}")
    if config.manual_roi_xywh is not None:
        return _validate_manual_roi(config.manual_roi_xywh, (height, width))

    step = max(1, int(math.ceil(max(height, width) / config.preview_max_dim)))
    preview_shape = frames[0][::step, ::step].shape[:2]
    aggregate = np.zeros(preview_shape, dtype=np.float32)
    usable = 0
    for frame in frames:
        signal = _normalized_signal(_luminance(np.asarray(frame[::step, ::step, :])))
        if np.count_nonzero(signal > 0) >= 16:
            signal = ndimage.gaussian_filter(signal, sigma=1.25)
            aggregate = np.maximum(aggregate, signal)
            usable += 1
    if usable == 0 or float(np.max(aggregate)) <= 0:
        raise RegistrationError(
            "No compact foreground signal was found for registration; specify --roi X,Y,W,H"
        )

    gy = ndimage.sobel(aggregate, axis=0, mode="nearest")
    gx = ndimage.sobel(aggregate, axis=1, mode="nearest")
    gradient = np.hypot(gx, gy)
    gradient_scale = float(np.percentile(gradient, 99.5))
    if gradient_scale > 0:
        gradient = np.clip(gradient / gradient_scale, 0.0, 1.0)
    energy = ndimage.gaussian_filter(aggregate + np.float32(0.35) * gradient, sigma=1.0)

    nonzero = energy[energy > 0]
    try:
        threshold = float(threshold_otsu(nonzero)) if nonzero.size >= 16 else 0.1
    except ValueError:
        threshold = 0.1
    threshold = max(0.04, threshold)
    foreground = energy >= threshold
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=2)
    labels, count = ndimage.label(foreground)
    if count == 0:
        raise RegistrationError(
            "Automatic Sun/corona localization found no connected signal; specify --roi X,Y,W,H"
        )

    components: list[tuple[float, int, tuple[slice, slice], float, float]] = []
    rejected_elongated = 0
    objects = ndimage.find_objects(labels)
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        ys, xs = slices
        component = labels[ys, xs] == label_index
        area = int(np.count_nonzero(component))
        if area < 9:
            continue
        box_h, box_w = component.shape
        aspect = max(box_h / max(box_w, 1), box_w / max(box_h, 1))
        span_y = box_h / foreground.shape[0]
        span_x = box_w / foreground.shape[1]
        line_like = aspect > 4.0 or (
            max(span_x, span_y) > 0.60 and min(span_x, span_y) < 0.25
        )
        if line_like:
            rejected_elongated += 1
            continue
        compactness = area / float(box_h * box_w)
        mass = float(np.sum(energy[ys, xs][component]))
        # Strongly elongated horizon/cloud structures lose to a compact solar limb.
        geometry_weight = compactness / max(1.0, aspect) ** 1.5
        score = mass * geometry_weight
        local_y, local_x = ndimage.center_of_mass(energy[ys, xs] * component)
        components.append((score, label_index, (ys, xs), ys.start + local_y, xs.start + local_x))
    if not components:
        detail = " Elongated horizon/cloud-like components were rejected." if rejected_elongated else ""
        raise RegistrationError(
            "No compact solar candidate remained after foreground checks; "
            f"specify --roi X,Y,W,H.{detail}"
        )
    components.sort(key=lambda item: item[0], reverse=True)
    best = components[0]
    warnings: list[str] = []
    if len(components) > 1 and components[1][0] >= 0.75 * best[0]:
        warnings.append(
            "Automatic ROI was ambiguous: the second-best foreground component was nearly as strong."
        )

    ys, xs = best[2]
    feature_h = max(1, ys.stop - ys.start) * step
    feature_w = max(1, xs.stop - xs.start) * step
    center_y = (best[3] + 0.5) * step
    center_x = (best[4] + 0.5) * step
    minimum = max(config.crop_size_px, int(math.ceil(0.20 * min(height, width))))
    margin = int(math.ceil(2 * config.max_shift_px + 32))
    crop_h = min(height, max(minimum, 2 * feature_h + margin))
    crop_w = min(width, max(minimum, 2 * feature_w + margin))
    y0, y1 = _clamped_bounds(center_y, crop_h, height)
    x0, x1 = _clamped_bounds(center_x, crop_w, width)
    if y0 == 0 or x0 == 0 or y1 == height or x1 == width:
        warnings.append("Registration ROI touches an image boundary; inspect the alignment preview.")
    return CropRegion(y0, y1, x0, x1, warnings=tuple(warnings))


def _robust_standardize(image: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32, copy=False)
    median = float(np.median(result))
    mad = float(np.median(np.abs(result - median)))
    scale = max(1.4826 * mad, float(np.std(result)), 1e-6)
    result = np.clip((result - median) / scale, -8.0, 8.0)
    wy = tukey(result.shape[0], alpha=0.15).astype(np.float32)
    wx = tukey(result.shape[1], alpha=0.15).astype(np.float32)
    return np.ascontiguousarray(result * wy[:, None] * wx[None, :], dtype=np.float32)


def registration_representations(rgb_crop: np.ndarray) -> dict[str, np.ndarray]:
    """Create three brightness-tolerant views; none changes registration geometry."""

    signal = _normalized_signal(_luminance(rgb_crop))
    if np.count_nonzero(signal) < 16:
        raise RegistrationError("Registration crop contains too little non-black signal")
    smooth = ndimage.gaussian_filter(signal, sigma=1.0)
    gy = ndimage.sobel(smooth, axis=0, mode="nearest")
    gx = ndimage.sobel(smooth, axis=1, mode="nearest")
    gradient = np.hypot(gx, gy)
    highpass = smooth - ndimage.gaussian_filter(smooth, sigma=8.0)
    positive = smooth[smooth > 0]
    try:
        threshold = float(threshold_otsu(positive)) if positive.size >= 16 else 0.5
    except ValueError:
        threshold = 0.5
    mtb = np.where(smooth >= threshold, 1.0, -1.0).astype(np.float32)
    # Suppress bitmap decisions in the uncertain band around the threshold.
    exclusion = np.abs(smooth - threshold) < max(0.02, 0.08 * threshold)
    mtb[exclusion] = 0.0
    return {
        "log-gradient": _robust_standardize(gradient),
        "log-highpass": _robust_standardize(highpass),
        "mtb": _robust_standardize(mtb),
    }


def _peak_statistics(scores: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    peak_value = float(scores[peak])
    sidelobes = scores.astype(np.float64, copy=True)
    y, x = peak
    sidelobes[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = np.nan
    finite = sidelobes[np.isfinite(sidelobes)]
    if finite.size == 0:
        return peak_value, 0.0
    standard_deviation = float(np.std(finite))
    psr = (peak_value - float(np.mean(finite))) / max(standard_deviation, 1e-9)
    return peak_value, psr


def _bounded_coarse_shift(
    reference: np.ndarray, moving: np.ndarray, max_shift: float
) -> tuple[float, float, float, float]:
    """Find an integer shift inside the configured box using bounded ZNCC."""

    margin = int(math.ceil(max_shift))
    if reference.shape != moving.shape:
        raise RegistrationError("Pairwise registration arrays differ in size")
    if min(reference.shape) <= 2 * margin + 32:
        raise RegistrationError(
            f"Registration crop {reference.shape} is too small for +/-{max_shift:g} px search"
        )
    template = reference[margin:-margin, margin:-margin]
    scores = match_template(moving, template, pad_input=False)
    if not np.any(np.isfinite(scores)):
        raise RegistrationError("Bounded translation correlation produced no finite score")
    peak_flat = int(np.nanargmax(scores))
    peak = np.unravel_index(peak_flat, scores.shape)
    coarse_score, psr = _peak_statistics(scores, peak)
    # A template starting at margin+d in the moving image means the moving
    # content was displaced by +d; applying margin-(margin+d) reverses it.
    dy = float(margin - peak[0])
    dx = float(margin - peak[1])
    return dy, dx, coarse_score, psr


def _overlap_slices(
    shape: tuple[int, int], dy: float, dx: float, inset: int = 4
) -> tuple[slice, slice]:
    height, width = shape
    border_y = int(math.ceil(abs(dy))) + inset
    border_x = int(math.ceil(abs(dx))) + inset
    if height <= 2 * border_y or width <= 2 * border_x:
        return slice(0, 0), slice(0, 0)
    return slice(border_y, height - border_y), slice(border_x, width - border_x)


def _aligned_zncc(reference: np.ndarray, moving: np.ndarray, dy: float, dx: float) -> float:
    aligned = ndimage.shift(
        moving,
        shift=(dy, dx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    ys, xs = _overlap_slices(reference.shape, dy, dx)
    a = reference[ys, xs].astype(np.float64, copy=False).ravel()
    b = aligned[ys, xs].astype(np.float64, copy=False).ravel()
    if a.size < 64:
        return -1.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(a, b) / denominator)


def _refine_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    coarse_dy: float,
    coarse_dx: float,
    upsample_factor: int,
    normalization: str | None,
) -> tuple[float, float, float | None]:
    coarse_aligned = ndimage.shift(
        moving,
        shift=(coarse_dy, coarse_dx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    ys, xs = _overlap_slices(reference.shape, coarse_dy, coarse_dx, inset=8)
    reference_inner = reference[ys, xs]
    moving_inner = coarse_aligned[ys, xs]
    try:
        residual, error, _ = phase_cross_correlation(
            reference_inner,
            moving_inner,
            upsample_factor=upsample_factor,
            disambiguate=True,
            normalization=normalization,
        )
        residual_dy, residual_dx = float(residual[0]), float(residual[1])
        if not np.isfinite([residual_dy, residual_dx]).all():
            raise ValueError("non-finite residual")
        # Coarse placement is already bounded.  A large residual signals a false
        # periodic solution, so keep the defensible integer result instead.
        if abs(residual_dy) > 2.0 or abs(residual_dx) > 2.0:
            return coarse_dy, coarse_dx, None
        finite_error = float(error) if np.isfinite(error) else None
        return coarse_dy + residual_dy, coarse_dx + residual_dx, finite_error
    except (ValueError, FloatingPointError):
        return coarse_dy, coarse_dx, None


def estimate_pair_translation(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    config: RegistrationConfig,
) -> tuple[RegistrationCandidate, ...]:
    """Return translation candidates sorted by aligned feature correlation."""

    reference_views = registration_representations(reference_rgb)
    moving_views = registration_representations(moving_rgb)
    candidates: list[RegistrationCandidate] = []
    for name in reference_views:
        reference = reference_views[name]
        moving = moving_views[name]
        coarse_dy, coarse_dx, coarse_score, psr = _bounded_coarse_shift(
            reference, moving, config.max_shift_px
        )
        seen_results: set[tuple[float, float, str]] = set()
        for normalization in ("phase", None):
            dy, dx, error = _refine_shift(
                reference,
                moving,
                coarse_dy,
                coarse_dx,
                config.upsample_factor,
                normalization,
            )
            if abs(dy) > config.max_shift_px or abs(dx) > config.max_shift_px:
                continue
            score = _aligned_zncc(reference, moving, dy, dx)
            if error is None:
                suffix = "coarse-only"
            else:
                suffix = "phase" if normalization == "phase" else "unnormalized"
            result_key = (round(dy, 6), round(dx, 6), suffix)
            if result_key in seen_results:
                continue
            seen_results.add(result_key)
            candidates.append(
                RegistrationCandidate(
                    dy=dy,
                    dx=dx,
                    method=f"bounded-ZNCC/{name}/{suffix}",
                    score=score,
                    coarse_score=coarse_score,
                    psr=psr,
                    refinement_error=error,
                )
            )
    if not candidates:
        raise RegistrationError("No translation candidate remained inside the maximum-shift box")
    # Collapse duplicates from the two refinements while retaining distinct feature evidence.
    candidates.sort(key=lambda item: (item.score, item.psr, item.coarse_score), reverse=True)
    return tuple(candidates)


def _best_candidate_per_family(frame: FrameAlignment) -> dict[str, RegistrationCandidate]:
    result: dict[str, RegistrationCandidate] = {}
    for candidate in frame.candidates:
        family = candidate.method.split("/")[1]
        if family not in result:
            result[family] = candidate
    return result


def _method_disagreement(
    frame: FrameAlignment, threshold: float, min_psr: float, min_score: float
) -> bool:
    if not frame.candidates:
        return False
    best = frame.candidates[0]
    best_family = best.method.split("/")[1]
    for family, candidate in _best_candidate_per_family(frame).items():
        if family == best_family:
            continue
        # ZNCC values from bitmap, high-pass, and gradient representations are
        # not on a directly comparable scale. Judge each family's own best peak
        # against absolute quality floors, then compare their translations.
        plausible = candidate.score >= min_score and candidate.psr >= min_psr
        if plausible and math.hypot(best.dx - candidate.dx, best.dy - candidate.dy) > threshold:
            return True
    return False


def _has_independent_support(
    frame: FrameAlignment, threshold: float, min_psr: float, min_score: float
) -> bool:
    if not frame.candidates:
        return False
    best = frame.candidates[0]
    best_family = best.method.split("/")[1]
    for family, candidate in _best_candidate_per_family(frame).items():
        if family == best_family:
            continue
        plausible = candidate.score >= min_score and candidate.psr >= min_psr
        if plausible and math.hypot(best.dx - candidate.dx, best.dy - candidate.dy) <= threshold:
            return True
    return False


def physical_sanity_checks(
    frames: Sequence[FrameAlignment],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    config: RegistrationConfig,
) -> tuple[list[str], str, tuple[float, float], float]:
    """Check bounds, confidence, constant motion, and adjacent continuity."""

    issues: list[str] = []
    offsets_yx = np.asarray([(frame.dy, frame.dx) for frame in frames], dtype=np.float64)
    for frame in frames:
        if abs(frame.dx) > config.max_shift_px or abs(frame.dy) > config.max_shift_px:
            issues.append(
                f"frame {frame.frame_index}: shift ({frame.dx:+.3f}, {frame.dy:+.3f}) exceeds "
                f"the +/-{config.max_shift_px:g} px component limit"
            )
        if frame.frame_index != reference_index and frame.score < config.min_score:
            issues.append(
                f"frame {frame.frame_index}: aligned feature score {frame.score:.3f} is below "
                f"the {config.min_score:.3f} minimum"
            )
        if frame.frame_index != reference_index and frame.candidates:
            selected = frame.candidates[0]
            if selected.refinement_error is None:
                issues.append(
                    f"frame {frame.frame_index}: subpixel phase refinement failed; "
                    "only the bounded integer translation was available"
                )
            if selected.psr < config.min_psr:
                issues.append(
                    f"frame {frame.frame_index}: bounded correlation PSR {selected.psr:.3f} "
                    f"is below the {config.min_psr:.3f} minimum"
                )
            if (
                abs(selected.dx) >= config.max_shift_px - 0.5
                or abs(selected.dy) >= config.max_shift_px - 0.5
            ):
                issues.append(
                    f"frame {frame.frame_index}: correlation peak touches the configured "
                    "maximum-shift boundary"
                )
            if not _has_independent_support(
                frame,
                config.max_method_disagreement_px,
                config.min_psr,
                config.min_score,
            ):
                issues.append(
                    f"frame {frame.frame_index}: selected translation lacks corroboration "
                    "from an independent registration representation"
                )
        if _method_disagreement(
            frame,
            config.max_method_disagreement_px,
            config.min_psr,
            config.min_score,
        ):
            issues.append(
                f"frame {frame.frame_index}: plausible registration representations disagree by "
                f"more than {config.max_method_disagreement_px:g} px"
            )

    times, basis = usable_time_axis(metadata, reference_index)
    denominator = float(np.dot(times, times))
    if denominator > 0:
        velocity_yx = np.sum(times[:, None] * offsets_yx, axis=0) / denominator
        predicted = times[:, None] * velocity_yx[None, :]
        residuals = np.linalg.norm(offsets_yx - predicted, axis=1)
        max_residual = float(np.max(residuals))
    else:
        velocity_yx = np.zeros(2, dtype=np.float64)
        max_residual = 0.0
    if max_residual > config.max_linear_residual_px:
        issues.append(
            f"offsets depart from constant linear motion by {max_residual:.3f} px "
            f"(limit {config.max_linear_residual_px:g} px)"
        )

    if len(frames) >= 3:
        steps = np.diff(times)
        uniform_steps = np.allclose(steps, steps[0], rtol=0.05, atol=1e-6)
        if uniform_steps:
            accelerations = np.diff(offsets_yx, n=2, axis=0)
            max_acceleration = float(np.max(np.linalg.norm(accelerations, axis=1)))
            acceleration_label = "adjacent offsets"
        else:
            velocities = np.diff(offsets_yx, axis=0) / steps[:, None]
            typical_step = float(np.median(np.abs(steps)))
            velocity_changes = np.diff(velocities, axis=0) * typical_step
            max_acceleration = float(np.max(np.linalg.norm(velocity_changes, axis=1)))
            acceleration_label = "time-normalized adjacent velocities"
        if max_acceleration > config.max_acceleration_px:
            issues.append(
                f"{acceleration_label} have a {max_acceleration:.3f} px-equivalent discontinuity "
                f"(limit {config.max_acceleration_px:g} px)"
            )
    return issues, basis, (float(velocity_yx[1]), float(velocity_yx[0])), max_residual


def estimate_group_alignment(
    frames: Sequence[np.ndarray],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    config: RegistrationConfig,
) -> AlignmentResult:
    """Align every frame directly to the central exposure using x/y translation only."""

    if len(frames) != len(metadata):
        raise RegistrationError("Frame and metadata counts differ during registration")
    if not 0 <= reference_index < len(frames):
        raise RegistrationError(f"Reference index {reference_index} is outside the bracket")
    crop = find_registration_crop(frames, config)
    rgb_crops = [frame[crop.y0 : crop.y1, crop.x0 : crop.x1, :] for frame in frames]
    reference = np.asarray(rgb_crops[reference_index])
    alignments: list[FrameAlignment] = []
    for index, moving in enumerate(rgb_crops):
        if index == reference_index:
            alignments.append(
                FrameAlignment(index, dy=0.0, dx=0.0, method="reference", score=1.0)
            )
            continue
        candidates = estimate_pair_translation(reference, np.asarray(moving), config)
        best = candidates[0]
        alignments.append(
            FrameAlignment(
                frame_index=index,
                dy=best.dy,
                dx=best.dx,
                method=best.method,
                score=best.score,
                candidates=candidates,
            )
        )

    issues, basis, velocity, max_residual = physical_sanity_checks(
        alignments, metadata, reference_index, config
    )
    issues = list(crop.warnings) + issues
    return AlignmentResult(
        reference_index=reference_index,
        frames=tuple(alignments),
        crop=crop,
        suspicious=bool(issues),
        issues=tuple(issues),
        time_basis=basis,
        linear_velocity_xy=velocity,
        max_linear_residual_px=max_residual,
    )
