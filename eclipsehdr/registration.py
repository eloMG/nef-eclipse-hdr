"""Exposure-tolerant, strictly translation-only registration."""

from __future__ import annotations

import logging
import math
from itertools import combinations
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

# Strong, independent image evidence may legitimately contradict the smooth
# motion model because of tracker correction or camera shake.  Larger jumps
# remain blocking even when two feature representations agree locally.
_MAX_CONSENSUS_LINEAR_RESIDUAL_PX = 3.0
_MAX_CONSENSUS_ACCELERATION_PX = 5.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Linear-light luminance from uint16 or normalized RGB."""

    scale = np.float32(1.0 / 65535.0) if rgb.dtype == np.uint16 else np.float32(1.0)
    values = rgb.astype(np.float32, copy=False)
    return scale * (
        np.float32(0.2126) * values[..., 0]
        + np.float32(0.7152) * values[..., 1]
        + np.float32(0.0722) * values[..., 2]
    )


def _normalized_signal(
    luminance: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    finite = np.isfinite(luminance)
    if valid_mask is not None:
        if valid_mask.shape != luminance.shape:
            raise RegistrationError(
                f"Registration validity mask {valid_mask.shape} does not match {luminance.shape}"
            )
        finite &= valid_mask.astype(bool, copy=False)
    if not np.any(finite):
        return np.zeros_like(luminance, dtype=np.float32)
    values = luminance[finite]
    background = float(np.percentile(values, 25.0))
    positive = np.maximum(luminance.astype(np.float32, copy=False) - background, 0.0)
    nonzero = positive[finite & (positive > 0)]
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
    result = np.clip(compressed / np.float32(top), 0.0, 1.0).astype(np.float32)
    if valid_mask is not None and not np.all(finite):
        # Replace unreliable highlights by a smooth continuation of nearby
        # unsaturated signal.  This removes their registration evidence without
        # introducing a sharp mask-shaped edge that could itself be correlated.
        weights = finite.astype(np.float32)
        numerator = ndimage.gaussian_filter(result * weights, sigma=4.0, mode="nearest")
        denominator = ndimage.gaussian_filter(weights, sigma=4.0, mode="nearest")
        filled = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(np.float32).eps,
        )
        result = np.where(finite, result, filled).astype(np.float32, copy=False)
    return result


def _registration_valid_mask(
    rgb: np.ndarray,
    saturation_mask: np.ndarray | None,
    *,
    dilation_px: int = 4,
) -> np.ndarray | None:
    if saturation_mask is not None and saturation_mask.shape != rgb.shape[:2]:
        raise RegistrationError(
            f"Saturation mask {saturation_mask.shape} does not match RGB crop {rgb.shape[:2]}"
        )
    invalid = np.zeros(rgb.shape[:2], dtype=bool)
    if saturation_mask is not None:
        invalid |= saturation_mask.astype(bool, copy=False)
    if rgb.dtype == np.uint16:
        invalid |= np.any(rgb >= int(math.ceil(0.98 * 65535.0)), axis=2)
    else:
        invalid |= np.any(rgb >= 0.98, axis=2)
    # Cover demosaic/bloom boundaries that remain unreliable immediately
    # outside the conservative sensor mask.  Do not reject a mask based on its
    # percentage of the ROI: tightening the ROI must not suddenly turn the same
    # clipped feature back into registration evidence.
    if dilation_px > 0:
        invalid = ndimage.binary_dilation(invalid, iterations=dilation_px)
    valid = ~invalid
    if np.count_nonzero(valid) < max(64, int(math.ceil(0.05 * valid.size))):
        # With almost no reliable support, retain the exposure-tolerant views
        # rather than correlating a tiny mask remnant.
        return None
    return valid


def _downsample_mask_any(mask: np.ndarray, step: int) -> np.ndarray:
    """Downsample a mask while preserving any invalid site in each source block."""

    boolean = mask.astype(bool, copy=False)
    row_starts = np.arange(0, boolean.shape[0], step)
    column_starts = np.arange(0, boolean.shape[1], step)
    reduced_rows = np.logical_or.reduceat(boolean, row_starts, axis=0)
    return np.logical_or.reduceat(reduced_rows, column_starts, axis=1)


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
    frames: Sequence[np.ndarray],
    config: RegistrationConfig,
    *,
    saturation_masks: Sequence[np.ndarray | None] | None = None,
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
    if saturation_masks is None:
        masks: Sequence[np.ndarray | None] = [None] * len(frames)
    elif len(saturation_masks) != len(frames):
        raise RegistrationError("Saturation-mask count must match frame count")
    else:
        masks = saturation_masks
    for index, mask in enumerate(masks):
        if mask is not None and tuple(mask.shape) != (height, width):
            raise RegistrationError(
                f"Saturation mask {index} shape {mask.shape} does not match {(height, width)}"
            )
    if config.manual_roi_xywh is not None:
        return _validate_manual_roi(config.manual_roi_xywh, (height, width))

    step = max(1, int(math.ceil(max(height, width) / config.preview_max_dim)))
    signals: list[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        preview_rgb = np.asarray(frame[::step, ::step, :])
        preview_mask = None if mask is None else _downsample_mask_any(mask, step)
        valid = _registration_valid_mask(
            preview_rgb,
            preview_mask,
            dilation_px=max(1, int(math.ceil(4 / step))),
        )
        signal = _normalized_signal(_luminance(preview_rgb), valid)
        if np.count_nonzero(signal > 0) >= 16:
            signal = ndimage.gaussian_filter(signal, sigma=1.25)
            signals.append(signal)
    if not signals:
        raise RegistrationError(
            "No compact foreground signal was found for registration; specify --roi X,Y,W,H"
        )
    # Persistent signal is safer than a pixelwise maximum when one exposure
    # contains a large photospheric breakout or bloom.  The second-highest view
    # retains a feature that is measurable in only two frames.
    stack = np.stack(signals, axis=0)
    rank = max(0, stack.shape[0] - 2)
    second_highest = np.partition(stack, rank, axis=0)[rank]
    aggregate = (
        np.float32(0.60) * np.median(stack, axis=0)
        + np.float32(0.40) * second_highest
    ).astype(np.float32)
    if float(np.max(aggregate)) <= 0:
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


def registration_representations(
    rgb_crop: np.ndarray, valid_mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Create three brightness-tolerant views; none changes registration geometry."""

    signal = _normalized_signal(_luminance(rgb_crop), valid_mask)
    if np.count_nonzero(signal) < 16:
        raise RegistrationError("Registration crop contains too little non-black signal")
    smooth = ndimage.gaussian_filter(signal, sigma=1.0)
    gy = ndimage.sobel(smooth, axis=0, mode="nearest")
    gx = ndimage.sobel(smooth, axis=1, mode="nearest")
    gradient = np.hypot(gx, gy)
    highpass = smooth - ndimage.gaussian_filter(smooth, sigma=8.0)
    threshold_support = smooth > 0
    if valid_mask is not None:
        threshold_support &= valid_mask
    positive = smooth[threshold_support]
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


def _aligned_zncc(
    reference: np.ndarray,
    moving: np.ndarray,
    dy: float,
    dx: float,
    reference_valid: np.ndarray | None = None,
    moving_valid: np.ndarray | None = None,
) -> float:
    aligned = ndimage.shift(
        moving,
        shift=(dy, dx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    ys, xs = _overlap_slices(reference.shape, dy, dx)
    reference_inner = reference[ys, xs]
    moving_inner = aligned[ys, xs]
    if reference_valid is not None or moving_valid is not None:
        common = np.ones(reference_inner.shape, dtype=bool)
        if reference_valid is not None:
            common &= reference_valid[ys, xs]
        if moving_valid is not None:
            aligned_invalid = ndimage.shift(
                (~moving_valid).astype(np.float32, copy=False),
                shift=(dy, dx),
                order=1,
                mode="constant",
                cval=1.0,
                prefilter=False,
            )
            # A bilinear footprint is valid only when every contributing source
            # sample is valid; any fractional invalid weight is rejected.
            aligned_valid = aligned_invalid <= np.finfo(np.float32).eps
            common &= aligned_valid[ys, xs]
        a = reference_inner[common].astype(np.float64, copy=False)
        b = moving_inner[common].astype(np.float64, copy=False)
    else:
        a = reference_inner.astype(np.float64, copy=False).ravel()
        b = moving_inner.astype(np.float64, copy=False).ravel()
    minimum_support = max(64, int(math.ceil(0.05 * reference_inner.size)))
    if a.size < minimum_support:
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
    *,
    reference_saturation_mask: np.ndarray | None = None,
    moving_saturation_mask: np.ndarray | None = None,
) -> tuple[RegistrationCandidate, ...]:
    """Return translation candidates sorted by aligned feature correlation."""

    if reference_rgb.shape != moving_rgb.shape:
        raise RegistrationError("Pairwise registration RGB crops differ in size")
    reference_valid = _registration_valid_mask(reference_rgb, reference_saturation_mask)
    moving_valid = _registration_valid_mask(moving_rgb, moving_saturation_mask)
    reference_is_all_valid = reference_valid is not None and bool(np.all(reference_valid))
    moving_is_all_valid = moving_valid is not None and bool(np.all(moving_valid))
    raw_mask_availability_differs = (reference_saturation_mask is None) != (
        moving_saturation_mask is None
    )
    mask_coverage_is_unbalanced = False
    if reference_valid is not None and moving_valid is not None:
        reference_invalid_fraction = 1.0 - float(np.mean(reference_valid))
        moving_invalid_fraction = 1.0 - float(np.mean(moving_valid))
        larger_fraction = max(reference_invalid_fraction, moving_invalid_fraction)
        smaller_fraction = min(reference_invalid_fraction, moving_invalid_fraction)
        mask_coverage_is_unbalanced = (
            larger_fraction > 0.0 and smaller_fraction < 0.25 * larger_fraction
        )
    if (
        reference_valid is None
        or moving_valid is None
        or raw_mask_availability_differs
        or mask_coverage_is_unbalanced
    ):
        # Strongly unbalanced clipping can remove the only unique feature from
        # one frame. Retain the ordinary exposure-tolerant views for both images
        # rather than creating asymmetric evidence.
        reference_valid = None
        moving_valid = None
    else:
        if reference_is_all_valid:
            reference_valid = None
        if moving_is_all_valid:
            moving_valid = None
    reference_views = registration_representations(reference_rgb, reference_valid)
    moving_views = registration_representations(moving_rgb, moving_valid)
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
            score = _aligned_zncc(
                reference,
                moving,
                dy,
                dx,
                reference_valid,
                moving_valid,
            )
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
    return _rank_candidates_by_consensus(candidates, config)


def _candidate_family(candidate: RegistrationCandidate) -> str:
    parts = candidate.method.split("/")
    return parts[1] if len(parts) > 1 else candidate.method


def _candidate_is_plausible(
    candidate: RegistrationCandidate, config: RegistrationConfig
) -> bool:
    return candidate.score >= config.min_score and candidate.psr >= config.min_psr


def _candidate_is_interior(
    candidate: RegistrationCandidate, config: RegistrationConfig
) -> bool:
    return (
        abs(candidate.dx) < config.max_shift_px - 0.5
        and abs(candidate.dy) < config.max_shift_px - 0.5
    )


def _candidate_quality_margin(
    candidate: RegistrationCandidate, config: RegistrationConfig
) -> float:
    score_span = max(1.0 - config.min_score, 1e-9)
    score_margin = float(
        np.clip((candidate.score - config.min_score) / score_span, 0.0, 1.0)
    )
    psr_margin = float(
        np.clip(
            1.0 - config.min_psr / max(candidate.psr, config.min_psr),
            0.0,
            1.0,
        )
    )
    return min(score_margin, psr_margin)


def _complete_family_clusters(
    candidates: Sequence[RegistrationCandidate], radius: float
) -> list[tuple[RegistrationCandidate, ...]]:
    """Enumerate one-candidate-per-family clusters with a bounded diameter."""

    family_count = len({_candidate_family(item) for item in candidates})
    clusters: list[tuple[RegistrationCandidate, ...]] = []
    for size in range(1, family_count + 1):
        for support in combinations(candidates, size):
            if len({_candidate_family(item) for item in support}) != size:
                continue
            if any(
                math.hypot(left.dx - right.dx, left.dy - right.dy) > radius
                for left, right in combinations(support, 2)
            ):
                continue
            clusters.append(support)
    return clusters


def _rank_candidates_by_consensus(
    candidates: Sequence[RegistrationCandidate], config: RegistrationConfig
) -> tuple[RegistrationCandidate, ...]:
    """Put the best independently corroborated translation first.

    ZNCC values from gradient, high-pass, and threshold-bitmap images do not
    share one useful numeric scale.  Each representation therefore gets one
    vote for a shift cluster.  Absolute score/PSR floors decide whether it may
    vote; raw ZNCC is used only as a deterministic final tie-breaker.
    """

    ordered = sorted(
        candidates,
        key=lambda item: (item.score, item.psr, item.coarse_score),
        reverse=True,
    )
    plausible = [item for item in ordered if _candidate_is_plausible(item, config)]
    if not plausible:
        return tuple(ordered)

    radius = config.max_method_disagreement_px
    best_representative: RegistrationCandidate | None = None
    best_key: tuple[float, ...] | None = None
    for support in _complete_family_clusters(plausible, radius):
        pair_distances = [
            math.hypot(left.dx - right.dx, left.dy - right.dy)
            for left, right in combinations(support, 2)
        ]
        representative = max(
            support,
            key=lambda item: (
                item.refinement_error is not None,
                _candidate_is_interior(item, config),
                _candidate_quality_margin(item, config),
                -sum(
                    math.hypot(item.dx - other.dx, item.dy - other.dy)
                    for other in support
                ),
                item.score,
                item.psr,
                item.coarse_score,
            ),
        )
        key = (
            float(len(support)),
            float(sum(item.refinement_error is not None for item in support)),
            float(sum(_candidate_is_interior(item, config) for item in support)),
            float(sum(_candidate_quality_margin(item, config) for item in support)),
            -max(pair_distances, default=0.0),
            -float(np.mean(pair_distances)) if pair_distances else 0.0,
            float(representative.refinement_error is not None),
            float(_candidate_is_interior(representative, config)),
            representative.score,
            representative.psr,
            representative.coarse_score,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_representative = representative

    assert best_representative is not None
    return (
        best_representative,
        *(item for item in ordered if item is not best_representative),
    )


def _method_disagreement(
    frame: FrameAlignment, threshold: float, min_psr: float, min_score: float
) -> bool:
    if not frame.candidates:
        return False
    best = frame.candidates[0]
    plausible = [
        candidate
        for candidate in frame.candidates
        if candidate.score >= min_score and candidate.psr >= min_psr
    ]
    best_family = _candidate_family(best)
    supported_clusters = [
        cluster
        for cluster in _complete_family_clusters(plausible, threshold)
        if len(cluster) >= 2
    ]
    selected_clusters = [
        cluster for cluster in supported_clusters if any(item is best for item in cluster)
    ]
    if not selected_clusters:
        return any(
            _candidate_family(candidate) != best_family for candidate in plausible
        )

    strongest_support = max(len(cluster) for cluster in supported_clusters)
    strongest_selected = [
        cluster for cluster in selected_clusters if len(cluster) == strongest_support
    ]
    if not strongest_selected:
        return True
    strongest_clusters = [
        cluster for cluster in supported_clusters if len(cluster) == strongest_support
    ]
    selected_cluster = max(
        strongest_selected,
        key=lambda cluster: (
            -max(
                (
                    math.hypot(left.dx - right.dx, left.dy - right.dy)
                    for left, right in combinations(cluster, 2)
                ),
                default=0.0,
            ),
            sum(candidate.score for candidate in cluster),
            sum(candidate.psr for candidate in cluster),
        ),
    )
    # Complete-link clusters prevent A~B and B~C from silently becoming one
    # consensus when A and C disagree.  Equally supported, incompatible
    # clusters are ambiguous.  A cluster that keeps a strict majority of the
    # selected candidates merely swaps one family's refinement and is treated
    # as the same solution.
    for alternative_cluster in strongest_clusters:
        shared = sum(
            any(candidate is selected for selected in selected_cluster)
            for candidate in alternative_cluster
        )
        if 2 * shared > strongest_support:
            continue
        if any(
            math.hypot(left.dx - right.dx, left.dy - right.dy) > threshold
            for left in selected_cluster
            for right in alternative_cluster
        ):
            return True
    return False


def _has_independent_support(
    frame: FrameAlignment, threshold: float, min_psr: float, min_score: float
) -> bool:
    if not frame.candidates:
        return False
    best = frame.candidates[0]
    best_family = _candidate_family(best)
    for candidate in frame.candidates:
        if _candidate_family(candidate) == best_family:
            continue
        plausible = candidate.score >= min_score and candidate.psr >= min_psr
        if (
            plausible
            and math.hypot(best.dx - candidate.dx, best.dy - candidate.dy)
            <= threshold
        ):
            return True
    return False


def _has_strong_consensus(
    frame: FrameAlignment, reference_index: int, config: RegistrationConfig
) -> bool:
    if frame.frame_index == reference_index:
        return True
    if not frame.candidates:
        return False
    selected = frame.candidates[0]
    return (
        _candidate_is_plausible(selected, config)
        and selected.refinement_error is not None
        and _candidate_is_interior(selected, config)
        and _has_independent_support(
            frame,
            config.max_method_disagreement_px,
            config.min_psr,
            config.min_score,
        )
        and not _method_disagreement(
            frame,
            config.max_method_disagreement_px,
            config.min_psr,
            config.min_score,
        )
    )


def _motion_statistics(
    frames: Sequence[FrameAlignment],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
) -> tuple[str, tuple[float, float], float, float, str]:
    offsets_yx = np.asarray([(frame.dy, frame.dx) for frame in frames], dtype=np.float64)
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

    max_acceleration = 0.0
    acceleration_label = "adjacent offsets"
    if len(frames) >= 3:
        steps = np.diff(times)
        uniform_steps = np.allclose(steps, steps[0], rtol=0.05, atol=1e-6)
        if uniform_steps:
            accelerations = np.diff(offsets_yx, n=2, axis=0)
            max_acceleration = float(np.max(np.linalg.norm(accelerations, axis=1)))
        else:
            velocities = np.diff(offsets_yx, axis=0) / steps[:, None]
            typical_step = float(np.median(np.abs(steps)))
            velocity_changes = np.diff(velocities, axis=0) * typical_step
            max_acceleration = float(np.max(np.linalg.norm(velocity_changes, axis=1)))
            acceleration_label = "time-normalized adjacent velocities"
    velocity_xy = (float(velocity_yx[1]), float(velocity_yx[0]))
    return basis, velocity_xy, max_residual, max_acceleration, acceleration_label


def physical_sanity_checks(
    frames: Sequence[FrameAlignment],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    config: RegistrationConfig,
) -> tuple[list[str], str, tuple[float, float], float]:
    """Check bounds, confidence, constant motion, and adjacent continuity."""

    issues: list[str] = []
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

    basis, velocity_xy, max_residual, max_acceleration, acceleration_label = (
        _motion_statistics(frames, metadata, reference_index)
    )
    if max_residual > config.max_linear_residual_px:
        issues.append(
            f"offsets depart from constant linear motion by {max_residual:.3f} px "
            f"(limit {config.max_linear_residual_px:g} px)"
        )

    if len(frames) >= 3 and max_acceleration > config.max_acceleration_px:
        issues.append(
            f"{acceleration_label} have a {max_acceleration:.3f} px-equivalent discontinuity "
            f"(limit {config.max_acceleration_px:g} px)"
        )
    return issues, basis, velocity_xy, max_residual


def estimate_group_alignment(
    frames: Sequence[np.ndarray],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    config: RegistrationConfig,
    *,
    saturation_masks: Sequence[np.ndarray | None] | None = None,
) -> AlignmentResult:
    """Align every frame to the selected reference using x/y translation only."""

    if len(frames) != len(metadata):
        raise RegistrationError("Frame and metadata counts differ during registration")
    if not 0 <= reference_index < len(frames):
        raise RegistrationError(f"Reference index {reference_index} is outside the bracket")
    if saturation_masks is None:
        masks: Sequence[np.ndarray | None] = [None] * len(frames)
    elif len(saturation_masks) != len(frames):
        raise RegistrationError("Saturation-mask count must match frame count")
    else:
        masks = saturation_masks
    height, width = frames[0].shape[:2]
    for index, mask in enumerate(masks):
        if mask is not None and tuple(mask.shape) != (height, width):
            raise RegistrationError(
                f"Saturation mask {index} shape {mask.shape} does not match {(height, width)}"
            )
    crop = find_registration_crop(frames, config, saturation_masks=masks)
    rgb_crops = [frame[crop.y0 : crop.y1, crop.x0 : crop.x1, :] for frame in frames]
    mask_crops = [
        None if mask is None else mask[crop.y0 : crop.y1, crop.x0 : crop.x1]
        for mask in masks
    ]
    reference = np.asarray(rgb_crops[reference_index])
    alignments: list[FrameAlignment] = []
    for index, moving in enumerate(rgb_crops):
        if index == reference_index:
            alignments.append(
                FrameAlignment(index, dy=0.0, dx=0.0, method="reference", score=1.0)
            )
            continue
        candidates = estimate_pair_translation(
            reference,
            np.asarray(moving),
            config,
            reference_saturation_mask=mask_crops[reference_index],
            moving_saturation_mask=mask_crops[index],
        )
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

    reported, basis, velocity, max_residual = physical_sanity_checks(
        alignments, metadata, reference_index, config
    )
    _, _, _, max_acceleration, _ = _motion_statistics(
        alignments, metadata, reference_index
    )
    warnings = list(crop.warnings)
    issues: list[str] = []
    consensus_is_strong = all(
        _has_strong_consensus(frame, reference_index, config) for frame in alignments
    )
    advisory_residual_limit = min(
        _MAX_CONSENSUS_LINEAR_RESIDUAL_PX,
        0.5 * config.max_shift_px,
    )
    advisory_acceleration_limit = min(
        _MAX_CONSENSUS_ACCELERATION_PX,
        config.max_shift_px,
    )
    for message in reported:
        is_linear_motion_check = message.startswith(
            "offsets depart from constant linear motion"
        )
        is_acceleration_check = message.startswith(
            ("adjacent offsets", "time-normalized")
        )
        is_bounded_motion_check = (
            is_linear_motion_check and max_residual <= advisory_residual_limit
        ) or (
            is_acceleration_check
            and max_acceleration <= advisory_acceleration_limit
        )
        if consensus_is_strong and is_bounded_motion_check:
            warnings.append(message)
        else:
            issues.append(message)
    return AlignmentResult(
        reference_index=reference_index,
        frames=tuple(alignments),
        crop=crop,
        suspicious=bool(issues),
        issues=tuple(issues),
        warnings=tuple(warnings),
        time_basis=basis,
        linear_velocity_xy=velocity,
        max_linear_residual_px=max_residual,
    )
