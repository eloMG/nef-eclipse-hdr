"""Small, deliberately display-stretched images for alignment inspection only."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .models import AlignmentResult, CropRegion
from .registration import _luminance, _normalized_signal


def _display_gray(rgb: np.ndarray) -> np.ndarray:
    signal = _normalized_signal(_luminance(rgb))
    # Debug-only display encoding.  This never enters the HDR master.
    return np.asarray(np.clip(np.sqrt(signal), 0.0, 1.0) * 255.0, dtype=np.uint8)


def _fit_panel(pixels: np.ndarray, max_side: int = 384) -> Image.Image:
    if pixels.ndim == 2:
        image = Image.fromarray(pixels, mode="L").convert("RGB")
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        image = Image.fromarray(pixels, mode="RGB")
    else:
        raise ValueError(f"Diagnostic panel must be grayscale or RGB, got {pixels.shape}")
    ratio = min(1.0, max_side / max(image.width, image.height))
    if ratio < 1.0:
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image


def _labeled_panel(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height + 28), "black")
    x = (width - image.width) // 2
    y = 28 + (height - image.height) // 2
    panel.paste(image, (x, y))
    ImageDraw.Draw(panel).text((8, 7), label, fill="white")
    return panel


def save_alignment_preview(
    path: Path, frames: Sequence[np.ndarray], alignment: AlignmentResult
) -> None:
    """Save before/after per-frame crops with measured offsets in the labels."""

    crop = alignment.crop
    before_images: list[Image.Image] = []
    after_images: list[Image.Image] = []
    for frame, estimate in zip(frames, alignment.frames):
        rgb = np.asarray(frame[crop.y0 : crop.y1, crop.x0 : crop.x1, :])
        before_images.append(_fit_panel(_display_gray(rgb)))
        aligned_rgb = ndimage.shift(
            rgb,
            shift=(estimate.dy, estimate.dx, 0.0),
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        after_images.append(_fit_panel(_display_gray(aligned_rgb)))

    panel_width = max(image.width for image in before_images + after_images)
    panel_height = max(image.height for image in before_images + after_images)
    labeled_before = [
        _labeled_panel(image, f"before f{i}", panel_width, panel_height)
        for i, image in enumerate(before_images)
    ]
    labeled_after = [
        _labeled_panel(
            image,
            f"after f{i}: dx={alignment.frames[i].dx:+.2f} dy={alignment.frames[i].dy:+.2f}",
            panel_width,
            panel_height,
        )
        for i, image in enumerate(after_images)
    ]
    canvas = Image.new(
        "RGB",
        (panel_width * len(frames), (panel_height + 28) * 2 + 30),
        "#202020",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        "Display-stretched diagnostics only; row 1 unaligned, row 2 translation-aligned",
        fill="white",
    )
    top = 30
    for index, panel in enumerate(labeled_before):
        canvas.paste(panel, (index * panel_width, top))
    top += panel_height + 28
    for index, panel in enumerate(labeled_after):
        canvas.paste(panel, (index * panel_width, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    image = gray.astype(np.float32) / 255.0
    gx = ndimage.sobel(image, axis=1, mode="nearest")
    gy = ndimage.sobel(image, axis=0, mode="nearest")
    edge = np.hypot(gx, gy)
    scale = float(np.percentile(edge, 99.5))
    if scale > 0:
        edge = np.clip(edge / scale, 0.0, 1.0)
    return edge


def _overlay(reference_edge: np.ndarray, moving_edge: np.ndarray) -> Image.Image:
    height, width = reference_edge.shape
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    overlay[..., 0] = reference_edge
    overlay[..., 1] = moving_edge
    overlay[..., 2] = moving_edge
    return _fit_panel(np.asarray(np.clip(overlay, 0.0, 1.0) * 255.0, dtype=np.uint8), 512)


def save_edge_overlay(
    path: Path, frames: Sequence[np.ndarray], alignment: AlignmentResult
) -> None:
    """Save red-reference/cyan-moving edges; correct overlap tends toward white."""

    crop = alignment.crop
    reference_rgb = np.asarray(
        frames[alignment.reference_index][crop.y0 : crop.y1, crop.x0 : crop.x1, :]
    )
    reference_edge = _edge_map(_display_gray(reference_rgb))
    rows: list[tuple[int, Image.Image, Image.Image]] = []
    for frame, estimate in zip(frames, alignment.frames):
        if estimate.frame_index == alignment.reference_index:
            continue
        rgb = np.asarray(frame[crop.y0 : crop.y1, crop.x0 : crop.x1, :])
        before = _edge_map(_display_gray(rgb))
        aligned_rgb = ndimage.shift(
            rgb,
            shift=(estimate.dy, estimate.dx, 0.0),
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        after = _edge_map(_display_gray(aligned_rgb))
        rows.append((estimate.frame_index, _overlay(reference_edge, before), _overlay(reference_edge, after)))

    if not rows:
        return
    panel_width = max(max(before.width, after.width) for _, before, after in rows)
    panel_height = max(max(before.height, after.height) for _, before, after in rows)
    label_height = 28
    canvas = Image.new(
        "RGB",
        (2 * panel_width, len(rows) * (panel_height + label_height) + 34),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        "Reference edges red, moving edges cyan; aligned overlap becomes pale/white",
        fill="white",
    )
    for row, (frame_index, before, after) in enumerate(rows):
        top = 34 + row * (panel_height + label_height)
        draw.text((8, top + 6), f"frame {frame_index} before", fill="white")
        draw.text((panel_width + 8, top + 6), f"frame {frame_index} after", fill="white")
        canvas.paste(before, ((panel_width - before.width) // 2, top + label_height))
        canvas.paste(
            after,
            (panel_width + (panel_width - after.width) // 2, top + label_height),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_hdr_preview(path: Path, hdr: np.ndarray, crop: CropRegion) -> None:
    """Save a display-stretched crop for inspection without altering the master."""

    crop_height, crop_width = crop.shape
    step = max(1, (max(crop_height, crop_width) + 1599) // 1600)
    rgb = np.asarray(
        hdr[crop.y0 : crop.y1 : step, crop.x0 : crop.x1 : step, :],
        dtype=np.float32,
    )
    finite = np.isfinite(rgb)
    if not np.any(finite):
        raise ValueError("HDR preview crop contains no finite samples")
    background = float(np.percentile(rgb[finite], 1.0))
    positive = np.maximum(rgb - background, 0.0)
    positive_values = positive[positive > 0]
    scale = float(np.percentile(positive_values, 99.95)) if positive_values.size else 1.0
    scale = max(scale, 1e-8)
    display = np.arcsinh(positive / np.float32(scale * 0.02))
    display_top = float(np.arcsinh(1.0 / 0.02))
    display = np.clip(display / display_top, 0.0, 1.0)
    display = np.power(display, np.float32(1.0 / 2.2))
    pixels = np.rint(display * 255.0).astype(np.uint8)
    image = _fit_panel(pixels, max_side=1600)
    label_height = 32
    canvas = Image.new("RGB", (image.width, image.height + label_height), "black")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text(
        (8, 8),
        "Display-stretched HDR diagnostic only; master remains linear and untone-mapped",
        fill="white",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
