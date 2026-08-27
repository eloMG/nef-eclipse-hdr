"""Read-only Nikon NEF metadata access and deterministic linear development."""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy import ndimage

from .errors import DependencyError, MetadataError
from .models import ExposureMetadata
from .output import write_linear_intermediate

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDevelopmentSettings:
    """Bracket-wide choices that must not vary with exposure brightness."""

    white_balance: tuple[float, float, float, float]
    user_flip: int
    user_saturation: int
    output_color: str = "linear sRGB"

    def to_dict(self) -> dict[str, Any]:
        return {
            "white_balance": list(self.white_balance),
            "libraw_flip_code": self.user_flip,
            "user_saturation": self.user_saturation,
            "output_color": self.output_color,
            "gamma": [1.0, 1.0],
            "no_auto_bright": True,
            "adjust_maximum_thr": 0.0,
            "output_bps": 16,
            "highlight_mode": "Ignore (no highlight reconstruction)",
            "demosaic": "AHD",
            "denoising": "disabled",
            "sharpening": "not performed by rawpy",
        }


@dataclass(frozen=True)
class DevelopedFrame:
    """Disk-backed developed RGB plus a conservative sensor-clipping mask."""

    metadata: ExposureMetadata
    rgb_npy: Path
    saturation_mask_npy: Optional[Path]
    shape: tuple[int, int, int]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def open_rgb(self) -> np.ndarray:
        return np.load(self.rgb_npy, mmap_mode="r")

    def open_saturation_mask(self) -> Optional[np.ndarray]:
        if self.saturation_mask_npy is None:
            return None
        return np.load(self.saturation_mask_npy, mmap_mode="r")


def import_rawpy() -> Any:
    """Import rawpy lazily so ``--help`` and array-only tests work without it."""

    try:
        return importlib.import_module("rawpy")
    except ImportError as exc:
        raise DependencyError(
            "rawpy is required to read NEF files. Install the Windows dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _first_attribute(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        # LibRaw's missing timestamp is zero; rawpy exposes that as the local
        # Unix epoch rather than None.  No eclipse capture here predates 1980.
        return value if value.year >= 1980 else None
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return None
    return None


def metadata_from_open_raw(raw: Any, path: Path) -> ExposureMetadata:
    """Read LibRaw's non-image metadata, accepting old and new field spellings."""

    other = getattr(raw, "other", None)
    if other is None:
        other = getattr(raw, "metadata", None)
    if other is None:
        raise MetadataError(f"LibRaw did not expose exposure metadata for {path.name}")
    return ExposureMetadata(
        path=path,
        shutter_s=_number(_first_attribute(other, "shutter_speed", "shutter")),
        iso=_number(_first_attribute(other, "iso_speed", "iso")),
        aperture=_number(_first_attribute(other, "aperture")),
        timestamp=_timestamp(_first_attribute(other, "timestamp")),
        focal_length_mm=_number(_first_attribute(other, "focal_length", "focal_len")),
    )


def read_metadata(path: Path, rawpy_module: Any = None) -> ExposureMetadata:
    """Open an NEF read-only and extract only its metadata."""

    rawpy_module = rawpy_module or import_rawpy()
    try:
        with rawpy_module.imread(str(path)) as raw:
            return metadata_from_open_raw(raw, path)
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError(f"Could not read metadata from {path.name}: {exc}") from exc


def reference_settings(path: Path, rawpy_module: Any = None) -> RawDevelopmentSettings:
    """Lock WB, orientation, and saturation to the central reference NEF."""

    rawpy_module = rawpy_module or import_rawpy()
    try:
        with rawpy_module.imread(str(path)) as raw:
            camera_wb = np.asarray(getattr(raw, "camera_whitebalance", ()), dtype=np.float64)
            if camera_wb.size != 4 or np.any(~np.isfinite(camera_wb)) or np.any(camera_wb <= 0):
                camera_wb = np.asarray(getattr(raw, "daylight_whitebalance", ()), dtype=np.float64)
                LOGGER.warning("Camera WB unavailable in %s; using its daylight WB", path.name)
            if camera_wb.size != 4 or np.any(~np.isfinite(camera_wb)) or np.any(camera_wb <= 0):
                raise MetadataError(f"No valid four-channel white balance is available in {path.name}")

            sizes = getattr(raw, "sizes", None)
            flip = int(getattr(sizes, "flip", 0))
            saturation = int(getattr(raw, "white_level", 0) or 0)
            if saturation <= 0:
                raise MetadataError(f"No valid sensor white level is available in {path.name}")
            return RawDevelopmentSettings(
                white_balance=tuple(float(value) for value in camera_wb),  # type: ignore[arg-type]
                user_flip=flip,
                user_saturation=saturation,
            )
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError(f"Could not read reference development settings from {path.name}: {exc}") from exc


def build_rawpy_params(rawpy_module: Any, settings: RawDevelopmentSettings) -> Any:
    """Build a radiometrically stable LibRaw conversion configuration.

    ``adjust_maximum_thr=0`` prevents LibRaw from choosing a different scale from
    each frame's histogram.  Black-to-white scaling and the fixed reference WB
    remain linear operations; ``no_auto_scale`` therefore intentionally remains
    false.  Highlight ``Ignore`` retains channel headroom but performs no invented
    reconstruction (unlike Blend/Reconstruct).
    """

    return rawpy_module.Params(
        demosaic_algorithm=rawpy_module.DemosaicAlgorithm.AHD,
        half_size=False,
        four_color_rgb=False,
        dcb_iterations=0,
        dcb_enhance=False,
        fbdd_noise_reduction=rawpy_module.FBDDNoiseReductionMode.Off,
        noise_thr=None,
        median_filter_passes=0,
        use_camera_wb=False,
        use_auto_wb=False,
        # rawpy 0.27's Cython boundary requires an actual list, not merely a
        # generic four-value sequence/tuple.
        user_wb=list(settings.white_balance),
        output_color=rawpy_module.ColorSpace.sRGB,
        output_bps=16,
        user_flip=settings.user_flip,
        user_sat=settings.user_saturation,
        no_auto_bright=True,
        no_auto_scale=False,
        auto_bright_thr=None,
        adjust_maximum_thr=0.0,
        bright=1.0,
        highlight_mode=rawpy_module.HighlightMode.Ignore,
        gamma=(1.0, 1.0),
        exp_shift=None,
        exp_preserve_highlights=0.0,
    )


def _camera_white_levels(raw: Any) -> np.ndarray:
    per_channel = getattr(raw, "camera_white_level_per_channel", None)
    if per_channel is not None:
        levels = np.asarray(per_channel, dtype=np.float64)
        if levels.size >= 4 and np.all(np.isfinite(levels[:4])) and np.all(levels[:4] > 0):
            return levels[:4]
    white = float(getattr(raw, "white_level", 0) or 0)
    if white <= 0:
        return np.asarray([], dtype=np.float64)
    return np.repeat(white, 4)


def _orient_mask(mask: np.ndarray, flip: int) -> np.ndarray:
    # LibRaw/rawpy flip codes: 3=180 degrees, 5=90 CCW, 6=90 CW.
    if flip == 3:
        return np.rot90(mask, 2)
    if flip == 5:
        return np.rot90(mask, 1)
    if flip == 6:
        return np.rot90(mask, -1)
    return mask


def _raw_saturation_mask(raw: Any, flip: int) -> Optional[np.ndarray]:
    """Map clipped CFA sites to a dilated visible-image mask before demosaicing."""

    try:
        raw_values = getattr(raw, "raw_image_visible", None)
        raw_colors = getattr(raw, "raw_colors_visible", None)
        white_levels = _camera_white_levels(raw)
    except Exception:
        return None
    if raw_values is None or raw_colors is None or white_levels.size < 4:
        return None

    values = np.asarray(raw_values)
    colors = np.asarray(raw_colors)
    if values.shape != colors.shape or colors.dtype.kind not in "ui":
        return None
    # A two-pixel dilation conservatively covers the footprint of AHD demosaicing
    # and prevents bilinear subpixel alignment from hiding a clipped CFA neighbor.
    mask = np.zeros(values.shape, dtype=bool)
    color_match = np.empty(values.shape, dtype=bool)
    value_clipped = np.empty(values.shape, dtype=bool)
    for channel in range(4):
        np.equal(colors, channel, out=color_match)
        np.greater_equal(values, max(0, int(white_levels[channel]) - 1), out=value_clipped)
        np.logical_and(color_match, value_clipped, out=color_match)
        np.logical_or(mask, color_match, out=mask)
    mask = ndimage.binary_dilation(mask, iterations=2)
    return np.ascontiguousarray(_orient_mask(mask, flip), dtype=np.uint8)


def develop_group(
    paths: Sequence[Path],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    work_dir: Path,
    *,
    keep_intermediates_dir: Optional[Path] = None,
    rawpy_module: Any = None,
) -> tuple[list[DevelopedFrame], RawDevelopmentSettings]:
    """Develop a bracket to disk-backed uint16 arrays without altering the NEFs."""

    if len(paths) != len(metadata):
        raise MetadataError("Path and metadata counts differ while developing a bracket")
    rawpy_module = rawpy_module or import_rawpy()
    settings = reference_settings(paths[reference_index], rawpy_module)
    params = build_rawpy_params(rawpy_module, settings)
    work_dir.mkdir(parents=True, exist_ok=True)
    if keep_intermediates_dir is not None:
        keep_intermediates_dir.mkdir(parents=True, exist_ok=True)

    developed: list[DevelopedFrame] = []
    expected_shape: Optional[tuple[int, int, int]] = None
    for index, (path, item_metadata) in enumerate(zip(paths, metadata)):
        warnings: list[str] = []
        LOGGER.info("Developing frame %d/%d: %s", index + 1, len(paths), path.name)
        try:
            with rawpy_module.imread(str(path)) as raw:
                flip = int(getattr(getattr(raw, "sizes", None), "flip", 0))
                if flip != settings.user_flip:
                    warnings.append(
                        f"LibRaw orientation code {flip} differs from reference "
                        f"code {settings.user_flip}; reference orientation was forced."
                    )
                white = int(getattr(raw, "white_level", 0) or 0)
                if white != settings.user_saturation:
                    warnings.append(
                        f"Sensor white level {white} differs from reference "
                        f"{settings.user_saturation}; the reference scale was forced."
                    )
                saturation_mask = _raw_saturation_mask(raw, settings.user_flip)
                rgb = raw.postprocess(params=params)
        except Exception as exc:
            raise MetadataError(f"Failed to develop {path.name}: {exc}") from exc

        if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise MetadataError(
                f"LibRaw returned {rgb.shape} {rgb.dtype} for {path.name}; expected HxWx3 uint16"
            )
        shape = tuple(int(value) for value in rgb.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise MetadataError(
                f"Developed dimensions differ inside the bracket: {shape} vs {expected_shape}"
            )

        mask_path: Optional[Path] = None
        if saturation_mask is not None:
            if saturation_mask.shape != rgb.shape[:2]:
                warnings.append(
                    "The CFA saturation mask dimensions did not match developed RGB; "
                    "postprocessed RGB clipping detection will be used instead."
                )
                saturation_mask = None
            else:
                mask_path = work_dir / f"frame_{index:02d}_saturated.npy"
                mask_mm = np.lib.format.open_memmap(
                    mask_path, mode="w+", dtype=np.uint8, shape=saturation_mask.shape
                )
                mask_mm[...] = saturation_mask
                mask_mm.flush()
                del mask_mm

        rgb_path = work_dir / f"frame_{index:02d}_linear.npy"
        rgb_mm = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint16, shape=rgb.shape)
        rgb_mm[...] = rgb
        rgb_mm.flush()
        del rgb_mm
        if keep_intermediates_dir is not None:
            write_linear_intermediate(keep_intermediates_dir / f"{path.stem}_linear.tif", rgb)
        del rgb

        developed.append(
            DevelopedFrame(
                metadata=item_metadata,
                rgb_npy=rgb_path,
                saturation_mask_npy=mask_path,
                shape=shape,  # type: ignore[arg-type]
                warnings=tuple(warnings),
            )
        )
    return developed, settings
