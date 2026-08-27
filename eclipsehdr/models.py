"""Small data objects shared by the eclipse HDR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExposureMetadata:
    """Exposure fields read directly from a RAW file."""

    path: Path
    shutter_s: Optional[float]
    iso: Optional[float]
    aperture: Optional[float]
    timestamp: Optional[datetime]
    focal_length_mm: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "path": str(self.path.resolve()),
            "shutter_s": self.shutter_s,
            "iso": self.iso,
            "aperture_f": self.aperture,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "focal_length_mm": self.focal_length_mm,
        }


@dataclass(frozen=True)
class RegistrationConfig:
    """Controls only translation estimation; no other transform exists here."""

    max_shift_px: float = 20.0
    crop_size_px: int = 2048
    preview_max_dim: int = 1024
    upsample_factor: int = 20
    max_linear_residual_px: float = 1.0
    max_acceleration_px: float = 1.5
    min_score: float = 0.05
    min_psr: float = 1.25
    max_method_disagreement_px: float = 1.5
    manual_roi_xywh: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class MergeConfig:
    """Linear-domain exposure weighting and interpolation settings."""

    black_start: float = 0.001
    black_full_weight: float = 0.02
    highlight_falloff: float = 0.85
    saturation_cutoff: float = 0.98
    chunk_rows: int = 128
    interpolation_order: int = 1


@dataclass(frozen=True)
class CropRegion:
    """A shared, unshifted image crop in reference-frame coordinates."""

    y0: int
    y1: int
    x0: int
    x1: int
    method: str = "automatic"
    warnings: Tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x0,
            "y": self.y0,
            "width": self.x1 - self.x0,
            "height": self.y1 - self.y0,
            "method": self.method,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RegistrationCandidate:
    """One translation-only estimate from one exposure-invariant representation."""

    dy: float
    dx: float
    method: str
    score: float
    coarse_score: float
    psr: float
    refinement_error: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameAlignment:
    """The shift applied to a frame to place it in reference coordinates."""

    frame_index: int
    dy: float
    dx: float
    method: str
    score: float
    candidates: Tuple[RegistrationCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "dx": self.dx,
            "dy": self.dy,
            "method": self.method,
            "score": self.score,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Offsets plus diagnostics and group-level physical checks."""

    reference_index: int
    frames: Tuple[FrameAlignment, ...]
    crop: CropRegion
    suspicious: bool
    issues: Tuple[str, ...] = ()
    time_basis: str = "frame_index"
    linear_velocity_xy: Tuple[float, float] = (0.0, 0.0)
    max_linear_residual_px: float = 0.0

    @property
    def offsets_yx(self) -> tuple[tuple[float, float], ...]:
        return tuple((frame.dy, frame.dx) for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "offset_convention": (
                "dx/dy is the translation applied to this frame to align it to "
                "the reference; +dx moves content right, +dy moves content down"
            ),
            "crop": self.crop.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "sanity": {
                "suspicious": self.suspicious,
                "issues": list(self.issues),
                "time_basis": self.time_basis,
                "linear_velocity_dx": self.linear_velocity_xy[0],
                "linear_velocity_dy": self.linear_velocity_xy[1],
                "max_linear_residual_px": self.max_linear_residual_px,
            },
        }


@dataclass(frozen=True)
class MergeStatistics:
    """Counts that expose where the five RAWs contained no ideal measurement."""

    pixel_count: int
    weighted_pixel_count: int
    fallback_pixel_count: int
    all_saturated_pixel_count: int
    all_dark_pixel_count: int
    min_value: float
    max_value: float
    exposure_factors: Tuple[float, ...] = field(default_factory=tuple)
    relative_exposures: Tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
