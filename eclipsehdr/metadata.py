"""Exposure metadata validation and relative radiometric normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from .errors import MetadataError
from .models import ExposureMetadata


def exposure_factors(
    metadata: Sequence[ExposureMetadata], reference_index: int
) -> tuple[np.ndarray, list[str]]:
    """Return ``shutter * ISO / f_number**2`` and explicit metadata warnings.

    The constant ISO/100 is omitted because only ratios are needed.  When every
    aperture is missing, the aperture term cancels if (as expected for a burst)
    it stayed fixed.  A partially missing but otherwise constant aperture is
    filled from the known values and reported.
    """

    if not metadata:
        raise MetadataError("Cannot compute exposure factors for an empty bracket")
    if not 0 <= reference_index < len(metadata):
        raise MetadataError(f"Reference index {reference_index} is outside the bracket")

    warnings: list[str] = []
    shutters = np.asarray([item.shutter_s or 0.0 for item in metadata], dtype=np.float64)
    isos = np.asarray([item.iso or 0.0 for item in metadata], dtype=np.float64)
    if np.any(~np.isfinite(shutters)) or np.any(shutters <= 0):
        bad = [metadata[i].path.name for i in np.flatnonzero((~np.isfinite(shutters)) | (shutters <= 0))]
        raise MetadataError(f"Missing or invalid shutter speed in: {', '.join(bad)}")
    if np.any(~np.isfinite(isos)) or np.any(isos <= 0):
        bad = [metadata[i].path.name for i in np.flatnonzero((~np.isfinite(isos)) | (isos <= 0))]
        raise MetadataError(f"Missing or invalid ISO in: {', '.join(bad)}")

    aperture_values = np.asarray(
        [item.aperture if item.aperture is not None else np.nan for item in metadata],
        dtype=np.float64,
    )
    known = np.isfinite(aperture_values) & (aperture_values > 0)
    if not np.any(known):
        aperture_values[:] = 1.0
        warnings.append(
            "Aperture metadata is absent in all frames; treated as constant, so its term cancels."
        )
    else:
        known_values = aperture_values[known]
        relative_spread = float(np.ptp(known_values) / np.median(known_values))
        if relative_spread > 0.02 and not np.all(known):
            raise MetadataError(
                "Aperture varies among known frames while other frames lack aperture metadata; "
                "relative exposure cannot be determined safely"
            )
        if not np.all(known):
            fill = float(np.median(known_values))
            aperture_values[~known] = fill
            missing = [metadata[i].path.name for i in np.flatnonzero(~known)]
            warnings.append(
                f"Aperture missing in {', '.join(missing)}; used the bracket median f/{fill:g}."
            )

    factors = shutters * isos / np.square(aperture_values)
    if np.any(~np.isfinite(factors)) or np.any(factors <= 0):
        raise MetadataError("Exposure normalization produced a non-positive or non-finite factor")
    return factors, warnings


def timestamp_gaps(
    metadata: Sequence[ExposureMetadata], max_gap_seconds: float
) -> list[str]:
    """Describe absent, reversed, or unexpectedly large capture-time gaps."""

    warnings: list[str] = []
    stamps = [item.timestamp for item in metadata]
    if any(stamp is None for stamp in stamps):
        warnings.append("One or more capture timestamps are missing; burst timing was not fully checked.")
        return warnings

    assert all(isinstance(stamp, datetime) for stamp in stamps)
    for index in range(len(stamps) - 1):
        assert stamps[index] is not None and stamps[index + 1] is not None
        gap = (stamps[index + 1] - stamps[index]).total_seconds()
        if gap < 0:
            warnings.append(
                f"Capture timestamps run backwards between frames {index} and {index + 1} ({gap:+.3f} s)."
            )
        elif gap > max_gap_seconds:
            warnings.append(
                f"Gap between frames {index} and {index + 1} is {gap:.3f} s, "
                f"above the configured {max_gap_seconds:.3f} s burst limit."
            )
    return warnings


def usable_time_axis(
    metadata: Sequence[ExposureMetadata], reference_index: int
) -> tuple[np.ndarray, str]:
    """Use subsecond timestamps when informative, otherwise stable frame indices."""

    stamps = [item.timestamp for item in metadata]
    if all(stamp is not None for stamp in stamps):
        reference = stamps[reference_index]
        assert reference is not None
        seconds = np.asarray([(stamp - reference).total_seconds() for stamp in stamps], dtype=np.float64)  # type: ignore[operator]
        if np.all(np.diff(seconds) > 0) and np.unique(seconds).size == len(seconds):
            return seconds, "capture_timestamp"
    return np.arange(len(metadata), dtype=np.float64) - float(reference_index), "frame_index"

