from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from eclipsehdr.errors import MetadataError
from eclipsehdr.metadata import exposure_factors, timestamp_gaps, usable_time_axis
from eclipsehdr.models import ExposureMetadata


def meta(
    name: str,
    shutter: float = 0.01,
    iso: float = 100.0,
    aperture: float | None = 8.0,
    timestamp: datetime | None = None,
) -> ExposureMetadata:
    return ExposureMetadata(Path(name), shutter, iso, aperture, timestamp)


def test_exposure_factor_uses_shutter_iso_and_aperture() -> None:
    items = [
        meta("a.nef", shutter=1 / 1000, iso=100, aperture=8),
        meta("b.nef", shutter=1 / 500, iso=200, aperture=11),
        meta("c.nef", shutter=1 / 250, iso=100, aperture=8),
    ]
    factors, warnings = exposure_factors(items, 1)
    expected = np.array([1 / 1000 * 100 / 64, 1 / 500 * 200 / 121, 1 / 250 * 100 / 64])
    np.testing.assert_allclose(factors, expected)
    assert not warnings


def test_all_missing_apertures_cancel_with_warning() -> None:
    factors, warnings = exposure_factors(
        [meta("a.nef", shutter=0.01, aperture=None), meta("b.nef", shutter=0.02, aperture=None)],
        0,
    )
    np.testing.assert_allclose(factors / factors[0], [1, 2])
    assert "treated as constant" in warnings[0]


def test_partially_missing_varying_aperture_is_rejected() -> None:
    with pytest.raises(MetadataError, match="Aperture varies"):
        exposure_factors(
            [
                meta("a.nef", aperture=5.6),
                meta("b.nef", aperture=None),
                meta("c.nef", aperture=8.0),
            ],
            1,
        )


def test_timestamp_checks_and_subsecond_axis() -> None:
    start = datetime(2024, 4, 8, 18, 0, 0)
    items = [meta(f"f{i}.nef", timestamp=start + timedelta(seconds=0.3 * i)) for i in range(5)]
    assert timestamp_gaps(items, 0.5) == []
    axis, basis = usable_time_axis(items, 2)
    np.testing.assert_allclose(axis, [-0.6, -0.3, 0.0, 0.3, 0.6])
    assert basis == "capture_timestamp"


def test_duplicate_second_timestamps_fall_back_to_indices() -> None:
    stamp = datetime(2024, 4, 8, 18, 0, 0)
    items = [meta(f"f{i}.nef", timestamp=stamp) for i in range(5)]
    axis, basis = usable_time_axis(items, 2)
    np.testing.assert_array_equal(axis, [-2, -1, 0, 1, 2])
    assert basis == "frame_index"

