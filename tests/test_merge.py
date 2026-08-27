from pathlib import Path

import numpy as np
import tifffile

from eclipsehdr.merge import merge_linear_hdr
from eclipsehdr.models import MergeConfig
from eclipsehdr.output import write_float_tiff


def _developed_bracket(radiance: np.ndarray, relative_exposures: np.ndarray):
    frames = []
    masks = []
    for exposure in relative_exposures:
        measured = np.clip(radiance * exposure, 0.0, 1.0)
        frames.append(np.rint(measured * 65535.0).astype(np.uint16))
        masks.append(np.max(measured, axis=2) >= 1.0)
    return frames, masks


def test_linear_merge_recovers_scene_radiance_across_exposures() -> None:
    x = np.geomspace(0.003, 8.0, 96, dtype=np.float32)
    base = np.broadcast_to(x[None, :, None], (24, 96, 3)).copy()
    base[..., 1] *= 0.7
    base[..., 2] *= 0.4
    relative = np.asarray([1 / 16, 1 / 4, 1.0, 4.0, 16.0], dtype=np.float64)
    frames, masks = _developed_bracket(base, relative)
    hdr, stats = merge_linear_hdr(
        frames,
        [(0.0, 0.0)] * 5,
        relative,
        2,
        MergeConfig(chunk_rows=7),
        saturation_masks=masks,
    )
    # Pixels with at least one comfortable measurement recover quantized truth.
    comfortable = (base[..., 0] >= 0.01) & (base[..., 0] <= 8.0)
    relative_error = np.abs(hdr[..., 0] - base[..., 0]) / base[..., 0]
    assert float(np.percentile(relative_error[comfortable], 99)) < 0.003
    assert stats.max_value > 1.0
    assert stats.weighted_pixel_count > 0


def test_all_saturated_and_all_dark_fallbacks_are_counted() -> None:
    radiance = np.zeros((2, 5, 3), dtype=np.float32)
    radiance[0, 0] = 100.0
    radiance[0, 1] = 0.2
    radiance[0, 2] = 0.0
    radiance[0, 3] = 0.2
    radiance[0, 4] = 0.2
    radiance[1] = radiance[0]
    relative = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    frames, masks = _developed_bracket(radiance, relative)
    hdr, stats = merge_linear_hdr(
        frames,
        [(0.0, 0.0)] * 5,
        relative,
        2,
        MergeConfig(chunk_rows=1),
        saturation_masks=masks,
    )
    # A fully clipped pixel is an explicit lower bound from the least exposure.
    np.testing.assert_allclose(hdr[:, 0], 4.0, atol=1e-5)
    np.testing.assert_allclose(hdr[:, 2], 0.0)
    assert stats.all_saturated_pixel_count >= 2  # mask dilation also protects neighbors
    assert stats.all_dark_pixel_count == 2
    assert stats.fallback_pixel_count >= 4


def test_subpixel_translation_has_constant_border_not_wraparound() -> None:
    reference = np.zeros((12, 12, 3), dtype=np.uint16)
    moving = np.zeros_like(reference)
    moving[:, -1, :] = 65535
    # Moving is shifted right into reference coordinates; its right edge must fall out,
    # never wrap around to column zero.
    hdr, _ = merge_linear_hdr(
        [reference, moving],
        [(0.0, 0.0), (0.0, 2.0)],
        [1.0, 1.0],
        0,
        MergeConfig(black_start=0.0, black_full_weight=0.01, chunk_rows=4),
    )
    assert np.all(hdr[:, 0, :] == 0)


def test_source_clipping_is_rejected_before_bilinear_interpolation_hides_it() -> None:
    reference = np.full((8, 8, 3), np.rint(0.2 * 65535), dtype=np.uint16)
    moving = np.zeros_like(reference)
    moving[:, 2, :] = 65535
    moving[:, 3, :] = np.rint(0.90 * 65535)
    hdr, _ = merge_linear_hdr(
        [reference, moving],
        [(0.0, 0.0), (0.0, 0.5)],
        [1.0, 1.0],
        0,
        MergeConfig(chunk_rows=3),
        saturation_masks=[None, None],
    )
    # x=3 samples the moving source halfway between 1.0 and 0.9. The source-space
    # clip mask must reject it, leaving only the clean 0.2 reference measurement.
    np.testing.assert_allclose(hdr[:, 3, :], 0.2, atol=2 / 65535)


def test_float_tiff_roundtrip_preserves_unbounded_values(tmp_path: Path) -> None:
    values = np.asarray([0.0, 0.18, 1.0, 2.0, 8.0], dtype=np.float32)
    rgb = np.broadcast_to(values[None, :, None], (3, 5, 3)).copy()
    path = tmp_path / "master.tif"
    write_float_tiff(path, rgb)
    loaded = tifffile.imread(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, rgb)
    assert float(loaded.max()) == 8.0
