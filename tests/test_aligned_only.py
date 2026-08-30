import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage

import eclipsehdr.pipeline as pipeline_module
from eclipsehdr.aligned import _publish_outputs
from eclipsehdr.models import (
    AlignmentResult,
    CropRegion,
    ExposureMetadata,
    FrameAlignment,
    MergeConfig,
)
from eclipsehdr.pipeline import PipelineConfig, process_bracket
from eclipsehdr.output import write_aligned_linear_tiff
from eclipsehdr.rawio import DevelopedFrame, RawDevelopmentSettings


def test_aligned_only_writes_indexed_linear_tiffs_and_never_merges(
    tmp_path: Path, monkeypatch
) -> None:
    height, width = 10, 12
    y, x = np.mgrid[:height, :width]
    frames: list[np.ndarray] = []
    for index in range(5):
        signal = (index + 1) * 7000 + y * 200 + x * 20
        frames.append(
            np.stack([signal, signal + 100, signal + 200], axis=2).astype(np.uint16)
        )

    offsets_yx = (
        (0.0, 2.0),
        (0.5, -1.0),
        (0.0, 0.0),
        (-1.25, 0.75),
        (1.0, -2.0),
    )
    paths = [tmp_path / f"DSC_{index:04d}.NEF" for index in range(5)]
    start = datetime(2024, 4, 8, 18, 0, 0)
    # Exposure fields are deliberately absent: translating developed pixels does
    # not require the radiometric metadata needed by the HDR merge.
    metadata = [
        ExposureMetadata(path, None, None, None, start + timedelta(seconds=0.3 * index))
        for index, path in enumerate(paths)
    ]

    def fake_develop_group(
        requested_paths,
        requested_metadata,
        reference_index,
        work_dir,
        **kwargs,
    ):
        assert list(requested_paths) == paths
        assert list(requested_metadata) == metadata
        assert reference_index == 2
        work_dir.mkdir(parents=True, exist_ok=True)
        developed = []
        for index, rgb in enumerate(frames):
            rgb_path = work_dir / f"frame_{index:02d}.npy"
            np.save(rgb_path, rgb)
            developed.append(DevelopedFrame(metadata[index], rgb_path, None, rgb.shape))
        return developed, RawDevelopmentSettings((2.0, 1.0, 1.5, 1.0), 0, 16383)

    alignment = AlignmentResult(
        reference_index=2,
        frames=tuple(
            FrameAlignment(
                frame_index=index,
                dy=dy,
                dx=dx,
                method="reference" if index == 2 else "test",
                score=1.0,
            )
            for index, (dy, dx) in enumerate(offsets_yx)
        ),
        crop=CropRegion(0, height, 0, width, method="test"),
        suspicious=False,
    )

    def unexpected_hdr_operation(*args, **kwargs):
        raise AssertionError("aligned-only mode must not run HDR exposure or merge operations")

    monkeypatch.setattr(pipeline_module, "develop_group", fake_develop_group)
    monkeypatch.setattr(pipeline_module, "estimate_group_alignment", lambda *args: alignment)
    monkeypatch.setattr(pipeline_module, "exposure_factors", unexpected_hdr_operation)
    monkeypatch.setattr(pipeline_module, "merge_linear_hdr", unexpected_hdr_operation)

    output_dir = tmp_path / "output"
    result = process_bracket(
        0,
        paths,
        metadata,
        PipelineConfig(
            input_dir=tmp_path,
            output_dir=output_dir,
            aligned_only=True,
            merge=MergeConfig(chunk_rows=3),
        ),
        rawpy_module=object(),
    )

    aligned_dir = output_dir / "aligned" / "DSC_0002"
    expected_paths = tuple(
        aligned_dir / f"{index:02d}_DSC_{index:04d}_aligned_linear.tif"
        for index in range(5)
    )
    assert result.status == "complete"
    assert result.output_path is None
    assert result.output_paths == expected_paths
    assert result.sidecar_path == aligned_dir / "DSC_0002_aligned.json"
    assert not (output_dir / "DSC_0002_HDR.tif").exists()

    for source, (dy, dx), aligned_path in zip(frames, offsets_yx, expected_paths):
        actual = tifffile.imread(aligned_path)
        expected = np.rint(
            ndimage.shift(
                source.astype(np.float32),
                shift=(dy, dx, 0.0),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        ).astype(np.uint16)
        assert actual.dtype == np.uint16
        assert actual.shape == source.shape
        np.testing.assert_allclose(actual, expected, atol=1)

    np.testing.assert_array_equal(tifffile.imread(expected_paths[2]), frames[2])
    assert np.all(tifffile.imread(expected_paths[0])[:, :2, :] == 0)

    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "complete"
    assert sidecar["mode"] == "aligned_only"
    assert [item["path"] for item in sidecar["output"]["files"]] == [
        str(path.resolve()) for path in expected_paths
    ]
    assert [item["source_filename"] for item in sidecar["output"]["files"]] == [
        path.name for path in paths
    ]
    assert sidecar["output"]["files"][0]["dx"] == 2.0
    assert sidecar["output"]["files"][0]["dy"] == 0.0
    assert "merge" not in sidecar


def test_aligned_tiff_writer_accepts_unicode_output_name(tmp_path: Path) -> None:
    rgb = np.arange(6 * 7 * 3, dtype=np.uint16).reshape(6, 7, 3)
    path = tmp_path / "sol_ø_aligned_linear.tif"
    write_aligned_linear_tiff(path, rgb, dy=-0.25, dx=1.5)
    np.testing.assert_array_equal(tifffile.imread(path), rgb)


def test_aligned_output_publish_rolls_back_overwrite_failure(
    tmp_path: Path, monkeypatch
) -> None:
    pairs: list[tuple[Path, Path]] = []
    for index in range(3):
        partial = tmp_path / f".frame_{index}.partial.tif"
        final = tmp_path / f"frame_{index}.tif"
        partial.write_bytes(f"new-{index}".encode("ascii"))
        final.write_bytes(f"old-{index}".encode("ascii"))
        pairs.append((partial, final))

    original_rename = Path.rename
    partial_publish_count = 0

    def fail_second_partial_publish(self: Path, target: Path):
        nonlocal partial_publish_count
        if self.name.endswith(".partial.tif"):
            partial_publish_count += 1
            if partial_publish_count == 2:
                raise OSError("injected publication failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_second_partial_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        _publish_outputs(pairs, overwrite=True)

    for index, (_, final) in enumerate(pairs):
        assert final.read_bytes() == f"old-{index}".encode("ascii")
    assert not list(tmp_path.glob("*.backup"))
