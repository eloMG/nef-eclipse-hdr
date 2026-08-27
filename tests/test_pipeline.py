from datetime import datetime
from pathlib import Path

from eclipsehdr.models import ExposureMetadata
import numpy as np

from eclipsehdr.pipeline import (
    PipelineConfig,
    _close_memmaps,
    _selected_indices,
    group_brackets,
    process_bracket,
)


def _items(count: int):
    paths = [Path(f"DSC_{index:04d}.NEF") for index in range(count)]
    metadata = [
        ExposureMetadata(path, 0.01, 100, 8, datetime(2024, 4, 8, 18, 0, index))
        for index, path in enumerate(paths)
    ]
    return paths, metadata


def test_grouping_uses_consecutive_complete_sets_and_warns_on_remainder() -> None:
    paths, metadata = _items(12)
    groups, warnings = group_brackets(paths, metadata, 5)
    assert len(groups) == 2
    assert [path.name for path in groups[1][0]] == [f"DSC_{index:04d}.NEF" for index in range(5, 10)]
    assert "Ignored 2 trailing" in warnings[0]


def test_single_bracket_selects_start_group_only(tmp_path: Path) -> None:
    config = PipelineConfig(
        tmp_path,
        tmp_path / "out",
        start_group=3,
        end_group=7,
        single_bracket=True,
    )
    assert _selected_indices(10, config) == [3]


def test_memmap_is_closed_before_windows_temp_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "mapped.npy"
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(4, 4))
    mapped[:] = 1
    _close_memmaps([mapped])
    path.unlink()
    assert not path.exists()


def test_existing_master_and_sidecar_are_preserved_without_overwrite(tmp_path: Path) -> None:
    paths, metadata = _items(5)
    output = tmp_path / "out"
    output.mkdir()
    master = output / "DSC_0002_HDR.tif"
    sidecar = output / "DSC_0002_HDR.json"
    master.write_bytes(b"existing master")
    sidecar.write_text("existing sidecar", encoding="utf-8")
    result = process_bracket(
        0,
        paths,
        metadata,
        PipelineConfig(tmp_path, output, overwrite=False),
    )
    assert result.status == "failed"
    assert master.read_bytes() == b"existing master"
    assert sidecar.read_text(encoding="utf-8") == "existing sidecar"
