import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

import eclipsehdr.pipeline as pipeline_module
from eclipsehdr.models import ExposureMetadata, MergeConfig, RegistrationConfig
from eclipsehdr.pipeline import PipelineConfig, process_bracket
from eclipsehdr.rawio import DevelopedFrame, RawDevelopmentSettings


def test_single_bracket_pipeline_writes_master_sidecar_and_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    size = 160
    y, x = np.mgrid[:size, :size].astype(np.float32)
    radius = np.hypot(y - 78.0, x - 83.0)
    limb = 0.55 * (radius <= 27.0) + 0.25 * np.exp(-0.5 * ((radius - 27.0) / 1.2) ** 2)
    bright_detail = 2.5 * np.exp(-((x - 105.0) ** 2 + (y - 65.0) ** 2) / 12.0)
    scene = np.stack(
        [limb + bright_detail, 0.85 * limb + bright_detail, 0.65 * limb + bright_detail],
        axis=2,
    )
    displacements = [(-3.0, 1.5), (-1.5, 0.75), (0.0, 0.0), (1.5, -0.75), (3.0, -1.5)]
    exposures = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    frames = []
    masks = []
    for displacement, exposure in zip(displacements, exposures):
        measured = np.clip(scene * exposure, 0.0, 1.0)
        shifted = ndimage.shift(
            measured,
            shift=(displacement[0], displacement[1], 0),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        frames.append(np.rint(shifted * 65535.0).astype(np.uint16))
        masks.append((np.max(shifted, axis=2) >= 1.0).astype(np.uint8))

    paths = [tmp_path / f"DSC_{index:04d}.NEF" for index in range(5)]
    start = datetime(2024, 4, 8, 18, 0, 0)
    metadata = [
        ExposureMetadata(
            path,
            shutter_s=float(exposure / 100.0),
            iso=100.0,
            aperture=1.0,
            timestamp=start + timedelta(seconds=0.3 * index),
        )
        for index, (path, exposure) in enumerate(zip(paths, exposures))
    ]

    def fake_develop_group(
        requested_paths,
        requested_metadata,
        reference_index,
        work_dir,
        **kwargs,
    ):
        assert list(requested_paths) == paths
        work_dir.mkdir(parents=True, exist_ok=True)
        developed = []
        for index, (rgb, mask) in enumerate(zip(frames, masks)):
            rgb_path = work_dir / f"frame_{index:02d}.npy"
            mask_path = work_dir / f"mask_{index:02d}.npy"
            np.save(rgb_path, rgb)
            np.save(mask_path, mask)
            developed.append(
                DevelopedFrame(metadata[index], rgb_path, mask_path, rgb.shape)
            )
        return developed, RawDevelopmentSettings((2.0, 1.0, 1.5, 1.0), 0, 16383)

    monkeypatch.setattr(pipeline_module, "develop_group", fake_develop_group)
    output_dir = tmp_path / "output"
    config = PipelineConfig(
        input_dir=tmp_path,
        output_dir=output_dir,
        save_diagnostics=True,
        registration=RegistrationConfig(
            max_shift_px=8,
            crop_size_px=size,
            manual_roi_xywh=(0, 0, size, size),
            max_linear_residual_px=0.5,
            max_acceleration_px=0.75,
        ),
        merge=MergeConfig(chunk_rows=31),
    )
    result = process_bracket(0, paths, metadata, config, rawpy_module=object())
    assert result.status == "complete"
    assert result.output_path is not None and result.output_path.exists()
    master = tifffile.imread(result.output_path)
    assert master.dtype == np.float32
    assert float(master.max()) > 1.0
    assert result.sidecar_path is not None
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "complete"
    assert sidecar["alignment"]["sanity"]["suspicious"] is False
    assert len(sidecar["sources"]) == 5
    assert (output_dir / "diagnostics" / "DSC_0002_alignment_preview.png").exists()
    assert (output_dir / "diagnostics" / "DSC_0002_edge_overlay.png").exists()
    assert (output_dir / "diagnostics" / "DSC_0002_HDR_preview.png").exists()
