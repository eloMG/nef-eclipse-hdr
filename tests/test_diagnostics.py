from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from eclipsehdr.diagnostics import save_alignment_preview, save_edge_overlay
from eclipsehdr.models import AlignmentResult, CropRegion, FrameAlignment


def test_alignment_diagnostics_accept_rgb_edge_overlays(tmp_path: Path) -> None:
    frames = []
    y, x = np.mgrid[:96, :96]
    for index in range(3):
        signal = np.exp(-((x - 48 - index) ** 2 + (y - 47) ** 2) / 80.0)
        rgb = np.stack([signal, 0.8 * signal, 0.5 * signal], axis=2)
        frames.append(np.rint(rgb * 65535).astype(np.uint16))
    alignment = AlignmentResult(
        reference_index=1,
        frames=(
            FrameAlignment(0, 0.0, 1.0, "test", 1.0),
            FrameAlignment(1, 0.0, 0.0, "reference", 1.0),
            FrameAlignment(2, 0.0, -1.0, "test", 1.0),
        ),
        crop=CropRegion(0, 96, 0, 96, method="manual"),
        suspicious=False,
    )
    preview = tmp_path / "preview.png"
    overlay = tmp_path / "overlay.png"
    save_alignment_preview(preview, frames, alignment)
    save_edge_overlay(overlay, frames, alignment)
    assert Image.open(preview).mode == "RGB"
    assert Image.open(overlay).mode == "RGB"

