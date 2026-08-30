"""Writers for linear masters, diagnostics, and machine-readable sidecars."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .errors import EclipseHDRError


def write_float_tiff(path: Path, rgb: np.ndarray, *, verify: bool = True) -> None:
    """Write one uncompressed, contiguous RGB IEEE-float TIFF.

    ``tifffile`` does not alter or normalize these values.  In particular, scene
    values greater than one remain greater than one.  No nonlinear ICC profile is
    attached because the samples are linear-sRGB, not transfer-encoded sRGB.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise EclipseHDRError(f"HDR master must have shape (height, width, 3), got {rgb.shape}")
    if rgb.dtype != np.float32:
        raise EclipseHDRError(f"HDR master must be float32 before writing, got {rgb.dtype}")
    for y0 in range(0, rgb.shape[0], 256):
        if not np.isfinite(rgb[y0 : y0 + 256]).all():
            raise EclipseHDRError("HDR master contains NaN or infinity; refusing to write it")

    tifffile.imwrite(
        path,
        rgb,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        compression=None,
        # Nikon-sized RGB float files are normally <4 GiB.  tifffile will switch
        # automatically if a larger array actually requires BigTIFF.
        bigtiff=None,
        description=(
            "Linear-light, scene-referred relative RGB; sRGB/BT.709 primaries; "
            "no tone mapping, gamma encoding, sharpening, or denoising"
        ),
        software="eclipse-hdr",
    )

    if verify:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if page.dtype != np.dtype(np.float32) or page.shape != rgb.shape:
                raise EclipseHDRError(
                    f"TIFF verification failed: wrote {page.shape} {page.dtype}, "
                    f"expected {rgb.shape} float32"
                )
            if str(page.photometric.name).upper() != "RGB":
                raise EclipseHDRError(
                    f"TIFF verification failed: photometric is {page.photometric.name}, not RGB"
                )


def write_linear_intermediate(path: Path, rgb_u16: np.ndarray) -> None:
    """Write an optional developed 16-bit linear RGB diagnostic."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        rgb_u16,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        compression=None,
        description="Linear LibRaw development; diagnostic intermediate only",
        software="eclipse-hdr",
    )


def write_aligned_linear_tiff(
    path: Path,
    rgb_u16: np.ndarray,
    *,
    dy: float,
    dx: float,
    verify: bool = True,
) -> None:
    """Write one full-resolution translated linear uint16 RGB frame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_u16.ndim != 3 or rgb_u16.shape[2] != 3:
        raise EclipseHDRError(
            f"Aligned frame must have shape (height, width, 3), got {rgb_u16.shape}"
        )
    if rgb_u16.dtype != np.uint16:
        raise EclipseHDRError(f"Aligned frame must be uint16 before writing, got {rgb_u16.dtype}")

    tifffile.imwrite(
        path,
        rgb_u16,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        compression=None,
        bigtiff=None,
        description=(
            f"Linear LibRaw development; translated by dx={dx:+.6f}, dy={dy:+.6f} "
            "pixels into the bracket reference coordinates; black outside source bounds; "
            "original exposure retained; no tone mapping or gamma encoding"
        ),
        software="eclipse-hdr",
    )

    if verify:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if page.dtype != np.dtype(np.uint16) or page.shape != rgb_u16.shape:
                raise EclipseHDRError(
                    f"Aligned TIFF verification failed: wrote {page.shape} {page.dtype}, "
                    f"expected {rgb_u16.shape} uint16"
                )
            if str(page.photometric.name).upper() != "RGB":
                raise EclipseHDRError(
                    "Aligned TIFF verification failed: "
                    f"photometric is {page.photometric.name}, not RGB"
                )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a UTF-8 JSON sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
