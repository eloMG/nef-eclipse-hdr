"""Full-resolution export of translation-aligned linear developments."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import EclipseHDRError, OutputExistsError
from .output import write_aligned_linear_tiff
from .resampling import translate_linear_u16

LOGGER = logging.getLogger(__name__)


def _close_memmap(array: np.memmap) -> None:
    try:
        array.flush()
    finally:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _publish_outputs(partials: Sequence[tuple[Path, Path]], overwrite: bool) -> None:
    """Publish a complete set, rolling back files already moved on failure."""

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        if overwrite:
            for _, final_path in partials:
                if not final_path.exists():
                    continue
                backup_path = final_path.with_name(
                    f".{final_path.name}.{uuid.uuid4().hex}.backup"
                )
                final_path.rename(backup_path)
                backups.append((backup_path, final_path))
        else:
            appeared = [final for _, final in partials if final.exists()]
            if appeared:
                names = ", ".join(path.name for path in appeared)
                raise OutputExistsError(
                    f"Aligned output appeared during processing; it was preserved: {names}"
                )

        for partial_path, final_path in partials:
            try:
                partial_path.rename(final_path)
            except OSError as exc:
                if final_path.exists() and not overwrite:
                    raise OutputExistsError(
                        "Aligned output appeared during processing; it was preserved: "
                        f"{final_path.name}"
                    ) from exc
                raise
            published.append(final_path)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for final_path in reversed(published):
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(f"remove {final_path.name}: {rollback_exc}")
        for backup_path, final_path in reversed(backups):
            try:
                if backup_path.exists():
                    backup_path.replace(final_path)
            except OSError as rollback_exc:
                rollback_errors.append(f"restore {final_path.name}: {rollback_exc}")
        if rollback_errors:
            raise EclipseHDRError(
                "Aligned output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise

    for backup_path, _ in backups:
        try:
            if backup_path.exists():
                backup_path.unlink()
        except OSError as exc:
            # The requested outputs are complete; retaining a hidden backup is
            # safer than turning successful publication into a reported failure.
            LOGGER.warning("Could not remove aligned-output backup %s: %s", backup_path, exc)


def export_aligned_linear_frames(
    frames: Sequence[np.ndarray],
    offsets_yx: Sequence[tuple[float, float]],
    output_paths: Sequence[Path],
    work_dir: Path,
    *,
    chunk_rows: int,
    interpolation_order: int,
    overwrite: bool,
) -> tuple[Path, ...]:
    """Write all frames in reference coordinates without exposure normalization."""

    count = len(frames)
    if not count or not (len(offsets_yx) == len(output_paths) == count):
        raise EclipseHDRError("Aligned frame, offset, and output counts must match")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise EclipseHDRError(f"Developed frame dimensions differ: {sorted(shapes)}")

    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise OutputExistsError(f"Aligned output already exists (use --overwrite): {names}")

    work_dir.mkdir(parents=True, exist_ok=True)
    partials: list[tuple[Path, Path]] = []
    try:
        for index, (frame, (dy, dx), final_path) in enumerate(
            zip(frames, offsets_yx, output_paths)
        ):
            LOGGER.info(
                "Writing aligned frame %d/%d: %s", index + 1, count, final_path.name
            )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = final_path.with_name(
                f".{final_path.stem}.{uuid.uuid4().hex}.partial.tif"
            )
            partials.append((partial_path, final_path))
            mapped_path = work_dir / f"aligned_{index:02d}_{uuid.uuid4().hex}.npy"
            aligned = np.lib.format.open_memmap(
                mapped_path, mode="w+", dtype=np.uint16, shape=frame.shape
            )
            try:
                translate_linear_u16(
                    frame,
                    float(dy),
                    float(dx),
                    chunk_rows=chunk_rows,
                    interpolation_order=interpolation_order,
                    output=aligned,
                )
                aligned.flush()
                write_aligned_linear_tiff(
                    partial_path,
                    aligned,
                    dy=float(dy),
                    dx=float(dx),
                    verify=True,
                )
            finally:
                _close_memmap(aligned)
                if mapped_path.exists():
                    mapped_path.unlink()

        # All five files are complete before any is published under its final name.
        _publish_outputs(partials, overwrite)
        return tuple(output_paths)
    finally:
        for partial_path, _ in partials:
            if partial_path.exists():
                partial_path.unlink()
