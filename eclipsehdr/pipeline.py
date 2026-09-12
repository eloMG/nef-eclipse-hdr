"""Single-bracket processor and chronological batch wrapper."""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import __version__
from .aligned import export_aligned_linear_frames
from .diagnostics import save_alignment_preview, save_edge_overlay, save_hdr_preview
from .errors import EclipseHDRError, OutputExistsError, SuspiciousAlignmentError
from .merge import create_float32_memmap, merge_linear_hdr
from .metadata import exposure_factors, timestamp_gaps
from .models import ExposureMetadata, MergeConfig, RegistrationConfig
from .output import write_float_tiff, write_json
from .rawio import develop_group, import_rawpy, read_metadata
from .registration import estimate_group_alignment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    group_size: int = 5
    start_group: int = 0
    end_group: Optional[int] = None
    single_bracket: bool = False
    max_gap_seconds: float = 2.0
    keep_intermediates: bool = False
    aligned_only: bool = False
    save_diagnostics: bool = False
    on_suspicious: str = "skip"
    overwrite: bool = False
    registration: RegistrationConfig = RegistrationConfig()
    merge: MergeConfig = MergeConfig()


@dataclass(frozen=True)
class GroupResult:
    group_index: int
    reference_name: str
    status: str
    output_path: Optional[Path] = None
    output_paths: tuple[Path, ...] = ()
    sidecar_path: Optional[Path] = None
    message: str = ""


@dataclass(frozen=True)
class BatchResult:
    groups_found: int
    selected_groups: int
    completed: int
    suspicious_skipped: int
    failed: int
    results: tuple[GroupResult, ...]
    warnings: tuple[str, ...] = ()


def _close_memmaps(arrays: Sequence[Optional[np.ndarray]]) -> None:
    """Release .npy mappings before Windows removes a temporary directory."""

    closed_handles: set[int] = set()
    for array in arrays:
        if not isinstance(array, np.memmap):
            continue
        mapping = getattr(array, "_mmap", None)
        if mapping is None or id(mapping) in closed_handles:
            continue
        try:
            array.flush()
        except (OSError, ValueError):
            pass
        mapping.close()
        closed_handles.add(id(mapping))


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_nefs(input_dir: Path) -> list[Path]:
    """Find NEFs in one directory without ever modifying the originals."""

    if not input_dir.exists():
        raise EclipseHDRError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise EclipseHDRError(f"Input path is not a directory: {input_dir}")
    try:
        paths = sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".nef"
            ),
            key=_natural_key,
        )
    except OSError as exc:
        raise EclipseHDRError(f"Could not enumerate input directory {input_dir}: {exc}") from exc
    if not paths:
        raise EclipseHDRError(f"No .NEF files found directly inside {input_dir}")
    return paths


def scan_and_order(
    input_dir: Path, *, rawpy_module: Any = None
) -> tuple[list[Path], list[ExposureMetadata], list[str]]:
    """Read metadata first, then prefer capture-time order when all times exist."""

    rawpy_module = rawpy_module or import_rawpy()
    paths = discover_nefs(input_dir)
    LOGGER.info("Reading metadata from %d NEF files", len(paths))
    metadata = [read_metadata(path, rawpy_module) for path in paths]
    warnings: list[str] = []
    if all(item.timestamp is not None for item in metadata):
        ordered = sorted(
            zip(paths, metadata),
            key=lambda pair: (pair[1].timestamp, _natural_key(pair[0])),
        )
        paths = [pair[0] for pair in ordered]
        metadata = [pair[1] for pair in ordered]
    else:
        warnings.append(
            "At least one NEF lacks a capture timestamp; natural filename order was used."
        )
    return paths, metadata, warnings


def group_brackets(
    paths: Sequence[Path],
    metadata: Sequence[ExposureMetadata],
    group_size: int,
) -> tuple[list[tuple[list[Path], list[ExposureMetadata]]], list[str]]:
    if group_size < 2:
        raise EclipseHDRError("--group-size must be at least 2")
    if len(paths) != len(metadata):
        raise EclipseHDRError("Internal path/metadata count mismatch")
    groups: list[tuple[list[Path], list[ExposureMetadata]]] = []
    complete_count = len(paths) // group_size
    for group_index in range(complete_count):
        start = group_index * group_size
        stop = start + group_size
        groups.append((list(paths[start:stop]), list(metadata[start:stop])))
    warnings: list[str] = []
    remainder = len(paths) % group_size
    if remainder:
        trailing = ", ".join(path.name for path in paths[-remainder:])
        warnings.append(
            f"Ignored {remainder} trailing NEF file(s) that do not make a complete "
            f"group of {group_size}: {trailing}"
        )
    if not groups:
        raise EclipseHDRError(
            f"Found {len(paths)} NEF files, fewer than one complete group of {group_size}"
        )
    return groups, warnings


def _base_sidecar(
    group_index: int,
    paths: Sequence[Path],
    metadata: Sequence[ExposureMetadata],
    reference_index: int,
    timing_warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": f"nef-eclipse-hdr {__version__}",
        "group_index": group_index,
        "status": "started",
        "reference_index": reference_index,
        "reference_filename": paths[reference_index].name,
        "sources": [item.to_dict() for item in metadata],
        "burst_timing_warnings": list(timing_warnings),
    }


def _log_offsets(group_index: int, alignment: Any) -> None:
    LOGGER.info("Group %d translation offsets (applied to each frame):", group_index)
    for frame in alignment.frames:
        if frame.frame_index == alignment.reference_index:
            LOGGER.info("  frame %d: reference", frame.frame_index)
        else:
            LOGGER.info(
                "  frame %d: dx=%+.3f dy=%+.3f score=%.3f (%s)",
                frame.frame_index,
                frame.dx,
                frame.dy,
                frame.score,
                frame.method,
            )


def process_bracket(
    group_index: int,
    paths: Sequence[Path],
    metadata: Sequence[ExposureMetadata],
    config: PipelineConfig,
    *,
    rawpy_module: Any = None,
) -> GroupResult:
    """Process exactly one bracket; this is independently usable by tests/callers."""

    if len(paths) != len(metadata):
        raise EclipseHDRError("Bracket path and metadata counts differ")
    if len(paths) != config.group_size:
        raise EclipseHDRError(
            f"Bracket contains {len(paths)} frames, expected configured group size {config.group_size}"
        )
    if len(paths) < 2:
        raise EclipseHDRError("A bracket must contain at least two frames")
    reference_index = len(paths) // 2
    reference_stem = paths[reference_index].stem
    output_path = config.output_dir / f"{reference_stem}_HDR.tif"
    aligned_dir = config.output_dir / "aligned" / reference_stem
    aligned_output_paths = tuple(
        aligned_dir / f"{index:02d}_{path.stem}_aligned_linear.tif"
        for index, path in enumerate(paths)
    )
    sidecar_path = (
        aligned_dir / f"{reference_stem}_aligned.json"
        if config.aligned_only
        else config.output_dir / f"{reference_stem}_HDR.json"
    )
    timing_warnings = timestamp_gaps(metadata, config.max_gap_seconds)
    payload = _base_sidecar(group_index, paths, metadata, reference_index, timing_warnings)
    payload["mode"] = "aligned_only" if config.aligned_only else "hdr_merge"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = (
        [path for path in aligned_output_paths if path.exists()]
        if config.aligned_only
        else ([output_path] if output_path.exists() else [])
    )
    if existing_outputs and not config.overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        message = f"Output already exists (use --overwrite): {names}"
        return GroupResult(
            group_index,
            paths[reference_index].name,
            "failed",
            sidecar_path=sidecar_path if sidecar_path.exists() else None,
            message=message,
        )

    for warning in timing_warnings:
        LOGGER.warning("Group %d: %s", group_index, warning)

    with tempfile.TemporaryDirectory(prefix=f"eclipse_hdr_g{group_index:04d}_") as temporary:
        work_dir = Path(temporary)
        intermediate_dir = (
            config.output_dir / "intermediates" / reference_stem
            if config.keep_intermediates
            else None
        )
        mapped_arrays: list[Optional[np.ndarray]] = []
        try:
            developed, raw_settings = develop_group(
                paths,
                metadata,
                reference_index,
                work_dir / "developed",
                keep_intermediates_dir=intermediate_dir,
                rawpy_module=rawpy_module,
            )
            frames = [frame.open_rgb() for frame in developed]
            masks = [frame.open_saturation_mask() for frame in developed]
            mapped_arrays.extend(frames)
            mapped_arrays.extend(masks)
            payload["raw_development"] = raw_settings.to_dict()
            payload["raw_development"]["frame_warnings"] = [
                {"frame_index": index, "warnings": list(frame.warnings)}
                for index, frame in enumerate(developed)
                if frame.warnings
            ]

            alignment = estimate_group_alignment(
                frames,
                metadata,
                reference_index,
                config.registration,
                saturation_masks=masks,
            )
            payload["alignment"] = alignment.to_dict()
            _log_offsets(group_index, alignment)
            for warning in alignment.warnings:
                LOGGER.warning("Group %d alignment advisory: %s", group_index, warning)

            if config.save_diagnostics:
                diagnostics_dir = config.output_dir / "diagnostics"
                save_alignment_preview(
                    diagnostics_dir / f"{reference_stem}_alignment_preview.png",
                    frames,
                    alignment,
                )
                save_edge_overlay(
                    diagnostics_dir / f"{reference_stem}_edge_overlay.png",
                    frames,
                    alignment,
                )
                payload["diagnostics"] = {
                    "alignment_preview": str(
                        (diagnostics_dir / f"{reference_stem}_alignment_preview.png").resolve()
                    ),
                    "edge_overlay": str(
                        (diagnostics_dir / f"{reference_stem}_edge_overlay.png").resolve()
                    ),
                }

            if alignment.suspicious:
                LOGGER.error("Group %d alignment is suspicious:", group_index)
                for issue in alignment.issues:
                    LOGGER.error("  - %s", issue)
                payload["status"] = "suspicious"
                write_json(sidecar_path, payload)
                if config.on_suspicious == "skip":
                    return GroupResult(
                        group_index,
                        paths[reference_index].name,
                        "suspicious_skipped",
                        sidecar_path=sidecar_path,
                        message="; ".join(alignment.issues),
                    )
                if config.on_suspicious == "error":
                    raise SuspiciousAlignmentError("; ".join(alignment.issues))
                LOGGER.warning(
                    "Continuing group %d only because --on-suspicious continue was selected",
                    group_index,
                )

            if config.aligned_only:
                exported_paths = export_aligned_linear_frames(
                    frames,
                    alignment.offsets_yx,
                    aligned_output_paths,
                    work_dir / "aligned_export",
                    chunk_rows=config.merge.chunk_rows,
                    interpolation_order=config.merge.interpolation_order,
                    overwrite=config.overwrite,
                )
                height, width, _ = frames[0].shape
                payload.update(
                    status=(
                        "complete"
                        if not alignment.suspicious
                        else "complete_suspicious_override"
                    ),
                    output={
                        "type": "aligned_linear_frames",
                        "directory": str(aligned_dir.resolve()),
                        "format": "TIFF, contiguous RGB, uint16, linear-sRGB primaries, unprofiled",
                        "dimensions": {"width": width, "height": height},
                        "interpolation": (
                            "bilinear translation"
                            if config.merge.interpolation_order == 1
                            else "nearest-neighbor translation"
                        ),
                        "outside_source_bounds": "black (integer code 0)",
                        "exposure_handling": (
                            "original exposure retained independently in each frame; "
                            "no normalization or HDR merge"
                        ),
                        "files": [
                            {
                                "frame_index": frame.frame_index,
                                "source_filename": paths[frame.frame_index].name,
                                "path": str(exported_paths[frame.frame_index].resolve()),
                                "dx": frame.dx,
                                "dy": frame.dy,
                            }
                            for frame in alignment.frames
                        ],
                    },
                )
                write_json(sidecar_path, payload)
                return GroupResult(
                    group_index,
                    paths[reference_index].name,
                    "complete",
                    output_paths=exported_paths,
                    sidecar_path=sidecar_path,
                )

            factors, exposure_warnings = exposure_factors(metadata, reference_index)
            payload["exposure_warnings"] = exposure_warnings
            for warning in exposure_warnings:
                LOGGER.warning("Group %d: %s", group_index, warning)

            output_memmap = create_float32_memmap(work_dir / "hdr.npy", developed[0].shape)
            mapped_arrays.append(output_memmap)
            hdr, merge_stats = merge_linear_hdr(
                frames,
                alignment.offsets_yx,
                factors,
                reference_index,
                config.merge,
                saturation_masks=masks,
                output=output_memmap,
            )
            payload["merge"] = {
                "method": (
                    "relative exposure-normalized RGB = RGB / (exposure/reference exposure); "
                    "smooth scalar black/highlight weight; no tone mapping"
                ),
                "settings": {
                    "black_start": config.merge.black_start,
                    "black_full_weight": config.merge.black_full_weight,
                    "highlight_falloff": config.merge.highlight_falloff,
                    "saturation_cutoff": config.merge.saturation_cutoff,
                    "interpolation": (
                        "bilinear translation" if config.merge.interpolation_order == 1 else "nearest"
                    ),
                },
                "statistics": merge_stats.to_dict(),
            }
            if config.save_diagnostics:
                diagnostics_dir = config.output_dir / "diagnostics"
                hdr_preview_path = diagnostics_dir / f"{reference_stem}_HDR_preview.png"
                save_hdr_preview(hdr_preview_path, hdr, alignment.crop)
                payload.setdefault("diagnostics", {})["hdr_preview"] = str(
                    hdr_preview_path.resolve()
                )

            partial_path = output_path.with_name(
                f".{output_path.stem}.{uuid.uuid4().hex}.partial.tif"
            )
            try:
                write_float_tiff(partial_path, hdr, verify=True)
                if config.overwrite:
                    partial_path.replace(output_path)
                else:
                    # On Windows rename fails atomically if a concurrent run created
                    # the destination after the initial no-overwrite check.
                    try:
                        partial_path.rename(output_path)
                    except OSError as exc:
                        if output_path.exists():
                            raise OutputExistsError(
                                f"Output appeared during processing; it was preserved: {output_path.name}"
                            ) from exc
                        raise
            finally:
                if partial_path.exists():
                    partial_path.unlink()

            payload.update(
                status="complete" if not alignment.suspicious else "complete_suspicious_override",
                output={
                    "path": str(output_path.resolve()),
                    "format": "TIFF, contiguous RGB, IEEE float32, linear-sRGB primaries, unprofiled",
                },
            )
            write_json(sidecar_path, payload)
            return GroupResult(
                group_index,
                paths[reference_index].name,
                "complete",
                output_path=output_path,
                sidecar_path=sidecar_path,
            )
        except OutputExistsError:
            # Preserve a sidecar that may belong to the concurrent winning run.
            raise
        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            payload.update(status="error", error=f"{type(exc).__name__}: {exc}")
            write_json(sidecar_path, payload)
            raise
        finally:
            _close_memmaps(mapped_arrays)


def _selected_indices(group_count: int, config: PipelineConfig) -> list[int]:
    if config.start_group < 0:
        raise EclipseHDRError("--start-group cannot be negative")
    if config.start_group >= group_count:
        raise EclipseHDRError(
            f"--start-group {config.start_group} is outside 0..{group_count - 1}"
        )
    end = group_count - 1 if config.end_group is None else config.end_group
    if end < config.start_group:
        raise EclipseHDRError("--end-group must be greater than or equal to --start-group")
    if end >= group_count:
        raise EclipseHDRError(f"--end-group {end} is outside 0..{group_count - 1}")
    if config.single_bracket:
        end = config.start_group
    return list(range(config.start_group, end + 1))


def run_batch(config: PipelineConfig, *, rawpy_module: Any = None) -> BatchResult:
    """Chronological wrapper around the independently testable bracket processor."""

    if config.on_suspicious not in {"skip", "continue", "error"}:
        raise EclipseHDRError(f"Invalid suspicious-alignment policy: {config.on_suspicious}")
    rawpy_module = rawpy_module or import_rawpy()
    paths, metadata, scan_warnings = scan_and_order(config.input_dir, rawpy_module=rawpy_module)
    groups, grouping_warnings = group_brackets(paths, metadata, config.group_size)
    warnings = scan_warnings + grouping_warnings
    for warning in warnings:
        LOGGER.warning(warning)
    indices = _selected_indices(len(groups), config)
    LOGGER.info(
        "Found %d complete bracket(s); processing group(s) %s",
        len(groups),
        ", ".join(str(index) for index in indices),
    )

    results: list[GroupResult] = []
    for group_index in indices:
        group_paths, group_metadata = groups[group_index]
        LOGGER.info(
            "Starting group %d: %s .. %s",
            group_index,
            group_paths[0].name,
            group_paths[-1].name,
        )
        try:
            result = process_bracket(
                group_index,
                group_paths,
                group_metadata,
                config,
                rawpy_module=rawpy_module,
            )
        except SuspiciousAlignmentError as exc:
            result = GroupResult(
                group_index,
                group_paths[len(group_paths) // 2].name,
                "failed",
                message=str(exc),
            )
            results.append(result)
            break
        except EclipseHDRError as exc:
            LOGGER.error("Group %d failed: %s", group_index, exc)
            result = GroupResult(
                group_index,
                group_paths[len(group_paths) // 2].name,
                "failed",
                message=str(exc),
            )
        except Exception as exc:
            LOGGER.exception("Unexpected failure in group %d", group_index)
            result = GroupResult(
                group_index,
                group_paths[len(group_paths) // 2].name,
                "failed",
                message=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        if result.status == "failed" and result.message:
            LOGGER.error("Group %d failed: %s", group_index, result.message)

    completed = sum(result.status == "complete" for result in results)
    suspicious_skipped = sum(result.status == "suspicious_skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    return BatchResult(
        groups_found=len(groups),
        selected_groups=len(indices),
        completed=completed,
        suspicious_skipped=suspicious_skipped,
        failed=failed,
        results=tuple(results),
        warnings=tuple(warnings),
    )
