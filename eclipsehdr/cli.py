"""Windows-friendly command-line interface."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .errors import EclipseHDRError
from .models import MergeConfig, RegistrationConfig
from .pipeline import PipelineConfig, run_batch


def _roi(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be four integers: X,Y,W,H") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be four integers: X,Y,W,H")
    return parts  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eclipse_hdr.py",
        description=(
            "Develop Nikon NEFs linearly, align each bracket with x/y translation only, "
            "and write an untone-mapped float32 TIFF master."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="directory containing chronological NEFs")
    parser.add_argument("output_dir", type=Path, help="directory for HDR masters and JSON sidecars")
    parser.add_argument("--group-size", type=int, default=5, help="consecutive NEFs per bracket")
    parser.add_argument(
        "--max-shift", type=float, default=20.0, help="maximum absolute dx and dy in pixels"
    )
    parser.add_argument(
        "--registration-crop",
        type=int,
        default=2048,
        help="minimum native-pixel crop around the automatically located Sun",
    )
    parser.add_argument(
        "--roi",
        type=_roi,
        metavar="X,Y,W,H",
        help="manual shared registration crop if automatic Sun localization is ambiguous",
    )
    parser.add_argument(
        "--upsample-factor", type=int, default=20, help="subpixel phase-correlation grid factor"
    )
    parser.add_argument(
        "--max-linear-residual",
        type=float,
        default=1.0,
        help="largest allowed departure from constant-motion fit in pixels",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=1.5,
        help="largest allowed adjacent offset second-difference in pixels",
    )
    parser.add_argument(
        "--min-registration-score",
        type=float,
        default=0.05,
        help="minimum aligned feature correlation before a frame is suspicious",
    )
    parser.add_argument(
        "--min-registration-psr",
        type=float,
        default=1.25,
        help="minimum bounded-correlation peak-to-sidelobe ratio",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=2.0,
        help="warn when successive capture timestamps are farther apart",
    )
    parser.add_argument("--start-group", type=int, default=0, help="first 0-based group, inclusive")
    parser.add_argument("--end-group", type=int, help="last 0-based group, inclusive")
    parser.add_argument(
        "--single-bracket",
        action="store_true",
        help="process only --start-group (use this for the first validation run)",
    )
    parser.add_argument(
        "--on-suspicious",
        choices=("skip", "continue", "error"),
        default="skip",
        help="policy after offsets fail physical checks",
    )
    parser.add_argument(
        "--alignment-preview",
        action="store_true",
        help="save before/after crops and red/cyan edge overlays",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="save large 16-bit linear developed TIFFs under output/intermediates",
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=0.001,
        help="zero-weight threshold in normalized developed linear RGB",
    )
    parser.add_argument(
        "--black-full-weight",
        type=float,
        default=0.02,
        help="end of the near-black weight ramp",
    )
    parser.add_argument(
        "--highlight-falloff",
        type=float,
        default=0.85,
        help="start of the highlight weight rolloff",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.98,
        help="postprocessed RGB clipping fallback threshold",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=128, help="rows merged at once to bound memory use"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing HDR masters")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="verbose logging and alignment diagnostic images",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        parser.error(f"{name} must be finite and positive")


def config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> PipelineConfig:
    _positive(parser, "--group-size", args.group_size)
    _positive(parser, "--max-shift", args.max_shift)
    _positive(parser, "--registration-crop", args.registration_crop)
    _positive(parser, "--upsample-factor", args.upsample_factor)
    _positive(parser, "--max-gap-seconds", args.max_gap_seconds)
    _positive(parser, "--chunk-rows", args.chunk_rows)
    _positive(parser, "--max-linear-residual", args.max_linear_residual)
    _positive(parser, "--max-acceleration", args.max_acceleration)
    if not math.isfinite(args.min_registration_score) or not -1.0 <= args.min_registration_score <= 1.0:
        parser.error("--min-registration-score must be between -1 and 1")
    _positive(parser, "--min-registration-psr", args.min_registration_psr)
    merge_levels = (
        args.black_threshold,
        args.black_full_weight,
        args.highlight_falloff,
        args.saturation_threshold,
    )
    if not all(math.isfinite(value) for value in merge_levels) or not (
        0 <= merge_levels[0] < merge_levels[1] < merge_levels[2] < merge_levels[3] <= 1
    ):
        parser.error(
            "merge levels must satisfy 0 <= --black-threshold < --black-full-weight < "
            "--highlight-falloff < --saturation-threshold <= 1"
        )
    registration = RegistrationConfig(
        max_shift_px=args.max_shift,
        crop_size_px=args.registration_crop,
        upsample_factor=args.upsample_factor,
        max_linear_residual_px=args.max_linear_residual,
        max_acceleration_px=args.max_acceleration,
        min_score=args.min_registration_score,
        min_psr=args.min_registration_psr,
        manual_roi_xywh=args.roi,
    )
    merge = MergeConfig(
        black_start=args.black_threshold,
        black_full_weight=args.black_full_weight,
        highlight_falloff=args.highlight_falloff,
        saturation_cutoff=args.saturation_threshold,
        chunk_rows=args.chunk_rows,
    )
    return PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        group_size=args.group_size,
        start_group=args.start_group,
        end_group=args.end_group,
        single_bracket=args.single_bracket,
        max_gap_seconds=args.max_gap_seconds,
        keep_intermediates=args.keep_intermediates,
        save_diagnostics=args.alignment_preview or args.debug,
        on_suspicious=args.on_suspicious,
        overwrite=args.overwrite,
        registration=registration,
        merge=merge,
    )


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug:
        logging.getLogger("PIL").setLevel(logging.INFO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args, parser)
    configure_logging(args.debug)
    try:
        result = run_batch(config)
    except EclipseHDRError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).error("Interrupted; NEF originals were not changed")
        return 130

    logging.getLogger(__name__).info(
        "Finished: %d complete, %d suspicious/skipped, %d failed",
        result.completed,
        result.suspicious_skipped,
        result.failed,
    )
    if result.failed:
        return 1
    if result.suspicious_skipped:
        return 2
    return 0
