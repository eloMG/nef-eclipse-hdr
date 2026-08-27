import pytest

from eclipsehdr.cli import build_parser, config_from_args


def test_nan_numeric_option_is_rejected(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args([str(tmp_path), str(tmp_path / "out"), "--max-shift", "nan"])
    with pytest.raises(SystemExit):
        config_from_args(args, parser)


def test_invalid_merge_threshold_order_is_rejected(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [str(tmp_path), str(tmp_path / "out"), "--highlight-falloff", "0.99", "--saturation-threshold", "0.98"]
    )
    with pytest.raises(SystemExit):
        config_from_args(args, parser)

