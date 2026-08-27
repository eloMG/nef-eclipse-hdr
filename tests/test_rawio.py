from pathlib import Path
from types import SimpleNamespace

from eclipsehdr.rawio import RawDevelopmentSettings, build_rawpy_params, metadata_from_open_raw


class FakeParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_linear_rawpy_parameters_are_fixed_and_noncreative() -> None:
    fake_rawpy = SimpleNamespace(
        Params=FakeParams,
        DemosaicAlgorithm=SimpleNamespace(AHD="AHD"),
        FBDDNoiseReductionMode=SimpleNamespace(Off="off"),
        ColorSpace=SimpleNamespace(sRGB="sRGB"),
        HighlightMode=SimpleNamespace(Ignore="ignore"),
    )
    settings = RawDevelopmentSettings((2.0, 1.0, 1.5, 1.0), 6, 16383)
    params = build_rawpy_params(fake_rawpy, settings).kwargs
    assert params["gamma"] == (1.0, 1.0)
    assert params["no_auto_bright"] is True
    assert params["adjust_maximum_thr"] == 0.0
    assert params["output_bps"] == 16
    assert params["user_wb"] == list(settings.white_balance)
    assert params["user_flip"] == 6
    assert params["user_sat"] == 16383
    assert params["fbdd_noise_reduction"] == "off"
    assert params["median_filter_passes"] == 0
    assert params["highlight_mode"] == "ignore"


def test_raw_other_metadata_field_names() -> None:
    raw = SimpleNamespace(
        other=SimpleNamespace(
            shutter=0.001,
            iso_speed=200,
            aperture=8.0,
            timestamp=1_700_000_000,
            focal_len=400.0,
        )
    )
    result = metadata_from_open_raw(raw, Path("sample.NEF"))
    assert result.shutter_s == 0.001
    assert result.iso == 200
    assert result.aperture == 8.0
    assert result.timestamp is not None
    assert result.focal_length_mm == 400.0
