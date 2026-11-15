"""The vertical exporter's shape checks, matching check_conv and symbex_bool_vertical_check."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def vertical():
    path = Path(__file__).resolve().parents[1] / "tools/compiler_core/export/vertical.py"
    spec = importlib.util.spec_from_file_location("vertical_for_problem_tests", path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conv(**overrides):
    conv = dict(k=3, s=1, h=16, oh=16, cin=24)
    conv.update(overrides)
    return conv


@pytest.mark.parametrize("convs, levels, in_size", [
    pytest.param([_conv(), _conv(k=1)], 8, 16, id="thermometer"),
    pytest.param([_conv(s=2, h=32, oh=16)], 8, 32, id="stride_2"),
    pytest.param([dict(int8=True, oh=16), _conv()], 11, 24, id="int8_stem"),
    pytest.param([_conv()], 10, 16, id="thermometer_30_bits"),
    pytest.param([_conv(k=1, cin=2040)], 8, 16, id="K_below_limit"),
])
def test_valid_shapes(vertical, convs, levels, in_size):
    assert vertical.vertical_problem(convs, levels, in_size) is None


@pytest.mark.parametrize("convs, levels, in_size, expected", [
    pytest.param([dict(int8=True, oh=24)], 8, 16, "int8 stem output size", id="int8_stem_width_not_dividing_32"),
    pytest.param([dict(int8=True, oh=4)], 8, 16, "int8 stem output size", id="int8_stem_area_not_multiple_of_32"),
    pytest.param([_conv()], 11, 16, "thermometer with more than 32 bits", id="thermometer_over_32_bits"),
    pytest.param([_conv()], 8, 24, "input size", id="input_width_not_dividing_32"),
    pytest.param([_conv()], 8, 4, "input size", id="input_area_not_multiple_of_32"),
    pytest.param([_conv(k=2)], 8, 16, "kernel other than 1 x 1 or 3 x 3", id="unsupported_kernel"),
    pytest.param([_conv(oh=8)], 8, 16, "stride 1 with kernel 3", id="stride_1_changes_size"),
    pytest.param([_conv(k=1, s=2, oh=8)], 8, 16, "stride 2 with kernel 1", id="stride_2_requires_kernel_3"),
    pytest.param([_conv(s=2)], 8, 16, "stride 2 with kernel 3", id="stride_2_requires_halving"),
    pytest.param([_conv(s=3)], 8, 16, "stride 3 with kernel 3", id="unsupported_stride"),
    pytest.param([_conv(h=24, oh=24)], 8, 16, "map size", id="map_width_not_dividing_32"),
    pytest.param([_conv(h=4, oh=4)], 8, 16, "map size", id="map_area_not_multiple_of_32"),
    pytest.param([_conv(s=2, h=8, oh=4)], 8, 16, "map size", id="output_map_area_not_multiple_of_32"),
    pytest.param([_conv(k=1, cin=7)], 8, 16, "k * k * in_ch = 7", id="K_not_multiple_of_8"),
    pytest.param([_conv(k=1, cin=2048)], 8, 16, "k * k * in_ch = 2048", id="K_at_upper_limit"),
    pytest.param([dict(int8=True, oh=16), _conv(k=2)], 11, 24, "kernel other than 1 x 1 or 3 x 3",
                 id="kernel_after_int8_stem"),
])
def test_rejected_shapes(vertical, convs, levels, in_size, expected):
    reason = vertical.vertical_problem(convs, levels, in_size)
    assert isinstance(reason, str)
    assert expected in reason
    if expected.startswith("k * k * in_ch"):
        assert "not a multiple of 8 or not below 2048" in reason


@pytest.mark.parametrize("convs, levels, in_size, expected", [
    pytest.param([_conv()], 11, 24, "thermometer with more than 32 bits", id="thermometer_before_input"),
    pytest.param([_conv(k=2, s=3)], 8, 16, "kernel other than 1 x 1 or 3 x 3", id="kernel_before_stride"),
    pytest.param([_conv(cin=7), _conv(k=2)], 8, 16, "k * k * in_ch = 63", id="first_layer_before_second"),
])
def test_first_failing_rule_wins(vertical, convs, levels, in_size, expected):
    reason = vertical.vertical_problem(convs, levels, in_size)
    assert isinstance(reason, str)
    assert expected in reason
