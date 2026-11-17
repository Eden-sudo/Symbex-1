"""The shared export report counts model storage and summarizes exported rows."""
import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("The export report tests require PyTorch") from exc


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).resolve().parents[1] / "tools/compiler_core/export/report.py"
    spec = importlib.util.spec_from_file_location("export_report_under_test", path)
    assert spec is not None and spec.loader is not None, f"Cannot load export report from {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def model():
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(2, 4, 3, bias=False)),
        ("bn", nn.BatchNorm2d(4)),
        ("fc", nn.Linear(8, 3)),
    ]))


@pytest.fixture
def rows():
    return [
        dict(name="CONV_main", shape="4x2x3x3", n_weights=72, bits=1, weights_bytes=12,
             params_bytes=32, parts={"thresholds": 32}, map_bytes=1024, xnor_words=3),
        dict(name="FC_main", shape="3x8", n_weights=24, bits=2, weights_bytes=8,
             params_bytes=16, parts={"thresholds": 16}, map_bytes=2048, xnor_words=1),
    ]


@pytest.fixture
def header(tmp_path):
    path = tmp_path / "export.h"
    path.write_bytes(b"x" * 1024)
    return path


def line(section, prefix):
    return next(line for line in section.splitlines() if line.strip().startswith(prefix)).split()


def total_bytes(model):
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def test_describe_input_counts_weights_parameters_and_buffers(report, model):
    model.register_parameter("auxiliary", nn.Parameter(torch.zeros(2, 3)))
    model.register_buffer("extra_buffer", torch.zeros(2, 2))
    weights, other = report.describe_input(model)
    expected = [(name, tuple(t.shape), t.numel(), t.numel() * t.element_size())
                for name, t in [("conv.weight", model.conv.weight), ("fc.weight", model.fc.weight)]]
    assert [(name, shape, count) for name, shape, count, _ in weights] == [
        ("conv.weight", (4, 2, 3, 3), 72), ("fc.weight", (3, 8), 24),
    ]
    assert weights == expected
    other_tensors = [model.bn.weight, model.bn.bias, model.fc.bias, model.bn.running_mean,
                     model.bn.running_var, model.bn.num_batches_tracked, model.auxiliary, model.extra_buffer]
    assert other == sum(t.numel() * t.element_size() for t in other_tensors)
    assert sum(w[3] for w in weights) + other == total_bytes(model)


def test_print_report_totals_padding_and_columns(report, model, rows, header, capsys):
    report.print_report(model, rows, header)
    output = capsys.readouterr().out
    input_section = output.split("INPUT MODEL", 1)[1].split("OUTPUT", 1)[0]
    output_section = output.split("OUTPUT  (header for the engine)", 1)[1].split("MEMORY", 1)[0]
    memory = output.split("MEMORY", 1)[1]
    assert line(input_section, "TOTAL") == ["TOTAL", "96", f"{total_bytes(model) / 1024:.2f}"]
    for row in rows:
        assert line(output_section, row["name"]) == [
            row["name"], row["shape"], "25%", f"{row['weights_bytes'] / 1024:.2f}",
            f"{row['params_bytes'] / 1024:.2f}", f"{(row['weights_bytes'] + row['params_bytes']) / 1024:.2f}",
        ]
    assert line(output_section, "TOTAL") == ["TOTAL", "0.02", "0.05", "0.07"]
    assert "1.0 KB" in " ".join(line(memory, "header file"))
    ram = " ".join(line(memory, "RAM"))
    assert "2.0 KB" in ram and "largest: FC_main" in ram
    assert "[!]" not in output and "* not exported" not in output
    assert "ROM BY ARRAY" not in output and "BY STAGE" not in output


def test_print_report_detail_optional_fields(report, model, rows, header, capsys):
    report.print_report(model, rows, header, detail=True)
    output = capsys.readouterr().out
    arrays = output.split("ROM BY ARRAY", 1)[1].split("BY STAGE", 1)[0]
    stages = output.split("BY STAGE", 1)[1]
    assert line(arrays, "weights, packed")[-2] == f"{15 / 1024:.2f}"
    assert line(arrays, "alignment padding")[-2] == f"{5 / 1024:.2f}"
    assert line(arrays, "thresholds")[-2] == f"{48 / 1024:.2f}"
    assert line(stages, "CONV")[-2:] == ["3", "75.0%"]
    assert line(stages, "FC")[-2:] == ["1", "25.0%"]


def test_print_report_marks_skipped_weight(report, model, rows, header, capsys):
    report.print_report(model, rows[:1], header, skipped=("fc.weight",))
    output = capsys.readouterr().out
    input_section = output.split("INPUT MODEL", 1)[1].split("OUTPUT", 1)[0]
    assert line(input_section, "fc.weight*")[0] == "fc.weight*"
    assert line(input_section, "conv.weight")[0] == "conv.weight"
    assert "* not exported (unused at inference)" in input_section
    assert line(input_section, "TOTAL") == ["TOTAL", "96", f"{total_bytes(model) / 1024:.2f}"]
    assert "[!]" not in output

    report.print_report(model, rows[:1], header)
    output = capsys.readouterr().out
    assert "the model has 96 weights to export but 72 were exported" in output
