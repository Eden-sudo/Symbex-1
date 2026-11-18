"""The boolean export's own checks: layer_bits (PyTorch vs the folded thresholds, layer by layer) and the C built with
the sanitizers, on random networks with odd channel counts, both stems and both classifiers."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from compiler_core.export import boolean

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "Symbex1_PIO/lib/lib_symbex"

# (channels per stage, blocks per stage, stem, head): odd counts, a count below 32, one above 64, stride 2 twice
SHAPES = [((3, 17, 33), (1, 1, 1), "thermometer", "binary"),
          ((17, 33, 65), (2, 1, 1), "int8", "binary"),
          ((5, 9, 33), (1, 2, 1), "thermometer", "pool2"),
          ((33, 17, 9), (1, 1, 2), "int8", "pool2")]


@pytest.fixture(scope="module")
def resnet():
    path = REPO / "benchmarks/conv2d/cnn_cifar_resnet8/model.py"
    spec = importlib.util.spec_from_file_location("checks_resnet8_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_net(mod, channels, blocks, stem, head, seed):
    """a BoolResNet8 of this shape with random BatchNorms: every sign of the scale (zero too) and thresholds that
    fall inside the range of raw, so the bits are not all equal"""
    torch.manual_seed(seed)
    model = mod.BoolResNet8(levels=4, channels=channels, blocks=blocks, stem=stem, head=head).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                n = m.num_features
                m.running_mean.normal_(0, 3)
                m.running_var.uniform_(0.5, 9)
                m.weight.copy_(torch.randn(n) * torch.where(torch.rand(n) < 0.1, 0.0, 1.0))
                m.bias.normal_(0, 0.5)
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_(0, 1)
                m.running_var.uniform_(0.5, 2)
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.5)
    return model


@pytest.mark.parametrize("channels,blocks,stem,head", SHAPES)
def test_random_shapes_exact_and_sanitized(resnet, tmp_path, channels, blocks, stem, head):
    model = random_net(resnet, channels, blocks, stem, head, seed=sum(channels))
    x = np.random.default_rng(sum(channels)).integers(0, 256, size=(24, 32, 32, 3), dtype=np.uint8)
    net = boolean.fold(model)
    for b in boolean.layer_bits(model, net, x):
        assert b["mismatched"] <= boolean.LAYER_BITS_LIMIT * b["bits"], b
    with torch.no_grad():
        p_torch = model(torch.as_tensor(x).permute(0, 3, 1, 2)).argmax(1).numpy()
    p_sim = boolean.simulate(net, x)
    assert (p_sim == p_torch).mean() >= 0.9
    header = tmp_path / "model_symbex.h"
    boolean.write_header(net, header)
    eq, tot, msg = boolean.validate_c(header, LIB, x, p_sim, sanitize=True)
    assert eq == tot == len(x), msg
    if "symbex_bool_net_run_vertical" in header.read_text():
        for prepared in (False, True, "compact"):
            eq, tot, msg = boolean.validate_c(header, LIB, x, p_sim, vertical=True, prepared=prepared, sanitize=True)
            assert eq == tot == len(x), (prepared, msg)


def test_layer_bits_points_at_the_wrong_layer(resnet):
    """a threshold moved in one layer shows in that layer only: each layer is checked on PyTorch's own input"""
    model = random_net(resnet, (17, 33, 65), (1, 1, 1), "thermometer", "binary", seed=7)
    x = np.random.default_rng(7).integers(0, 256, size=(16, 32, 32, 3), dtype=np.uint8)
    net = boolean.fold(model)
    net["blocks"][1]["conv"]["thr"] = net["blocks"][1]["conv"]["thr"] + 3
    bad = {b["layer"]: b["mismatched"] / b["bits"] for b in boolean.layer_bits(model, net, x)}
    assert bad["block 1"] > boolean.LAYER_BITS_LIMIT
    assert all(v <= boolean.LAYER_BITS_LIMIT for k, v in bad.items() if k != "block 1"), bad


def test_sanitizer_finds_a_short_buffer(resnet, tmp_path):
    """a map buffer one word too short: the sanitized build reports it (a plain build may still give equal results)"""
    model = random_net(resnet, (17, 33, 65), (1, 1, 1), "thermometer", "binary", seed=3)
    x = np.random.default_rng(3).integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    net = boolean.fold(model)
    header = tmp_path / "model_symbex.h"
    boolean.write_header(net, header)
    text = header.read_text()
    m = re.search(r"static uint32_t B_MAP1\[SYMBEX_BOOL_MAPW\((\d+)\)\]", text)
    header.write_text(text.replace(m.group(0), f"static uint32_t B_MAP1[SYMBEX_BOOL_MAPW({int(m.group(1)) - 1})]"))
    eq, _, msg = boolean.validate_c(header, LIB, x, boolean.simulate(net, x), sanitize=True)
    assert eq is None and "sanitizer" in msg, msg
