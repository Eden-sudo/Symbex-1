"""The real four-output FOMO head agrees with its integer and C exports, cell by cell."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from compiler_core.export import boolean

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_SYMBEX = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
MODEL = os.path.join(REPO, "benchmarks", "conv2d", "fomo_voc_person", "model.py")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _model():
    spec = importlib.util.spec_from_file_location("fomo_mc_model", MODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)  # fixed model initialization, independent of other tests
        model = mod.BoolFOMO(channels=(8, 8, 16), blocks=(1, 1, 1), classes=4)
    return model.cpu().eval()


def _validate_c_exact(header, x_u8, expected_grids):
    expected = np.asarray([boolean._grid_checksum(g) for g in expected_grids])
    equal, total, message = boolean.validate_c(header, LIB_SYMBEX, x_u8, expected)
    if (equal, total, message) == (None, 0, "gcc not found: the C exactness check was skipped"):
        pytest.skip(message)
    assert equal == total == len(x_u8), (equal, total, message)
    assert message == "", (equal, total, message)


def test_fomo_mc_export_exact(tmp_path, rng):
    model = _model()
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for bn in model.modules():
            if isinstance(bn, torch.nn.BatchNorm2d):
                bn.running_mean.copy_(torch.randn(bn.num_features, generator=gen) * 3)
                bn.running_var.copy_(torch.rand(bn.num_features, generator=gen) * 4 + 0.1)
        # Keep the randomized statistics but put head logits on a common scale.
        model.head_bn.weight.copy_(torch.sqrt(model.head_bn.running_var + model.head_bn.eps))
        model.head_bn.bias.copy_(model.head_bn.running_mean)

    x_u8 = rng.integers(0, 256, size=(32, 96, 96, 3), dtype=np.uint8)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_u8).permute(0, 3, 1, 2))
    assert logits.shape == (32, 4, 12, 12)
    expected = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    assert expected.shape == (32, 12, 12)
    assert len(np.unique(expected)) >= 3, "fewer than three classes win any cell"

    net = boolean.fold(model)
    assert net['cells']
    assert len(net['fc_w']) == 4
    actual = boolean.simulate(net, x_u8, batch=8, device='cpu', return_bits=True)
    assert actual.shape == (32, 12, 12)
    np.testing.assert_array_equal(actual, expected)
    header = tmp_path / 'model_symbex.h'
    boolean.write_header(net, header, in_size=96)
    _validate_c_exact(header, x_u8, expected)


def test_fomo_mc_export_first_maximum(tmp_path, rng):
    model = _model()
    with torch.no_grad():
        model.head_bn.weight.zero_()
        model.head_bn.bias.copy_(torch.tensor([0., 1., 1., 1.]))
    x_u8 = rng.integers(0, 256, size=(1, 96, 96, 3), dtype=np.uint8)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_u8).permute(0, 3, 1, 2))
    assert logits.shape == (1, 4, 12, 12)
    np.testing.assert_array_equal(logits.numpy(), np.broadcast_to(
        np.array([0., 1., 1., 1.])[None, :, None, None], logits.shape))
    expected = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    np.testing.assert_array_equal(expected, np.ones((1, 12, 12), dtype=np.uint8))

    net = boolean.fold(model)
    assert net['cells']
    assert len(net['fc_w']) == 4
    actual = boolean.simulate(net, x_u8, batch=8, device='cpu', return_bits=True)
    np.testing.assert_array_equal(actual, expected)
    header = tmp_path / 'model_symbex.h'
    boolean.write_header(net, header, in_size=96)
    _validate_c_exact(header, x_u8, expected)
