"""tools/lowbit: the folded sequence reproduces the model, the requantization is ESP-NN's bit for bit, the integer
simulation follows Brevitas' fake-quant model, and the exported C (ESP-NN reference kernels) equals the simulation."""
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("brevitas")

from lowbit.export_c import check_c, find_esp_nn, input_bytes, write_header  # noqa: E402
from lowbit.quantize import QNet, export, fold, quantize_input, quantize_multiplier, requantize, run_integer  # noqa: E402
from lowbit.study import calibrate  # noqa: E402


def _ref_requantize(x, mult, shift):
    """esp_nn_multiply_by_quantized_mult transcribed with Python integers (common_functions.h)"""
    left, right = max(shift, 0), max(-shift, 0)
    a = ((x * (1 << left) + (1 << 31)) % (1 << 32)) - (1 << 31)          # int32 multiply
    nudge = (1 - (1 << 30)) if (a < 0) ^ (mult < 0) else (1 << 30)
    v = a * mult + nudge
    r = (v + ((1 << 31) - 1 if v < 0 else 0)) >> 31
    mask = (1 << right) - 1
    q = r >> right
    return q + (1 if (r & mask) > (mask >> 1) + (q < 0) else 0)


def test_requantize_is_esp_nn():
    rng = np.random.default_rng(0)
    acc = rng.integers(-2 ** 31, 2 ** 31, 3000).tolist() + rng.integers(-300000, 300000, 3000).tolist()
    ms = [quantize_multiplier(float(v)) for v in 10 ** rng.uniform(-6, 0.5, len(acc))]
    got = requantize(np.array(acc), np.array([q for q, _ in ms]), np.array([s for _, s in ms]))
    want = [_ref_requantize(a, q, s) for a, (q, s) in zip(acc, ms)]
    assert got.tolist() == want


def _nets():
    torch.manual_seed(0)
    nets = {
        "mlp": (nn.Sequential(nn.Flatten(), nn.Linear(64, 37, bias=False), nn.BatchNorm1d(37), nn.ReLU(),
                              nn.Linear(37, 10)), (200, 1, 8, 8)),
        "conv1d": (nn.Sequential(nn.Conv1d(3, 13, 5, padding=2), nn.BatchNorm1d(13), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(13, 11, 3, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(11 * 7, 6)),
                   (200, 3, 32)),
        "conv2d": (nn.Sequential(nn.Conv2d(3, 9, 3, padding=1), nn.BatchNorm2d(9), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(9, 14, 3), nn.ReLU(), nn.Flatten(), nn.Linear(14 * 4, 4)), (200, 3, 8, 8)),
    }
    for net, _ in nets.values():
        for m in net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.uniform_(-.5, .5)
                m.running_var.uniform_(.5, 2)
                m.weight.data.uniform_(-1.5, 1.5)
    return nets


def test_fold_rejects_a_residual_network():
    class Res(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(8, 8), nn.Linear(8, 8)

        def forward(self, x):
            return self.b(self.a(x)) + x

    with pytest.raises(SystemExit):
        fold(Res(), torch.randn(4, 8))


@pytest.mark.parametrize("name", ["mlp", "conv1d", "conv2d"])
@pytest.mark.parametrize("bits", [8, 4, 2])
def test_integer_follows_fake_quant_and_c_is_exact(name, bits, tmp_path):
    net, shape = _nets()[name]
    x = torch.randn(*shape)
    q = QNet(fold(net, x[:16]), bits, "learned")
    calibrate(q, x)
    with torch.no_grad():
        fq = q(x).numpy()
    layers = export(q)
    qmax = 2 ** (bits - 1) - 1
    for c in layers:
        if "W" in c:
            assert c["W"].min() >= -qmax and c["W"].max() <= qmax
    out = run_integer(layers, quantize_input(layers, x.numpy()))
    steps = np.abs((out - layers[-1]["zero_out"]) * layers[-1]["s_out"] - fq) / layers[-1]["s_out"]
    # the fixed-point and the float requantization round apart by one step now and then, and a step in one layer can
    # move the next: close, not equal (what must be equal is the C against the simulation, below)
    assert np.median(steps) <= 1.0 + 1e-6 and steps.max() <= 8.0
    if find_esp_nn() is None:
        pytest.skip("ESP-NN sources not found (ESP_NN_PATH): C check skipped")
    sim = out.reshape(len(out), -1).argmax(1)
    header = tmp_path / "model_symbex.h"
    write_header(layers, x.shape[1:], str(header), bits)
    xb = input_bytes(layers, x.numpy())
    for flags in ((), ("-DSYMBEX_ROM=",)):
        equal, msg = check_c(str(header), xb, sim, flags)
        assert equal == len(sim), msg
