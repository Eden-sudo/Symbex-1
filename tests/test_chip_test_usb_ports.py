import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import chip_test


def fake_ports(monkeypatch, ports):
    calls = []

    def comports():
        calls.append(True)
        return ports

    serial = types.ModuleType("serial")
    tools = types.ModuleType("serial.tools")
    list_ports = types.ModuleType("serial.tools.list_ports")
    serial.tools = tools
    tools.list_ports = list_ports
    list_ports.comports = comports
    monkeypatch.setitem(sys.modules, "serial", serial)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports)
    return calls


def mac_ports():
    return [types.SimpleNamespace(device="/dev/cu.debug-console", vid=None),
            types.SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", vid=None)]


@pytest.mark.parametrize("vid", [0x1234, 0])
def test_only_usb_port(monkeypatch, vid):
    calls = fake_ports(monkeypatch, mac_ports() + [types.SimpleNamespace(device="/dev/cu.usbmodem", vid=vid)])
    assert chip_test.resolve_port(None) == "/dev/cu.usbmodem"
    assert len(calls) == 1


@pytest.mark.parametrize("ports", [[], mac_ports()])
def test_no_usb_ports(monkeypatch, ports):
    calls = fake_ports(monkeypatch, ports)
    with pytest.raises(ValueError) as exc:
        chip_test.resolve_port(None)
    assert str(exc.value) == "no serial ports found; specify --port."
    assert len(calls) == 1


def test_multiple_usb_ports(monkeypatch):
    debug, bluetooth = mac_ports()
    ports = [debug, types.SimpleNamespace(device="COM4", vid=1), bluetooth,
             types.SimpleNamespace(device="COM3", vid=2)]
    calls = fake_ports(monkeypatch, ports)
    with pytest.raises(ValueError) as exc:
        chip_test.resolve_port(None)
    assert str(exc.value) == "multiple serial ports found: COM4, COM3; specify --port."
    assert len(calls) == 1


def test_explicit_non_usb_port(monkeypatch):
    calls = fake_ports(monkeypatch, [])
    def unexpected():
        pytest.fail("enumerator called")
    monkeypatch.setattr(sys.modules["serial.tools.list_ports"], "comports", unexpected)
    assert chip_test.resolve_port("/dev/cu.debug-console") == "/dev/cu.debug-console"
    assert not calls


@pytest.mark.parametrize("board", ["pico2_arm", "pico2_riscv"])
@pytest.mark.parametrize("kernels", [True, False])
def test_pico2_no_usb_port(monkeypatch, tmp_path, board, kernels):
    (tmp_path / "model_symbex.h").write_text("#define SYMBEX_MLP_N_INPUTS 1\n")
    (tmp_path / "validation_x.u8").write_bytes(b"\x00")
    monkeypatch.setattr(chip_test, "load_validation", lambda *args: ([0], [0], [0]))
    def unexpected(*args, **kwargs):
        pytest.fail("consumer called")
    monkeypatch.setattr(chip_test, "run_kernels", unexpected)
    monkeypatch.setattr(chip_test, "measure_on_chip", unexpected)
    monkeypatch.setattr(chip_test, "save_profile", unexpected)
    calls = fake_ports(monkeypatch, mac_ports())
    monkeypatch.setattr(sys, "argv", ["chip_test.py", "--board", board] +
                        (["--kernels"] if kernels else [str(tmp_path)]))
    with pytest.raises(SystemExit) as exc:
        chip_test.main()
    if kernels:  # chip_test.py refuses the kernel benchmark on the Pico 2 before it looks for a port
        assert exc.value.code.startswith("[X] the kernel benchmark is for the ESP32 boards")
        assert len(calls) == 0
    else:
        assert exc.value.code == "[X] no serial ports found; specify --port."
        assert len(calls) == 1
    assert not list(tmp_path.glob("*.json"))
