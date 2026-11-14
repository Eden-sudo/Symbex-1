import builtins
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import chip_test


def fake_ports(monkeypatch, names, error=None):
    calls = []

    def comports():
        calls.append(True)
        if error:
            raise error
        return [types.SimpleNamespace(device=name) for name in names]

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


def block_serial(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "serial" or name.startswith("serial."):
            raise ImportError("serial unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("name", ["/dev/ttyUSB0", "/dev/cu.usbserial-123", "/dev/cu.usbmodem123", "COM3"])
def test_choose_one(name):
    names = [name]
    assert chip_test.choose_port(names) == name
    assert names == [name]


@pytest.mark.parametrize("names,message", [
    ([], "No serial ports found; specify --port."),
    (["COM3", "/dev/x"], "Multiple serial ports found: COM3, /dev/x; specify --port."),
])
def test_choose_errors(names, message):
    with pytest.raises(ValueError) as exc:
        chip_test.choose_port(names)
    assert str(exc.value) == message


@pytest.mark.parametrize("port,expected", [
    ("/dev/x", "/dev/x"), ("ttyUSB0", "/dev/ttyUSB0"),
    ("COM3", "COM3"), ("com3", "com3"), ("", ""),
])
def test_explicit_port_does_not_import_serial(monkeypatch, port, expected):
    block_serial(monkeypatch)
    assert chip_test.resolve_port(port) == expected


@pytest.mark.parametrize("names,expected", [
    ([], "No serial ports found; specify --port."),
    (["COM3"], "COM3"),
    (["COM3", "COM4"], "Multiple serial ports found: COM3, COM4; specify --port."),
])
def test_discovery(monkeypatch, names, expected):
    calls = fake_ports(monkeypatch, names)
    if len(names) == 1:
        assert chip_test.resolve_port(None) == expected
    else:
        with pytest.raises(ValueError) as exc:
            chip_test.resolve_port(None)
        assert str(exc.value) == expected
    assert len(calls) == 1


def test_import_error(monkeypatch):
    block_serial(monkeypatch)
    with pytest.raises(RuntimeError, match="Cannot list serial ports: serial unavailable; specify --port"):
        chip_test.resolve_port(None)


def test_discovery_oserror(monkeypatch):
    calls = fake_ports(monkeypatch, [], OSError("unavailable"))
    with pytest.raises(RuntimeError, match="Cannot list serial ports: unavailable; specify --port"):
        chip_test.resolve_port(None)
    assert len(calls) == 1


@pytest.mark.parametrize("kernels", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_main_passes_port(monkeypatch, tmp_path, kernels, explicit):
    (tmp_path / "model_symbex.h").write_text("#define SYMBEX_MLP_N_INPUTS 1\n")
    (tmp_path / "validation_x.u8").write_bytes(b"\x00")
    monkeypatch.setattr(chip_test, "load_validation", lambda *args: ([0], [0], [0]))
    calls = []
    monkeypatch.setattr(chip_test, "run_kernels", lambda *args, **kwargs: calls.append(args[1]) or {})
    monkeypatch.setattr(chip_test, "measure_on_chip", lambda *args, **kwargs: calls.append(args[7]) or {})
    ports = fake_ports(monkeypatch, ["COM4"])
    argv = ["chip_test.py", "--no-upload"] + (["--kernels"] if kernels else [str(tmp_path)])
    if explicit:
        argv += ["--port", "ttyUSB0"]
    monkeypatch.setattr(sys, "argv", argv)
    chip_test.main()
    assert calls == (["/dev/ttyUSB0"] if explicit else ["COM4"])
    assert len(ports) == (0 if explicit else 1)


@pytest.mark.parametrize("kernels", [True, False])
@pytest.mark.parametrize("names,message", [
    ([], "[X] No serial ports found; specify --port."),
    (["COM3", "COM4"], "[X] Multiple serial ports found: COM3, COM4; specify --port."),
])
def test_main_selection_errors(monkeypatch, tmp_path, kernels, names, message):
    (tmp_path / "model_symbex.h").write_text("#define SYMBEX_MLP_N_INPUTS 1\n")
    (tmp_path / "validation_x.u8").write_bytes(b"\x00")
    monkeypatch.setattr(chip_test, "load_validation", lambda *args: ([0], [0], [0]))
    def unexpected(*args, **kwargs):
        pytest.fail("consumer called")
    monkeypatch.setattr(chip_test, "run_kernels", unexpected)
    monkeypatch.setattr(chip_test, "measure_on_chip", unexpected)
    monkeypatch.setattr(chip_test, "save_profile", unexpected)
    fake_ports(monkeypatch, names)
    monkeypatch.setattr(sys, "argv", ["chip_test.py"] + (["--kernels"] if kernels else [str(tmp_path)]))
    with pytest.raises(SystemExit) as exc:
        chip_test.main()
    assert exc.value.code == message
    assert not list(tmp_path.glob("*.json"))


def test_help_without_serial(monkeypatch, capsys):
    block_serial(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["chip_test.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        chip_test.main()
    assert exc.value.code == 0
    assert "use the only serial port found" in capsys.readouterr().out


@pytest.mark.parametrize("args,message", [
    ([], "[X] give the output folder"),
    (["--kernels", "--board", "uno"], "[X] the kernel benchmark is for the ESP32 boards"),
])
def test_existing_errors_precede_discovery(monkeypatch, args, message):
    block_serial(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["chip_test.py"] + args)
    with pytest.raises(SystemExit) as exc:
        chip_test.main()
    assert exc.value.code.startswith(message)
