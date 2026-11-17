"""flash(): builds and uploads by default, upload=False only builds (no board needed).
No network, no board, no real pio: shutil.which and subprocess.run are replaced."""
import types

import pytest

import compiler_core.chip.core as core


def _fake_pio(monkeypatch, returncode=0, found=True):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(core.shutil, "which", lambda name: "/usr/bin/pio" if found else None)
    monkeypatch.setattr(core.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("kwargs", [{}, {"upload": True}])
def test_upload_default_runs_pio_run_with_upload_target(monkeypatch, kwargs):
    calls = _fake_pio(monkeypatch)
    core.flash("/tmp/proj", verbose=False, **kwargs)
    assert calls == [(["pio", "run", "-d", "/tmp/proj", "--target", "upload"], {"capture_output": True})]


def test_build_only_runs_pio_run_without_upload_target(monkeypatch, capsys):
    calls = _fake_pio(monkeypatch)
    core.flash("/tmp/proj", verbose=False, upload=False)
    assert calls == [(["pio", "run", "-d", "/tmp/proj"], {"capture_output": True})]
    calls.clear()
    core.flash("/tmp/proj", verbose=True, upload=False)
    assert calls == [(["pio", "run", "-d", "/tmp/proj"], {"capture_output": False})]
    out = capsys.readouterr().out
    assert "building" in out and "flashing" not in out


def test_upload_failure_message_unchanged(monkeypatch):
    _fake_pio(monkeypatch, returncode=1)
    with pytest.raises(RuntimeError) as e:
        core.flash("/tmp/proj", verbose=False, upload=True)
    assert str(e.value) == "chip: build or upload failed (run `pio run -d /tmp/proj -t upload` to see it)"


def test_build_only_failure_says_build(monkeypatch):
    _fake_pio(monkeypatch, returncode=2)
    with pytest.raises(RuntimeError) as e:
        core.flash("/tmp/proj", verbose=False, upload=False)
    assert str(e.value) == "chip: build failed (run `pio run -d /tmp/proj` to see it)"
    assert "upload" not in str(e.value)


@pytest.mark.parametrize("upload", [True, False])
def test_missing_pio_raises_before_running(monkeypatch, upload):
    calls = _fake_pio(monkeypatch, found=False)
    with pytest.raises(RuntimeError) as e:
        core.flash("/tmp/proj", verbose=False, upload=upload)
    assert str(e.value) == "chip: PlatformIO (pio) not found"
    assert calls == []


def test_upload_is_keyword_only(monkeypatch):
    _fake_pio(monkeypatch)
    with pytest.raises(TypeError):
        core.flash("/tmp/proj", False, "", False)
