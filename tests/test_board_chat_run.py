import json
import pytest
from tiny_import_helpers import load

REF = dict(reference_id=42, vocab=8, max_tokens=2, expected_ids=[6, 6], stop_reason=3)
RECORDING = b"boot: synthetic fixture\r\nCHAT1 BEGIN 42\nCHAT1 TOKEN 0 6\nCHAT1 TOKEN 1 6\nCHAT1 END 3 2\n"


def test_chunks_and_result(tmp_path):
    m = load("board_chat_run")
    d = m.Decoder(REF)
    for byte in RECORDING:
        d.feed(bytes([byte]))
    assert d.result("recording")["passed"]
    assert not d.result("recording")["board_verified"]
    assert not d.result("recording")["output_hashes_verified"]
    ref, recording, out = [tmp_path / p for p in ("ref.json", "serial.bin", "result.json")]
    ref.write_text(json.dumps(REF))
    recording.write_bytes(RECORDING)
    args = ["--reference", str(ref), "--recording", str(recording), "--result", str(out)]
    assert m.main(args) == 0
    assert json.loads(out.read_text())["token_ids"] == [6, 6]
    recording.write_bytes(RECORDING.replace(b"TOKEN 1 6", b"TOKEN 1 7"))
    assert m.main(args) == 1
    recording.write_bytes(RECORDING.split(b"CHAT1 END")[0])
    assert m.main(args) == 1
    assert not json.loads(out.read_text())["complete"]
    recording.write_bytes(b"CHAT1 BEGIN 99\n")
    assert m.main(args) == 1
    assert "mismatch" in json.loads(out.read_text())["error"]


@pytest.mark.parametrize("data", [b"CHAT1 TOKEN 0 6\n", b"CHAT1 BEGIN 99\n",
    b"CHAT1 BEGIN 42\nCHAT1 TOKEN 1 6\n", b"CHAT1 BEGIN 42\nCHAT1 END 3 1\n",
    b"CHAT1 BEGIN 42\nCHAT1 TOKEN 0 8\n", RECORDING + b"CHAT1 END 3 2\n",
    b"x" * 513, b"CHAT1 BEGIN 42\nERROR\n"])
def test_reject_frames(data):
    with pytest.raises(ValueError):
        load("board_chat_run").Decoder(REF).feed(data)


def test_arguments_and_flash_plan_only(tmp_path):
    m = load("board_chat_run")
    args = m.parser().parse_args(["--reference", "r", "--result", "o", "--hardware"])
    assert args.port == "/dev/board_s3" and args.flash == []
    image = tmp_path / "model.bin"
    image.write_bytes(b"fake")
    command = m.flash_command([f"0x310000:{image}"], args.port)
    assert "pio" not in command and command[-2:] == ["0x310000", str(image)]
    for offsets in ([0x310001], [0x1000000], [0x310000, 0x310000]):
        with pytest.raises(ValueError):
            m.flash_command([f"{offset}:{image}" for offset in offsets], args.port)
    with pytest.raises(SystemExit):
        m.main(["--reference", "r", "--result", "o", "--recording", "f", "--flash", "0:x"])
