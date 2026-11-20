import re
from pathlib import Path
from tiny_import_helpers import TINY, load


def test_documented_assets_and_checks_exist():
    text = (TINY / "board_prep.md").read_text()
    tests = re.findall(r"`(test_board_[a-z_]+\.py)`", text)
    assert len(tests) == 6 and len(set(tests)) == 6
    for name in tests:
        assert (Path(__file__).parent / name).is_file()
    for name in ("board_flash_plan.py", "board_ram_plan.py", "board_chat_reference.py",
                 "c/board_chat.h", "c/board_chat.c", "board_chat_run.py"):
        assert f"`{name}`" in text and (TINY / name).is_file()
    for boundary in ("not measured free RAM", "roughly 35 KiB", "score hashes",
                     "GQA/LittleBit", "No board access", "0x310000"):
        assert boundary in text


def test_documented_offline_runner_syntax():
    m = load("board_chat_run")
    args = m.parser().parse_args([
        "--reference", "chat_reference.json", "--recording", "serial.bin",
        "--result", "result.json"])
    assert not args.hardware and not args.flash
    assert str(args.recording) == "serial.bin"
    assert args.port == "/dev/board_s3"
