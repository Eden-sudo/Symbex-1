"""Keep documented build/source links and fixture limitations honest."""
from tiny_runtime_support import LM, ROOT


def test_runtime_documentation():
    text=(LM/"chat_runtime.md").read_text()
    for name in ("chat_host", "chat_loop", "chat_sample", "chat_penalty", "chat_stop", "chat_pieces"):
        assert (LM/"c"/(name+".c")).is_file()
        assert name in text
    for suffix in ("rng","sample","penalty","stop","format","memory","loop","pieces"):
        assert (ROOT/"tests"/f"test_tiny_runtime_{suffix}.py").is_file()
    assert "No board performance is claimed" in text
    assert "It is not a C tokenizer" in text
    assert "inference was verified only with a stub" in text


def test_round_two_documented_contracts():
    text = (LM / "chat_runtime.md").read_text()
    for suffix in ("data", "train", "synthetic", "cache", "serial"):
        name = f"chat_{suffix}"
        assert (LM / f"{name}.py").is_file()
        assert name in text
        assert (ROOT / "tests" / f"test_tiny_{name}.py").is_file()
    for contract in ("mask[:, 1:]", "force_recompute=recompute", "five CPU toy steps",
                     "No production fine-tuning", "C host CLI still resets/replays"):
        assert contract in text
    protocol = (LM / "chat_serial.md").read_text()
    assert "CRC-16/CCITT-FALSE" in protocol and "0x29B1" in protocol
    assert "1038 bytes" in protocol and "not firmware integration" in protocol
    assert (LM / "c/chat_serial.c").is_file()
    assert (LM / "c/chat_serial.h").is_file()
