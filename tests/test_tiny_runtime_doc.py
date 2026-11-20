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
