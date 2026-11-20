"""Host GQA forward feeds Python/C samplers and streaming piece decoders."""
import ctypes
import subprocess
import sys

import pytest
import torch

from tiny_import_helpers import TINY, load


class StubTokenizer:
    """Small explicit vocabulary; no claim of a trained chat tokenizer."""
    def unk_id(self): return 0
    def bos_id(self): return 1
    def eos_id(self): return 2
    def encode(self, text):
        return {"User:": [3], "hello": [4], "Assistant:": [5]}[text]


class CDecoder(ctypes.Structure):
    _fields_ = [("offsets", ctypes.c_void_p), ("payload", ctypes.c_void_p),
                ("vocab", ctypes.c_uint32), ("started", ctypes.c_int)]


@pytest.mark.parametrize("temperature,top_p", [(0, 65536), (0xffffffff, 49152)])
def test_chat_gqa_sampler_stream(tmp_path, monkeypatch, temperature, top_p):
    train = load("train")
    monkeypatch.setitem(sys.modules, "train", train)
    integer = load("intsim")
    monkeypatch.setitem(sys.modules, "intsim", integer)
    gqa, sample, pieces = load("gqa_intsim"), load("sample_int"), load("chat_pieces")
    library = tmp_path / "chat.so"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(TINY / "c/chat_sample.c"),
                    str(TINY / "c/chat_pieces.c"), "-o", str(library)],
                   check=True, capture_output=True, timeout=15)
    c = ctypes.CDLL(str(library))
    c.chat_sample.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
                             ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                             ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    c.chat_decoder_init.argtypes = [ctypes.POINTER(CDecoder), ctypes.c_void_p, ctypes.c_size_t]
    c.chat_decode.argtypes = [ctypes.POINTER(CDecoder), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    blob = pieces.pack([b"?", b"", b"", b" User:", b" hello", b" Assistant:", b" world", b"!"])
    raw, decoder = ctypes.create_string_buffer(blob), CDecoder()
    assert c.chat_decoder_init(ctypes.byref(decoder), raw, len(blob)) == 0
    py_decoder = pieces.Decoder(blob)
    py_ids = load("chat_format").build_chat(StubTokenizer(), [("user", "hello")])
    assert py_ids == [1, 3, 4, 2, 5]
    c_ids = list(py_ids)
    py_state, c_state = 42, ctypes.c_uint32(42)
    scratch = (ctypes.c_int * 8)()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(29)
            model = train.TinyLM(8, 128, 1, 2, 32, 16, "binary", kv_heads=1).eval()
        params = gqa.integerize(model, "binary")
        assert params["kv_heads"] == 1 and params["heads"] == 2
        py_chunks, c_chunks = [], []
        for _ in range(6):
            # Each side feeds its own selected token back into the actual GQA path.
            py_scores = gqa.forward(params, torch.tensor([py_ids]))[0][0, -1].tolist()
            c_scores = gqa.forward(params, torch.tensor([c_ids]))[0][0, -1].tolist()
            token, py_state = sample.sample(py_scores, 0, temperature, top_p, py_state)
            c_token = c.chat_sample((ctypes.c_int64 * 8)(*c_scores), 8, 0, temperature,
                                    top_p, ctypes.byref(c_state), scratch, 8)
            assert c_token == token and c_state.value == py_state
            py_ids.append(token)
            c_ids.append(c_token)
            py_chunks.append(py_decoder.feed(token))
            ptr, size = ctypes.c_void_p(), ctypes.c_size_t()
            assert c.chat_decode(ctypes.byref(decoder), c_token, ctypes.byref(ptr), ctypes.byref(size)) == 0
            c_chunks.append(ctypes.string_at(ptr, size.value))
            assert c_chunks[-1] == py_chunks[-1]
        assert py_ids == c_ids
        assert b"".join(py_chunks) == b"".join(c_chunks)
        assert py_state == (42 if temperature == 0 else sample.xorshift32(
            sample.xorshift32(sample.xorshift32(sample.xorshift32(sample.xorshift32(sample.xorshift32(42)))))))
    finally:
        torch.set_num_threads(previous)
