import pytest
from tiny_import_helpers import load


def test_alignment_sums_and_header():
    m = load("board_ram_plan")
    sizes = dict(weights=17, kv=33, luts=3, activations=19, scratch=5)
    p = m.plan(sizes, sram_bytes=1024, sram_reserved=64)
    assert p["regions"]["kv"]["offset"] == 32
    assert p["arenas"]["psram"]["used"] == 80
    assert p["arenas"]["sram"]["used"] == 64
    for arena, info in p["arenas"].items():
        assert info["used"] == info["payload"] + info["padding"]
        assert info["used"] + info["reserved"] + info["free"] == info["capacity"]
        end = 0
        for region in p["regions"].values():
            if region["arena"] == arena:
                assert region["offset"] % 16 == 0 and region["offset"] >= end
                end = region["offset"] + region["size"]
    assert "#define BOARD_PSRAM_KV_OFFSET 32u" in m.c_header(p)
    assert "#define BOARD_SRAM_SCRATCH_BYTES 5u" in m.c_header(p)


def test_existing_architecture_numbers():
    m = load("board_ram_plan")
    arch = dict(dim=384, layers=8, heads=6, hidden=1024, seq=256, vocab=4096,
                weights="binary", head="int4")
    b = load("memory_budget").budget(**arch)
    sizes = m.legacy_sizes(**arch)
    assert sizes["weights"] == b["model_bin_bytes"]
    assert sizes["kv"] == b["kv_cache_bytes"]
    assert sum(sizes.values()) == sum(b[k] for k in
        ("model_bin_bytes", "kv_cache_bytes", "per_token_internal_ram_bytes"))
    assert m.plan(sizes, sram_bytes=256*1024)["arenas"]["psram"]["free"] > 0


def test_reject_and_exact_fit():
    m = load("board_ram_plan")
    sizes = dict(weights=16, kv=0, luts=0, activations=16, scratch=0)
    assert m.plan(sizes, sram_bytes=16, psram_bytes=16)["arenas"]["sram"]["free"] == 0
    for kw in (dict(sram_bytes=15), dict(sram_bytes=16, sram_reserved=1),
               dict(sram_bytes=16, alignment=3), dict(sram_bytes=True),
               dict(sram_bytes=16, psram_bytes=8*1024*1024+1)):
        with pytest.raises(ValueError):
            m.plan(sizes, **kw)
    with pytest.raises(ValueError):
        m.plan(dict(sizes, scratch=-1), sram_bytes=1024)
