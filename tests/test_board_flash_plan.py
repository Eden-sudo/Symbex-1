import csv
import json

import pytest

from tiny_import_helpers import load


def test_layout_and_cli(tmp_path):
    m = load("board_flash_plan")
    result = m.plan({"model": 4097, "pieces": 17})
    assert result["offsets"]["model"] == 0x310000
    assert result["offsets"]["pieces"] == 0x312000
    rows = list(csv.reader(m.partitions_csv(result).splitlines()[1:]))
    assert rows[1][:5] == ["app", "app", "factory", "0x10000", "0x300000"]
    for left, right in zip(result["partitions"], result["partitions"][1:]):
        assert left["offset"] + left["size"] <= right["offset"]
    assert all(r["offset"] % 4096 == r["size"] % 4096 == 0 for r in result["partitions"])
    source, target, offsets = [tmp_path / n for n in ("in.json", "partitions.csv", "out.json")]
    source.write_text('{"model":4097,"pieces":17}')
    m.main([str(source), "--csv", str(target), "--offsets", str(offsets)])
    assert json.loads(offsets.read_text()) == result
    assert target.read_text() == m.partitions_csv(result)


def test_exact_fit_and_overflow():
    m = load("board_flash_plan")
    size = 16 * 1024 * 1024 - 0x310000
    assert m.plan({"model": size})["free_bytes"] == 0
    with pytest.raises(ValueError, match="exceed"):
        m.plan({"model": size + 1})


@pytest.mark.parametrize("sizes", [{}, {"pieces": 4}, {"model": 0}, {"model": -1},
                                  {"model": True}, {"model": 1.5},
                                  {"model": 1, "app": 4}, {"model": 1, "bad,name": 1}])
def test_reject_sizes(sizes):
    with pytest.raises(ValueError):
        load("board_flash_plan").plan(sizes)
