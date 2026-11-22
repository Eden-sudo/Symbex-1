import pytest
import torch
from tiny_runtime_support import LM, module


@pytest.mark.parametrize("head", ["fp32", "int4", "int2"])
def test_integer_prefix_reuse(monkeypatch, head):
    monkeypatch.syspath_prepend(str(LM))
    train, integer, cache_module = module("train"), module("intsim"), module("chat_cache")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            # Real 64-wide heads, two layers; no checkpoint or training.
            model = train.TinyLM(16, 64, 2, 1, 32, 16, "binary", head=head)
            P = integer.integerize(model, "binary")
            cache = cache_module.PrefixCache(P)
            first = [1, 3, 4]
            for ids, steps in [(first, 3), (first, 0), (first+[5, 2, 6, 7], 4),
                               ([1, 3, 4, 5, 2, 6, 7, 8], 1), ([1, 9, 10], 3), ([1], 1)]:
                z, hm, actual_steps = cache.update(ids)
                full_z, full_hm = integer.forward(P, torch.tensor([ids]))
                assert actual_steps == steps
                assert torch.equal(z, full_z[:, -1:])
                assert torch.equal(hm, full_hm[:, -1:])
            ids, dropped, replay = cache_module.retention_plan([1], [[3, 4], [5, 6]], [7], cache.ids, context=5)
            assert dropped == 1 and replay and ids == [1, 5, 6, 7]
            z, hm, steps = cache.update(ids, force_recompute=replay)
            expected = integer.forward(P, torch.tensor([ids]))
            assert steps == len(ids) and torch.equal(z, expected[0][:, -1:])
            assert torch.equal(hm, expected[1][:, -1:])
            with pytest.raises(ValueError):
                cache.update([1, 16])
            assert cache.ids == ids
    finally:
        torch.set_num_threads(previous_threads)


def test_policy_reports_append_change_and_eviction(monkeypatch):
    monkeypatch.syspath_prepend(str(LM))
    plan = module("chat_cache").retention_plan
    assert plan([1], [[3, 4]], [5], [1, 3], context=8) == ([1, 3, 4, 5], 0, False)
    assert plan([1], [[3, 4]], [5], [1, 9], context=8)[2]
    assert plan([1], [], [5], [1, 5, 6], context=8)[2]
    # Repeated values cannot conceal an eviction from the policy.
    assert plan([1], [[3], [3]], [], [1, 3], context=3) == ([1, 3], 1, True)
