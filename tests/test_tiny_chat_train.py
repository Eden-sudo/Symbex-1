import pytest
import torch
from torch.nn import functional as F
from tiny_runtime_support import module


def test_five_cpu_steps_and_manual_loss():
    train, chat = module("train"), module("chat_train")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            base = train.TinyLM(16, 8, 1, 2, 16, 8, "binary")
            model = chat.ChatTrainingModel(base)
            ids = torch.tensor([[1, 3, 4, 5, 2, 6], [1, 7, 8, 9, 10, 2]])
            mask = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.bool)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            before = base.emb.weight.detach().clone()
            for _ in range(5):
                optimizer.zero_grad()
                logits, loss = model(ids, mask)
                manual = torch.stack([-logits[b, t-1].log_softmax(-1)[ids[b, t]]
                                      for b in range(2) for t in range(1, 6) if mask[b, t]]).mean()
                torch.testing.assert_close(loss, manual)
                logits.retain_grad()
                loss.backward()
                assert torch.count_nonzero(logits.grad[:, :-1][~mask[:, 1:]]) == 0
                assert torch.count_nonzero(logits.grad[:, -1]) == 0
                assert torch.count_nonzero(logits.grad[:, :-1][mask[:, 1:]]) > 0
                optimizer.step()
            assert not torch.equal(before, base.emb.weight)
            logits, ordinary = model(ids)
            torch.testing.assert_close(ordinary, F.cross_entropy(logits[:, :-1].reshape(-1, 16), ids[:, 1:].reshape(-1)))
            labels = ids.clone()
            labels[~mask] = -100
            torch.testing.assert_close(chat.causal_loss(logits, labels, mask), model(ids, mask)[1])
    finally:
        torch.set_num_threads(previous)


def test_reject_empty_or_misaligned_mask():
    chat = module("chat_train")
    ids, logits = torch.tensor([[1, 2, 3]]), torch.zeros(1, 3, 4)
    for mask in (torch.zeros_like(ids, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool),
                 torch.ones_like(ids), torch.tensor([[True, False, False]])):
        with pytest.raises(ValueError):
            chat.causal_loss(logits, ids, mask)
