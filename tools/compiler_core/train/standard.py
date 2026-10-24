"""
Standard training: CrossEntropy + the STE gradient of the binarized layers.
No teacher, no distillation, no gating. This is the loop that experiment confirmed works equally well for
conv and dense layers (once the BatchNorm fix below is applied) -- see fix_batchnorm().

Use this for any model built with methods.apply_method(model, "plain") (or "gated" / "bitslice", once those
exist for conv too: the loop itself does not care which method built the layers, it only calls model(x) and
loss.backward()).

Fixes in SYMBEX v3:
  - evaluate() returns the model to the mode it was in (before: always left it in training mode).
  - checkpoints: save="last" (default) saves the LAST epoch, the fair number to compare;
    save="best" keeps the old behavior (best TEST epoch: optimistic, it selects on the test set).
    save_every really saves every N epochs (before it only printed a message).
  - max_norm: the same gradient-clipping option as distillation.py (default None = no clipping, as before),
    so the two loops can be compared changing ONE variable.

The returned dict has the new keys (best_test, final_average, history) and, until every caller is translated,
the old ones too (mejor_test, promedio_final, historial). See RENAMES.md.
"""
import time

import torch
import torch.nn as nn


def fix_batchnorm(model, momentum=0.3):
    """
    Raises the momentum of every BatchNorm layer in `model` from PyTorch's default (0.1) to `momentum`.

    Why: confirmed experimentally (conv_vs_linear.py) that training a binarized network with the default
    BatchNorm momentum makes its running mean/var lag behind the activation statistics, which change fast
    because weights flip sign often during training. This causes wild, misleading swings in eval-mode (test)
    accuracy while train accuracy climbs smoothly -- in one run it looked like conv and dense learned very
    differently (3.05 points apart); raising momentum to 0.5 made that gap disappear (-0.46 points, i.e.
    noise). This has nothing to do with which method built the layers (plain / gated / bitslice): any
    BatchNorm sitting on top of binarized weights is exposed to the same lag.

    Only touches layers still at the PyTorch default (0.1): if the user's own model already set a custom
    momentum, that choice is respected. Call this once, right after building the model, before training.
    Returns the number of layers changed.
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.momentum == 0.1:
            module.momentum = momentum
            changed += 1
    return changed


@torch.no_grad()
def evaluate(model, loader, device):
    """Accuracy (%) of `model` on `loader`, in eval mode. The model is returned to the mode it was in."""
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    model.train(was_training)
    return 100.0 * correct / total


_SAVE_OLD = {"ultima": "last", "mejor": "best"}


class Checkpoints:
    """Shared checkpoint policy of the training loops.
    save="last": save the current model every `save_every` epochs (if set) and after the last epoch.
    save="best": save whenever the test accuracy improves (old behavior; selects on the test set).
    (The old values "ultima" / "mejor" are still accepted.)"""
    def __init__(self, path, save="last", save_every=None, verbose=True):
        save = _SAVE_OLD.get(save, save)
        if save not in ("last", "best"):
            raise ValueError(f"save must be 'last' or 'best', got {save!r}")
        if save_every is not None and save_every < 1:
            raise ValueError("save_every must be >= 1")
        self.path, self.save, self.every, self.verbose = path, save, save_every, verbose
        self.best = float("-inf")

    def after_epoch(self, model, epoch, epochs, acc_test):
        if self.path is None:
            return
        last = (epoch + 1) == epochs
        if self.save == "best":
            if acc_test > self.best:
                self.best = acc_test
                torch.save(model.state_dict(), self.path)
            if self.every and (epoch + 1) % self.every == 0 and self.verbose:
                print(f"  [*] Checkpoint ({self.path}): best so far {self.best:.2f}% "
                      f"-- the optimizer and the scheduler are NOT restarted")
        elif last or (self.every and (epoch + 1) % self.every == 0):
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"  [*] Checkpoint ({self.path}): epoch {epoch + 1}"
                      + (" (last)" if last else "") + " -- the optimizer and the scheduler are NOT restarted")

    despues_de_epoca = after_epoch              # old name (kept until every caller is translated)


def _summary(history, average_last):
    """Result dict of the training loops, with the old key names too (until every caller is translated)."""
    n = min(average_last, len(history))
    best = max(h[3] for h in history)
    average = sum(h[3] for h in history[-n:]) / n
    return dict(best_test=best, final_average=average, history=history,
                mejor_test=best, promedio_final=average, historial=history)


def train_model(model, trainloader, testloader, epochs=30, lr=1e-3, weight_decay=1e-5,
                device=None, bn_momentum=0.3, average_last=5, verbose=True,
                save_every=None, checkpoint_path=None, save="last", max_norm=None,
                promedio_ultimas=None, guardar_cada=None, ruta_checkpoint=None, guardar=None):
    """
    Standard loop: Adam + CosineAnnealingLR + CrossEntropy.

    Returns a dict with:
      best_test      : the single highest test accuracy seen (noisy: see final_average)
      final_average  : average test accuracy of the last `average_last` epochs (the robust number: use THIS
                       to compare architectures / methods, not best_test -- a single lucky epoch is not a
                       fair comparison)
      history        : list of (epoch, loss, train_acc, test_acc)

    checkpoint_path / save / save_every: see Checkpoints. The optimizer and the LR schedule are never
    restarted (unlike calling train_model() again in a loop from the outside, which resets Adam's state and
    the LR schedule). max_norm: gradient clipping (None = off, as before).
    promedio_ultimas / guardar_cada / ruta_checkpoint / guardar: old names of average_last / save_every /
    checkpoint_path / save (kept until every caller is translated).
    """
    average_last = average_last if promedio_ultimas is None else promedio_ultimas
    save_every = save_every if guardar_cada is None else guardar_cada
    checkpoint_path = checkpoint_path if ruta_checkpoint is None else ruta_checkpoint
    save = save if guardar is None else guardar
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if bn_momentum is not None:
        changed = fix_batchnorm(model, bn_momentum)
        if verbose and changed:
            print(f"[*] BatchNorm momentum set to {bn_momentum} in {changed} layer(s) "
                  f"(they were at PyTorch's default 0.1)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    ckpt = Checkpoints(checkpoint_path, save, save_every, verbose)

    history = []
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            loss_sum += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        acc_train = 100.0 * correct / total
        acc_test = evaluate(model, testloader, device)
        history.append((epoch, loss_sum / len(trainloader), acc_train, acc_test))
        if verbose:
            print(f"  Epoch [{epoch + 1:03d}/{epochs}] | Loss: {loss_sum / len(trainloader):.4f} | "
                  f"Train: {acc_train:5.2f}% | Test: {acc_test:5.2f}% | {time.time() - t0:.1f}s")
        ckpt.after_epoch(model, epoch, epochs, acc_test)

    return _summary(history, average_last)


class InputBitQuantizer:
    """Keeps only the `bits` most significant bits of every 8-bit input value (x = value / 255, a tensor).
    A class at module level (not a local function) so DataLoader workers can pickle it: Python 3.14 starts
    them with 'forkserver', which pickles the dataset and its transforms."""

    def __init__(self, bits):
        self.shift = 8 - int(bits)

    def __call__(self, x):
        if self.shift == 0:
            return x
        p = torch.round(x * 255).to(torch.int32)
        return ((p >> self.shift) << self.shift).float() / 255.0

