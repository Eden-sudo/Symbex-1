"""
Standard training: CrossEntropy + the STE gradient of the binarized layers.
No teacher, no distillation, no gating. This is the loop that today's
experiment confirmed works equally well for conv and linear (once the
BatchNorm fix below is applied) -- see corregir_batchnorm().

Use this for any model built with methods.apply_method(modelo, "plain")
(or "gated" / "bitslice", once those exist for conv too: the loop itself
does not care which method built the layers, it only calls model(x) and
loss.backward()).

Fixes in SYMBEX v3:
  - evaluar() returns the model to the mode it was in (before: always left it in training mode).
  - checkpoints: guardar="ultima" (default) saves the LAST epoch, the fair number to compare;
    guardar="mejor" keeps the old behavior (best TEST epoch: optimistic, it selects on the test set).
    guardar_cada now really saves every N epochs (before it only printed a message).
  - max_norm: the same gradient-clipping option as distillation.py (default None = no clipping,
    as before), so the two loops can be compared changing ONE variable.
"""
import time

import torch
import torch.nn as nn


def corregir_batchnorm(modelo, momentum=0.3):
    """
    Raises the momentum of every BatchNorm layer in `modelo` from PyTorch's
    default (0.1) to `momentum`.

    Why: confirmed experimentally (conv_vs_linear.py, today) that training a
    binarized network with the default BatchNorm momentum makes its running
    mean/var lag behind the activation statistics, which change fast because
    weights flip sign often during training. This causes wild, misleading
    swings in eval-mode (test) accuracy while train accuracy climbs smoothly
    -- in one run it looked like conv and linear learned very differently
    (3.05 points apart); raising momentum to 0.5 made that gap disappear
    (-0.46 points, i.e. noise). This has nothing to do with which method
    built the layers (plain / gated / bitslice): any BatchNorm sitting on
    top of binarized weights is exposed to the same lag.

    Only touches layers still at the PyTorch default (0.1): if the user's
    own model already set a custom momentum, that choice is respected.
    Call this once, right after building the model, before training.
    """
    tocadas = 0
    for modulo in modelo.modules():
        if isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and modulo.momentum == 0.1:
            modulo.momentum = momentum
            tocadas += 1
    return tocadas


@torch.no_grad()
def evaluar(modelo, loader, device):
    """Accuracy (%) of `modelo` on `loader`, in eval mode. The model is returned to the mode it was in."""
    estaba_entrenando = modelo.training
    modelo.eval()
    correctos, totales = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correctos += (modelo(x).argmax(1) == y).sum().item()
        totales += y.size(0)
    modelo.train(estaba_entrenando)
    return 100.0 * correctos / totales


class Checkpoints:
    """Shared checkpoint policy of the training loops.
    guardar="ultima": save the current model every `guardar_cada` epochs (if set) and after the last epoch.
    guardar="mejor" : save whenever the test accuracy improves (old behavior; selects on the test set)."""
    def __init__(self, ruta, guardar="ultima", guardar_cada=None, verbose=True):
        if guardar not in ("ultima", "mejor"):
            raise ValueError(f"guardar must be 'ultima' or 'mejor', got {guardar!r}")
        if guardar_cada is not None and guardar_cada < 1:
            raise ValueError("guardar_cada must be >= 1")
        self.ruta, self.guardar, self.cada, self.verbose = ruta, guardar, guardar_cada, verbose
        self.mejor = float("-inf")

    def despues_de_epoca(self, modelo, epoch, epochs, acc_test):
        if self.ruta is None:
            return
        ultima = (epoch + 1) == epochs
        if self.guardar == "mejor":
            if acc_test > self.mejor:
                self.mejor = acc_test
                torch.save(modelo.state_dict(), self.ruta)
            if self.cada and (epoch + 1) % self.cada == 0 and self.verbose:
                print(f"  [*] Checkpoint ({self.ruta}): best so far {self.mejor:.2f}% "
                      f"-- optimizer y scheduler NO se reinician")
        elif ultima or (self.cada and (epoch + 1) % self.cada == 0):
            torch.save(modelo.state_dict(), self.ruta)
            if self.verbose:
                print(f"  [*] Checkpoint ({self.ruta}): epoch {epoch + 1}"
                      + (" (last)" if ultima else "") + " -- optimizer y scheduler NO se reinician")


def entrenar(modelo, trainloader, testloader, epochs=30, lr=1e-3, weight_decay=1e-5,
            device=None, bn_momentum=0.3, promedio_ultimas=5, verbose=True,
            guardar_cada=None, ruta_checkpoint=None, guardar="ultima", max_norm=None):
    """
    Standard loop: Adam + CosineAnnealingLR + CrossEntropy.

    Returns a dict with:
      mejor_test        : the single highest test accuracy seen (noisy: see promedio_final)
      promedio_final     : average test accuracy of the last `promedio_ultimas` epochs
                           (the robust number: use THIS to compare architectures/methods,
                           not mejor_test -- a single lucky epoch is not a fair comparison)
      historial          : list of (epoch, loss, train_acc, test_acc)

    ruta_checkpoint / guardar / guardar_cada: see Checkpoints. The optimizer and the LR schedule are
    never restarted (unlike calling entrenar() again in a loop from the outside, which resets Adam's
    state and the LR schedule -- see entrenar_bn_corregido.py's original bug for what that costs).
    max_norm: gradient clipping (None = off, as before).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    if bn_momentum is not None:
        tocadas = corregir_batchnorm(modelo, bn_momentum)
        if verbose and tocadas:
            print(f"[*] BatchNorm momentum ajustado a {bn_momentum} en {tocadas} capa(s) "
                  f"(estaban en el default 0.1 de PyTorch)")

    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    ckpt = Checkpoints(ruta_checkpoint, guardar, guardar_cada, verbose)

    historial = []
    modelo.train()
    for epoch in range(epochs):
        t0 = time.time()
        loss_acc, correctos, totales = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            salida = modelo(x)
            loss = criterion(salida, y)
            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=max_norm)
            optimizer.step()
            loss_acc += loss.item()
            correctos += (salida.argmax(1) == y).sum().item()
            totales += y.size(0)
        scheduler.step()

        acc_train = 100.0 * correctos / totales
        acc_test = evaluar(modelo, testloader, device)
        historial.append((epoch, loss_acc / len(trainloader), acc_train, acc_test))
        if verbose:
            print(f"  Epoca [{epoch + 1:03d}/{epochs}] | Loss: {loss_acc / len(trainloader):.4f} | "
                  f"Train: {acc_train:5.2f}% | Test: {acc_test:5.2f}% | {time.time() - t0:.1f}s")
        ckpt.despues_de_epoca(modelo, epoch, epochs, acc_test)

    n = min(promedio_ultimas, len(historial))
    return dict(
        mejor_test=max(h[3] for h in historial),
        promedio_final=sum(h[3] for h in historial[-n:]) / n,
        historial=historial,
    )
