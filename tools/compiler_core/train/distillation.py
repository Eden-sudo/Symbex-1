"""
Distillation: an OPTIONAL wrapper around the standard loop. Trains an FP32
teacher first, then trains the binarized student against a hybrid loss
(KL-divergence to the teacher's soft outputs + CrossEntropy to the real
labels), instead of CrossEntropy alone.

This is not "what linear needs" or "what conv needs" -- it is a technique
that may or may not help, independent of layer type (see the note in
compiler_core/sensitivity.py and the digits duel: gating + distillation
together gave 93.89% vs 92.22% plain, but that experiment mixed two
variables and did not isolate distillation's own contribution).

Same BatchNorm fix as standard.py applies here (see standard.corregir_batchnorm):
distillation does not change WHY that lag happens, only what loss trains the weights.

Fixes in SYMBEX v3:
  - entrada_maestro: optional function applied to the batch before the teacher sees it, for when the
    teacher expects a different input than the student (e.g. the student takes uint8 pixels and the
    teacher normalized floats). Default None: both get the same x, as before.
  - checkpoints with the same policy and options as standard.entrenar (default: no file, as before).
  - max_norm is now a parameter (default 1.0, as before); standard.entrenar has the same option.
  - entrenar_teacher returns the teacher in eval mode.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard import corregir_batchnorm, evaluar, Checkpoints


def entrenar_teacher(teacher, trainloader, testloader, epochs=150, lr=5e-3, device=None, verbose=True):
    """Plain FP32 training (CrossEntropy only). No STE, no BatchNorm fix needed: the teacher is not binarized.
    Returns (teacher in eval mode, its final test accuracy)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    teacher.train()
    for epoch in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(teacher(x), y)
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  [teacher] Epoca [{epoch + 1:03d}/{epochs}] | Test: {evaluar(teacher, testloader, device):5.2f}%")

    acc = evaluar(teacher, testloader, device)
    return teacher.eval(), acc


def destilar_estudiante(student, teacher, trainloader, testloader, epochs=300, lr=1e-3,
                        weight_decay=1e-5, temperatura=4.0, alpha=0.85, device=None,
                        bn_momentum=0.3, promedio_ultimas=5, verbose=True,
                        entrada_maestro=None, max_norm=1.0,
                        ruta_checkpoint=None, guardar="ultima", guardar_cada=None):
    """
    Hybrid loss: alpha * KL(student_soft, teacher_soft) * T^2 + (1 - alpha) * CrossEntropy(student, y_true).

    Returns the same dict shape as standard.entrenar (mejor_test, promedio_final, historial),
    so the two are interchangeable from the caller's point of view.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    teacher = teacher.to(device).eval()
    if bn_momentum is not None:
        tocadas = corregir_batchnorm(student, bn_momentum)
        if verbose and tocadas:
            print(f"[*] BatchNorm momentum ajustado a {bn_momentum} en {tocadas} capa(s) del estudiante")

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ckpt = Checkpoints(ruta_checkpoint, guardar, guardar_cada, verbose)

    historial = []
    student.train()
    for epoch in range(epochs):
        t0 = time.time()
        loss_acc, correctos, totales = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(x if entrada_maestro is None else entrada_maestro(x))
            s_out = student(x)

            loss_kl = F.kl_div(F.log_softmax(s_out / temperatura, dim=1),
                               F.softmax(t_out / temperatura, dim=1),
                               reduction="batchmean") * (temperatura ** 2)
            loss_ce = F.cross_entropy(s_out, y)
            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce

            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=max_norm)
            optimizer.step()

            loss_acc += loss.item()
            correctos += (s_out.argmax(1) == y).sum().item()
            totales += y.size(0)
        scheduler.step()

        acc_train = 100.0 * correctos / totales
        acc_test = evaluar(student, testloader, device)
        historial.append((epoch, loss_acc / len(trainloader), acc_train, acc_test))
        if verbose:
            print(f"  [student] Epoca [{epoch + 1:03d}/{epochs}] | Loss: {loss_acc / len(trainloader):.4f} | "
                  f"Train: {acc_train:5.2f}% | Test: {acc_test:5.2f}% | {time.time() - t0:.1f}s")
        ckpt.despues_de_epoca(student, epoch, epochs, acc_test)

    n = min(promedio_ultimas, len(historial))
    return dict(
        mejor_test=max(h[3] for h in historial),
        promedio_final=sum(h[3] for h in historial[-n:]) / n,
        historial=historial,
    )
