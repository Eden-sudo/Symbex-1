"""
Distillation: an OPTIONAL wrapper around the standard loop. Trains an FP32 teacher first, then trains the
binarized student against a hybrid loss (KL-divergence to the teacher's soft outputs + CrossEntropy to the real
labels), instead of CrossEntropy alone.

This is not "what dense needs" or "what conv needs" -- it is a technique that may or may not help, independent
of layer type (see the note in compiler_core/sensitivity.py and the digits duel: gating + distillation together
gave 93.89% vs 92.22% plain, but that experiment mixed two variables and did not isolate distillation's own
contribution).

The same BatchNorm fix as standard.py applies here (see standard.fix_batchnorm): distillation does not change
WHY that lag happens, only what loss trains the weights.

Fixes in SYMBEX v3:
  - teacher_input: optional function applied to the batch before the teacher sees it, for when the teacher
    expects a different input than the student (e.g. the student takes uint8 pixels and the teacher normalized
    floats). Default None: both get the same x, as before.
  - checkpoints with the same policy and options as standard.train_model (default: no file, as before).
  - max_norm is a parameter (default 1.0, as before); standard.train_model has the same option.
  - train_teacher returns the teacher in eval mode.
Old names of functions, arguments and result keys are still accepted (see RENAMES.md).
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard import fix_batchnorm, evaluate, Checkpoints, _summary


def train_teacher(teacher, trainloader, testloader, epochs=150, lr=5e-3, device=None, verbose=True):
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
            print(f"  [teacher] Epoch [{epoch + 1:03d}/{epochs}] | Test: {evaluate(teacher, testloader, device):5.2f}%")

    acc = evaluate(teacher, testloader, device)
    return teacher.eval(), acc


def distill_student(student, teacher, trainloader, testloader, epochs=300, lr=1e-3,
                    weight_decay=1e-5, temperature=4.0, alpha=0.85, device=None,
                    bn_momentum=0.3, average_last=5, verbose=True,
                    teacher_input=None, max_norm=1.0,
                    checkpoint_path=None, save="last", save_every=None,
                    temperatura=None, promedio_ultimas=None, entrada_maestro=None,
                    ruta_checkpoint=None, guardar=None, guardar_cada=None):
    """
    Hybrid loss: alpha * KL(student_soft, teacher_soft) * T^2 + (1 - alpha) * CrossEntropy(student, y_true).

    Returns the same dict as standard.train_model (best_test, final_average, history, plus the old keys),
    so the two are interchangeable from the caller's point of view.
    temperatura / promedio_ultimas / entrada_maestro / ruta_checkpoint / guardar / guardar_cada: old names
    (kept until every caller is translated).
    """
    temperature = temperature if temperatura is None else temperatura
    average_last = average_last if promedio_ultimas is None else promedio_ultimas
    teacher_input = teacher_input if entrada_maestro is None else entrada_maestro
    checkpoint_path = checkpoint_path if ruta_checkpoint is None else ruta_checkpoint
    save = save if guardar is None else guardar
    save_every = save_every if guardar_cada is None else guardar_cada
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    teacher = teacher.to(device).eval()
    if bn_momentum is not None:
        changed = fix_batchnorm(student, bn_momentum)
        if verbose and changed:
            print(f"[*] BatchNorm momentum set to {bn_momentum} in {changed} layer(s) of the student")

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ckpt = Checkpoints(checkpoint_path, save, save_every, verbose)

    history = []
    student.train()
    for epoch in range(epochs):
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(x if teacher_input is None else teacher_input(x))
            s_out = student(x)

            loss_kl = F.kl_div(F.log_softmax(s_out / temperature, dim=1),
                               F.softmax(t_out / temperature, dim=1),
                               reduction="batchmean") * (temperature ** 2)
            loss_ce = F.cross_entropy(s_out, y)
            loss = alpha * loss_kl + (1.0 - alpha) * loss_ce

            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=max_norm)
            optimizer.step()

            loss_sum += loss.item()
            correct += (s_out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        acc_train = 100.0 * correct / total
        acc_test = evaluate(student, testloader, device)
        history.append((epoch, loss_sum / len(trainloader), acc_train, acc_test))
        if verbose:
            print(f"  [student] Epoch [{epoch + 1:03d}/{epochs}] | Loss: {loss_sum / len(trainloader):.4f} | "
                  f"Train: {acc_train:5.2f}% | Test: {acc_test:5.2f}% | {time.time() - t0:.1f}s")
        ckpt.after_epoch(student, epoch, epochs, acc_test)

    return _summary(history, average_last)

