"""Token-mean KL(teacher || student), T²-scaled, plus optional hidden MSE."""
import math

import torch.nn.functional as F


def distillation_loss(student, teacher, *, temperature=1.0,
                      student_hidden=None, teacher_hidden=None, hidden_weight=0.0):
    """Detach teacher targets; hidden states must have identical, nonempty shapes.

    Logits are [..., vocabulary]. KL is averaged over tokens, not batches.
    Half/bfloat logits are promoted to float32; float64 remains float64.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(hidden_weight) or hidden_weight < 0:
        raise ValueError("hidden_weight must be finite and nonnegative")
    if student.shape != teacher.shape or student.ndim < 2 or not student.numel():
        raise ValueError("logits must have matching nonempty [..., vocabulary] shapes")
    def work(x):
        return x if x.element_size() >= 4 else x.float()
    s = work(student).reshape(-1, student.shape[-1]) / temperature
    t = work(teacher.detach()).reshape_as(s) / temperature
    loss = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                    log_target=True, reduction="batchmean") * temperature**2
    if hidden_weight:
        if (student_hidden is None or teacher_hidden is None
                or student_hidden.shape != teacher_hidden.shape or not student_hidden.numel()):
            raise ValueError("hidden states must have matching nonempty shapes")
        loss = loss + hidden_weight * F.mse_loss(work(student_hidden), work(teacher_hidden.detach()))
    return loss
