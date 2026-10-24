"""Training loops: standard (CrossEntropy) and distillation (FP32 teacher -> binarized student)."""
from .standard import train_model, evaluate, fix_batchnorm, Checkpoints, InputBitQuantizer
from .distillation import train_teacher, distill_student
