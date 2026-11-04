"""Training loops: standard (CrossEntropy) and distillation (FP32 teacher -> binarized student); Bop, the optimizer
for 1-bit weights of method "boolean" ([training] binary_optimizer = "bop")."""
from .standard import train_model, evaluate, fix_batchnorm, Checkpoints, InputBitQuantizer
from .distillation import train_teacher, distill_student
from .bop import Bop
