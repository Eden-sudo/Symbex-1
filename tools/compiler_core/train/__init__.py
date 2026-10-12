"""Training loops. `standard` covers conv and linear alike; `distillation` is an
optional wrapper around it; `attention` is reserved, not implemented yet (see that file)."""
from .standard import entrenar, evaluar, corregir_batchnorm
from .distillation import entrenar_teacher, destilar_estudiante
