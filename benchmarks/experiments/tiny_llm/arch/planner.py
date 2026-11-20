"""Analytical architecture budgets, not training or hardware measurements.

Only the dense MHA W1A8 layout matches tiny_llm's existing exporter. Proposed
GQA/factor layouts use the explicit packing contract in projection_layout().
"""

from dataclasses import dataclass
import math


PROJECTIONS = ("q", "k", "v", "o", "w1", "w3", "w2")


def positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def packed_rows(rows, columns, bits):
    """Each row is independently padded to a 32-bit word."""
    return rows * ((columns * bits + 31) // 32) * 4


@dataclass(frozen=True)
class Config:
    dim: int = 384
    layers: int = 8
    heads: int = 6
    kv_heads: int = 6
    hidden: int = 1024
    vocab: int = 4096
    ctx: int = 256
    weight_bits: tuple = (1, 1, 1, 1, 1, 1, 1)
    head_bits: int = 8
    kv_bits: int = 8

    def __post_init__(self):
        for name in ("dim", "layers", "heads", "kv_heads", "hidden", "vocab", "ctx"):
            positive_int(name, getattr(self, name))
        if self.dim % self.heads or self.head_dim % 2:
            raise ValueError("dim must divide into even-width rotary heads")
        if self.heads % self.kv_heads:
            raise ValueError("kv_heads must divide heads")
        if self.head_bits not in (2, 4, 8) or self.kv_bits not in (2, 4, 8):
            raise ValueError("head_bits and kv_bits must be 2, 4, or 8")
        if not isinstance(self.weight_bits, tuple) or len(self.weight_bits) != 7:
            raise ValueError("weight_bits must be a tuple in q,k,v,o,w1,w3,w2 order")
        for bits in self.weight_bits:
            if isinstance(bits, bool) or not isinstance(bits, (int, float)) or not 0 < bits <= 1:
                raise ValueError("projection bits must be 1 or a LittleBit target in (0,1)")
        # Reject a sub-bit budget that cannot store even rank one and its scales.
        self.projections()

    @property
    def head_dim(self):
        return self.dim // self.heads

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    def projections(self):
        shapes = ((self.dim, self.dim), (self.kv_dim, self.dim),
                  (self.kv_dim, self.dim), (self.dim, self.dim),
                  (self.hidden, self.dim), (self.hidden, self.dim),
                  (self.dim, self.hidden))
        return {name: projection_layout(rows, cols, bits)
                for name, (rows, cols), bits in zip(PROJECTIONS, shapes, self.weight_bits)}


def projection_layout(rows, columns, bits):
    """Exact bytes for this *planning* layout, not a LittleBit export claim.

    Dense: row-packed binary weights + int32 output multipliers. Factored:
    U[rows,rank] and V[rank,columns], independently word-padded rows, plus
    16-bit row/column/latent scales, padded to 4 bytes. No residual branch.
    Select the largest rank whose weights AND scales meet the target BPW.
    Metadata is accounted separately by size_model().
    """
    dense = rows * columns
    if bits == 1:
        weights = packed_rows(rows, columns, 1)
        return dict(dense_parameters=dense, parameters=dense, matmul_parameters=dense,
                    weight_bytes=weights, scale_bytes=4 * rows, rank=None,
                    achieved_bpw=8 * (weights + 4 * rows) / dense)

    def layout(rank):
        weights = packed_rows(rows, rank, 1) + packed_rows(rank, columns, 1)
        scales = ((2 * (rows + columns + rank) + 3) // 4) * 4
        return weights, scales

    budget = math.floor(dense * bits / 8)
    low, high = 0, min(rows, columns)
    while low < high:
        mid = (low + high + 1) // 2
        if sum(layout(mid)) <= budget:
            low = mid
        else:
            high = mid - 1
    if low == 0:
        raise ValueError(f"{bits} bpw cannot store rank one for {rows}x{columns}")
    weights, scales = layout(low)
    products = low * (rows + columns)
    return dict(dense_parameters=dense, parameters=products + rows + columns + low,
                matmul_parameters=products, weight_bytes=weights, scale_bytes=scales,
                rank=low, achieved_bpw=8 * (weights + scales) / dense)


def size_model(config):
    """Return exact counts/bytes under the documented layout (tied head)."""
    c = config
    projections = c.projections()
    norms = (2 * c.layers + 1) * c.dim
    embedding = c.vocab * c.dim
    proposed = c.kv_heads != c.heads or any(b != 1 for b in c.weight_bits)
    # Versions 3/4 add 16 bytes per linear. Proposed layouts reserve that too.
    metadata = 16 * 7 * c.layers if c.head_bits != 8 or proposed else 0
    parts = dict(header=32, embedding=packed_rows(c.vocab, c.dim, c.head_bits),
                 embedding_scales=4 * c.vocab, norms=4 * norms,
                 rope=2 * c.ctx * c.head_dim, nonlinear_tables=2 * (2048 + 1024),
                 projection_weights=c.layers * sum(p["weight_bytes"] for p in projections.values()),
                 projection_scales=c.layers * sum(p["scale_bytes"] for p in projections.values()),
                 projection_metadata=metadata)
    return dict(parameters=embedding + norms + c.layers * sum(p["parameters"] for p in projections.values()),
                dense_parameters=embedding + norms + c.layers * sum(p["dense_parameters"] for p in projections.values()),
                linear_parameters=c.layers * sum(p["matmul_parameters"] for p in projections.values()),
                flash_bytes=sum(parts.values()), flash_parts=parts, projections=projections)
