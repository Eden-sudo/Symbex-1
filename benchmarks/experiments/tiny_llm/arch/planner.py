"""Analytical architecture budgets, not training or hardware measurements.

Only the dense MHA W1A8 layout matches tiny_llm's existing exporter. Proposed
GQA/factor layouts use the explicit packing contract in projection_layout().
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


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


def latency(config, workload="mean"):
    """Milliseconds/token, calibrated on Eden's seed-42 lib_symbex JSON.

    mean: positions 0..ctx-1; full: ctx cached tokens; story: inferred effective
    occupancy reproducing the 120-token story, scaled with context capacity.
    Linear scales with executed binary products (W1A8); head with vocab*d*bits.
    Attention's rope/cache term is fixed per step, remaining work scales with
    occupied tokens*query_heads*head_dim*layers. GQA/KV bits do NOT imply a
    free attention speedup: repeated query-head work and unpacking remain.
    Factor scale/requantization overhead is unknown and excluded (optimistic).
    """
    if workload not in ("mean", "full", "story"):
        raise ValueError("workload must be mean, full, or story")
    ref = json.loads((Path(__file__).resolve().parents[1] /
                      "results/chip_s3_binary_d384l8_s42_symbex.json").read_text())
    p = ref["ms_per_token_by_part"]
    linear_ms = sum(p[k] for k in ("qkv", "o", "w1_w3", "w2"))
    rest_ms = p["embedding_norms"] + p["swiglu"]
    fixed = p["attention_rope_cache"]
    variable = sum(p[k] for k in ("attention_values", "attention_scores",
                                  "attention_softmax", "attention_division"))
    c = config
    reference_tokens = (256 + 1) / 2
    occupied = (c.ctx + 1) / 2
    if workload == "full":
        occupied = c.ctx
    elif workload == "story":
        story_attention = ref["ms_per_token_story_mean"] - linear_ms - rest_ms - p["output_head"]
        occupied *= (story_attention - fixed) / variable
        occupied = max(1, occupied)
    scale = c.layers * c.heads * c.head_dim / (8 * 6 * 64)
    parts = dict(linear=linear_ms * size_model(c)["linear_parameters"] / 14155776,
                 attention=scale * (fixed + variable * occupied / reference_tokens),
                 head=p["output_head"] * (c.vocab * c.dim * c.head_bits) / (4096 * 384 * 8),
                 rest=p["embedding_norms"] * c.layers * c.dim / (8 * 384)
                      + p["swiglu"] * c.layers * c.hidden / (8 * 1024))
    total = sum(parts.values())
    return dict(parts_ms=parts, ms_per_token=total, tokens_per_second=1000 / total,
                occupied_tokens=occupied, workload=workload)


def firmware_gaps(c):
    """Known structural restrictions, not a proof of runnable firmware."""
    gaps = []
    if c.kv_heads != c.heads:
        gaps.append("GQA needs exporter and attention kernels")
    if c.kv_bits != 8:
        gaps.append("packed KV needs quantization and kernels")
    if any(b != 1 for b in c.weight_bits):
        gaps.append("LittleBit needs training, exporter and factor kernels")
    if c.head_dim != 64 or c.dim % 64 or c.hidden % 32 or c.vocab % 4:
        gaps.append("current kernels require head_dim=64 and aligned dimensions/tables")
    if c.ctx > 258 or c.hidden < 2 * c.ctx or max(c.dim, c.hidden) < 4 * c.ctx:
        gaps.append("context exceeds current score/attention scratch constraints")
    if max(c.dim, c.hidden) > 1024:
        gaps.append("lib_symbex input workspace is capped at 1024")
    return gaps


def memory(config, *, sram_bytes=512 * 1024, psram_bytes=8 * 1024**2,
           flash_bytes=16 * 1024**2, sram_reserve=160 * 1024,
           psram_reserve=256 * 1024, flash_reserve=0x310000,
           model_partition_bytes=0x500000):
    """Placement estimate with explicit application/runtime headroom.

    Replicates current state buffers (including the allocated-but-unused lut8
    buffer for symbex). Proposed packed KV pads each head's K/V payload to
    bytes and retains int32 scales per token/head. No automatic SRAM spill:
    changing placement would invalidate latency calibration. Reserves cover
    stacks, static workspaces, allocator overhead, tokenizer/application, not
    a claim that all physical SRAM is heap. Current flash partition is 5 MiB.
    """
    limits = locals().copy()
    for name, value in limits.items():
        if name == "config":
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    c = config
    size = size_model(c)
    big = max(c.dim, c.hidden, 4 * c.ctx)
    # Keep existing d-wide k/v scratch even for GQA, a conservative extension.
    activations = 20 * c.dim + 21 * big + 2 * c.ctx + 4 * (big // 32 + 1)
    factor_rank = max((p["rank"] or 0 for p in size["projections"].values()), default=0)
    activations += 5 * factor_rank  # int32 latent + int8 requantized latent
    sram = dict(activations=activations, linear_lut=64 * ((big + 7) // 8) * 8,
                logits=8 * c.vocab, block_descriptors=c.layers * (8 + 7 * 32),
                reserve=sram_reserve)
    kv = 2 * c.layers * c.kv_heads * ((c.ctx * c.head_dim * c.kv_bits + 7) // 8)
    kv_scales = 8 * c.layers * c.ctx * c.kv_heads
    row_counts = 0
    for name, p in size["projections"].items():
        rows = c.kv_dim if name in ("k", "v") else c.hidden if name in ("w1", "w3") else c.dim
        row_counts += 4 * (rows + (p["rank"] or 0))
    psram = dict(model=size["flash_bytes"], kv_payload=kv, kv_scales=kv_scales,
                 kv_guard=16, binary_row_counts=c.layers * row_counts, reserve=psram_reserve)
    totals = dict(sram=sum(sram.values()), psram=sum(psram.values()),
                  flash=size["flash_bytes"] + flash_reserve)
    violations = []
    for name, capacity in (("sram", sram_bytes), ("psram", psram_bytes), ("flash", flash_bytes)):
        if totals[name] > capacity:
            violations.append(f"{name} exceeds budget by {totals[name] - capacity} bytes")
    if size["flash_bytes"] > model_partition_bytes:
        violations.append("model exceeds current flash partition")
    return dict(fits=not violations, violations=violations, totals=totals,
                sram=sram, psram=psram, firmware_gaps=firmware_gaps(c))
