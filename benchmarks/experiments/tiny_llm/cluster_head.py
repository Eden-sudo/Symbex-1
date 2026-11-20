"""CPU clustering and bounded int8 two-stage head reference (no row scales).

Integer sums use int64, dimension <= 65536. Ties prefer lower cluster/token ID.
Candidates are returned as sparse IDs/scores, never fabricated full logits.
"""
import numpy as np


def cluster_embeddings(embedding, clusters, *, iterations=8, seed=0):
    """Seeded Lloyd k-means, bounded to small CPU experiments.

    Empty centers retain their previous value. The final assignment is to the
    final float centers; rounded int8 centers are a routing approximation only.
    Input embeddings must already be in the shared int8 score scale.
    """
    e = np.asarray(embedding)
    if e.dtype != np.int8 or e.ndim != 2 or not all(e.shape):
        raise ValueError("embedding must be a nonempty int8 matrix")
    if e.size > 2_000_000 or type(clusters) is not int or not 1 <= clusters <= min(len(e), 256):
        raise ValueError("clustering exceeds the small CPU budget")
    if type(iterations) is not int or not 1 <= iterations <= 32:
        raise ValueError("iterations must be in [1, 32]")
    x = e.astype(np.float64)
    centers = x[np.random.default_rng(seed).choice(len(e), clusters, replace=False)].copy()

    def assign():
        # Avoid allocating vocab x clusters x dim distances.
        distance = ((x * x).sum(1)[:, None] + (centers * centers).sum(1)
                    - 2 * x @ centers.T)
        return distance.argmin(axis=1)

    for _ in range(iterations):
        labels = assign()
        for c in range(clusters):
            members = x[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return assign().astype(np.int32), np.rint(centers).clip(-128, 127).astype(np.int8)


def _inputs(h, weights):
    h, weights = np.asarray(h), np.asarray(weights)
    if (h.dtype != np.int8 or weights.dtype != np.int8 or h.ndim != 1
            or weights.ndim != 2 or weights.shape[1:] != h.shape
            or not 1 <= h.size <= 65536 or not len(weights)):
        raise ValueError("expected int8 h[dim], weights[rows,dim], dim in [1,65536]")
    return h, weights


def full_logits(h, weights):
    h, weights = _inputs(h, weights)
    return weights.astype(np.int64) @ h.astype(np.int64)


def shortlist_logits(h, weights, labels, centers, k):
    """Return (ascending token IDs, exact int64 scores, ranked cluster IDs).

    All selected clusters are searched. Empty clusters are permitted; an empty
    candidate set is an error. Greedy parity requires the full head's winning
    token (lowest ID on ties) to be present, not merely any tied maximizer.
    """
    h, weights = _inputs(h, weights)
    _, centers = _inputs(h, centers)
    labels = np.asarray(labels)
    if (labels.shape != (len(weights),) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= len(centers))):
        raise ValueError("labels must assign each token to a valid cluster")
    if type(k) is not int or not 1 <= k <= len(centers):
        raise ValueError("k must be in [1, clusters]")
    routing = full_logits(h, centers)
    selected = np.argsort(-routing, kind="stable")[:k]
    ids = np.flatnonzero(np.isin(labels, selected))
    if not len(ids):
        raise ValueError("shortlist contains no tokens")
    return ids, full_logits(h, weights[ids]), selected


def greedy(ids, scores):
    """Choose from a nonempty sparse result produced by shortlist_logits."""
    return int(ids[np.argmax(scores)])
