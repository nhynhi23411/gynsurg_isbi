"""Prediction-only diagnostic logging; never supplies features to a selector.

Use on private outputs. Digests do not reconstruct prediction tensors: a later
paired numerical comparison also requires private snapshots at identical steps.
"""
import hashlib
import numpy as np


def compare_predictions(reference, current):
    a, b = np.asarray(reference), np.asarray(current)
    if a.shape != b.shape or not a.size:
        raise ValueError('Nonempty identical shapes required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite predictions')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(maximum_absolute_difference=float(delta.max()),
                mean_absolute_difference=float(delta.mean()),
                threshold_05_disagreement_fraction=float(np.mean((a > .5) != (b > .5))),
                exact_numeric_equality=bool(np.array_equal(a, b)))


def fingerprint(array):
    a = np.asarray(array)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Nonempty finite predictions required')
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                minimum=float(a.min()), maximum=float(a.max()),
                threshold_05_foreground_fraction=float(np.mean(a > .5)))
