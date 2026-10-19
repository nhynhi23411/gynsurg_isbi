"""Prediction-only hard gate and group-safe parameter-free grid selection."""
import numpy as np

COUNTS = [0, 1, 2, 5]
WEIGHTS = np.array([.1, .2, .4, .3])

def candidates():
    result = [dict(name='raw', distance_s=None, entropy=None)]
    for d in [.5, 1., 2., 3., 5.]:
        result.append(dict(name=f'distance_{d:g}', distance_s=d, entropy=1.))
    for d in [.5, 1., 2., 3., 5.]:
        for h in [.05, .1, .2, .4]:
            result.append(dict(name=f'distance_{d:g}_entropy_{h:g}', distance_s=d, entropy=h))
    return result

def keep_raw(prompted, nearest_query_s, sam_entropy, candidate):
    """No reference masks, reference presence, Dice or source IDs accepted."""
    prompted = np.asarray(prompted, dtype=bool)
    distance = np.asarray(nearest_query_s, dtype=float)
    entropy = np.asarray(sam_entropy, dtype=float)
    if distance.shape != prompted.shape or entropy.shape != prompted.shape:
        raise ValueError('Feature shape mismatch')
    if np.any(~np.isfinite(distance[prompted])) or np.any(distance[prompted] < 0):
        raise ValueError('Invalid prompted distance')
    if np.any(~np.isfinite(entropy)) or np.any((entropy < 0) | (entropy > 1)):
        raise ValueError('Invalid entropy')
    if candidate['name'] == 'raw':
        return np.ones(prompted.shape, dtype=bool)
    return (~prompted) | ((distance <= candidate['distance_s']) & (entropy <= candidate['entropy']))

def macro_auc(rows, dice):
    """One source/clip: equal structures/frames within each budget."""
    if len(rows) != 24 or rows.duplicated(['query_count', 'frame', 'structure']).any():
        raise ValueError('Expected complete one-source primary grid')
    if set(rows['query_count']) != set(COUNTS) or set(rows['frame']) != {74,99,124} or set(rows['structure']) != {0,1}:
        raise ValueError('Unexpected primary endpoint grid')
    values = np.asarray(dice, dtype=float)
    if len(values) != len(rows) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid Dice')
    curve = [float(values[rows.query_count.to_numpy() == n].mean()) for n in COUNTS]
    return float(np.dot(curve, WEIGHTS))

def select_from_training(score_table, held_out):
    """Exclude outer source first; each remaining source is one inner holdout.

    No coefficients are fitted: candidate columns are fixed threshold rules.
    Returns full inner memberships so the parameter-free selection is explicit.
    """
    train = score_table.drop(index=held_out)
    if len(train) < 2 or not np.isfinite(train.to_numpy()).all():
        raise ValueError('Insufficient/invalid inner sources')
    means = train.mean(axis=0)
    best = next(name for name in score_table.columns if means[name] >= means.max()-1e-12)
    folds = [dict(validation_source=str(pid), training_sources=[str(x) for x in train.index if x != pid]) for pid in train.index]
    return best, {str(k):float(v) for k,v in means.items()}, folds
