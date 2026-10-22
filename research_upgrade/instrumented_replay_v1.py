"""Versioned diagnostic hook around frozen raw replay; one run per process.

The observer receives prediction snapshots, never gold. Adaptive selection calls
the original selector unchanged. Fixed replay deliberately substitutes only the
registered query history. The frozen source file is not edited.
"""
import numpy as np
from research_upgrade import online_replay
from research_upgrade.bank import FRAMES, QUERY, entropy
from research_upgrade.reproducibility_observer_v1 import fingerprint

_ACTIVE = False


def run_instrumented(baseline, engine, purchase_mask, observer, fixed_queries=None):
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('One instrumented run per process')
    if fixed_queries is not None:
        fixed_queries = [tuple(map(int, row)) for row in fixed_queries]
        if len(fixed_queries) != 5 or len(set(fixed_queries)) != 5 or any(t not in QUERY or k not in (0, 1) for t, k in fixed_queries):
            raise ValueError('Five distinct registered candidate queries required')
    original = online_replay.select_next
    current = np.asarray(baseline)[list(FRAMES)].copy()
    logs = []

    class Proxy:
        @property
        def prompts(self):
            return engine.prompts

        @prompts.setter
        def prompts(self, value):
            engine.prompts = value

        def correct(self, t, head, mask):
            prediction = engine.correct(t, head, mask)
            current[:, head] = prediction[list(FRAMES), head]
            observer(len(logs), current.copy())
            return prediction

    def select(states, paid, policy, rng):
        for head in (0, 1):
            if not np.array_equal(states[head, paid[head]], current[:, head]):
                raise ValueError('Observer state differs from frozen raw replay')
        eligible = [(j, k) for j in range(3) for k in range(2) if not paid[k] & (1 << j)]
        candidates = [dict(frame=QUERY[j], head=k, entropy=entropy(states[k, paid[k]][FRAMES.index(QUERY[j])])) for j, k in eligible]
        candidates.sort(key=lambda r: (-r['entropy'], r['frame'], r['head']))
        adaptive = original(states, paid, policy, rng)
        selected = adaptive if fixed_queries is None else (QUERY.index(fixed_queries[len(logs)][0]), fixed_queries[len(logs)][1])
        if selected not in eligible:
            raise ValueError('Registered fixed query already purchased')
        logs.append(dict(query=len(logs) + 1, paid_subsets_before=list(paid), candidates=candidates,
                         top_margin=candidates[0]['entropy'] - candidates[1]['entropy'],
                         adaptive_query=[QUERY[adaptive[0]], adaptive[1]], selected_query=[QUERY[selected[0]], selected[1]],
                         state_before=fingerprint(current)))
        return selected

    _ACTIVE = True
    try:
        observer(0, current.copy())
        online_replay.select_next = select
        result = online_replay.run(baseline, Proxy(), purchase_mask, policy='uncertainty', seed=20261017)
    finally:
        online_replay.select_next = original
        _ACTIVE = False
    result['diagnostic_queries'] = logs
    result['diagnostic_mode'] = 'adaptive' if fixed_queries is None else 'fixed'
    return result
