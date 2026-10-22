"""CPU-only paired-record audit. Reads execution records, never pixels or models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FRAMES = (0, 24, 49)
BUDGETS = (0, 1, 2, 5)
WEIGHTS = (.1, .2, .4, .3)


def read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def query_comparison(reference, current):
    if len(reference) != 5 or len(current) != 5:
        raise ValueError('Exactly five charged queries required')
    first = next((i + 1 for i, (a, b) in enumerate(zip(reference, current)) if a != b), None)
    return first, set(reference) == set(current)


def historical_margins(observables, queries):
    lookup = {(int(r['head']), int(r['subset']), int(r['frame'])): float(r['entropy']) for r in observables}
    paid = [0, 0]
    rows = []
    for step, (frame, head) in enumerate(queries, 1):
        options = [(lookup[k, paid[k], f], f, k) for f in FRAMES for k in (0, 1)
                   if not paid[k] & (1 << FRAMES.index(f))]
        options.sort(key=lambda x: (-x[0], x[1], x[2]))
        if options[0][1:] != (frame, head):
            raise ValueError('Historical observable replay does not reconstruct query')
        rows.append(dict(query=step, frame=frame, head=head,
                         entropy_top=options[0][0], entropy_second=options[1][0],
                         entropy_margin=options[0][0] - options[1][0],
                         historical_paid_head0=paid[0], historical_paid_head1=paid[1]))
        paid[head] |= 1 << FRAMES.index(frame)
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(historical, current, representatives, destination):
    destination.mkdir(parents=True, exist_ok=False)
    clip_rows, budget_rows, query_rows, evidence = [], [], [], {}
    for number in range(1, 171):
        cid = f'{number:03d}'
        old = historical / cid
        reused = not (current / cid).is_dir()
        new = (representatives if reused else current) / cid
        paths = [old / n for n in ['execution.json', 'policy_queries.csv', 'policy_curves.csv', 'selector_observables.csv']]
        paths += [new / n for n in ['execution.json', 'curves.csv', 'traces.json']]
        for path in paths:
            evidence[f'{"historical" if path.parent == old else "current"}/{cid}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        old_record = json.loads(paths[0].read_text())
        new_record = json.loads((new / 'execution.json').read_text())
        assert old_record['input_sha256'] == new_record['input_sha256'], cid
        assert old_record['procedure_id'] == new_record['procedure_id'], cid
        old_query = [(int(r['frame']), int(r['head'])) for r in read_csv(old / 'policy_queries.csv') if r['policy'] == 'uncertainty']
        traces = json.loads((new / 'traces.json').read_text())
        new_query = [tuple(x) for x in next(t for t in traces if t['mode'] == 'raw')['bought_queries']]
        first_query, same_set = query_comparison(old_query, new_query)
        margins = historical_margins(read_csv(old / 'selector_observables.csv'), old_query)
        for row, actual in zip(margins, new_query):
            query_rows.append(dict(clip_id=cid, **row, current_frame=actual[0], current_head=actual[1],
                                   historical_paid_subset_matches_current_prefix=row['query'] <= (first_query or 6)))
        previous = {int(r['n_queries']): r for r in read_csv(old / 'policy_curves.csv') if r['policy'] == 'uncertainty'}
        present = {int(r['n_queries']): r for r in read_csv(new / 'curves.csv') if r['mode'] == 'raw' and r['pool'] == 'original'}
        differences = []
        for n in BUDGETS:
            delta = [float(present[n][m]) - float(previous[n][m]) for m in ['dice', 'iou']]
            differences.append(delta)
            budget_rows.append(dict(clip_id=cid, n_queries=n, dice_difference=delta[0], iou_difference=delta[1],
                                    metric_tolerance_exceeded=max(map(abs, delta)) > .001))
        maximum = max(abs(x) for pair in differences for x in pair)
        bound = max(sum(w * abs(pair[k]) for w, pair in zip(WEIGHTS, differences)) for k in (0, 1))
        first_budget = next((n for n, pair in zip(BUDGETS, differences) if max(map(abs, pair)) > .001), None)
        clip_rows.append(dict(clip_id=cid, procedure_id=old_record['procedure_id'], reused_representative=reused,
            historical_gpu=old_record['actual_gpu'], current_gpu=new_record['actual_gpu'],
            historical_torch=old_record.get('torch', 'unrecorded'), current_torch=new_record.get('torch', 'unrecorded'),
            identical_input_hash=True, first_query_divergence=first_query, final_query_set_same=same_set,
            first_budget_over_metric_tolerance=first_budget, max_metric_difference=maximum, auc_absolute_difference_bound=bound,
            historical_parity_pass=first_query is None and maximum <= .001 and bound <= .0001,
            metric_failure_before_query_divergence=first_budget is not None and (first_query is None or first_budget < first_query)))
    write_csv(destination / 'clip_diagnostics.csv', clip_rows)
    write_csv(destination / 'budget_differences.csv', budget_rows)
    write_csv(destination / 'historical_query_margins.csv', query_rows)
    failed = [r for r in clip_rows if not r['historical_parity_pass']]
    summary = dict(records=170, sources=len({r['procedure_id'] for r in clip_rows}),
                   parity_failures=len(failed), query_divergent_clips=[r['clip_id'] for r in clip_rows if r['first_query_divergence']],
                   failed_with_same_full_query_history=sum(r['first_query_divergence'] is None for r in failed),
                   failed_with_no_earlier_query_divergence=sum(r['metric_failure_before_query_divergence'] for r in failed),
                   baseline_budget0_maximum=max(abs(r[k]) for r in budget_rows if r['n_queries'] == 0 for k in ['dice_difference', 'iou_difference']),
                   current_entropy_margins_available=False, tensor_level_outputs_available=False,
                   hardware_causality_established=False, gpu_dispatched=False, final_test=False,
                   original_gate_modified=False)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    (destination / 'evidence_sha256.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['historical', 'current', 'representatives', 'destination']:
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    audit(args.historical, args.current, args.representatives, args.destination)
