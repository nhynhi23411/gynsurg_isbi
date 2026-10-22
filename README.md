## Latest bounded precision and runtime diagnostic

Four paired GPU calls were attempted; three completed (six/eight registered arms), and one stopped at a no-overwrite guard with partial states retained. No fifth call was submitted. Independent scalar/hash QA passed for the six complete arms, 60 pair-scope steps and 216 head/frame metrics. Both precision arms disabled TF32 and shared a physical device within each process. On the single cross-class clip 132, maximum A10-versus-A10G probability drift was 1.0 under BF16 and 0.000732 without external BF16 autocast; the latter remained nonidentical and did not ensure higher Dice. Official SAM2 still stores mask memory in BF16 internally, so this is not a pure-FP32-all-tensors run. The original 20/170 failures, negative outcomes and closed validation/final-test gates remain unchanged.

[Scientific report and four figures](docs/reproducibility_20261022/precision_runtime/REPORT.md) · [Unapproved four-page working draft](docs/reproducibility_20261022/precision_runtime/paper_working_draft.pdf). Run `python tools/reproduce_precision_diagnostic.py` for released scalar/hash consistency. Device identifiers are anonymized aliases; private probability arrays, medical media, weights and account launchers are excluded.

## Prior allocator-observed reproducibility diagnostic

All 24 registered additional executions completed (20 A10, four A10G). Independent metadata/hash/scalar QA passed for 43 records, 516 pair-scope steps and 1,548 head/frame scores. All 192 same-class scoring-step pairs were exact; fixed-query cross-class comparisons still differed by up to 0.216309 in probability. This is execution-environment association, not isolated GPU causality or a repair of the original failed gate. The original 20/170 failures remain; validation and final test remain closed. Full report and four figures: [allocator-observed report](docs/reproducibility_20261022/allocator_observed/REPORT.md). Run `python tools/reproduce_allocator_diagnostic.py` to audit the aggregate release. No prediction arrays or medical imagery are published.

# Gynecologic surgical video correction experiments

Research code for budgeted full-mask correction and offline propagation in gynecologic video segmentation. A frozen framewise segmenter provides the no-correction baseline; SAM2 propagates charged masks selected from a fixed candidate pool. Queries and scoring frames are disjoint.

## Current evidence

**22 October instrumented technical diagnostic:** [report and coverage](docs/reproducibility_20261022/instrumented_diagnostic/REPORT.md) retain 48 registered cells: 25 submitted, 19 complete on A10, six skipped for wrong hardware and 23 not dispatched. All 96 available step pairs are bitwise identical on saved query/scoring probabilities; missing pairs are not zeros. This supports conditional repeatability and does not repair the original failed20/170 gate. The AWS-constrained Modal arm failed at function creation before any scientific call because cloud pinning was unsupported. No A10G inference or validation/test completion is claimed. Code snapshot `adb41c1` remains tagged `freeze-completion-20261022`; investigation is isolated on `research/reproducibility-20261022`. Run `python tools/reproduce_reproducibility_diagnostic.py` to verify the numeric release.

**22 October full execution and failed parity gate:** [scientific report](docs/completion_appendix_20261022/REPORT.md) and [numerical appendix](benchmarks/completion_appendix_20261022/) retain all 170 execution records (161 new + nine reuses). Original historical raw parity fails on 20 clips, including three different query histories; all failures occurred on A10G. Structural provenance/hash/budget checks pass separately and do not override this failed primary acceptance gate. The all-170 results are a **nonconforming exploratory appendix**, not accepted primary-protocol results or independent validation. EVA-minus-current-raw AUC is -0.003079, descriptive 95% interval [-0.007819,+0.001327]. The separate zero-gold automatic Dice difference is -0.012065, interval [-0.026281,+0.001406]. No hardware-based dropping or favorable rerun was performed. Uniform/reject new-compute coverage remains nine. Validation is protocol-only, final test closed. Run `python tools/reproduce_completion_appendix.py` to reconstruct source-equal summaries and verify retained failures.


**Prior 21 October time-stop handover (superseded for execution coverage by the full appendix above):** [scientific report](docs/completion_extension_20261021/REPORT.md) and [aggregate tables](benchmarks/completion_extension_20261021/). Ten fixed feature/fallback ablations cover 170 training clips/nine sources; every original-pool mean is below raw. Nine representatives provide same-hardware compute and limited EVA-QNet/SASVi-inspired adaptations, not original-system reproductions. EVA-minus-raw AUC is -0.001289 with an interval crossing zero. The zero-gold predicted-prompt endpoint is reported separately. All 161 additional inputs were transferred privately, but the registered time gate stopped before new GPU inference. Closest-control efficacy remains nine clips; validation is protocol-only, final test closed. Private media, weights, credentials and account/transport launchers are excluded.

**20 October 2026 update:** completed source-grouped correction-verification and utility-scheduling analyses are documented in [the research findings](docs/research_upgrade_20261020.md), with [aggregate tables](benchmarks/research_upgrade_20261020/). Paid-anchor acceptance changes Dice AUC by -0.000431 versus raw uncertainty; entropy-based utility changes it by +0.001480, with both intervals crossing zero. Both branches fail their prespecified promotion gate. The separate external fixed-controls batch is complete on 101 windows/17 sources from a previously explored release: entropy-minus-random AUC +0.011855, while the interval versus uniform crosses zero. No final-test result is claimed.

The added `research_upgrade` package has 55 synthetic tests covering paid-mask isolation, matched budget replay, utility features, void labels and source-equal aggregation:

```bash
python -m unittest discover -s research_upgrade -p 'test_*.py'
```

The earlier factorized-bank training expansion covers **170 clips from nine previously observed source videos**, not 170 independent procedures. That earlier bank cohort contains 161 new inference runs and nine verified reuses; the subsequent all-170 close-comparator appendix has a failed historical parity gate, as disclosed above.

| Comparison | Source-macro normalized Dice AUC |
|---|---:|
| Frozen framewise, no corrections |0.803468|
| Raw SAM with uncertainty queries |0.893236|
| Paired difference |+0.089768|
| Descriptive source-bootstrap 95% interval |[+0.079290,+0.099948]|

This comparison establishes a correction-versus-no-correction result on the stated development scope. It **does not establish superiority of the proposed scheduling or reliability heuristics**. Those experiments have negative or inconclusive development results. Final-test evaluation remains unopened. No new clinical annotations or clinician annotation-time measurements are claimed.

## Quick start

Python 3.11 or 3.12 is used by the project. Install PyTorch for your own hardware, then install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python examples/reproduce_published_metrics.py
```

The software tests use synthetic inputs and temporary files; they do not start GPU jobs or download the datasets. Modal is imported by packaging regression tests, but the tests do not invoke remote inference. The original desktop execution completed 62 tests.

## Project layout

- `src/`: policy features, charged correction loop, SAM2 adapter, metrics, grouped analysis and reference-scope checks.
- `scripts/`: preparation, experiment orchestration, grouped gate/oracle diagnostics, validation and analysis helpers.
- `configs/`: policy recipes, metadata templates and experiment registries.
- `tests/`: synthetic leakage, metric, source-grouping, packaging and transport regression tests.
- `protocols/`: versioned scientific and recovery protocols with historical freezes.
- `benchmarks/`: numeric summaries and per-clip/per-source AUC from the full-training expansion. No image, video or mask pixels are included.
- `examples/`: a portable CPU reproduction of the released aggregate result.

## Computational scope

Released AutoLaparo references support 125 raw frames per clip, query frames[0,24,49], scoring frames[74,99,124], two dataset-defined targets and charged-mask counts[0,1,2,5]. Actual densities are[0,0.8,1.6,4] masks per 100 raw frames. Clip means are averaged within source, then sources receive equal weight. Descriptive bootstrap resamples sources, never frames or clips as independent cases.

Obtain datasets directly from their providers, including [AutoLaparo](https://autolaparo.github.io/). Dataset access, checkpoints, original baseline model assets and the upstream SAM2 installation are separate prerequisites for real inference. Dataset permissions do not transfer through this repository. Raw medical media, masks, model weights and provider access links are excluded.

The recorded GPU protocol used actual NVIDIA A10, BF16 and fixed checkpoint/config/input hashes. Frozen inference should not be silently moved to different hardware or altered parameters and reported as the same experiment. Archived desktop/Modal scripts include machine-specific run paths; adjust those in a new, versioned protocol for a new environment. The core CLI and released-metric example are separate from these archived deployment helpers.

## Reproducibility and access gates

`configs/review_status.json` is distributed with execution approvals disabled for a new checkout. Historical user authorization records document the archived study; they do not authorize another operator or open the final test. Configure local permissions, source grouping, reference completeness and exposure records before real inference.

Original freeze files include hashes of licensed assets that are intentionally absent here. They document experimental provenance; the public checkout alone cannot pass a complete private-data audit. Do not replace missing reference masks with predictions or infer clinical adjudication from engineering QA.

See [scientific interpretation](docs/scientific_results.md) for the distinction between observed correction benefits, failed heuristics, oracle diagnostics and remaining scientific requirements.
