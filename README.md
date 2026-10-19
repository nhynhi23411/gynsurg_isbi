# Gynecologic surgical video correction experiments

Research code for budgeted full-mask correction and offline propagation in gynecologic video segmentation. A frozen framewise segmenter provides the no-correction baseline; SAM2 propagates charged masks selected from a fixed candidate pool. Queries and scoring frames are disjoint.

## Current evidence

The full-training expansion covers **170 clips from nine previously observed source videos**, not 170 independent procedures. There are 161 new inference runs and nine verified reuses.

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
