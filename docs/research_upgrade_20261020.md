# Correction verification and utility scheduling: completed exploratory findings

The accepted development cohort contains 170 AutoLaparo training clips from nine verified source videos. These are previously observed sources, not 170 independent procedures or untouched held-out cases. Official validation and final-test evaluation have not been promoted by these results. All effects below average clips within source and then weight sources equally; descriptive paired source-bootstrap intervals reflect this small development sample.

## Fixed evaluation and matched controls

Each clip contributes 125 raw frames at 25 fps. Two independent binary targets are uterus and the release-annotated primary instrument union. Candidate masks occur at frames 0,24,49; scoring uses the disjoint frames 74,99,124. One correction purchases one full binary structure/frame mask, including initialization. Budgets 0,1,2,5 correspond to actual densities 0,0.8,1.6,4 masks per 100 frames. Normalized trapezoidal Dice AUC uses weights 0.1,0.2,0.4,0.3. The frozen framewise model supplies budget zero; SAM2.1 Tiny propagates purchased masks offline in both directions, with BF16 and postprocessing disabled. Every within-clip contrast uses the same factorized bank/hardware.

Raw uncertainty, temporal coverage (uniform), and random selection with ten fixed seeds are matched selector controls. They are not reproductions of complete EVA-VOS, SASVi, or memory-based surgical tracking systems. Full-mask counts do not measure clinician minutes.

| Method | Dice AUC | Difference from raw | Descriptive 95% interval | Positive sources |
|---|---:|---:|---:|---:|
| Raw uncertainty |0.893236|reference|--|--|
| Uniform coverage |0.892092|-0.001144|[-0.006896,+0.004109]|3/9|
| Random, ten-seed mean |0.869604|-0.023632|[-0.026483,-0.020431]|0/9|
| Observable acceptance, nested OOF |0.892851|-0.000385|[-0.001556,+0.000576]|4/9|
| Paid-anchor acceptance, nested OOF |0.892805|-0.000431|[-0.001679,+0.000549]|3/9|
| Static role/time utility, OOF |0.892383|-0.000853|[-0.006717,+0.004302]|4/9|
| Entropy-based utility, OOF |0.894716|+0.001480|[-0.001520,+0.004449]|5/9|

The acceptance model compares new and retained outputs using observable features. Anchor support reconstructs previously purchased masks with leave-one-prompt-out states, avoiding trivial comparison with an overwritten prompt. Nested source-grouped OOF chooses among the frozen families/thresholds. Anchor-minus-observable control is -0.0000458 AUC, with interval crossing zero: these results do not establish incremental benefit of purchased-mask verification.

After that negative result, one separately frozen utility branch predicts marginal gain of the next correction using current paid-state query entropy and a fixed source-balanced Ridge model. It uses additional offline outcome supervision; it is not training-free. The full-minus-static effect is +0.002333, with interval [-0.001137,+0.006517]. Both primary acceptance and this bounded utility branch fail their prespecified internal promotion requirement (at least +0.005 AUC and six of nine positive sources). No further family/threshold search is used to rescue them. No fitted candidate has been promoted to validation or final test.

## Why the budget curves matter

Raw Dice increases from 0.803468 at zero masks to 0.886549,0.902732,0.914955 at one,two,five masks. The first mask accounts for approximately 74.5% of that mean zero-to-five improvement. However, 43/170 individual clips lose Dice from two to five masks, including 13 losses greater than 0.02. Anchor acceptance reduces those counts to 40 and 10 but slightly decreases mean AUC. Suppressing some bad updates can also suppress useful updates; fewer harmful transitions alone do not demonstrate a better allocator. These are segmentation-score decreases, not patient-harm outcomes.

Uniform and uncertainty are close at the source level, while random is lower. These observations support careful matched baselines and diminishing-return analysis, not a general superiority claim for a new surgical AI method. Many algorithmic components have close prior art, and technical novelty has not been established.

## Numerical reproducibility and runtime scope

All 170 accepted banks pass the original strict query/metric replay checks. Eight failed original A10G runs were preserved and replaced only through a prospectively bounded exact-A10 repair; tolerances were not relaxed, favorable sources were not removed, and original failures remain in the private evidence archive. The accepted cohort uses 164 A10 and six A10G runs. A selected-fold numerical audit reproduces the chosen OOF endpoint curves to absolute tolerance 1e-12 with finite predictions. Constant-column scaler roundoff warnings were audited rather than used to change outcomes.

The published `runtime_by_clip.csv` describes counterfactual bank construction (14 propagations per clip). It is not deployment latency, total billed GPU time, end-to-end processing time, or human annotation time. An on-demand sequential runtime audit remains a separate engineering task. Public unit tests exercise paid-history guards and synthetic replay parity; synthetic success does not establish clinical validity.

## External descriptive replication is running

A separate frozen experiment uses all 101 CholecSeg8k windows from 17 sources, 80 consecutive frames each. Only fixed baseline/raw/uniform/random controls are included. Both failed learned candidates are excluded. The structural/compute pilot passed, and the remaining windows were dispatched with one GPU worker. No external efficacy result is included in this release, and no partial-cohort analysis is used for tuning.

This external release has been explored in related prior work; it is not a blinded, pristine unseen-data validation. The exact current checkpoint training-manifest audit finds no identical external image files, but transformed-image overlap and foundation-model pretraining overlap remain unexcluded/unknown. Release annotations cover grasper/L-hook, whereas the baseline tool head has wider scope. Void pixels are excluded from scoring; zero-filled void prompt pixels are an explicit mechanical approximation. Three-query budgets and short-window duration differ from AutoLaparo, so the two AUCs must not be pooled.

## Reproduction and limitations

Install the documented dependencies and run `python -m unittest discover -s research_upgrade -p 'test_*.py'`. These 32 tests use synthetic arrays and temporary files, without GPU dispatch or dataset download. Core bank, acceptance, utility and external analysis code is included; launchers, accounts, raw media, private volumes and model weights are excluded. Full scientific reruns require the licensed data, exact frozen model artifacts, manifests and study inputs. Aggregate CSVs under `benchmarks/research_upgrade_20261020/` support inspection of the reported findings; they do not reconstruct private masks or predictions.

The official sparse reference scope is narrower than the original dense-tail proposal. No new clinician reference annotations, expert adjudication, patient-level independence, ethics approval, safety improvement, measured clinician time, SOTA performance or submission readiness is claimed. Final-test access remains closed. The results are exploratory, including historical design exposure and the sequentially introduced utility branch.

Relevant primary sources: [AutoLaparo](https://arxiv.org/abs/2208.02049), [CholecSeg8k](https://arxiv.org/abs/2012.12453), [SAM2](https://arxiv.org/abs/2408.00714), [EVA-VOS](https://openaccess.thecvf.com/content/WACV2024/html/Delatolas_Learning_the_What_and_How_of_Annotation_in_Video_Object_WACV_2024_paper.html). Dataset annotations and foundation models remain attributed to their providers; this code release does not redistribute them.
