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

The published `runtime_by_clip.csv` describes counterfactual bank construction (14 propagations per clip). It is not deployment latency, total billed GPU time, end-to-end processing time, or human annotation time. A completed one-training-clip sequential software audit preserves query history and meets original metric tolerances: raw46.55s/5propagations versus constant-reject software control83.62s/9propagations, both5paidmasks onA10. This is software parity, not fitted-gate efficacy or an average latency benchmark. Prepared baseline/JPEGs precede timing; mask-copy callbacks are not human annotation time. Public unit tests exercise paid-history guards and synthetic replay parity; synthetic success does not establish clinical validity.

## Completed external descriptive replication

All101CholecSeg8k windows from17sources completed. The released grasper/L-hook union is narrower than the model tool head; scoring excludes void, binary query void is mechanically zero-filled. Query0/15/31, score47/63/79, 80frames at25fps. Budgets0/1/2/3 use densities0/1.25/2.5/3.75 and AUC weights1/6,1/3,1/3,1/6. Do not pool this grid with AutoLaparo.

Source-macro AUC: entropy0.916173,uniform0.914145,random10seeds0.904317. Entropy-minus-random+0.011855 (descriptive sourcebootstrap95%[0.001561,0.024806],10/17positive); entropy-minus-uniform+0.002028 ([-0.005708,0.012754],4/17positive). Entropy Dice0/1/2/3=.840703/.924833/.935524/.935617. All selectors converge after buying every action. Entropy2-to3 declines30/101windows,2above.02. Prespecified present-only sensitivity uses16contributing sources; endpoint difference at2vs3 masks is descriptive.

This release was previously explored in related work: NOT blinded/pristine external validation. Zero exact training-file matches cannot exclude transformed or pretraining overlap. No failed learned candidate is evaluated/retuned. All101file hashes,scope,chargedmaskcounts and full17source aggregation/bootstrap passed audit. Hardware85A10+16A10G, matched within window; accepted-bank1.615662workerh is neither billedGPUtime nor policy latency. Numeric CSVs reproduce these results without redistributing licensed media.

## Evaluator-only finite-space diagnostic

Retrospective enumeration of720ordered five-mask sequences from six actions uses existing bank states. Gold-aware per-clip maxima give source-macroAUC0.920028 vsraw0.893236,gap0.026792 (descriptiveCI[.016340,.038904]); exact-random expectation.872149. Oracle advantage is partly nonnegative by construction; it is forbidden future-outcome access, not a deployable selector, a global architecture bound or evidence the gap is learnable. No new learner/promotion or finaltest access follows. Registered10seedrandom remains the original comparator. Published diagnostic aggregate CSVs and two synthetic tests accompany the core evaluator code.


## Reproduction and limitations

Install the documented dependencies and run `python -m unittest discover -s research_upgrade -p 'test_*.py'`. These 32 tests use synthetic arrays and temporary files, without GPU dispatch or dataset download. Core bank, acceptance, utility and external analysis code is included; launchers, accounts, raw media, private volumes and model weights are excluded. Full scientific reruns require the licensed data, exact frozen model artifacts, manifests and study inputs. Aggregate CSVs under `benchmarks/research_upgrade_20261020/` support inspection of the reported findings; they do not reconstruct private masks or predictions.

The official sparse reference scope is narrower than the original dense-tail proposal. No new clinician reference annotations, expert adjudication, patient-level independence, ethics approval, safety improvement, measured clinician time, SOTA performance or submission readiness is claimed. Final-test access remains closed. The results are exploratory, including historical design exposure and the sequentially introduced utility branch.

Relevant primary sources: [AutoLaparo](https://arxiv.org/abs/2208.02049), [CholecSeg8k](https://arxiv.org/abs/2012.12453), [SAM2](https://arxiv.org/abs/2408.00714), [EVA-VOS](https://openaccess.thecvf.com/content/WACV2024/html/Delatolas_Learning_the_What_and_How_of_Annotation_in_Video_Object_WACV_2024_paper.html). Dataset annotations and foundation models remain attributed to their providers; this code release does not redistribute them.
