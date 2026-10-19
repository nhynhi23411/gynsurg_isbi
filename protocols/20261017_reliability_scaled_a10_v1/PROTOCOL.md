# Hardware-only execution revision: NVIDIA A10

The 2026-10-17 first attempt received NVIDIA A10 and was rejected by the frozen A10G contract before inference. This new version requires actual NVIDIA A10. Algorithm, six recipes, lambda/delta/tau, reference/scoring pools and budgets are unchanged. Original A10G freeze remains immutable. No new-method outcome was inspected to make this change. Compare all recipes within this fixed actual hardware; never pool A10/A10G numerical runs silently.

# Prototype v2: prediction-based fallback and relative diversity

Status: experimental design and implementation, recorded before new-method GPU runs. The user requested this version after negative v1 development and diagnostic replays. This is an outcome-informed redirection, not a prespecified confirmation of the original interaction-aware hypothesis. Preserve original plan, configs and all negative results. Do not claim novelty or efficacy from this implementation.

## Scope and invariants

Use the same official-only 125-frame inputs, query frames [0,24,49], scoring [74,99,124], uterus/tools targets, budgets [0,1,2,5], actual densities [0,.8,1.6,4]. All selected full-mask prompts remain charged, including a prompt whose propagation is subsequently downweighted. At budget zero, output is exactly the frozen framewise baseline. Never read candidate/scoring gold for selection, reliability weights, threshold choice or acceptance/rejection of propagation. Exact masks on already charged prompt frames are allowed and remain exact. Raw data, checkpoint and original approved plan are immutable.

## Relative diversity policy

New policy identifier: `scaled_diversity_v1`; no proximity multiplier in this prototype.

H = mean binary entropy / log(2); D = mean absolute difference between current probabilities and frozen framewise probabilities; E=(H+D)/2. Clip probabilities only for entropy; compute disagreement from original probability values so identical arrays give D=0.

R(t,k)=max exp(-|t-tq|/(fps*tau)) over prior queries of the same structure, or 0 with none. Select the unused candidate maximizing **S=E*(1-lambda*R)**, with frame/structure ascending tie-break. Lambda=.25, constrained to [0,1]; tau=2s. Diversity can reduce E by at most 25% at these defaults and cannot turn a nonnegative E negative. It does not enforce structure balance or guarantee better choices. These constants are engineering choices informed by the scale failure, not estimates fitted to scoring labels.

## Observable propagation fallback

SAM2 continues to replay every charged prompt, with the existing model, BF16 and postprocessing settings. Cache raw SAM probabilities, not trust-filtered outputs. Keep all charged prompts for future replay even if a propagated prediction is downweighted; never refund budget.

For every prompted structure k and frame t, let S be raw SAM probability and F the frozen framewise probability. Define ROI = (S>.5) union (F>.5); if empty, use the whole image. In this ROI compute normalized entropies Hs/Hf and disagreement Dsf=mean(|S-F|). Set **w=(1-Hs)*exp(-Dsf*(1-Hf)/delta)**, delta=.25, and output **P=w*S+(1-w)*F**. Clamp w to [0,1]. Unprompted structures remain exactly F; restore every charged prompt frame to its selected mask after blending. Weights depend only on observable predictions; no gold or clinical quality score enters them.

This is a heuristic trust weight, not calibrated uncertainty or a verified mask-quality estimator. Confidently wrong F can suppress a beneficial SAM correction; confidently wrong SAM can still pass. Empty-mask handling and foreground ROI can introduce biases. Report these risks and do not claim the fallback cannot harm Dice.

## Matched comparison registry

Six fixed recipes: uncertainty/raw; uncertainty/fallback; error/raw; scaled-diversity/raw; scaled-diversity/fallback (prototype); legacy-full/raw. Same inputs/models/budgets; compare quality on identical scoring frames. Factor contrasts: scaled/raw minus error/raw isolates relative diversity; uncertainty/fallback minus uncertainty/raw isolates fallback under that selector. Prototype minus uncertainty/raw combines two changes and cannot by itself establish either component's contribution. Legacy full is an anchor, not the sole comparator. No lambda/delta grid in this version.

## Execution sequence and interpretation

1. Unit tests and CPU synthetic smoke must pass, including no-gold API, bounded weights, exact zero-budget identity, exact charged masks, no unprompted changes, nonnegative relative scores, and old-policy equivalence.
2. Freeze protocol/config/source hashes. Engineering replay may then use 059/086, with debug, one actual GPU model and runtime. Record both requested and actual GPU; do not treat A10 and A10G as numerically interchangeable. Do not use replay outcomes to silently change parameters.
3. A later explicitly recorded development run may evaluate all nine previously used training sources. All such results are exploratory; analyze original and new recipes, including negative outcomes. If code changes, invalidate the previous freeze and create a new version before rerunning.
4. The three validation sources were already inspected, so further clips from those sources are extended validation, not unseen-procedure confirmation. No independence claim from unused clip IDs. Final test remains locked; do not use test gains to choose mechanism.

Primary computational endpoint remains procedure-macro normalized Dice AUC with paired procedure bootstrap; report IoU, present-only and propagation-harm diagnostics. Debug intermediate counts 3/4 are observational, not a replacement endpoint. No clinical utility, annotation-time saving or significance claim from the two-case engineering replay. Do not tune acceptance weights on gold, including scoring masks already viewed.

## Current delivery boundary

This delivery implements and tests the prototype and supplies an explicitly runnable launcher. New-method GPU execution is not automatically started by preparation scripts or tests. No final-test permission is changed. User authorization for development/Modal persists; this boundary is staging of the requested design, not a new permission requirement.
