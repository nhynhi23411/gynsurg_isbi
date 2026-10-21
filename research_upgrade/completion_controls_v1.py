"""Explicit, limited close-method adaptations. Not upstream system reproductions."""
from pathlib import Path
import importlib.util
import sys
import time
import types
import numpy as np
from research_upgrade.bank import QUERY, SCORE, BUDGETS, entropy, popcount


def eva_action(embeddings, current, paid):
    """Original farthest-from-nearest L2 rule, restricted to charged head/frame pool.

    Unseeded heads use entropy first. No future mask or scoring label is accepted.
    Embeddings are recomputed from the current paid-state predictions.
    """
    eligible = [(j, k) for j in range(3) for k in range(2) if not paid[k] & (1 << j)]
    if not eligible:
        raise ValueError('No unpaid action')
    unseeded = [(j, k) for j, k in eligible if paid[k] == 0]
    if unseeded:
        return max(unseeded, key=lambda z: (entropy(current[QUERY[z[0]], z[1]]), -QUERY[z[0]], -z[1]))

    def value(item):
        j, k = item
        known = [a for a in range(3) if paid[k] & (1 << a)]
        distance = min(float(np.linalg.norm(embeddings[k][j] - embeddings[k][a])) for a in known)
        if not np.isfinite(distance):
            raise ValueError('Nonfinite QNet distance')
        return distance, -QUERY[j], -k
    return max(eligible, key=value)


def predicted_reset_schedule(probabilities, period=25, persistence=4):
    """Two binary semantic heads, predicted-mask presence only; fixed code defaults.

    Periodic resets and newly appearing classes are adaptations of the inspected
    SASVi code. No instances, detections, gold masks, or scorer validity are inputs.
    Four-frame lookahead is explicit; this is not a causal streaming control.
    """
    p = np.asarray(probabilities)
    if p.ndim != 4 or p.shape[:2] != (125, 2) or not np.isfinite(p).all():
        raise ValueError('Expected finite frozen 125-frame two-head input')
    presence = (p > .5).any(axis=(2, 3))
    schedules = []
    for head in range(2):
        resets = [0]
        old_present = bool(presence[0, head])
        for frame in range(1, 125):
            stable_new = (not old_present and frame + persistence <= 125
                          and bool(presence[frame:frame + persistence, head].all()))
            if frame - resets[-1] >= period or stable_new:
                resets.append(frame)
            old_present = bool(presence[frame, head])
        schedules.append(resets)
    return schedules


def load_qnet(code_root, weights):
    """Load unchanged upstream architecture with a strict complete trained state.

    Disable only redundant ImageNet initialization downloads; the official QNet
    checkpoint replaces every learned parameter and persistent buffer strictly.
    """
    import torch
    import torchvision.models
    root = Path(code_root) / 'models'
    package = types.ModuleType('eva_reference_private')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    for name in ['modules', 'qnet']:
        spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if name == 'modules':
            module.resnet18 = lambda weights=None: torchvision.models.resnet18(weights=None)
    qnet = sys.modules['eva_reference_private.qnet'].QualityNet()
    state = torch.load(weights, map_location='cpu', weights_only=True)
    qnet.load_state_dict(state, strict=True)
    return qnet.cuda().eval()


class Embeddings:
    def __init__(self, qnet, jpeg_dir):
        import torch
        from PIL import Image
        from torchvision.transforms import Normalize, Resize, InterpolationMode
        frames = sorted(Path(jpeg_dir).glob('*.jpg'))
        if len(frames) != 125:
            raise ValueError('Incomplete frames')
        tensors = []
        norm = Normalize([.485, .456, .406], [.229, .224, .225])
        resize = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
        for frame in QUERY:
            rgb = np.array(Image.open(frames[frame]).convert('RGB'))
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
            tensors.append(resize(norm(x)))
        self.rgb = torch.stack(tensors).cuda()
        self.qnet = qnet

    def __call__(self, current):
        import torch
        from torchvision.transforms import Resize, InterpolationMode
        resize = Resize((224, 224), interpolation=InterpolationMode.NEAREST, antialias=True)
        result = []
        with torch.inference_mode():
            for head in range(2):
                masks = torch.from_numpy((current[list(QUERY), head] > .5).astype(np.float32)).cuda()
                masks = resize(masks)[:, None].repeat(1, 3, 1, 1)
                result.append(self.qnet.extract_features(self.rgb, masks).float().cpu().numpy())
        if any(x.shape != (3, 1024) or not np.isfinite(x).all() for x in result):
            raise ValueError('Unexpected QNet embedding shape/domain')
        return result


def run_eva(baseline, engine, purchase_mask, embedding):
    current = np.asarray(baseline).copy()
    paid = [0, 0]
    prompts, queries, calls = {}, [], []
    output = {0: current[list(SCORE)].copy()}
    start = time.perf_counter()
    selection_seconds = 0.
    for n in range(1, 6):
        before = time.perf_counter()
        # With an unseeded head, first-query entropy needs no QNet feature.
        features = None if 0 in paid else embedding(current)
        j, head = eva_action(features, current, paid)
        selection_seconds += time.perf_counter() - before
        frame = QUERY[j]
        mask = np.asarray(purchase_mask(frame, head))
        if mask.shape != current.shape[2:] or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid purchased mask')
        prompts[head, j] = mask.copy()
        paid[head] |= 1 << j
        existing = [(QUERY[a], head, prompts[head, a]) for a in range(3) if paid[head] & (1 << a)]
        engine.prompts = [(t, k, m.copy()) for t, k, m in existing[:-1]]
        before = time.perf_counter()
        pred = engine.correct(*existing[-1])
        calls.append(dict(head=head, subset=paid[head], seconds=time.perf_counter() - before))
        current[:, head] = pred[:, head]
        del pred
        queries.append(dict(query=n, frame=frame, head=head, paid_subset=paid[head]))
        if n in BUDGETS:
            output[n] = current[list(SCORE)].copy()
    return dict(predictions=output, queries=queries, propagations=calls, paid_masks=5,
                predicted_prompts=0, qnet_selection_seconds=selection_seconds,
                runtime=dict(total_seconds=time.perf_counter() - start,
                             propagation_seconds=sum(r['seconds'] for r in calls)),
                upstream_reproduction=False, adaptation='EVA-VOS pretrained QNet frame selection, mask-only SAM2')


def run_predicted(baseline, engine, segment_factory=None):
    """Zero-gold predicted-prompt endpoint; never represented as a paid-mask AUC."""
    baseline = np.asarray(baseline)
    schedules = predicted_reset_schedule(baseline)
    output = baseline[list(SCORE)].copy()
    calls = []
    start = time.perf_counter()
    for head, resets in enumerate(schedules):
        for i, frame in enumerate(resets):
            end = resets[i + 1] if i + 1 < len(resets) else 125
            # Compute every reset, including segments without scoring snapshots.
            # Thus call count is not reduced by looking at the evaluation pool.
            mask = baseline[frame, head] > .5
            before = time.perf_counter()
            if segment_factory is None:
                engine.prompts = []
                prediction = engine.correct(frame, head, mask)
                relative_start = 0
            else:
                # Process each segment, rather than repeatedly processing the
                # complete 125-frame video after every automatic reset.
                segment, cleanup = segment_factory(frame, end)
                try:
                    prediction = segment.correct(0, head, mask)
                finally:
                    cleanup()
                relative_start = frame
            calls.append(dict(head=head, frame=frame, segment_end=end,
                              prompt_foreground_pixels=int(mask.sum()),
                              processed_frames=end-frame if segment_factory else 125,
                              seconds=time.perf_counter() - before))
            for index, t in enumerate(SCORE):
                if frame <= t < end:
                    output[index, head] = prediction[t-relative_start, head]
            del prediction
    return dict(predictions={0: output}, queries=[], propagations=calls, paid_masks=0,
                predicted_prompts=len(calls), reset_schedule=schedules,
                runtime=dict(total_seconds=time.perf_counter() - start,
                             propagation_seconds=sum(r['seconds'] for r in calls)),
                upstream_reproduction=False, adaptation='SASVi-inspired semantic predicted-mask reset/stitch control',
                gold_budget_auc_defined=False, causal_streaming=False)
