"""Pure controls for the preregistered paired precision diagnostic."""
def apply_matched_flags(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pair_factor(a, b):
    if a['clip_id'] != b['clip_id']:
        return None
    if a['pair_id'] == b['pair_id'] and a['precision'] != b['precision']:
        return 'within_device_precision'
    if a['precision'] == b['precision'] and a['repeat'] != b['repeat']:
        return 'same_precision_repeat'
    return None
