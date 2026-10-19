"""Decode the audited AutoLaparo grayscale-in-RGB release into two task targets.

No individual instrument type is inferred from the order of the eight part IDs.
Uterus=180 follows the archived release decoder; overlay review remains required.
"""
import numpy as np
PALETTE=set(range(0,181,20))
def decode_targets(raw):
    raw=np.asarray(raw)
    if raw.ndim==3:
        if raw.shape[-1]!=3 or not np.array_equal(raw[...,0],raw[...,1]) or not np.array_equal(raw[...,1],raw[...,2]):raise ValueError('Unknown RGB color encoding')
        raw=raw[...,0]
    if raw.ndim!=2 or raw.dtype!=np.uint8:raise ValueError('Expected uint8 label raster')
    unknown=set(np.unique(raw).tolist())-PALETTE
    if unknown:raise ValueError('Unknown AutoLaparo palette: '+str(sorted(unknown)))
    return np.stack([raw==180,(raw>0)&(raw<180)])
