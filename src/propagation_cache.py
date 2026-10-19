"""Bounded CPU cache of already-charged, structure-specific propagation outputs.

The caller supplies an immutable clip/model context and the exact selected prompt
set. No gold label may be read to form a key before its query is selected.
"""
from collections import OrderedDict
import numpy as np

class PropagationCache:
    def __init__(self,max_bytes=10*1024**3):
        if max_bytes<0:raise ValueError('Invalid cache bound')
        self.max_bytes=int(max_bytes);self.bytes=0;self.entries=OrderedDict();self.hits=0;self.misses=0

    def get(self,key):
        if key not in self.entries:self.misses+=1;return None
        self.hits+=1;self.entries.move_to_end(key);return self.entries[key]

    def put(self,key,value):
        value=np.asarray(value)
        if value.nbytes>self.max_bytes:return
        if key in self.entries:self.bytes-=self.entries.pop(key).nbytes
        while self.entries and self.bytes+value.nbytes>self.max_bytes:
            _,old=self.entries.popitem(last=False);self.bytes-=old.nbytes
        saved=value.copy();saved.flags.writeable=False
        self.entries[key]=saved;self.bytes+=saved.nbytes
