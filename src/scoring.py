"""Policy-independent temporal exclusion for matched sensitivity analyses."""
import numpy as np
def common_scoring(scoring_ids,all_query_frames,fps,seconds):
    if fps<=0 or seconds<0:raise ValueError('Invalid temporal exclusion')
    s=np.asarray(scoring_ids,dtype=int);q=set(map(int,all_query_frames))
    if len(set(s))!=len(s):raise ValueError('Duplicate scoring frame IDs')
    keep=np.array([t for t in s if all(abs(int(t)-a)>fps*seconds for a in q)],dtype=int)
    if not len(keep):raise ValueError('No common scoring frames remain; add independent reference annotations')
    return keep
