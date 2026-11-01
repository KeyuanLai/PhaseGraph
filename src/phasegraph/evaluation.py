"""Region matching and contact scoring for labelled benchmark masks."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def match_regions(reference, predicted, reference_ids, predicted_ids, threshold=.5):
    """Maximum-IoU one-to-one matching, preserving absent/background labels."""
    if reference.shape!=predicted.shape:
        raise ValueError('Reference and prediction masks must have equal shape')
    reference_ids=list(reference_ids);predicted_ids=list(predicted_ids)
    if not reference_ids or not predicted_ids:return {},np.zeros((len(predicted_ids),len(reference_ids)))
    m=int(max(reference.max(initial=0),max(reference_ids)))+1
    n=int(max(predicted.max(initial=0),max(predicted_ids)))+1
    joint=np.bincount(predicted.ravel().astype(np.int64)*m+reference.ravel(),minlength=m*n).reshape(n,m)
    inter=joint[np.ix_(predicted_ids,reference_ids)]
    union=joint.sum(1)[predicted_ids,None]+joint.sum(0)[None,reference_ids]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    ir,jr=linear_sum_assignment(-iou)
    return {predicted_ids[i]:reference_ids[j] for i,j in zip(ir,jr) if iou[i,j]>=threshold},iou

def score_contacts(reference, predicted, mapping, reference_nodes, predicted_nodes):
    expected={tuple(sorted(map(int,e))) for e in reference}
    got={tuple(sorted((mapping.get(int(u),-int(u)),mapping.get(int(v),-int(v))))) for u,v in predicted}
    tp=len(got&expected);fp=len(got-expected);fn=len(expected-got)
    return {'node_F1':2*len(mapping)/(reference_nodes+predicted_nodes),
            'edge_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'exact_graph_recovery':len(mapping)==reference_nodes==predicted_nodes and got==expected,
            'region_count_absolute_error':abs(reference_nodes-predicted_nodes)}
