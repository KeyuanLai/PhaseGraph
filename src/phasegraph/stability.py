"""Decision stability over explicit, coupled extraction hypotheses.

Certificates quantify a finite sensitivity set, never probability of correct
segmentation or coverage of an unknown true graph. Every hypothesis is retained.
"""
import numpy as np


def assess_distances(distances, names, *, top_k=3, nominal=0, tolerance=1e-10):
    """Compare a coupled-hypothesis certificate with an independent interval box.

    A positive coupled margin preserves the nominal top-k *set* under each
    supplied row. A positive box margin also preserves it under every independent
    combination of column intervals, a stronger and more conservative condition.
    Within-set ordering is not certified. Ties receive no positive certificate.
    """
    d = np.asarray(distances, float)
    names = list(names)
    if d.ndim != 2 or d.shape[0] < 1 or d.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('A nonempty hypothesis-by-candidate matrix with unique names is required')
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('All hypothesis distances must be finite and nonnegative')
    if not isinstance(top_k, int) or not 1 <= top_k < len(names) or not 0 <= nominal < len(d):
        raise ValueError('Require 1 <= top_k < gallery size and a valid nominal hypothesis')
    if not np.isfinite(tolerance) or tolerance < 1e-12:
        raise ValueError('Tolerance must be finite and at least the ranking resolution 1e-12')
    rounded = np.round(d, 12)
    orders = np.array([sorted(range(len(names)), key=lambda j: (row[j], names[j])) for row in rounded])
    ranks = np.empty_like(orders)
    for row in range(len(d)):
        ranks[row, orders[row]] = np.arange(1, len(names)+1)
    selected = orders[nominal, :top_k]
    outside = orders[nominal, top_k:]
    lower, upper = rounded.min(0), rounded.max(0)
    coupled_margin = float(np.min(rounded[:, outside].min(1)-rounded[:, selected].max(1)))
    box_margin = float(lower[outside].min()-upper[selected].max())
    top_sets = [set(order[:top_k]) for order in orders]
    core = set.intersection(*top_sets)
    union = set.union(*top_sets)
    nominal_set = set(selected)
    spread = float(np.max(np.ptp(d, axis=0)))
    return {'hypotheses': len(d), 'gallery_size': len(names), 'top_k': top_k, 'nominal_hypothesis': nominal,
            'nominal_top_k': [names[j] for j in selected],
            'stable_core': sorted(names[j] for j in core),
            'possible_top_k': sorted(names[j] for j in union),
            'coupled_margin': coupled_margin, 'box_margin': box_margin,
            'coupled_certified': coupled_margin > tolerance, 'box_certified': box_margin > tolerance,
            'observed_sets_identical': all(t == nominal_set for t in top_sets),
            'minimum_nominal_overlap': min(len(t & nominal_set)/top_k for t in top_sets),
            'max_distance_spread': spread,
            'nominal_gap': float(rounded[nominal, orders[nominal, top_k]]-rounded[nominal, orders[nominal, top_k-1]]),
            'tolerance': tolerance, 'scope': 'supplied finite hypotheses; not graph-truth confidence',
            'candidates': [{'name': name, 'nominal_rank': int(ranks[nominal, j]),
                            'best_rank': int(ranks[:, j].min()), 'worst_rank': int(ranks[:, j].max()),
                            'distance_min': float(d[:, j].min()), 'distance_max': float(d[:, j].max()),
                            'nominal_distance': float(d[nominal, j])} for j, name in enumerate(names)]}


def query_hypotheses(model, hypotheses, *, top_k=3, nominal=0, exclude=()):
    """Use the frozen model; any failed hypothesis blocks certification."""
    hypotheses = list(hypotheses)
    if not hypotheses:
        raise ValueError('At least one extraction hypothesis is required')
    names = [name for name in model.names if name not in set(exclude)]
    rows, failures = [], []
    for i, graph in enumerate(hypotheses):
        try:
            values = {r['name']: r['distance'] for r in model.query(graph)}
            rows.append([values[name] for name in names])
        except ValueError as error:
            failures.append({'hypothesis': i, 'name': graph.graph.get('name', ''), 'error': str(error)})
    if failures:
        return {'status': 'incomplete', 'expected_hypotheses': len(hypotheses),
                'successful_hypotheses': len(rows), 'failures': failures,
                'coupled_certified': False, 'box_certified': False,
                'scope': 'a missing hypothesis blocks certification'}
    result = assess_distances(rows, names, top_k=top_k, nominal=nominal)
    result.update(status='complete', hypothesis_names=[g.graph.get('name', str(i)) for i, g in enumerate(hypotheses)])
    return result
