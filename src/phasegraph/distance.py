"""Historical geometric D equations with explicit frozen calibration and validation."""
from pathlib import Path
import json
import numpy as np
import networkx as nx
from .io import validate_graph, graph_digest

def entropy(p):
    p = np.asarray(p); p = p[p > 0]
    return -float(np.dot(p, np.log(p)))

def js_root(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sqrt(max(0., (entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2) / np.log(2))))

def paths(g):
    validate_graph(g)
    return np.asarray(nx.floyd_warshall_numpy(g, nodelist=sorted(g), weight="edge_distance"), float)

class DistanceModel:
    """Fit on a collection, or freeze a gallery and compare later query graphs.

    Out-of-range finite paths raise an error by default. Explicit clipping is
    available for diagnostics, and its occurrence is returned with each query.
    """
    def __init__(self, bins=30, weights=(.45, .45, .10), overflow="error", encoding='historical'):
        if not isinstance(bins, int) or bins < 2:
            raise ValueError("At least two finite bins are required")
        if len(weights) != 3 or not np.all(np.isfinite(weights)) or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError("Three nonnegative weights summing to one are required")
        if overflow not in {"error", "clip", "tail"}:
            raise ValueError("overflow must be error, clip or tail")
        if encoding not in {'historical','continuous'}:
            raise ValueError('encoding must be historical or continuous')
        self.bins = bins; self.weights = tuple(float(x) for x in weights); self.overflow = overflow
        self.encoding = encoding

    def fit(self, graphs):
        graphs = list(graphs)
        if not graphs:
            raise ValueError("An empty gallery cannot be fitted")
        names = [g.graph.get("name", f"graph_{i}") for i, g in enumerate(graphs)]
        if len(set(names)) != len(names):
            raise ValueError("Gallery graph names must be unique")
        profiles = {g.graph.get('coordinate_metric', 'diagonal') for g in graphs}
        if not profiles <= {'diagonal', 'axes'} or len(profiles) != 1:
            raise ValueError('Gallery graphs must use one coordinate_metric: diagonal or axes')
        self.coordinate_metric = next(iter(profiles))
        matrices = [paths(g) for g in graphs]
        maxima = [float(d[np.isfinite(d)].max()) for d in matrices]
        self.breaks = np.linspace(0, max(max(maxima), 1e-12) * 1.000001, self.bins + 1)
        costs = [float(d["edge_distance"]) for g in graphs for _, _, d in g.edges(data=True)]
        self.sigma = float(np.median(costs)) if costs else 1.
        self.names = names
        self.fingerprints = [graph_digest(g) for g in graphs]
        self.features = [self.encode(g, d) for g, d in zip(graphs, matrices)]
        return self

    def encode(self, g, matrix=None):
        if not hasattr(self, "breaks"):
            raise ValueError("Fit or load a distance model first")
        if g.graph.get('coordinate_metric', 'diagonal') != self.coordinate_metric:
            raise ValueError('Query coordinate_metric differs from the frozen gallery; recompute under one convention')
        d = paths(g) if matrix is None else matrix
        n = len(g); finite = d[np.isfinite(d)]
        exceeded = bool(np.any(finite > self.breaks[-1]))
        if exceeded and self.overflow == "error":
            raise ValueError(f"Query paths exceed frozen support {self.breaks[-1]:.8g}; explicit recalibration or diagnostic overflow='clip' is required")
        finite_columns=self.bins+(self.overflow=='tail')
        pdf = np.zeros((n, finite_columns + 1))
        if n == 1:
            pdf[0, 0] = 1
        else:
            for i in range(n):
                row = np.delete(d[i], i); values = row[np.isfinite(row)]
                if self.encoding == 'historical':
                    ids = np.clip(np.searchsorted(self.breaks, values, side="left") - 1, 0, finite_columns - 1)
                    pdf[i, :finite_columns] = np.bincount(ids, minlength=finite_columns)
                else:
                    width = self.breaks[1]-self.breaks[0]
                    coordinate = np.clip((values-(self.breaks[0]+width/2))/width,0,self.bins-1)
                    left = np.floor(coordinate).astype(int); right = np.minimum(left+1,self.bins-1)
                    fraction = coordinate-left
                    tail_weight = np.clip((values-self.breaks[-1])/width,0,1) if self.overflow=='tail' else np.zeros(len(values))
                    pdf[i,:self.bins] = np.bincount(left,weights=(1-fraction)*(1-tail_weight),minlength=self.bins)
                    pdf[i,:self.bins] += np.bincount(right,weights=fraction*(1-tail_weight),minlength=self.bins)
                    if self.overflow=='tail':pdf[i,self.bins]=tail_weight.sum()
                pdf[i, -1] = np.count_nonzero(~np.isfinite(row))
                pdf[i] /= n - 1
        mean = pdf.mean(axis=0)
        normalizer = np.log(max(2,np.count_nonzero(mean))) if self.encoding=='historical' else np.log(self.bins+2)
        nnd = max(0., entropy(mean) - np.mean([entropy(p) for p in pdf])) / normalizer
        a = np.zeros((n, n)); order = {v: i for i, v in enumerate(sorted(g))}
        for u, v, e in g.edges(data=True):
            i, j = order[u], order[v]; a[i, j] = a[j, i] = np.exp(-e["edge_distance"] / self.sigma)
        radius = float(np.max(np.abs(np.linalg.eigvalsh(a))))
        strength = a.sum(1); exo = strength / strength.sum() if strength.sum() else np.full(n, 1/n)
        alpha = .85/radius if radius > 1e-12 else 0.
        centrality = np.maximum(np.linalg.solve(np.eye(n) - alpha*a.T, exo), 0)
        centrality /= centrality.sum()
        return {"mean": mean.tolist(), "nnd": float(nnd), "alpha": np.sort(centrality).tolist(), "overflow": exceeded}

    def compare_features(self, a, b):
        n = max(len(a["alpha"]), len(b["alpha"]))
        ca = np.pad(a["alpha"], (n-len(a["alpha"]), 0)); cb = np.pad(b["alpha"], (n-len(b["alpha"]), 0))
        parts = [js_root(a["mean"], b["mean"]), abs(np.sqrt(a["nnd"])-np.sqrt(b["nnd"])), js_root(ca, cb)]
        return float(np.dot(self.weights, parts))

    def matrix(self):
        n = len(self.features); d = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                d[i, j] = d[j, i] = self.compare_features(self.features[i], self.features[j])
        return d

    def query(self, g):
        f = self.encode(g)
        scores = [{"name": name, "distance": self.compare_features(f, ref), "query_overflow": f["overflow"]} for name, ref in zip(self.names, self.features)]
        return sorted(scores, key=lambda x: (round(x["distance"],12), x["name"]))

    def save(self, path):
        data = {"schema": "phasegraph.distance/1", "bins": self.bins, "weights": self.weights,
                "overflow": self.overflow, "encoding": self.encoding, "coordinate_metric": self.coordinate_metric, "breaks": self.breaks.tolist(), "sigma": self.sigma,
                "names": self.names, "fingerprints": self.fingerprints, "features": self.features}
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != "phasegraph.distance/1":
            raise ValueError("Unsupported distance model schema")
        model = cls(data["bins"], data["weights"], data["overflow"],data.get('encoding','historical'))
        model.coordinate_metric = data.get('coordinate_metric', 'diagonal')
        if model.coordinate_metric not in {'diagonal', 'axes'}:
            raise ValueError('Invalid saved coordinate_metric')
        model.breaks = np.asarray(data["breaks"], float); model.sigma = float(data["sigma"])
        model.names = data["names"]; model.fingerprints = data["fingerprints"]; model.features = data["features"]
        if len(model.breaks) != model.bins+1 or not np.all(np.isfinite(model.breaks)) or not np.all(np.diff(model.breaks) > 0) or not np.isfinite(model.sigma) or model.sigma <= 0:
            raise ValueError("Invalid model calibration")
        if not model.names or len(set(model.names)) != len(model.names) or not len(model.names) == len(model.fingerprints) == len(model.features):
            raise ValueError("Invalid model gallery")
        for feature in model.features:
            mean = np.asarray(feature["mean"]); alpha = np.asarray(feature["alpha"])
            if len(mean) != model.bins+1+(model.overflow=='tail') or len(alpha) < 1 or not np.all(np.isfinite(mean)) or np.min(mean) < 0 or not np.isclose(mean.sum(), 1) or not np.all(np.isfinite(alpha)) or np.min(alpha) < 0 or not np.isclose(alpha.sum(), 1) or not np.isfinite(feature["nnd"]) or feature["nnd"] < 0:
                raise ValueError("Invalid cached graph feature")
        return model
