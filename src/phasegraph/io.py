"""Versioned graph interchange. JSON preserves isolated regions; edge CSV cannot."""
from pathlib import Path
import hashlib
import json
import math
import networkx as nx
import pandas as pd

SCHEMA = "phasegraph.graph/1"

def validate_graph(g):
    if g.is_directed() or g.is_multigraph() or not len(g):
        raise ValueError("Expected a nonempty simple undirected graph")
    if any(not isinstance(n, str) or not n for n in g):
        raise ValueError("Node identifiers must be nonempty strings")
    for u, v, data in g.edges(data=True):
        d = data.get("edge_distance")
        if u == v or isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
            raise ValueError(f"Invalid edge {u!r}, {v!r}: a finite positive cost is required")
    return g

def graph_record(g):
    validate_graph(g)
    return {"schema": SCHEMA, "metadata": dict(g.graph),
            "nodes": [{"id": n, **g.nodes[n]} for n in sorted(g)],
            "edges": [{"source": u, "target": v, **g.edges[u, v]} for u, v in sorted(tuple(sorted(e)) for e in g.edges)]}

def graph_digest(g):
    # Provenance fingerprint: node IDs and geometric attributes are retained.
    payload = graph_record(g)
    payload.pop("metadata")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write_graph(g, path):
    Path(path).write_text(json.dumps(graph_record(g), indent=2, allow_nan=False), encoding="utf-8")

def read_graph(path):
    path = Path(path)
    g = nx.Graph(name=path.stem)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported graph schema in {path}")
        g.graph.update(data.get("metadata", {}))
        for node in data["nodes"]:
            node = dict(node); name = node.pop("id")
            if name in g:
                raise ValueError(f"Duplicate node {name}")
            g.add_node(name, **node)
        for edge in data["edges"]:
            edge = dict(edge); u = edge.pop("source"); v = edge.pop("target")
            if u not in g or v not in g or g.has_edge(u, v):
                raise ValueError("Undefined node or duplicate edge in graph JSON")
            g.add_edge(u, v, **edge)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"node_u": str, "node_v": str}, keep_default_na=False)
        if not {"node_u", "node_v", "edge_distance"}.issubset(df.columns):
            raise ValueError(f"Not an edge CSV: {path}")
        for row in df.to_dict("records"):
            u, v = row["node_u"], row["node_v"]
            if g.has_edge(u, v):
                raise ValueError("Duplicate undirected edge; resolve it before loading")
            g.add_edge(u, v, edge_distance=float(row["edge_distance"]))
        g.graph.update(source=str(path), interchange="edge_csv_without_isolate_manifest")
    else:
        raise ValueError("Expected a graph JSON or adjacency edge CSV")
    return validate_graph(g)

def read_collection(folder):
    folder = Path(folder)
    if folder.is_file():
        return [read_graph(folder)]
    jsons = sorted(folder.rglob("*.json"))
    paths = []
    for p in jsons:
        try:
            if json.loads(p.read_text(encoding="utf-8-sig")).get("schema") == SCHEMA:
                paths.append(p)
        except (ValueError, AttributeError):
            continue
    # Extraction output contains both JSON and CSV: prefer explicit node manifests.
    if not paths:
        for p in sorted(folder.rglob("*.csv")):
            if {"node_u", "node_v", "edge_distance"}.issubset(pd.read_csv(p, nrows=0).columns):
                paths.append(p)
    graphs = [read_graph(p) for p in paths]
    names = [g.graph["name"] for g in graphs]
    if not graphs or len(set(names)) != len(names):
        raise ValueError("No graph inputs, or duplicate graph names")
    return graphs
