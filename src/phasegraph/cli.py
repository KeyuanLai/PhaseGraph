"""Each run records its command, configuration, result and provenance."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import shlex
import sys
import time
import cv2
import numpy as np
import pandas as pd
from . import __version__
from .distance import DistanceModel
from .io import read_graph, read_collection, write_graph
from .geometry import extract_image
from .analysis import cluster_matrix
from .stability import query_hypotheses

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="phasegraph", description="Binary phase diagram geometry and graph comparison")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["extract", "matrix", "fit", "query", "cluster"]:
        p = sub.add_parser(command); p.add_argument("--out", required=True, type=Path)
        if command == "extract":
            p.add_argument("--images", type=Path, required=True); p.add_argument("--no-repair", action="store_true")
            p.add_argument('--repair-strategy',choices=['none','local','global'])
            p.add_argument('--canonical-width',type=int,help='Optional canvas width before pixel-scale morphology; preserves aspect ratio')
            p.add_argument('--coordinate-metric',choices=['diagonal','axes'],default='diagonal',help='Historical image layout, or independent axis normalisation; use one profile consistently per gallery')
        elif command in {"matrix", "fit"}:
            p.add_argument("--graphs", type=Path, required=True); p.add_argument("--bins", type=int, default=30)
            p.add_argument("--weights", type=float, nargs=3, default=[.45,.45,.1]); p.add_argument("--overflow", choices=["error","clip","tail"], default="error")
            p.add_argument('--encoding',choices=['historical','continuous'],default='historical')
        elif command == "query":
            p.add_argument("--model", type=Path, required=True)
            query_input = p.add_mutually_exclusive_group(required=True)
            query_input.add_argument("--graph", type=Path)
            query_input.add_argument('--hypotheses',type=Path,help='Graphs from alternative extractions of one query image; all are required')
            p.add_argument('--nominal-hypothesis',help='Graph name defining the nominal shortlist; otherwise use a unique local-repair graph, then the first graph')
            p.add_argument("--top-k", type=int, default=5)
        else:
            p.add_argument("--matrix", type=Path, required=True); p.add_argument("--groups", type=int, required=True)
            p.add_argument("--dimensions", type=int, default=2); p.add_argument("--method", choices=["ward","kmeans"], default="ward"); p.add_argument("--seed", type=int, default=42)
    sub.add_parser("doctor")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps({"phasegraph": __version__, "python": sys.version.split()[0], "dependencies": {n: importlib.metadata.version(n) for n in ["numpy","pandas","networkx","opencv-python","scipy","scikit-learn"]}, "OCR_required": False, "thinning": "vendored NumPy Zhang-Suen"}, indent=2)); return
    out = args.out
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; select a new run directory")
    out.mkdir(parents=True,exist_ok=True)
    command = ["phasegraph", *(argv if argv is not None else sys.argv[1:])]
    (out/"command.txt").write_text(shlex.join(map(str,command)),encoding="utf-8")
    # JSON is a YAML 1.2-compatible representation and avoids an extra dependency.
    dump(out/"config.yaml", {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    start = time.perf_counter()
    try:
        if args.command == "extract":
            images = [args.images] if args.images.is_file() else sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"})
            if not images:
                raise ValueError("No input images")
            stems = [p.stem.removesuffix("_1") for p in images]
            if len(set(stems)) != len(stems):
                raise ValueError("Duplicate image names; process collections separately")
            rows = []
            for image in images:
                graph, debug = extract_image(image, repair=not args.no_repair,
                                              repair_strategy=args.repair_strategy,canonical_width=args.canonical_width,coordinate_metric=args.coordinate_metric)
                folder = out/graph.graph["name"]; folder.mkdir()
                write_graph(graph,folder/"graph.json")
                pd.DataFrame([{"id":n,**a} for n,a in graph.nodes(data=True)]).to_csv(folder/"nodes.csv",index=False)
                pd.DataFrame([{"node_u":u,"node_v":v,**a} for u,v,a in graph.edges(data=True)],columns=["node_u","node_v","edge_distance","shared_boundary_px"]).to_csv(folder/"adjacency_edges.csv",index=False)
                for key in ["raw","fixed","wall"]:
                    ok, data = cv2.imencode(".png",debug[key])
                    if not ok: raise ValueError("Mask encoding failed")
                    (folder/(key+".png")).write_bytes(data.tobytes())
                np.savez_compressed(folder/"regions.npz", labels=debug["labels"],roi=debug["roi"])
                dump(folder/"extraction_config.json", debug["config"])
                rows.append({"system":graph.graph["name"],"nodes":len(graph),"edges":graph.number_of_edges(),"source_sha256":graph.graph["source_sha256"]})
            result = pd.DataFrame(rows)
        elif args.command in {"matrix","fit"}:
            graphs = read_collection(args.graphs)
            model = DistanceModel(args.bins,args.weights,args.overflow,args.encoding).fit(graphs)
            model.save(out/"model.json")
            matrix = pd.DataFrame(model.matrix(),index=model.names,columns=model.names)
            matrix.to_csv(out/"distance_matrix.csv")
            result = pd.DataFrame({"system":model.names,"nodes":[len(g) for g in graphs],"edges":[g.number_of_edges() for g in graphs]})
        elif args.command == "query":
            if args.top_k < 1: raise ValueError("top-k must be positive")
            model = DistanceModel.load(args.model)
            if args.hypotheses:
                hypotheses=read_collection(args.hypotheses)
                nominal=[i for i,g in enumerate(hypotheses) if g.graph.get('name')==args.nominal_hypothesis] if args.nominal_hypothesis else [i for i,g in enumerate(hypotheses) if g.graph.get('repair_strategy')=='local']
                if args.nominal_hypothesis and len(nominal)!=1:raise ValueError('Nominal hypothesis name must match one supplied graph')
                report = query_hypotheses(model,hypotheses,top_k=args.top_k,nominal=nominal[0] if len(nominal)==1 else 0)
                dump(out/'stability.json',report)
                if report['status'] != 'complete':
                    raise ValueError('An extraction hypothesis failed; stability certification is blocked; see stability.json')
                result = pd.DataFrame(report['candidates']).sort_values('nominal_rank').head(args.top_k)
            else:
                result = pd.DataFrame(model.query(read_graph(args.graph))).head(args.top_k)
        else:
            result, coords, diag = cluster_matrix(pd.read_csv(args.matrix,index_col=0),args.groups,args.dimensions,args.method,args.seed)
            coords.to_csv(out/"pcoa_coordinates.csv"); dump(out/"diagnostics.json",diag)
        result.to_csv(out/"result.csv",index=False)
        dump(out/"run.json",{"status":"complete","version":__version__,"elapsed_seconds":time.perf_counter()-start,"command":command})
        (out/"run.log").write_text(f"complete: {len(result)} output records\n",encoding="utf-8")
        print(f"{args.command}: complete -> {out}")
    except Exception as exc:
        dump(out/"run.json",{"status":"failed","error":str(exc),"version":__version__})
        (out/"run.log").write_text(str(exc),encoding="utf-8")
        parser.exit(1, f"phasegraph: {exc}\n")
