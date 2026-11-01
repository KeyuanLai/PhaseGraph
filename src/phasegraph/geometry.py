"""OCR-free phase-boundary extraction and region geometry."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import math
import cv2
import numpy as np
import networkx as nx
from ._vendor import extraction_kernel as cv

# A format screen, not a confidence or accuracy threshold. A candidate boundary
# mask occupying most of a canvas is incompatible with thin-line extraction.
MAX_BOUNDARY_COVERAGE = .35


def boundary_coverage(bgr):
    mask = cv.colored_phase_mask(bgr, cv.PreprocessConfig(use_text_hint=False, paddle_device='cpu'))
    return float(np.count_nonzero(mask)/mask.size)

def coordinate_cost(delta, size, metric='diagonal'):
    """Normalise a point displacement in a declared plot coordinate frame.

    ``axes`` removes independent x/y plotting scale; ``diagonal`` preserves
    the historical image-layout descriptor. Neither changes graph contacts.
    """
    dx, dy = map(float, delta)
    width, height = map(float, size)
    if not all(math.isfinite(v) for v in [dx, dy, width, height]) or min(width, height) <= 0:
        raise ValueError('Finite displacement and positive plot dimensions are required')
    if metric == 'diagonal':
        return math.hypot(dx, dy)/math.hypot(width, height)
    if metric == 'axes':
        return round(math.hypot(dx/width, dy/height)/math.sqrt(2), 14)
    raise ValueError('coordinate_metric must be diagonal or axes')


def extract_array(bgr, *, name="diagram", repair=True, repair_strategy=None, canonical_width=None, coordinate_metric='diagonal'):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")
    if coordinate_metric not in {'diagonal', 'axes'}:
        raise ValueError('coordinate_metric must be diagonal or axes')
    strategy = repair_strategy if repair_strategy is not None else ('local' if repair else 'none')
    if strategy not in {'none','local','global'} or (not repair and strategy!='none'):
        raise ValueError('Invalid or conflicting repair strategy')
    input_height, input_width = bgr.shape[:2]
    if canonical_width is not None:
        if not isinstance(canonical_width,int) or canonical_width < 128:
            raise ValueError('canonical_width must be an integer of at least 128 pixels')
        if input_width != canonical_width:
            height=max(1,round(input_height*canonical_width/input_width))
            bgr=cv2.resize(bgr,(canonical_width,height),interpolation=cv2.INTER_AREA if canonical_width<input_width else cv2.INTER_CUBIC)
    cfg = cv.PreprocessConfig(use_text_hint=False, paddle_device="cpu")
    raw = cv.colored_phase_mask(bgr, cfg)
    if not np.any(raw):
        raise ValueError("No blue/red boundary evidence within the configured colour ranges")
    coverage = float(np.count_nonzero(raw)/raw.size)
    if coverage > MAX_BOUNDARY_COVERAGE:
        raise ValueError(f'Unsupported image format: blue/red evidence covers {coverage:.1%} of the canvas; expected sparse boundaries on a light background')
    roi, info = cv.detect_plot_frame(bgr, raw, cfg)
    projection = cv._frame_from_dark_projection(bgr, replace(cfg, frame_band_ratio=.15))
    if projection is not None:
        roi = projection; info = {"method": "verified_outer_dark_projection", "found": True}
    x1, y1, x2, y2 = map(int, roi)
    crop = bgr[y1:y2, x1:x2]; line = raw[y1:y2, x1:x2]
    black = cv.detect_black_mask(crop, line, cfg)
    clear = cv.remove_black_from_line_mask(line, black)
    contact = cv.build_black_line_contact_mask(black, line, cfg)
    zone = cv.build_black_repair_roi(black, contact, cfg)
    if strategy=='local':
        fixed=cv.repair_lines_in_mask_regions(clear,zone,cfg)
    elif strategy=='global':
        fixed=clear.copy()
        for kernel in cv._build_directional_kernels(cfg.repair_kernel_sizes_black):
            fixed=np.maximum(fixed,cv2.morphologyEx(clear,cv2.MORPH_CLOSE,kernel))
    else:
        fixed=clear
    wall = cv.make_wall_mask(cv.build_line_skeleton_mask(fixed, cfg), cfg)
    regions, _, labels = cv.extract_total_regions(wall, cfg)
    contacts = cv.compute_region_adjacency(wall, labels, regions, cfg.adjacency_min_shared_boundary_px, cfg.adjacency_probe_radius)
    if not regions:
        raise ValueError("No regions survived the configured area threshold")
    graph = nx.Graph(name=name, roi=[x1,y1,x2,y2], coordinate_frame="crop_pixel_xy",
                     normalization="euclidean_internal_point_distance/roi_diagonal" if coordinate_metric=='diagonal' else "euclidean_axis_normalised_internal_point_distance/sqrt(2)", repair=strategy!='none',
                     coordinate_metric=coordinate_metric,
                     boundary_coverage=coverage,
                     repair_strategy=strategy, canonical_width=canonical_width,
                     input_size=[input_width,input_height],processing_size=[bgr.shape[1],bgr.shape[0]], OCR_for_geometry=False)
    centres = {}
    for r in regions:
        n = str(r["label"]); p = [r["centroid_x"], r["centroid_y"]]; centres[n] = np.asarray(p)
        graph.add_node(n, position=p, area_px=int(r["area"]), center_method=r["center_method"])
    for e in contacts:
        u, v = str(e["u_label"]), str(e["v_label"])
        # Preserve historical floating-point arithmetic for the default path.
        cost = float(np.linalg.norm(centres[u]-centres[v])/math.hypot(x2-x1,y2-y1)) if coordinate_metric=='diagonal' else coordinate_cost(centres[u]-centres[v], [x2-x1,y2-y1], 'axes')
        graph.add_edge(u, v, edge_distance=cost, shared_boundary_px=int(e["shared_boundary_px"]))
    diagnostics = {"roi": np.asarray(roi), "raw": line, "fixed": fixed, "black": black,
                   "wall": wall, "labels": labels, "regions": regions, "edges": contacts, "frame": info,
                   "config": {**asdict(cfg), "frame_projection_search_band": .15,
                              "repair_strategy":strategy,"canonical_width":canonical_width,"coordinate_metric":coordinate_metric,
                              "max_boundary_coverage":MAX_BOUNDARY_COVERAGE},
                   "input_scale_xy": [bgr.shape[1]/input_width,bgr.shape[0]/input_height]}
    return graph, diagnostics

def extract_image(path, *, repair=True, repair_strategy=None, canonical_width=None, coordinate_metric='diagonal'):
    path = Path(path)
    graph, debug = extract_array(cv.read_image_bgr(path), name=path.stem.removesuffix("_1"), repair=repair,
                                repair_strategy=repair_strategy,canonical_width=canonical_width,coordinate_metric=coordinate_metric)
    graph.graph.update(source_name=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return graph, debug
