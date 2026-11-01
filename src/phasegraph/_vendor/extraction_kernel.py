#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    Image = None
    _PIL_AVAILABLE = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PreprocessConfig:
    blue_h_min: int = 90
    blue_h_max: int = 145
    blue_s_min: int = 35
    blue_v_min: int = 35

    red1_h_min: int = 0
    red1_h_max: int = 12
    red2_h_min: int = 168
    red2_h_max: int = 180
    red_s_min: int = 35
    red_v_min: int = 35

    dark_threshold: int = 155
    frame_band_ratio: float = 0.08
    frame_occ_thr: float = 0.22
    frame_min_span_ratio: float = 0.60
    frame_inset_max_ratio: float = 0.12
    frame_line_tol: int = 8
    fallback_color_bbox_pad: int = 6

    use_text_hint: bool = True
    paddle_device: str = "cpu"
    paddle_model_name: str = "PP-OCRv5_server_det"
    paddle_limit_side_len: int = 1400
    paddle_limit_type: str = "max"
    ocr_text_detection_model_name: str = "PP-OCRv5_server_det"
    ocr_text_recognition_model_name: str = "PP-OCRv5_server_rec"
    ocr_textline_orientation: bool = True
    ocr_group_scale: float = 3.0
    ocr_group_scale_rotated: float = 3.5
    ocr_use_formula_fallback: bool = True
    formula_model_name: str = "PP-FormulaNet_plus-M"
    formula_min_box_w: int = 20
    formula_min_box_h: int = 12

    black_gray_max: int = 130
    black_rgb_max: int = 140
    black_chroma_max: int = 32
    black_mask_dilate_radius: int = 0
    line_contact_dilate_radius: int = 1
    black_contact_expand_radius: int = 1
    black_ignore_border_px: int = 2
    repair_context_margin: int = 8
    repair_kernel_sizes_black: Tuple[int, ...] = (3, 5, 7)
    skeleton_post_dilate_radius: int = 1
    wall_dilate_radius: int = 0
    use_skeleton_for_wall: bool = True
    border_thickness: int = 5
    min_region_area: int = 20

    adjacency_min_shared_boundary_px: int = 3
    adjacency_probe_radius: int = 2
    axis_band_ratio_x: float = 0.16
    axis_band_ratio_y: float = 0.18
    axis_fit_min_points: int = 2
    axis_value_round: int = 4

    label_assign_near_px: int = 6
    label_arrow_search_px: int = 80
    label_arrow_text_exclusion_dilate: int = 2
    label_arrow_border_ignore_px: int = 14
    label_arrow_tail_text_max_px: int = 72
    label_max_chars: int = 24
    label_min_score: float = 0.25

    output_dpi: int = 600


def parse_args():
    p = argparse.ArgumentParser(description="Extract enclosed regions, adjacency networks and point features")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--main_prefix", default="Al-")
    p.add_argument("--paddle_device", default="cpu")
    p.add_argument("--save_debug", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_device_string(device: str) -> str:
    d = str(device).strip().lower()
    if d in {"", "auto"}:
        return "gpu:0"
    if d == "gpu":
        return "gpu:0"
    if d.startswith("gpu"):
        return d
    return d


def list_images(input_dir: Path, prefix: str) -> List[Path]:
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.name.startswith(prefix)]
    files.sort()
    return files


def phase_display_name(image_name: str) -> str:
    stem = Path(image_name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_image_unicode(path: Path, image: np.ndarray, dpi: Optional[int] = None):
    if dpi is not None and _PIL_AVAILABLE:
        if image.ndim == 2:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img.save(str(path), dpi=(dpi, dpi))
        return
    ok, buf = cv2.imencode(path.suffix if path.suffix else ".png", image)
    if not ok:
        raise ValueError(f"Cannot save image: {path}")
    path.write_bytes(buf.tobytes())


def hsv_blue_mask(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.blue_h_min, cfg.blue_s_min, cfg.blue_v_min], dtype=np.uint8)
    upper = np.array([cfg.blue_h_max, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def hsv_red_mask(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([cfg.red1_h_min, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8)
    upper1 = np.array([cfg.red1_h_max, 255, 255], dtype=np.uint8)
    lower2 = np.array([cfg.red2_h_min, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8)
    upper2 = np.array([cfg.red2_h_max, 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def colored_phase_mask(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    return cv2.bitwise_or(hsv_blue_mask(bgr, cfg), hsv_red_mask(bgr, cfg))


def _frame_from_dark_projection(bgr: np.ndarray, cfg: PreprocessConfig):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray <= cfg.dark_threshold).astype(np.uint8)
    bx = max(8, int(w * cfg.frame_band_ratio))
    by = max(8, int(h * cfg.frame_band_ratio))
    left_cols = dark[:, :bx].mean(axis=0)
    right_cols = dark[:, w - bx:].mean(axis=0)
    top_rows = dark[:by, :].mean(axis=1)
    bottom_rows = dark[h - by:, :].mean(axis=1)
    left_idx = np.where(left_cols >= cfg.frame_occ_thr)[0]
    right_idx = np.where(right_cols >= cfg.frame_occ_thr)[0]
    top_idx = np.where(top_rows >= cfg.frame_occ_thr)[0]
    bottom_idx = np.where(bottom_rows >= cfg.frame_occ_thr)[0]
    if len(left_idx) == 0 or len(right_idx) == 0 or len(top_idx) == 0 or len(bottom_idx) == 0:
        return None
    x1 = int(left_idx.min())
    x2 = int(w - bx + right_idx.max())
    y1 = int(top_idx.min())
    y2 = int(h - by + bottom_idx.max())
    if x1 > int(w * cfg.frame_inset_max_ratio):
        return None
    if y1 > int(h * cfg.frame_inset_max_ratio):
        return None
    if (w - 1 - x2) > int(w * cfg.frame_inset_max_ratio):
        return None
    if (h - 1 - y2) > int(h * cfg.frame_inset_max_ratio):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < int(w * cfg.frame_min_span_ratio):
        return None
    if (y2 - y1) < int(h * cfg.frame_min_span_ratio):
        return None
    return x1, y1, x2 + 1, y2 + 1


def _frame_from_hough_spines(bgr: np.ndarray, line_mask: np.ndarray, cfg: PreprocessConfig):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray <= cfg.dark_threshold, 255, 0).astype(np.uint8)
    min_h = int(w * cfg.frame_min_span_ratio)
    min_v = int(h * cfg.frame_min_span_ratio)
    lines = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=50,
                            minLineLength=min(min_h, min_v), maxLineGap=3)
    if lines is None:
        return None
    verticals, horizontals = [], []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if abs(dx) <= 2 and abs(dy) >= max(min_v, 0.45 * h):
            verticals.append({"x": int(round((x1 + x2) / 2)), "y1": min(y1, y2), "y2": max(y1, y2), "length": length})
        elif abs(dy) <= 2 and abs(dx) >= max(min_h, 0.45 * w):
            horizontals.append({"y": int(round((y1 + y2) / 2)), "x1": min(x1, x2), "x2": max(x1, x2), "length": length})
    if not verticals:
        return None
    vgroups = []
    for v in sorted(verticals, key=lambda t: t["x"]):
        if not vgroups or abs(v["x"] - np.mean([z["x"] for z in vgroups[-1]])) > cfg.frame_line_tol:
            vgroups.append([v])
        else:
            vgroups[-1].append(v)
    vmerged = []
    for grp in vgroups:
        total_len = sum(g["length"] for g in grp)
        x = int(round(sum(g["x"] * g["length"] for g in grp) / max(total_len, 1e-6)))
        y1 = min(g["y1"] for g in grp)
        y2 = max(g["y2"] for g in grp)
        vmerged.append({"x": x, "y1": y1, "y2": y2, "height": y2 - y1 + 1, "score": total_len})

    ys, xs = np.where(line_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin, xmax = int(xs.min()), int(xs.max())
    cand_left = [v for v in vmerged if v["x"] <= xmin + max(10, int(0.08 * w))]
    cand_right = [v for v in vmerged if v["x"] >= xmax - max(10, int(0.08 * w))]
    if not cand_left or not cand_right:
        cand_left = [min(vmerged, key=lambda v: v["x"])]
        cand_right = [max(vmerged, key=lambda v: v["x"])]
    left = max(cand_left, key=lambda v: (v["height"], -abs(v["x"] - xmin), -v["x"]))
    right = max(cand_right, key=lambda v: (v["height"], -abs(v["x"] - xmax), v["x"]))
    if right["x"] <= left["x"]:
        return None

    y_bottom = None
    if horizontals:
        good_h = []
        for hh in horizontals:
            overlap = min(hh["x2"], right["x"]) - max(hh["x1"], left["x"]) + 1
            if overlap >= 0.75 * (right["x"] - left["x"] + 1):
                good_h.append(hh)
        if good_h:
            y_bottom = max(good_h, key=lambda t: (t["y"], t["length"]))["y"]
    if y_bottom is None:
        y_bottom = max(left["y2"], right["y2"])

    y_top = None
    if horizontals:
        top_h = []
        for hh in horizontals:
            overlap = min(hh["x2"], right["x"]) - max(hh["x1"], left["x"]) + 1
            if overlap >= 0.75 * (right["x"] - left["x"] + 1):
                top_h.append(hh)
        if top_h:
            y_top = min(top_h, key=lambda t: t["y"])["y"]
    if y_top is None:
        y_top = min(left["y1"], right["y1"])

    x1 = max(0, int(left["x"]))
    x2 = min(w, int(right["x"] + 1))
    y1 = max(0, int(y_top))
    y2 = min(h, int(y_bottom + 1))
    if x2 - x1 < min_h or y2 - y1 < min_v:
        return None
    return x1, y1, x2, y2


def detect_plot_frame(bgr: np.ndarray, line_mask: np.ndarray, cfg: PreprocessConfig):
    h, w = bgr.shape[:2]
    hough_spine = _frame_from_hough_spines(bgr, line_mask, cfg)
    if hough_spine is not None:
        return hough_spine, {"method": "hough_spines", "found": True}
    proj = _frame_from_dark_projection(bgr, cfg)
    if proj is not None:
        return proj, {"method": "dark_projection", "found": True}
    ys, xs = np.where(line_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, w, h), {"method": "full_image", "found": False}
    x1 = max(0, int(xs.min()) - cfg.fallback_color_bbox_pad)
    y1 = max(0, int(ys.min()) - cfg.fallback_color_bbox_pad)
    x2 = min(w, int(xs.max()) + 1 + cfg.fallback_color_bbox_pad)
    y2 = min(h, int(ys.max()) + 1 + cfg.fallback_color_bbox_pad)
    return (x1, y1, x2, y2), {"method": "color_bbox_fallback", "found": False}


def build_paddle_text_detector(cfg: PreprocessConfig):
    if not cfg.use_text_hint:
        return None
    from paddleocr import TextDetection
    return TextDetection(model_name=cfg.paddle_model_name, device=normalize_device_string(cfg.paddle_device),
                         limit_side_len=cfg.paddle_limit_side_len, limit_type=cfg.paddle_limit_type)


def _extract_dt_polys_from_result_obj(res):
    if isinstance(res, dict):
        if "res" in res and isinstance(res["res"], dict) and "dt_polys" in res["res"]:
            return res["res"]["dt_polys"]
        if "dt_polys" in res:
            return res["dt_polys"]
    if hasattr(res, "json"):
        payload = res.json
        if isinstance(payload, dict):
            if "res" in payload and isinstance(payload["res"], dict) and "dt_polys" in payload["res"]:
                return payload["res"]["dt_polys"]
            if "dt_polys" in payload:
                return payload["dt_polys"]
    if hasattr(res, "res"):
        payload = res.res
        if isinstance(payload, dict) and "dt_polys" in payload:
            return payload["dt_polys"]
    return []


def detect_text_mask_paddleocr(crop_bgr: np.ndarray, detector):
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    if detector is None:
        return mask, boxes
    output = detector.predict(crop_bgr, batch_size=1)
    for res in output:
        polys = _extract_dt_polys_from_result_obj(res)
        if polys is None:
            continue
        arr = np.asarray(polys)
        if arr.size == 0:
            continue
        for poly in arr:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            boxes.append(pts.reshape(-1, 2).tolist())
            cv2.fillPoly(mask, [pts], 255)
    return mask, boxes


def build_ocr_reader(cfg: PreprocessConfig):
    from paddleocr import PaddleOCR

    device = normalize_device_string(cfg.paddle_device)

    bundle: Dict[str, Any] = {"ocr": None, "formula": None, "device": device}

    try:
        bundle["ocr"] = PaddleOCR(
            lang="en",
            device=device,
            text_detection_model_name=cfg.ocr_text_detection_model_name,
            text_recognition_model_name=cfg.ocr_text_recognition_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=bool(cfg.ocr_textline_orientation),
            text_det_limit_side_len=max(int(cfg.paddle_limit_side_len), 1600),
            text_det_limit_type=cfg.paddle_limit_type,
            text_rec_score_thresh=0.0,
        )
    except TypeError:
        try:
            bundle["ocr"] = PaddleOCR(
                lang="en",
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=bool(cfg.ocr_textline_orientation),
            )
        except TypeError:
            bundle["ocr"] = PaddleOCR(
                lang="en",
                use_angle_cls=True,
                use_gpu=(device != "cpu"),
            )

    if bool(getattr(cfg, "ocr_use_formula_fallback", True)):
        try:
            from paddlex import create_model
            try:
                bundle["formula"] = create_model(model_name=cfg.formula_model_name, device=device)
            except TypeError:
                bundle["formula"] = create_model(model_name=cfg.formula_model_name)
        except Exception:
            bundle["formula"] = None

    return bundle


def _rotate_image_and_inverse(angle: int, img: np.ndarray):
    h, w = img.shape[:2]
    if angle == 0:
        def inv(pt):
            x, y = pt
            return float(x), float(y)
        return img, inv
    if angle == 90:
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        def inv(pt):
            x, y = pt
            return float(y), float(h - 1 - x)
        return rot, inv
    if angle == 270:
        rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        def inv(pt):
            x, y = pt
            return float(w - 1 - y), float(x)
        return rot, inv
    raise ValueError(angle)


def _bbox_from_poly(poly: np.ndarray) -> Tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _extract_text_items_from_ocr_result_obj(res_obj) -> List[Dict[str, Any]]:
    def _first_payload(obj):
        cand = []
        if isinstance(obj, dict):
            cand.append(obj)
        for attr in ("json", "res"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, dict):
                    cand.append(val)
        for payload in cand:
            stack = [payload]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "dt_polys" in cur and ("rec_texts" in cur or "texts" in cur):
                        return cur
                    for v in cur.values():
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(cur, list):
                    for v in cur:
                        if isinstance(v, (dict, list)):
                            stack.append(v)
        return None

    payload = _first_payload(res_obj)
    if payload is None:
        return []
    polys = payload.get("dt_polys", [])
    texts = payload.get("rec_texts", payload.get("texts", []))
    scores = payload.get("rec_scores", payload.get("scores", [1.0] * len(texts)))
    items = []
    n = min(len(polys), len(texts))
    for i in range(n):
        try:
            poly = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        txt = str(texts[i]).strip()
        if not txt:
            continue
        try:
            score = float(scores[i]) if i < len(scores) else 1.0
        except Exception:
            score = 1.0
        items.append({"poly": poly, "text": txt, "score": score})
    return items


def _run_ocr_once(reader, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    if reader is None:
        return []
    engine = reader.get("ocr") if isinstance(reader, dict) else reader
    if engine is None:
        return []
    if hasattr(engine, "predict"):
        try:
            results = engine.predict(img_bgr)
            out = []
            for res in results:
                out.extend(_extract_text_items_from_ocr_result_obj(res))
            if out:
                return out
        except Exception:
            pass
    if hasattr(engine, "ocr"):
        try:
            res = engine.ocr(img_bgr, cls=True)
            out = []
            if res:
                lines = res[0] if isinstance(res, list) else res
                for line in lines:
                    if not line or len(line) < 2:
                        continue
                    try:
                        poly = np.array(line[0], dtype=np.float32).reshape(-1, 2)
                    except Exception:
                        continue
                    txt = str(line[1][0]).strip()
                    if not txt:
                        continue
                    try:
                        score = float(line[1][1]) if len(line[1]) > 1 else 1.0
                    except Exception:
                        score = 1.0
                    out.append({"poly": poly, "text": txt, "score": score})
            return out
        except Exception:
            return []
    return []


def _ocr_patch_items(img_bgr: np.ndarray, reader, origin_xy: Tuple[int, int], scale: float = 1.0) -> List[Dict[str, Any]]:
    ox, oy = origin_xy
    if scale != 1.0:
        patch = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        patch = img_bgr
    raw_items = _run_ocr_once(reader, patch)
    items = []
    for it in raw_items:
        poly = np.asarray(it["poly"], dtype=np.float32) / float(scale)
        poly[:, 0] += float(ox)
        poly[:, 1] += float(oy)
        bbox = _bbox_from_poly(poly)
        items.append({
            "text": it["text"],
            "score": float(it.get("score", 1.0)),
            "poly": poly.tolist(),
            "bbox": bbox,
            "angle": 0,
            "source": "patch",
        })
    return items


def _run_axis_patch_ocr(img_bgr: np.ndarray, roi: np.ndarray, reader, cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, roi.tolist())
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    bx = max(40, int(round(w * 0.18)))
    by = max(40, int(round(h * 0.18)))
    overlap = 18
    patches = []
    # bottom numeric ticks
    patches.append((max(0, x1 - 10), max(0, y2 - overlap), min(W, x2 + 10), min(H, y2 + by), 2.0))
    # left y-axis numeric ticks
    patches.append((max(0, x1 - bx), max(0, y1 - 10), min(W, x1 + overlap), min(H, y2 + 10), 2.0))
    # right y-axis numeric ticks fallback
    patches.append((max(0, x2 - overlap), max(0, y1 - 10), min(W, x2 + bx), min(H, y2 + 10), 2.0))
    all_items = []
    for px1, py1, px2, py2, scale in patches:
        if px2 - px1 < 8 or py2 - py1 < 8:
            continue
        patch = img_bgr[py1:py2, px1:px2]
        all_items.extend(_ocr_patch_items(patch, reader, (px1, py1), scale=scale))
    return all_items


def run_multiorientation_ocr(img_bgr: np.ndarray, reader, roi: Optional[np.ndarray] = None,
                             cfg: Optional[PreprocessConfig] = None) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    if reader is None:
        return all_items
    for angle in (0, 90, 270):
        rot, inv = _rotate_image_and_inverse(angle, img_bgr)
        raw_items = _run_ocr_once(reader, rot)
        for it in raw_items:
            poly = np.asarray(it["poly"], dtype=np.float32).reshape(-1, 2)
            text = str(it["text"]).strip()
            score = float(it.get("score", 1.0))
            if not text:
                continue
            mapped = np.array([inv(tuple(pt)) for pt in poly], dtype=np.float32)
            bbox = _bbox_from_poly(mapped)
            all_items.append({
                "text": text,
                "score": score,
                "poly": mapped.tolist(),
                "bbox": bbox,
                "angle": angle,
                "source": "full",
            })
    if roi is not None and cfg is not None:
        all_items.extend(_run_axis_patch_ocr(img_bgr, roi, reader, cfg))
    dedup: List[Dict[str, Any]] = []
    for item in sorted(all_items, key=lambda t: t["score"], reverse=True):
        keep = True
        for ex in dedup:
            if item["text"].lower() == ex["text"].lower() and _bbox_iou(item["bbox"], ex["bbox"]) >= 0.35:
                keep = False
                break
        if keep:
            dedup.append(item)
    return dedup




def build_label_ocr_image(crop_bgr: np.ndarray, black_mask: np.ndarray) -> np.ndarray:
    h, w = black_mask.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[black_mask > 0] = (0, 0, 0)
    return out


def _build_textlike_mask_from_black(black_mask: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    if black_mask.size == 0:
        return black_mask
    if text_mask is None or text_mask.size == 0 or not np.any(text_mask > 0):
        return black_mask.copy()
    return cv2.bitwise_and(black_mask, text_mask)


def _bbox_mask_overlap_labels(box, label_img: np.ndarray) -> Dict[int, int]:
    h, w = label_img.shape[:2]
    x1, y1, x2, y2 = map(int, map(round, box))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return {}
    patch = label_img[y1:y2, x1:x2]
    vals, counts = np.unique(patch[patch > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def dedup_label_ocr_items(ocr_items: List[Dict[str, Any]], black_mask: np.ndarray, text_mask: np.ndarray) -> List[Dict[str, Any]]:
    if not ocr_items:
        return []
    text_like = _build_textlike_mask_from_black(black_mask, text_mask)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    grouped = cv2.dilate(text_like, ker, iterations=1)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    by_group: Dict[int, List[Dict[str, Any]]] = {}
    leftovers: List[Dict[str, Any]] = []
    for item in ocr_items:
        overlaps = _bbox_mask_overlap_labels(item['bbox'], lab)
        if overlaps:
            gid = max(overlaps.items(), key=lambda kv: kv[1])[0]
            if gid > 0:
                by_group.setdefault(int(gid), []).append(item)
                continue
        leftovers.append(item)

    kept: List[Dict[str, Any]] = []
    for gid, items in by_group.items():
        # keep the widest/highest-area text for this black-text group
        best = max(items, key=lambda it: (_bbox_area(it['bbox']), float(it.get('score', 1.0)), len(str(it['text']))))
        best = dict(best)
        best['group_id'] = int(gid)
        kept.append(best)

    # dedup leftovers by iou/text, keep only the larger box
    for item in sorted(leftovers, key=lambda it: (_bbox_area(it['bbox']), float(it.get('score', 1.0))), reverse=True):
        dup = False
        for ex in kept:
            if _bbox_iou(item['bbox'], ex['bbox']) >= 0.45:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


def run_label_ocr(crop_bgr: np.ndarray, black_mask: np.ndarray, text_mask: np.ndarray, reader, roi: np.ndarray, cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    label_img = build_label_ocr_image(crop_bgr, black_mask)
    items = run_multiorientation_ocr(label_img, reader, None, None)
    # Map crop-local boxes to global coordinates so downstream code stays unchanged
    gx, gy = float(roi[0]), float(roi[1])
    mapped = []
    for it in items:
        poly = np.asarray(it['poly'], dtype=np.float32)
        poly[:, 0] += gx
        poly[:, 1] += gy
        bbox = _bbox_from_poly(poly)
        mapped.append({
            'text': it['text'],
            'score': float(it.get('score', 1.0)),
            'poly': poly.tolist(),
            'bbox': bbox,
            'angle': it.get('angle', 0),
            'source': 'label_black',
        })
    dedup_global = dedup_label_ocr_items(mapped, black_mask, text_mask)
    return dedup_global


def _point_to_bbox_distance(x: float, y: float, box) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def detect_black_mask(crop_bgr: np.ndarray, line_mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    b = crop_bgr[:, :, 0].astype(np.int16)
    g = crop_bgr[:, :, 1].astype(np.int16)
    r = crop_bgr[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    maxc = np.maximum(np.maximum(b, g), r)
    minc = np.minimum(np.minimum(b, g), r)
    chroma = maxc - minc
    black_candidate = ((gray <= cfg.black_gray_max) & (chroma <= cfg.black_chroma_max)) | (maxc <= cfg.black_rgb_max)
    black_candidate = black_candidate.astype(np.uint8) * 255
    black_candidate = cv2.bitwise_and(black_candidate, cv2.bitwise_not(line_mask))
    m = max(0, int(cfg.black_ignore_border_px))
    if m > 0:
        black_candidate[:m, :] = 0
        black_candidate[-m:, :] = 0
        black_candidate[:, :m] = 0
        black_candidate[:, -m:] = 0
    return black_candidate


def remove_black_from_line_mask(line_mask: np.ndarray, black_mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(line_mask, cv2.bitwise_not(black_mask))


def build_black_line_contact_mask(black_mask: np.ndarray, line_mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not np.any(black_mask > 0) or not np.any(line_mask > 0):
        return np.zeros_like(black_mask)
    rb = max(0, int(cfg.black_contact_expand_radius))
    if rb > 0:
        ker_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb + 1, 2 * rb + 1))
        black_for_contact = cv2.dilate(black_mask, ker_b, iterations=1)
    else:
        black_for_contact = black_mask.copy()
    rl = max(0, int(cfg.line_contact_dilate_radius))
    if rl > 0:
        ker_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rl + 1, 2 * rl + 1))
        line_for_contact = cv2.dilate(line_mask, ker_l, iterations=1)
    else:
        line_for_contact = line_mask.copy()
    return cv2.bitwise_and(black_for_contact, line_for_contact)


def build_black_repair_roi(black_mask: np.ndarray, black_line_contact_mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not np.any(black_mask > 0) or not np.any(black_line_contact_mask > 0):
        return np.zeros_like(black_mask)
    nlab, lab, _, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    roi = np.zeros_like(black_mask)
    for i in range(1, nlab):
        comp = np.where(lab == i, 255, 0).astype(np.uint8)
        if np.any(cv2.bitwise_and(comp, black_line_contact_mask) > 0):
            roi = np.maximum(roi, comp)
    return roi


def _build_directional_kernels(kernel_sizes: Tuple[int, ...]) -> List[np.ndarray]:
    kernels: List[np.ndarray] = []
    for k in kernel_sizes:
        k = int(k)
        if k < 3:
            continue
        kernels.extend([
            cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)),
            np.eye(k, dtype=np.uint8),
            np.fliplr(np.eye(k, dtype=np.uint8)),
        ])
    return kernels


def repair_lines_in_mask_regions(line_mask: np.ndarray, repair_roi: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    repaired = line_mask.copy()
    if not np.any(repair_roi > 0):
        return repaired
    h, w = repaired.shape[:2]
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(repair_roi, connectivity=8)
    kernels = _build_directional_kernels(cfg.repair_kernel_sizes_black)
    for i in range(1, nlab):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        margin = int(max(cfg.repair_context_margin, max(cfg.repair_kernel_sizes_black) // 2 + 1))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + ww + margin)
        y2 = min(h, y + hh + margin)
        patch_lines = repaired[y1:y2, x1:x2].copy()
        patch_roi = np.where(lab[y1:y2, x1:x2] == i, 255, 0).astype(np.uint8)
        merged = patch_lines.copy()
        for ker in kernels:
            closed = cv2.morphologyEx(patch_lines, cv2.MORPH_CLOSE, ker)
            merged = np.where(patch_roi > 0, np.maximum(merged, closed), merged).astype(np.uint8)
        repaired[y1:y2, x1:x2] = np.where(patch_roi > 0, merged, repaired[y1:y2, x1:x2]).astype(np.uint8)
    return repaired


def _zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    img = (binary_mask > 0).astype(np.uint8)
    changed = True
    h, w = img.shape[:2]
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] != 1:
                        continue
                    p2 = img[y - 1, x]
                    p3 = img[y - 1, x + 1]
                    p4 = img[y, x + 1]
                    p5 = img[y + 1, x + 1]
                    p6 = img[y + 1, x]
                    p7 = img[y + 1, x - 1]
                    p8 = img[y, x - 1]
                    p9 = img[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    B = sum(neighbors)
                    if B < 2 or B > 6:
                        continue
                    A = 0
                    seq = neighbors + [neighbors[0]]
                    for i in range(8):
                        if seq[i] == 0 and seq[i + 1] == 1:
                            A += 1
                    if A != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0
    return (img * 255).astype(np.uint8)


def build_line_skeleton_mask(line_mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    thin = _zhang_suen_thinning(line_mask)
    r = max(0, int(cfg.skeleton_post_dilate_radius))
    if r <= 0:
        return thin
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(thin, ker, iterations=1)


def make_wall_mask(line_mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    h, w = line_mask.shape[:2]
    wall = line_mask.copy()
    if cfg.wall_dilate_radius > 0:
        r = cfg.wall_dilate_radius
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        wall = cv2.dilate(wall, ker, iterations=1)
    cv2.line(wall, (0, 0), (w - 1, 0), 255, thickness=cfg.border_thickness)
    cv2.line(wall, (0, h - 1), (w - 1, h - 1), 255, thickness=cfg.border_thickness)
    cv2.line(wall, (0, 0), (0, h - 1), 255, thickness=cfg.border_thickness)
    cv2.line(wall, (w - 1, 0), (w - 1, h - 1), 255, thickness=cfg.border_thickness)
    return wall


def compute_internal_center(component_mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return {"center_x": 0.0, "center_y": 0.0, "raw_centroid_x": 0.0, "raw_centroid_y": 0.0,
                "center_method": "empty", "center_shift": 0.0}
    raw_cx = float(xs.mean())
    raw_cy = float(ys.mean())
    cx_i = int(round(raw_cx))
    cy_i = int(round(raw_cy))
    h, w = component_mask.shape[:2]
    if 0 <= cx_i < w and 0 <= cy_i < h and component_mask[cy_i, cx_i] > 0:
        return {"center_x": float(cx_i), "center_y": float(cy_i), "raw_centroid_x": raw_cx,
                "raw_centroid_y": raw_cy, "center_method": "raw_inside", "center_shift": 0.0}
    dist = cv2.distanceTransform((component_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.where(component_mask > 0)
    pts = np.stack([xx, yy], axis=1).astype(np.float32)
    target = np.array([raw_cx, raw_cy], dtype=np.float32)
    d2 = np.sum((pts - target[None, :]) ** 2, axis=1)
    nearest_idx = int(np.argmin(d2))
    px = int(pts[nearest_idx, 0])
    py = int(pts[nearest_idx, 1])
    if np.max(dist) > 0:
        best_idx = nearest_idx
        best_score = -1.0
        local_radius = 6
        for k, (tx, ty) in enumerate(pts.astype(np.int32)):
            if abs(int(tx) - px) > local_radius or abs(int(ty) - py) > local_radius:
                continue
            score = float(dist[int(ty), int(tx)])
            if score > best_score:
                best_score = score
                best_idx = k
        px = int(pts[best_idx, 0])
        py = int(pts[best_idx, 1])
    shift = float(math.hypot(px - raw_cx, py - raw_cy))
    return {"center_x": float(px), "center_y": float(py), "raw_centroid_x": raw_cx,
            "raw_centroid_y": raw_cy, "center_method": "projected_inside", "center_shift": shift}


def extract_total_regions(wall_mask: np.ndarray, cfg: PreprocessConfig):
    free_total = cv2.bitwise_not(wall_mask)
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(free_total, connectivity=8)
    regions = []
    region_mask = np.zeros_like(wall_mask)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_region_area:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        center_info = compute_internal_center(comp)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        regions.append({
            "label": int(i),
            "area": area,
            "centroid_x": float(center_info["center_x"]),
            "centroid_y": float(center_info["center_y"]),
            "raw_centroid_x": float(center_info["raw_centroid_x"]),
            "raw_centroid_y": float(center_info["raw_centroid_y"]),
            "center_method": str(center_info["center_method"]),
            "center_shift": float(center_info["center_shift"]),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": ww,
            "bbox_h": hh,
        })
        region_mask[labels == i] = 255
    return regions, region_mask, labels


def draw_centers_overlay(crop_bgr: np.ndarray, regions) -> np.ndarray:
    out = crop_bgr.copy()
    for reg in regions:
        cv2.circle(out, (int(round(reg["centroid_x"])), int(round(reg["centroid_y"]))), 4, (0, 255, 0), -1)
    return out


def preprocess_one(path: Path, cfg: PreprocessConfig, text_detector, save_debug_dir: Optional[Path] = None):
    bgr = read_image_bgr(path)
    raw_line_mask = colored_phase_mask(bgr, cfg)
    roi, frame_debug = detect_plot_frame(bgr, raw_line_mask, cfg)
    x1, y1, x2, y2 = roi
    crop_bgr = bgr[y1:y2, x1:x2]
    crop_line_raw = raw_line_mask[y1:y2, x1:x2]
    text_mask, text_boxes = detect_text_mask_paddleocr(crop_bgr, text_detector)
    black_mask = detect_black_mask(crop_bgr, crop_line_raw, cfg)
    crop_line_no_black = remove_black_from_line_mask(crop_line_raw, black_mask)
    black_line_contact_mask = build_black_line_contact_mask(black_mask, crop_line_raw, cfg)
    black_repair_roi = build_black_repair_roi(black_mask, black_line_contact_mask, cfg)
    line_repaired = repair_lines_in_mask_regions(crop_line_no_black, black_repair_roi, cfg)
    skeleton = build_line_skeleton_mask(line_repaired, cfg)
    wall_base = skeleton if cfg.use_skeleton_for_wall else line_repaired
    wall_mask = make_wall_mask(wall_base, cfg)
    regions_total, region_mask_total, labels_total = extract_total_regions(wall_mask, cfg)
    centers_overlay = draw_centers_overlay(crop_bgr, regions_total)
    if save_debug_dir is not None:
        ensure_dir(save_debug_dir)
        save_image_unicode(save_debug_dir / f"{path.stem}_01_crop.png", crop_bgr, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_02_line_raw.png", crop_line_raw, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_03_text_mask.png", text_mask, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_04_black_mask.png", black_mask, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_05_line_no_black.png", crop_line_no_black, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_06_black_line_contact_mask.png", black_line_contact_mask, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_07_black_repair_roi.png", black_repair_roi, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_08_line_repaired.png", line_repaired, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_09_wall_mask.png", wall_mask, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_10_region_mask_total.png", region_mask_total, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_11_skeleton.png", skeleton, dpi=cfg.output_dpi)
        save_image_unicode(save_debug_dir / f"{path.stem}_12_centers_overlay.png", centers_overlay, dpi=cfg.output_dpi)
    return {
        "bgr": bgr,
        "crop_bgr": crop_bgr,
        "crop_line_raw": crop_line_raw,
        "text_mask": text_mask,
        "black_mask": black_mask,
        "crop_line_no_black": crop_line_no_black,
        "black_line_contact_mask": black_line_contact_mask,
        "black_repair_roi": black_repair_roi,
        "line_repaired": line_repaired,
        "skeleton": skeleton,
        "wall_mask": wall_mask,
        "regions_total": regions_total,
        "region_mask_total": region_mask_total,
        "labels_total": labels_total,
        "centers_overlay": centers_overlay,
        "roi": np.array(roi, dtype=np.int32),
        "frame_debug": frame_debug,
        "text_boxes": text_boxes,
    }



def clean_numeric_text(text: str) -> Optional[float]:
    s0 = str(text).strip()
    if not s0:
        return None
    if re.search(r"[A-HJ-NP-Za-hj-knp-z]", s0):
        return None
    s = s0.replace(" ", "")
    trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"})
    s = s.translate(trans)
    s = s.replace("°", "").replace("C", "").replace("c", "")
    s = s.rstrip("+-")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def fit_linear_axis(candidates: List[Dict[str, Any]], coord_key: str, min_points: int) -> Optional[Dict[str, Any]]:
    if len(candidates) < min_points:
        return None
    cand = [dict(c) for c in candidates]
    cand.sort(key=lambda c: (float(c.get("axis_dist", 0.0)), -float(c.get("score", 1.0))))
    dedup_value = {}
    for c in cand:
        vkey = round(float(c["value"]), 6)
        dedup_value.setdefault(vkey, c)
    cand = list(dedup_value.values())

    coords = np.array([float(c[coord_key]) for c in cand], dtype=np.float64)
    values = np.array([float(c["value"]) for c in cand], dtype=np.float64)
    if len(coords) < min_points or np.std(coords) < 1e-6:
        return None

    a, b = np.polyfit(coords, values, 1)
    pred = a * coords + b
    resid = np.abs(pred - values)
    med = float(np.median(resid)) if resid.size > 0 else 0.0
    thr = max(0.02 if np.max(np.abs(values)) <= 2 else 25.0, 2.5 * med)
    keep = resid <= thr
    if np.count_nonzero(keep) >= min_points and np.count_nonzero(keep) < len(coords):
        coords2 = coords[keep]
        values2 = values[keep]
        a, b = np.polyfit(coords2, values2, 1)
        pred2 = a * coords2 + b
        rmse = float(np.sqrt(np.mean((pred2 - values2) ** 2)))
        used = [cand[i] for i in range(len(cand)) if bool(keep[i])]
        return {"a": float(a), "b": float(b), "rmse": rmse, "n_points": int(len(coords2)), "used_points": used}

    rmse = float(np.sqrt(np.mean((pred - values) ** 2)))
    return {"a": float(a), "b": float(b), "rmse": rmse, "n_points": int(len(coords)), "used_points": cand}


def build_axis_mapping(ocr_items: List[Dict[str, Any]], roi: np.ndarray, cfg: PreprocessConfig) -> Dict[str, Any]:
    x1, y1, x2, y2 = map(int, roi.tolist())
    w = x2 - x1
    h = y2 - y1
    x_band = max(28, int(round(h * max(cfg.axis_band_ratio_x, 0.18))))
    y_band = max(28, int(round(w * max(cfg.axis_band_ratio_y, 0.20))))
    x_candidates, y_candidates = [], []
    for item in ocr_items:
        val = clean_numeric_text(item["text"])
        if val is None:
            continue
        bx1, by1, bx2, by2 = item["bbox"]
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        score = float(item.get("score", 1.0))
        if 0.0 <= val <= 1.0001:
            if (x1 - 0.06 * w) <= cx <= (x2 + 0.06 * w) and (y2 - 0.035 * h) <= cy <= (y2 + x_band):
                x_candidates.append({"text": item["text"], "value": val, "pixel": cx, "cx": cx, "cy": cy, "bbox": item["bbox"], "score": score, "axis_dist": abs(cy - y2)})
        if 0.0 <= val <= 5000.0:
            on_left = (x1 - y_band) <= cx <= (x1 + 0.045 * w)
            on_right = (x2 - 0.045 * w) <= cx <= (x2 + y_band)
            in_y_span = (y1 - 0.08 * h) <= cy <= (y2 + 0.08 * h)
            if in_y_span and on_left:
                y_candidates.append({"text": item["text"], "value": val, "pixel": cy, "cx": cx, "cy": cy, "bbox": item["bbox"], "side": "left", "bw": bw, "bh": bh, "score": score, "axis_dist": abs(cx - x1)})
            elif in_y_span and on_right:
                y_candidates.append({"text": item["text"], "value": val, "pixel": cy, "cx": cx, "cy": cy, "bbox": item["bbox"], "side": "right", "bw": bw, "bh": bh, "score": score, "axis_dist": abs(cx - x2)})

    y_left = [c for c in y_candidates if c.get("side") == "left"]
    y_right = [c for c in y_candidates if c.get("side") == "right"]
    y_use = y_left if len(y_left) >= len(y_right) else y_right

    x_fit = fit_linear_axis(x_candidates, "pixel", cfg.axis_fit_min_points)
    y_fit = fit_linear_axis(y_use, "pixel", cfg.axis_fit_min_points)
    return {
        "roi": roi.tolist(),
        "x_candidates": x_candidates,
        "y_candidates": y_candidates,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_mapping_ok": x_fit is not None,
        "y_mapping_ok": y_fit is not None,
    }


def pixel_to_axis(axis_fit: Optional[Dict[str, Any]], pixel: float) -> Optional[float]:

    if axis_fit is None:
        return None
    return float(axis_fit["a"] * pixel + axis_fit["b"])


def format_coord(x: Optional[float], y: Optional[float], decimals: int) -> str:
    if x is None or y is None:
        return "(nan,nan)"
    return f"({round(float(x), decimals)},{round(float(y), decimals)})"




def _axis_value_range_from_roi(axis_fit: Optional[Dict[str, Any]], p1: float, p2: float) -> Tuple[Optional[float], Optional[float]]:
    v1 = pixel_to_axis(axis_fit, p1)
    v2 = pixel_to_axis(axis_fit, p2)
    if v1 is None or v2 is None:
        return None, None
    return float(min(v1, v2)), float(max(v1, v2))


def _safe_minmax_norm(value: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> Optional[float]:
    if value is None or vmin is None or vmax is None:
        return None
    span = float(vmax - vmin)
    if abs(span) < 1e-12:
        return 0.0
    return float((value - vmin) / span)


def _region_component_mask(labels_total: np.ndarray, label_id: int) -> np.ndarray:
    return np.where(labels_total == int(label_id), 255, 0).astype(np.uint8)


def _region_max_feret(labels_total: np.ndarray, label_id: int) -> float:
    comp_mask = _region_component_mask(labels_total, label_id)
    if not np.any(comp_mask > 0):
        return 0.0
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if cnt is None or len(cnt) == 0:
        return 0.0
    hull = cv2.convexHull(cnt, returnPoints=True)
    pts = hull.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] <= 1:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.max(d2)))


def _feature_point_rows(regions: List[Dict[str, Any]], labels_total: np.ndarray, roi: np.ndarray,
                        x_fit: Optional[Dict[str, Any]], y_fit: Optional[Dict[str, Any]],
                        label_map: Dict[int, str], cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    roi_w = int(roi[2] - roi[0])
    roi_h = int(roi[3] - roi[1])
    total_frame_px = max(1, roi_w * roi_h)

    x_min, x_max = _axis_value_range_from_roi(x_fit, float(roi[0]), float(roi[2] - 1))
    y_min, y_max = _axis_value_range_from_roi(y_fit, float(roi[1]), float(roi[3] - 1))
    feret_norm_denom = float(math.hypot(float(roi_w), float(roi_h)))
    if feret_norm_denom < 1e-12:
        feret_norm_denom = None

    rows = []
    for reg in regions:
        cx_global = float(reg["centroid_x"] + roi[0])
        cy_global = float(reg["centroid_y"] + roi[1])
        x_val = pixel_to_axis(x_fit, cx_global)
        y_val = pixel_to_axis(y_fit, cy_global)
        x_norm = _safe_minmax_norm(x_val, x_min, x_max)
        y_norm = _safe_minmax_norm(y_val, y_min, y_max)
        max_feret = _region_max_feret(labels_total, int(reg["label"]))
        max_feret_norm = None
        if feret_norm_denom is not None:
            max_feret_norm = float(max_feret / feret_norm_denom)
        area_norm = float(int(reg["area"]) / float(total_frame_px))
        coord_id = format_coord(x_val, y_val, cfg.axis_value_round)

        reg["x_value"] = x_val
        reg["y_value"] = y_val
        reg["coord_id"] = coord_id
        reg["phase_label"] = label_map.get(int(reg["label"]), "")

        rows.append({
            "node_id": coord_id,
            "x": x_val,
            "y": y_val,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "max_feret": max_feret,
            "max_feret_norm": max_feret_norm,
            "area_px": int(reg["area"]),
            "area_norm": area_norm,
            "phase_label": reg["phase_label"],
            "center_px_x_local": float(reg["centroid_x"]),
            "center_px_y_local": float(reg["centroid_y"]),
            "center_px_x_global": cx_global,
            "center_px_y_global": cy_global,
            "raw_centroid_x_local": float(reg["raw_centroid_x"]),
            "raw_centroid_y_local": float(reg["raw_centroid_y"]),
            "center_method": reg["center_method"],
            "center_shift": float(reg["center_shift"]),
            "bbox_x": int(reg["bbox_x"]),
            "bbox_y": int(reg["bbox_y"]),
            "bbox_w": int(reg["bbox_w"]),
            "bbox_h": int(reg["bbox_h"]),
        })
    return rows



def _first_region_label_along(labels_total: np.ndarray, x: int, y: int, dx: int, dy: int,
                              max_step: int) -> int:
    h, w = labels_total.shape[:2]
    for step in range(1, max_step + 1):
        xx = x + dx * step
        yy = y + dy * step
        if xx < 0 or xx >= w or yy < 0 or yy >= h:
            break
        lab = int(labels_total[yy, xx])
        if lab > 0:
            return lab
    return 0


def compute_region_adjacency(wall_mask: np.ndarray, labels_total: np.ndarray,
                             regions: List[Dict[str, Any]], min_shared_px: int,
                             probe_radius: int = 2) -> List[Dict[str, Any]]:
    region_by_label = {int(r["label"]): r for r in regions}
    labels_set = set(region_by_label.keys())
    pair_pixels: Dict[Tuple[int, int], set] = {}
    ys, xs = np.where(wall_mask > 0)
    direction_pairs = [
        ((-1, 0), (1, 0)),    # horizontal
        ((0, -1), (0, 1)),    # vertical
        ((-1, -1), (1, 1)),   # diagonal \
        ((1, -1), (-1, 1)),   # diagonal /
    ]

    for y, x in zip(ys.tolist(), xs.tolist()):
        pair_hits = set()
        for (dx1, dy1), (dx2, dy2) in direction_pairs:
            lab1 = _first_region_label_along(labels_total, x, y, dx1, dy1, probe_radius)
            lab2 = _first_region_label_along(labels_total, x, y, dx2, dy2, probe_radius)
            if lab1 <= 0 or lab2 <= 0 or lab1 == lab2:
                continue
            if lab1 not in labels_set or lab2 not in labels_set:
                continue
            pair = tuple(sorted((int(lab1), int(lab2))))
            pair_hits.add(pair)
        for pair in pair_hits:
            pair_pixels.setdefault(pair, set()).add((int(x), int(y)))

    edges = []
    for (li, lj), pixset in pair_pixels.items():
        shared = len(pixset)
        if shared >= int(min_shared_px):
            edges.append({"u_label": int(li), "v_label": int(lj), "shared_boundary_px": int(shared)})

    edges.sort(key=lambda t: (-t["shared_boundary_px"], t["u_label"], t["v_label"]))
    return edges


def poly_to_mask(shape_hw: Tuple[int, int], poly: List[List[float]]) -> np.ndarray:

    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def box_center_local(item: Dict[str, Any], roi: np.ndarray) -> Tuple[float, float]:
    bx1, by1, bx2, by2 = item["bbox"]
    return 0.5 * (bx1 + bx2) - float(roi[0]), 0.5 * (by1 + by2) - float(roi[1])


def point_in_region(labels_total: np.ndarray, x: float, y: float) -> int:
    h, w = labels_total.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return int(labels_total[yi, xi])
    return 0


def nearest_region_label(labels_total: np.ndarray, x: float, y: float, radius: int) -> int:
    h, w = labels_total.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    r = int(max(1, radius))
    x1, x2 = max(0, xi - r), min(w, xi + r + 1)
    y1, y2 = max(0, yi - r), min(h, yi + r + 1)
    patch = labels_total[y1:y2, x1:x2]
    if patch.size == 0:
        return 0
    labels = [int(v) for v in np.unique(patch) if int(v) > 0]
    if not labels:
        return 0
    best_label = 0
    best_d2 = 1e18
    for yy in range(y1, y2):
        for xx in range(x1, x2):
            lab = int(labels_total[yy, xx])
            if lab <= 0:
                continue
            d2 = float((xx - x) ** 2 + (yy - y) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best_label = lab
    return best_label


def bbox_distance(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return float(math.hypot(dx, dy))


def bbox_from_component(stats_row) -> Tuple[int, int, int, int]:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h



def likely_phase_label(text: str, score: float, cfg: PreprocessConfig) -> bool:
    t = str(text).strip()
    if not t or score < cfg.label_min_score or len(t) > cfg.label_max_chars:
        return False
    if clean_numeric_text(t) is not None:
        return False
    lower = t.lower()
    banned = ["temperature", "composition", "at.%", "wt.%", "mol.%", "°c", "celsius", "t[c]", "x(", "x（"]
    if any(b in lower for b in banned):
        return False
    return True


def _bbox_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _text_rank(item: Dict[str, Any]) -> float:
    area = _bbox_area(item["bbox"])
    length_bonus = min(len(str(item["text"]).strip()), 24)
    return float(item.get("score", 1.0)) * (1.0 + 0.015 * math.sqrt(max(area, 1.0))) + 0.01 * length_bonus


def _component_endpoints_pca(xs: np.ndarray, ys: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    if len(pts) == 1:
        p = (float(pts[0, 0]), float(pts[0, 1]))
        return p, p
    mean = pts.mean(axis=0, keepdims=True)
    cen = pts - mean
    _, _, vt = np.linalg.svd(cen, full_matrices=False)
    axis = vt[0]
    proj = cen @ axis
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))




def normalize_phase_text(text: str) -> str:
    t = str(text).strip()
    if not t:
        return ""
    t = t.replace(" ", "")
    t = t.replace("Ａ", "A").replace("ｌ", "l")
    t = t.replace("—", "-").replace("–", "-")
    # common OCR confusions in this dataset
    if t.startswith("AI"):
        t = "Al" + t[2:]
    t = t.replace("AICu", "AlCu").replace("AIFe", "AlFe").replace("AIAu", "AlAu")
    t = t.replace("Bce", "Bcc").replace("Bee", "Bcc").replace("Fce", "Fcc")
    t = t.replace("_|_", "_").replace("__", "_")
    lower = t.lower()
    if lower == 'liquid':
        return 'Liquid'
    if lower == 'bcc':
        return 'Bcc'
    if lower == 'fcc':
        return 'Fcc'
    if lower == 'hcp':
        return 'Hcp'
    suffix_map = {
        '8': 'δ', 'B': 'β', 'y': 'γ', 'Y': 'γ', 'r': 'γ',
        'n': 'η', 'h': 'η', 'H': 'H', 'S': 'ζ', 's': 'ζ',
        '5': 'ξ', 'x': 'ξ', 'X': 'ξ', '0': 'θ', 'O': 'θ'
    }
    m = re.match(r'^(Al[A-Za-z0-9]+_)(.+)$', t)
    if m:
        stem, suf = m.groups()
        suf = suf.strip('_')
        if len(suf) == 1 and suf in suffix_map:
            t = stem + suffix_map[suf]
    t = t.replace('Al5Fe,', 'Al5Fe').replace('Al5Fe4', 'Al5Fe4')
    return t

def _is_common_phase_name(text: str) -> bool:
    t = normalize_phase_text(text)
    return t in {'Liquid', 'Bcc', 'Fcc', 'Hcp'}


def _arrow_text_candidate_ok(text: str) -> bool:
    t = normalize_phase_text(text)
    if not t:
        return False
    if _is_common_phase_name(t):
        return False
    if len(t) <= 1:
        return False
    if clean_numeric_text(t) is not None:
        return False
    return True



def _item_local_bbox(item: Dict[str, Any], roi: np.ndarray) -> Tuple[float, float, float, float]:
    bx1, by1, bx2, by2 = item["bbox"]
    return float(bx1 - roi[0]), float(by1 - roi[1]), float(bx2 - roi[0]), float(by2 - roi[1])


def _point_to_bbox_distance(x: float, y: float, box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(math.hypot(dx, dy))


def _endpoint_box_pair_score(end_a, end_b, item: Dict[str, Any], roi: np.ndarray, tail_max: float):
    box = _item_local_bbox(item, roi)
    cx, cy = _candidate_item_local_center(item, roi)
    cand = []
    for tail, tip in ((end_a, end_b), (end_b, end_a)):
        d_tail_box = _point_to_bbox_distance(tail[0], tail[1], box)
        d_tip_box = _point_to_bbox_distance(tip[0], tip[1], box)
        d_tail_center = math.hypot(cx - tail[0], cy - tail[1])
        d_tip_center = math.hypot(cx - tip[0], cy - tip[1])
        if d_tail_box > tail_max:
            continue
        if d_tip_box <= d_tail_box + 4.0:
            continue
        axis = np.array([tip[0] - tail[0], tip[1] - tail[1]], dtype=np.float32)
        to_text = np.array([cx - tail[0], cy - tail[1]], dtype=np.float32)
        n1 = float(np.linalg.norm(axis))
        n2 = float(np.linalg.norm(to_text))
        cosang = 0.0
        if n1 > 1e-6 and n2 > 1e-6:
            cosang = float(np.dot(axis, to_text) / (n1 * n2))
        if cosang > 0.45:
            continue
        score = _text_rank(item) - 0.05 * d_tail_box - 0.015 * max(cosang, 0.0)
        score -= 0.01 * min(d_tail_center, 120.0)
        score += 0.01 * min(len(str(item.get('text',''))), 24)
        if '_' in str(item.get('text','')) or any(ch in str(item.get('text','')) for ch in ('β','γ','δ','ζ','η','ξ','θ')):
            score += 0.18
        cand.append({
            'tail': (float(tail[0]), float(tail[1])),
            'tip': (float(tip[0]), float(tip[1])),
            'd_tail_box': float(d_tail_box),
            'd_tip_box': float(d_tip_box),
            'd_tail_center': float(d_tail_center),
            'd_tip_center': float(d_tip_center),
            'cosang': float(cosang),
            'pair_score': float(score),
        })
    if not cand:
        return None
    cand.sort(key=lambda z: z['pair_score'], reverse=True)
    return cand[0]

def _candidate_item_local_center(item: Dict[str, Any], roi: np.ndarray) -> Tuple[float, float]:
    return box_center_local(item, roi)


def _make_arrow_text_pairs(candidates: List[Dict[str, Any]], arrow_comps: List[Dict[str, Any]],
                           roi: np.ndarray, cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    pairs = []
    tail_max = float(getattr(cfg, 'label_arrow_tail_text_max_px', 72))
    for comp in arrow_comps:
        end_a = comp['end_a']
        end_b = comp['end_b']
        for item in candidates:
            txt = normalize_phase_text(item['text'])
            if not _arrow_text_candidate_ok(txt):
                continue
            best = _endpoint_box_pair_score(end_a, end_b, item, roi, tail_max)
            if best is None:
                continue
            p = dict(best)
            p.update({
                'item': item,
                'comp': comp,
            })
            pairs.append(p)
    pairs.sort(key=lambda z: z['pair_score'], reverse=True)
    chosen = []
    used_items = set()
    used_comps = set()
    for p in pairs:
        iid = id(p['item'])
        cid = int(p['comp']['id'])
        if iid in used_items or cid in used_comps:
            continue
        chosen.append(p)
        used_items.add(iid)
        used_comps.add(cid)
    return chosen




def _mask_from_labels(labels: np.ndarray, keep_ids: List[int]) -> np.ndarray:
    if labels.size == 0 or not keep_ids:
        return np.zeros_like(labels, dtype=np.uint8)
    out = np.isin(labels, np.asarray(keep_ids, dtype=labels.dtype)).astype(np.uint8) * 255
    return out


def _comp_geometry(mask: np.ndarray) -> Dict[str, Any]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"area": 0}
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    area = int(len(xs))
    bbox_area = max(1, w * h)
    fill = float(area) / float(bbox_area)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_n = 0
    tip_angle = 180.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        approx_n = len(approx)
        if len(approx) >= 3:
            pts = approx[:, 0, :].astype(np.float32)
            angs = []
            for i in range(len(pts)):
                a = pts[i - 1]
                b = pts[i]
                c = pts[(i + 1) % len(pts)]
                v1 = a - b
                v2 = c - b
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angs.append(float(np.degrees(np.arccos(cosang))))
            if angs:
                tip_angle = min(angs)
    return {
        "bbox": (x1, y1, x2 + 1, y2 + 1),
        "w": w,
        "h": h,
        "area": area,
        "fill": fill,
        "aspect": float(max(w, h)) / float(max(1, min(w, h))),
        "approx_n": approx_n,
        "tip_angle": tip_angle,
        "xs": xs,
        "ys": ys,
    }

def split_black_text_and_arrows(black_mask: np.ndarray, text_mask: np.ndarray, cfg: PreprocessConfig):
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    comps = []
    for i in range(1, nlab):
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        geom = _comp_geometry(comp)
        if geom.get('area', 0) <= 0:
            continue
        x1, y1, x2, y2 = geom['bbox']
        patch_t = text_mask[y1:y2, x1:x2] if text_mask is not None and text_mask.size else np.zeros((y2-y1, x2-x1), np.uint8)
        patch_c = comp[y1:y2, x1:x2]
        overlap = int(np.count_nonzero((patch_t > 0) & (patch_c > 0)))
        overlap_ratio = overlap / float(max(1, geom['area']))
        geom.update({"id": i, "mask": comp, "text_overlap_ratio": overlap_ratio})
        # arrow heuristics: narrow/triangular component with low text overlap
        arrow_like = False
        if overlap_ratio < 0.08:
            if geom['fill'] < 0.62 and geom['aspect'] >= 1.35 and geom['area'] <= 1800:
                arrow_like = True
            if geom['tip_angle'] <= 65.0 and geom['area'] <= 2200:
                arrow_like = True
            # tiny dots are usually punctuation/noise, keep as text-like only if OCR detector agreed
            if geom['area'] < 12:
                arrow_like = False
        geom['arrow_like'] = arrow_like
        comps.append(geom)

    text_ids = [c['id'] for c in comps if not c['arrow_like']]
    arrow_ids = [c['id'] for c in comps if c['arrow_like']]
    text_only_mask = _mask_from_labels(labels, text_ids)
    arrow_only_mask = _mask_from_labels(labels, arrow_ids)
    return comps, labels, text_only_mask, arrow_only_mask



def assign_phase_labels(ocr_items: List[Dict[str, Any]], roi: np.ndarray, labels_total: np.ndarray,
                        text_mask: np.ndarray, black_mask: np.ndarray, cfg: PreprocessConfig):
    h, w = labels_total.shape[:2]
    debug_links = []
    candidates = []
    for item in ocr_items:
        t = normalize_phase_text(item['text'])
        if likely_phase_label(t, float(item.get('score', 1.0)), cfg):
            item2 = dict(item)
            item2['text'] = t
            candidates.append(item2)

    # recompute arrow-only from black mask using grouped text components, then use endpoints
    _, _, text_only_mask, arrow_only_mask = split_black_text_and_arrows(black_mask, text_mask, cfg)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(getattr(cfg, 'label_arrow_text_exclusion_dilate', 2)) + 1,
                                                        2 * int(getattr(cfg, 'label_arrow_text_exclusion_dilate', 2)) + 1))
    arrow_only = cv2.bitwise_and(arrow_only_mask, cv2.bitwise_not(cv2.dilate(text_only_mask, ker, iterations=1)))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(arrow_only, connectivity=8)
    arrow_comps = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = np.where(lab == i, 255, 0).astype(np.uint8)
        ends = _skeleton_endpoints(comp)
        if len(ends) < 2:
            continue
        bbox = bbox_from_component(stats[i])
        arrow_comps.append({'id': i, 'bbox': bbox, 'mask': comp, 'area': area, 'end_a': ends[0], 'end_b': ends[1]})

    arrow_pairs = _make_arrow_text_pairs(candidates, arrow_comps, roi, cfg)
    arrow_item_ids = {id(p['item']) for p in arrow_pairs}
    arrow_prox_item_ids = set(arrow_item_ids)
    tail_soft = float(getattr(cfg, 'label_arrow_tail_text_max_px', 72)) * 1.25
    for comp in arrow_comps:
        end_a = comp['end_a']
        end_b = comp['end_b']
        for item in candidates:
            txt = normalize_phase_text(item['text'])
            if not _arrow_text_candidate_ok(txt):
                continue
            best = _endpoint_box_pair_score(end_a, end_b, item, roi, tail_soft)
            if best is not None:
                arrow_prox_item_ids.add(id(item))

    best_region_item: Dict[int, Dict[str, Any]] = {}

    # direct assignment only for labels not likely controlled by arrows
    for item in candidates:
        if id(item) in arrow_prox_item_ids:
            continue
        cx_local, cy_local = box_center_local(item, roi)
        direct_label = point_in_region(labels_total, cx_local, cy_local)
        if direct_label > 0:
            cur = best_region_item.get(int(direct_label))
            if cur is None or _text_rank(item) > _text_rank(cur):
                best_region_item[int(direct_label)] = item
            debug_links.append({'text': item['text'], 'mode': 'direct', 'target_label': int(direct_label),
                                'text_center': [float(cx_local), float(cy_local)], 'bbox': item['bbox']})

    # arrow-driven assignment can override weak direct labels in broad container regions.
    for pair in arrow_pairs:
        item = pair['item']
        tip_x, tip_y = pair['tip']
        tail_x, tail_y = pair['tail']
        cx_local, cy_local = box_center_local(item, roi)
        if tip_x <= cfg.label_arrow_border_ignore_px or tip_x >= (w - 1 - cfg.label_arrow_border_ignore_px):
            debug_links.append({'text': item['text'], 'mode': 'ignored_boundary_arrow', 'target_label': 0,
                                'text_center': [float(cx_local), float(cy_local)], 'tail': [tail_x, tail_y],
                                'tip': [tip_x, tip_y], 'bbox': item['bbox']})
            continue
        target_label = point_in_region(labels_total, tip_x, tip_y)
        if target_label <= 0:
            target_label = nearest_region_label(labels_total, tip_x, tip_y, max(cfg.label_assign_near_px, 8))
        if target_label <= 0:
            debug_links.append({'text': item['text'], 'mode': 'arrow_unresolved', 'target_label': 0,
                                'text_center': [float(cx_local), float(cy_local)], 'tail': [tail_x, tail_y],
                                'tip': [tip_x, tip_y], 'bbox': item['bbox']})
            continue

        # remove same item from any previous direct region.
        to_del = [lab_id for lab_id, ex in best_region_item.items() if id(ex) == id(item) and int(lab_id) != int(target_label)]
        for lab_id in to_del:
            del best_region_item[lab_id]

        cur = best_region_item.get(int(target_label))
        prefer = cur is None or (_text_rank(item) + 0.20 > _text_rank(cur))
        if prefer:
            best_region_item[int(target_label)] = item
        debug_links.append({'text': item['text'], 'mode': 'arrow', 'target_label': int(target_label),
                            'text_center': [float(cx_local), float(cy_local)],
                            'tail': [tail_x, tail_y], 'tip': [tip_x, tip_y], 'bbox': item['bbox'],
                            'pair_score': float(pair.get('pair_score', 0.0))})

    final_map: Dict[int, str] = {}
    final_items: List[Dict[str, Any]] = []
    for lab_id, item in sorted(best_region_item.items(), key=lambda kv: kv[0]):
        final_map[int(lab_id)] = normalize_phase_text(str(item['text']).strip())
        it = dict(item)
        it['target_label'] = int(lab_id)
        final_items.append(it)
    return final_map, debug_links, final_items



def _skeleton_endpoints(comp_mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = (comp_mask > 0).astype(np.uint8) * 255
    if mask.ndim != 2 or mask.size == 0:
        return []
    skel = cv2.ximgproc.thinning(mask) if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning") else mask
    pts = np.argwhere(skel > 0)
    if len(pts) == 0:
        return []
    h, w = skel.shape[:2]
    ends = []
    for y, x in pts:
        x1 = max(0, x - 1); x2 = min(w, x + 2)
        y1 = max(0, y - 1); y2 = min(h, y + 2)
        neigh = int(np.count_nonzero(skel[y1:y2, x1:x2] > 0)) - 1
        if neigh == 1:
            ends.append((int(x), int(y)))
    if len(ends) >= 2:
        # choose farthest pair
        best = None
        best_d2 = -1
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                dx = ends[i][0] - ends[j][0]
                dy = ends[i][1] - ends[j][1]
                d2 = dx * dx + dy * dy
                if d2 > best_d2:
                    best_d2 = d2
                    best = (ends[i], ends[j])
        return [best[0], best[1]] if best is not None else ends[:2]
    # fallback: use farthest bbox corners along foreground points
    xs = pts[:, 1].astype(np.int32)
    ys = pts[:, 0].astype(np.int32)
    idx_minx = int(np.argmin(xs)); idx_maxx = int(np.argmax(xs))
    p1 = (int(xs[idx_minx]), int(ys[idx_minx]))
    p2 = (int(xs[idx_maxx]), int(ys[idx_maxx]))
    if p1 == p2 and len(pts) >= 2:
        idx_miny = int(np.argmin(ys)); idx_maxy = int(np.argmax(ys))
        p1 = (int(xs[idx_miny]), int(ys[idx_miny]))
        p2 = (int(xs[idx_maxy]), int(ys[idx_maxy]))
    return [p1, p2] if p1 != p2 else [p1]


def _draw_text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int],
                            font_scale: float = 0.45, thickness: int = 1,
                            color: Tuple[int, int, int] = (0, 0, 0),
                            outline: Tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def build_region_network_overlay(crop_bgr: np.ndarray, regions: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> np.ndarray:
    out = crop_bgr.copy()
    region_by_label = {int(r["label"]): r for r in regions}
    # draw edges first
    for edge in edges:
        ru = region_by_label.get(int(edge["u_label"]))
        rv = region_by_label.get(int(edge["v_label"]))
        if ru is None or rv is None:
            continue
        p1 = (int(round(ru["centroid_x"])), int(round(ru["centroid_y"])))
        p2 = (int(round(rv["centroid_x"])), int(round(rv["centroid_y"])))
        cv2.line(out, p1, p2, (0, 128, 255), 1, cv2.LINE_AA)
    # then draw nodes and ids
    for reg in regions:
        p = (int(round(reg["centroid_x"])), int(round(reg["centroid_y"])))
        cv2.circle(out, p, 4, (0, 255, 0), -1, cv2.LINE_AA)
        txt = str(reg.get("coord_id", reg.get("label", "")))
        tx = min(max(2, p[0] + 6), out.shape[1] - 10)
        ty = min(max(12, p[1] - 6), out.shape[0] - 4)
        _draw_text_with_outline(out, txt, (tx, ty), font_scale=0.35, thickness=1)
    return out


def build_ocr_debug_overlay(crop_bgr: np.ndarray, selected_label_items: List[Dict[str, Any]],
                            roi: np.ndarray, label_debug_links: List[Dict[str, Any]]) -> np.ndarray:
    out = crop_bgr.copy()
    selected_map = {}
    for item in selected_label_items:
        selected_map[(str(item.get("text", "")), int(item.get("target_label", 0)))] = item
    # draw final selected boxes
    for item in selected_label_items:
        bx1, by1, bx2, by2 = item["bbox"]
        x1 = int(round(bx1 - float(roi[0]))); y1 = int(round(by1 - float(roi[1])))
        x2 = int(round(bx2 - float(roi[0]))); y2 = int(round(by2 - float(roi[1])))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
        txt = str(item.get("text", ""))
        _draw_text_with_outline(out, txt, (x1, max(12, y1 - 4)), font_scale=0.45, thickness=1)
    # draw assignment links
    for link in label_debug_links:
        lab = int(link.get("target_label", 0))
        mode = str(link.get("mode", ""))
        cx, cy = link.get("text_center", [None, None])
        if cx is None or cy is None:
            continue
        p_text = (int(round(cx)), int(round(cy)))
        if mode.startswith("arrow"):
            tail = link.get("tail")
            tip = link.get("tip")
            if tail is not None:
                p_tail = (int(round(tail[0])), int(round(tail[1])))
                cv2.line(out, p_text, p_tail, (255, 0, 255), 1, cv2.LINE_AA)
            if tip is not None:
                p_tip = (int(round(tip[0])), int(round(tip[1])))
                cv2.arrowedLine(out, p_text, p_tip, (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.15)
        elif lab > 0:
            # target region center not directly passed; keep a small marker at text center
            cv2.circle(out, p_text, 2, (0, 255, 255), -1, cv2.LINE_AA)
    return out




def export_region_outputs(image_path: Path, out_dir: Path, preprocess_out: Dict[str, Any], label_ocr_items: List[Dict[str, Any]],
                          axis_map: Dict[str, Any], cfg: PreprocessConfig):
    ensure_dir(out_dir)
    roi = preprocess_out["roi"]
    crop_bgr = preprocess_out["crop_bgr"]
    regions = preprocess_out["regions_total"]
    labels_total = preprocess_out["labels_total"]
    wall_mask = preprocess_out["wall_mask"]

    label_map, label_debug_links, selected_label_items = assign_phase_labels(
        ocr_items=label_ocr_items,
        roi=roi,
        labels_total=labels_total,
        text_mask=preprocess_out["text_mask"],
        black_mask=preprocess_out["black_mask"],
        cfg=cfg,
    )

    x_fit = axis_map.get("x_fit")
    y_fit = axis_map.get("y_fit")
    region_rows = _feature_point_rows(regions, labels_total, roi, x_fit, y_fit, label_map, cfg)
    region_by_label = {int(r["label"]): r for r in regions}

    edges = compute_region_adjacency(wall_mask, labels_total, regions, cfg.adjacency_min_shared_boundary_px, cfg.adjacency_probe_radius)
    edge_rows = []
    roi_w = float(roi[2] - roi[0])
    roi_h = float(roi[3] - roi[1])
    edge_distance_norm_denom = math.hypot(roi_w, roi_h)
    if edge_distance_norm_denom < 1e-12:
        edge_distance_norm_denom = None
    for edge in edges:
        ru = region_by_label[int(edge["u_label"])]
        rv = region_by_label[int(edge["v_label"])]
        edge_distance_px = math.hypot(
            float(ru["centroid_x"]) - float(rv["centroid_x"]),
            float(ru["centroid_y"]) - float(rv["centroid_y"]),
        )
        edge_distance = None
        if edge_distance_norm_denom is not None:
            edge_distance = float(edge_distance_px / edge_distance_norm_denom)
        edge_rows.append({
            "node_u": ru["coord_id"],
            "node_v": rv["coord_id"],
            "shared_boundary_px": int(edge["shared_boundary_px"]),
            "edge_distance": edge_distance,
        })

    pd.DataFrame(edge_rows).to_csv(out_dir / "adjacency_edges.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(region_rows).to_excel(out_dir / "point_features.xlsx", index=False)

    network_overlay = build_region_network_overlay(crop_bgr, regions, edges)
    save_image_unicode(out_dir / "region_network_overlay.png", network_overlay, dpi=cfg.output_dpi)

    ocr_debug = build_ocr_debug_overlay(crop_bgr, selected_label_items, roi, label_debug_links)
    save_image_unicode(out_dir / "ocr_label_debug.png", ocr_debug, dpi=cfg.output_dpi)

    with open(out_dir / "axis_mapping.json", "w", encoding="utf-8") as f:
        json.dump(axis_map, f, ensure_ascii=False, indent=2)
    with open(out_dir / "phase_label_debug.json", "w", encoding="utf-8") as f:
        json.dump(label_debug_links, f, ensure_ascii=False, indent=2)



def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    cfg = PreprocessConfig(paddle_device=normalize_device_string(args.paddle_device))

    image_paths = list_images(input_dir, args.main_prefix)
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {input_dir}, prefix={args.main_prefix}")

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    text_detector = build_paddle_text_detector(cfg)
    ocr_reader = build_ocr_reader(cfg)
    per_image_dir = output_dir / "per_image_outputs"
    ensure_dir(per_image_dir)
    debug_dir = output_dir / "debug" if args.save_debug else None
    if debug_dir is not None:
        ensure_dir(debug_dir)

    summary_rows = []
    for path in tqdm(image_paths, desc="extract graph"):
        one_debug_dir = (debug_dir / path.stem) if debug_dir is not None else None
        pre = preprocess_one(path, cfg, text_detector, save_debug_dir=one_debug_dir)
        axis_ocr_items = run_multiorientation_ocr(pre["bgr"], ocr_reader, pre["roi"], cfg)
        label_ocr_items = run_label_ocr(pre["crop_bgr"], pre["black_mask"], pre["text_mask"], ocr_reader, pre["roi"], cfg)
        axis_map = build_axis_mapping(axis_ocr_items, pre["roi"], cfg)
        out_dir = per_image_dir / path.stem
        export_region_outputs(path, out_dir, pre, label_ocr_items, axis_map, cfg)
        summary_rows.append({
            "image_name": path.name,
            "display_name": phase_display_name(path.name),
            "n_regions": len(pre["regions_total"]),
            "x_mapping_ok": bool(axis_map.get("x_mapping_ok", False)),
            "y_mapping_ok": bool(axis_map.get("y_mapping_ok", False)),
            "frame_method": pre["frame_debug"].get("method", "unknown"),
        })

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print("Processing complete")
    print(f"Input images: {len(image_paths)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
