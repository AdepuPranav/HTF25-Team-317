from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn
import cv2
import tempfile
import os
import time
import asyncio
import json
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
import csv
import math
import hashlib
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# Azure integrations (minimal, optional)
try:
    from azure_integrations import (
        blob_enabled,
        blob_upload_bytes,
        blob_list_videos,
        resolve_path_for_processing,
        sb_enabled,
        sb_send_json,
    )
except Exception:
    blob_enabled = lambda: False
    blob_upload_bytes = None
    blob_list_videos = lambda: []
    resolve_path_for_processing = None
    sb_enabled = lambda: False
    sb_send_json = None

from fastapi import FastAPI
app = FastAPI(title="Crowd Data Ingestion - Video Uploader")

@app.get("/")
async def root():
    return {"status": "ok", "service": "video-uploader"}

# In-memory location configs keyed by filename.
# Each config: {"location_id": str, "area_m2": float, "roi": List[[x,y], ...], "lat": float, "lon": float}
LOCATION_CONFIGS: Dict[str, Dict[str, object]] = {}

# Persisted locations file
LOCATIONS_DB = "locations.json"

def _load_locations() -> None:
    try:
        if os.path.isfile(LOCATIONS_DB):
            with open(LOCATIONS_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Basic validation per entry
                    for k, v in list(data.items()):
                        if not isinstance(v, dict):
                            del data[k]
                    LOCATION_CONFIGS.update(data)
    except Exception:
        pass

# Load persisted locations after helpers and dict are defined
_load_locations()

def _save_locations() -> None:
    try:
        with open(LOCATIONS_DB, "w", encoding="utf-8") as f:
            json.dump(LOCATION_CONFIGS, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Global live metrics per filename for map/dashboard
# { filename: { ts, location_id, count, density_mp, density_m2 (optional), lat, lon, severity } }
GLOBAL_METRICS: Dict[str, Dict[str, object]] = {}

# Per-source severity smoothing and cooldown state
SEVERITY_STATE: Dict[str, Dict[str, object]] = {}

# Traffic dataset and prediction helpers (vehicle congestion)
TRAFFIC_DATA: List[Dict[str, object]] = []

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _load_traffic_csv(path: str = 'hyderabad_synth.csv') -> None:
    global TRAFFIC_DATA
    TRAFFIC_DATA = []
    if not os.path.isfile(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts = row.get('timestamp')
                    vc = float(row.get('vehicle_count') or 0.0)
                    sp = float(row.get('average_speed') or 0.0)
                    coords = (row.get('location_coordinates') or '').strip().strip('"')
                    lat_str, lon_str = coords.split(',') if ',' in coords else (None, None)
                    if not lat_str or not lon_str:
                        continue
                    lat, lon = float(lat_str), float(lon_str)
                    # derive hour and weekday from timestamp
                    dt = datetime.fromisoformat(ts)
                    TRAFFIC_DATA.append({
                        'ts': dt,
                        'hour': dt.hour,
                        'weekday': dt.weekday(),
                        'lat': lat,
                        'lon': lon,
                        'speed': sp,
                        'vehicles': vc,
                    })
                except Exception:
                    continue
    except Exception:
        TRAFFIC_DATA = []

def _predict_speed(lat: float, lon: float, when: datetime) -> float:
    # Simple spatiotemporal KNN with inverse-distance weighting + hour proximity
    if not TRAFFIC_DATA:
        return 30.0  # km/h fallback
    # parameters
    max_neighbors = 50
    spatial_radius_m = 3000.0  # 3 km
    hour = when.hour
    wd = when.weekday()
    neighbors = []
    for rec in TRAFFIC_DATA:
        if abs(rec['weekday'] - wd) > 6:  # not needed, but keep structure
            pass
        hdist = min(abs(rec['hour'] - hour), 24 - abs(rec['hour'] - hour))
        # prefer same part of day
        hour_w = 1.0 + (max(0, 6 - hdist) / 6.0)
        d = _haversine_m(lat, lon, rec['lat'], rec['lon'])
        if d <= spatial_radius_m:
            neighbors.append((d, hour_w, rec['speed']))
    if not neighbors:
        # fallback to global median
        try:
            speeds = sorted([rec['speed'] for rec in TRAFFIC_DATA if rec['speed'] > 0])
            return float(speeds[len(speeds)//2]) if speeds else 30.0
        except Exception:
            return 30.0
    neighbors.sort(key=lambda t: t[0])
    neighbors = neighbors[:max_neighbors]
    # inverse distance weighting with hour weight
    num = 0.0
    den = 0.0
    for d, hw, sp in neighbors:
        w = (hw) * (1.0 / max(1.0, d))
        num += w * sp
        den += w
    return max(5.0, num / den) if den > 0 else 30.0

def _predict_route_time_seconds(coords: List[Dict[str, float]], when_iso: Optional[str]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    when = None
    try:
        when = datetime.fromisoformat(when_iso) if when_iso else datetime.utcnow()
    except Exception:
        when = datetime.utcnow()
    total_sec = 0.0
    for i in range(len(coords)-1):
        a = coords[i]; b = coords[i+1]
        seg_m = _haversine_m(a['lat'], a['lng'], b['lat'], b['lng']) if 'lng' in a else _haversine_m(a['lat'], a['lon'], b['lat'], b['lon'])
        mid_lat = 0.5 * (a['lat'] + b['lat'])
        mid_lon = 0.5 * ((a.get('lng', a.get('lon'))) + (b.get('lng', b.get('lon'))))
        sp_kmh = _predict_speed(mid_lat, mid_lon, when)
        sp_ms = max(0.5, sp_kmh / 3.6)
        total_sec += (seg_m / sp_ms)
    return total_sec

# Load traffic data at startup
_load_traffic_csv()

# Alerts config and state
CONFIG: Dict[str, float] = {
    "density_threshold": 12.0,
    "density_threshold_m2": 2.5,
    "panic_threshold": 0.7,
    "panic_window": 10,
    "panic_scale": 8.0,
    "ema_alpha": 0.4,
    "hysteresis_margin": 0.15,
    "alert_cooldown_sec": 10.0,
    "occupancy_mod_threshold": 0.18,
    "occupancy_crit_threshold": 0.32,
    "occupancy_boost": 1.2,
    "min_people_for_critical": 4,
}
ALERT_STATE: Dict[str, object] = {"latest": None, "by_source": {}}

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _compute_panic_index(history: deque, current_count: int) -> float:
    if not history:
        return 0.0
    avg = sum(history) / len(history)
    diff = abs(current_count - avg)
    scale = CONFIG.get("panic_scale", 8.0) or 1.0
    v = diff / scale
    return 0.0 if v < 0 else (1.0 if v > 1 else v)

def _record_alert(source: str, count: int, density_per_mp: float, panic_index: float, severity: str):
    alert = {
        "time": _now_iso(),
        "source": source,
        "count": int(count),
        "density_per_mp": round(density_per_mp, 3),
        "panic_index": round(panic_index, 3),
        "severity": severity,
        "message": f"{severity}: density={density_per_mp:.2f}/MP, panic={panic_index:.2f}, count={count}",
    }
    ALERT_STATE["latest"] = alert
    by_src = ALERT_STATE.get("by_source") or {}
    by_src[source] = alert
    ALERT_STATE["by_source"] = by_src

@app.get("/")
async def root():
    return {"status": "ok", "service": "video-uploader"}

@app.post("/video/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        # Allow unknown as well; OpenCV may still read. Just warn.
        pass

    # Save to a temporary file so OpenCV can read it
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed saving uploaded file: {e}")

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Failed to open video. Unsupported or corrupted file.")

    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # Iterate frames quickly without decoding to display
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
    finally:
        cap.release()
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    duration = frame_count / fps if fps and fps > 0 else None

    return JSONResponse(
        {
            "filename": file.filename,
            "frames": frame_count,
            "fps": round(fps, 3) if fps else None,
            "resolution": {"width": width, "height": height},
            "duration_seconds": round(duration, 3) if duration else None,
        }
    )

@app.post("/video/upload")
async def upload_to_blob(file: UploadFile = File(...)):
    """Upload video to Azure Blob Storage if configured."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not blob_enabled():
        raise HTTPException(status_code=500, detail="Blob storage not configured")
    try:
        size = 0
        chunks = []
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        blob_upload_bytes(file.filename, b"".join(chunks), overwrite=True)
        return JSONResponse({"ok": True, "blob": file.filename, "size": size})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to blob: {e}")

def _list_demo_videos() -> List[Dict[str, object]]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    items: List[Dict[str, object]] = []
    for name in os.listdir('.'):
        p = os.path.join('.', name)
        if os.path.isfile(p):
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                try:
                    size = os.path.getsize(p)
                except Exception:
                    size = None
                if size and size > 0:
                    items.append({"filename": name, "path": p, "size_bytes": size})
    return items

_YOLO_MODEL = None

def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if YOLO is None:
            raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
        _YOLO_MODEL = YOLO('yolov8n.pt')
    return _YOLO_MODEL

def _yolo_detect(frame, conf: float = 0.25, iou: float = 0.5, imgsz: int = 960):
    model = _get_yolo()
    res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes = []
    if res and hasattr(res, 'boxes') and res.boxes is not None:
        for b in res.boxes:
            try:
                cls_id = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            except Exception:
                continue
    return boxes

def _nms_boxes(boxes: List[tuple], iou_thresh: float = 0.5) -> List[tuple]:
    if not boxes:
        return []
    # boxes: (x, y, w, h)
    b2 = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
    areas = [max(1, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2) in b2]
    order = sorted(range(len(b2)), key=lambda i: areas[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        x1, y1, x2, y2 = b2[i]
        remain = []
        for j in order:
            xx1 = max(x1, b2[j][0]); yy1 = max(y1, b2[j][1])
            xx2 = min(x2, b2[j][2]); yy2 = min(y2, b2[j][3])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[j] - inter
            iou = inter / union if union > 0 else 0.0
            if iou < iou_thresh:
                remain.append(j)
        order = remain
    return [boxes[i] for i in keep]

def _stable_index(key: str, n: int) -> int:
    if not key or n <= 0:
        return 0
    d = hashlib.md5(key.encode('utf-8')).digest()
    # take 8 bytes for a positive integer
    val = int.from_bytes(d[:8], byteorder='big', signed=False)
    return val % n

def _detect_people(frame) -> List[tuple]:
    h, w = frame.shape[:2]
    boxes: List[tuple] = []
    try:
        boxes += _yolo_detect(frame, conf=0.25, iou=0.5, imgsz=960)
    except Exception:
        pass

    # Tiled 2x2 pass to capture smaller/occluded persons in crowded scenes
    tiles = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w - w // 2, h // 2),
        (0, h // 2, w // 2, h - h // 2),
        (w // 2, h // 2, w - w // 2, h - h // 2),
    ]
    for (tx, ty, tw, th) in tiles:
        if tw < 160 or th < 160:
            continue
        crop = frame[ty:ty + th, tx:tx + tw]
        try:
            b2 = _yolo_detect(crop, conf=0.22, iou=0.5, imgsz=960)
        except Exception:
            b2 = []
        for (x, y, rw, rh) in b2:
            boxes.append((x + tx, y + ty, rw, rh))
    return _nms_boxes(boxes, iou_thresh=0.45)

_YOLO_MODEL = None

def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if YOLO is None:
            raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
        _YOLO_MODEL = YOLO('yolov8n.pt')
    return _YOLO_MODEL

def _analyze_video_file(path: str, frame_stride: int = 5, resize_width: int = 960) -> Dict[str, object]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Failed to open video: {os.path.basename(path)}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    detections_total = 0
    max_in_frame = 0
    timeline: List[int] = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop: rewind to start and try again
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                continue
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            if resize_width and frame is not None and frame.shape[1] > resize_width:
                ratio = resize_width / float(frame.shape[1])
                new_h = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (resize_width, new_h))
            rects = _detect_people(frame)

            count = len(rects)
            timeline.append(count)
            detections_total += count
            if count > max_in_frame:
                max_in_frame = count
            processed += 1
            idx += 1
    finally:
        cap.release()
    avg_per_frame = (detections_total / processed) if processed else 0.0
    duration = (total_frames / fps) if fps and fps > 0 else None
    # Density per megapixel as a camera-agnostic proxy
    area_mp = (width * height) / 1_000_000.0 if width and height else 0.0
    avg_density_per_mp = (avg_per_frame / area_mp) if area_mp else 0.0
    max_density_per_mp = (max_in_frame / area_mp) if area_mp else 0.0
    out = {
        "file": os.path.basename(path),
        "resolution": {"width": width, "height": height},
        "fps": round(fps, 3) if fps else None,
        "frames_total": total_frames,
        "frames_processed": processed,
        "detections_total": int(detections_total),
        "avg_per_processed_frame": round(avg_per_frame, 3),
        "max_in_frame": int(max_in_frame),
        "duration_seconds": round(duration, 3) if duration else None,
        "timeline_sample": timeline[:200],
        "avg_density_per_mp": round(avg_density_per_mp, 3),
        "max_density_per_mp": round(max_density_per_mp, 3)
    }
    # If we have a location config for this filename, include true density per m^2 using average count
    fname = out["file"]
    cfg = LOCATION_CONFIGS.get(fname)
    if cfg and isinstance(cfg.get("area_m2"), (int, float)) and cfg.get("area_m2"):
        area_m2 = float(cfg["area_m2"])  # user-provided calibrated area
        out["avg_density_per_m2"] = round((avg_per_frame / area_m2), 3)
        out["max_density_per_m2"] = round((max_in_frame / area_m2), 3)
        out["location_id"] = cfg.get("location_id")
    return out

@app.get("/videos/demo")
async def list_demo_videos():
    """List videos from Azure Blob Storage (primary) or local directory (fallback)."""
    items = []
    # Try Blob Storage first
    if blob_enabled():
        try:
            items = blob_list_videos()
        except Exception:
            pass
    # Fallback to local if no blobs found
    if not items:
        items = _list_demo_videos()
    if not items:
        return JSONResponse({"videos": [], "message": "No videos found in blob storage or local directory."})
    # Hardcode geotags for first 4 videos (static, non-randomized)
    try:
        presets = [
            ("Secunderabad railway station", 17.4399, 78.4983),
            ("Marredpally", 17.4479, 78.5029),
            ("Nagole", 17.3693, 78.5609),
            ("Charminar", 17.3850, 78.4867),
        ]
        for i, (lid, lat, lon) in enumerate(presets):
            if i >= len(items):
                break
            fname = items[i]["filename"]
            cfg = LOCATION_CONFIGS.get(fname, {})
            # Only set defaults if user hasn't set them yet
            if "location_id" not in cfg: cfg["location_id"] = lid
            if "area_m2" not in cfg: cfg["area_m2"] = 1000.0
            if "roi" not in cfg: cfg["roi"] = []
            if "lat" not in cfg: cfg["lat"] = float(lat)
            if "lon" not in cfg: cfg["lon"] = float(lon)
            LOCATION_CONFIGS[fname] = cfg
        # Do not delete any existing user-defined locations
        _save_locations()
    except Exception:
        pass
    return JSONResponse({"videos": items})

@app.get("/videos/source")
async def list_all_sources():
    """List both local and blob videos for unified access."""
    local = _list_demo_videos()
    blobs = []
    try:
        if blob_enabled():
            blobs = blob_list_videos()
    except Exception:
        pass
    try:
        spots = [
            (17.4399, 78.4983, "Secunderabad railway station"),
            (17.4479, 78.5029, "Marredpally"),
            (17.3693, 78.5609, "Nagole"),
            (17.3850, 78.4867, "Charminar"),
        ]
        # Build a set of already-used coords to avoid duplicates
        used = set()
        for v in LOCATION_CONFIGS.values():
            try:
                la = v.get("lat"); lo = v.get("lon")
                if isinstance(la, (int, float)) and isinstance(lo, (int, float)):
                    used.add((float(la), float(lo)))
            except Exception:
                pass
        assigned = 0
        for i, b in enumerate(blobs or []):
            fname = b.get("filename") if isinstance(b, dict) else None
            if not fname:
                continue
            if fname not in LOCATION_CONFIGS:
                # Pick the first unused spot; if all used, fall back to index
                pick = None
                for s in spots:
                    if (float(s[0]), float(s[1])) not in used:
                        pick = s
                        break
                if pick is None:
                    pick = spots[_stable_index(fname, len(spots))]
                lat, lon, lid = pick
                LOCATION_CONFIGS[fname] = {
                    "location_id": lid,
                    "area_m2": 1000.0,
                    "roi": [],
                    "lat": float(lat),
                    "lon": float(lon),
                }
                used.add((float(lat), float(lon)))
                _save_locations()
                assigned += 1
                if assigned >= 4:
                    break
    except Exception:
        pass
    return JSONResponse({"local": local, "blob": blobs})

import numpy as np

# Example demo endpoint with predictive analysis
@app.post("/videos/analyze/demo")
def analyze_demo_videos(predict_horizon_sec: int = 30):
    # Example: simulated time series for crowd density
    past_density = np.array([8.0, 10.5, 12.3, 14.0, 15.5, 17.2])

    # Simple linear prediction using least squares (you can replace with ML model later)
    x = np.arange(len(past_density))
    coeffs = np.polyfit(x, past_density, 1)  # linear fit
    future_x = np.arange(len(past_density), len(past_density) + predict_horizon_sec // 5)
    predicted_density = np.polyval(coeffs, future_x)

    # Define risk levels
    alerts = []
    for i, val in enumerate(predicted_density):
        if val > 15:
            alerts.append({
                "time": int(i * 5),
                "predicted_density": round(float(val), 2),
                "risk_level": "⚠️ HIGH",
                "message": "High crowd density predicted — potential congestion ahead"
            })
        else:
            alerts.append({
                "time": int(i * 5),
                "predicted_density": round(float(val), 2),
                "risk_level": "🟢 Normal",
                "message": "Crowd stable"
            })

    return JSONResponse({
        "historical_density": past_density.tolist(),
        "predicted_density": predicted_density.tolist(),
        "alerts": alerts
    })


def _severity_for_count(count: int) -> Dict[str, object]:
    if count <= 3:
        return {"label": "SAFE", "color": (0, 200, 0)}
    if count <= 8:
        return {"label": "MODERATE", "color": (0, 215, 255)}  # yellow-ish (BGR)
    return {"label": "CRITICAL", "color": (0, 0, 255)}

def _point_in_polygon(x: int, y: int, poly: List[List[int]]) -> bool:
    # Ray casting algorithm
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside

def _polygon_area(poly: List[List[int]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += (x1 * y2) - (x2 * y1)
    return abs(s) * 0.5

def _severity_for_density(density: float, per_m2: bool, mp_threshold: float, m2_threshold: float) -> Dict[str, object]:
    if per_m2:
        if density <= (0.4 * m2_threshold):
            return {"label": "SAFE", "color": (0, 200, 0)}
        if density <= m2_threshold:
            return {"label": "MODERATE", "color": (0, 215, 255)}
        return {"label": "CRITICAL", "color": (0, 0, 255)}
    else:
        if density <= (0.6 * mp_threshold):
            return {"label": "SAFE", "color": (0, 200, 0)}
        if density <= mp_threshold:
            return {"label": "MODERATE", "color": (0, 215, 255)}
        return {"label": "CRITICAL", "color": (0, 0, 255)}

def _update_severity_with_hysteresis(source: str, dens_value: float, per_m2: bool, mp_thr: float, m2_thr: float) -> Dict[str, object]:
    state = SEVERITY_STATE.get(source, {"ema": None, "last": "SAFE", "last_change": 0.0, "last_alert": 0.0})
    alpha = float(CONFIG.get("ema_alpha", 0.4) or 0.4)
    ema_prev = state.get("ema")
    ema = (alpha * dens_value) + ((1 - alpha) * ema_prev) if isinstance(ema_prev, (int, float)) else dens_value
    margin = float(CONFIG.get("hysteresis_margin", 0.15) or 0.15)
    thr = float(m2_thr if per_m2 else mp_thr)
    # Base bands
    safe_up = 0.6 * thr
    mod_up = thr
    # Hysteresis for downgrades (require a margin below boundaries)
    safe_dn = safe_up * (1 - margin)
    mod_dn = mod_up * (1 - margin)
    last = state.get("last") or "SAFE"
    label = last
    if last == "SAFE":
        if ema >= mod_up:
            label = "CRITICAL"
        elif ema >= safe_up:
            label = "MODERATE"
    elif last == "MODERATE":
        if ema >= mod_up:
            label = "CRITICAL"
        elif ema < safe_dn:
            label = "SAFE"
    else:  # last == CRITICAL
        if ema < mod_dn:
            label = "MODERATE" if ema >= safe_up else "SAFE"
    if label != last:
        state["last_change"] = time.time()
    state["ema"] = float(ema)
    state["last"] = label
    SEVERITY_STATE[source] = state
    color = (0, 200, 0) if label == "SAFE" else ((0, 215, 255) if label == "MODERATE" else (0, 0, 255))
    return {"label": label, "color": color, "ema": ema}

def _stream_overlay(path: str, frame_stride: int = 2, resize_width: int = 960):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield b""
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    idx = 0
    filename = os.path.basename(path)
    cfg = LOCATION_CONFIGS.get(filename)
    if not cfg:
        spots = [
            (17.4399, 78.4983, "Secunderabad railway station"),
            (17.4479, 78.5029, "Marredpally"),
            (17.3693, 78.5609, "Nagole"),
            (17.3850, 78.4867, "Charminar"),
        ]
        idx = _stable_index(filename, len(spots))
        lat, lon, lid = spots[idx]
        cfg = {
            "location_id": lid,
            "area_m2": 1000.0,
            "roi": [],
            "lat": float(lat),
            "lon": float(lon),
        }
        LOCATION_CONFIGS[filename] = cfg
        _save_locations()
    # alert history window for panic index
    panic_window = int(CONFIG.get("panic_window", 10) or 10)
    recent_counts: deque = deque(maxlen=panic_window)
    logged_once = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop: rewind to start and continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                continue
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            if resize_width and frame is not None and frame.shape[1] > resize_width:
                ratio = resize_width / float(frame.shape[1])
                new_h = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (resize_width, new_h))
            rects = _detect_people(frame)

            count = 0
            if rects:
                # If ROI is defined, filter boxes whose center lies inside ROI
                roi = cfg.get("roi") if cfg else None
                filtered = []
                if roi and isinstance(roi, list) and len(roi) >= 3:
                    for (x, y, w, h) in rects:
                        cx, cy = x + w // 2, y + h // 2
                        if _point_in_polygon(cx, cy, roi):
                            filtered.append((x, y, w, h))
                else:
                    filtered = rects
                count = len(filtered)
                for (x, y, w, h) in filtered:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay banner with severity and count
            h, w = frame.shape[:2]
            # Compute density per effective area (ROI if present)
            area_mp = (w * h) / 1_000_000.0 if w and h else 0.0
            if cfg and isinstance(cfg.get("roi"), list) and len(cfg.get("roi")) >= 3:
                roi_px = _polygon_area(cfg.get("roi"))
                eff_mp = (roi_px / 1_000_000.0) if roi_px and roi_px > 0 else area_mp
            else:
                eff_mp = area_mp
            dens = (count / eff_mp) if eff_mp else 0.0

            # Occupancy: fraction of effective pixel area covered by detected boxes
            eff_px = 0.0
            if cfg and isinstance(cfg.get("roi"), list) and len(cfg.get("roi")) >= 3 and roi_px and roi_px > 0:
                eff_px = float(roi_px)
            else:
                eff_px = float(w * h)
            occ = 0.0
            if eff_px > 0 and rects:
                # Use filtered boxes (already ROI-clipped by center) to approximate coverage
                total_person_px = 0.0
                for (bx, by, bw, bh) in (filtered if 'filtered' in locals() and filtered else rects):
                    total_person_px += float(max(0, bw) * max(0, bh))
                occ = min(1.0, total_person_px / eff_px)

            # Temporal stabilization: reduce effect of transient undercounts
            try:
                import numpy as _np
                hist = list(recent_counts) + [count]
                count_for_decision = int(_np.percentile(hist, 75)) if hist else count
            except Exception:
                count_for_decision = count

            # Choose severity metric: m^2 preferred if area is calibrated
            use_m2 = False
            dens_m2 = None
            if cfg and isinstance(cfg.get("area_m2"), (int, float)) and float(cfg.get("area_m2") or 0) > 0:
                use_m2 = True
                area_m2 = float(cfg.get("area_m2"))
                dens_m2 = (count_for_decision / area_m2)
            mp_thr = float(CONFIG.get("density_threshold", 12.0) or 12.0)
            m2_thr = float(CONFIG.get("density_threshold_m2", 2.5) or 2.5)
            # Use decision density for severity/alerts with EMA + hysteresis
            decision_dens = (count_for_decision / eff_mp) if eff_mp else dens
            # Occupancy-aware boost to decision density
            try:
                occ_boost = float(CONFIG.get("occupancy_boost", 1.2) or 1.0)
            except Exception:
                occ_boost = 1.0
            decision_dens_boosted = decision_dens * (1.0 + occ_boost * occ)
            sev = _update_severity_with_hysteresis(filename, (dens_m2 if use_m2 and dens_m2 is not None else decision_dens_boosted), use_m2, mp_thr, m2_thr)

            # Apply occupancy floor upgrades (force at least MODERATE/CRITICAL)
            try:
                occ_mod = float(CONFIG.get("occupancy_mod_threshold", 0.18) or 0.18)
                occ_crit = float(CONFIG.get("occupancy_crit_threshold", 0.32) or 0.32)
                min_crit = int(CONFIG.get("min_people_for_critical", 4) or 4)
            except Exception:
                occ_mod, occ_crit, min_crit = 0.18, 0.32, 4
            upgrade_label = None
            if occ >= occ_crit and count >= min_crit:
                upgrade_label = "CRITICAL"
            elif occ >= occ_mod and count >= 2:
                upgrade_label = "MODERATE"
            if upgrade_label:
                order = {"SAFE": 0, "MODERATE": 1, "CRITICAL": 2}
                if order.get(upgrade_label, 0) > order.get(sev.get('label', 'SAFE'), 0):
                    sev['label'] = upgrade_label
                    sev['color'] = (0, 200, 0) if upgrade_label == 'SAFE' else ((0, 215, 255) if upgrade_label == 'MODERATE' else (0, 0, 255))

            banner_h = max(30, h // 16)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), sev["color"], thickness=-1)
            alpha = 0.35
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            # Compute panic and record alert on threshold or CRITICAL severity (with cooldown)
            panic = _compute_panic_index(recent_counts, count)
            recent_counts.append(count)
            density_to_check = (dens_m2 if use_m2 and dens_m2 is not None else decision_dens_boosted)
            thr = (m2_thr if use_m2 else mp_thr)
            cooldown = float(CONFIG.get('alert_cooldown_sec', 10.0) or 10.0)
            st = SEVERITY_STATE.get(filename, {})
            last_alert = float(st.get('last_alert') or 0.0)
            now = time.time()
            should_alert = (sev['label'] == 'CRITICAL') or (density_to_check > thr) or (panic > (CONFIG.get('panic_threshold', 0.7) or 0.7))
            if should_alert and (now - last_alert >= cooldown):
                _record_alert(filename, count, dens, panic, 'CRITICAL' if sev['label'] == 'CRITICAL' else sev['label'])
                st['last_alert'] = now
                SEVERITY_STATE[filename] = st

            label = f"{sev['label']} | Count: {count} | Dens: {dens:.2f}/MP"
            if use_m2 and dens_m2 is not None:
                label += f" | {dens_m2:.2f}/m^2"
            cv2.putText(frame, label, (10, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # Optional timestamp/second estimate
            if fps and fps > 0:
                sec = int(idx / fps)
                cv2.putText(frame, f"t={sec}s", (w - 90, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw ROI polygon if configured
            if cfg and isinstance(cfg.get("roi"), list) and len(cfg["roi"]) >= 3:
                pts = cfg["roi"]
                poly = [(int(px), int(py)) for (px, py) in pts]
                # Use numpy if available to draw filled polygon with transparency
                try:
                    import numpy as np  # local import to avoid top-level change
                    overlay2 = frame.copy()
                    cv2.fillPoly(overlay2, [np.array(poly, dtype=np.int32)], color=(255, 255, 0))
                    frame = cv2.addWeighted(overlay2, 0.15, frame, 0.85, 0)
                    cv2.polylines(frame, [np.array(poly, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
                except Exception:
                    # Fallback: draw just polyline without fill
                    for i in range(len(poly)):
                        p1 = poly[i]
                        p2 = poly[(i + 1) % len(poly)]
                        cv2.line(frame, p1, p2, (255, 255, 0), 2)

            # Publish live metrics for map/dashboard
            loc_id = cfg.get("location_id") if cfg else filename
            lat = cfg.get("lat") if cfg else None
            lon = cfg.get("lon") if cfg else None
            if not logged_once:
                try:
                    print(f"[STREAM] {filename} placed at location_id='{loc_id}', lat={lat}, lon={lon}")
                except Exception:
                    pass
                logged_once = True
            payload = {
                "ts": int(time.time()),
                "filename": filename,
                "location_id": loc_id,
                "count": int(count),
                "density_mp": round(float(dens), 3),
                "severity": _severity_for_count(count)["label"],
            }
            if cfg and isinstance(cfg.get("area_m2"), (int, float)) and cfg.get("area_m2"):
                payload["density_m2"] = round((count / float(cfg["area_m2"])) if float(cfg["area_m2"]) > 0 else 0.0, 3)
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                payload["lat"] = float(lat)
                payload["lon"] = float(lon)
            GLOBAL_METRICS[filename] = payload

            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                idx += 1
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            idx += 1
    finally:
        cap.release()

@app.get("/videos/stream")
async def stream_demo_video(filename: str, frame_stride: int = 2):
    items = _list_demo_videos()
    # Also check blob videos if enabled
    if blob_enabled():
        try:
            items += blob_list_videos()
        except Exception:
            pass
    if not any(it["filename"] == filename for it in items):
        raise HTTPException(status_code=404, detail="Requested filename not found or empty")
    path = next(it["path"] for it in items if it["filename"] == filename)
    if YOLO is None:
        raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
    # Resolve blob:// paths to temp files; cleanup after streaming
    proc_path, cleanup = resolve_path_for_processing(path) if resolve_path_for_processing else (path, lambda: None)
    def _gen():
        try:
            gen = _stream_overlay(proc_path, frame_stride=frame_stride)
            for chunk in gen:
                yield chunk
        finally:
            try:
                cleanup()
            except Exception:
                pass
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/ml/score_route")
async def score_route(body: Dict[str, object]):
    coords = body.get('path') if isinstance(body, dict) else None
    when = body.get('timestamp') if isinstance(body, dict) else None
    if not isinstance(coords, list) or len(coords) < 2:
        raise HTTPException(status_code=400, detail="path must be an array of coordinates with lat and lng/lon")
    # normalize coords
    norm = []
    for p in coords:
        try:
            lat = float(p.get('lat'))
            lng = float(p.get('lng', p.get('lon')))
            norm.append({'lat': lat, 'lng': lng})
        except Exception:
            continue
    if len(norm) < 2:
        raise HTTPException(status_code=400, detail="insufficient coordinates after normalization")
    total_sec = _predict_route_time_seconds(norm, when if isinstance(when, str) else None)
    return JSONResponse({"predicted_time_seconds": round(float(total_sec), 2)})

@app.get("/dashboard")
async def dashboard():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Crowd Demo Dashboard</title>
      <style>
        body { font-family: system-ui, sans-serif; margin: 20px; }
        #list { margin-bottom: 12px; }
        #controls { margin-bottom: 12px; }
        button { margin: 4px; padding: 8px 12px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .panel { border: 1px solid #ddd; border-radius: 6px; padding: 6px; }
        .panel h4 { margin: 4px 0; font-size: 14px; }
        img { width: 100%; border-radius: 6px; }
        #map { height: 420px; margin-top: 16px; border: 1px solid #ddd; border-radius: 6px; }
        .sev-safe { color: #1a7f37; }
        .sev-moderate { color: #b58900; }
        .sev-critical { color: #d73a49; }
      </style>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
      <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.css" />
      <script src="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.js"></script>
    </head>
    <body>
      <h2>Crowd Detection - Demo Videos</h2>
      <div id="alertBox" style="display:none;padding:12px 14px;border-radius:8px;color:#fff;position:fixed;top:16px;right:16px;z-index:1000;box-shadow:0 6px 16px rgba(0,0,0,0.2);"></div>
      <div id="list">Loading...</div>
      <div id="controls">
        <label>Frame stride: <input id="stride" type="number" value="2" min="1" max="10" /></label>
        <button id="start">Start Selected</button>
        <button id="stop">Stop All</button>
        <button id="useLoc">Select Current Location</button>
        <input id="uloc" type="text" placeholder="Enter location (e.g., Secunderabad Station)" style="width:280px;" />
        <button id="geocode">Locate</button>
        <button id="pickOnMap">Pick on Map</button>
        <button id="recenter">Recenter</button>
        <label style="margin-left:10px;">🔊 Voice Guidance <input type="checkbox" id="voiceToggle" /></label>
        <label style="margin-left:8px;">Lang <select id="voiceLang" style="min-width:120px;"></select></label>
        <label style="margin-left:8px;">Rate <input id="voiceRate" type="range" min="0.5" max="1.5" step="0.1" value="1.0" /></label>
        <label style="margin-left:8px;">Step-by-step <input type="checkbox" id="voiceStep" checked /></label>
        <button id="voiceRepeat" type="button">Repeat</button>
        <button id="voiceStop" type="button">Stop Voice</button>
        <button id="avoidBtn" style="display:none;background:#c62828;color:#fff;">Avoid Crowds</button>
        <br/>
        <input id="dest" type="text" placeholder="Enter destination" style="width:280px;margin-top:6px;" />
        <button id="routeBtn">Route to Destination</button>
      </div>
      <div class="grid" id="grid"></div>

      <h3>Live Congestion Map (OSM + OSRM)</h3>
      <div id="map"></div>
      <script>
        let videos = [];
        let alertHideTimer = null;
        let perSourcePopupAt = {}; // source -> last popup ms
        async function load() {
          const r = await fetch('/videos/demo');
          const j = await r.json();
          const list = document.getElementById('list');
          list.innerHTML = '';
          if (!j.videos || j.videos.length === 0) { list.textContent = 'No videos found.'; return; }
          videos = j.videos;
          j.videos.forEach((v, i) => {
            const id = 'sel_' + i;
            const row = document.createElement('div');
            const cb = document.createElement('input'); cb.type = 'checkbox'; cb.id = id; cb.value = v.filename;
            const lab = document.createElement('label'); lab.setAttribute('for', id); lab.textContent = v.filename;
            const gi = document.createElement('input'); gi.type = 'text'; gi.placeholder = 'Type location to geotag'; gi.style.marginLeft = '8px'; gi.size = 28;
            const gb = document.createElement('button'); gb.textContent = 'Set Geotag'; gb.style.marginLeft = '6px';
            gb.onclick = async () => {
              const q = (gi.value || '').trim();
              if(!q){ alert('Enter a location'); return; }
              try{
                const res = await geocodeOnce(q);
                const payload = { filename: v.filename, lat: res.lat, lon: res.lon, location_id: q };
                const r = await fetch('/locations/geotag', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                if(!r.ok){ throw new Error('server rejected'); }
                locationsCache[v.filename] = Object.assign(locationsCache[v.filename]||{}, { lat: res.lat, lon: res.lon, location_id: q });
                alert('Geotag updated');
              }catch(e){ alert('Failed to set geotag'); }
            };
            row.appendChild(cb); row.appendChild(lab); row.appendChild(gi); row.appendChild(gb);
            list.appendChild(row);
          });
          try {
            const lr = await fetch('/locations');
            const lj = await lr.json();
            locationsCache = lj.locations || {};
          } catch(e) { locationsCache = {}; }
          // Auto-start: select all and start streaming
          try {
            const cbs = Array.from(document.querySelectorAll('#list input[type=checkbox]'));
            cbs.forEach(cb => cb.checked = true);
            setTimeout(startSelected, 200);
          } catch(_) { }
        }

        function stopAll() {
          const grid = document.getElementById('grid');
          if (grid) { grid.innerHTML = ''; }
        }

        function startSelected() {
          stopAll();
          const s = document.getElementById('stride').value || 2;
          const checked = Array.from(document.querySelectorAll('#list input[type=checkbox]:checked'));
          const grid = document.getElementById('grid');
          checked.forEach((c, idx) => {
            const panel = document.createElement('div');
            panel.className = 'panel';
            const title = document.createElement('h4');
            title.textContent = c.value;
            const img = document.createElement('img');
            img.alt = c.value;
            img.src = '/videos/stream?filename=' + encodeURIComponent(c.value) + '&frame_stride=' + s;
            panel.appendChild(title);
            panel.appendChild(img);
            grid.appendChild(panel);
          });

          // Pan map to selected locations and open popups once markers are ready
          const pts = [];
          checked.forEach(c => {
            const cfg = locationsCache[c.value];
            if(cfg && typeof cfg.lat === 'number' && typeof cfg.lon === 'number'){
              pts.push([cfg.lat, cfg.lon]);
            }
          });
          if(pts.length === 1){ map.flyTo(pts[0], 14); }
          if(pts.length > 1){
            const b = L.latLngBounds(pts.map(p => L.latLng(p[0], p[1])));
            map.fitBounds(b.pad(0.2));
          }
          // Try to open the popups shortly after, when markers exist
          setTimeout(() => {
            checked.forEach(c => {
              const key = c.value;
              if(markers.has(key)){
                markers.get(key).marker.openPopup();
              }
            });
            // Initialize and update heat layer
            try{
              if(!heatLayer){
                heatLayer = L.heatLayer([], { radius: 26, blur: 18, maxZoom: 19 });
                heatLayer.addTo(map);
              }
              heatLayer.setLatLngs(heat);
            }catch(_){ }
          }, 800);
        }

        async function pollAlerts(){
          try{
            const r = await fetch('/alerts/by_source');
            if(!r.ok) return;
            const j = await r.json();
            const by = j && j.by_source ? j.by_source : {};
            const entries = Object.values(by);
            if(!entries.length) return;
            if(!userMarker) return; // No user location -> suppress alerts
            const u = userMarker.getLatLng();
            const now = Date.now();
            for(const a of entries){
              if(!a || (a.severity||'').toUpperCase() !== 'CRITICAL') continue;
              const src = a.source || 'unknown';
              // Require a visible marker in range
              if(!markers.has(src)) continue;
              const m = markers.get(src).marker;
              const dist = map.distance(m.getLatLng(), u);
              if(dist > USER_RADIUS_METERS) continue;
              const lastAt = perSourcePopupAt[src] || 0;
              if(now - lastAt < 30000) continue; // 30s cooldown per source
              const box = document.getElementById('alertBox');
              box.style.background = '#c62828';
              box.textContent = `[${a.time}] ${a.severity} @ ${src} — ` + a.message;
              box.style.display = 'block';
              if(alertHideTimer) clearTimeout(alertHideTimer);
              alertHideTimer = setTimeout(()=>{ box.style.display='none'; }, 3000);
              perSourcePopupAt[src] = now;
              break; // show one at a time
            }
          }catch(e){}
        }

        document.getElementById('start').onclick = startSelected;
        document.getElementById('stop').onclick = stopAll;
        load();
        setInterval(pollAlerts, 2000);

        // OSM + OSRM map setup
        const map = L.map('map').setView([20.5937, 78.9629], 5); // India center
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        const markers = new Map();
        let router = null;
        let selected = [];
        let locationsCache = {};
        // Heatmap layer for live crowd severity
        let heatLayer = null;
        let userMarker = null;
        let userCircle = null;
        let pickMode = false;
        const USER_RADIUS_METERS = 3000; // 3 km radius gate for visibility
        let voiceEnabled = false;
        let voiceLang = 'en';
        let voiceRate = 1.0;
        let voiceStepMode = true;
        let lastSpeechText = '';
        const synth = window.speechSynthesis;
        let voiceList = [];
        let geoWatchId = null; // geolocation watch id
        // Persistence keys
        const LS_KEYS = {
          voiceEnabled: 'va_voice_enabled',
          voiceLang: 'va_voice_lang',
          voiceRate: 'va_voice_rate',
          voiceStep: 'va_voice_step',
          userLoc: 'va_user_loc',
          destLoc: 'va_dest_loc'
        };
        // Route state for dynamic re-route
        let lastWaypoints = null; // array of L.latLng
        let lastRouteCoords = null; // array of {lat,lng}

        function populateVoices(){
          try{
            voiceList = synth.getVoices() || [];
            const langs = Array.from(new Set(voiceList.map(v => v.lang))).sort();
            const sel = document.getElementById('voiceLang');
            if(!sel) return;
            const prev = sel.value;
            sel.innerHTML = '';
            langs.forEach(l => {
              const opt = document.createElement('option');
              opt.value = l; opt.textContent = l;
              sel.appendChild(opt);
            });
            if(prev){ sel.value = prev; }
            else {
              const best = langs.find(l => /^en/i.test(l)) || langs[0] || 'en-US';
              sel.value = best;
              voiceLang = best;
            }
          }catch(e){/* ignore */}
        }
        if('speechSynthesis' in window){
          populateVoices();
          if (typeof synth.onvoiceschanged !== 'undefined') {
            synth.onvoiceschanged = populateVoices;
          }
        }

        function loadVoiceSettings(){
          try{
            const ve = localStorage.getItem(LS_KEYS.voiceEnabled);
            if(ve !== null){ voiceEnabled = ve === '1'; document.getElementById('voiceToggle').checked = voiceEnabled; }
            const vl = localStorage.getItem(LS_KEYS.voiceLang);
            if(vl){ voiceLang = vl; const sel = document.getElementById('voiceLang'); if(sel) sel.value = vl; }
            const vr = localStorage.getItem(LS_KEYS.voiceRate);
            if(vr){ const v = parseFloat(vr); if(!isNaN(v)){ voiceRate = v; const r = document.getElementById('voiceRate'); if(r) r.value = String(v); } }
            const vs = localStorage.getItem(LS_KEYS.voiceStep);
            if(vs !== null){ voiceStepMode = vs === '1'; const cb = document.getElementById('voiceStep'); if(cb) cb.checked = voiceStepMode; }
          }catch(_){}
        }
        function saveVoiceSettings(){
          try{
            localStorage.setItem(LS_KEYS.voiceEnabled, voiceEnabled ? '1' : '0');
            localStorage.setItem(LS_KEYS.voiceLang, voiceLang || '');
            localStorage.setItem(LS_KEYS.voiceRate, String(voiceRate));
            localStorage.setItem(LS_KEYS.voiceStep, voiceStepMode ? '1' : '0');
          }catch(_){}}

        function loadUserLoc(){
          try{
            const s = localStorage.getItem(LS_KEYS.userLoc);
            if(!s) return;
            const obj = JSON.parse(s);
            if(obj && typeof obj.lat==='number' && typeof obj.lon==='number'){
              placeUser(obj.lat, obj.lon, null);
            }
          }catch(_){}}
        function saveUserLoc(lat, lon){
          try{ localStorage.setItem(LS_KEYS.userLoc, JSON.stringify({lat, lon})); }catch(_){}}

        function loadDest(){
          try{
            const s = localStorage.getItem(LS_KEYS.destLoc);
            if(!s) return;
            const obj = JSON.parse(s);
            if(obj && typeof obj.lat==='number' && typeof obj.lon==='number'){
              if(destMarker){ destMarker.setLatLng([obj.lat, obj.lon]); }
              else { destMarker = L.marker([obj.lat, obj.lon], { title: 'Destination' }).addTo(map).bindPopup('Destination'); }
              if(destCircle){ destCircle.setLatLng([obj.lat, obj.lon]); destCircle.setRadius(USER_RADIUS_METERS); }
              else { destCircle = L.circle([obj.lat, obj.lon], { radius: USER_RADIUS_METERS, color: '#1976d2', weight: 2, opacity: 0.9, dashArray: '6 8', fillOpacity: 0, interactive: false }).addTo(map); }
              const tf = document.getElementById('dest'); if(tf) tf.value = obj.name || '';
            }
          }catch(_){}}
        function saveDest(lat, lon, name){
          try{ localStorage.setItem(LS_KEYS.destLoc, JSON.stringify({lat, lon, name: name||''})); }catch(_){}}

        // Load persisted settings/state after DOM is ready
        loadVoiceSettings();
        // Do not auto-set user location; wait for explicit user action
        loadDest();

        function speak(text){
          try{
            if(!voiceEnabled || !text || !('speechSynthesis' in window)) return;
            // Stop any ongoing speech
            synth.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = voiceRate; u.pitch = 1.0; u.volume = 1.0; u.lang = voiceLang || 'en-US';
            // choose voice by selected language
            const en = (voiceList || synth.getVoices() || []).find(v => v.lang === voiceLang) ||
                       (voiceList || synth.getVoices() || []).find(v => v.lang && v.lang.startsWith(voiceLang.split('-')[0])) ||
                       null;
            if(en) u.voice = en;
            synth.speak(u);
            lastSpeechText = text;
          }catch(e){/* ignore */}
        }
        function attachVoice(router){
          if(!router) return;
          router.on('routesfound', function(e){
            if(!voiceEnabled) return;
            try{
              const route = e && e.routes && e.routes[0];
              // cache coordinates for dynamic re-route
              lastRouteCoords = route && route.coordinates ? route.coordinates.map(p=>({lat:p.lat,lng:p.lng})) : null;
              const parts = [];
              if(route){
                const hasSteps = route.instructions && route.instructions.length;
                if(voiceStepMode && hasSteps){
                  parts.push('Starting navigation.');

                  route.instructions.forEach((ins, idx) => {
                    if(!ins) return;
                    let t = ins.text || '';
                    parts.push(`Step ${idx+1}. ${t}`);
                  });

                }
                // Always include summary at end
                if(route.summary){
                  const dkm = Math.round((route.summary.totalDistance||0)/100)/10;
                  const min = Math.round((route.summary.totalTime||0)/60);
                  parts.push(`Total distance ${dkm} kilometers. Estimated time ${min} minutes.`);
                } else if(!voiceStepMode || !hasSteps) {
                  parts.push('Route ready.');
                }
              }
              const msg = parts.join(' ');
              speak(msg);
            }catch(err){/* ignore */}
          });
        }
        document.getElementById('voiceToggle').addEventListener('change', (e)=>{
          voiceEnabled = !!e.target.checked;
          if(!voiceEnabled && synth){ synth.cancel(); }
          saveVoiceSettings();
          if(voiceEnabled){
            // simple confirmation to validate audio works post user gesture
            setTimeout(()=>{ try{ speak('Voice guidance enabled'); }catch(_){ } }, 100);
          }
        });
        document.getElementById('voiceLang').addEventListener('change', (e)=>{
          voiceLang = e.target.value || 'en';
          saveVoiceSettings();
        });
        document.getElementById('voiceRate').addEventListener('input', (e)=>{
          const v = parseFloat(e.target.value);
          if(!isNaN(v)) voiceRate = v;
          saveVoiceSettings();
        });
        document.getElementById('voiceStep').addEventListener('change', (e)=>{
          voiceStepMode = !!e.target.checked;
          saveVoiceSettings();
        });
        document.getElementById('voiceRepeat').addEventListener('click', ()=>{
          if(lastSpeechText) speak(lastSpeechText);
        });
        document.getElementById('voiceStop').addEventListener('click', ()=>{
          try{ synth.cancel(); }catch(_){}}
        );

        // Safer-routing helpers and state
        const AVOID_DISTANCE_METERS = 1000; // target distance to move away from crowds
        let destMarker = null; // single destination marker
        let destCircle = null; // destination vicinity radius

        function toRad(d){ return d * Math.PI / 180; }
        function toDeg(r){ return r * 180 / Math.PI; }
        function destPoint(lat, lon, bearingDeg, distanceMeters){
          const R = 6371000; // Earth radius m
          const br = toRad(bearingDeg);
          const φ1 = toRad(lat);
          const λ1 = toRad(lon);
          const δ = distanceMeters / R;
          const sinφ1 = Math.sin(φ1), cosφ1 = Math.cos(φ1);
          const sinδ = Math.sin(δ), cosδ = Math.cos(δ);
          const sinφ2 = sinφ1 * cosδ + cosφ1 * sinδ * Math.cos(br);
          const φ2 = Math.asin(sinφ2);
          const y = Math.sin(br) * sinδ * cosφ1;
          const x = cosδ - sinφ1 * sinφ2;
          const λ2 = λ1 + Math.atan2(y, x);
          return { lat: toDeg(φ2), lon: toDeg(λ2) };
        }

        // Collect CRITICAL markers near either of two centers
        function collectCriticalNearAny(u, d){
          const crit = [];
          markers.forEach(({ marker }) => {
            const mll = marker.getLatLng();
            let near = false;
            if(u){ near = near || (map.distance(mll, u) <= USER_RADIUS_METERS); }
            if(d){ near = near || (map.distance(mll, d) <= USER_RADIUS_METERS); }
            if(!near) return;
            const content = marker.getPopup() && marker.getPopup().getContent();
            if(typeof content === 'string' && content.includes('CRITICAL')){
              crit.push([mll.lat, mll.lng]);
            }
          });
          return crit;
        }

        // Coarse route safety check: any critical marker within threshold to route vertices
        function isRouteUnsafe(routeCoords, crit, thresholdMeters = 200){
          if(!crit.length || !routeCoords || !routeCoords.length) return false;
          for(const c of crit){
            const cll = L.latLng(c[0], c[1]);
            for(const rc of routeCoords){
              const rll = L.latLng(rc.lat, rc.lng);
              if(map.distance(cll, rll) <= thresholdMeters){ return true; }
            }
          }
          return false;
        }

        function computeDetourWaypoint(u, dest, crit){
          if(!crit.length) return null;
          const cx = crit.reduce((a,c)=>a+c[0],0)/crit.length;
          const cy = crit.reduce((a,c)=>a+c[1],0)/crit.length;
          // Vector from user to centroid -> perpendicular
          const vx = cx - u.lat;
          const vy = cy - u.lng;
          let px = vy, py = -vx;
          const norm = Math.hypot(px, py) || 1;
          px /= norm; py /= norm;
          const perpBearing = toDeg(Math.atan2(py, px));
          const detour = destPoint(u.lat, u.lng, perpBearing, AVOID_DISTANCE_METERS);
          return L.latLng(detour.lat, detour.lon);
        }

        let rerouteGuard = false;
        async function scoreRouteML(route){
          try{
            const path = (route.coordinates||[]).map(p => ({ lat: p.lat, lng: p.lng }));
            const body = { path, timestamp: new Date().toISOString() };
            const r = await fetch('/ml/score_route', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if(!r.ok) throw new Error('ml score failed');
            const j = await r.json();
            return typeof j.predicted_time_seconds === 'number' ? j.predicted_time_seconds : Infinity;
          }catch(_){ return Infinity; }
        }

        async function selectBestAlternative(router, e){
          try{
            const routes = e && e.routes ? e.routes : [];
            if(!routes.length) return;
            // Score all alternatives
            const scores = await Promise.all(routes.map(rt => scoreRouteML(rt)));
            let bestIdx = 0;
            for(let i=1;i<scores.length;i++){ if(scores[i] < scores[bestIdx]) bestIdx = i; }
            // If best is not first, select it
            if(bestIdx !== 0){
              try{ if(typeof router._selectRoute === 'function'){ router._selectRoute(routes[bestIdx]); } }catch(_){ }
              // speak summary for selected route
              const best = routes[bestIdx];
              const min = Math.round((scores[bestIdx]||0)/60);
              if(best && best.summary){ speak(`Optimized route selected. Estimated time ${min} minutes.`); }
            }
          }catch(_){ }
        }

        function createRouterWithSafety(waypoints){
          if(router){ map.removeControl(router); router=null; }
          router = L.Routing.control({
            waypoints,
            routeWhileDragging: false,
            addWaypoints: false,
            draggableWaypoints: false,
            showAlternatives: true,
            router: L.Routing.osrmv1({
              serviceUrl: 'https://router.project-osrm.org/route/v1',
              profile: 'car',
              alternatives: 3,
            }),
            lineOptions: {
              styles: [
                { color: '#2ecc71', opacity: 0.9, weight: 6 }, // main: green
                { color: '#27ae60', opacity: 0.6, weight: 8 }
              ]
            },
            altLineOptions: {
              styles: [
                { color: '#9e9e9e', opacity: 0.7, weight: 5, dashArray: '6,6' } // alternatives: grey dashed
              ]
            },
            createMarker: () => null,
          }).addTo(map);
          lastWaypoints = waypoints.slice();
          attachVoice(router);
          router.on('routesfound', async (e) => {
            try{
              const route = e.routes && e.routes[0];
              if(!route) return;
              const coords = (route.coordinates||[]).map(p => ({ lat: p.lat, lng: p.lng }));
              const u = waypoints[0];
              const d = waypoints[waypoints.length-1];
              const crit = collectCriticalNearAny(u, d);
              if(isRouteUnsafe(coords, crit) && !rerouteGuard){
                rerouteGuard = true;
                const detour = computeDetourWaypoint(u, d, crit);
                if(detour){ createRouterWithSafety([ L.latLng(u.lat, u.lng), detour, L.latLng(d.lat, d.lng) ]); }
                setTimeout(() => { rerouteGuard = false; }, 1000);
                return;
              }
              // If route is safe, pick best alternative using ML scoring
              await selectBestAlternative(router, e);
            }catch(_){ /* ignore */ }
          });
        }

        async function geocodeOnce(q){
          const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=1`;
          const resp = await fetch(url, { headers: { 'Accept': 'application/json' } });
          if(!resp.ok) throw new Error('geocoding failed');
          const arr = await resp.json();
          if(!arr || !arr.length) throw new Error('no results');
          return { lat: parseFloat(arr[0].lat), lon: parseFloat(arr[0].lon), name: arr[0].display_name };
        }

        function sevClass(label){
          if(label === 'CRITICAL') return 'sev-critical';
          if(label === 'MODERATE') return 'sev-moderate';
          return 'sev-safe';
        }

        function sevColor(label){
          if(label === 'CRITICAL') return '#d73a49';
          if(label === 'MODERATE') return '#b58900';
          return '#1a7f37';
        }

        async function poll(){
          try{
            const r = await fetch('/congestion?include_idle=0');
            const j = await r.json();
            if(!j.items) return;
            const heat = [];
            const u = userMarker ? userMarker.getLatLng() : null;
            const dLL = destMarker ? destMarker.getLatLng() : null;
            const showAll = !userMarker; // before user location, show all markers
            j.items.forEach(it => {
              if(typeof it.lat !== 'number' || typeof it.lon !== 'number') return;
              const key = it.filename;
              const mll = L.latLng(it.lat, it.lon);
              const sev = (it.severity||'').toString().toUpperCase();
              const inRangeUser = u ? (map.distance(mll, u) <= USER_RADIUS_METERS) : false;
              const inRangeDest = dLL ? (map.distance(mll, dLL) <= USER_RADIUS_METERS) : false;
              const inRange = (u || dLL) ? (inRangeUser || inRangeDest) : true;
              const crowded = sev !== 'SAFE';
              const shouldShow = showAll ? true : (inRange && crowded);
              // Heat intensity by severity (always contribute to overview)
              const intensity = sev === 'CRITICAL' ? 1.0 : (sev === 'MODERATE' ? 0.6 : 0.25);
              heat.push([it.lat, it.lon, intensity]);
              if(!shouldShow){
                if(markers.has(key)){
                  const { marker } = markers.get(key);
                  map.removeLayer(marker);
                  markers.delete(key);
                }
                return;
              }

              const html = `<b>${it.location_id || key}</b><br/>`+
                `<span class="${sevClass(it.severity)}">${it.severity}</span><br/>`+
                `Count: ${it.count}<br/>`+
                `Density: ${(it.density_mp??0).toFixed(2)}/MP` + (it.density_m2? `<br/>${it.density_m2.toFixed(2)}/m²` : '');
              const radius = Math.min(30, 6 + (it.count||0));
              const color = sevColor(it.severity);
              if(markers.has(key)){
                const { marker } = markers.get(key);
                marker.setLatLng([it.lat, it.lon]);
                marker.setStyle({ color, fillColor: color, radius });
                marker.bindPopup(html);
              } else {
                const marker = L.circleMarker([it.lat, it.lon], { radius, color, fillColor: color, fillOpacity: 0.7, weight: 2 }).addTo(map).bindPopup(html);
                marker.on('click', () => {
                  // If user location is known, route from user to this marker.
                  if(userMarker){
                    const u = userMarker.getLatLng();
                    if(router){ map.removeControl(router); router=null; }
                    createRouterWithSafety([ L.latLng(u.lat, u.lng), L.latLng(it.lat, it.lon) ]);
                    return;
                  }
                  // Fallback to original 2-click routing between markers
                  if(selected.length === 2){ selected = []; if(router){ map.removeControl(router); router=null; } }
                  selected.push([it.lat, it.lon]);
                  if(selected.length === 2){
                    createRouterWithSafety([ L.latLng(selected[0][0], selected[0][1]), L.latLng(selected[1][0], selected[1][1]) ]);
                  }
                });
                markers.set(key, { marker });
              }
            });
            // Dynamic re-route if new CRITICAL markers appear along current route
            if(router && lastWaypoints && lastRouteCoords && userMarker){
              try{
                const u = lastWaypoints[0];
                const d = lastWaypoints[lastWaypoints.length-1];
                const crit = collectCriticalNearAny(u, d);
                if(isRouteUnsafe(lastRouteCoords, crit) && !rerouteGuard){
                  rerouteGuard = true;
                  createRouterWithSafety(lastWaypoints);
                  setTimeout(() => { rerouteGuard = false; }, 1000);
                }
              }catch(_){ }
            }
            // Toggle Avoid button visibility based on nearby CRITICAL markers
            const ab = document.getElementById('avoidBtn');
            if(ab && userMarker){
              // quick proximity check using current markers and user
              const u = userMarker.getLatLng();

              let anyCritical = false;
              markers.forEach(({ marker }) => {
                const mll = marker.getLatLng();
                if(map.distance(mll, u) <= USER_RADIUS_METERS){
                  const content = marker.getPopup() && marker.getPopup().getContent();
                  if(typeof content === 'string' && content.includes('CRITICAL')){
                    anyCritical = true;
                  }
                }
              });
              ab.style.display = anyCritical ? 'inline-block' : 'none';
            }
          }catch(e){/* ignore */}
          setTimeout(poll, 3000);
        }
        poll();

        function onUserMoved(){
          try{
            if(!router) return;
            const u = userMarker && userMarker.getLatLng();
            if(!u) return;
            let destLatLng = null;
            if(destMarker){ destLatLng = destMarker.getLatLng(); }
            else if(lastWaypoints && lastWaypoints.length >= 2){
              const d = lastWaypoints[lastWaypoints.length-1];
              destLatLng = L.latLng(d.lat, d.lng);
            }
            if(destLatLng){
              createRouterWithSafety([ L.latLng(u.lat, u.lng), L.latLng(destLatLng.lat, destLatLng.lng) ]);
            }
          }catch(_){ }
        }

        function placeUser(lat, lon, accuracy){
          const latlng = L.latLng(lat, lon);
          if(userMarker){ userMarker.setLatLng(latlng); }
          else {
            userMarker = L.marker(latlng, { title: 'You are here', draggable: true }).addTo(map).bindPopup('You are here');
            userMarker.on('dragend', () => {
              const ll = userMarker.getLatLng();
              if(userCircle){ userCircle.setLatLng(ll); }
              const tf = document.getElementById('uloc');
              if(tf) tf.value = `${ll.lat.toFixed(6)}, ${ll.lng.toFixed(6)}`;
              saveUserLoc(ll.lat, ll.lng);
              onUserMoved();
            });
          }
          if(userCircle){ userCircle.setLatLng(latlng); userCircle.setRadius(USER_RADIUS_METERS); }
          else { userCircle = L.circle(latlng, { radius: USER_RADIUS_METERS, color: '#1976d2', fillColor: '#64b5f6', fillOpacity: 0.2 }).addTo(map); }
          map.flyTo(latlng, Math.max(14, map.getZoom()));
          userMarker.openPopup();
          // Sync text field
          const tf = document.getElementById('uloc');
          if(tf) tf.value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
          saveUserLoc(lat, lon);
          onUserMoved();
        }

        function updateUserLive(lat, lon){
          const latlng = L.latLng(lat, lon);
          if(userMarker){ userMarker.setLatLng(latlng); }
          if(userCircle){ userCircle.setLatLng(latlng); userCircle.setRadius(USER_RADIUS_METERS); }
          saveUserLoc(lat, lon);
          onUserMoved();
        }

        function startGeoWatch(){
          try{
            if(!navigator.geolocation) return;
            if(geoWatchId !== null) return;
            geoWatchId = navigator.geolocation.watchPosition(
              (pos) => {
                const { latitude, longitude } = pos.coords;
                updateUserLive(latitude, longitude);
              },
              (_err) => {},
              { enableHighAccuracy: true, timeout: 15000, maximumAge: 2000 }
            );
          }catch(_){ geoWatchId = null; }
        }
        function stopGeoWatch(){
          try{
            if(geoWatchId !== null && navigator.geolocation && navigator.geolocation.clearWatch){
              navigator.geolocation.clearWatch(geoWatchId);
            }
          }catch(_){ }
          geoWatchId = null;
        }

        document.getElementById('useLoc').onclick = async () => {
          if(!navigator.geolocation){ alert('Geolocation not supported'); return; }
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              const { latitude, longitude, accuracy } = pos.coords;
              placeUser(latitude, longitude, accuracy);
            },
            (err) => { alert('Failed to get location: ' + err.message); },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
          );
        };

        async function geocodeAndPlace(){
          const q = (document.getElementById('uloc').value || '').trim();
          if(!q){ alert('Enter a location'); return; }
          try{
            const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=1`;
            const resp = await fetch(url, { headers: { 'Accept': 'application/json' } });
            if(!resp.ok){ alert('Geocoding failed'); return; }
            const arr = await resp.json();
            if(!arr || !arr.length){ alert('No results found'); return; }
            const best = arr[0];
            const lat = parseFloat(best.lat), lon = parseFloat(best.lon);
            if(Number.isFinite(lat) && Number.isFinite(lon)){
              placeUser(lat, lon, null);
            } else {
              alert('Invalid geocoding result');
            }
          }catch(e){ alert('Geocoding error'); }
        }
        document.getElementById('geocode').onclick = geocodeAndPlace;

        document.getElementById('routeBtn').onclick = async () => {
          if(!userMarker){ alert('Set your current location first'); return; }
          const q = (document.getElementById('dest').value || '').trim();
          if(!q){ alert('Enter a destination'); return; }
          try{
            const dest = await geocodeOnce(q);
            const u = userMarker.getLatLng();
            const waypoints = [ L.latLng(u.lat, u.lng), L.latLng(dest.lat, dest.lon) ];
            createRouterWithSafety(waypoints);
            if(destMarker){ destMarker.setLatLng([dest.lat, dest.lon]); }
            else { destMarker = L.marker([dest.lat, dest.lon], { title: 'Destination' }).addTo(map).bindPopup('Destination'); }
            if(destCircle){ destCircle.setLatLng([dest.lat, dest.lon]); destCircle.setRadius(USER_RADIUS_METERS); }
            else { destCircle = L.circle([dest.lat, dest.lon], { radius: USER_RADIUS_METERS, color: '#1976d2', weight: 2, opacity: 0.9, dashArray: '6 8', fillOpacity: 0, interactive: false }).addTo(map); }
            saveDest(dest.lat, dest.lon, dest.name);
          }catch(e){ alert('Failed to route to destination: ' + e.message); }
        };

        document.getElementById('avoidBtn').onclick = () => {
          if(!userMarker) return;
          const u = userMarker.getLatLng();
          // Collect CRITICAL markers near user
          const crit = [];
          markers.forEach(({ marker }) => {
            const mll = marker.getLatLng();
            if(map.distance(mll, u) <= USER_RADIUS_METERS){
              const content = marker.getPopup() && marker.getPopup().getContent();
              if(typeof content === 'string' && content.includes('CRITICAL')) crit.push([mll.lat, mll.lng]);
            }
          });
          if(!crit.length) return;
          // Compute bearing away from centroid
          const cx = crit.reduce((a,c)=>a+c[0],0)/crit.length;
          const cy = crit.reduce((a,c)=>a+c[1],0)/crit.length;
          const dy = toRad(cx - u.lat);
          const dx = toRad(cy - u.lng) * Math.cos(toRad((u.lat+cx)/2));
          const bearingToCrowd = Math.atan2(dx, dy);
          const bearingAwayDeg = (toDeg(bearingToCrowd) + 180) % 360;
          const dest = destPoint(u.lat, u.lng, bearingAwayDeg, AVOID_DISTANCE_METERS);
          if(destMarker){ destMarker.setLatLng([dest.lat, dest.lon]); }
          else { destMarker = L.marker([dest.lat, dest.lon], { title: 'Suggested safer route' }).addTo(map).bindPopup('Suggested safer route'); }
          if(destCircle){ destCircle.setLatLng([dest.lat, dest.lon]); destCircle.setRadius(USER_RADIUS_METERS); }
          else { destCircle = L.circle([dest.lat, dest.lon], { radius: USER_RADIUS_METERS, color: '#1976d2', weight: 2, opacity: 0.9, dashArray: '6 8', fillOpacity: 0, interactive: false }).addTo(map); }
          destMarker.openPopup();
          createRouterWithSafety([ L.latLng(u.lat, u.lng), L.latLng(dest.lat, dest.lon) ]);
          saveDest(dest.lat, dest.lon, 'Suggested safer route');
        };

        document.getElementById('pickOnMap').onclick = () => {
          pickMode = !pickMode;
          document.getElementById('pickOnMap').textContent = pickMode ? 'Picking… (click on map)' : 'Pick on Map';
        };

        map.on('click', (e) => {
          if(!pickMode) return;
          pickMode = false;
          document.getElementById('pickOnMap').textContent = 'Pick on Map';
          placeUser(e.latlng.lat, e.latlng.lng, null);
        });

        document.getElementById('recenter').onclick = () => {
          if(userMarker){
            const ll = userMarker.getLatLng();
            map.flyTo(ll, Math.max(14, map.getZoom()));
            userMarker.openPopup();
          }
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/videos/analyze/seconds")
async def analyze_per_second(filename: str, frame_stride: int = 5):
    items = _list_demo_videos()
    targets = [it for it in items if it["filename"] == filename]
    if not targets:
        raise HTTPException(status_code=404, detail="Requested filename not found or empty")
    path = targets[0]["path"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Failed to open video: {filename}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if YOLO is None:
        raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
    idx = 0
    per_sec: Dict[int, int] = {}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            rects = _yolo_detect(frame, conf=0.35, iou=0.5, imgsz=640)
            count = len(rects) if rects else 0
            sec = int(idx / fps) if fps and fps > 0 else 0
            per_sec[sec] = per_sec.get(sec, 0) + count
            idx += 1
    finally:
        cap.release()
    # convert to sorted list + density per MP (based on frame size from first frame we read)
    # Re-open once to get resolution for density normalization
    cap2 = cv2.VideoCapture(path)
    w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap2.release()
    area_mp = (w * h) / 1_000_000.0 if w and h else 0.0
    series = []
    for s in sorted(per_sec.keys()):
        cnt = per_sec[s]
        dens = (cnt / area_mp) if area_mp else 0.0
        item = {"second": s, "count": cnt, "density_per_mp": round(dens, 3)}
        cfg = LOCATION_CONFIGS.get(filename)
        if cfg and isinstance(cfg.get("area_m2"), (int, float)) and cfg.get("area_m2"):
            area_m2 = float(cfg["area_m2"])
            item["density_per_m2"] = round((cnt / area_m2), 3) if area_m2 > 0 else 0.0
        series.append(item)
    return JSONResponse({"filename": filename, "frame_stride": frame_stride, "series": series, "area_mp": round(area_mp, 3) if area_mp else 0.0})

# Alerts API
@app.get("/alerts/latest")
async def alerts_latest():
    return JSONResponse({"latest": ALERT_STATE.get("latest")})

@app.get("/alerts/config")
async def alerts_get_config():
    return JSONResponse({"config": CONFIG})

@app.post("/alerts/config")
async def alerts_set_config(density_threshold: float = None, panic_threshold: float = None, panic_window: int = None, panic_scale: float = None):
    if density_threshold is not None:
        CONFIG["density_threshold"] = float(density_threshold)
    if panic_threshold is not None:
        CONFIG["panic_threshold"] = float(panic_threshold)
    if panic_window is not None and int(panic_window) > 0:
        CONFIG["panic_window"] = int(panic_window)
    if panic_scale is not None and float(panic_scale) > 0:
        CONFIG["panic_scale"] = float(panic_scale)
    return JSONResponse({"config": CONFIG})

@app.get("/alerts/by_source")
async def alerts_by_source():
    return JSONResponse({"by_source": ALERT_STATE.get("by_source", {})})

@app.get("/locations")
async def get_locations():
    return JSONResponse({"locations": LOCATION_CONFIGS})

@app.post("/locations/set")
async def set_location(
    filename: str = Body(...),
    area_m2: float = Body(...),
    location_id: str | None = Body(None),
    roi: List[List[int]] | None = Body(None)
):
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    if not area_m2 or area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 must be > 0")
    if roi and (not isinstance(roi, list) or len(roi) < 3):
        raise HTTPException(status_code=400, detail="roi must be list of at least 3 [x,y] points")
    LOCATION_CONFIGS[filename] = {
        "location_id": location_id or filename,
        "area_m2": float(area_m2),
        "roi": roi or []
    }
    return JSONResponse({"ok": True, "config": LOCATION_CONFIGS[filename]})

@app.post("/locations/geotag")
async def geotag_location(filename: str = Body(...), lat: float = Body(...), lon: float = Body(...), location_id: Optional[str] = Body(None)):
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise HTTPException(status_code=400, detail="lat/lon must be numbers")
    cfg = LOCATION_CONFIGS.get(filename, {"location_id": filename, "area_m2": 0.0, "roi": []})
    cfg["lat"], cfg["lon"] = float(lat), float(lon)
    if location_id:
        cfg["location_id"] = str(location_id)
    LOCATION_CONFIGS[filename] = cfg
    try:
        _save_locations()
    except Exception:
        pass
    return JSONResponse({"ok": True, "config": cfg})

@app.delete("/locations/delete")
async def delete_location(filename: str):
    if filename in LOCATION_CONFIGS:
        del LOCATION_CONFIGS[filename]
        return JSONResponse({"ok": True})
    raise HTTPException(status_code=404, detail="filename not configured")

@app.get("/congestion")
async def congestion_feed(include_idle: int = 0):
    # By default, return only active stream metrics (markers show only when videos are playing).
    items = [dict(v) for v in GLOBAL_METRICS.values()]
    # If explicitly requested, include geotagged but idle locations as SAFE with count=0
    if include_idle:
        for fname, cfg in LOCATION_CONFIGS.items():
            lat = cfg.get("lat")
            lon = cfg.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and fname not in GLOBAL_METRICS:
                payload = {
                    "ts": int(time.time()),
                    "filename": fname,
                    "location_id": cfg.get("location_id") or fname,
                    "count": 0,
                    "density_mp": 0.0,
                    "severity": "SAFE",
                    "lat": float(lat),
                    "lon": float(lon),
                }
                if isinstance(cfg.get("area_m2"), (int, float)) and cfg.get("area_m2"):
                    payload["density_m2"] = 0.0
                items.append(payload)
    return JSONResponse({"items": items})

if __name__ == "__main__":
    # For local testing: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
