from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn
import cv2
import tempfile
import os
from typing import Dict, List
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

app = FastAPI(title="Crowd Data Ingestion - Video Uploader")

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

def _hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def _hog_detect(frame, hog, hit_threshold: float = 0.0, win_stride=(6, 6), padding=(8, 8), scale: float = 1.05, nms_thresh: float = 0.3):
    rects, weights = hog.detectMultiScale(frame, hitThreshold=hit_threshold, winStride=win_stride, padding=padding, scale=scale)
    if rects is None or len(rects) == 0:
        return []
    bboxes = []
    confs = []
    for (x, y, w, h), wgt in zip(rects, weights if weights is not None else [0.0] * len(rects)):
        bboxes.append([int(x), int(y), int(w), int(h)])
        confs.append(float(wgt))
    idxs = cv2.dnn.NMSBoxes(bboxes, confs, score_threshold=0.0, nms_threshold=nms_thresh)
    if idxs is None or len(idxs) == 0:
        return []
    sel = []
    for i in idxs:
        j = int(i[0]) if hasattr(i, '__len__') else int(i)
        x, y, w, h = bboxes[j]
        sel.append((x, y, w, h))
    return sel

_YOLO_MODEL = None

def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if YOLO is None:
            raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
        _YOLO_MODEL = YOLO('yolov8n.pt')
    return _YOLO_MODEL

def _yolo_detect(frame, conf: float = 0.3, iou: float = 0.5, imgsz: int = 640):
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

def _analyze_video_file(path: str, frame_stride: int = 5, resize_width: int = 960, detector: str = 'hog') -> Dict[str, object]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Failed to open video: {os.path.basename(path)}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    hog = _hog_detector()
    processed = 0
    detections_total = 0
    max_in_frame = 0
    timeline: List[int] = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            if resize_width and frame is not None and frame.shape[1] > resize_width:
                ratio = resize_width / float(frame.shape[1])
                new_h = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (resize_width, new_h))
            if detector == 'yolo':
                rects = _yolo_detect(frame, conf=0.3, iou=0.5, imgsz=640)
            else:
                rects = _hog_detect(frame, hog, hit_threshold=0.0, win_stride=(6, 6), padding=(8, 8), scale=1.05, nms_thresh=0.3)
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
    return {
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

@app.get("/videos/demo")
async def list_demo_videos():
    items = _list_demo_videos()
    if not items:
        return JSONResponse({"videos": [], "message": "No non-empty demo videos found in current directory."})
    return JSONResponse({"videos": items})

@app.post("/videos/analyze/demo")
async def analyze_demo_videos(filename: str = None, frame_stride: int = 5, detector: str = 'hog'):
    items = _list_demo_videos()
    if not items:
        raise HTTPException(status_code=400, detail="No demo videos found")
    targets: List[Dict[str, object]]
    if filename:
        targets = [it for it in items if it["filename"] == filename]
        if not targets:
            raise HTTPException(status_code=404, detail="Requested filename not found or empty")
    else:
        targets = items
    if detector not in {"hog", "yolo"}:
        raise HTTPException(status_code=400, detail="detector must be 'hog' or 'yolo'")
    if detector == 'yolo' and YOLO is None:
        raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
    results: List[Dict[str, object]] = []
    for it in targets:
        try:
            res = _analyze_video_file(it["path"], frame_stride=frame_stride, detector=detector)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed analyzing {it['filename']}: {e}")
        results.append(res)
    return JSONResponse({"results": results})

def _severity_for_count(count: int) -> Dict[str, object]:
    if count <= 3:
        return {"label": "SAFE", "color": (0, 200, 0)}
    if count <= 8:
        return {"label": "MODERATE", "color": (0, 215, 255)}  # yellow-ish (BGR)
    return {"label": "CRITICAL", "color": (0, 0, 255)}

def _stream_overlay(path: str, frame_stride: int = 2, resize_width: int = 960, detector: str = 'hog'):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield b""
        return
    hog = _hog_detector() if detector != 'yolo' else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue
            if resize_width and frame is not None and frame.shape[1] > resize_width:
                ratio = resize_width / float(frame.shape[1])
                new_h = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (resize_width, new_h))
            if detector == 'yolo':
                rects = _yolo_detect(frame, conf=0.35, iou=0.5, imgsz=640)
            else:
                rects = _hog_detect(frame, hog, hit_threshold=0.0, win_stride=(6, 6), padding=(8, 8), scale=1.05, nms_thresh=0.3)
            count = 0
            if rects:
                count = len(rects)
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay banner with severity and count
            h, w = frame.shape[:2]
            sev = _severity_for_count(count)
            banner_h = max(30, h // 16)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), sev["color"], thickness=-1)
            alpha = 0.35
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            # Density per megapixel (people/MP)
            area_mp = (w * h) / 1_000_000.0 if w and h else 0.0
            dens = (count / area_mp) if area_mp else 0.0
            label = f"{sev['label']} | Count: {count} | Dens: {dens:.2f}/MP"
            cv2.putText(frame, label, (10, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # Optional timestamp/second estimate
            if fps and fps > 0:
                sec = int(idx / fps)
                cv2.putText(frame, f"t={sec}s", (w - 90, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
async def stream_demo_video(filename: str, frame_stride: int = 2, detector: str = 'hog'):
    items = _list_demo_videos()
    if not any(it["filename"] == filename for it in items):
        raise HTTPException(status_code=404, detail="Requested filename not found or empty")
    path = next(it["path"] for it in items if it["filename"] == filename)
    if detector not in {"hog", "yolo"}:
        raise HTTPException(status_code=400, detail="detector must be 'hog' or 'yolo'")
    if detector == 'yolo' and YOLO is None:
        raise HTTPException(status_code=500, detail="YOLO not available. Install ultralytics.")
    gen = _stream_overlay(path, frame_stride=frame_stride, detector=detector)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")

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
      </style>
    </head>
    <body>
      <h2>Crowd Detection - Demo Videos</h2>
      <div id=\"list\">Loading...</div>
      <div id=\"controls\">
        <label>Detector: <select id=\"det\"><option value=\"hog\" selected>HOG</option><option value=\"yolo\">YOLOv8</option></select></label>
        <label>Frame stride: <input id=\"stride\" type=\"number\" value=\"2\" min=\"1\" max=\"10\" /></label>
        <button id=\"start\">Start Selected (max 3)</button>
        <button id=\"stop\">Stop All</button>
      </div>
      <div class=\"grid\">
        <div class=\"panel\"><h4 id=\"t1\">Panel 1</h4><img id=\"p1\" src=\"\" alt=\"panel1\" /></div>
        <div class=\"panel\"><h4 id=\"t2\">Panel 2</h4><img id=\"p2\" src=\"\" alt=\"panel2\" /></div>
        <div class=\"panel\"><h4 id=\"t3\">Panel 3</h4><img id=\"p3\" src=\"\" alt=\"panel3\" /></div>
      </div>
      <script>
        let videos = [];
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
            row.appendChild(cb); row.appendChild(lab);
            list.appendChild(row);
          });
        }

        function stopAll() {
          ['p1','p2','p3'].forEach(id => { document.getElementById(id).src = ''; });
          ['t1','t2','t3'].forEach(id => { document.getElementById(id).textContent = 'Panel'; });
        }

        function startSelected() {
          stopAll();
          const det = document.getElementById('det').value;
          const s = document.getElementById('stride').value || 2;
          const checked = Array.from(document.querySelectorAll('#list input[type=checkbox]:checked')).slice(0,3);
          const panels = [ ['p1','t1'], ['p2','t2'], ['p3','t3'] ];
          checked.forEach((c, idx) => {
            const [imgId, tId] = panels[idx];
            document.getElementById(imgId).src = '/videos/stream?filename=' + encodeURIComponent(c.value) + '&frame_stride=' + s + '&detector=' + det;
            document.getElementById(tId).textContent = c.value;
          });
        }

        document.getElementById('start').onclick = startSelected;
        document.getElementById('stop').onclick = stopAll;
        load();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/videos/analyze/seconds")
async def analyze_per_second(filename: str, frame_stride: int = 5, detector: str = 'hog'):
    items = _list_demo_videos()
    targets = [it for it in items if it["filename"] == filename]
    if not targets:
        raise HTTPException(status_code=404, detail="Requested filename not found or empty")
    path = targets[0]["path"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Failed to open video: {filename}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if detector not in {"hog", "yolo"}:
        raise HTTPException(status_code=400, detail="detector must be 'hog' or 'yolo'")
    hog = _hog_detector() if detector != 'yolo' else None
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
            if detector == 'yolo':
                rects = _yolo_detect(frame, conf=0.35, iou=0.5, imgsz=640)
            else:
                rects = _hog_detect(frame, hog, win_stride=(8, 8), padding=(8, 8), scale=1.05)
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
        series.append({"second": s, "count": cnt, "density_per_mp": round(dens, 3)})
    return JSONResponse({"filename": filename, "frame_stride": frame_stride, "series": series, "area_mp": round(area_mp, 3) if area_mp else 0.0})

if __name__ == "__main__":
    # For local testing: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
