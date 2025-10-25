from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import tempfile
import os

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

if __name__ == "__main__":
    # For local testing: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
