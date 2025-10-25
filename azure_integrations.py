"""
Azure integrations for Blob Storage and Service Bus.

Usage (examples):

from azure_integrations import (
    blob_enabled, blob_upload_bytes, blob_download_to_temp,
    blob_list_videos, resolve_path_for_processing,
    sb_enabled, sb_send_json
)

# Upload bytes
if blob_enabled():
    blob_upload_bytes("sample.mp4", b"...", overwrite=True)

# List videos
videos = blob_list_videos()

# Prepare path for OpenCV (downloads blob:// to a temp file)
proc_path, cleanup = resolve_path_for_processing("blob://sample.mp4")
try:
    # use proc_path with cv2.VideoCapture
    pass
finally:
    cleanup()  # ensure temp is deleted

# Send a JSON message to Service Bus
if sb_enabled():
    sb_send_json({"event": "crowd_metrics", "count": 12})
"""
from __future__ import annotations

import os
import json
import tempfile
from typing import Dict, List, Tuple, Callable, Optional
from dotenv import load_dotenv

try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None  # type: ignore

try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage  # type: ignore
except Exception:
    ServiceBusClient = None  # type: ignore
    ServiceBusMessage = None  # type: ignore

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Load environment variables from a .env file if present
load_dotenv()
AZ_BLOB_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZ_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "videos")
AZ_SB_CONN = os.getenv("AZURE_SERVICEBUS_CONNECTION_STRING", "")
AZ_SB_QUEUE = os.getenv("AZURE_SERVICEBUS_QUEUE", "crowd-metrics")

_blob_client = (
    BlobServiceClient.from_connection_string(AZ_BLOB_CONN) if (BlobServiceClient and AZ_BLOB_CONN) else None
)
_sb_client = (
    ServiceBusClient.from_connection_string(AZ_SB_CONN) if (ServiceBusClient and AZ_SB_CONN) else None
)

# ----------------------------------------------------------------------------
# Blob helpers
# ----------------------------------------------------------------------------

def blob_enabled() -> bool:
    return _blob_client is not None


def _get_container_client():
    if not blob_enabled():
        raise RuntimeError("Blob storage not configured")
    bc = _blob_client.get_container_client(AZ_BLOB_CONTAINER)
    try:
        bc.create_container()
    except Exception:
        pass
    return bc


def blob_upload_bytes(name: str, data: bytes, overwrite: bool = True) -> None:
    """Upload a bytes payload to the configured container."""
    bc = _get_container_client()
    bc.upload_blob(name, data, overwrite=overwrite)


def blob_download_to_temp(name: str) -> str:
    """Download a blob to a temporary file and return its path."""
    bc = _get_container_client()
    b = bc.get_blob_client(name)
    if not b.exists():
        raise FileNotFoundError(f"Blob not found: {name}")
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(name)[1].lower())
    os.close(fd)
    with open(tmp_path, "wb") as f:
        stream = b.download_blob()
        stream.readinto(f)
    return tmp_path


def blob_list_videos() -> List[Dict[str, object]]:
    """List video-like blobs in the container."""
    if not blob_enabled():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    bc = _get_container_client()
    out: List[Dict[str, object]] = []
    for b in bc.list_blobs():
        ext = os.path.splitext(b.name)[1].lower()
        if ext in exts and (getattr(b, "size", 0) or 0) > 0:
            out.append({
                "filename": b.name,
                "path": f"blob://{b.name}",
                "size_bytes": getattr(b, "size", None)
            })
    return out


def resolve_path_for_processing(item_path: str) -> Tuple[str, Callable[[], None]]:
    """
    Resolve a path that may be local or a blob URL (blob://<name>) for OpenCV.
    Returns (real_path, cleanup_fn).
    """
    if item_path.startswith("blob://"):
        name = item_path[len("blob://"):]
        tmp_path = blob_download_to_temp(name)
        def _cleanup():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return tmp_path, _cleanup
    else:
        return item_path, (lambda: None)

# ----------------------------------------------------------------------------
# Service Bus helpers
# ----------------------------------------------------------------------------

def sb_enabled() -> bool:
    return _sb_client is not None and bool(AZ_SB_QUEUE)


def sb_send_json(payload: Dict[str, object]) -> None:
    if not sb_enabled():
        return
    if ServiceBusMessage is None:
        return
    # Use context managers per Azure SDK best practice
    with _sb_client:
        sender = _sb_client.get_queue_sender(queue_name=AZ_SB_QUEUE)
        with sender:
            msg = ServiceBusMessage(json.dumps(payload))
            sender.send_messages(msg)


__all__ = [
    "blob_enabled",
    "blob_upload_bytes",
    "blob_download_to_temp",
    "blob_list_videos",
    "resolve_path_for_processing",
    "sb_enabled",
    "sb_send_json",
]
