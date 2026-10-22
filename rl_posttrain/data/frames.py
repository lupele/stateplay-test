from pathlib import Path
import io
import cv2
import numpy as np
from PIL import Image
from ..io import atomic_bytes, sha256

def loaded_frame_count(available, requested=101):
    n = min(int(available), int(requested))
    return (n - 1) // 4 * 4 + 1

def frame_indices(n, count=24):
    if n < 1: raise ValueError("Empty video")
    return np.linspace(0, n - 1, count).astype(int).tolist()

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    result = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    if result["frames"] < 1 or result["fps"] <= 0: raise ValueError(f"Unreadable video: {path}")
    return result

def read_frames(path, n=None):
    cap = cv2.VideoCapture(str(path)); frames = []
    while n is None or len(frames) < n:
        ok, frame = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if n is not None and len(frames) != n: raise ValueError("Video ended early")
    return frames

def jpeg_asset(image, path, max_side=832, quality=90):
    image = image.convert("RGB"); scale = min(1, max_side / max(image.size))
    if scale < 1: image = image.resize(tuple(max(1, round(x * scale)) for x in image.size), Image.Resampling.LANCZOS)
    stream = io.BytesIO(); image.save(stream, format="JPEG", quality=quality)
    atomic_bytes(path, stream.getvalue())
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "width": image.width, "height": image.height, "mode": "RGB"}

def extract_assets(video, root, count=24, max_side=832, quality=90):
    info = video_info(video); indices = frame_indices(info["frames"], count); frames = read_frames(video)
    return [{**jpeg_asset(frames[index], root / f"{j:02d}.jpg", max_side, quality), "index": index, "timestamp": index / info["fps"]} for j, index in enumerate(indices)]
