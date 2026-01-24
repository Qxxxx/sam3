"""
SAM 3 FastAPI Service - Simple & Clean Version

Thread-safe service with request queue for GPU concurrency control.
Supports both local video paths and video file uploads.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue, Full
from threading import Thread, Event
from typing import Optional, List

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent directory for uploaded videos
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "sam3_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")


# ==================== Models ====================

class StartSessionRequest(BaseModel):
    resource_path: Optional[str] = None
    video_id: Optional[str] = None
    session_id: Optional[str] = None


class UploadVideoResponse(BaseModel):
    video_id: str
    resource_path: str
    file_size_mb: float


class VideoInfo(BaseModel):
    video_id: str
    resource_path: str
    file_size_mb: float
    filename: str


class VideoListResponse(BaseModel):
    videos: List[VideoInfo]
    total: int


class StartSessionResponse(BaseModel):
    session_id: str
    resource_path: str
    uploaded: bool  # True if video was uploaded, False if using local path


class AddPromptRequest(BaseModel):
    session_id: str
    frame_index: int
    text: Optional[str] = None
    points: Optional[List[List[float]]] = None
    bounding_boxes: Optional[List[List[float]]] = None


class PropagateRequest(BaseModel):
    session_id: str
    direction: str = "both"


# ==================== Worker Thread ====================

class SAM3Worker:
    """Single worker thread for processing all SAM3 requests"""

    def __init__(self, queue_size: int = 100):
        self.queue = Queue(maxsize=queue_size)
        self.ready = Event()
        self.shutdown_flag = Event()
        self.video_predictor = None
        self.image_model = None
        self.image_processor = None

        # Track uploaded files per session for cleanup
        self.session_temp_files = {}  # session_id -> temp_file_path
        # Track uploaded videos by video_id
        self.uploaded_videos = {}  # video_id -> resource_path

    def start(self):
        """Start worker thread and load models"""
        Thread(target=self._worker_loop, daemon=True).start()

        if not self.ready.wait(timeout=120):
            raise RuntimeError("Model loading timeout")

        # Load existing videos from upload directory
        self._load_existing_videos()

        logger.info("✅ Worker ready")

    def _load_existing_videos(self):
        """Load existing videos from upload directory on startup"""
        if not os.path.exists(UPLOAD_DIR):
            return

        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                # Extract video_id from filename (format: {video_id}.ext)
                video_id = os.path.splitext(filename)[0]
                self.uploaded_videos[video_id] = file_path
                logger.info(f"📂 Loaded existing video: {video_id} -> {file_path}")

        if self.uploaded_videos:
            logger.info(f"📂 Loaded {len(self.uploaded_videos)} existing videos from {UPLOAD_DIR}")

    def _worker_loop(self):
        """Main worker loop - processes requests sequentially"""
        try:
            self._load_models()
            self.ready.set()

            while not self.shutdown_flag.is_set():
                try:
                    item = self.queue.get(timeout=1.0)
                    self._process(item)
                except:
                    pass

        except Exception as e:
            logger.error(f"Worker error: {e}")
            self.ready.set()

    def _load_models(self):
        """Load SAM3 models"""
        logger.info("🚀 Loading models...")

        from sam3.model_builder import build_sam3_video_predictor, build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.video_predictor = build_sam3_video_predictor(gpus_to_use=[0])
        self.image_model = build_sam3_image_model(device="cuda", eval_mode=True)
        self.image_processor = Sam3Processor(self.image_model)

    def _process(self, item):
        """Process a single request"""
        request_id, request_type, data, future = item

        try:
            if request_type == "video":
                result = self.video_predictor.handle_request(data)

                # Track temp file for cleanup if this is a start_session with uploaded file
                if data.get("type") == "start_session" and data.get("_temp_file"):
                    session_id = result["session_id"]
                    self.session_temp_files[session_id] = data["_temp_file"]
                    logger.info(f"📌 Tracked temp file for session {session_id}")

                # Cleanup temp file if closing session
                elif data.get("type") == "close_session":
                    session_id = data["session_id"]
                    self._cleanup_session(session_id)

            elif request_type == "video_stream":
                results = []
                for frame in self.video_predictor.handle_stream_request(data):
                    results.append(frame)
                result = {"frames": results}

            elif request_type == "image":
                result = self._segment_image(data)

            else:
                raise ValueError(f"Unknown type: {request_type}")

            asyncio.get_event_loop().call_soon_threadsafe(future.set_result, result)

        except Exception as e:
            logger.error(f"Process error: {e}")
            asyncio.get_event_loop().call_soon_threadsafe(future.set_exception, e)

    def _segment_image(self, data):
        """Process image segmentation"""
        image, prompt = data["image"], data["prompt"]

        state = self.image_processor.set_image(image)
        output = self.image_processor.set_text_prompt(state=state, prompt=prompt)

        return {
            "image_width": image.size[0],
            "image_height": image.size[1],
            "boxes": output["boxes"].cpu().tolist(),
            "scores": output["scores"].cpu().tolist(),
            "num_objects": len(output["scores"]),
            "prompt": prompt
        }

    def _cleanup_session(self, session_id):
        """Clean up temporary files for a session"""
        temp_file = self.session_temp_files.pop(session_id, None)
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"🗑️  Deleted temp file for session {session_id}: {temp_file}")
                # Also remove from uploaded_videos if it exists there
                video_id_to_remove = None
                for vid_id, path in self.uploaded_videos.items():
                    if path == temp_file:
                        video_id_to_remove = vid_id
                        break
                if video_id_to_remove:
                    self.uploaded_videos.pop(video_id_to_remove, None)
            except Exception as e:
                logger.error(f"Failed to delete temp file {temp_file}: {e}")

    async def submit(self, request_type, data, timeout=30.0):
        """Submit request and wait for result"""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()

        try:
            self.queue.put_nowait((request_id, request_type, data, future))
        except Full:
            raise HTTPException(503, "Queue full, try again later")

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Timeout after {timeout}s")

    def stats(self):
        """Get worker stats"""
        return {
            "queue_size": self.queue.qsize(),
            "ready": self.ready.is_set(),
            "active_sessions": len(self.video_predictor._ALL_INFERENCE_STATES) if self.video_predictor else 0,
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }


# ==================== FastAPI App ====================

worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global worker

    # Startup
    logger.info("Starting SAM3 service...")
    worker = SAM3Worker(queue_size=100)
    worker.start()
    logger.info("SAM3 service started")

    yield

    # Shutdown
    logger.info("Shutting down SAM3 service...")
    if worker:
        worker.shutdown_flag.set()

    logger.info(f"📁 Upload directory preserved: {UPLOAD_DIR}")


app = FastAPI(title="SAM 3 Service", version="1.0.0", lifespan=lifespan)

# Configure CORS to allow frontend requests from any origin (LAN-accessible)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for LAN access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ==================== Endpoints ====================

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "SAM 3 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    if not worker or not worker.ready.is_set():
        raise HTTPException(503, "Service not ready")

    return worker.stats()


@app.post("/api/v1/image/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Segment objects in an image"""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = await worker.submit("image", {"image": image, "prompt": prompt})
    return result


@app.post("/api/v1/video/session/start")
async def start_session(req: StartSessionRequest):
    """
    Start video session and load frames (this is when OpenCV splits the video).

    You can provide either:
    - video_id: from a previous POST /api/v1/video/upload
    - resource_path: direct path to video file on server

    This endpoint loads ALL video frames into memory, which can take time
    for large videos.
    """
    # Resolve video_id to resource_path if provided
    if req.video_id:
        resource_path = worker.uploaded_videos.get(req.video_id)
        if not resource_path:
            raise HTTPException(404, f"Video ID not found: {req.video_id}")
        is_uploaded = True
    elif req.resource_path:
        resource_path = req.resource_path
        is_uploaded = False
    else:
        raise HTTPException(400, "Must provide either video_id or resource_path")

    # Start session (this loads frames)
    result = await worker.submit("video", {
        "type": "start_session",
        "resource_path": resource_path,
        "session_id": req.session_id,
        "_temp_file": resource_path if is_uploaded else None
    }, timeout=60.0)

    return StartSessionResponse(
        session_id=result["session_id"],
        resource_path=resource_path,
        uploaded=is_uploaded
    )


@app.post("/api/v1/video/upload", response_model=UploadVideoResponse)
async def upload_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)")
):
    """
    Upload a video file WITHOUT starting a session.

    This only saves the video to the server. To process it, call
    POST /api/v1/video/session/start with the returned video_id.

    Supported formats: MP4, AVI, MOV, MKV, etc.
    """
    # Validate file extension
    filename = file.filename or "video.mp4"
    file_ext = Path(filename).suffix.lower()
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    if file_ext not in allowed_extensions:
        raise HTTPException(
            400,
            f"Unsupported video format: {file_ext}. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{video_id}{file_ext}")

    # Save uploaded file
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"📤 Uploaded video: {filename} ({file_size_mb:.1f} MB) -> {temp_path}")

        # Register video in worker
        worker.uploaded_videos[video_id] = temp_path

        return UploadVideoResponse(
            video_id=video_id,
            resource_path=temp_path,
            file_size_mb=round(file_size_mb, 2)
        )

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"Failed to save video: {str(e)}")


@app.post("/api/v1/video/session/upload", response_model=StartSessionResponse)
async def upload_and_start_session(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)"),
    session_id: Optional[str] = Form(None, description="Optional custom session ID")
):
    """
    Upload a video file and start a tracking session (combined operation).

    The video is saved temporarily on the server and will be automatically
    deleted when the session is closed.

    Supported formats: MP4, AVI, MOV, MKV, etc.

    For more control, use POST /api/v1/video/upload followed by
    POST /api/v1/video/session/start separately.
    """
    # Validate file extension
    filename = file.filename or "video.mp4"
    file_ext = Path(filename).suffix.lower()
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    if file_ext not in allowed_extensions:
        raise HTTPException(
            400,
            f"Unsupported video format: {file_ext}. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded file
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"📤 Uploaded video: {filename} ({file_size_mb:.1f} MB) -> {temp_path}")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"Failed to save video: {str(e)}")

    # Start session with the uploaded video
    try:
        result = await worker.submit("video", {
            "type": "start_session",
            "resource_path": temp_path,
            "session_id": session_id,
            "_temp_file": temp_path  # Mark as temporary for cleanup
        }, timeout=60.0)

        return StartSessionResponse(
            session_id=result["session_id"],
            resource_path=temp_path,
            uploaded=True
        )

    except Exception as e:
        # Clean up file if session start failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


@app.post("/api/v1/video/prompt/add")
async def add_prompt(req: AddPromptRequest):
    """Add prompt to video frame"""
    result = await worker.submit("video", {
        "type": "add_prompt",
        "session_id": req.session_id,
        "frame_index": req.frame_index,
        "text": req.text,
        "points": req.points,
        "bounding_boxes": req.bounding_boxes
    })

    return result


@app.post("/api/v1/video/propagate")
async def propagate(req: PropagateRequest):
    """Propagate tracking across video"""
    result = await worker.submit("video_stream", {
        "type": "propagate_in_video",
        "session_id": req.session_id,
        "propagation_direction": req.direction
    }, timeout=300.0)

    return result


@app.get("/api/v1/video/list", response_model=VideoListResponse)
async def list_videos():
    """List all uploaded videos"""
    videos = []
    for video_id, resource_path in worker.uploaded_videos.items():
        if os.path.exists(resource_path):
            file_size = os.path.getsize(resource_path)
            file_size_mb = file_size / (1024 * 1024)
            filename = os.path.basename(resource_path)
            videos.append(VideoInfo(
                video_id=video_id,
                resource_path=resource_path,
                file_size_mb=round(file_size_mb, 2),
                filename=filename
            ))

    return VideoListResponse(videos=videos, total=len(videos))


@app.delete("/api/v1/video/upload/{video_id}")
async def delete_uploaded_video(video_id: str):
    """Delete an uploaded video file that hasn't been used in a session yet"""
    resource_path = worker.uploaded_videos.pop(video_id, None)
    if not resource_path:
        raise HTTPException(404, f"Video ID not found: {video_id}")

    if os.path.exists(resource_path):
        try:
            os.remove(resource_path)
            logger.info(f"🗑️  Deleted uploaded video {video_id}: {resource_path}")
            return {"is_success": True, "video_id": video_id}
        except Exception as e:
            logger.error(f"Failed to delete video {resource_path}: {e}")
            raise HTTPException(500, f"Failed to delete video: {str(e)}")
    else:
        return {"is_success": True, "video_id": video_id, "note": "File already deleted"}


@app.delete("/api/v1/video/session/{session_id}")
async def close_session(session_id: str):
    """Close video session"""
    result = await worker.submit("video", {
        "type": "close_session",
        "session_id": session_id
    })

    return result


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
