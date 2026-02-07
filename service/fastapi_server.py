"""
SAM 3 FastAPI Service (Unified Session Core)

This service exposes a unified API over Sam3VideoPredictor for both images and videos.
All inference paths are session-based, and one-shot segmentation is implemented as
start_session -> semantic_prompt -> close_session.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sam3.train.masks_ops import rle_encode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Constants ====================

BASE_STORAGE_DIR = os.path.join(tempfile.gettempdir(), "sam3_api")
RESOURCE_DIR = os.path.join(BASE_STORAGE_DIR, "resources")
ONESHOT_DIR = os.path.join(BASE_STORAGE_DIR, "oneshot")

os.makedirs(RESOURCE_DIR, exist_ok=True)
os.makedirs(ONESHOT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

QUEUE_SIZE = 100
START_TIMEOUT_SEC = 120.0
PROMPT_TIMEOUT_SEC = 60.0
PROPAGATE_TIMEOUT_SEC = 300.0
CLOSE_TIMEOUT_SEC = 30.0

SESSION_TTL_SEC = 900
MAX_ACTIVE_SESSIONS = 64


# ==================== Errors ====================

class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ==================== Models ====================

class ResourceInfo(BaseModel):
    resource_id: str
    media_type: Literal["image", "video"]
    resource_path: str
    size_bytes: int
    uploaded_at: float
    ref_count: int


class SessionInfo(BaseModel):
    session_id: str
    resource_type: Literal["image", "video"]
    resource_path: str
    resource_id: Optional[str] = None
    num_frames: int
    orig_width: int
    orig_height: int
    created_at: float
    last_access_at: float
    active: bool


class StartSessionRequest(BaseModel):
    resource_id: Optional[str] = None
    resource_path: Optional[str] = None
    session_id: Optional[str] = None


class SemanticPromptRequest(BaseModel):
    frame_index: Optional[int] = 0
    text: Optional[str] = None
    boxes_xywh_norm: Optional[List[List[float]]] = None
    box_labels: Optional[List[bool]] = None


class PointsPromptRequest(BaseModel):
    frame_index: int
    obj_id: int
    points_xy_norm: List[List[float]]
    point_labels: List[int]


class PropagateRequest(BaseModel):
    direction: Literal["forward", "backward", "both"] = "both"
    start_frame_index: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None


class ObjectResponse(BaseModel):
    obj_id: int
    score: float
    bbox_xywh_norm: List[float]
    mask_rle: Dict[str, Any]


class FrameResponse(BaseModel):
    frame_index: int
    objects: List[ObjectResponse]
    frame_stats: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    session_id: str
    frame_index: int
    objects: List[ObjectResponse]
    frame_stats: Optional[Dict[str, Any]] = None


class PropagateResponse(BaseModel):
    session_id: str
    total_frames: int
    frames: List[FrameResponse]


@dataclass
class ResourceRecord:
    resource_id: str
    media_type: Literal["image", "video"]
    resource_path: str
    size_bytes: int
    uploaded_at: float
    ref_count: int = 0
    owned_by_service: bool = False


@dataclass
class SessionRecord:
    session_id: str
    resource_type: Literal["image", "video"]
    resource_path: str
    resource_id: Optional[str]
    num_frames: int
    orig_width: int
    orig_height: int
    created_at: float
    last_access_at: float
    temporary_file: Optional[str] = None


# ==================== Helpers ====================

def _media_type_from_path(path: str) -> Optional[Literal["image", "video"]]:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _resource_to_response(record: ResourceRecord) -> Dict[str, Any]:
    return {
        "resource_id": record.resource_id,
        "media_type": record.media_type,
        "resource_path": record.resource_path,
        "size_bytes": record.size_bytes,
        "uploaded_at": record.uploaded_at,
        "ref_count": record.ref_count,
    }


def _session_to_response(record: SessionRecord) -> Dict[str, Any]:
    return {
        "session_id": record.session_id,
        "resource_type": record.resource_type,
        "resource_path": record.resource_path,
        "resource_id": record.resource_id,
        "num_frames": record.num_frames,
        "orig_width": record.orig_width,
        "orig_height": record.orig_height,
        "created_at": record.created_at,
        "last_access_at": record.last_access_at,
        "active": True,
    }


def _validate_boxes_xywh_norm(boxes: List[List[float]]) -> None:
    for i, box in enumerate(boxes):
        if len(box) != 4:
            raise ServiceError(400, f"boxes_xywh_norm[{i}] must have exactly 4 values")
        for v in box:
            if not isinstance(v, (int, float)):
                raise ServiceError(400, "boxes_xywh_norm must contain numeric values")
            if v < 0.0 or v > 1.0:
                raise ServiceError(400, "boxes_xywh_norm values must be in [0, 1]")


def _validate_points_xy_norm(points: List[List[float]]) -> None:
    for i, point in enumerate(points):
        if len(point) != 2:
            raise ServiceError(400, f"points_xy_norm[{i}] must have exactly 2 values")
        for v in point:
            if not isinstance(v, (int, float)):
                raise ServiceError(400, "points_xy_norm must contain numeric values")
            if v < 0.0 or v > 1.0:
                raise ServiceError(400, "points_xy_norm values must be in [0, 1]")


def _json_form_field(value: Optional[str], field_name: str) -> Optional[Any]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid JSON in form field '{field_name}'") from exc


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


# ==================== Worker Thread ====================

class SAM3Worker:
    """Single worker thread for processing all SAM3 requests."""

    def __init__(self, queue_size: int = QUEUE_SIZE):
        self.queue: Queue = Queue(maxsize=queue_size)
        self.ready = Event()
        self.shutdown_flag = Event()

        self.video_predictor = None
        self.resource_store: Dict[str, ResourceRecord] = {}
        self.session_store: Dict[str, SessionRecord] = {}

    def start(self) -> None:
        Thread(target=self._worker_loop, daemon=True).start()
        if not self.ready.wait(timeout=180):
            raise RuntimeError("Model loading timeout")

    def _worker_loop(self) -> None:
        try:
            self._load_models()
            self._load_existing_resources()
            self.ready.set()
            logger.info("Worker is ready")

            while not self.shutdown_flag.is_set():
                try:
                    item = self.queue.get(timeout=1.0)
                except Empty:
                    continue

                self._cleanup_expired_sessions()
                self._process(item)

        except Exception as exc:
            logger.exception("Worker startup failure: %s", exc)
            self.ready.set()

    def _load_models(self) -> None:
        logger.info("Loading SAM3 predictor...")
        from sam3.model_builder import build_sam3_video_predictor

        self.video_predictor = build_sam3_video_predictor(gpus_to_use=[0])

    def _load_existing_resources(self) -> None:
        if not os.path.exists(RESOURCE_DIR):
            return

        for filename in os.listdir(RESOURCE_DIR):
            path = os.path.join(RESOURCE_DIR, filename)
            if not os.path.isfile(path):
                continue
            media_type = _media_type_from_path(path)
            if media_type is None:
                continue

            resource_id = Path(filename).stem
            stat = os.stat(path)
            self.resource_store[resource_id] = ResourceRecord(
                resource_id=resource_id,
                media_type=media_type,
                resource_path=path,
                size_bytes=int(stat.st_size),
                uploaded_at=float(stat.st_mtime),
                ref_count=0,
                owned_by_service=False,
            )

        if self.resource_store:
            logger.info("Loaded %d resources from disk", len(self.resource_store))

    def _process(self, item: Tuple[str, str, Dict[str, Any], asyncio.Future, asyncio.AbstractEventLoop]) -> None:
        request_id, op, data, future, loop = item
        try:
            result = self._dispatch(op, data)
        except Exception as exc:
            logger.exception("Worker process error (%s): %s", request_id, exc)
            self._set_future_exception(loop, future, exc)
        else:
            self._set_future_result(loop, future, result)

    @staticmethod
    def _set_future_result(
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future,
        result: Any,
    ) -> None:
        if future.done():
            return
        loop.call_soon_threadsafe(future.set_result, result)

    @staticmethod
    def _set_future_exception(
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future,
        exc: Exception,
    ) -> None:
        if future.done():
            return
        loop.call_soon_threadsafe(future.set_exception, exc)

    def _dispatch(self, op: str, data: Dict[str, Any]) -> Any:
        if op == "register_resource":
            return self._register_resource(data)
        if op == "list_resources":
            return self._list_resources()
        if op == "delete_resource":
            return self._delete_resource(data)

        if op == "start_session":
            return self._start_session(data)
        if op == "get_session":
            return self._get_session(data)
        if op == "list_sessions":
            return self._list_sessions()
        if op == "close_session":
            return self._close_session(data)
        if op == "reset_session":
            return self._reset_session(data)
        if op == "remove_object":
            return self._remove_object(data)
        if op == "prompt_semantic":
            return self._prompt_semantic(data)
        if op == "prompt_points":
            return self._prompt_points(data)
        if op == "propagate":
            return self._propagate(data)

        raise ServiceError(400, f"Unknown operation: {op}")

    def _predictor_handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.video_predictor.handle_request(request)
        except RuntimeError as exc:
            message = str(exc)
            if "Cannot find session" in message:
                raise ServiceError(404, message) from exc
            raise ServiceError(500, message) from exc
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        except Exception as exc:
            raise ServiceError(500, str(exc)) from exc

    def _predictor_handle_stream(self, request: Dict[str, Any]):
        try:
            yield from self.video_predictor.handle_stream_request(request)
        except RuntimeError as exc:
            message = str(exc)
            if "Cannot find session" in message:
                raise ServiceError(404, message) from exc
            raise ServiceError(500, message) from exc
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        except Exception as exc:
            raise ServiceError(500, str(exc)) from exc

    def _register_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        resource_id = data["resource_id"]
        if resource_id in self.resource_store:
            raise ServiceError(409, f"Resource already exists: {resource_id}")

        media_type = data["media_type"]
        if media_type not in ("image", "video"):
            raise ServiceError(400, f"Invalid media_type: {media_type}")

        record = ResourceRecord(
            resource_id=resource_id,
            media_type=media_type,
            resource_path=data["resource_path"],
            size_bytes=int(data["size_bytes"]),
            uploaded_at=float(data.get("uploaded_at", time.time())),
            ref_count=0,
            owned_by_service=bool(data.get("owned_by_service", False)),
        )
        self.resource_store[resource_id] = record
        return _resource_to_response(record)

    def _list_resources(self) -> Dict[str, Any]:
        resources = [_resource_to_response(rec) for rec in self.resource_store.values()]
        resources.sort(key=lambda item: item["uploaded_at"], reverse=True)
        return {"resources": resources, "total": len(resources)}

    def _delete_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        resource_id = data["resource_id"]
        record = self.resource_store.get(resource_id)
        if record is None:
            raise ServiceError(404, f"Resource not found: {resource_id}")
        if record.ref_count > 0:
            raise ServiceError(409, f"Resource {resource_id} is in use by {record.ref_count} session(s)")

        if os.path.exists(record.resource_path):
            try:
                os.remove(record.resource_path)
            except OSError as exc:
                raise ServiceError(500, f"Failed to delete resource file: {exc}") from exc

        self.resource_store.pop(resource_id, None)
        return {"is_success": True, "resource_id": resource_id}

    def _start_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        resource_id = data.get("resource_id")
        resource_path = data.get("resource_path")
        custom_session_id = data.get("session_id")
        temporary_file = data.get("temporary_file")

        if bool(resource_id) == bool(resource_path):
            raise ServiceError(400, "Provide exactly one of resource_id or resource_path")

        if len(self.session_store) >= MAX_ACTIVE_SESSIONS:
            raise ServiceError(409, f"Maximum active sessions reached ({MAX_ACTIVE_SESSIONS})")

        resource_record = None
        if resource_id is not None:
            resource_record = self.resource_store.get(resource_id)
            if resource_record is None:
                raise ServiceError(404, f"Resource not found: {resource_id}")
            resource_path = resource_record.resource_path
        else:
            if not os.path.exists(resource_path):
                raise ServiceError(404, f"resource_path not found: {resource_path}")

        start_result = self._predictor_handle_request(
            {
                "type": "start_session",
                "resource_path": resource_path,
                "session_id": custom_session_id,
            }
        )
        session_id = start_result["session_id"]

        predictor_info = self.video_predictor.get_session_info(session_id)
        now = time.time()

        if resource_record is not None:
            resource_record.ref_count += 1

        session_record = SessionRecord(
            session_id=session_id,
            resource_type=predictor_info["resource_type"],
            resource_path=resource_path,
            resource_id=resource_id,
            num_frames=int(predictor_info["num_frames"]),
            orig_width=int(predictor_info["orig_width"]),
            orig_height=int(predictor_info["orig_height"]),
            created_at=now,
            last_access_at=now,
            temporary_file=temporary_file,
        )
        self.session_store[session_id] = session_record
        return _session_to_response(session_record)

    def _get_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        record = self.session_store.get(session_id)
        if record is None:
            raise ServiceError(404, f"Session not found: {session_id}")
        return _session_to_response(record)

    def _list_sessions(self) -> Dict[str, Any]:
        sessions = [_session_to_response(rec) for rec in self.session_store.values()]
        sessions.sort(key=lambda item: item["created_at"], reverse=True)
        return {"sessions": sessions, "total": len(sessions)}

    def _close_session_internal(self, session_id: str) -> None:
        self._predictor_handle_request({"type": "close_session", "session_id": session_id})

        record = self.session_store.pop(session_id, None)
        if record is None:
            return

        if record.resource_id is not None:
            resource_record = self.resource_store.get(record.resource_id)
            if resource_record is not None and resource_record.ref_count > 0:
                resource_record.ref_count -= 1

        if record.temporary_file and os.path.exists(record.temporary_file):
            try:
                os.remove(record.temporary_file)
            except OSError:
                logger.warning("Failed to delete temporary file: %s", record.temporary_file)

    def _close_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        self._close_session_internal(session_id)
        return {"is_success": True, "session_id": session_id}

    def _reset_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        if session_id not in self.session_store:
            raise ServiceError(404, f"Session not found: {session_id}")

        self._predictor_handle_request({"type": "reset_session", "session_id": session_id})
        self.session_store[session_id].last_access_at = time.time()
        return {"is_success": True, "session_id": session_id}

    def _remove_object(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        obj_id = int(data["obj_id"])
        if session_id not in self.session_store:
            raise ServiceError(404, f"Session not found: {session_id}")

        self._predictor_handle_request(
            {"type": "remove_object", "session_id": session_id, "obj_id": obj_id}
        )
        self.session_store[session_id].last_access_at = time.time()
        return {"is_success": True, "session_id": session_id, "obj_id": obj_id}

    def _prompt_semantic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        if session_id not in self.session_store:
            raise ServiceError(404, f"Session not found: {session_id}")

        frame_index = int(data.get("frame_index", 0))
        text = data.get("text")
        boxes = data.get("boxes_xywh_norm")
        box_labels = data.get("box_labels")

        if text is None and not boxes:
            raise ServiceError(400, "At least one of text or boxes_xywh_norm must be provided")

        if boxes is not None:
            if not isinstance(boxes, list) or len(boxes) == 0:
                raise ServiceError(400, "boxes_xywh_norm must be a non-empty list")
            _validate_boxes_xywh_norm(boxes)

            if box_labels is None:
                box_labels = [True] * len(boxes)
            if len(box_labels) != len(boxes):
                raise ServiceError(400, "box_labels length must match boxes_xywh_norm length")
            normalized_labels = [1 if bool(v) else 0 for v in box_labels]
        else:
            normalized_labels = None

        result = self._predictor_handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": text,
                "bounding_boxes": boxes,
                "bounding_box_labels": normalized_labels,
            }
        )

        self.session_store[session_id].last_access_at = time.time()
        outputs = result["outputs"]
        response: Dict[str, Any] = {
            "session_id": session_id,
            "frame_index": int(result["frame_index"]),
            "objects": self._format_objects(outputs),
        }
        frame_stats = outputs.get("frame_stats")
        if frame_stats is not None:
            response["frame_stats"] = frame_stats
        return response

    def _prompt_points(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        if session_id not in self.session_store:
            raise ServiceError(404, f"Session not found: {session_id}")

        frame_index = int(data["frame_index"])
        obj_id = int(data["obj_id"])
        points = data.get("points_xy_norm")
        point_labels = data.get("point_labels")

        if not isinstance(points, list) or len(points) == 0:
            raise ServiceError(400, "points_xy_norm must be a non-empty list")
        if not isinstance(point_labels, list) or len(point_labels) != len(points):
            raise ServiceError(400, "point_labels length must match points_xy_norm length")

        _validate_points_xy_norm(points)

        normalized_labels: List[int] = []
        for i, label in enumerate(point_labels):
            if int(label) not in (0, 1):
                raise ServiceError(400, f"point_labels[{i}] must be 0 or 1")
            normalized_labels.append(int(label))

        result = self._predictor_handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "obj_id": obj_id,
                "points": points,
                "point_labels": normalized_labels,
            }
        )

        self.session_store[session_id].last_access_at = time.time()
        outputs = result["outputs"]
        response: Dict[str, Any] = {
            "session_id": session_id,
            "frame_index": int(result["frame_index"]),
            "objects": self._format_objects(outputs),
        }
        frame_stats = outputs.get("frame_stats")
        if frame_stats is not None:
            response["frame_stats"] = frame_stats
        return response

    def _propagate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data["session_id"]
        if session_id not in self.session_store:
            raise ServiceError(404, f"Session not found: {session_id}")

        request: Dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": data.get("direction", "both"),
        }
        if data.get("start_frame_index") is not None:
            request["start_frame_index"] = int(data["start_frame_index"])
        if data.get("max_frame_num_to_track") is not None:
            request["max_frame_num_to_track"] = int(data["max_frame_num_to_track"])

        frames: List[Dict[str, Any]] = []
        for frame in self._predictor_handle_stream(request):
            outputs = frame["outputs"]
            frame_response: Dict[str, Any] = {
                "frame_index": int(frame["frame_index"]),
                "objects": self._format_objects(outputs),
            }
            frame_stats = outputs.get("frame_stats")
            if frame_stats is not None:
                frame_response["frame_stats"] = frame_stats
            frames.append(frame_response)

        self.session_store[session_id].last_access_at = time.time()
        return {"session_id": session_id, "total_frames": len(frames), "frames": frames}

    def _format_objects(self, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        scores = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
        boxes = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)
        masks_np = np.asarray(outputs.get("out_binary_masks", []), dtype=np.bool_)

        if masks_np.size == 0:
            return []

        masks = torch.from_numpy(masks_np).to(torch.bool)
        rles = rle_encode(masks)

        objects: List[Dict[str, Any]] = []
        for i in range(len(obj_ids)):
            mask_rle = rles[i] if i < len(rles) else {"size": [0, 0], "counts": ""}
            objects.append(
                {
                    "obj_id": int(obj_ids[i]),
                    "score": float(scores[i]),
                    "bbox_xywh_norm": [float(v) for v in boxes[i].tolist()],
                    "mask_rle": {
                        "size": mask_rle.get("size", [0, 0]),
                        "counts": mask_rle.get("counts", ""),
                    },
                }
            )
        return objects

    def _cleanup_expired_sessions(self) -> None:
        if SESSION_TTL_SEC <= 0:
            return

        now = time.time()
        expired_ids = [
            session_id
            for session_id, record in self.session_store.items()
            if now - record.last_access_at > SESSION_TTL_SEC
        ]
        for session_id in expired_ids:
            logger.info("Session expired and will be closed: %s", session_id)
            try:
                self._close_session_internal(session_id)
            except Exception as exc:
                logger.warning("Failed to close expired session %s: %s", session_id, exc)

    async def submit(self, op: str, data: Dict[str, Any], timeout: float = 30.0) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = str(uuid.uuid4())

        try:
            self.queue.put_nowait((request_id, op, data, future, loop))
        except Full as exc:
            raise HTTPException(503, "Queue full, try again later") from exc

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, f"Timeout after {timeout}s") from exc
        except ServiceError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected worker error")
            raise HTTPException(500, str(exc)) from exc

    def stats(self) -> Dict[str, Any]:
        return {
            "queue_size": self.queue.qsize(),
            "ready": self.ready.is_set(),
            "active_sessions": len(self.session_store),
            "active_resources": len(self.resource_store),
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }


# ==================== FastAPI App ====================

worker: Optional[SAM3Worker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker
    logger.info("Starting SAM3 API service...")
    worker = SAM3Worker(queue_size=QUEUE_SIZE)
    worker.start()
    logger.info("SAM3 API service started")

    yield

    logger.info("Shutting down SAM3 API service...")
    if worker is not None:
        worker.shutdown_flag.set()


app = FastAPI(title="SAM 3 API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Endpoints ====================

@app.get("/")
async def root() -> Dict[str, Any]:
    return {"service": "SAM 3 API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    if not worker or not worker.ready.is_set():
        raise HTTPException(503, "Service not ready")
    return worker.stats()


@app.post("/api/v1/resources/upload", response_model=ResourceInfo)
async def upload_resource(file: UploadFile = File(...)):
    filename = file.filename or "resource"
    ext = Path(filename).suffix.lower()
    media_type = _media_type_from_path(filename)
    if media_type is None:
        raise HTTPException(
            400,
            f"Unsupported file extension: {ext}. Supported image={sorted(IMAGE_EXTENSIONS)}, video={sorted(VIDEO_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    resource_id = str(uuid.uuid4())
    path = os.path.join(RESOURCE_DIR, f"{resource_id}{ext}")

    try:
        with open(path, "wb") as f:
            f.write(content)
    except OSError as exc:
        raise HTTPException(500, f"Failed to save uploaded file: {exc}") from exc

    try:
        return await worker.submit(
            "register_resource",
            {
                "resource_id": resource_id,
                "media_type": media_type,
                "resource_path": path,
                "size_bytes": len(content),
                "uploaded_at": time.time(),
            },
            timeout=30.0,
        )
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise


@app.get("/api/v1/resources")
async def list_resources() -> Dict[str, Any]:
    return await worker.submit("list_resources", {}, timeout=20.0)


@app.delete("/api/v1/resources/{resource_id}")
async def delete_resource(resource_id: str) -> Dict[str, Any]:
    return await worker.submit("delete_resource", {"resource_id": resource_id}, timeout=20.0)


@app.post("/api/v1/sessions/start", response_model=SessionInfo)
async def start_session(req: StartSessionRequest):
    return await worker.submit("start_session", _model_to_dict(req), timeout=START_TIMEOUT_SEC)


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    return await worker.submit("get_session", {"session_id": session_id}, timeout=20.0)


@app.get("/api/v1/sessions")
async def list_sessions() -> Dict[str, Any]:
    return await worker.submit("list_sessions", {}, timeout=20.0)


@app.post("/api/v1/sessions/{session_id}/prompts/semantic", response_model=PromptResponse)
async def add_semantic_prompt(session_id: str, req: SemanticPromptRequest):
    payload = _model_to_dict(req)
    payload["session_id"] = session_id
    return await worker.submit("prompt_semantic", payload, timeout=PROMPT_TIMEOUT_SEC)


@app.post("/api/v1/sessions/{session_id}/prompts/points", response_model=PromptResponse)
async def add_points_prompt(session_id: str, req: PointsPromptRequest):
    payload = _model_to_dict(req)
    payload["session_id"] = session_id
    return await worker.submit("prompt_points", payload, timeout=PROMPT_TIMEOUT_SEC)


@app.post("/api/v1/sessions/{session_id}/propagate", response_model=PropagateResponse)
async def propagate(session_id: str, req: PropagateRequest):
    payload = _model_to_dict(req)
    payload["session_id"] = session_id
    return await worker.submit("propagate", payload, timeout=PROPAGATE_TIMEOUT_SEC)


@app.post("/api/v1/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> Dict[str, Any]:
    return await worker.submit("reset_session", {"session_id": session_id}, timeout=20.0)


@app.post("/api/v1/sessions/{session_id}/objects/{obj_id}/remove")
async def remove_object(session_id: str, obj_id: int) -> Dict[str, Any]:
    return await worker.submit(
        "remove_object",
        {"session_id": session_id, "obj_id": obj_id},
        timeout=20.0,
    )


@app.delete("/api/v1/sessions/{session_id}")
async def close_session(session_id: str) -> Dict[str, Any]:
    return await worker.submit("close_session", {"session_id": session_id}, timeout=CLOSE_TIMEOUT_SEC)


@app.post("/api/v1/one-shot/segment", response_model=PromptResponse)
async def one_shot_segment(
    file: Optional[UploadFile] = File(None),
    resource_path: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    boxes_xywh_norm: Optional[str] = Form(None),
    box_labels: Optional[str] = Form(None),
):
    if bool(file) == bool(resource_path):
        raise HTTPException(400, "Provide exactly one of file or resource_path")

    boxes = _json_form_field(boxes_xywh_norm, "boxes_xywh_norm")
    labels = _json_form_field(box_labels, "box_labels")

    if text is None and boxes is None:
        raise HTTPException(400, "At least one of text or boxes_xywh_norm must be provided")

    temporary_file = None
    effective_resource_path = resource_path

    if file is not None:
        filename = file.filename or "resource"
        ext = Path(filename).suffix.lower()
        media_type = _media_type_from_path(filename)
        if media_type is None:
            raise HTTPException(
                400,
                f"Unsupported file extension: {ext}. Supported image={sorted(IMAGE_EXTENSIONS)}, video={sorted(VIDEO_EXTENSIONS)}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(400, "Uploaded file is empty")

        temporary_file = os.path.join(ONESHOT_DIR, f"oneshot-{uuid.uuid4()}{ext}")
        try:
            with open(temporary_file, "wb") as f:
                f.write(content)
        except OSError as exc:
            raise HTTPException(500, f"Failed to save one-shot file: {exc}") from exc

        effective_resource_path = temporary_file

    start_resp = await worker.submit(
        "start_session",
        {
            "resource_path": effective_resource_path,
            "resource_id": None,
            "session_id": None,
            "temporary_file": temporary_file,
        },
        timeout=START_TIMEOUT_SEC,
    )

    session_id = start_resp["session_id"]
    try:
        return await worker.submit(
            "prompt_semantic",
            {
                "session_id": session_id,
                "frame_index": 0,
                "text": text,
                "boxes_xywh_norm": boxes,
                "box_labels": labels,
            },
            timeout=PROMPT_TIMEOUT_SEC,
        )
    finally:
        try:
            await worker.submit("close_session", {"session_id": session_id}, timeout=CLOSE_TIMEOUT_SEC)
        except Exception as exc:
            logger.warning("Failed to close one-shot session %s: %s", session_id, exc)


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
