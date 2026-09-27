from __future__ import annotations

import hmac
import json
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .auto_index import SceneAutoIndexer
from .llm.answerer import SpatialAnswerer
from .llm.fallback import FallbackInterpreter
from .llm.openai_compat import OpenAICompatibleInterpreter
from .llm.rules import RuleBasedInterpreter
from .models import Observation, QueryResult, Scene, SpatialEntity
from .object_corrections import ObjectCorrection, ObjectCorrectionUpdate
from .pointcloud import get_point_cloud_variant
from .reconstruction import ReconstructionService
from .relocalization import PoseSmoother, SceneRelocalizer, pose_to_dict, target_guidance
from .semantic import SemanticSearchEngine
from .service import SpatialMemoryService
from .search_answer import search_answer, entity_answer, recorded_location
from .spatial_model import build_spatial_model
from .speech import SpeechTranscriber
from .storage import SpatialMemoryDB
from .video_memory import SceneVideoMemoryIndexer
from .vision.grounding_dino import GroundingDinoDetector, select_near_point, select_target
from .vision.prompts import prompt_from_transcript

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "spatial-memory-v2.db"
DEFAULT_SCENE_ROOT = DEFAULT_DATA_DIR / "scenes-v2"
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_VGGT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vggt" / "model.pt"
DEFAULT_VGGT_REPO = PROJECT_ROOT / "third_party" / "vggt"


def _resolve_path(env_name: str, default_path: Path, legacy_unix_path: str) -> Path:
    env_val = os.getenv(env_name)
    if env_val:
        return Path(env_val)
    if os.name != "nt" and Path(legacy_unix_path).exists():
        return Path(legacy_unix_path)
    return default_path


DB_PATH = _resolve_path("SPATIAL_DB", DEFAULT_DB_PATH, "/results/spatial-memory.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_interpreter():
    fallback = RuleBasedInterpreter()
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if base_url and model:
        primary = OpenAICompatibleInterpreter(
            base_url=base_url,
            model=model,
            api_key=os.getenv("LLM_API_KEY", "local"),
        )
        return FallbackInterpreter(primary=primary, fallback=fallback)
    return fallback


db = SpatialMemoryDB(DB_PATH)
answerer = None
if os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"):
    answerer = SpatialAnswerer(
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.getenv("LLM_API_KEY", "local"),
    )

semantic_engine = None
if os.getenv("SEMANTIC_SEARCH_ENABLED", "1") not in {"0", "false", "False"}:
    semantic_engine = SemanticSearchEngine(
        model_id=os.getenv(
            "SEMANTIC_MODEL",
            "intfloat/multilingual-e5-small",
        )
    )
service = SpatialMemoryService(
    db,
    build_interpreter(),
    semantic=semantic_engine,
)
vision_detector = GroundingDinoDetector(
    model_id=os.getenv("VISION_MODEL", "IDEA-Research/grounding-dino-tiny")
)
speech_transcriber = SpeechTranscriber(
    model_id=os.getenv("SPEECH_MODEL", "openai/whisper-large-v3-turbo")
)
SCENE_ROOT = _resolve_path("SCENE_ROOT", DEFAULT_SCENE_ROOT, "/results/scenes")
SCENE_ROOT.mkdir(parents=True, exist_ok=True)

reconstruction_service = ReconstructionService(
    db=db,
    root_dir=SCENE_ROOT,
    backend=os.getenv("RECONSTRUCTION_BACKEND", "vggt"),
    model_id=os.getenv("RECONSTRUCTION_MODEL", "depth-anything/DA3-LARGE-1.1"),
    vggt_checkpoint=_resolve_path(
        "VGGT_CHECKPOINT",
        DEFAULT_VGGT_CHECKPOINT,
        "/results/checkpoints/vggt/model.pt",
    ),
    vggt_repo=_resolve_path(
        "VGGT_REPO",
        DEFAULT_VGGT_REPO,
        "/workspace/third_party/vggt",
    ),
    vggt_omega_checkpoint=os.getenv("VGGT_OMEGA_CHECKPOINT") or None,
    vggt_omega_resolution=int(os.getenv("VGGT_OMEGA_RESOLUTION", "512")),
)
auto_indexer = SceneAutoIndexer(
    db=db,
    memory=service,
    detector=vision_detector,
)
video_memory_indexer = SceneVideoMemoryIndexer(
    db=db,
    memory=service,
    detector=vision_detector,
    transcriber=speech_transcriber,
)
relocalizer = SceneRelocalizer()
pose_smoother = PoseSmoother()
app = FastAPI(title="Spatial Memory API V2", version="0.2.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )


WEB_DIR = _resolve_path("SPATIAL_WEB_DIR", DEFAULT_WEB_DIR, "/workspace/web")
DEMO_ACCESS_TOKEN = os.getenv("DEMO_ACCESS_TOKEN", "").strip()
DEMO_ACCESS_COOKIE = "spatial_v2_access"


@app.middleware("http")
async def demo_access_control(request: Request, call_next):
    if not DEMO_ACCESS_TOKEN or request.url.path == "/health":
        return await call_next(request)

    supplied = request.query_params.get("access_token") or request.cookies.get(DEMO_ACCESS_COOKIE)
    auth = request.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()

    if not supplied or not hmac.compare_digest(supplied, DEMO_ACCESS_TOKEN):
        return Response(
            "Spatial Memory V2 access token required.",
            status_code=401,
            media_type="text/plain",
        )

    response = await call_next(request)
    if request.query_params.get("access_token"):
        response.set_cookie(
            DEMO_ACCESS_COOKIE,
            DEMO_ACCESS_TOKEN,
            httponly=True,
            samesite="lax",
            secure=request.headers.get("x-forwarded-proto", "").lower() == "https",
            max_age=60 * 60 * 8,
        )
    return response


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def demo_page(request: Request):
    if request.query_params.get("scene"):
        return FileResponse(WEB_DIR / "index.html")
    return FileResponse(WEB_DIR / "home.html")


@app.get("/room/new", include_in_schema=False)
def new_room_page():
    return FileResponse(WEB_DIR / "new-room.html")


@app.get("/rooms", include_in_schema=False)
def rooms_page():
    return FileResponse(WEB_DIR / "rooms.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


class QueryRequest(BaseModel):
    scene_id: str
    text: str
    limit: int = 5


@app.get("/health")
def health() -> dict:
    health = {
        "status": "ok",
        "version": "v2",
    }
    if DEMO_ACCESS_TOKEN:
        return health
    health.update({
        "db": str(DB_PATH),
        "interpreter": type(service.interpreter).__name__,
        "reconstruction_backend": reconstruction_service.backend,
        "reconstruction_model": reconstruction_service.model_label,
    })
    return health


@app.get("/demo/health")
def demo_health() -> dict:
    import httpx
    import torch

    gpu_available = bool(torch.cuda.is_available())
    gpu = {
        "available": gpu_available,
        "name": torch.cuda.get_device_name(0) if gpu_available else None,
        "allocated_gib": (
            torch.cuda.memory_allocated(0) / 2**30
            if gpu_available else 0.0
        ),
        "reserved_gib": (
            torch.cuda.memory_reserved(0) / 2**30
            if gpu_available else 0.0
        ),
    }

    if reconstruction_service.backend == "vggt":
        reconstruction_available = reconstruction_service.vggt_checkpoint.exists()
        artifact = str(reconstruction_service.vggt_checkpoint)
    elif reconstruction_service.backend == "vggt-omega":
        checkpoint = reconstruction_service.vggt_omega_checkpoint
        reconstruction_available = bool(checkpoint and checkpoint.exists())
        artifact = str(checkpoint) if checkpoint else None
    else:
        reconstruction_available = True
        artifact = reconstruction_service.model_id

    llm_base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    llm_health_url = (
        llm_base_url[:-3] + "/health"
        if llm_base_url.endswith("/v1")
        else llm_base_url + "/health"
    )
    llm_online = False
    llm_loaded = False
    llm_detail = None
    if llm_base_url:
        try:
            response = httpx.get(llm_health_url, timeout=0.8)
            response.raise_for_status()
            payload = response.json()
            llm_online = payload.get("status") == "ok"
            llm_loaded = bool(payload.get("loaded"))
            llm_detail = payload
        except Exception:
            llm_detail = None

    scenes = db.list_scenes()
    core_ready = bool(gpu_available and reconstruction_available)
    return {
        "status": "ready" if core_ready else "degraded",
        "api": {
            "ready": True,
            "version": "v2",
            "db": str(DB_PATH),
            "scene_count": len(scenes),
        },
        "gpu": gpu,
        "reconstruction": {
            "available": reconstruction_available,
            "loaded": reconstruction_service._adapter is not None,
            "backend": reconstruction_service.backend,
            "model": reconstruction_service.model_label,
            "artifact": artifact,
        },
        "vision": {
            "available": True,
            "loaded": vision_detector.model is not None,
            "model": vision_detector.model_id,
        },
        "whisper": {
            "available": True,
            "loaded": speech_transcriber.pipe is not None,
            "model": speech_transcriber.model_id,
        },
        "llm": {
            "configured": bool(llm_base_url),
            "online": llm_online,
            "loaded": llm_loaded,
            "model": os.getenv("LLM_MODEL"),
            "mode": "local_qwen" if llm_online else "deterministic_fallback",
            "detail": llm_detail,
        },
    }


@app.post("/scenes", response_model=Scene)
def create_scene(scene: Scene) -> Scene:
    if db.is_scene_deleted(scene.id):
        raise HTTPException(status_code=409, detail="削除した部屋は再登録できません")
    return db.upsert_scene(scene)


@app.get("/scenes", response_model=list[Scene])
def list_scenes() -> list[Scene]:
    return db.list_scenes()


@app.get("/scenes/{scene_id}", response_model=Scene)
def get_scene(scene_id: str) -> Scene:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene


@app.delete("/scenes/{scene_id}")
def delete_scene(scene_id: str) -> dict:
    if not db.delete_scene(scene_id):
        raise HTTPException(status_code=404, detail="部屋が見つかりません")
    return {"deleted": scene_id}


@app.get("/scenes/{scene_id}/object-corrections", response_model=list[ObjectCorrection])
def list_object_corrections(scene_id: str, source_prediction: str = Query(min_length=1, max_length=200)):
    if db.get_scene(scene_id) is None:
        raise HTTPException(status_code=404, detail="部屋が見つかりません")
    return db.list_object_corrections(scene_id, source_prediction)


@app.post("/scenes/{scene_id}/object-corrections", response_model=ObjectCorrection)
def update_object_correction(scene_id: str, update: ObjectCorrectionUpdate):
    try:
        return db.update_object_correction(scene_id, update)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/entities", response_model=SpatialEntity)
def create_entity(entity: SpatialEntity) -> SpatialEntity:
    if db.get_scene(entity.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return db.upsert_entity(entity)


@app.get("/scenes/{scene_id}/entities", response_model=list[SpatialEntity])
def list_entities(scene_id: str) -> list[SpatialEntity]:
    return db.list_entities(scene_id)


@app.delete("/entities/{entity_id}")
def delete_entity(entity_id: str) -> dict:
    if not db.delete_entity(entity_id):
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"deleted": entity_id}


@app.post("/observations", response_model=SpatialEntity)
async def ingest_observation(observation: Observation) -> SpatialEntity:
    if db.get_scene(observation.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return await service.ingest_observation(observation)


@app.post("/query", response_model=QueryResult)
async def query_memory(request: QueryRequest) -> QueryResult:
    if db.get_scene(request.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return await service.query(request.scene_id, request.text, request.limit)


@app.get("/viewer/state/{scene_id}")
def viewer_state(scene_id: str, focus: str | None = Query(default=None)) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    entities = db.list_entities(scene_id)
    focus_entity = db.get_entity(focus) if focus else None
    return {
        "scene": scene.model_dump(mode="json"),
        "entities": [e.model_dump(mode="json") for e in entities],
        "focus": focus_entity.model_dump(mode="json") if focus_entity else None,
    }


class ProjectionRequest(BaseModel):
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int]


@app.post("/scenes/{scene_id}/project")
def project_detection(scene_id: str, request: ProjectionRequest) -> dict:
    from .projection import project_bbox_from_npz

    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.reconstruction_npz:
        raise HTTPException(status_code=409, detail="Scene has no reconstruction_npz")
    try:
        point = project_bbox_from_npz(
            scene.reconstruction_npz,
            request.frame_index,
            request.bbox_xyxy,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"scene_id": scene_id, "position": point.model_dump()}


from .models import EntityKind, NavEdge, NavNode, RouteResult, Vec3
from .navigation import route_to_entity


class NavigateRequest(BaseModel):
    scene_id: str
    start: Vec3
    target_entity_id: str


class GuideRequest(BaseModel):
    scene_id: str
    start: Vec3
    text: str


@app.post("/navigation/nodes", response_model=NavNode)
def create_nav_node(node: NavNode) -> NavNode:
    if db.get_scene(node.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return db.upsert_nav_node(node)


@app.post("/navigation/edges", response_model=NavEdge)
def create_nav_edge(edge: NavEdge) -> NavEdge:
    if db.get_scene(edge.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return db.upsert_nav_edge(edge)


@app.get("/scenes/{scene_id}/navigation")
def navigation_state(scene_id: str) -> dict:
    return {
        "nodes": [n.model_dump(mode="json") for n in db.list_nav_nodes(scene_id)],
        "edges": [e.model_dump(mode="json") for e in db.list_nav_edges(scene_id)],
    }


@app.post("/navigate", response_model=RouteResult)
def navigate(request: NavigateRequest) -> RouteResult:
    try:
        return route_to_entity(db, request.scene_id, request.start, request.target_entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/guide")
async def guide(request: GuideRequest) -> dict:
    result = await service.query(request.scene_id, request.text, limit=5)
    if result.target is None:
        raise HTTPException(status_code=404, detail="No matching entity")
    try:
        route = route_to_entity(db, request.scene_id, request.start, result.target.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "query": result.model_dump(mode="json"),
        "route": route.model_dump(mode="json"),
    }


class DetectionObservationRequest(BaseModel):
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int]
    source_width: int | None = None
    source_height: int | None = None
    transcript: str | None = None
    detected_label: str | None = None
    detection_confidence: float | None = None
    kind: EntityKind = "object"
    parent_id: str | None = None


@app.post("/scenes/{scene_id}/detections/ingest", response_model=SpatialEntity)
async def ingest_detection(scene_id: str, request: DetectionObservationRequest) -> SpatialEntity:
    from .projection import project_bbox_from_npz

    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.reconstruction_npz:
        raise HTTPException(status_code=409, detail="Scene has no reconstruction_npz")
    try:
        source_size = None
        if request.source_width and request.source_height:
            source_size = (request.source_width, request.source_height)
        position = project_bbox_from_npz(
            scene.reconstruction_npz,
            request.frame_index,
            request.bbox_xyxy,
            source_size=source_size,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    observation = Observation(
        scene_id=scene_id,
        position=position,
        kind=request.kind,
        parent_id=request.parent_id,
        transcript=request.transcript,
        detected_label=request.detected_label,
        detection_confidence=request.detection_confidence,
    )
    return await service.ingest_observation(observation)


@app.post("/vision/detect")
async def vision_detect(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    transcript: str = Form(""),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> dict:
    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    detection_prompt = prompt_from_transcript(transcript, explicit=prompt or None)
    if not detection_prompt:
        raise HTTPException(status_code=422, detail="Could not derive an object prompt")
    labels = [part.strip() for part in detection_prompt.split(",") if part.strip()]
    detections = vision_detector.detect(
        pil,
        labels=labels,
        threshold=threshold,
        text_threshold=text_threshold,
    )
    target = select_target(detections)
    return {
        "prompt": detection_prompt,
        "detections": [d.model_dump() for d in detections],
        "target": target.model_dump() if target else None,
        "metrics": vision_detector.last_metrics,
    }


@app.post("/scenes/{scene_id}/vision/remember")
async def vision_remember(
    scene_id: str,
    image: UploadFile = File(...),
    frame_index: int = Form(...),
    transcript: str = Form(""),
    prompt: str | None = Form(None),
    kind: EntityKind = Form("object"),
    parent_id: str | None = Form(None),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> dict:
    from .projection import project_bbox_from_npz

    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.reconstruction_npz:
        raise HTTPException(status_code=409, detail="Scene has no reconstruction_npz")

    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    detection_prompt = prompt_from_transcript(transcript, explicit=prompt)
    if not detection_prompt:
        raise HTTPException(status_code=422, detail="Could not derive an object prompt")


    detections = vision_detector.detect(
        pil,
        labels=[detection_prompt],
        threshold=threshold,
        text_threshold=text_threshold,
    )
    target = select_target(detections)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"No object matched prompt: {detection_prompt}",
        )

    bbox = tuple(int(round(v)) for v in target.bbox_xyxy)
    try:
        position = project_bbox_from_npz(
            scene.reconstruction_npz,
            frame_index,
            bbox,
            source_size=(pil.width, pil.height),
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    observation = Observation(
        scene_id=scene_id,
        position=position,
        kind=kind,
        parent_id=parent_id,
        transcript=transcript or None,
        detected_label=target.label,
        detection_confidence=target.score,
    )
    entity = await service.ingest_observation(observation)

    return {
        "prompt": detection_prompt,
        "detection": target.model_dump(),
        "position": position.model_dump(),
        "entity": entity.model_dump(mode="json"),
        "metrics": vision_detector.last_metrics,
    }


async def _remember_visual_from_pil(
    scene_id: str,
    pil: Image.Image,
    frame_index: int,
    transcript: str,
    prompt: str | None,
    kind: EntityKind,
    parent_id: str | None,
    threshold: float,
    text_threshold: float,
) -> dict:
    from .projection import project_bbox_from_npz

    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.reconstruction_npz:
        raise HTTPException(status_code=409, detail="Scene has no reconstruction_npz")

    detection_prompt = prompt_from_transcript(transcript, explicit=prompt)
    if not detection_prompt:
        raise HTTPException(status_code=422, detail="Could not derive an object prompt")


    detections = vision_detector.detect(
        pil,
        labels=[detection_prompt],
        threshold=threshold,
        text_threshold=text_threshold,
    )
    target = select_target(detections)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"No object matched prompt: {detection_prompt}",
        )

    bbox = tuple(int(round(v)) for v in target.bbox_xyxy)
    try:
        position = project_bbox_from_npz(
            scene.reconstruction_npz,
            frame_index,
            bbox,
            source_size=(pil.width, pil.height),
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


    observation = Observation(
        scene_id=scene_id,
        position=position,
        kind=kind,
        parent_id=parent_id,
        transcript=transcript or None,
        detected_label=target.label,
        detection_confidence=target.score,
    )
    entity = await service.ingest_observation(observation)
    return {
        "prompt": detection_prompt,
        "detection": target.model_dump(),
        "position": position.model_dump(),
        "entity": entity.model_dump(mode="json"),
        "vision_metrics": vision_detector.last_metrics,
    }


@app.post("/speech/transcribe")
async def speech_transcribe(
    audio: UploadFile = File(...),
    language: str = Form("ja"),
) -> dict:
    raw = await audio.read()
    try:
        result = speech_transcriber.transcribe(raw, language=language or None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio decode/transcription failed: {exc}") from exc
    return {
        "text": result.text,
        "language": result.language,
        "metrics": result.metrics,
    }


@app.post("/scenes/{scene_id}/voice/remember")
async def voice_remember(
    scene_id: str,
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    frame_index: int = Form(...),
    language: str = Form("ja"),
    prompt: str | None = Form(None),
    kind: EntityKind = Form("object"),
    parent_id: str | None = Form(None),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> dict:
    raw_audio = await audio.read()
    try:
        transcription = speech_transcriber.transcribe(
            raw_audio,
            language=language or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Audio decode/transcription failed: {exc}",
        ) from exc

    if not transcription.text:
        raise HTTPException(status_code=422, detail="No speech detected")

    raw_image = await image.read()
    try:
        pil = Image.open(BytesIO(raw_image)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc


    memory = await _remember_visual_from_pil(
        scene_id=scene_id,
        pil=pil,
        frame_index=frame_index,
        transcript=transcription.text,
        prompt=prompt,
        kind=kind,
        parent_id=parent_id,
        threshold=threshold,
        text_threshold=text_threshold,
    )
    return {
        "transcription": {
            "text": transcription.text,
            "language": transcription.language,
            "metrics": transcription.metrics,
        },
        **memory,
    }


def _load_spatial_model(scene: Scene) -> tuple[dict, Path]:
    if not scene.spatial_model_path:
        raise HTTPException(status_code=404, detail="Scene has no spatial model")
    manifest_path = Path(scene.spatial_model_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Spatial model manifest not found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Spatial model manifest is invalid") from exc
    manifest["revision"] = str(manifest_path.stat().st_mtime_ns)
    if isinstance(manifest.get("shell"), dict):
        manifest["shell"]["url"] = f"/scenes/{scene.id}/spatial-model/shell"
    return manifest, manifest_path


@app.get("/scenes/{scene_id}/spatial-model")
def scene_spatial_model(scene_id: str) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    manifest, _ = _load_spatial_model(scene)
    return manifest


@app.get("/scenes/{scene_id}/spatial-model/shell")
def scene_spatial_model_shell(scene_id: str, variant: str = Query("raw", pattern="^(raw|hybrid|fusion)$")):
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    manifest, manifest_path = _load_spatial_model(scene)
    shell = manifest.get("shell") or {}
    shell_path = Path(shell.get("path") or manifest_path.parent / "shell.ply")
    if variant in {"hybrid", "fusion"}:
        hybrid_path = manifest.get(variant, {}).get("shell_path")
        if not hybrid_path:
            raise HTTPException(status_code=404, detail=f"{variant} model is not built yet")
        shell_path = Path(hybrid_path)
    if not shell_path.exists():
        raise HTTPException(status_code=404, detail="Spatial model shell not found")
    return FileResponse(
        shell_path,
        media_type="application/octet-stream",
        filename=shell_path.name,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@app.get("/scenes/{scene_id}/spatial-model/texture/{asset}")
def scene_texture_asset(scene_id: str, asset: str):
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    manifest, _ = _load_spatial_model(scene)
    key = {"geometry": "geometry_path", "atlas": "atlas_path"}.get(asset)
    value = manifest.get("texture", {}).get(key) if key else None
    if not value or not Path(value).is_file():
        raise HTTPException(status_code=404, detail="Photo texture not built")
    return FileResponse(value, media_type="application/json" if asset == "geometry" else "image/jpeg",
                        headers={"Cache-Control": "no-cache"})


@app.get("/scenes/{scene_id}/spatial-model/appearance-2dgs")
def scene_appearance_2dgs(scene_id: str, variant: str = Query('current', pattern='^(current|baseline)$')):
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    manifest, _ = _load_spatial_model(scene)
    key = 'appearance_2dgs_baseline' if variant == 'baseline' else 'appearance_2dgs'
    value = manifest.get(key, {}).get("path")
    if not value or not Path(value).is_file():
        raise HTTPException(status_code=404, detail="2DGS appearance is not trained yet")
    return FileResponse(value, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})


@app.post("/scenes/{scene_id}/spatial-model/rebuild")
def rebuild_scene_spatial_model(scene_id: str) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.point_cloud_path:
        raise HTTPException(status_code=409, detail="Scene has no point cloud")
    point_cloud = Path(scene.point_cloud_path)
    if not point_cloud.exists():
        raise HTTPException(status_code=404, detail="Point cloud file not found")
    if scene.frames_dir:
        output_dir = Path(scene.frames_dir).parent / "spatial_model"
    else:
        output_dir = point_cloud.parent / "spatial_model"
    try:
        manifest = build_spatial_model(
            point_cloud_path=point_cloud,
            output_dir=output_dir,
            scene_id=scene_id,
            target_triangles=20_000,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    updated = scene.model_copy(update={
        "spatial_model_path": manifest["manifest_path"],
    })
    db.upsert_scene(updated)
    manifest["revision"] = str(Path(manifest["manifest_path"]).stat().st_mtime_ns)
    manifest["shell"]["url"] = f"/scenes/{scene_id}/spatial-model/shell"
    return manifest


@app.get("/scenes/{scene_id}/point-cloud")
def scene_point_cloud(
    scene_id: str,
    max_points: int | None = Query(None, ge=5_000, le=500_000),
    variant: str = Query("raw", pattern="^(raw|hybrid)$"),
):
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.point_cloud_path:
        raise HTTPException(status_code=404, detail="Scene has no point cloud")

    path = Path(scene.point_cloud_path)
    if variant == "hybrid":
        manifest, _ = _load_spatial_model(scene)
        hybrid_path = manifest.get("hybrid", {}).get("point_path")
        if not hybrid_path:
            raise HTTPException(status_code=404, detail="Hybrid model is not built yet")
        path = Path(hybrid_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Point cloud file not found")
    served_path = get_point_cloud_variant(path, max_points=max_points)
    return FileResponse(
        served_path,
        media_type="application/octet-stream",
        filename=served_path.name,
        headers={"Cache-Control": "private, max-age=86400"},
    )


from .room_jobs import RoomJobs
room_jobs = RoomJobs(PROJECT_ROOT, SCENE_ROOT)
from .capture_jobs import CaptureJobs
from .capture_api import build_capture_router
capture_jobs = CaptureJobs(PROJECT_ROOT, reconstruction_service, room_jobs)
app.include_router(build_capture_router(PROJECT_ROOT, db, capture_jobs, room_jobs))


@app.get("/scenes/{scene_id}/solid-room-job")
def solid_room_job_status(scene_id: str):
    if not db.get_scene(scene_id):raise HTTPException(404, '部屋が見つかりません')
    try:return room_jobs.status(scene_id)
    except ValueError as exc:raise HTTPException(422, str(exc)) from exc


@app.post("/scenes/{scene_id}/solid-room-job")
def solid_room_job_start(scene_id: str, up_axis: str = Query('image-top'), force: bool = Query(False)):
    if not db.get_scene(scene_id):raise HTTPException(404, '部屋が見つかりません')
    if capture_jobs.active:raise HTTPException(409, '追加撮影を処理中です。完了後に開始してください。')
    try:return room_jobs.start(scene_id, up_axis, force=force)
    except ValueError as exc:raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:raise HTTPException(409, str(exc)) from exc


@app.post("/scenes/reconstruct")
def reconstruct_scene(
    video: UploadFile = File(...),
    name: str = Form("New Scene"),
    description: str | None = Form(None),
    sample_fps: float = Form(2.0),
    max_frames: int = Form(60),
    point_stride: int = Form(4),
    quality_filter: bool = Form(True),
) -> dict:
    if sample_fps <= 0 or sample_fps > 30:
        raise HTTPException(status_code=422, detail="sample_fps must be in (0, 30]")
    if max_frames < 2 or max_frames > 240:
        raise HTTPException(status_code=422, detail="max_frames must be between 2 and 240")
    if point_stride < 1 or point_stride > 32:
        raise HTTPException(status_code=422, detail="point_stride must be between 1 and 32")

    try:
        with capture_jobs.legacy_reconstruction():
            scene, metrics = reconstruction_service.reconstruct_upload(
                upload=video,
                name=name,
                description=description,
                sample_fps=sample_fps,
                max_frames=max_frames,
                point_stride=point_stride,
                quality_filter=quality_filter,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"3D再構成モデルまたはチェックポイントが見つかりません: {exc}。初期デモルームをご利用いただくか、モデルチェックポイントを配置してください。",
        ) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"3D再構成の依存モジュールが見つかりません: {exc}。初期デモルームをご利用いただくか、依存関係を導入してください。",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"3D再構成処理中にエラーが発生しました: {exc}",
        ) from exc

    return {
        "scene": scene.model_dump(mode="json"),
        "metrics": metrics,
    }


@app.get("/scenes/{scene_id}/frames")
def scene_frames(scene_id: str) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.frames_dir:
        raise HTTPException(status_code=404, detail="Scene has no extracted frames")

    frames_dir = Path(scene.frames_dir)
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames directory not found")

    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    manifest_by_name: dict[str, dict] = {}
    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_by_name = {
                str(item.get("name")): item
                for item in manifest.get("frames", [])
                if item.get("name")
            }
        except (OSError, ValueError, TypeError):
            manifest_by_name = {}

    return {
        "scene_id": scene_id,
        "count": len(frames),
        "frames": [
            {
                "index": i,
                "name": frame.name,
                "url": f"/scenes/{scene_id}/frames/{i}",
                "timestamp_seconds": manifest_by_name.get(frame.name, {}).get("timestamp_seconds"),
                "source_frame_index": manifest_by_name.get(frame.name, {}).get("source_frame_index"),
            }
            for i, frame in enumerate(frames)
        ],
    }


@app.get("/scenes/{scene_id}/frames/{frame_index}")
def scene_frame(scene_id: str, frame_index: int):
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.frames_dir:
        raise HTTPException(status_code=404, detail="Scene has no extracted frames")

    frames_dir = Path(scene.frames_dir)
    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")

    frame = frames[frame_index]
    media = "image/jpeg" if frame.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(frame, media_type=media)


@app.post("/scenes/{scene_id}/auto-index")
async def auto_index_scene(
    scene_id: str,
    labels: str | None = Form(None),
    frame_step: int = Form(5),
    max_frames: int = Form(12),
    max_detections_per_frame: int = Form(20),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    cluster_radius: float = Form(0.45),
    max_shape_distance: float = Form(0.95),
    max_appearance_distance: float = Form(0.32),
    singleton_min_confidence: float = Form(0.55),
) -> dict:
    if frame_step < 1 or frame_step > 120:
        raise HTTPException(status_code=422, detail="frame_step must be between 1 and 120")
    if max_frames < 1 or max_frames > 120:
        raise HTTPException(status_code=422, detail="max_frames must be between 1 and 120")
    if max_detections_per_frame < 1 or max_detections_per_frame > 100:
        raise HTTPException(
            status_code=422,
            detail="max_detections_per_frame must be between 1 and 100",
        )
    if cluster_radius < 0.05 or cluster_radius > 3.0:
        raise HTTPException(
            status_code=422,
            detail="cluster_radius must be between 0.05 and 3.0",
        )
    if max_shape_distance < 0.05 or max_shape_distance > 3.0:
        raise HTTPException(
            status_code=422,
            detail="max_shape_distance must be between 0.05 and 3.0",
        )
    if max_appearance_distance < 0.01 or max_appearance_distance > 1.5:
        raise HTTPException(
            status_code=422,
            detail="max_appearance_distance must be between 0.01 and 1.5",
        )
    if singleton_min_confidence < 0.0 or singleton_min_confidence > 1.0:
        raise HTTPException(
            status_code=422,
            detail="singleton_min_confidence must be in [0, 1]",
        )

    parsed_labels = None
    if labels:
        parsed_labels = [part.strip() for part in labels.split(",") if part.strip()]

    try:
        return await auto_indexer.run(
            scene_id=scene_id,
            labels=parsed_labels,
            frame_step=frame_step,
            max_frames=max_frames,
            max_detections_per_frame=max_detections_per_frame,
            threshold=threshold,
            text_threshold=text_threshold,
            cluster_radius=cluster_radius,
            max_shape_distance=max_shape_distance,
            max_appearance_distance=max_appearance_distance,
            singleton_min_confidence=singleton_min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/scenes/{scene_id}/audio-memory")
async def audio_memory_scene(
    scene_id: str,
    language: str = Form("ja"),
    detection_threshold: float = Form(0.25),
    text_threshold: float = Form(0.25),
    auto_confirm_threshold: float = Form(0.60),
) -> dict:
    if not 0.0 <= detection_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="detection_threshold must be in [0, 1]")
    if not 0.0 <= text_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="text_threshold must be in [0, 1]")
    if not 0.0 <= auto_confirm_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="auto_confirm_threshold must be in [0, 1]")
    if auto_confirm_threshold < detection_threshold:
        raise HTTPException(
            status_code=422,
            detail="auto_confirm_threshold must be >= detection_threshold",
        )
    try:
        return await video_memory_indexer.run(
            scene_id=scene_id,
            language=language or None,
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            auto_confirm_threshold=auto_confirm_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/scenes/{scene_id}/audio-memory")
def audio_memory_report(scene_id: str) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    empty = {
        "scene_id": scene_id,
        "transcript": "",
        "segments": [],
        "utterances": [],
        "confirmed": [],
        "pending": [],
        "confirmed_count": 0,
        "pending_count": 0,
        "status": "not_indexed",
    }
    if not scene.frames_dir:
        return empty
    report_path = Path(scene.frames_dir).parent / "audio_memory.json"
    if not report_path.exists():
        return empty
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Audio memory report is invalid") from exc


class AskRequest(BaseModel):
    scene_id: str
    text: str
    limit: int = 5
    start: Vec3 | None = None


def _fallback_spatial_answer(result: QueryResult) -> str:
    return search_answer(result, db.get_entity)


@app.post("/ask")
async def ask_spatial_memory(request: AskRequest) -> dict:
    if db.get_scene(request.scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")

    result = await service.query(
        request.scene_id,
        request.text,
        limit=max(1, min(request.limit, 10)),
    )
    answer = _fallback_spatial_answer(result)
    answer_source = "deterministic"
    # Keep location answers grounded in registered fields, including abstentions.

    route = None
    if request.start is not None and result.target is not None:
        try:
            route = route_to_entity(
                db,
                request.scene_id,
                request.start,
                result.target.id,
            ).model_dump(mode="json")
        except ValueError:
            route = None

    return {
        "answer": answer,
        "answer_source": answer_source,
        "candidate_details": {hit.entity.id: {
            "answer": entity_answer(hit.entity, db.get_entity),
            "location": recorded_location(hit.entity, db.get_entity),
        } for hit in (result.candidates or result.hits)},
        "query": result.model_dump(mode="json"),
        "route": route,
    }


@app.post("/scenes/{scene_id}/locate-in-image")
async def locate_entity_in_image(
    scene_id: str,
    image: UploadFile = File(...),
    query: str = Form(...),
    prompt: str | None = Form(None),
    threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
) -> dict:
    if db.get_scene(scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")

    result = await service.query(scene_id, query, limit=5)
    target = result.target
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="No matching Spatial Memory entity",
        )

    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    detection_prompt = prompt_from_transcript(
        " ".join(
            [target.name, *target.aliases, *target.tags]
        ),
        explicit=prompt,
    )
    if not detection_prompt:
        raise HTTPException(
            status_code=422,
            detail="Could not derive a detector prompt for the target entity",
        )

    detections = vision_detector.detect(
        pil,
        labels=[detection_prompt],
        threshold=threshold,
        text_threshold=text_threshold,
    )
    detected = select_target(detections)

    return {
        "query": result.model_dump(mode="json"),
        "target_entity": target.model_dump(mode="json"),
        "prompt": detection_prompt,
        "found_in_image": detected is not None,
        "detection": detected.model_dump() if detected else None,
        "detections": [item.model_dump() for item in detections],
        "metrics": vision_detector.last_metrics,
    }


@app.post("/scenes/{scene_id}/relocalize")
async def relocalize_camera(
    scene_id: str,
    image: UploadFile = File(...),
    max_reference_frames: int = Form(60),
) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.frames_dir or not scene.reconstruction_npz:
        raise HTTPException(
            status_code=409,
            detail="Scene requires frames and reconstruction data",
        )

    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    try:
        pose = relocalizer.relocalize(
            scene,
            pil,
            max_reference_frames=max(1, min(max_reference_frames, 120)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "scene_id": scene_id,
        "pose": pose_to_dict(pose),
    }


@app.post("/scenes/{scene_id}/ar-guide")
async def ar_guide(
    scene_id: str,
    image: UploadFile = File(...),
    query: str = Form(...),
    verify_visible: bool = Form(True),
    threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
    max_reference_frames: int = Form(60),
) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.frames_dir or not scene.reconstruction_npz:
        raise HTTPException(
            status_code=409,
            detail="Scene requires frames and reconstruction data",
        )

    result = await service.query(scene_id, query, limit=5)
    target = result.target
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="No matching Spatial Memory entity",
        )

    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    try:
        pose = relocalizer.relocalize(
            scene,
            pil,
            max_reference_frames=max(1, min(max_reference_frames, 120)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    guidance = target_guidance(pose, target)
    detection = None
    detection_prompt = None

    if verify_visible and guidance["on_screen"]:
        combined = " ".join(
            [target.name, *target.aliases, *target.tags]
        )
        detection_prompt = prompt_from_transcript(combined)
        if detection_prompt:
            detections = vision_detector.detect(
                pil,
                labels=[detection_prompt],
                threshold=threshold,
                text_threshold=text_threshold,
            )
            point = guidance["screen_target"]
            detection = select_near_point(
                detections,
                point["x"],
                point["y"],
            )

    mode = "bbox" if detection is not None else "arrow"
    return {
        "scene_id": scene_id,
        "query": result.model_dump(mode="json"),
        "target_entity": target.model_dump(mode="json"),
        "pose": pose_to_dict(pose),
        "guidance": guidance,
        "mode": mode,
        "detection_prompt": detection_prompt,
        "detection": detection.model_dump() if detection else None,
    }


@app.post("/scenes/{scene_id}/relocalization/prepare")
def prepare_relocalization(
    scene_id: str,
    max_reference_frames: int = Form(60),
) -> dict:
    import time

    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not scene.frames_dir:
        raise HTTPException(status_code=409, detail="Scene has no frames")

    started = time.perf_counter()
    try:
        refs = relocalizer.prepare(
            scene,
            max_reference_frames=max(1, min(max_reference_frames, 120)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "scene_id": scene_id,
        "reference_frames": len(refs),
        "seconds": time.perf_counter() - started,
        "index": dict(relocalizer.last_prepare_info),
    }


@app.post("/scenes/{scene_id}/ar-track")
async def ar_track(
    scene_id: str,
    image: UploadFile = File(...),
    target_entity_id: str = Form(...),
    tracking_session: str = Form("default"),
    verify_visible: bool = Form(False),
    threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
    max_reference_frames: int = Form(60),
) -> dict:
    scene = db.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    target = db.get_entity(target_entity_id)
    if target is None or target.scene_id != scene_id:
        raise HTTPException(status_code=404, detail="Target entity not found in Scene")
    if not scene.frames_dir or not scene.reconstruction_npz:
        raise HTTPException(
            status_code=409,
            detail="Scene requires frames and reconstruction data",
        )

    raw = await image.read()
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    tracking_key = f"{scene_id}:{tracking_session}"
    recent_before = pose_smoother.recent_pose(
        tracking_key,
        max_age_seconds=5.0,
    )
    fast_tracking_search = recent_before is not None
    try:
        raw_pose = relocalizer.relocalize(
            scene,
            pil,
            max_reference_frames=max(1, min(max_reference_frames, 120)),
            search_all_references=not fast_tracking_search,
        )
        pose, smoothing = pose_smoother.update(
            tracking_key,
            raw_pose,
        )
        pose_source = "relocalized"
    except ValueError as exc:
        if recent_before is None:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        pose, fallback_age = recent_before
        pose_smoother.refresh_recent_pose(tracking_key)
        raw_pose = None
        pose_source = "temporal_fallback"
        smoothing = {
            "applied": True,
            "fallback": True,
            "fallback_age_seconds": fallback_age,
            "reason": str(exc),
        }

    guidance = target_guidance(pose, target)
    detection = None
    detection_prompt = None

    if verify_visible and guidance["on_screen"]:
        combined = " ".join([target.name, *target.aliases, *target.tags])
        detection_prompt = prompt_from_transcript(combined)
        if detection_prompt:
            detections = vision_detector.detect(
                pil,
                labels=[detection_prompt],
                threshold=threshold,
                text_threshold=text_threshold,
            )
            screen = guidance["screen_target"]
            detection = select_near_point(
                detections,
                screen["x"],
                screen["y"],
            )

    # Keep the temporal fallback window anchored to the last successfully
    # completed tracking response, not to the earlier pose-estimation moment.
    pose_smoother.refresh_recent_pose(tracking_key)

    return {
        "scene_id": scene_id,
        "target_entity": target.model_dump(mode="json"),
        "pose": pose_to_dict(pose),
        "raw_pose": pose_to_dict(raw_pose) if raw_pose is not None else None,
        "pose_source": pose_source,
        "reference_search": (
            "primary_only" if fast_tracking_search else "exhaustive"
        ),
        "smoothing": smoothing,
        "guidance": guidance,
        "mode": "bbox" if detection is not None else "arrow",
        "detection_prompt": detection_prompt,
        "detection": detection.model_dump() if detection else None,
    }


class RelationMemoryRequest(BaseModel):
    parent_entity_id: str
    transcript: str
    relation: str = "inside"
    kind: str = "object"


@app.post("/scenes/{scene_id}/relations/remember", response_model=SpatialEntity)
async def remember_entity_relation(
    scene_id: str,
    request: RelationMemoryRequest,
) -> SpatialEntity:
    if db.get_scene(scene_id) is None:
        raise HTTPException(status_code=404, detail="Scene not found")

    parent = db.get_entity(request.parent_entity_id)
    if parent is None or parent.scene_id != scene_id:
        raise HTTPException(status_code=404, detail="Parent entity not found in Scene")

    if request.relation not in {"inside", "on", "at", "near"}:
        raise HTTPException(status_code=422, detail="Unsupported relation")
    if request.kind not in {
        "object", "furniture", "container", "room", "place", "region", "unknown"
    }:
        raise HTTPException(status_code=422, detail="Unsupported entity kind")

    observation = Observation(
        scene_id=scene_id,
        position=parent.position,
        kind=request.kind,
        parent_id=parent.id,
        relation_to_parent=request.relation,
        transcript=request.transcript,
    )
    return await service.ingest_observation(observation)


def _ancestor_chain(entity: SpatialEntity, max_depth: int = 6) -> list[SpatialEntity]:
    chain: list[SpatialEntity] = []
    seen = {entity.id}
    current = entity
    for _ in range(max_depth):
        if not current.parent_id or current.parent_id in seen:
            break
        parent = db.get_entity(current.parent_id)
        if parent is None:
            break
        chain.append(parent)
        seen.add(parent.id)
        current = parent
    return chain
