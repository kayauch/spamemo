from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Vec3(BaseModel):
    x: float
    y: float
    z: float

class BoundingBox3D(BaseModel):
    center: Vec3
    size: Vec3

EntityKind = Literal["object","furniture","container","room","place","region","unknown"]
SourceKind = Literal["human","vision","ai","fused"]

class Scene(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str | None = None
    point_cloud_path: str | None = None
    spatial_model_path: str | None = None
    reconstruction_npz: str | None = None
    frames_dir: str | None = None
    source_video_path: str | None = None
    parent_scene_id: str | None = None
    capture_manifest_path: str | None = None
    reconstruction_model: str | None = None
    frame_count: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=utc_now)

class SpatialEntity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    scene_id: str
    name: str
    kind: EntityKind = "unknown"
    position: Vec3
    bbox: BoundingBox3D | None = None
    parent_id: str | None = None
    relation_to_parent: Literal["inside", "on", "at", "near"] | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: SourceKind = "human"
    last_observed_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)

class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    scene_id: str
    position: Vec3
    kind: EntityKind = "object"
    bbox: BoundingBox3D | None = None
    parent_id: str | None = None
    relation_to_parent: Literal["inside", "on", "at", "near"] | None = None
    transcript: str | None = None
    detected_label: str | None = None
    detection_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    image_path: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    detection_bbox_xyxy: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

class SearchHit(BaseModel):
    entity: SpatialEntity
    score: float
    reasons: list[str] = Field(default_factory=list)

class QueryResult(BaseModel):
    query: str
    scene_id: str
    hits: list[SearchHit]
    target: SpatialEntity | None = None
    status: Literal["matched", "ambiguous", "not_found"] = "not_found"
    candidates: list[SearchHit] = Field(default_factory=list)
    search_mode: Literal["hybrid", "lexical", "lexical_fallback"] = "lexical"


class NavNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    scene_id: str
    name: str
    position: Vec3


class NavEdge(BaseModel):
    scene_id: str
    source_id: str
    target_id: str
    cost: float | None = Field(default=None, gt=0)


class RouteResult(BaseModel):
    scene_id: str
    target_entity_id: str
    node_ids: list[str]
    points: list[Vec3]
    total_cost: float
