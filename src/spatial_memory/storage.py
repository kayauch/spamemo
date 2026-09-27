from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import NavEdge, NavNode, Observation, Scene, SpatialEntity
from .object_corrections import (
    ObjectCorrection, ObjectCorrectionUpdate, apply_object_names, hidden_entity_ids,
)


class SpatialMemoryDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deleted_scenes (
                    scene_id TEXT PRIMARY KEY,
                    deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS object_corrections (
                    scene_id TEXT NOT NULL,
                    source_prediction TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(scene_id, source_prediction, object_id)
                );
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entities_scene ON entities(scene_id);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observations_scene ON observations(scene_id);
                CREATE TABLE IF NOT EXISTS nav_nodes (
                    id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nav_nodes_scene ON nav_nodes(scene_id);
                CREATE TABLE IF NOT EXISTS nav_edges (
                    scene_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(scene_id, source_id, target_id)
                );
                """
            )

    def upsert_scene(self, scene: Scene) -> Scene:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO scenes(id,payload) SELECT ?,? "
                "WHERE NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (scene.id, scene.model_dump_json(), scene.id),
            )
        return scene

    def delete_scene(self, scene_id: str) -> bool:
        """Hide one room permanently, retaining assets shared by other revisions.

        A tombstone also prevents a late reconstruction update from restoring the
        card. Repeating a successful deletion is safe; an unknown ID returns False.
        """
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO deleted_scenes(scene_id) "
                "SELECT id FROM scenes WHERE id=?",
                (scene_id,),
            )
            return conn.execute(
                "SELECT 1 FROM deleted_scenes WHERE scene_id=?", (scene_id,)
            ).fetchone() is not None

    def is_scene_deleted(self, scene_id: str) -> bool:
        with self.connect() as conn:
            return conn.execute(
                "SELECT 1 FROM deleted_scenes WHERE scene_id=?", (scene_id,)
            ).fetchone() is not None

    def get_scene(self, scene_id: str) -> Scene | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM scenes WHERE id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=scenes.id)",
                (scene_id,),
            ).fetchone()
        return Scene.model_validate_json(row["payload"]) if row else None

    def list_scenes(self) -> list[Scene]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM scenes "
                "WHERE NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=scenes.id) "
                "ORDER BY rowid DESC"
            ).fetchall()
        return [Scene.model_validate_json(r["payload"]) for r in rows]

    def upsert_entity(self, entity: SpatialEntity) -> SpatialEntity:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            corrections = self._object_corrections(conn, entity.scene_id)
            if not any(correction.name is not None for correction in corrections):
                self._write_entity(conn, entity)
                return entity
            originals = self._scene_entities(conn, entity.scene_id)
            entities = [old for old in originals if old.id != entity.id] + [entity]
            corrected = apply_object_names(entities, corrections)
            previous = {old.id: old for old in originals}
            for item in corrected:
                if item.id == entity.id or item != previous.get(item.id):
                    self._write_entity(conn, item)
                if item.id == entity.id:
                    entity = item
        return entity

    @staticmethod
    def _write_entity(conn: sqlite3.Connection, entity: SpatialEntity) -> None:
        conn.execute(
            "INSERT INTO entities(id,scene_id,name,payload) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET scene_id=excluded.scene_id,name=excluded.name,payload=excluded.payload",
            (entity.id, entity.scene_id, entity.name, entity.model_dump_json()),
        )

    @staticmethod
    def _object_corrections(conn: sqlite3.Connection, scene_id: str) -> list[ObjectCorrection]:
        rows = conn.execute(
            "SELECT payload FROM object_corrections WHERE scene_id=? ORDER BY rowid", (scene_id,)
        ).fetchall()
        return [ObjectCorrection.model_validate_json(row['payload']) for row in rows]

    def list_object_corrections(self, scene_id: str, source_prediction: str) -> list[ObjectCorrection]:
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM deleted_scenes WHERE scene_id=?", (scene_id,)).fetchone():
                return []
            return [correction for correction in self._object_corrections(conn, scene_id)
                    if correction.source_prediction == source_prediction]

    def update_object_correction(self, scene_id: str, update: ObjectCorrectionUpdate) -> ObjectCorrection:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute(
                "SELECT 1 FROM scenes WHERE id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=scenes.id)",
                (scene_id,),
            ).fetchone():
                raise ValueError('部屋が見つかりません')
            corrections = self._object_corrections(conn, scene_id)
            current = next((item for item in corrections
                            if item.source_prediction == update.source_prediction
                            and item.object_id == update.object_id),
                           ObjectCorrection(source_prediction=update.source_prediction, object_id=update.object_id))
            correction = current.model_copy(update=update.model_dump(exclude_unset=True))
            conn.execute(
                "INSERT INTO object_corrections(scene_id,source_prediction,object_id,payload) VALUES(?,?,?,?) "
                "ON CONFLICT(scene_id,source_prediction,object_id) DO UPDATE SET payload=excluded.payload",
                (scene_id, correction.source_prediction, correction.object_id, correction.model_dump_json()),
            )
            originals = self._scene_entities(conn, scene_id)
            corrected = apply_object_names(originals, self._object_corrections(conn, scene_id))
            for old, new in zip(originals, corrected):
                if old != new:
                    self._write_entity(conn, new)
        return correction

    @staticmethod
    def _scene_entities(conn: sqlite3.Connection, scene_id: str) -> list[SpatialEntity]:
        rows = conn.execute(
            "SELECT payload FROM entities WHERE scene_id=? "
            "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=entities.scene_id) "
            "ORDER BY rowid DESC", (scene_id,)
        ).fetchall()
        return [SpatialEntity.model_validate_json(row['payload']) for row in rows]

    def _visible_entities(self, conn: sqlite3.Connection, scene_id: str) -> list[SpatialEntity]:
        entities = self._scene_entities(conn, scene_id)
        hidden = hidden_entity_ids(entities, self._object_corrections(conn, scene_id))
        return [entity for entity in entities if entity.id not in hidden]

    def get_entity(self, entity_id: str) -> SpatialEntity | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT scene_id,payload FROM entities WHERE id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=entities.scene_id)",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            if not any(correction.hidden for correction in self._object_corrections(conn, row['scene_id'])):
                return SpatialEntity.model_validate_json(row['payload'])
            return next((entity for entity in self._visible_entities(conn, row['scene_id'])
                         if entity.id == entity_id), None)

    def list_entities(self, scene_id: str) -> list[SpatialEntity]:
        with self.connect() as conn:
            return self._visible_entities(conn, scene_id)

    def delete_entity(self, entity_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
        return cur.rowcount > 0

    def add_observation(self, observation: Observation) -> Observation:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO observations(id,scene_id,payload) VALUES(?,?,?)",
                (observation.id, observation.scene_id, observation.model_dump_json()),
            )
        return observation

    def list_observations(self, scene_id: str) -> list[Observation]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM observations WHERE scene_id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=observations.scene_id) "
                "ORDER BY rowid DESC", (scene_id,)
            ).fetchall()
        return [Observation.model_validate_json(r["payload"]) for r in rows]

    def upsert_nav_node(self, node: NavNode) -> NavNode:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO nav_nodes(id,scene_id,payload) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET scene_id=excluded.scene_id,payload=excluded.payload",
                (node.id, node.scene_id, node.model_dump_json()),
            )
        return node

    def list_nav_nodes(self, scene_id: str) -> list[NavNode]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM nav_nodes WHERE scene_id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=nav_nodes.scene_id) "
                "ORDER BY rowid", (scene_id,)
            ).fetchall()
        return [NavNode.model_validate_json(r["payload"]) for r in rows]

    def upsert_nav_edge(self, edge: NavEdge) -> NavEdge:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO nav_edges(scene_id,source_id,target_id,payload) VALUES(?,?,?,?) "
                "ON CONFLICT(scene_id,source_id,target_id) DO UPDATE SET payload=excluded.payload",
                (edge.scene_id, edge.source_id, edge.target_id, edge.model_dump_json()),
            )
        return edge

    def list_nav_edges(self, scene_id: str) -> list[NavEdge]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM nav_edges WHERE scene_id=? "
                "AND NOT EXISTS (SELECT 1 FROM deleted_scenes WHERE scene_id=nav_edges.scene_id)",
                (scene_id,)
            ).fetchall()
        return [NavEdge.model_validate_json(r["payload"]) for r in rows]
