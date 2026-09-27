from __future__ import annotations

import heapq
import math

from .models import NavEdge, NavNode, RouteResult, Vec3
from .storage import SpatialMemoryDB


def distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def nearest_node(position: Vec3, nodes: list[NavNode]) -> NavNode:
    if not nodes:
        raise ValueError("Navigation graph has no nodes")
    return min(nodes, key=lambda node: distance(position, node.position))


def _edge_cost(edge: NavEdge, nodes: dict[str, NavNode]) -> float:
    if edge.cost is not None:
        return edge.cost
    return distance(nodes[edge.source_id].position, nodes[edge.target_id].position)


def shortest_path(nodes: list[NavNode], edges: list[NavEdge], start_id: str, goal_id: str) -> tuple[list[str], float]:
    by_id = {n.id: n for n in nodes}
    graph: dict[str, list[tuple[str, float]]] = {n.id: [] for n in nodes}
    for edge in edges:
        if edge.source_id not in by_id or edge.target_id not in by_id:
            continue
        cost = _edge_cost(edge, by_id)
        graph[edge.source_id].append((edge.target_id, cost))
        graph[edge.target_id].append((edge.source_id, cost))

    queue: list[tuple[float, str]] = [(0.0, start_id)]
    best = {start_id: 0.0}
    prev: dict[str, str] = {}
    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal_id:
            break
        if cost > best.get(current, float("inf")):
            continue
        for nxt, edge_cost in graph.get(current, []):
            new_cost = cost + edge_cost
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                prev[nxt] = current
                heapq.heappush(queue, (new_cost, nxt))

    if goal_id not in best:
        raise ValueError("No route to target")
    path = [goal_id]
    while path[-1] != start_id:
        path.append(prev[path[-1]])
    path.reverse()
    return path, best[goal_id]


def route_to_entity(
    db: SpatialMemoryDB,
    scene_id: str,
    start: Vec3,
    target_entity_id: str,
) -> RouteResult:
    entity = db.get_entity(target_entity_id)
    if entity is None or entity.scene_id != scene_id:
        raise ValueError("Target entity not found in scene")
    nodes = db.list_nav_nodes(scene_id)
    edges = db.list_nav_edges(scene_id)
    if not nodes:
        return RouteResult(
            scene_id=scene_id,
            target_entity_id=target_entity_id,
            node_ids=[],
            points=[start, entity.position],
            total_cost=distance(start, entity.position),
        )

    start_node = nearest_node(start, nodes)
    goal_node = nearest_node(entity.position, nodes)
    node_ids, graph_cost = shortest_path(nodes, edges, start_node.id, goal_node.id)
    by_id = {n.id: n for n in nodes}
    points = [start] + [by_id[node_id].position for node_id in node_ids] + [entity.position]
    total = distance(start, start_node.position) + graph_cost + distance(goal_node.position, entity.position)
    return RouteResult(
        scene_id=scene_id,
        target_entity_id=target_entity_id,
        node_ids=node_ids,
        points=points,
        total_cost=total,
    )
