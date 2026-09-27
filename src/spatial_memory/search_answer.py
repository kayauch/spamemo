from .models import SpatialEntity


def recorded_location(entity: SpatialEntity, get_entity) -> str:
    parts = []
    current = entity
    seen = {entity.id}
    relations = {"inside": "の中", "on": "の上", "near": "の近く", "at": "の場所"}
    for _ in range(6):
        if not current.parent_id or current.parent_id in seen:
            break
        parent = get_entity(current.parent_id)
        if parent is None or parent.scene_id != entity.scene_id:
            break
        parts.append(f"「{parent.name}」{relations.get(current.relation_to_parent, 'の場所')}")
        seen.add(parent.id)
        current = parent
    parts.reverse()
    if entity.metadata.get("shelf_item"):
        level = entity.metadata.get("level")
        if isinstance(level, int) and not isinstance(level, bool) and level > 0:
            parts.append(f"上から{level}段目")
        side = entity.metadata.get("side")
        if side in ("左", "中央", "右"):
            parts.append(side)
    return "・".join(parts)


def entity_answer(entity: SpatialEntity, get_entity) -> str:
    location = recorded_location(entity, get_entity)
    if location:
        return f"「{entity.name}」は{location}にあると記録されています。"
    return f"「{entity.name}」の位置が記録されています。3D上の目印を確認してください。"


def search_answer(result, get_entity) -> str:
    if result.status == "ambiguous":
        return "候補が複数あり、どれか判断できません。写真や収納場所から選んでください。"
    if result.target is None:
        return "登録された情報からは見つかりませんでした。名前を変えて検索するか、物を登録してください。"
    return entity_answer(result.target, get_entity)
