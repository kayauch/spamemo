const labels = {
  cabinet:'キャビネット',shelf:'棚',bookshelf:'本棚',locker:'ロッカー',wardrobe:'収納',box:'収納箱',
  desk:'机',bed:'ベッド',sofa:'ソファ',chair:'椅子',dining_chair:'椅子',coffee_table:'テーブル',
  table:'テーブル',curtain:'カーテン',illumination:'照明',chandelier:'照明',painting:'壁の飾り',
  wall_decoration:'壁の飾り',pillow:'枕',cushion:'クッション',tub:'浴槽',plant:'植物'
};

export function prepareCorrectedRoom(raw, corrections = [], entities = []) {
  const records = new Map(corrections.filter(c => c.source_prediction === raw.source_prediction).map(c => [String(c.object_id), c]));
  const objects = new Map(), counts = {};
  function add(id, label) {
    if (id == null || objects.has(String(id))) return;
    counts[label] = (counts[label] || 0) + 1;
    const saved = entities.find(e => e.metadata?.source_prediction === raw.source_prediction && String(e.metadata?.source_object_id) === String(id));
    const correction = records.get(String(id));
    objects.set(String(id), {id:String(id), name:correction?.name || saved?.name || `${labels[label] || label || '家具'} ${counts[label]}`, hidden:correction?.hidden === true});
  }
  for (const s of raw.surfaces || []) add(s.object_id, s.object_label);
  for (const b of raw.furniture || []) if (!b.hidden) add(b.id, b.label);
  const hidden = new Set([...objects.values()].filter(o => o.hidden).map(o => o.id));
  const keep = id => id == null || !hidden.has(String(id));
  const data = {...raw,
    surfaces:(raw.surfaces || []).filter(s => keep(s.object_id)),
    furniture:(raw.furniture || []).filter(b => !b.hidden && keep(b.id)),
    object_names:Object.fromEntries([...objects].map(([id, o]) => [id, o.name])),
    current_entities:entities,
    navigation:raw.navigation ? {...raw.navigation, colliders:(raw.navigation.colliders || []).filter(b => keep(b.id))} : raw.navigation
  };
  return {data, objects:[...objects.values()]};
}

export async function loadObjectCorrections(raw) {
  const sceneId = raw.transform?.scene_id, prediction = raw.source_prediction;
  if (!sceneId || !prediction) {
    const result = prepareCorrectedRoom(raw);
    delete result.data.current_entities;
    return {...result, available:false};
  }
  const endpoint = '/scenes/' + encodeURIComponent(sceneId) + '/object-corrections';
  async function request(url, options) {
    const response = await fetch(url, {cache:'no-store', ...options});
    if (!response.ok) throw Error('認識の修正を読み込み・保存できませんでした。もう一度お試しください。');
    return response.json();
  }
  const [corrections, entities] = await Promise.all([
    request(endpoint + '?' + new URLSearchParams({source_prediction:prediction})),
    request('/scenes/' + encodeURIComponent(sceneId) + '/entities')
  ]);
  if (!Array.isArray(corrections) || !Array.isArray(entities)) throw Error('認識の修正データを確認できませんでした。');
  return {...prepareCorrectedRoom(raw, corrections, entities), available:true,
    async save(id, patch) {
      await request(endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_prediction:prediction, object_id:String(id), ...patch})});
      // Rebuild every consumer together: model, collisions, capture queue and saved items.
      location.reload();
    }
  };
}
