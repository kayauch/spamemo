// Only an authoritative scene list may resolve an old link to a parent room.
export function resolveCaptureEntry(scenes, sceneId) {
  if (!Array.isArray(scenes)) throw new Error('部屋一覧を確認できませんでした。もう一度お試しください。');
  const scene = scenes.find(item => item.id === sceneId);
  return {sceneId: scene?.id ?? null, missing: Boolean(sceneId && !scene)};
}

export function isMissingCaptureParent(error) {
  return error?.status === 422 && error.message === '追加先の部屋が見つかりません';
}
