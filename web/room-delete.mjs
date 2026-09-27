// Room deletion is deliberate: confirm the named room, then wait for the server.
export function installRoomDeletion({onDeleted, focusAfterDelete}) {
  const make = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const dialog = make('dialog', null, 'sm-dialog room-delete-dialog');
  dialog.id = 'roomDeleteDialog';
  dialog.setAttribute('aria-labelledby', 'roomDeleteTitle');
  dialog.setAttribute('aria-describedby', 'roomDeleteDescription');
  const title = make('h2', 'この部屋を削除しますか？');
  title.id = 'roomDeleteTitle';
  const name = make('strong', '', 'room-delete-name');
  const date = make('p', '', 'room-delete-date');
  const description = make('p', 'この部屋と登録したものを一覧から削除します。元の写真・動画は残ります。');
  description.id = 'roomDeleteDescription';
  const error = make('p', '', 'room-delete-error');
  error.setAttribute('role', 'alert');
  error.hidden = true;
  const actions = make('div', null, 'room-delete-actions');
  const cancel = make('button', 'キャンセル', 'secondary-btn');
  const confirm = make('button', '削除する', 'room-delete-confirm');
  cancel.type = confirm.type = 'button';
  cancel.autofocus = true;
  actions.append(cancel, confirm);
  dialog.append(title, name, date, description, error, actions);
  const notice = make('p', '', 'room-delete-notice');
  notice.setAttribute('role', 'status');
  notice.setAttribute('aria-live', 'polite');
  notice.hidden = true;
  document.getElementById('roomsStatus').after(notice);
  document.body.append(dialog);
  let selected = null, opener = null, busy = false;
  cancel.addEventListener('click', () => { if (!busy) dialog.close(); });
  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  dialog.addEventListener('close', () => {
    if (opener?.isConnected) opener.focus();
    selected = null;
    opener = null;
  });
  confirm.addEventListener('click', async () => {
    if (busy || !selected) return;
    const scene = selected;
    busy = true;
    cancel.disabled = confirm.disabled = true;
    confirm.textContent = '削除中…';
    dialog.setAttribute('aria-busy', 'true');
    error.hidden = true;
    try {
      const response = await fetch('/scenes/' + encodeURIComponent(scene.id), {method: 'DELETE'});
      // A room removed in another tab is already in the requested state.
      if (!response.ok && response.status !== 404) throw new Error('delete failed');
      onDeleted(scene.id);
      notice.textContent = '「' + (scene.name || '名称未設定') + '」を削除しました。';
      notice.hidden = false;
      opener = null;
      dialog.close();
      focusAfterDelete?.();
    } catch {
      error.textContent = '削除できませんでした。通信状態を確認して、もう一度お試しください。';
      error.hidden = false;
    } finally {
      busy = false;
      cancel.disabled = confirm.disabled = false;
      confirm.textContent = '削除する';
      dialog.removeAttribute('aria-busy');
    }
  });
  return (scene, trigger) => {
    if (busy || dialog.open) return;
    selected = scene;
    opener = trigger;
    name.textContent = scene.name || '名称未設定';
    const created = scene.created_at ? new Date(scene.created_at) : null;
    date.textContent = created && !Number.isNaN(created.valueOf()) ? '作成日時：' + created.toLocaleString('ja-JP') : '';
    date.hidden = !date.textContent;
    error.hidden = true;
    dialog.showModal();
    cancel.focus();
  };
}
