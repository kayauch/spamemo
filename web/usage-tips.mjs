const tips = [
  '「充電ケーブルどこ？」のような質問でも探せます。',
  '検索で候補が複数出たら、探している物を選んでください。',
  '「マップ」で、部屋の形と今見ている位置を確認できます。',
  '「俯瞰へ」で、部屋全体を上から見渡せます。',
  '「登録」で収納を選ぶと、その場所が青く光ります。',
  '物の場所は「上から何段目・左右」まで記録できます。',
  '物の写真やメモも一緒に残せます。',
  '「部屋の修正」から、不足している場所を順番に撮影できます。',
  '追加撮影はスマホを横向きにして撮りましょう。',
  '壁を撮るときは、床との境目や部屋の角も入れましょう。',
  '棚や箱は正面だけでなく、側面も撮ると奥行きが伝わります。',
  '少しずつ位置を変え、前の写真と重なる部分を入れて撮りましょう。'
];

// Keep hints user-controlled: nothing changes while someone is reading.
const storageKey = 'spamemo.last-usage-tip';
let last = '';
try { last = sessionStorage.getItem(storageKey) || ''; } catch {}
let remaining = [];
function nextTip() {
  if (!remaining.length) {
    remaining = [...tips];
    for (let i = remaining.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
    }
    if (remaining.at(-1) === last) {
      [remaining[0], remaining[remaining.length - 1]] = [remaining.at(-1), remaining[0]];
    }
  }
  last = remaining.pop();
  try { sessionStorage.setItem(storageKey, last); } catch {}
  return last;
}

for (const container of document.querySelectorAll('[data-usage-tips]')) {
  const text = container.querySelector('[data-tip-text]');
  const button = container.querySelector('[data-tip-next]');
  if (!text) continue;
  const showNext = () => { text.textContent = nextTip(); };
  showNext();
  button?.addEventListener('click', showNext);
}
