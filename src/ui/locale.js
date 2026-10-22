const storageKey = 'local-yolo-trainer.language';
let current = 'zh';
try {
  const saved = globalThis.localStorage?.getItem(storageKey);
  current = saved === 'en' || saved === 'zh' ? saved : /^zh/i.test(globalThis.navigator?.language ?? 'zh') ? 'zh' : 'en';
} catch { /* Browser preferences can be unavailable in private contexts. */ }

export function language() { return current; }
export function tr(zh, en) { return current === 'zh' ? zh : en; }
export function applyLocale(root = globalThis.document) {
  if (!root) return;
  if (root === globalThis.document) root.documentElement.lang = current === 'zh' ? 'zh-CN' : 'en';
  for (const node of root.querySelectorAll('[data-zh][data-en]')) node.textContent = current === 'zh' ? node.dataset.zh : node.dataset.en;
  for (const node of root.querySelectorAll('[data-zh-label][data-en-label]')) node.setAttribute('aria-label', current === 'zh' ? node.dataset.zhLabel : node.dataset.enLabel);
  const toggle = root.querySelector('#language-toggle');
  if (toggle) toggle.textContent = current === 'zh' ? 'EN' : '中文';
}
export function setLanguage(value) {
  if (value !== 'zh' && value !== 'en') throw new TypeError('Language must be zh or en.');
  const changed = current !== value;
  current = value;
  try { globalThis.localStorage?.setItem(storageKey, value); } catch { /* Optional preference only. */ }
  applyLocale();
  if (changed && globalThis.window) window.dispatchEvent(new CustomEvent('languagechange', { detail: { language: value } }));
  return value;
}
applyLocale();
