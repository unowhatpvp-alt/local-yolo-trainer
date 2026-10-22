const storageKey = 'local-yolo-trainer.theme';
let preference = 'system';
let media;
try {
  const saved = globalThis.localStorage?.getItem(storageKey);
  if (['light', 'dark', 'system'].includes(saved)) preference = saved;
} catch { /* Appearance preferences must not block the app. */ }
export function theme() { return preference; }
export function resolvedTheme() { return preference === 'system' ? media?.matches ? 'dark' : 'light' : preference; }
function apply() {
  if (!globalThis.document) return;
  const value = resolvedTheme();
  document.documentElement.dataset.theme = value;
  document.documentElement.style.colorScheme = value;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', value === 'dark' ? '#141414' : '#f7f7f5');
  const button = document.getElementById('theme-toggle');
  if (button) {
    button.textContent = preference === 'system' ? '◐' : value === 'dark' ? '☾' : '☼';
    button.dataset.themePreference = preference;
    button.title = `${preference === 'system' ? '跟随系统 / System' : preference === 'dark' ? '深色 / Dark' : '浅色 / Light'} · 切换外观 / Change appearance`;
  }
}
export function setTheme(value) {
  if (!['light', 'dark', 'system'].includes(value)) throw new TypeError('Theme must be light, dark or system.');
  preference = value;
  try { globalThis.localStorage?.setItem(storageKey, value); } catch { /* Optional preference only. */ }
  apply();
  globalThis.window?.dispatchEvent(new CustomEvent('themechange', { detail: { preference, resolved: resolvedTheme() } }));
  return value;
}
export function cycleTheme() { return setTheme(({ system: 'light', light: 'dark', dark: 'system' })[preference]); }
if (globalThis.window?.matchMedia) {
  media = window.matchMedia('(prefers-color-scheme: dark)');
  media.addEventListener?.('change', () => { if (preference === 'system') apply(); });
}
apply();
