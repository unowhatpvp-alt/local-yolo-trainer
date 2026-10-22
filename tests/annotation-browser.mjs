import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const harness = `<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/src/styles.css"></head><body><main id="app"><div id="editor"></div></main><script type="module">
import { createAnnotationEditor } from '/src/ui/annotation-editor.js';
import { setLanguage } from '/src/ui/locale.js';
import { setTheme } from '/src/ui/theme.js';
window.state = { changes: [], status: null, revoked: [], created: [] };
const originalCreate = URL.createObjectURL.bind(URL), originalRevoke = URL.revokeObjectURL.bind(URL);
URL.createObjectURL = (...args) => { const url = originalCreate(...args); state.created.push(url); return url; };
URL.revokeObjectURL = (url) => { state.revoked.push(url); return originalRevoke(url); };
window.setLanguage = setLanguage;
window.setTheme = setTheme;
window.editor = createAnnotationEditor({ root: document.querySelector('#editor'), onChange: boxes => state.changes.push(boxes), onStatus: status => state.status = status });
const fixture = document.createElement('canvas'); fixture.width = 640; fixture.height = 360;
const context = fixture.getContext('2d'); context.fillStyle = '#d9d9d4'; context.fillRect(0,0,640,360); context.fillStyle = '#383834'; context.fillRect(80,50,220,130);
window.fixture = await new Promise(resolve => fixture.toBlob(resolve, 'image/png'));
window.loadFixture = async (boxes = []) => editor.load({ blob: window.fixture, width: 640, height: 360, boxes, classes: ['Apple', 'Berry'] });
await loadFixture(); window.ready = true;
</script></body></html>`;

async function withBrowser(run) {
  const server = createServer(async (request, response) => {
    if (request.url === '/editor-harness') { response.setHeader('Content-Type', 'text/html'); response.end(harness); return; }
    if (request.url === '/shell-harness') {
      const shell = await readFile(resolve(projectRoot, 'index.html'), 'utf8');
      const fixtureScript = harness.match(/<script type="module">([\s\S]*?)<\/script>/)[0];
      response.setHeader('Content-Type', 'text/html');
      response.end(shell.replace('<main id="app" tabindex="-1" aria-live="off"></main>', '<main id="app" tabindex="-1"><div id="editor"></div></main>').replace('<script type="module" src="/src/app.js"></script>', `<link rel="stylesheet" href="/src/styles.css">${fixtureScript}`));
      return;
    }
    try {
      const path = resolve(projectRoot, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
      if (!path.startsWith(resolve(projectRoot) + sep)) { response.writeHead(403).end(); return; }
      const content = await readFile(path);
      response.setHeader('Content-Type', ({ '.js': 'text/javascript', '.css': 'text/css' })[extname(path)] ?? 'application/octet-stream');
      response.end(content);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    await run(browser, `http://127.0.0.1:${server.address().port}/editor-harness`);
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
}

async function imagePoint(page, x, y) {
  return page.evaluate(({ x, y }) => {
    const matrix = document.querySelector('.annotation-canvas svg > g').getScreenCTM();
    const point = new DOMPoint(x, y).matrixTransform(matrix);
    return { x: point.x, y: point.y };
  }, { x, y });
}
async function draw(page, start, end) {
  const a = await imagePoint(page, ...start);
  const b = await imagePoint(page, ...end);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 8 });
  await page.mouse.up();
}
async function lastBoxes(page) { return page.evaluate(() => state.changes.at(-1)); }
function near(actual, expected, tolerance = .1) { assert.ok(Math.abs(actual - expected) < tolerance, `${actual} ≠ ${expected}`); }

test('real editor gestures preserve original pixels through zoom, resize, move, class, history and safe delete', { timeout: 60000 }, async () => withBrowser(async (browser, url) => {
  const page = await browser.newPage({ viewport: { width: 1100, height: 850 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => window.ready);
  await draw(page, [80, 50], [300, 180]);
  let boxes = await lastBoxes(page);
  assert.equal(boxes.length, 1);
  for (const [key, expected] of Object.entries({ x: 80, y: 50, width: 220, height: 130 })) near(boxes[0][key], expected);

  const handle = await imagePoint(page, 300, 115);
  const larger = await imagePoint(page, 380, 115);
  await page.mouse.move(handle.x + 5, handle.y); // Grab away from exact handle center; no jumping.
  await page.mouse.down();
  await page.mouse.move(larger.x + 5, larger.y, { steps: 4 });
  await page.mouse.up();
  boxes = await lastBoxes(page);
  near(boxes[0].x, 80);
  near(boxes[0].width, 300);
  await page.locator('.editor-class-select').selectOption('1');
  assert.equal((await lastBoxes(page))[0].classId, 1);
  await page.locator('[data-action="move-selected"]').click();
  const a = await imagePoint(page, 200, 120);
  const b = await imagePoint(page, 250, 150);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 4 });
  await page.mouse.up();
  near((await lastBoxes(page))[0].x, 130);
  near((await lastBoxes(page))[0].y, 80);
  await page.locator('[data-action="undo"]').click();
  near((await lastBoxes(page))[0].x, 80);
  await page.locator('[data-action="redo"]').click();
  near((await lastBoxes(page))[0].x, 130);

  await page.evaluate(() => editor.load({ blob: fixture, width: 640, height: 360, boxes: [], classes: ['Apple', 'Berry'], activeClassIndex: 1 }));
  await page.locator('[data-action="zoom-in"]').click();
  await page.locator('[data-action="zoom-in"]').click();
  await draw(page, [200, 110], [300, 190]);
  boxes = await lastBoxes(page);
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0].classId, 1);
  for (const [key, expected] of Object.entries({ x: 200, y: 110, width: 100, height: 80 })) near(boxes[0][key], expected);
  await page.locator('[data-action="delete"]').click();
  assert.equal((await lastBoxes(page)).length, 1, 'first click must only arm deletion');
  await page.locator('[data-action="delete"]').click();
  assert.equal((await lastBoxes(page)).length, 0);
  await page.locator('[data-action="undo"]').click();
  assert.equal((await lastBoxes(page)).length, 1);

  // Exercise each actual SVG handle, not only the pure geometry helper.
  const handlePositions = { n: [300, 100], s: [300, 240], w: [200, 170], e: [400, 170], nw: [200, 100], ne: [400, 100], sw: [200, 240], se: [400, 240] };
  for (const [handleName, position] of Object.entries(handlePositions)) {
    await page.evaluate(() => loadFixture([{ id: 'eight-handles', classId: 0, x: 200, y: 100, width: 200, height: 140 }]));
    await page.locator('[data-select-box="eight-handles"]').click();
    const start = await imagePoint(page, ...position);
    const dx = handleName.includes('w') ? -20 : handleName.includes('e') ? 20 : 0;
    const dy = handleName.includes('n') ? -20 : handleName.includes('s') ? 20 : 0;
    const end = await imagePoint(page, position[0] + dx, position[1] + dy);
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(end.x, end.y, { steps: 3 });
    await page.mouse.up();
    const actual = (await lastBoxes(page))[0];
    near(actual.x, 200 + Math.min(0, dx));
    near(actual.y, 100 + Math.min(0, dy));
    near(actual.width, 200 + Math.abs(dx));
    near(actual.height, 140 + Math.abs(dy));
  }

  const resources = await page.evaluate(async () => {
    await loadFixture();
    const old = state.created[0];
    editor.destroy();
    return { oldRevoked: state.revoked.includes(old), allRevoked: state.created.every(url => state.revoked.includes(url)), editorChildren: document.querySelector('#editor').children.length };
  });
  assert.deepEqual(resources, { oldRevoked: true, allRevoked: true, editorChildren: 0 });
  assert.deepEqual(errors, []);
}));

test('actual shared shell fits phone, landscape, tablet and desktop in both languages and themes', { timeout: 60000 }, async () => withBrowser(async (browser, url) => {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true });
  await page.goto(url.replace('/editor-harness', '/shell-harness'));
  await page.waitForFunction(() => window.ready);
  for (const lang of ['zh', 'en']) {
    await page.evaluate(language => setLanguage(language), lang);
    for (const theme of ['light', 'dark']) {
      await page.evaluate(value => setTheme(value), theme);
      for (const [width, height] of [[375, 650], [390, 844], [844, 390], [768, 1024], [1024, 768], [1280, 900]]) {
        await page.setViewportSize({ width, height });
        const layout = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: innerWidth, brand: document.querySelector('.brand').getBoundingClientRect().toJSON(), controls: [...document.querySelectorAll('header button, nav button')].map(node => { const b = node.getBoundingClientRect(); return { x: b.x, right: b.right, top: b.top, bottom: b.bottom, width: b.width, height: b.height, text: node.textContent }; }), theme: document.documentElement.dataset.theme, language: document.documentElement.lang }));
        assert.ok(layout.scroll <= layout.viewport, `${lang}/${theme} shell overflow at ${width}×${height}`);
        for (const control of layout.controls) {
          assert.ok(control.x >= 0 && control.right <= width, `offscreen ${control.text} at ${width}×${height}`);
          assert.ok(control.width >= 44 && control.height >= 44, `small ${control.text} at ${width}×${height}`);
        }
        const regions = [{ ...layout.brand, text: 'brand' }, ...layout.controls];
        for (let a = 0; a < regions.length; a++) for (let b = a + 1; b < regions.length; b++) {
          const first = regions[a], second = regions[b];
          const overlaps = first.x < second.right - .5 && second.x < first.right - .5 && first.top < second.bottom - .5 && second.top < first.bottom - .5;
          assert.equal(overlaps, false, `${first.text}/${second.text} overlap at ${width}×${height}: ${JSON.stringify({ first, second })}`);
        }
        assert.equal(layout.theme, theme);
        assert.equal(layout.language, lang === 'zh' ? 'zh-CN' : 'en');
      }
    }
  }
}));

test('trusted mobile touch draws original pixels, pinches without extra boxes, and layouts fit six viewports', { timeout: 60000 }, async () => withBrowser(async (browser, url) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  await page.goto(url);
  await page.waitForFunction(() => window.ready);
  const cdp = await context.newCDPSession(page);
  const a = await imagePoint(page, 60, 30);
  const b = await imagePoint(page, 350, 220);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: a.x, y: a.y, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: b.x, y: b.y, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await page.waitForFunction(() => state.changes.length === 1);
  const box = (await lastBoxes(page))[0];
  for (const [key, expected] of Object.entries({ x: 60, y: 30, width: 290, height: 190 })) near(box[key], expected, .15);
  const canvas = await page.locator('.annotation-canvas').boundingBox();
  const cx = canvas.x + canvas.width / 2;
  const cy = canvas.y + canvas.height / 2;
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: cx - 35, y: cy, id: 1 }, { x: cx + 35, y: cy, id: 2 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: cx - 70, y: cy + 5, id: 1 }, { x: cx + 70, y: cy + 5, id: 2 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  assert.ok(await page.evaluate(() => state.status.zoom > 1.8));
  assert.equal(await page.evaluate(() => state.changes.length), 1, 'pinch must not create or mutate a box');
  await page.evaluate(() => editor.fit());
  const cancelA = await imagePoint(page, 450, 40);
  const cancelB = await imagePoint(page, 600, 160);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: cancelA.x, y: cancelA.y, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: cancelB.x, y: cancelB.y, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchCancel', touchPoints: [] });
  assert.equal(await page.evaluate(() => state.changes.length), 1, 'interrupted touch must restore the last saved annotation');
  assert.equal(await page.locator('.annotation-box').count(), 1);
  for (const [width, height] of [[375, 650], [390, 844], [844, 390], [768, 1024], [1024, 768], [1280, 900]]) {
    await page.setViewportSize({ width, height });
    await page.evaluate(() => editor.fit());
    const layout = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth, editor: document.querySelector('#editor').getBoundingClientRect().toJSON(), smallButtons: [...document.querySelectorAll('#editor button:not(:disabled)')].filter(node => { const b = node.getBoundingClientRect(); return b.width && b.height && (b.width < 44 || b.height < 44); }).length }));
    assert.ok(layout.scroll <= layout.width, `horizontal overflow at ${width}×${height}`);
    assert.ok(layout.editor.width <= width, `editor overflow at ${width}×${height}`);
    assert.equal(layout.smallButtons, 0, `small touch controls at ${width}×${height}`);
  }
  await page.evaluate(() => setLanguage('en'));
  assert.equal(await page.locator('[data-action="pan"]').textContent(), 'Pan');
  await context.close();
}));
