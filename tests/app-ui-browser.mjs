import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { build, preview } from 'vite';
import { chromium } from 'playwright';

const resultRoot = fileURLToPath(new URL('../test-results/ui/', import.meta.url));
const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const builtDirectory = fileURLToPath(new URL('../test-results/ui/build/', import.meta.url));
const viewports = [[375, 650], [390, 844], [844, 390], [768, 1024], [1024, 768], [1280, 900]];
const screenshots = { images: [1280, 900], gallery: [1280, 900], annotate: [390, 844], train: [768, 1024], test: [844, 390] };

async function sourceFingerprint() {
  async function walk(relative) {
    const entries = await readdir(resolve(projectRoot, relative), { withFileTypes: true });
    return (await Promise.all(entries.map(entry => entry.isDirectory() ? walk(`${relative}/${entry.name}`) : `${relative}/${entry.name}`))).flat();
  }
  const files = ['index.html', 'vite.config.js', 'package-lock.json', 'public/models/manifest.json', ...await walk('src')].sort();
  const digest = createHash('sha256');
  const records = [];
  for (const path of files) {
    const content = await readFile(resolve(projectRoot, path));
    digest.update(path).update('\0').update(content).update('\0');
    records.push({ path, bytes: content.length, sha256: createHash('sha256').update(content).digest('hex') });
  }
  return { algorithm: 'sha256', normalization: 'exact file bytes, sorted path + NUL + bytes + NUL', value: digest.digest('hex'), files: records };
}

async function fixture(page, index = 0) {
  const bytes = await page.evaluate(async index => {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = ['#e8e1d8', '#ddd8e3', '#dce4df', '#e3dfd2'][index % 4];
    ctx.fillRect(0, 0, 128, 96);
    ctx.fillStyle = '#30302d';
    ctx.fillRect(18 + index * 3, 20, 52, 52);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    return [...new Uint8Array(await blob.arrayBuffer())];
  }, index);
  return { name: `example-${index + 1}.png`, mimeType: 'image/png', buffer: Buffer.from(bytes) };
}

async function filesFromClick(page, selector, files) {
  const chosen = page.waitForEvent('filechooser');
  await page.locator(selector).click();
  await (await chosen).setFiles(files);
}
async function imageStorageState(page, id) {
  return page.evaluate(id => new Promise((resolve, reject) => {
    const request = indexedDB.open('local-yolo-trainer-v1', 1);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => { request.transaction.abort(); reject(new Error('The app database was not created.')); };
    request.onsuccess = () => {
      const database = request.result;
      const transaction = database.transaction(['images', 'thumbnails', 'projects'], 'readonly');
      const image = transaction.objectStore('images').get(id);
      const thumbnail = transaction.objectStore('thumbnails').get(id);
      const projects = transaction.objectStore('projects').getAll();
      transaction.onabort = () => { database.close(); reject(transaction.error); };
      transaction.oncomplete = () => { database.close(); resolve({ imageExists: !!image.result, thumbnailExists: !!thumbnail.result, referenced: projects.result.some(project => project.images.some(image => image.id === id)) }); };
    };
  }), id);
}
async function mark(page, start = [18, 20], end = [70, 72]) {
  await page.locator('.annotation-canvas').scrollIntoViewIfNeeded();
  const points = await page.evaluate(({ start, end }) => {
    const matrix = document.querySelector('.annotation-canvas svg > g').getScreenCTM();
    return [start, end].map(([x, y]) => { const point = new DOMPoint(x, y).matrixTransform(matrix); return { x: point.x, y: point.y }; });
  }, { start, end });
  await page.mouse.move(points[0].x, points[0].y);
  await page.mouse.down();
  await page.mouse.move(points[1].x, points[1].y, { steps: 5 });
  await page.mouse.up();
}
async function switchLocale(page, language) {
  if (await page.locator('html').getAttribute('lang') !== (language === 'zh' ? 'zh-CN' : 'en')) await page.locator('#language-toggle').click();
  await page.waitForFunction(language => document.documentElement.lang === (language === 'zh' ? 'zh-CN' : 'en'), language);
}
async function switchTheme(page, theme) {
  for (let i = 0; i < 3 && await page.locator('html').getAttribute('data-theme') !== theme; i++) await page.locator('#theme-toggle').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'), theme);
}

async function inspectLayout(page, state, results, { dialog = false } = {}) {
  const languages = dialog ? [await page.locator('html').getAttribute('lang') === 'zh-CN' ? 'zh' : 'en'] : ['zh', 'en'];
  const themes = dialog ? [await page.locator('html').getAttribute('data-theme')] : ['light', 'dark'];
  for (const language of languages) {
    if (!dialog) await switchLocale(page, language);
    for (const theme of themes) {
      if (!dialog) await switchTheme(page, theme);
      for (const [width, height] of viewports) {
        await page.setViewportSize({ width, height });
        if (state === 'train' && !(await page.locator('#advanced-training').getAttribute('open') !== null)) await page.locator('#advanced-training summary').click();
        if (state === 'annotate') await page.locator('#active-class:not(:disabled)').waitFor();
        if (state === 'test') await page.locator('#test-preview .annotation-canvas image[href^="blob:"]').waitFor();
        if (state === 'gallery') await page.waitForFunction(() => [...document.querySelectorAll('.image-card img')].every(image => image.complete && image.naturalWidth > 0));
        await page.evaluate(() => window.scrollTo(0, 0));
        const data = await page.evaluate(dialog => {
          const scope = dialog ? document.querySelector('dialog') : document;
          const controls = [...scope.querySelectorAll('button,input:not([type=checkbox]):not([type=radio]),select,summary,.checkbox-row')].map(node => {
            const box = node.getBoundingClientRect();
            return { text: node.textContent.slice(0, 45), tag: node.tagName, id: node.id, disabled: node.disabled, x: box.x, right: box.right, width: box.width, height: box.height };
          }).filter(box => box.width > 0 && box.height > 0);
          return { width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth, heading: document.querySelector('#app h1')?.textContent, controls };
        }, dialog);
        assert.ok(data.heading, `the actual app was not rendered in ${state}`);
        assert.ok(data.scrollWidth <= data.width + 1, `actual app overflow ${state}/${language}/${theme}/${width}×${height}: ${data.scrollWidth}`);
        for (const control of data.controls) {
          assert.ok(control.x >= -.5 && control.right <= width + .5, `offscreen control ${JSON.stringify(control)} in ${state}/${width}×${height}`);
          if (control.tag === 'INPUT' && control.disabled) continue;
          // A declared 44 CSS px may read as 43.999969 after transforms. The
          // 0.001px arithmetic epsilon does not reduce the 44px product target.
          assert.ok(control.width >= 44 - 1e-3 && control.height >= 44 - 1e-3, `small control ${JSON.stringify(control)} in ${state}/${width}×${height}`);
        }
        results.layouts.push({ state, language, theme, width, height, controls: data.controls.length, pass: true });
        if (!dialog && language === 'zh' && theme === 'light' && screenshots[state]?.[0] === width && screenshots[state]?.[1] === height) {
          await page.locator('#toast').waitFor({ state: 'hidden' });
          await page.screenshot({ path: `${resultRoot}${state}-${width}x${height}.png`, fullPage: true, animations: 'disabled' });
        }
      }
    }
  }
  await page.setViewportSize({ width: 1280, height: 900 });
  if (!dialog) { await switchLocale(page, 'zh'); await switchTheme(page, 'light'); }
}

test('actual Trainer workflow, responsive layouts, bilingual themes, project recovery and real training', { timeout: 240000 }, async () => {
  await mkdir(resultRoot, { recursive: true });
  const results = { startedAt: new Date().toISOString(), scope: 'Desktop Chromium. Mobile touch and viewports are emulated; no physical mobile training claim.', layouts: [], checks: [], errors: [] };
  const runStamp = results.startedAt.replaceAll(':', '-');
  await mkdir(`${resultRoot}history/`, { recursive: true });
  let server, browser;
  try {
    results.source = await sourceFingerprint();
    await build({ build: { outDir: builtDirectory, emptyOutDir: true }, logLevel: 'warn' });
    results.builtHtmlSha256 = createHash('sha256').update(await readFile(resolve(builtDirectory, 'index.html'))).digest('hex');
    server = await preview({ build: { outDir: builtDirectory }, preview: { host: '127.0.0.1', port: 0 }, logLevel: 'warn' });
    const url = `http://127.0.0.1:${server.httpServer.address().port}${server.config.base}`;
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
    await context.addInitScript(() => { localStorage.setItem('local-yolo-trainer.language', 'zh'); localStorage.setItem('local-yolo-trainer.theme', 'light'); });
    const page = await context.newPage();
    page.on('pageerror', error => results.errors.push(error.message));
    const uploads = [];
    page.on('request', request => { if (request.method() !== 'GET' && request.method() !== 'HEAD') uploads.push({ method: request.method(), url: request.url() }); });
    await page.goto(url);
    await page.locator('#drop-zone').waitFor();
    await page.evaluate(() => {
      window.__qaBusyStates = { uploadProjectDisabled: false, uploadNavigationDisabled: false };
      new MutationObserver(() => {
        if (document.querySelector('#drop-zone[aria-disabled="true"]')) {
          window.__qaBusyStates.uploadProjectDisabled ||= document.querySelector('#project-menu').disabled;
          window.__qaBusyStates.uploadNavigationDisabled ||= [...document.querySelectorAll('#step-nav button')].every(button => button.disabled);
        }
      }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled', 'aria-disabled'] });
    });
    await inspectLayout(page, 'images', results);
    const samples = await Promise.all([0, 1, 2, 3].map(index => fixture(page, index)));
    await filesFromClick(page, '#drop-zone', samples);
    await page.waitForFunction(() => document.querySelectorAll('.image-card').length === 4 && !document.querySelector('#go-annotate')?.disabled);
    results.busyStates = await page.evaluate(() => window.__qaBusyStates);
    assert.deepEqual(results.busyStates, { uploadProjectDisabled: true, uploadNavigationDisabled: true });
    assert.equal(await page.locator('#project-menu').isDisabled(), false);
    await inspectLayout(page, 'gallery', results);
    await filesFromClick(page, '#drop-zone', [samples[0]]);
    await page.waitForFunction(() => document.querySelector('#toast')?.textContent.includes('重复 1'));
    assert.equal(await page.locator('.image-card').count(), 4);
    results.checks.push('real multi-image upload and duplicate detection; Project and navigation remain disabled during image processing');
    await page.locator('#go-annotate').click();
    await page.locator('.annotation-canvas image').waitFor();
    await page.locator('#edit-classes').click();
    await page.locator('[data-class-name="0"]').fill('Square');
    await page.getByRole('button', { name: '保存', exact: true }).click();
    await page.locator('.annotation-canvas image').waitFor();
    await mark(page);
    assert.equal(await page.locator('.annotation-box').count(), 1);
    await page.locator('#class-mode').selectOption('multi');
    await page.locator('#edit-classes').click();
    await page.locator('#new-class').fill('Circle');
    await page.getByRole('button', { name: '保存', exact: true }).click();
    await page.locator('#active-class').selectOption('1');
    await mark(page, [88, 12], [116, 40]);
    assert.equal(await page.locator('.annotation-box').count(), 2);
    assert.deepEqual(await page.locator('.annotation-box .box-label').allTextContents(), ['Square', 'Circle']);
    assert.equal(await page.locator('#operation-error').isVisible(), false, 'fast class selection after loading must not lose the chosen category');
    await inspectLayout(page, 'annotate', results);
    // Visiting training computes splits. Removing a photo afterwards must commit
    // a consistent project and delete its original and thumbnail atomically.
    await page.locator('[data-step="train"]').click();
    await page.locator('#start-training').waitFor();
    await page.locator('[data-step="images"]').click();
    await page.locator('.image-card').last().waitFor();
    const removedId = await page.locator('[data-delete]').last().getAttribute('data-delete');
    await page.locator('[data-delete]').last().click();
    await page.getByRole('button', { name: '取消', exact: true }).click();
    assert.equal(await page.locator('.image-card').count(), 4, 'cancel must preserve all photos');
    await page.locator('[data-delete]').last().click();
    await page.getByRole('button', { name: '确认', exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('.image-card').length === 3 || !document.querySelector('#operation-error')?.hidden);
    assert.equal(await page.locator('#operation-error').isVisible(), false, await page.locator('#operation-error').textContent());
    assert.equal(await page.locator('.image-card').count(), 3);
    await page.reload();
    await page.locator('#go-annotate').waitFor();
    assert.equal(await page.locator('.image-card').count(), 3);
    results.removedImageStorage = await imageStorageState(page, removedId);
    assert.deepEqual(results.removedImageStorage, { imageExists: false, thumbnailExists: false, referenced: false });
    results.checks.push('cancel preserves a photo; confirmed removal after visiting training survives reload with no image, thumbnail or project-reference orphan');
    await page.locator('#go-annotate').click();
    await page.locator('#active-class:not(:disabled)').waitFor();
    await page.locator('#next-image').click();
    await page.locator('.annotation-canvas image').waitFor();
    await mark(page);
    await page.locator('#next-image').click();
    await page.locator('.annotation-canvas image').waitFor();
    await page.locator('#reviewed').check();
    await page.locator('#save-status').filter({ hasText: '已保存在此浏览器' }).waitFor();
    await page.reload();
    await page.locator('.annotation-canvas image').waitFor();
    assert.equal(await page.locator('.annotation-box').count(), 2);
    results.checks.push('actual single/multi-category annotation and IndexedDB reload recovery');
    await page.locator('#go-train').click();
    await page.locator('#training-mode').selectOption('scratch');
    await page.locator('#advanced-training summary').click();
    await page.locator('#epochs').fill('0');
    await page.locator('#epochs').dispatchEvent('change');
    await page.locator('#operation-error').waitFor({ state: 'visible' });
    await page.locator('#project-menu').click();
    const validDataBackup = page.waitForEvent('download');
    await page.getByRole('button', { name: '导出完整备份', exact: true }).click();
    assert.ok((await validDataBackup).suggestedFilename().endsWith('.lyproj'), 'invalid form input must not corrupt the last valid project or prevent its backup');
    await page.locator('#epochs').fill('1');
    await page.locator('#epochs').dispatchEvent('change');
    if (await page.locator('#dismiss-error').isVisible()) await page.locator('#dismiss-error').click();
    await page.locator('#training-backend').selectOption('cpu');
    await inspectLayout(page, 'train', results);
    await page.locator('#start-training').click();
    await page.waitForFunction(() => !!document.querySelector('#go-test') || !document.querySelector('#operation-error')?.hidden, null, { timeout: 90000 });
    assert.equal(await page.locator('#operation-error').isVisible(), false, await page.locator('#operation-error').textContent());
    results.progressDisplay = await page.evaluate(() => ({ state: document.querySelector('#training-state').textContent, value: document.querySelector('#training-bar').value, numbers: document.querySelector('#training-numbers').textContent }));
    assert.equal(results.progressDisplay.value, 1, 'completed progress must reflect actual finished steps');
    assert.match(results.progressDisplay.state, /完成/);
    await page.locator('#toast').waitFor({ state: 'hidden' });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${resultRoot}training-completed-1280x900.png`, fullPage: true, animations: 'disabled' });
    await page.locator('#go-test').click();
    const trainingRecord = JSON.parse(await page.locator('#app details.panel pre').textContent());
    assert.equal(trainingRecord.architecture, 'tiny-grid-v1');
    assert.ok(trainingRecord.step > 0);
    assert.equal(trainingRecord.evidence.parametersChanged, true);
    assert.notEqual(trainingRecord.evidence.initialWeightHash, trainingRecord.evidence.finalWeightHash);
    assert.ok(Number.isFinite(trainingRecord.evidence.firstLoss));
    assert.ok(trainingRecord.evidence.maxGradientNorm > 0);
    results.training = trainingRecord;
    await page.locator('#continue-learning').click();
    await page.locator('#advanced-training summary').click();
    assert.match(await page.locator('#epochs').locator('..').textContent(), /再学习多少轮/);
    await page.locator('#start-training').click();
    await page.waitForFunction(() => !!document.querySelector('#go-test') || !document.querySelector('#operation-error')?.hidden, null, { timeout: 90000 });
    assert.equal(await page.locator('#operation-error').isVisible(), false, await page.locator('#operation-error').textContent());
    await page.locator('#go-test').click();
    const continuedRecord = JSON.parse(await page.locator('#app details.panel pre').textContent());
    assert.ok(continuedRecord.step > trainingRecord.step, 'continue learning must perform additional real updates');
    assert.equal(continuedRecord.evidence.initialWeightHash, trainingRecord.evidence.finalWeightHash);
    assert.equal(continuedRecord.evidence.parametersChanged, true);
    assert.notEqual(continuedRecord.evidence.initialWeightHash, continuedRecord.evidence.finalWeightHash);
    assert.equal(continuedRecord.validation.available, true);
    assert.equal(continuedRecord.validation.images, 1);
    results.continuedTraining = continuedRecord;
    results.checks.push('Continue learning performs new checkpoint updates and begins from the previously trained weight hash');
    await filesFromClick(page, '#test-picture', [await fixture(page, 3)]);
    await page.locator('#test-preview .annotation-canvas image').waitFor({ timeout: 60000 });
    await inspectLayout(page, 'test', results);
    const download = page.waitForEvent('download');
    await page.locator('#save-model').click();
    assert.ok((await download).suggestedFilename().endsWith('.lymodel'));
    const onnx = page.waitForEvent('download');
    await page.locator('#export-onnx').click();
    assert.ok((await onnx).suggestedFilename().endsWith('.onnx'));
    results.checks.push('real CPU scratch training, fresh-image inference, checkpoint and ONNX downloads');
    await page.locator('#project-menu').click();
    await page.locator('dialog').waitFor();
    await inspectLayout(page, 'project-dialog', results, { dialog: true });
    await page.getByRole('button', { name: '关闭', exact: true }).click();
    await page.locator('#help').click();
    await page.locator('dialog').waitFor();
    await inspectLayout(page, 'help-dialog', results, { dialog: true });
    await page.getByRole('button', { name: '开始使用 / 跳过', exact: true }).click();
    await page.locator('#project-menu').click();
    const backupDownload = page.waitForEvent('download');
    await page.getByRole('button', { name: '导出完整备份', exact: true }).click();
    const backup = await backupDownload;
    const backupPath = `${resultRoot}ui-recovery-fixture.lyproj`;
    assert.ok(backup.suggestedFilename().endsWith('.lyproj'));
    await backup.saveAs(backupPath);
    await page.locator('#project-menu').click();
    await page.getByRole('button', { name: '新建', exact: true }).click();
    await page.locator('#drop-zone').waitFor();
    assert.equal(await page.locator('.image-card').count(), 0);
    await page.locator('#project-menu').click();
    const restoreChooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: '导入备份', exact: true }).click();
    await (await restoreChooser).setFiles(backupPath);
    await page.waitForFunction(() => document.querySelectorAll('.image-card').length === 3);
    await page.locator('#go-annotate').click();
    await page.locator('#active-class:not(:disabled)').waitFor();
    assert.deepEqual(await page.locator('.annotation-box .box-label').allTextContents(), ['Square', 'Circle']);
    await page.locator('[data-step="test"]').click();
    await page.locator('#save-model').waitFor();
    const restoredRecord = JSON.parse(await page.locator('#app details.panel pre').textContent());
    assert.deepEqual(restoredRecord.validation, continuedRecord.validation, 'reloading the checkpoint must preserve actual validation metrics from its history');
    results.restoredValidation = restoredRecord.validation;
    await filesFromClick(page, '#test-picture', [await fixture(page, 3)]);
    await page.locator('#test-preview .annotation-canvas image[href^="blob:"]').waitFor();
    assert.equal(await page.locator('#operation-error').isVisible(), false);
    results.checks.push('actual complete project export/import preserves images, both classes, annotations and an inference-capable checkpoint');
    assert.deepEqual(uploads, [], 'no user-data uploads');
    assert.deepEqual(results.errors, []);
    results.checks.push('project/help dialogs remain usable; no image or training data uploads');
    results.sourceAfter = (await sourceFingerprint()).value;
    assert.equal(results.sourceAfter, results.source.value, 'runtime source changed during this acceptance run; rebuild and rerun');
    results.status = 'pass';
  } catch (error) {
    results.status = 'fail';
    results.failure = error.stack;
    const failedPage = browser?.contexts()[0]?.pages()[0];
    if (failedPage) {
      await failedPage.screenshot({ path: `${resultRoot}history/failure-${runStamp}.png`, fullPage: true, animations: 'disabled' }).catch(() => {});
      results.failureMarkup = (await failedPage.locator('#app').innerHTML().catch(() => '')).slice(0, 25000);
    }
    throw error;
  } finally {
    results.finishedAt = new Date().toISOString();
    await writeFile(`${resultRoot}actual-app-ui.json`, JSON.stringify(results, null, 2));
    await writeFile(`${resultRoot}history/app-ui-${runStamp}.json`, JSON.stringify(results, null, 2));
    await browser?.close();
    if (server) await new Promise(resolve => server.httpServer.close(resolve));
  }
});
