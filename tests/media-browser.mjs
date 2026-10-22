// Actual browser image decoding. Fixtures are generated locally; no API mocks.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = new URL('../', import.meta.url);
const reportDirectory = fileURLToPath(new URL('../test-results/media/', import.meta.url));
const observedSources = ['src/data/media.js', 'src/data/project.js'];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { startedAt: new Date().toISOString(), scope: 'Desktop Chromium real decoding; generated fixtures; no physical-phone or Safari claim.', cases: [], pageErrors: [], nonGetRequests: [] };
let server, browser, url;

async function sourceHashes() {
  return Object.fromEntries(await Promise.all(observedSources.map(async path => [path, digest(await readFile(new URL(path, root)))])));
}

function addExifOrientation(jpeg, orientation) {
  assert.equal(jpeg[0], 0xff);
  assert.equal(jpeg[1], 0xd8);
  const payload = Buffer.alloc(32);
  payload.write('Exif\0\0', 0, 'binary');
  payload.write('II', 6, 'ascii');
  payload.writeUInt16LE(42, 8);
  payload.writeUInt32LE(8, 10);
  payload.writeUInt16LE(1, 14);
  payload.writeUInt16LE(0x0112, 16); // TIFF Orientation
  payload.writeUInt16LE(3, 18);     // SHORT
  payload.writeUInt32LE(1, 20);
  payload.writeUInt16LE(orientation, 24);
  payload.writeUInt32LE(0, 28);
  const marker = Buffer.from([0xff, 0xe1, 0, payload.length + 2]);
  return Buffer.concat([jpeg.subarray(0, 2), marker, payload, jpeg.subarray(2)]);
}

function exifMetadata(bytes) {
  const buffer = Buffer.from(bytes), orientations = [];
  let segments = 0;
  if (buffer[0] !== 0xff || buffer[1] !== 0xd8) return { segments, orientations };
  let offset = 2;
  while (offset + 4 <= buffer.length) {
    if (buffer[offset] !== 0xff) throw new Error('Malformed JPEG segment marker.');
    const marker = buffer[offset + 1];
    if (marker === 0xda || marker === 0xd9) break;
    const length = buffer.readUInt16BE(offset + 2);
    if (length < 2 || offset + 2 + length > buffer.length) throw new Error('Malformed JPEG segment length.');
    const start = offset + 4;
    if (marker === 0xe1 && buffer.subarray(start, start + 6).equals(Buffer.from('Exif\0\0', 'binary'))) {
      segments++;
      const tiff = start + 6;
      const little = buffer.toString('ascii', tiff, tiff + 2) === 'II';
      const u16 = position => little ? buffer.readUInt16LE(position) : buffer.readUInt16BE(position);
      const u32 = position => little ? buffer.readUInt32LE(position) : buffer.readUInt32BE(position);
      assert.equal(u16(tiff + 2), 42);
      const directory = tiff + u32(tiff + 4);
      for (let i = 0; i < u16(directory); i++) {
        const entry = directory + 2 + i * 12;
        if (u16(entry) === 0x0112) orientations.push(u16(entry + 8));
      }
    }
    offset += 2 + length;
  }
  return { segments, orientations };
}

before(async () => {
  report.sourceBefore = await sourceHashes();
  server = createServer(async (request, response) => {
    const path = new URL(request.url, 'http://localhost').pathname;
    if (path === '/media-harness') {
      response.setHeader('Content-Type', 'text/html');
      response.end(`<!doctype html><html><meta charset="utf-8"><title>Actual media import acceptance</title><body><p>Generated image decoding fixtures</p><script type="module">import * as media from '/src/data/media.js';window.media=media;</script></body></html>`);
      return;
    }
    if (!observedSources.includes(path.slice(1))) { response.writeHead(404).end(); return; }
    try { response.setHeader('Content-Type', 'text/javascript'); response.end(await readFile(new URL(path.slice(1), root))); }
    catch { response.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  url = `http://127.0.0.1:${server.address().port}/media-harness`;
  browser = await chromium.launch({ headless: true });
  report.browser = await browser.version();
});

async function runCase(name, operation) {
  const page = await browser.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  page.on('request', request => { if (!['GET', 'HEAD'].includes(request.method())) report.nonGetRequests.push({ method: request.method(), url: request.url() }); });
  const entry = { name, status: 'running' };
  report.cases.push(entry);
  try {
    await page.goto(url);
    await page.waitForFunction(() => window.media);
    entry.result = await operation(page);
    entry.status = 'pass';
  } catch (error) { entry.status = 'fail'; entry.error = error.stack; throw error; }
  finally { await page.close(); }
}

async function importAndInspect(page, bytes, name, mime, thumbnailSize = 64) {
  return page.evaluate(async ({ bytes, name, mime, thumbnailSize }) => {
    const file = new File([new Uint8Array(bytes)], name, { type: mime });
    const before = new Uint8Array(await file.arrayBuffer());
    const imported = await media.importImage(file);
    const after = new Uint8Array(await file.arrayBuffer());
    async function inspect(blob) {
      const image = await createImageBitmap(blob);
      try {
        const canvas = document.createElement('canvas');
        canvas.width = image.width; canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        const samples = [[.25, .25], [.75, .25], [.25, .75], [.75, .75]].map(([x, y]) => [...ctx.getImageData(Math.floor(x * image.width), Math.floor(y * image.height), 1, 1).data]);
        return { width: image.width, height: image.height, mime: blob.type, samples };
      } finally { image.close(); }
    }
    return {
      metadata: imported.image,
      inputBytesUnchanged: before.length === after.length && before.every((value, index) => value === after[index]),
      canonicalBytes: [...new Uint8Array(await imported.blob.arrayBuffer())],
      canonical: await inspect(imported.blob),
      thumbnail: await inspect(imported.thumbnail),
      customThumbnail: await inspect(await media.makeThumbnail(file, thumbnailSize)),
      decoder: typeof createImageBitmap === 'function' ? 'createImageBitmap/from-image' : 'HTMLImageElement',
    };
  }, { bytes: [...bytes], name, mime, thumbnailSize });
}

const colors = { red: [220, 30, 40, 255], green: [30, 190, 70, 255], blue: [30, 70, 220, 255], yellow: [230, 210, 40, 255] };
function nearColor(actual, expected, tolerance = 12) {
  for (let channel = 0; channel < 4; channel++) assert.ok(Math.abs(actual[channel] - expected[channel]) <= tolerance, `Pixel ${JSON.stringify(actual)} differs from ${JSON.stringify(expected)} at channel ${channel}.`);
}

test('real EXIF 6 and 8 JPEG import rotates dimensions and all four markers, strips EXIF and preserves input bytes', { timeout: 30000 }, async () => runCase('EXIF JPEG orientations 6 and 8', async page => {
  const base = Buffer.from(await page.evaluate(async colors => {
    const canvas = document.createElement('canvas'); canvas.width = 120; canvas.height = 80;
    const ctx = canvas.getContext('2d');
    for (const [index, value] of Object.values(colors).entries()) { ctx.fillStyle = `rgb(${value.slice(0, 3).join(',')})`; ctx.fillRect(index % 2 * 60, Math.floor(index / 2) * 40, 60, 40); }
    const jpeg = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 1));
    return [...new Uint8Array(await jpeg.arrayBuffer())];
  }, colors));
  const observations = [];
  for (const [orientation, expected] of [[6, ['blue', 'red', 'yellow', 'green']], [8, ['green', 'yellow', 'red', 'blue']]]) {
    const bytes = addExifOrientation(base, orientation);
    assert.deepEqual(exifMetadata(bytes), { segments: 1, orientations: [orientation] }, 'fixture must contain a real EXIF APP1 segment and Orientation tag');
    const result = await importAndInspect(page, bytes, `orientation-${orientation}.jpg`, 'image/jpeg');
    assert.deepEqual([result.metadata.width, result.metadata.height], [80, 120]);
    assert.deepEqual([result.canonical.width, result.canonical.height], [80, 120]);
    assert.equal(result.canonical.mime, 'image/jpeg');
    for (const [index, name] of expected.entries()) nearColor(result.canonical.samples[index], colors[name]);
    const canonicalExif = exifMetadata(result.canonicalBytes);
    assert.deepEqual(canonicalExif, { segments: 0, orientations: [] }, 'canonical output must strip the whole EXIF segment, not only the Orientation tag');
    assert.equal(result.inputBytesUnchanged, true);
    assert.equal(result.metadata.sourceSha256, digest(bytes));
    assert.equal(result.metadata.sha256, digest(Buffer.from(result.canonicalBytes)));
    assert.notEqual(result.metadata.sha256, result.metadata.sourceSha256);
    assert.deepEqual([result.thumbnail.width, result.thumbnail.height], [80, 120], 'small pictures must not be enlarged');
    assert.deepEqual([result.customThumbnail.width, result.customThumbnail.height], [43, 64]);
    observations.push({ orientation, inputSha256: digest(bytes), canonicalExif, expectedMarkerOrder: expected, ...result, canonicalBytes: undefined });
  }
  return observations;
}));

test('transparent PNG retains alpha and original dimensions; thumbnails preserve aspect ratio and bounded size', { timeout: 30000 }, async () => runCase('Transparent PNG and thumbnail sizing', async page => {
  const bytes = Buffer.from(await page.evaluate(async () => {
    const canvas = document.createElement('canvas'); canvas.width = 1024; canvas.height = 768;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(230,30,60,.5)'; ctx.fillRect(512, 0, 512, 384);
    ctx.fillStyle = 'rgb(30,180,210)'; ctx.fillRect(0, 384, 512, 384);
    ctx.fillStyle = 'rgb(230,210,40)'; ctx.fillRect(512, 384, 512, 384);
    const png = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    return [...new Uint8Array(await png.arrayBuffer())];
  }));
  const result = await importAndInspect(page, bytes, 'transparent.png', 'image/png');
  assert.equal(result.inputBytesUnchanged, true);
  assert.equal(result.metadata.sourceSha256, digest(bytes));
  assert.equal(result.metadata.sha256, digest(Buffer.from(result.canonicalBytes)));
  assert.equal(result.canonical.mime, 'image/png');
  assert.deepEqual([result.canonical.width, result.canonical.height], [1024, 768]);
  assert.deepEqual(result.canonical.samples[0], [0, 0, 0, 0]);
  nearColor(result.canonical.samples[1], [230, 30, 60, 128], 2);
  nearColor(result.canonical.samples[2], [30, 180, 210, 255], 0);
  nearColor(result.canonical.samples[3], [230, 210, 40, 255], 0);
  assert.deepEqual([result.thumbnail.width, result.thumbnail.height], [256, 192]);
  assert.deepEqual([result.customThumbnail.width, result.customThumbnail.height], [64, 48]);
  assert.equal(result.thumbnail.samples[0][3], 0);
  assert.ok(Math.abs(result.thumbnail.samples[1][3] - 128) <= 1);
  assert.equal(result.customThumbnail.samples[0][3], 0);
  assert.ok(Math.abs(result.customThumbnail.samples[1][3] - 128) <= 1);
  return { ...result, canonicalBytes: undefined };
}));

test('actual decoder rejects corrupt JPEG, truncated PNG and empty input without yielding a canonical image', { timeout: 30000 }, async () => runCase('Corrupt image rejection', async page => {
  const failures = await page.evaluate(async () => {
    const fixtures = [
      { name: 'corrupt.jpg', mime: 'image/jpeg', bytes: [255, 216, 255, 224, 0, 2, 255, 217] },
      { name: 'truncated.png', mime: 'image/png', bytes: [137, 80, 78, 71, 13, 10, 26, 10] },
      { name: 'empty.jpg', mime: 'image/jpeg', bytes: [] },
    ];
    const results = [];
    for (const fixture of fixtures) {
      try { await media.importImage(new File([new Uint8Array(fixture.bytes)], fixture.name, { type: fixture.mime })); results.push({ name: fixture.name, rejected: false }); }
      catch (error) { results.push({ name: fixture.name, rejected: true, errorName: error.name, errorMessage: error.message, code: error.code ?? null }); }
    }
    return results;
  });
  assert.equal(failures.length, 3);
  for (const failure of failures) assert.equal(failure.rejected, true, `${failure.name} must not be accepted`);
  return failures;
}));

after(async () => {
  report.sourceAfter = await sourceHashes();
  report.sourceStable = JSON.stringify(report.sourceBefore) === JSON.stringify(report.sourceAfter);
  report.finishedAt = new Date().toISOString();
  report.status = report.cases.length === 3 && report.cases.every(entry => entry.status === 'pass') && report.sourceStable && report.pageErrors.length === 0 && report.nonGetRequests.length === 0 ? 'pass' : 'fail';
  await mkdir(reportDirectory, { recursive: true });
  await writeFile(`${reportDirectory}media-browser.json`, JSON.stringify(report, null, 2));
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
  assert.equal(report.sourceStable, true, 'media source changed during the run');
  assert.deepEqual(report.pageErrors, []);
  assert.deepEqual(report.nonGetRequests, []);
});
