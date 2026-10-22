// Synthetic tiny runtime files exercise the verifier, not inference. The
// real build CLI separately compares actual installed runtime bytes.
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, writeFile, rm} from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {checkBuild, RUNTIME_FILES} from '../scripts/check-build.mjs';
import {BACKBONE, BACKBONE_ASSET_PATH} from '../src/model/backbone.js';

const actualModel = await readFile(new URL('../public/models/yolo26n-features-128.onnx', import.meta.url));
const wasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
async function fixture(run, base = '/local-yolo-trainer/') {
  const directory = await mkdtemp(path.join(tmpdir(), 'trainer-build-check-')), dist = path.join(directory, 'dist'), sourceRoot = path.join(directory, 'source');
  const put = async (filename, content) => { const target = path.join(dist, filename); await mkdir(path.dirname(target), {recursive: true}); await writeFile(target, content); };
  try {
    await put('.nojekyll', '');
    await put('index.html', `<script type="module" src="${base}assets/main.js"></script><link rel="stylesheet" href="${base}assets/style.css">`);
    await put('assets/main.js', 'import "./entry.js";new Worker(new URL("./worker-test.js",import.meta.url),{type:"module"});');
    await put('assets/entry.js', 'export const ready=true;'); await put('assets/worker-test.js', 'import("./entry.js");'); await put('assets/style.css', 'body{color:black}');
    await put(BACKBONE_ASSET_PATH, actualModel);
    await put('models/manifest.json', JSON.stringify({...BACKBONE, filename: path.basename(BACKBONE_ASSET_PATH)}));
    for (const [destination, source] of RUNTIME_FILES) {
      const content = destination.endsWith('.wasm') ? wasm : destination.endsWith('.mjs') ? 'export const syntheticVerifierFixture=true;' : 'Synthetic verifier notice fixture; not a redistributed runtime license.';
      await put(destination, content); const original = path.join(sourceRoot, source); await mkdir(path.dirname(original), {recursive: true}); await writeFile(original, content);
    }
    await run({directory: dist, sourceRoot, base, put});
  } finally {
    const relative = path.relative(path.resolve(tmpdir()), path.resolve(directory));
    if (relative.startsWith('..') || path.isAbsolute(relative) || !path.basename(directory).startsWith('trainer-build-check-')) throw new Error('Refusing to remove a test directory outside the named temporary workspace.');
    await rm(directory, {recursive: true, force: true});
  }
}

test('build verifier resolves correct nested-base HTML, ESM and Worker resources', async () => fixture(async options => {
  const result = await checkBuild(options); assert.equal(result.status, 'pass', result.errors.join('\n')); assert.equal(result.errors.length, 0);
  assert(result.checks.find(check => check.name === 'subpath-resources-and-portability').references >= 4);
}, '/nested/trainer/'));

test('build verifier rejects incorrect Pages paths and missing emitted Worker dependencies', async () => fixture(async options => {
  await options.put('index.html', '<script type="module" src="/assets/main.js"></script>');
  await options.put('assets/worker-test.js', 'import("./missing.js");');
  const result = await checkBuild(options); assert.equal(result.status, 'fail');
  assert(result.errors.some(error => error.includes('escapes deployment base'))); assert(result.errors.some(error => error.includes('missing.js')));
}));

test('build verifier pins model bytes independently of a tampered manifest', async () => fixture(async options => {
  const damaged = Buffer.from(actualModel); damaged[damaged.length - 1] ^= 1; await options.put(BACKBONE_ASSET_PATH, damaged);
  await options.put('models/manifest.json', JSON.stringify({...BACKBONE, sha256: '0'.repeat(64), filename: path.basename(BACKBONE_ASSET_PATH)}));
  const result = await checkBuild(options); assert.equal(result.status, 'fail'); assert(result.errors.some(error => error.includes('pinned byte count/SHA'))); assert(result.errors.some(error => error.includes('manifest')));
}));

test('build verifier rejects missing notices, altered WASM and accidental developer fixtures', async () => fixture(async options => {
  await options.put('runtime/FFLATE_LICENSE.txt', ''); await options.put('runtime/ort/ort-wasm-simd-threaded.wasm', new Uint8Array([1, 2, 3]));
  await options.put('tests/fixtures/private-picture.png', new Uint8Array([1]));
  const result = await checkBuild(options); assert.equal(result.status, 'fail');
  for (const expected of ['FFLATE_LICENSE', 'Invalid WASM', 'Development fixtures']) assert(result.errors.some(error => error.includes(expected)), expected);
}));

test('build verifier rejects real local service/path literals but permits generic vendor portability branches', async () => fixture(async options => {
  await options.put('assets/entry.js', 'const bad="http://127.0.0.1:5187/api";const path="C:/Users/example/model.bin";');
  let result = await checkBuild(options); assert.equal(result.status, 'fail'); assert(result.errors.some(error => error.includes('local/private network'))); assert(result.errors.some(error => error.includes('filesystem path')));
  await options.put('assets/entry.js', 'export const virtual="/home/web_user";const isFile=s=>s.startsWith("file://");if(typeof process!=="undefined"&&process.versions?.node)import("node:fs");');
  result = await checkBuild(options); assert.equal(result.status, 'pass', result.errors.join('\n'));
}));
