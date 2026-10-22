import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {zipSync, unzipSync, strToU8, strFromU8, zlibSync} from 'fflate';
import {createProject, validateProject, validateClasses, validateBoxes, validateConfig, assignSplits, datasetDigest, sha256, isIsoTimestamp} from '../src/data/project.js';
import {packCheckpoint, unpackCheckpoint, packProject, unpackProject, exportYoloDataset, validateCheckpoint} from '../src/data/formats.js';
import {SaveQueue} from '../src/data/storage.js';
import {decodeImage} from '../src/data/media.js';

function png(width = 200, height = 100, color = [140, 60, 30]) {
  const crc = bytes => { let value = 0xffffffff; for (const byte of bytes) { value ^= byte; for (let i = 0; i < 8; i++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0); } return (value ^ 0xffffffff) >>> 0; };
  const chunk = (name, bytes) => { const type = Buffer.from(name), result = Buffer.alloc(12 + bytes.length); result.writeUInt32BE(bytes.length); type.copy(result, 4); result.set(bytes, 8); result.writeUInt32BE(crc(Buffer.concat([type, Buffer.from(bytes)])), 8 + bytes.length); return result; };
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 2;
  const pixels = new Uint8Array((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) pixels.set(color, y * (width * 3 + 1) + 1 + x * 3);
  return new Blob([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', zlibSync(pixels)), chunk('IEND', new Uint8Array())], {type: 'image/png'});
}
async function projectFixture() {
  const project = createProject({title: 'Original project', className: 'person'}), blobs = new Map();
  for (const [index, color] of [[150, 30, 10], [20, 160, 40], [20, 40, 170]].entries()) {
    const blob = png(200, 100, color), id = `original-image-${index}`;
    blobs.set(id, blob);
    project.images.push({id, name: `photo-${index}.png`, mime: 'image/png', width: 200, height: 100, sha256: await sha256(blob), boxes: index === 2 ? [] : [{id: `box-${index}`, classId: 0, x: 20, y: 10, width: 40, height: 30}], reviewed: true});
  }
  assignSplits(project);
  return {project, blobs};
}
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; }
const turn = () => new Promise(resolve => setImmediate(resolve));
async function mutateArchive(blob, mutate) {
  const files = unzipSync(new Uint8Array(await blob.arrayBuffer()));
  const manifest = JSON.parse(strFromU8(files['manifest.json']));
  await mutate(files, manifest);
  files['manifest.json'] = strToU8(JSON.stringify(manifest));
  return new Blob([zipSync(files, {level: 0})], {type: 'application/zip'});
}
let trainedCheckpointPromise;
async function trainedCheckpoint() {
  if (!trainedCheckpointPromise) trainedCheckpointPromise = (async () => {
    const {TrainerEngine} = await import('../src/training/engine.js');
    const backboneBytes = await readFile(new URL('../public/models/yolo26n-features-128.onnx', import.meta.url));
    const engine = new TrainerEngine({backboneBytes});
    try {
      const pixels = new Uint8Array(128 * 128 * 3).fill(20);
      for (let y = 30; y < 70; y++) for (let x = 40; x < 60; x++) pixels[(y * 128 + x) * 3] = 240;
      const result = await engine.train({samples: [{id: 'actual-step', pixels, width: 128, height: 128, split: 'train', boxes: [{id: 'box', classId: 0, x: 40, y: 30, width: 20, height: 40}]}], classes: ['person'], config: {mode: 'pretrained', backend: 'cpu', epochs: 2, batchSize: 1, augment: false, seed: 12345, learningRate: .003}});
      assert(result.evidence.parametersChanged); assert(result.evidence.maxGradientNorm > 0);
      return result.checkpoint;
    } finally { await engine.dispose(); }
  })();
  return structuredClone(await trainedCheckpointPromise);
}

test('split assignment groups identical SHA images and excludes every unreviewed photo', async () => {
  const {project} = await projectFixture();
  project.images.push({...structuredClone(project.images[0]), id: 'same-photo-new-id', boxes: []});
  project.images.push({...structuredClone(project.images[1]), id: 'not-confirmed', reviewed: false, split: 'validation'});
  const result = assignSplits(project);
  assert.equal(project.images[0].split, project.images[3].split, 'Identical photos cannot leak across train/validation.');
  assert.equal(project.images[4].split, undefined);
  assert.equal(result.train + result.validation, 4); assert.equal(result.unreviewed, 1); assert.equal(result.unique, 3);
  const train = new Set(project.images.filter(image => image.split === 'train').map(image => image.sha256));
  assert(project.images.filter(image => image.split === 'validation').every(image => !train.has(image.sha256)));
});

test('dataset digest survives UUID/title/order changes and tracks material annotation changes', async () => {
  const {project} = await projectFixture(), original = await datasetDigest(project), restored = structuredClone(project);
  restored.id = crypto.randomUUID(); restored.title = 'Another project'; restored.images.reverse();
  for (const image of restored.images) { image.id = crypto.randomUUID(); image.name = 'renamed.png'; for (const box of image.boxes) box.id = crypto.randomUUID(); }
  assert.equal(await datasetDigest(restored), original);
  restored.images.find(image => image.boxes.length).boxes[0].width += 1;
  assert.notEqual(await datasetDigest(restored), original);
  const unreviewed = structuredClone(project); unreviewed.images.push({...structuredClone(project.images[0]), id: 'unreviewed', reviewed: false});
  assert.equal(await datasetDigest(unreviewed), original);
});

test('classes and project coordinates reject invalid labels, nonfinite numbers and escaped boxes', async () => {
  for (const classes of [[], ['person', 'PERSON'], ['é', 'e\u0301'], [''], ['bad\nclass']]) assert.throws(() => validateClasses(classes));
  const box = {id: 'valid', classId: 0, x: 0, y: 0, width: 20, height: 30};
  assert.deepEqual(validateBoxes([box], 200, 100, 1), [box]);
  for (const invalid of [{...box, x: NaN}, {...box, width: Infinity}, {...box, x: -1}, {...box, width: 201}, {...box, classId: 1}, {...box, classId: .5}]) assert.throws(() => validateBoxes([invalid], 200, 100, 1));
  assert.throws(() => validateBoxes([box, {...box}], 200, 100, 1), /BOX_ID/);
  const {project} = await projectFixture();
  for (const edit of [p => { p.schemaVersion = 2; }, p => { p.images[0].width = NaN; }, p => { p.images[0].sha256 = 'not-a-sha'; }, p => { p.images[0].reviewed = 'yes'; }, p => { p.images[0].split = 'val'; }]) { const invalid = structuredClone(project); edit(invalid); assert.throws(() => validateProject(invalid)); }
});

test('project class/config limits agree with the actually supported training engine', () => {
  assert.throws(() => validateClasses(Array.from({length: 33}, (_, index) => `class ${index}`)));
  assert.throws(() => validateClasses(['x'.repeat(81)]));
  const config = createProject().config;
  assert.throws(() => validateConfig({...config, seed: 0}));
  assert.throws(() => validateConfig({...config, augment: 'yes'}));
});

test('project timestamps must be real canonical UTC ISO strings before import/persistence', async () => {
  const {project, blobs} = await projectFixture();
  assert(isIsoTimestamp(project.createdAt)); assert(isIsoTimestamp('2024-02-29T12:00:00.000Z'));
  for (const bad of [undefined, null, 42, '', 'not a date', '2026-02-30T12:00:00.000Z', '2026-10-21', '2026-10-21T12:00:00Z', '2026-10-21T12:00:00.000+00:00']) {
    assert.equal(isIsoTimestamp(bad), false);
    for (const field of ['createdAt', 'updatedAt']) { const invalid = structuredClone(project); invalid[field] = bad; assert.throws(() => validateProject(invalid), /PROJECT_TIMESTAMPS/); }
  }
  const archive = await packProject(project, id => blobs.get(id));
  const badArchive = await mutateArchive(archive, (_files, manifest) => { manifest.project.updatedAt = null; });
  await assert.rejects(unpackProject(badArchive), /PROJECT_TIMESTAMPS/);
});

test('real optimized checkpoint ZIP preserves all binary weights, int32 scalar iteration and exact Adam beta powers', async () => {
  const checkpoint = await trainedCheckpoint(), before = structuredClone(checkpoint), archive = await packCheckpoint(checkpoint), restored = await unpackCheckpoint(archive);
  assert.equal(restored.training.step, 2); assert.deepEqual(restored, before); assert.deepEqual(checkpoint, before);
  const iteration = restored.optimizer.weights.find(weight => weight.name === 'iter');
  assert(iteration.data instanceof Int32Array); assert.deepEqual(iteration.shape, []); assert.equal(iteration.data[0], 2);
  assert.deepEqual(restored.optimizer.betaPowers, checkpoint.optimizer.betaPowers);
  assert.equal(restored.optimizer.betaPowers.length, 2);
  for (const [index, weight] of restored.weights.entries()) {
    assert(weight.data instanceof Float32Array);
    assert.deepEqual(new Uint8Array(weight.data.buffer), new Uint8Array(checkpoint.weights[index].data.buffer));
  }
});

test('checkpoint transport rejects damaged binary payload, overlapped tensor offsets and unsupported versions', async () => {
  const archive = await packCheckpoint(await trainedCheckpoint());
  const corrupted = await mutateArchive(archive, files => { files['weights.bin'][3] ^= 128; });
  await assert.rejects(unpackCheckpoint(corrupted), /CHECKPOINT_INTEGRITY/);
  const overlapped = await mutateArchive(archive, (_files, manifest) => { manifest.weights[1].offset = manifest.weights[0].offset; });
  await assert.rejects(unpackCheckpoint(overlapped), /CHECKPOINT_OFFSETS/);
  const version = await mutateArchive(archive, (_files, manifest) => { manifest.schemaVersion = 99; });
  await assert.rejects(unpackCheckpoint(version));
  const wrongFormat = await mutateArchive(archive, (_files, manifest) => { manifest.format = 'onnx'; });
  await assert.rejects(unpackCheckpoint(wrongFormat), /CHECKPOINT_INTEGRITY/);
});

test('checkpoint validation rejects missing/forged Adam state and wrong architecture tensor shapes', async () => {
  const checkpoint = await trainedCheckpoint();
  for (const mutate of [state => { delete state.optimizer; }, state => { delete state.optimizer.betaPowers; }, state => { state.optimizer.betaPowers[0] = NaN; }, state => { state.optimizer.weights.pop(); }, state => { state.optimizer.weights[0].data[0] += 1; }, state => { state.weights[0].shape = [1, state.weights[0].data.length]; }, state => { state.training.config.lossVersion = 'unknown'; }, state => { state.training.config.learningRate *= 2; }, state => { state.source = {backboneId: 'yolo26n-features-stride8-v1', sha256: 'd07f0d496799d77361cb621a67c86b3ceaa856eb0007d1a1c8deb1383b14a897'}; }]) {
    const invalid = structuredClone(checkpoint); mutate(invalid); assert.throws(() => validateCheckpoint(invalid));
  }
  const nonfinite = structuredClone(checkpoint); nonfinite.weights[0].data[0] = Infinity;
  assert.throws(() => validateCheckpoint(nonfinite));
});

test('archive parser rejects traversal members and model expansion limits before import', async () => {
  const manifest = strToU8('{}');
  for (const name of ['../escape.txt', '/absolute.txt', 'folder\\escape.txt', 'safe/../../escape.txt', 'constructor/bad']) {
    const malicious = new Blob([zipSync({[name]: new Uint8Array([1]), 'manifest.json': manifest})]);
    await assert.rejects(unpackProject(malicious), /ARCHIVE_UNSAFE/);
  }
  const overExpanded = new Blob([zipSync({'weights.bin': new Uint8Array(16 * 1024 * 1024 + 1), 'manifest.json': manifest}, {level: 9})]);
  await assert.rejects(unpackCheckpoint(overExpanded), /ARCHIVE_UNSAFE/);
  const tooManyMembers = new Blob([zipSync({'manifest.json': manifest, 'weights.bin': new Uint8Array(4), 'unexpected.bin': new Uint8Array(4)})]);
  await assert.rejects(unpackCheckpoint(tooManyMembers), /ARCHIVE_UNSAFE/);
  await assert.rejects(unpackCheckpoint(new Blob([new Uint8Array(20 * 1024 * 1024 + 1)])), /ARCHIVE_SIZE/);
});

test('complete project restoration remaps IDs and preserves source classes, annotations, binary images and checkpoint', async () => {
  const {project, blobs} = await projectFixture(), checkpoint = await trainedCheckpoint(), original = structuredClone(project);
  const archive = await packProject(project, id => blobs.get(id), checkpoint), restored = await unpackProject(archive);
  assert.notEqual(restored.project.id, project.id); assert.deepEqual(project, original); assert.deepEqual(restored.checkpoint, checkpoint);
  assert.deepEqual(restored.project.classes, project.classes); assert.equal(restored.project.images.length, project.images.length);
  for (let index = 0; index < project.images.length; index++) {
    const image = restored.project.images[index], source = project.images[index];
    assert.notEqual(image.id, source.id); assert.deepEqual(image.boxes, source.boxes);
    assert.equal(await sha256(restored.images.get(image.id)), source.sha256);
    assert.deepEqual(new Uint8Array(await restored.images.get(image.id).arrayBuffer()), new Uint8Array(await blobs.get(source.id).arrayBuffer()));
  }
  assert.equal(await datasetDigest(restored.project), await datasetDigest(project));
});

test('project restoration rejects image SHA tampering, missing images, wrong format and checkpoint class conflicts', async () => {
  const {project, blobs} = await projectFixture(), archive = await packProject(project, id => blobs.get(id), await trainedCheckpoint());
  for (const mutate of [
    (files, manifest) => { files[manifest.images[0].path][0] ^= 1; },
    (files, manifest) => { delete files[manifest.images[0].path]; },
    (_files, manifest) => { manifest.schemaVersion = 2; },
    (_files, manifest) => { manifest.project.classes = ['another']; },
  ]) await assert.rejects(unpackProject(await mutateArchive(archive, mutate)));
  await assert.rejects(packProject(project, () => new Blob(['damaged'])), /PROJECT_IMAGE_INTEGRITY/);
});

test('YOLO export writes real original PNG bytes, normalized xywh labels and reviewed empty negatives only', async () => {
  const {project, blobs} = await projectFixture();
  project.images[0].split = 'train'; project.images[1].split = 'validation'; project.images[2].split = 'train';
  project.images.push({...structuredClone(project.images[0]), id: 'unreviewed', reviewed: false});
  const archive = await exportYoloDataset(project, id => blobs.get(id)), files = unzipSync(new Uint8Array(await archive.arrayBuffer()));
  assert.equal(Object.keys(files).filter(name => name.startsWith('images/')).length, 3);
  assert.equal(strFromU8(files['labels/train/000000.txt']).trim(), '0 0.20000000 0.25000000 0.20000000 0.30000000');
  assert.equal(strFromU8(files['labels/val/000001.txt']).trim(), '0 0.20000000 0.25000000 0.20000000 0.30000000');
  assert.equal(files['labels/train/000002.txt'].length, 0);
  assert.equal(await sha256(files['images/train/000000.png']), project.images[0].sha256);
  assert.match(strFromU8(files['data.yaml']), /names:\n  - "person"/);
});

test('SaveQueue stays dirty until its actual deferred save promise commits, then drains the newest edit', async () => {
  // This is an injected save-state unit test, not a claim that IndexedDB was run.
  const first = deferred(), second = deferred(), calls = [], states = [];
  const queue = new SaveQueue(value => { calls.push(value); return calls.length === 1 ? first.promise : second.promise; }, state => states.push(state));
  const initial = {value: 1}; queue.enqueue(initial); initial.value = 99;
  const flush = queue.flush(); await turn();
  assert(queue.dirty); assert.deepEqual(calls, [{value: 1}]); assert(!states.includes('saved'));
  queue.enqueue({value: 2}); queue.enqueue({value: 3});
  first.resolve(); await turn();
  assert(queue.dirty); assert.deepEqual(calls, [{value: 1}, {value: 3}]); assert(!states.includes('saved'));
  second.resolve(); await flush;
  assert.equal(queue.dirty, false); assert.equal(states.at(-1), 'saved');
});

test('SaveQueue deferred failure preserves unsaved data and retries it without a false saved state', async () => {
  const first = deferred(), calls = [], states = [];
  const queue = new SaveQueue(value => { calls.push(value); return calls.length === 1 ? first.promise : Promise.resolve(); }, state => states.push(state));
  queue.enqueue({value: 'important annotation'}); const flushing = queue.flush();
  first.reject(new Error('real injected storage commit failure'));
  await assert.rejects(flushing, /storage commit failure/);
  assert(queue.dirty); assert.equal(states.at(-1), 'error'); assert(!states.includes('saved'));
  await queue.flush(); assert.equal(queue.dirty, false); assert.deepEqual(calls[1], calls[0]); assert.equal(states.at(-1), 'saved');
});

test('SaveQueue synchronous storage validation failure can be retried and retains a later queued edit', async () => {
  let fail = true; const calls = [], queue = new SaveQueue(value => { calls.push(value); if (fail) throw new Error('validation failure'); return Promise.resolve(); });
  queue.enqueue({value: 1}); await assert.rejects(queue.flush(), /validation failure/); assert(queue.dirty);
  queue.enqueue({value: 2}); fail = false;
  await queue.flush(); assert.equal(queue.dirty, false); assert.deepEqual(calls, [{value: 1}, {value: 2}]);
});

test('media decoder rejects empty/oversized actual Blobs before decoding or allocating a canvas', async () => {
  await assert.rejects(decodeImage(new Blob()), /IMAGE_FILE_TOO_LARGE/);
  await assert.rejects(decodeImage(new Blob([new Uint8Array(32 * 1024 * 1024 + 1)])), /IMAGE_FILE_TOO_LARGE/);
});
