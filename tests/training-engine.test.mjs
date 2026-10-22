import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {TrainerEngine, validateCheckpoint, normalizeConfig} from '../src/training/engine.js';
import {probeBackend, tf} from '../src/training/backend.js';

function square(id, x, y, {split = 'train', channel = 0, classId = 0} = {}) {
  const pixels = new Float32Array(128 * 128 * 3).fill(.05);
  for (let yy = y; yy < y + 12; yy++) for (let xx = x; xx < x + 12; xx++) pixels[(yy * 128 + xx) * 3 + channel] = 1;
  return {id, width: 128, height: 128, pixels, boxes: [{id: `box-${id}`, classId, x, y, width: 12, height: 12}], split};
}
const config = {mode: 'pretrained', backend: 'cpu', epochs: 50, batchSize: 1, augment: false, learningRate: .01, seed: 12345};
const backboneBytes = await readFile(new URL('../public/models/yolo26n-features-128.onnx', import.meta.url));
const dataset = () => [square('a', 18, 18), square('b', 66, 26), square('c', 34, 74), square('held-out', 82, 82, {split: 'val'})];

test('actual pretrained spatial detector learns boxes, saves/reloads and detects an unseen image', {timeout: 60000}, async () => {
  const engine = new TrainerEngine({backboneBytes}), restored = new TrainerEngine({backboneBytes});
  const beforeTensors = tf.memory().numTensors;
  try {
    const samples = dataset(), result = await engine.train({samples, classes: ['red square'], config});
    assert.equal(result.status, 'completed'); assert.equal(result.evidence.steps, 150);
    assert(result.evidence.parametersChanged); assert(result.evidence.firstGradient.finite); assert(result.evidence.firstGradient.norm > 0);
    assert(result.evidence.lastLoss < result.evidence.firstLoss * .1);
    assert.equal(result.evidence.initialMetrics.recall50, 0);
    assert.equal(result.metrics.validationImages, undefined); // Never fabricate a metric field.
    assert.equal(result.metrics.images, 1); assert.equal(result.metrics.truePositives, 1); assert.equal(result.metrics.falsePositives, 0); assert(result.metrics.meanMatchedIou > .8);
    validateCheckpoint(result.checkpoint);
    const original = await engine.predict({...samples[3], confidence: .25});
    const loaded = await restored.predict({...samples[3], checkpoint: structuredClone(result.checkpoint), backend: 'cpu', confidence: .25});
    assert.deepEqual(loaded.items, original.items);
    console.log('REAL_TRAINING_EVIDENCE', JSON.stringify({case: 'generated-shapes-only-not-natural-photo-accuracy', ...result.evidence, metrics: result.metrics, reloadedPrediction: loaded.items}));
  } finally { await engine.dispose(); await restored.dispose(); }
  assert.equal(tf.memory().numTensors, beforeTensors, 'Dispose must free variables, Adam moments, and temporary tensors');
});

test('CPU scratch convolutions update; different images change the learned model', {timeout: 60000}, async () => {
  const a = new TrainerEngine(), b = new TrainerEngine();
  try {
    const settings = {...config, mode: 'scratch', epochs: 3};
    const left = await a.train({samples: [square('a', 18, 18)], classes: ['object'], config: settings});
    const right = await b.train({samples: [square('b', 18, 18, {channel: 1})], classes: ['object'], config: settings});
    assert(left.evidence.parametersChanged); assert(right.evidence.parametersChanged);
    assert.notEqual(left.evidence.finalWeightHash, right.evidence.finalWeightHash);
    for (let i = 1; i <= 3; i++) {
      const first = left.checkpoint.weights.find(w => w.name === `conv${i}/kernel`).data;
      const second = right.checkpoint.weights.find(w => w.name === `conv${i}/kernel`).data;
      assert(first.some((value, index) => value !== second[index]), `conv${i} is genuinely data-dependent`);
    }
    assert.equal(left.metrics.available, false);
  } finally { await a.dispose(); await b.dispose(); }
});

test('pause at a real batch boundary preserves Adam and RNG; resumed CPU training equals continuous training', {timeout: 60000}, async () => {
  const complete = new TrainerEngine({backboneBytes}), split = new TrainerEngine({backboneBytes}), resumed = new TrainerEngine({backboneBytes});
  const settings = {...config, epochs: 3, augment: true}, samples = dataset();
  split.onProgress = event => { if (event.state === 'training' && event.step === 2) split.pause(); };
  try {
    const expected = await complete.train({samples, classes: ['red'], config: settings});
    const paused = await split.train({samples, classes: ['red'], config: settings});
    assert.equal(paused.status, 'paused'); assert.equal(paused.checkpoint.training.step, 2); assert.equal(paused.checkpoint.training.batchOffset, 2);
    const restoredSamples = samples.map((sample, i) => ({...sample, id: `restored-${i}`, boxes: sample.boxes.map(box => ({...box, id: `new-box-${i}`}))}));
    const actual = await resumed.train({samples: restoredSamples, classes: ['red'], config: settings, checkpoint: paused.checkpoint});
    assert.equal(actual.status, 'completed'); assert.equal(actual.evidence.datasetChanged, false);
    assert.equal(actual.checkpoint.training.rngState, expected.checkpoint.training.rngState);
    assert.equal(actual.evidence.finalWeightHash, expected.evidence.finalWeightHash);
    assert.deepEqual(actual.checkpoint.optimizer, expected.checkpoint.optimizer);
    console.log('REAL_RESUME_EVIDENCE', JSON.stringify({continuousSteps: expected.checkpoint.training.step, pausedStep: 2, resumedSteps: actual.evidence.steps, continuousHash: expected.evidence.finalWeightHash, resumedHash: actual.evidence.finalWeightHash, rngState: actual.checkpoint.training.rngState}));
  } finally { await complete.dispose(); await split.dispose(); await resumed.dispose(); }
});

test('invalid checkpoint/config/data fail explicitly and keep a previous valid checkpoint', {timeout: 30000}, async () => {
  const engine = new TrainerEngine({backboneBytes});
  try {
    const good = await engine.train({samples: [square('a', 18, 18)], classes: ['red'], config: {...config, epochs: 1}}), previousHash = good.evidence.finalWeightHash;
    const broken = structuredClone(good.checkpoint); broken.optimizer.weights.pop();
    assert.throws(() => validateCheckpoint(broken), error => error.code === 'INVALID_CHECKPOINT');
    const badValue = structuredClone(good.checkpoint); badValue.weights[0].data[0] = NaN;
    assert.throws(() => validateCheckpoint(badValue), error => error.code === 'INVALID_CHECKPOINT');
    const oldLowQuality = structuredClone(good.checkpoint); delete oldLowQuality.training.config.preprocessingVersion;
    await assert.rejects(engine.load(oldLowQuality, {backend: 'cpu'}), error => error.code === 'INVALID_CHECKPOINT');
    const incompatibleScale = structuredClone(good.checkpoint); incompatibleScale.training.config.preprocessingVersion = 'canvas-low-v0';
    assert.throws(() => validateCheckpoint(incompatibleScale), error => error.code === 'INVALID_CHECKPOINT');
    for (const field of ['epochOrder', 'epochLossSum', 'epochBatches']) {
      const incomplete = structuredClone(good.checkpoint); delete incomplete.training[field];
      assert.throws(() => validateCheckpoint(incomplete), error => error.code === 'INVALID_CHECKPOINT', `Missing ${field} must not silently change resumed training.`);
    }
    await assert.rejects(engine.predict({pixels: new Uint8Array(64 * 64 * 3), width: 64, height: 64}), error => error.code === 'PIXEL_PREPROCESSING_UNSUPPORTED');
    assert.throws(() => normalizeConfig({resolution: 256}), error => error.code === 'UNSUPPORTED_RESOLUTION');
    await assert.rejects(engine.train({samples: [{...square('a', 18, 18), boxes: []}], classes: ['red'], config: {...config, epochs: 1}}), error => error.code === 'NO_TRAIN_BOXES');
    assert.equal(engine.lastCheckpoint.training.step, good.checkpoint.training.step);
    await engine.load(engine.lastCheckpoint, {backend: 'cpu'}); assert.equal(await engine.weightHash(), previousHash);
    await assert.rejects(engine.train({samples: [square('a', 18, 18)], classes: ['different'], checkpoint: good.checkpoint, config: {...config, epochs: 1}}), error => error.code === 'CHECKPOINT_CLASS_MISMATCH');
    await assert.rejects(engine.train({samples: [square('train', 18, 18), square('val-copy', 18, 18, {split: 'val'})], classes: ['red'], config: {...config, epochs: 1}}), error => error.code === 'DATA_SPLIT_LEAKAGE');
    await assert.rejects(engine.train({samples: [square('must-stay-independent', 18, 18, {split: 'test'})], classes: ['red'], config: {...config, epochs: 1}}), error => error.code === 'INVALID_DATA_SPLIT');
  } finally { await engine.dispose(); }
});

test('all offered CPU batch sizes execute actual full detector gradient/update probes', {timeout: 60000}, async () => {
  for (const batchSize of [1, 2, 4, 8]) {
    const result = await probeBackend('cpu', {mode: 'scratch', batchSize});
    assert.equal(result.supported, true, result.reason); assert.equal(result.parameterChanged, true); assert.equal(result.batchSize, batchSize);
  }
});

test('Node WASM uses actual package assets; the head trains while missing convolution gradients remain unsupported', async () => {
  const head = await probeBackend('wasm', {mode: 'pretrained', baseUrl: '/', refresh: true});
  assert.equal(head.supported, true, head.reason); assert.equal(head.parameterChanged, true);
  assert(head.numericalReference.maximumGradientRelativeError < .002);
  const scratch = await probeBackend('wasm', {mode: 'scratch', baseUrl: '/', refresh: true});
  assert.equal(scratch.supported, false); assert.match(scratch.reason, /Conv2DBackpropFilter/);
  assert.doesNotMatch(scratch.reason, /ENOENT|no such file|runtime\/tfjs/i);
});

test('two tiny targets in one spatial cell learn distinct classes and non-collapsed boxes on a new image', {timeout: 60000}, async () => {
  const pair = (id, shift, split = 'train') => {
    const pixels = new Float32Array(128 * 128 * 3).fill(.05);
    const boxes = [{classId: 0, x: 8 + shift, y: 8, width: 4, height: 4}, {classId: 1, x: 12 + shift, y: 12, width: 4, height: 4}];
    for (const box of boxes) for (let y = box.y; y < box.y + box.height; y++) for (let x = box.x; x < box.x + box.width; x++) pixels[(y * 128 + x) * 3 + box.classId] = 1;
    return {id, width: 128, height: 128, pixels, boxes, split};
  };
  const samples = [pair('a', 0), pair('b', 16), pair('c', 32), pair('unseen', 64, 'val')], engine = new TrainerEngine({backboneBytes});
  try {
    const result = await engine.train({samples, classes: ['red', 'green'], config: {...config, epochs: 60}});
    const {items} = await engine.predict({...samples[3], confidence: .25});
    assert.equal(items.length, 2); assert.deepEqual(items.map(item => item.classId).sort(), [0, 1]);
    assert.equal(result.metrics.truePositives, 2); assert.equal(result.metrics.falsePositives, 0);
    assert(result.metrics.meanMatchedIou > .8);
    assert(items.every(item => item.box.width > 3 && item.box.height > 3), 'Small-box regression must not collapse while loss appears low');
  } finally { await engine.dispose(); }
});

test('additional epochs apply after a changed dataset resets its order; paused absolute targets are unchanged', {timeout: 30000}, async () => {
  const engine = new TrainerEngine({backboneBytes}), next = new TrainerEngine({backboneBytes}), changed = new TrainerEngine({backboneBytes});
  try {
    const initial = await engine.train({samples: [square('first', 18, 18)], classes: ['red'], config: {...config, epochs: 2}});
    const continued = await next.train({samples: [square('restored-id', 18, 18)], classes: ['red'], config: {...config, epochs: 30}, checkpoint: initial.checkpoint, additionalEpochs: 1});
    assert.equal(continued.checkpoint.training.epoch, 3); assert.equal(continued.checkpoint.training.config.epochs, 3); assert.equal(continued.evidence.steps, 1); assert.equal(continued.evidence.datasetChanged, false);
    const updated = await changed.train({samples: [square('new-photo', 34, 18)], classes: ['red'], config: {...config, epochs: 30}, checkpoint: initial.checkpoint, additionalEpochs: 1});
    assert.equal(updated.checkpoint.training.epoch, 1); assert.equal(updated.checkpoint.training.config.epochs, 1); assert.equal(updated.evidence.steps, 1); assert.equal(updated.evidence.datasetChanged, true);
    await assert.rejects(engine.train({samples: [square('a', 18, 18)], classes: ['red'], config, additionalEpochs: 1}), error => error.code === 'INVALID_CONFIG');
    await assert.rejects(engine.train({samples: [square('a', 18, 18)], classes: ['red'], config, checkpoint: initial.checkpoint, additionalEpochs: 1000}), error => error.code === 'INVALID_CONFIG');
  } finally { await engine.dispose(); await next.dispose(); await changed.dispose(); }
});

test('the UI validation split enters evaluation and never contributes a training step', {timeout: 30000}, async () => {
  const engine = new TrainerEngine({backboneBytes});
  try {
    const result = await engine.train({samples: [square('train', 18, 18), square('ui-validation', 66, 26, {split: 'validation'})], classes: ['red'], config: {...config, epochs: 1}});
    assert.equal(result.evidence.steps, 1); assert.equal(result.evidence.trainingImages, 1); assert.equal(result.evidence.validationImages, 1); assert.equal(result.metrics.images, 1);
  } finally { await engine.dispose(); }
});
