import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import * as ort from 'onnxruntime-web';
import {exportOnnx, sameUpperPadding} from '../src/model/export-onnx.js';
import {BACKBONE, verifyBackbone} from '../src/model/backbone.js';
import {readGraph, fields, decodeString, varint} from '../src/model/onnx-proto.js';

ort.env.wasm.numThreads = 1;
await tf.setBackend('cpu');
await tf.ready();
const backbone = new Uint8Array(await readFile(new URL('../public/models/yolo26n-features-128.onnx', import.meta.url)));
let seed = 20261021;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 4294967296; }
function checkpoint(architecture, resolution = 128, count = 2) {
  const weights = [], add = (name, shape, scale = .1) => weights.push({name, shape, dtype: 'float32', data: Float32Array.from({length: shape.reduce((a, b) => a * b, 1)}, () => (random() - .5) * scale)});
  if (architecture === 'tiny-grid-v1') {
    let incoming = 3;
    for (const [index, outgoing] of [8, 12, 16].entries()) {
      add(`conv${index + 1}/kernel`, [3, 3, incoming, outgoing], .35); add(`conv${index + 1}/bias`, [outgoing]); incoming = outgoing;
    }
  }
  add('head/hidden/kernel', [architecture === 'tiny-grid-v1' ? 16 : 128, 32], .1);
  add('head/hidden/bias', [32]); add('head/output/kernel', [32, 2 * (5 + count)], .2); add('head/output/bias', [2 * (5 + count)], .5);
  return {schemaVersion: 1, architecture, resolution, slots: 2, classes: Array.from({length: count}, (_, index) => `class-${index}`), weights, source: {backboneId: BACKBONE.id, sha256: BACKBONE.sha256}, training: {step: 0}};
}

async function tfPrediction(checkpoint, image, pretrainedFeatures) {
  const {resolution, classes, weights} = checkpoint, grid = resolution / 8, anchors = grid * grid * 2;
  const get = name => { const weight = weights.find(item => item.name === name); return tf.tensor(weight.data, weight.shape, 'float32'); };
  const output = tf.tidy(() => {
    let features;
    if (pretrainedFeatures) features = tf.transpose(tf.tensor(pretrainedFeatures, [1, 128, grid, grid]), [0, 2, 3, 1]);
    else {
      features = tf.transpose(tf.tensor(image, [1, 3, resolution, resolution]), [0, 2, 3, 1]);
      for (let index = 1; index <= 3; index++) features = tf.relu(tf.add(tf.conv2d(features, get(`conv${index}/kernel`), 2, 'same'), get(`conv${index}/bias`)));
    }
    const hidden = tf.relu(tf.add(tf.matMul(features, get('head/hidden/kernel')), get('head/hidden/bias')));
    const raw = tf.reshape(tf.add(tf.matMul(hidden, get('head/output/kernel')), get('head/output/bias')), [1, anchors, 5 + classes.length]);
    const offsets = new Float32Array(anchors * 4);
    for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) for (let slot = 0; slot < 2; slot++) {
      const index = ((y * grid + x) * 2 + slot) * 4; offsets[index] = x * 8; offsets[index + 1] = y * 8;
    }
    const boxes = tf.add(tf.mul(tf.sigmoid(tf.slice(raw, [0, 0, 0], [1, anchors, 4])), tf.tensor([8, 8, resolution, resolution], [1, 1, 4])), tf.tensor(offsets, [1, anchors, 4]));
    const scores = tf.mul(tf.sigmoid(tf.slice(raw, [0, 0, 4], [1, anchors, 1])), tf.softmax(tf.slice(raw, [0, 0, 5], [1, anchors, classes.length]), -1));
    return tf.transpose(tf.concat([boxes, scores], 2), [0, 2, 1]);
  });
  try { return Float32Array.from(await output.data()); } finally { output.dispose(); }
}

test('bundled pretrained feature artifact is pinned, valid and not an entire trainable YOLO26 model', async () => {
  assert.equal(backbone.length, BACKBONE.bytes);
  assert.equal(createHash('sha256').update(backbone).digest('hex'), BACKBONE.sha256);
  assert.equal(await verifyBackbone(backbone), backbone);
  const damaged = backbone.slice(); damaged[damaged.length - 1] ^= 1;
  await assert.rejects(verifyBackbone(damaged), /SHA-256/);
  const graph = readGraph(backbone);
  assert.equal(graph.nodes.length, BACKBONE.nodeCount); assert.equal(graph.initializers.length, BACKBONE.initializerCount);
  const session = await ort.InferenceSession.create(backbone, {executionProviders: ['wasm']});
  try {
    const input = new ort.Tensor('float32', Float32Array.from({length: 3 * 128 * 128}, (_, index) => (index % 251) / 255), [1, 3, 128, 128]);
    const outputs = await session.run({images: input}); const output = outputs[BACKBONE.output];
    assert.deepEqual(output.dims, [1, 128, 16, 16]); assert(output.data.every(Number.isFinite));
    assert(output.data.some(value => Math.abs(value) > .01), 'Pretrained feature inference must produce real nonzero features.');
    output.dispose(); input.dispose();
  } finally { await session.release(); }
});

test('SAME_UPPER pads have the asymmetric trailing pad required by TensorFlow stride two', () => {
  assert.deepEqual(sameUpperPadding(128), {before: 0, after: 1, output: 64});
  assert.deepEqual(sameUpperPadding(64), {before: 0, after: 1, output: 32});
  assert.deepEqual(sameUpperPadding(127), {before: 1, after: 1, output: 64});
  assert.deepEqual([...varint(300)], [172, 2]);
});

for (const architecture of ['tiny-grid-v1', 'yolo26n-frozen-head-v1']) for (const count of [1, 3]) {
  test(`${architecture} ${count}-class untrained numeric fixture exports a real ONNX matching TensorFlow predictions`, async () => {
    // Deliberately untrained weights test export arithmetic only. This is not
    // evidence of training success, accuracy or generalization.
    const state = checkpoint(architecture, 128, count), size = state.resolution;
    const image = Float32Array.from({length: 3 * size * size}, (_, index) => ((index * 13 + 7) % 256) / 255);
    let features;
    if (architecture === 'yolo26n-frozen-head-v1') {
      const featureSession = await ort.InferenceSession.create(backbone, {executionProviders: ['wasm']});
      try { const result = await featureSession.run({images: new ort.Tensor('float32', image, [1, 3, size, size])}); features = Float32Array.from(result[BACKBONE.output].data); result[BACKBONE.output].dispose(); }
      finally { await featureSession.release(); }
    }
    const expected = await tfPrediction(state, image, features), exported = exportOnnx(state, {backboneBytes: backbone});
    const modelFields = fields(exported), metadata = Object.fromEntries(modelFields.filter(field => field.number === 14).map(field => { const entries = fields(field.value); return [decodeString(entries.find(item => item.number === 1).value), decodeString(entries.find(item => item.number === 2).value)]; }));
    assert.equal(metadata.task, 'detect'); assert.equal(metadata.architecture, architecture); assert.equal(metadata.trained_steps, '0'); assert.equal(metadata.end2end, 'false'); assert.deepEqual(JSON.parse(metadata.names), state.classes);
    const session = await ort.InferenceSession.create(exported, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    try {
      const input = new ort.Tensor('float32', image, [1, 3, size, size]), results = await session.run({images: input}), actual = results.output0;
      assert.deepEqual(actual.dims, [1, 4 + count, (size / 8) ** 2 * 2]);
      let maximumAbsoluteError = 0, squaredError = 0;
      for (let index = 0; index < expected.length; index++) { const error = Math.abs(expected[index] - actual.data[index]); maximumAbsoluteError = Math.max(maximumAbsoluteError, error); squaredError += error * error; }
      assert(maximumAbsoluteError < .0002, `ONNX/TFJS max absolute error ${maximumAbsoluteError} must be < .0002 input pixels/score.`);
      assert(actual.data.every(Number.isFinite));
      console.log('REAL_ONNX_TFJS_PARITY', JSON.stringify({architecture, classes: count, backend: 'ORT WASM / TFJS CPU', training: 'none; numerical fixture only', modelBytes: exported.length, maxAbsoluteError: maximumAbsoluteError, rmse: Math.sqrt(squaredError / expected.length)}));
      actual.dispose(); input.dispose();
    } finally { await session.release(); }
  });
}

test('export rejects incompatible, corrupt and misleading checkpoints', () => {
  const original = checkpoint('tiny-grid-v1');
  assert.throws(() => exportOnnx({...original, architecture: 'yolo26-from-scratch'}), /Unsupported/);
  assert.throws(() => exportOnnx({...original, classes: ['same', 'same']}), /class names/);
  assert.throws(() => exportOnnx({...original, weights: original.weights.slice(1)}), /conv1\/kernel/);
  assert.throws(() => exportOnnx({...original, slots: 1}), /two detection slots/);
  const invalid = checkpoint('tiny-grid-v1'); invalid.weights[0].data[0] = NaN;
  assert.throws(() => exportOnnx(invalid), /Invalid checkpoint weight/);
  const frozen = checkpoint('yolo26n-frozen-head-v1');
  assert.throws(() => exportOnnx(frozen), /requires verified/);
  assert.throws(() => exportOnnx({...frozen, source: {backboneId: BACKBONE.id, sha256: 'wrong'}}, {backboneBytes: backbone}), /pinned backbone/);
  assert.throws(() => exportOnnx({...frozen, source: {backboneId: 'yolo26n-features-stride8-v1', sha256: 'd07f0d496799d77361cb621a67c86b3ceaa856eb0007d1a1c8deb1383b14a897'}}, {backboneBytes: backbone}), /pinned backbone/);
});

for (const mode of ['scratch', 'pretrained']) test(`${mode} real gradient-updated checkpoint exports its learned parameters, not initialization`, async () => {
  const {TrainerEngine} = await import('../src/training/engine.js');
  const size = 128;
  const sample = (id, x, split) => {
    const pixels = new Uint8Array(size * size * 3).fill(32);
    for (let y = 35; y < 70; y++) for (let px = x; px < x + 30; px++) { const offset = (y * size + px) * 3; pixels[offset] = 230; pixels[offset + 1] = 80; pixels[offset + 2] = 40; }
    return {id, width: size, height: size, pixels, split, boxes: [{id: `${id}-box`, classId: 0, x, y: 35, width: 30, height: 35}]};
  };
  const engine = new TrainerEngine({backboneBytes: backbone});
  let session;
  try {
    const trained = await engine.train({samples: [sample('train-a', 25, 'train'), sample('train-b', 60, 'train'), sample('held-out', 42, 'val')], classes: ['red object'], config: {mode, backend: 'cpu', resolution: 128, epochs: 2, batchSize: 1, learningRate: .003, augment: false, seed: 12345}});
    assert(trained.evidence.parametersChanged, 'Actual optimization must alter a saved model weight.');
    assert.notEqual(trained.evidence.initialWeightHash, trained.evidence.finalWeightHash);
    assert(trained.checkpoint.training.step >= 4);
    assert(trained.evidence.maxGradientNorm > 0);
    const input = Float32Array.from({length: 3 * size * size}, (_, index) => ((index * 19 + 11) % 256) / 255);
    let features;
    if (mode === 'pretrained') {
      const result = await engine.featureSession.run({images: new ort.Tensor('float32', input, [1, 3, size, size])});
      features = Float32Array.from(result[BACKBONE.output].data); result[BACKBONE.output].dispose();
    }
    const expected = await tfPrediction(trained.checkpoint, input, features), exported = await engine.export({checkpoint: trained.checkpoint});
    session = await ort.InferenceSession.create(exported.bytes, {executionProviders: ['wasm']});
    const result = await session.run({images: new ort.Tensor('float32', input, [1, 3, size, size])});
    const actual = result.output0;
    let maximumAbsoluteError = 0;
    for (let index = 0; index < expected.length; index++) maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(expected[index] - actual.data[index]));
    assert(maximumAbsoluteError < .0002, `Learned ONNX/TFJS values differ by ${maximumAbsoluteError}.`);
    actual.dispose();
    console.log('REAL_TRAINED_ONNX_PARITY', JSON.stringify({mode, steps: trained.checkpoint.training.step, initialWeightHash: trained.evidence.initialWeightHash, finalWeightHash: trained.evidence.finalWeightHash, maximumAbsoluteError, dataset: 'Generated rectangles; gradient/export regression only, not real-world accuracy evidence'}));
  } finally { if (session) await session.release(); await engine.dispose(); }
});
