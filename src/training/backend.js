import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import {TrainingError} from './targets.js';
import {detectorForward, detectorLoss} from './model-ops.js';

let probeSequence = 0;
const loaded = new Set(['cpu']);
const cache = new Map();
const references = new Map();
export {tf};

async function registerBackend(name, baseUrl = '/') {
  if (loaded.has(name)) return;
  if (name === 'wasm') {
    const wasm = await import('@tensorflow/tfjs-backend-wasm');
    wasm.setThreadsCount(1); // Static GitHub Pages does not promise cross-origin isolation.
    // Node's package entry resolves its installed WASM files beside __dirname.
    // Static browser workers must use the explicitly hosted same-origin assets.
    if (baseUrl && !tf.env().get('IS_NODE')) wasm.setWasmPaths(`${baseUrl.replace(/\/$/, '')}/runtime/tfjs/`);
  } else if (name === 'webgl') await import('@tensorflow/tfjs-backend-webgl');
  else if (name === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu');
  else throw new TrainingError('Unknown training backend.', 'UNSUPPORTED_BACKEND');
  loaded.add(name);
}

/** A gradient/update test on the actual architecture primitives; no capability guess. */
export async function probeBackend(name, {mode = 'pretrained', baseUrl = '/', refresh = false, batchSize = 1, classCount = 2} = {}) {
  const key = `${name}:${mode}:${batchSize}:${classCount}:${baseUrl}`;
  if (!refresh && cache.has(key)) return cache.get(key);
  const referenceKey = `${mode}:${batchSize}:${classCount}`;
  if (name !== 'cpu' && !references.has(referenceKey)) {
    const cpu = await probeBackend('cpu', {mode, baseUrl, batchSize, classCount});
    if (!cpu.supported) return {backend: name, mode, supported: false, verified: false, reason: 'The CPU numerical reference could not run.'};
  }
  const prior = tf.getBackend();
  const variables = [], prefix = `probe-${++probeSequence}/`;
  let optimizer, loss, grads;
  const started = performance.now();
  try {
    await registerBackend(name, baseUrl);
    if (!await tf.setBackend(name)) throw new Error(`Backend ${name} failed to initialize.`);
    await tf.ready();
    const variable = (values, shape, suffix) => { const initial = tf.tensor(values, shape); const value = tf.variable(initial, true, prefix + suffix); initial.dispose(); variables.push(value); return value; };
    const map = new Map(), scratch = mode === 'scratch', channels = scratch ? 16 : 128;
    const add = (suffix, shape, value) => map.set(suffix, variable(new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(value), shape, suffix));
    if (scratch) for (const [i, incoming, outgoing] of [[1, 3, 8], [2, 8, 12], [3, 12, 16]]) { add(`conv${i}/kernel`, [3, 3, incoming, outgoing], .02); add(`conv${i}/bias`, [outgoing], .01); }
    add('head/hidden/kernel', [channels, 32], .02); add('head/hidden/bias', [32], .01);
    add('head/output/kernel', [32, 2 * (5 + classCount)], .02); add('head/output/bias', [2 * (5 + classCount)], .01);
    const head = map.get('head/output/kernel');
    optimizer = tf.train.adam(.003);
    ({value: loss, grads} = tf.variableGrads(() => tf.tidy(() => {
      const input = tf.ones(scratch ? [batchSize, 128, 128, 3] : [batchSize, 16, 16, 128]);
      const logits = detectorForward(tf, suffix => map.get(suffix), input, scratch ? 'tiny-grid-v1' : 'yolo26n-frozen-head-v1', 128, classCount);
      const target = new Float32Array(batchSize * 16 * 16 * 2 * (5 + classCount));
      for (let batch = 0; batch < batchSize; batch++) { const offset = batch * 16 * 16 * 2 * (5 + classCount); target.set([.5, .5, .2, .2, 1, 1], offset); }
      return detectorLoss(tf, logits, tf.tensor(target, [batchSize, 16, 16, 2, 5 + classCount]), classCount);
    }), variables));
    const before = Array.from(await head.data()), lossValue = (await loss.data())[0];
    const values = Object.values(grads);
    const arrays = await Promise.all(values.map(tensor => tensor.data()));
    if (!Number.isFinite(lossValue) || arrays.some(array => !array.every(Number.isFinite)) || !arrays.some(array => array.some(value => Math.abs(value) > 1e-12))) throw new Error('The real gradient probe returned invalid or zero gradients.');
    let numericalReference = {backend: 'cpu', lossRelativeError: 0, maximumGradientRelativeError: 0, tolerance: .002};
    if (name === 'cpu') references.set(referenceKey, {loss: lossValue, gradients: arrays.map(array => Float32Array.from(array))});
    else {
      const reference = references.get(referenceKey);
      numericalReference.lossRelativeError = Math.abs(lossValue - reference.loss) / Math.max(1, Math.abs(reference.loss));
      if (arrays.length !== reference.gradients.length) throw new Error('The GPU gradient structure differs from the real CPU reference.');
      arrays.forEach((array, index) => {
        if (array.length !== reference.gradients[index].length) throw new Error('The gradient shape differs from the real CPU reference.');
        for (let i = 0; i < array.length; i++) numericalReference.maximumGradientRelativeError = Math.max(numericalReference.maximumGradientRelativeError, Math.abs(array[i] - reference.gradients[index][i]) / Math.max(1, Math.abs(reference.gradients[index][i])));
      });
      if (numericalReference.lossRelativeError > numericalReference.tolerance || numericalReference.maximumGradientRelativeError > numericalReference.tolerance) throw new Error(`Training arithmetic disagrees with CPU: loss error ${numericalReference.lossRelativeError}, gradient error ${numericalReference.maximumGradientRelativeError}.`);
    }
    optimizer.applyGradients(grads);
    const after = await head.data();
    if (!after.every(Number.isFinite) || !after.some((value, i) => Math.abs(value - before[i]) > 1e-9)) throw new Error('The optimizer did not update parameters.');
    const result = {backend: name, mode, batchSize, classCount, supported: true, verified: true, loss: lossValue, parameterChanged: true, numericalReference, milliseconds: performance.now() - started, scope: mode === 'scratch' ? 'actual three-convolution detector forward, loss, gradients, Adam update' : 'actual frozen-feature spatial head forward, loss, gradients, Adam update'};
    cache.set(key, result); return result;
  } catch (error) {
    const result = {backend: name, mode, supported: false, verified: false, reason: error.message, milliseconds: performance.now() - started};
    cache.set(key, result); return result;
  } finally {
    loss?.dispose(); if (grads) Object.values(grads).forEach(tensor => tensor.dispose());
    optimizer?.dispose(); variables.forEach(value => value.dispose());
    if (prior && tf.findBackend(prior)) await tf.setBackend(prior);
  }
}

export async function probeBackends(options = {}) {
  const results = [];
  for (const backend of ['cpu', 'wasm', 'webgl', 'webgpu']) results.push(await probeBackend(backend, options));
  return results;
}

export async function selectBackend(requested = 'auto', options = {}) {
  const candidates = requested === 'auto' ? ['webgpu', 'webgl', 'wasm', 'cpu'] : [requested];
  const attempts = [];
  for (const backend of candidates) {
    const probe = await probeBackend(backend, options); attempts.push(probe);
    if (probe.supported) { await tf.setBackend(backend); await tf.ready(); return {...probe, attempts}; }
  }
  throw new TrainingError(`This training mode is unavailable: ${attempts.map(attempt => `${attempt.backend}: ${attempt.reason}`).join('; ')}`, 'UNSUPPORTED_BACKEND');
}
