// Shared by archive/storage validation and the training worker. Keep this
// module free of TensorFlow/ONNX Runtime imports so opening a project does not
// initialize a training backend on the main thread.
import {BACKBONE} from './backbone.js';
import {TrainingError} from '../training/targets.js';

export const ARCHITECTURES = Object.freeze({pretrained: 'yolo26n-frozen-head-v1', scratch: 'tiny-grid-v1'});
const invalid = message => { throw new TrainingError(message, 'INVALID_CHECKPOINT'); };
const integer = (value, low, high) => Number.isInteger(value) && value >= low && value <= high;
const sameShape = (a, b) => Array.isArray(a) && a.length === b.length && a.every((value, index) => value === b[index]);

export function weightShapes(architecture, classCount) {
  if (!Object.values(ARCHITECTURES).includes(architecture) || !integer(classCount, 1, 32)) invalid('This model architecture/class count is not a supported training checkpoint.');
  const shapes = architecture === ARCHITECTURES.scratch ? {'conv1/kernel': [3, 3, 3, 8], 'conv1/bias': [8], 'conv2/kernel': [3, 3, 8, 12], 'conv2/bias': [12], 'conv3/kernel': [3, 3, 12, 16], 'conv3/bias': [16]} : {};
  return {...shapes, 'head/hidden/kernel': [architecture === ARCHITECTURES.scratch ? 16 : 128, 32], 'head/hidden/bias': [32], 'head/output/kernel': [32, 2 * (5 + classCount)], 'head/output/bias': [2 * (5 + classCount)]};
}

function validateClasses(classes) {
  if (!Array.isArray(classes) || !integer(classes.length, 1, 32)) invalid('A checkpoint must contain 1–32 classes.');
  const seen = new Set();
  for (const name of classes) {
    if (typeof name !== 'string' || !name.trim() || name.length > 80 || /[\u0000-\u001f\u007f]/.test(name)) invalid('Checkpoint class names must be nonempty strings of at most 80 characters.');
    const canonical = name.normalize('NFC').trim().toLocaleLowerCase();
    if (seen.has(canonical)) invalid('Checkpoint class names must be unique.');
    seen.add(canonical);
  }
}

function checkedWeight(weight, name, shape, dtype = 'float32') {
  const size = shape.reduce((a, b) => a * b, 1), Type = dtype === 'int32' ? Int32Array : Float32Array;
  if (!weight || weight.name !== name || weight.dtype !== dtype || !sameShape(weight.shape, shape) || !(Array.isArray(weight.data) || weight.data instanceof Type) || weight.data.length !== size) invalid(`Invalid checkpoint tensor: ${name}.`);
  for (const value of weight.data) if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value)) || (dtype === 'int32' && !integer(value, -2147483648, 2147483647))) invalid(`Invalid checkpoint tensor value: ${name}.`);
}

function validateConfig(config, architecture, optimizer) {
  const mode = architecture === ARCHITECTURES.scratch ? 'scratch' : 'pretrained';
  if (!config || config.mode !== mode || config.resolution !== 128 || ![1, 2, 4, 8].includes(config.batchSize) || !integer(config.epochs, 1, 1000) || !Number.isFinite(config.learningRate) || config.learningRate < .000001 || config.learningRate > .1 || config.learningRate !== optimizer.learningRate || typeof config.augment !== 'boolean' || !integer(config.seed, 1, 4294967295) || !['auto', 'cpu', 'wasm', 'webgl', 'webgpu'].includes(config.backend) || config.lossVersion !== 'focal-kl-v1' || config.preprocessingVersion !== 'canvas-high-v1') invalid('The checkpoint training configuration is incomplete or incompatible. Browser image scaling must use canvas-high-v1; earlier low-quality-scaling development checkpoints cannot resume this engine.');
}

function validateHistory(history) {
  if (!Array.isArray(history) || history.length > 10000) invalid('The checkpoint training history is invalid.');
  // Metadata is untrusted too. Bound traversal and reject cycles/nonfinite
  // diagnostic numbers before a ZIP/JSON save could silently convert them.
  let visited = 0;
  const ancestors = new Set();
  const inspect = (value, depth) => {
    if (++visited > 500000 || depth > 16) invalid('The checkpoint history is excessively complex.');
    if (typeof value === 'number' && !Number.isFinite(value)) invalid('The checkpoint history contains a nonfinite value.');
    if (value && typeof value === 'object') {
      if (ancestors.has(value)) invalid('The checkpoint history contains a cycle.');
      ancestors.add(value); for (const item of Object.values(value)) inspect(item, depth + 1); ancestors.delete(value);
    }
  };
  inspect(history, 0);
}

export function validateCheckpoint(checkpoint) {
  if (!checkpoint || checkpoint.schemaVersion !== 1 || checkpoint.resolution !== 128 || checkpoint.slots !== 2) invalid('Unsupported checkpoint version or resolution. Import a Local YOLO Trainer checkpoint, not an inference ONNX.');
  validateClasses(checkpoint.classes);
  const shapes = weightShapes(checkpoint.architecture, checkpoint.classes.length), names = Object.keys(shapes);
  if (!Array.isArray(checkpoint.weights) || checkpoint.weights.length !== names.length || checkpoint.weights.some(weight => !weight || typeof weight.name !== 'string') || new Set(checkpoint.weights.map(weight => weight.name)).size !== names.length) invalid('Checkpoint weights are incomplete or duplicated.');
  names.forEach(name => checkedWeight(checkpoint.weights.find(weight => weight.name === name), name, shapes[name]));

  const state = checkpoint.training, optimizer = checkpoint.optimizer;
  if (!state || !integer(state.epoch, 0, 1000) || !integer(state.batchOffset, 0, 2000) || !integer(state.step, 0, 100_000_000) || !integer(state.rngState, 1, 4294967295) || typeof state.datasetHash !== 'string' || !(state.datasetHash === '' && state.step === 0 || /^[a-f0-9]{64}$/.test(state.datasetHash))) invalid('The checkpoint training state is incomplete.');
  validateHistory(state.history);
  if (!Array.isArray(state.epochOrder) || state.epochOrder.length > 2000 || state.epochOrder.some(index => !integer(index, 0, state.epochOrder.length - 1)) || new Set(state.epochOrder).size !== state.epochOrder.length || state.epochOrder.length && state.batchOffset > state.epochOrder.length || !state.epochOrder.length && state.batchOffset !== 0) invalid('Invalid or missing checkpoint sample order. Exact continuation requires the original epoch order.');
  if (!Number.isFinite(state.epochLossSum) || state.epochLossSum < 0 || !integer(state.epochBatches, 0, 2000) || state.datasetRevision !== undefined && !integer(state.datasetRevision, 0, 100_000_000)) invalid('Invalid or missing partial epoch state.');
  if (!state.epochOrder.length && (state.epochLossSum !== 0 || state.epochBatches !== 0) || state.batchOffset === 0 && (state.epochLossSum !== 0 || state.epochBatches !== 0) || state.batchOffset > 0 && state.epochBatches === 0) invalid('The saved partial epoch totals disagree with its sample position.');
  if (!optimizer || optimizer.type !== 'adam' || !Number.isFinite(optimizer.learningRate) || !Array.isArray(optimizer.weights)) invalid('The checkpoint is missing Adam optimizer state.');
  validateConfig(state.config, checkpoint.architecture, optimizer);
  const expected = state.step ? ['iter', ...names.map(name => `${name}/m`), ...names.map(name => `${name}/v`)] : ['iter'];
  if (optimizer.weights.length !== expected.length) invalid('The checkpoint is missing Adam moments.');
  expected.forEach((name, index) => checkedWeight(optimizer.weights[index], name, name === 'iter' ? [] : shapes[name.slice(0, -2)], name === 'iter' ? 'int32' : 'float32'));
  if (optimizer.weights[0].data[0] !== state.step) invalid('Adam iteration count disagrees with the training step.');
  if (!Array.isArray(optimizer.betaPowers) || optimizer.betaPowers.length !== 2 || optimizer.betaPowers.some(value => !Number.isFinite(value) || value < 0 || value > 1)) invalid('The checkpoint is missing exact Adam bias-correction state.');
  if (checkpoint.architecture === ARCHITECTURES.pretrained) {
    if (checkpoint.source?.backboneId !== BACKBONE.id || checkpoint.source?.sha256 !== BACKBONE.sha256) invalid('The pretrained checkpoint references incompatible feature weights. Earlier shallow-feature checkpoints cannot load this neck feature model.');
  } else if (checkpoint.source?.backboneId != null || checkpoint.source?.sha256 != null) invalid('A randomly initialized detector cannot claim a pretrained backbone source.');
  return checkpoint;
}
