import {tf, probeBackends, selectBackend} from './backend.js';
import {TrainingError, validateClasses, validateBoxes, createRng, shuffledIndices, letterboxGeometry, inputBoxes, encodeTargets, decodePredictions, evaluateDetections} from './targets.js';
import {BACKBONE, loadBackbone, verifyBackbone} from '../model/backbone.js';
import {detectorForward, detectorLoss} from './model-ops.js';
import {ARCHITECTURES, weightShapes, validateCheckpoint} from '../model/checkpoint-schema.js';

export {ARCHITECTURES, weightShapes, validateCheckpoint};
const DEFAULTS = Object.freeze({mode: 'pretrained', resolution: 128, batchSize: 1, epochs: 30, learningRate: .003, augment: true, seed: 12345, backend: 'auto', lossVersion: 'focal-kl-v1', preprocessingVersion: 'canvas-high-v1'});
const delay = () => new Promise(resolve => setTimeout(resolve, 0));
let engineSequence = 0;
export async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
}

export function normalizeConfig(input = {}, checkpoint) {
  const config = {...DEFAULTS, ...input};
  if (config.mode === 'checkpoint') {
    if (!checkpoint) throw new TrainingError('Choose a compatible training checkpoint first. An inference ONNX cannot resume training.', 'CHECKPOINT_REQUIRED');
    config.mode = checkpoint.architecture === ARCHITECTURES.scratch ? 'scratch' : 'pretrained';
  }
  if (!ARCHITECTURES[config.mode]) throw new TrainingError('Choose pretrained features or the small detector from scratch.', 'INVALID_CONFIG');
  if (config.resolution !== 128) throw new TrainingError('Only 128 × 128 has been enabled for this engine. Higher resolutions have not been validated.', 'UNSUPPORTED_RESOLUTION');
  if (config.batchSize === 'auto') config.batchSize = 1;
  if (![1, 2, 4, 8].includes(config.batchSize)) throw new TrainingError('Batch size must be 1, 2, 4, 8, or auto.', 'INVALID_CONFIG');
  if (!Number.isInteger(config.epochs) || config.epochs < 1 || config.epochs > 1000 || !Number.isFinite(config.learningRate) || config.learningRate < .000001 || config.learningRate > .1) throw new TrainingError('Use 1–1000 epochs and a learning rate between 0.000001 and 0.1.', 'INVALID_CONFIG');
  if (!['auto', 'cpu', 'wasm', 'webgl', 'webgpu'].includes(config.backend) || !Number.isInteger(config.seed) || config.seed < 1 || config.seed > 4294967295 || typeof config.augment !== 'boolean') throw new TrainingError('Invalid backend, augmentation, or random seed.', 'INVALID_CONFIG');
  if (config.lossVersion !== DEFAULTS.lossVersion) throw new TrainingError('This checkpoint uses an incompatible experimental training loss.', 'INVALID_CONFIG');
  if (config.preprocessingVersion !== DEFAULTS.preprocessingVersion) throw new TrainingError('This checkpoint uses incompatible image scaling. Use a canvas-high-v1 training checkpoint.', 'INVALID_CONFIG');
  return config;
}

const sameShape = (a, b) => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);

async function decodeImage(sample, resolution) {
  let bitmap;
  if (sample.pixels) {
    // Test-only canonical input: resizing raw photographs requires the actual
    // browser Canvas high-quality filter, which Node cannot faithfully mimic.
    const width = sample.width, height = sample.height, geometry = letterboxGeometry(width, height, resolution);
    if (width !== resolution || height !== resolution) throw new TrainingError('Typed pixels must already be the canonical 128 × 128 input. Use an original image Blob for high-quality browser resizing.', 'PIXEL_PREPROCESSING_UNSUPPORTED');
    const channels = sample.pixels.length / (width * height);
    if (![3, 4].includes(channels) || !Array.from(sample.pixels).every(Number.isFinite)) throw new TrainingError('Invalid source pixels.', 'INVALID_IMAGE');
    const normalized = sample.pixels instanceof Float32Array;
    const data = new Float32Array(resolution * resolution * 3).fill(114 / 255);
    for (let y = 0; y < geometry.resizedHeight; y++) for (let x = 0; x < geometry.resizedWidth; x++) {
      const sx = Math.min(width - 1, Math.floor((x + .5) * width / geometry.resizedWidth));
      const sy = Math.min(height - 1, Math.floor((y + .5) * height / geometry.resizedHeight));
      const alpha = channels === 4 ? sample.pixels[(sy * width + sx) * channels + 3] / (normalized ? 1 : 255) : 1;
      for (let c = 0; c < 3; c++) {
        const value = sample.pixels[(sy * width + sx) * channels + c] / (normalized ? 1 : 255);
        if (value < 0 || value > 1 || alpha < 0 || alpha > 1) throw new TrainingError('Pixel values are outside their declared format.', 'INVALID_IMAGE');
        data[((y + geometry.padY) * resolution + x + geometry.padX) * 3 + c] = value * alpha + 114 / 255 * (1 - alpha);
      }
    }
    return {data, geometry};
  }
  if (!(sample.blob instanceof Blob) || typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') throw new TrainingError('This browser cannot decode images in a training worker. Use an updated desktop browser.', 'IMAGE_DECODE_UNAVAILABLE');
  try {
    if (!sample.blob.size || sample.blob.size > 50 * 1024 * 1024) throw new TrainingError('Use an image smaller than 50 MiB.', 'INVALID_IMAGE');
    bitmap = await createImageBitmap(sample.blob, {imageOrientation: 'from-image'});
    const geometry = letterboxGeometry(bitmap.width, bitmap.height, resolution);
    if (sample.width && sample.width !== bitmap.width || sample.height && sample.height !== bitmap.height) throw new TrainingError('The oriented image dimensions changed. Reimport this image before training.', 'IMAGE_DIMENSION_MISMATCH');
    const canvas = new OffscreenCanvas(resolution, resolution), context = canvas.getContext('2d', {willReadFrequently: true});
    if (!context) throw new TrainingError('A 2D image canvas is unavailable.', 'IMAGE_DECODE_UNAVAILABLE');
    context.fillStyle = '#727272'; context.fillRect(0, 0, resolution, resolution);
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    context.drawImage(bitmap, geometry.padX, geometry.padY, geometry.resizedWidth, geometry.resizedHeight);
    const rgba = context.getImageData(0, 0, resolution, resolution).data, data = new Float32Array(resolution * resolution * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4) { data[j++] = rgba[i] / 255; data[j++] = rgba[i + 1] / 255; data[j++] = rgba[i + 2] / 255; }
    return {data, geometry};
  } catch (error) {
    if (error instanceof TrainingError) throw error;
    throw new TrainingError(`This image could not be decoded: ${error.message}`, 'INVALID_IMAGE');
  } finally { bitmap?.close(); }
}

function flipPixels(data, resolution) {
  const result = new Float32Array(data.length);
  for (let y = 0; y < resolution; y++) for (let x = 0; x < resolution; x++) for (let c = 0; c < 3; c++) result[(y * resolution + x) * 3 + c] = data[(y * resolution + resolution - x - 1) * 3 + c];
  return result;
}

export class TrainerEngine {
  constructor({onProgress = () => {}, onCheckpoint = () => {}, baseUrl, backboneBytes} = {}) {
    this.onProgress = onProgress; this.onCheckpoint = onCheckpoint; this.baseUrl = baseUrl; this.backboneBytes = backboneBytes;
    this.prefix = `trainer-${++engineSequence}/`; this.variables = new Map(); this.running = false; this.interruption = null; this.lastCheckpoint = null;
    this.activeOperation = null;
  }
  async exclusive(name, action) {
    if (this.activeOperation) throw new TrainingError(`Wait for ${this.activeOperation} before starting ${name}.`, 'BUSY');
    this.activeOperation = name;
    try { return await action(); } finally { this.activeOperation = null; }
  }
  probe(payload) { return this.exclusive('probe', () => this._probe(payload)); }
  train(payload) { return this.exclusive('train', () => this._train(payload)); }
  load(checkpoint, options) { return this.exclusive('load', () => this._load(checkpoint, options)); }
  predict(payload) { return this.exclusive('predict', () => this._predict(payload)); }
  export(payload) { return this.exclusive('export', () => this._export(payload)); }
  dispose() { return this.exclusive('dispose', () => this._dispose()); }
  async _probe({mode = 'pretrained', refresh = false} = {}) {
    if (this.running) throw new TrainingError('Finish or pause training before checking other backends.', 'BUSY');
    if (mode === 'checkpoint') mode = this.architecture === ARCHITECTURES.scratch ? 'scratch' : 'pretrained';
    return {mode, backends: await probeBackends({mode, baseUrl: this.baseUrl, refresh})};
  }
  pause() { if (this.running) this.interruption = 'paused'; return {requested: this.running, state: 'pause-requested'}; }
  stop() { if (this.running) this.interruption = 'stopped'; return {requested: this.running, state: 'stop-requested'}; }

  async releaseModel() {
    this.optimizer?.dispose(); this.optimizer = null;
    for (const variable of this.variables.values()) variable.dispose(); this.variables.clear();
    if (this.featureSession) await this.featureSession.release(); this.featureSession = null;
    this.prepared = []; this.architecture = null;
  }
  async initialize(config, classes, checkpoint) {
    if (checkpoint) validateCheckpoint(checkpoint);
    const architecture = ARCHITECTURES[config.mode];
    if (checkpoint && (checkpoint.architecture !== architecture || JSON.stringify(checkpoint.classes) !== JSON.stringify(classes))) throw new TrainingError('Checkpoint architecture and class order must match the project. Existing annotations were retained.', 'CHECKPOINT_CLASS_MISMATCH');
    await this.releaseModel();
    this.config = config; this.classes = classes; this.architecture = architecture; this.rng = createRng(config.seed);
    this.backendInfo = await selectBackend(config.backend, {mode: config.mode, baseUrl: this.baseUrl, batchSize: config.batchSize, classCount: classes.length});
    const shapes = weightShapes(architecture, classes.length);
    for (const [name, shape] of Object.entries(shapes)) {
      const size = shape.reduce((a, b) => a * b, 1); let data;
      if (checkpoint) data = Float32Array.from(checkpoint.weights.find(w => w.name === name).data);
      else {
        data = new Float32Array(size);
        if (name.endsWith('/kernel')) {
          const fanIn = shape.length === 4 ? shape[0] * shape[1] * shape[2] : shape[0], scale = name === 'head/output/kernel' ? .03 : Math.sqrt(2 / fanIn);
          for (let i = 0; i < size; i++) data[i] = Math.sqrt(-2 * Math.log(Math.max(1e-12, this.rng.next()))) * Math.cos(2 * Math.PI * this.rng.next()) * scale;
        }
        if (name === 'head/output/bias') for (let slot = 0; slot < 2; slot++) { const offset = slot * (5 + classes.length); data[offset + 2] = -1.4; data[offset + 3] = -1.4; data[offset + 4] = -4; }
      }
      const tensor = tf.tensor(data, shape, 'float32'), variable = tf.variable(tensor, true, this.prefix + name); tensor.dispose(); this.variables.set(name, variable);
    }
    this.optimizer = tf.train.adam(config.learningRate, .9, .999, 1e-7);
    this.state = checkpoint ? structuredClone(checkpoint.training) : {epoch: 0, batchOffset: 0, step: 0, rngState: this.rng.state, datasetHash: '', history: [], epochOrder: [], epochLossSum: 0, epochBatches: 0};
    if (checkpoint) {
      this.rng.state = checkpoint.training.rngState;
      const weights = checkpoint.optimizer.weights.map(w => ({name: w.name === 'iter' ? 'iter' : this.prefix + w.name, tensor: tf.tensor(w.data, w.shape, w.dtype)}));
      try {
        await this.optimizer.setWeights(weights);
        // TF.js 4.22 setWeights recomputes beta powers using pow, whereas an
        // uninterrupted run uses float32 multiplication. Persist the actual
        // accumulators to resume bit-for-bit rather than silently drift.
        for (const [i, name] of ['accBeta1', 'accBeta2'].entries()) {
          const value = tf.scalar(checkpoint.optimizer.betaPowers[i]);
          try { this.optimizer[name].assign(value); } finally { value.dispose(); }
        }
      } finally { weights.forEach(w => w.tensor.dispose()); }
    }
    if (config.mode === 'pretrained') {
      this.backboneBytes = this.backboneBytes ? await verifyBackbone(this.backboneBytes) : await loadBackbone({baseUrl: this.baseUrl});
      const ort = await import('onnxruntime-web/wasm'); this.ort = ort; ort.env.wasm.numThreads = 1;
      if (this.baseUrl) ort.env.wasm.wasmPaths = `${this.baseUrl.replace(/\/$/, '')}/runtime/ort/`;
      this.featureSession = await ort.InferenceSession.create(this.backboneBytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    }
  }
  forward(input) {
    return detectorForward(tf, name => this.variables.get(name), input, this.architecture, this.config.resolution, this.classes.length);
  }
  loss(input, targets) {
    return tf.tidy(() => {
      return detectorLoss(tf, this.forward(input), targets, this.classes.length);
    });
  }
  async features(data) {
    const resolution = this.config.resolution, plane = resolution * resolution, nchw = new Float32Array(data.length);
    for (let p = 0; p < plane; p++) for (let c = 0; c < 3; c++) nchw[c * plane + p] = data[p * 3 + c];
    const outputs = await this.featureSession.run({images: new this.ort.Tensor('float32', nchw, [1, 3, resolution, resolution])});
    const tensor = outputs[BACKBONE.output], grid = resolution / 8, featurePlane = grid * grid;
    if (!tensor || !sameShape(tensor.dims, [1, 128, grid, grid]) || !tensor.data.every(Number.isFinite)) throw new TrainingError('Pretrained features returned invalid values.', 'INVALID_OUTPUT');
    const nhwc = new Float32Array(tensor.data.length);
    for (let p = 0; p < featurePlane; p++) for (let c = 0; c < 128; c++) nhwc[p * 128 + c] = tensor.data[c * featurePlane + p];
    for (const value of Object.values(outputs)) value.dispose?.();
    return nhwc;
  }
  async prepareSamples(samples) {
    if (!Array.isArray(samples) || !samples.length || samples.length > 2000 || new Set(samples.map(s => s.id)).size !== samples.length) throw new TrainingError('Choose a dataset with unique image IDs.', 'INVALID_DATASET');
    const resolution = this.config.resolution, perImage = resolution * resolution * 3 * 4 + (this.featureSession ? 2 * 128 * (resolution / 8) ** 2 * 4 : 0);
    if (samples.length * perImage > 192 * 1024 * 1024) throw new TrainingError('This dataset exceeds the engine’s 192 MiB prepared-image budget. Train a smaller project or export a backup first.', 'DATASET_MEMORY_LIMIT');
    const prepared = [], hashes = [], splitByPixels = new Map();
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i], decoded = await decodeImage(sample, resolution);
      if (sample.split !== undefined && !['train', 'val', 'validation'].includes(sample.split)) throw new TrainingError('Only training and validation samples belong in a training run. Keep final test images separate.', 'INVALID_DATA_SPLIT');
      const boxes = validateBoxes(sample.boxes || [], decoded.geometry.width, decoded.geometry.height, this.classes.length);
      encodeTargets([inputBoxes(boxes, decoded.geometry)], this.classes.length, resolution);
      if (this.config.augment) encodeTargets([inputBoxes(boxes, decoded.geometry, true)], this.classes.length, resolution);
      const item = {id: String(sample.id), ...decoded, boxes, split: ['val', 'validation'].includes(sample.split) ? 'val' : 'train', featureCache: new Map()};
      prepared.push(item);
      const pixelHash = await sha256(decoded.data);
      if (splitByPixels.has(pixelHash) && splitByPixels.get(pixelHash) !== item.split) throw new TrainingError('Identical decoded images appear in both training and validation. Keep duplicates in one split so the evaluation does not reuse training images. All images were retained.', 'DATA_SPLIT_LEAKAGE');
      splitByPixels.set(pixelHash, item.split);
      hashes.push({width: decoded.geometry.width, height: decoded.geometry.height, split: item.split, boxes: boxes.map(({classId, x, y, width, height}) => ({classId, x, y, width, height})), pixelHash});
      this.onProgress({runId: this.runId, state: 'preparing', completedImages: i + 1, totalImages: samples.length, progress: 0, backend: this.backendInfo.backend});
      await delay();
    }
    const datasetHash = await sha256(new TextEncoder().encode(JSON.stringify({classes: this.classes, images: hashes})));
    return {prepared, datasetHash};
  }
  async batch(samples, flips) {
    const resolution = this.config.resolution, grid = resolution / 8, pretrained = Boolean(this.featureSession), channels = pretrained ? 128 : 3;
    const data = new Float32Array(samples.length * (pretrained ? grid : resolution) ** 2 * channels), boxSets = [];
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i], flip = flips[i];
      let values;
      if (pretrained) {
        if (!sample.featureCache.has(flip)) sample.featureCache.set(flip, await this.features(flip ? flipPixels(sample.data, resolution) : sample.data));
        values = sample.featureCache.get(flip);
      } else values = flip ? flipPixels(sample.data, resolution) : sample.data;
      data.set(values, i * values.length); boxSets.push(inputBoxes(sample.boxes, sample.geometry, flip));
    }
    const target = encodeTargets(boxSets, this.classes.length, resolution), side = pretrained ? grid : resolution;
    return {input: tf.tensor4d(data, [samples.length, side, side, channels]), targets: tf.tensor(target.data, target.shape), positiveCount: target.count};
  }
  async checkpoint() {
    if (!this.optimizer) throw new TrainingError('There is no initialized model to save.', 'MODEL_REQUIRED');
    const weights = [];
    for (const [name, tensor] of this.variables) weights.push({name, shape: [...tensor.shape], dtype: tensor.dtype, data: Float32Array.from(await tensor.data())});
    const optimizer = [];
    const named = await this.optimizer.getWeights();
    try {
      for (const weight of named) optimizer.push({name: weight.name.startsWith(this.prefix) ? weight.name.slice(this.prefix.length) : weight.name, shape: [...weight.tensor.shape], dtype: weight.tensor.dtype, data: weight.tensor.dtype === 'int32' ? Int32Array.from(await weight.tensor.data()) : Float32Array.from(await weight.tensor.data())});
    } finally { named.find(weight => weight.name === 'iter')?.tensor.dispose(); } // Adam owns its moment tensors.
    const betaPowers = [(await this.optimizer.accBeta1.data())[0], (await this.optimizer.accBeta2.data())[0]];
    const result = {schemaVersion: 1, architecture: this.architecture, resolution: this.config.resolution, slots: 2, classes: [...this.classes], weights, optimizer: {type: 'adam', learningRate: this.config.learningRate, weights: optimizer, betaPowers}, training: {...structuredClone(this.state), rngState: this.rng.state, config: {...this.config}}, source: this.featureSession ? {backboneId: BACKBONE.id, sha256: BACKBONE.sha256} : {backboneId: null, sha256: null}};
    validateCheckpoint(result); return result;
  }
  async saveCheckpoint() { const checkpoint = await this.checkpoint(); this.lastCheckpoint = checkpoint; await this.onCheckpoint({runId: this.runId, checkpoint}); return checkpoint; }
  async weightHash() {
    const parts = []; let length = 0;
    for (const variable of this.variables.values()) { const values = await variable.data(), data = new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice(); parts.push(data); length += data.length; }
    const bytes = new Uint8Array(length); let offset = 0; for (const part of parts) { bytes.set(part, offset); offset += part.length; }
    return sha256(bytes);
  }
  async evaluate(samples) {
    if (!samples.length) return {available: false, images: 0, reason: 'NO_HELD_OUT_IMAGES'};
    const results = [];
    for (const sample of samples) {
      const batch = await this.batch([sample], [false]);
      let tensor;
      try { tensor = tf.tidy(() => this.forward(batch.input)); results.push(decodePredictions(await tensor.data(), this.classes, sample.geometry, .25)); }
      finally { tensor?.dispose(); batch.input.dispose(); batch.targets.dispose(); }
    }
    return {available: true, ...evaluateDetections(results, samples)};
  }
  async _train({samples, classes: rawClasses, config: inputConfig = {}, checkpoint, additionalEpochs} = {}) {
    if (this.running) throw new TrainingError('A training run is already active.', 'BUSY');
    const classes = validateClasses(rawClasses), config = normalizeConfig(inputConfig, checkpoint);
    if (additionalEpochs !== undefined && (!checkpoint || !Number.isInteger(additionalEpochs) || additionalEpochs < 1 || additionalEpochs > 1000)) throw new TrainingError('Additional training requires a compatible checkpoint and 1–1000 extra epochs.', 'INVALID_CONFIG');
    this.running = true; this.interruption = null; this.runId = crypto.randomUUID();
    let firstLoss = null, lastLoss = null, firstGradient = null, maxGradientNorm = 0, stepsThisRun = 0, initialWeightHash;
    const started = performance.now();
    try {
      await this.initialize(config, classes, checkpoint);
      const {prepared, datasetHash} = await this.prepareSamples(samples); this.prepared = prepared;
      const train = prepared.filter(s => s.split === 'train'), validation = prepared.filter(s => s.split === 'val');
      if (!train.length) throw new TrainingError('Keep at least one image in the training split.', 'NO_TRAIN_IMAGES');
      if (!train.some(s => s.boxes.length)) throw new TrainingError('Mark at least one training object before starting.', 'NO_TRAIN_BOXES');
      const datasetChanged = Boolean(this.state.datasetHash && this.state.datasetHash !== datasetHash);
      if (datasetChanged) { this.state.epoch = 0; this.state.batchOffset = 0; this.state.epochOrder = []; this.state.epochLossSum = 0; this.state.epochBatches = 0; this.state.datasetRevision = (this.state.datasetRevision || 0) + 1; }
      if (additionalEpochs !== undefined) {
        config.epochs = this.state.epoch + additionalEpochs;
        if (config.epochs > 1000) throw new TrainingError('The combined training target exceeds this engine’s 1000-epoch limit. Save the current model before starting a new project.', 'INVALID_CONFIG');
      }
      this.state.datasetHash = datasetHash;
      if (this.state.batchOffset > train.length || this.state.epochOrder?.length && (this.state.epochOrder.length !== train.length || this.state.epochOrder.some(i => i >= train.length))) throw new TrainingError('The saved sample order does not match this dataset.', 'INVALID_CHECKPOINT');
      initialWeightHash = await this.weightHash();
      const initialMetrics = await this.evaluate(validation);
      await this.saveCheckpoint();
      const variables = [...this.variables.values()], stepsPerEpoch = Math.ceil(train.length / config.batchSize), totalSteps = config.epochs * stepsPerEpoch;
      while (this.state.epoch < config.epochs && !this.interruption) {
        if (!this.state.epochOrder?.length) { this.state.epochOrder = shuffledIndices(train.length, this.rng); this.state.batchOffset = 0; this.state.epochLossSum = 0; this.state.epochBatches = 0; }
        while (this.state.batchOffset < train.length && !this.interruption) {
          const selected = this.state.epochOrder.slice(this.state.batchOffset, this.state.batchOffset + config.batchSize).map(i => train[i]);
          const flips = selected.map(() => config.augment && this.rng.next() < .5);
          const batch = await this.batch(selected, flips); let lossTensor, grads;
          try {
            ({value: lossTensor, grads} = tf.variableGrads(() => this.loss(batch.input, batch.targets), variables));
            const loss = (await lossTensor.data())[0];
            if (!Number.isFinite(loss)) throw new TrainingError('Training loss became non-finite. The last valid checkpoint was retained. Lower the learning rate.', 'NONFINITE_LOSS');
            let sumSquares = 0, maxAbsolute = 0, gradientValues = 0;
            for (const gradient of Object.values(grads)) for (const value of await gradient.data()) {
              if (!Number.isFinite(value)) throw new TrainingError('A gradient became non-finite. The last valid checkpoint was retained.', 'NONFINITE_GRADIENT');
              sumSquares += value * value; maxAbsolute = Math.max(maxAbsolute, Math.abs(value)); gradientValues++;
            }
            const norm = Math.sqrt(sumSquares); maxGradientNorm = Math.max(maxGradientNorm, norm);
            const stats = {norm, maxAbsolute, values: gradientValues, finite: true}; firstGradient ||= stats;
            if (norm > 10) { for (const name of Object.keys(grads)) { const old = grads[name]; grads[name] = tf.mul(old, 10 / norm); old.dispose(); } }
            this.optimizer.applyGradients(grads);
            this.state.step++; stepsThisRun++; this.state.batchOffset += selected.length;
            this.state.epochLossSum = (this.state.epochLossSum || 0) + loss; this.state.epochBatches = (this.state.epochBatches || 0) + 1;
            firstLoss ??= loss; lastLoss = loss;
            const completedSteps = this.state.epoch * stepsPerEpoch + Math.ceil(this.state.batchOffset / config.batchSize), elapsedMs = performance.now() - started;
            this.onProgress({runId: this.runId, state: 'training', epoch: this.state.epoch + 1, totalEpochs: config.epochs, step: this.state.step, completedSteps, totalSteps, progress: Math.min(1, completedSteps / totalSteps), loss, gradient: stats, elapsedMs, remainingMs: stepsThisRun < 3 ? null : Math.max(0, totalSteps - completedSteps) * elapsedMs / stepsThisRun, backend: this.backendInfo.backend, backboneBackend: this.featureSession ? 'wasm' : null});
          } finally { lossTensor?.dispose(); if (grads) Object.values(grads).forEach(tensor => tensor.dispose()); batch.input.dispose(); batch.targets.dispose(); }
          if (this.state.step % 10 === 0) await this.saveCheckpoint();
          await delay(); // Worker receives pause/stop messages between genuine updates.
        }
        if (this.state.batchOffset >= train.length) {
          const metrics = await this.evaluate(validation);
          this.state.history.push({epoch: this.state.epoch + 1, loss: this.state.epochLossSum / this.state.epochBatches, metrics, datasetHash, step: this.state.step});
          this.state.epoch++; this.state.batchOffset = 0; this.state.epochOrder = []; this.state.epochLossSum = 0; this.state.epochBatches = 0;
          await this.saveCheckpoint();
        }
      }
      const finalCheckpoint = await this.saveCheckpoint(), finalWeightHash = await this.weightHash(), metrics = await this.evaluate(validation), status = this.interruption || 'completed';
      const evidence = {runId: this.runId, steps: stepsThisRun, firstLoss, lastLoss, firstGradient, maxGradientNorm, initialWeightHash, finalWeightHash, parametersChanged: initialWeightHash !== finalWeightHash, datasetHash, datasetChanged, trainingImages: train.length, validationImages: validation.length, initialMetrics, elapsedMs: performance.now() - started, scope: this.featureSession ? 'frozen YOLO26n features; spatial head parameters trained' : 'all small randomly initialized detector parameters trained', limitations: ['This is not full YOLO26 training.', ...(!validation.length ? ['No held-out validation images; generalization was not measured.'] : [])]};
      this.onProgress({runId: this.runId, state: status, progress: this.state.epoch >= config.epochs ? 1 : Math.min(1, (this.state.epoch * train.length + this.state.batchOffset) / (config.epochs * train.length)), step: this.state.step, epoch: this.state.epoch, totalEpochs: config.epochs, loss: lastLoss, elapsedMs: evidence.elapsedMs, remainingMs: status === 'completed' ? 0 : null, backend: this.backendInfo.backend});
      return {status, checkpoint: finalCheckpoint, metrics, evidence, backend: {...this.backendInfo, backboneBackend: this.featureSession ? 'wasm' : null}};
    } finally { this.running = false; }
  }
  async _load(checkpoint, {backend = 'auto'} = {}) {
    if (this.running) throw new TrainingError('Pause training before loading another model.', 'BUSY');
    validateCheckpoint(checkpoint);
    const config = normalizeConfig({...checkpoint.training.config, mode: checkpoint.architecture === ARCHITECTURES.scratch ? 'scratch' : 'pretrained', backend}, checkpoint);
    await this.initialize(config, [...checkpoint.classes], checkpoint); this.lastCheckpoint = checkpoint;
    return {architecture: this.architecture, backend: this.backendInfo.backend};
  }
  async _predict({blob, pixels, width, height, checkpoint, confidence = .25, backend = 'auto'} = {}) {
    if (this.running) throw new TrainingError('Pause training before testing a model.', 'BUSY');
    if (checkpoint) await this._load(checkpoint, {backend});
    if (!this.optimizer) throw new TrainingError('Train or import a compatible checkpoint before testing.', 'MODEL_REQUIRED');
    const started = performance.now(), sample = await decodeImage({blob, pixels, width, height}, this.config.resolution);
    const data = this.featureSession ? await this.features(sample.data) : sample.data, side = this.featureSession ? this.config.resolution / 8 : this.config.resolution, channels = this.featureSession ? 128 : 3;
    const input = tf.tensor4d(data, [1, side, side, channels]); let output;
    try { output = tf.tidy(() => this.forward(input)); return {items: decodePredictions(await output.data(), this.classes, sample.geometry, confidence), sourceWidth: sample.geometry.width, sourceHeight: sample.geometry.height, latency: performance.now() - started, backend: this.backendInfo.backend, backboneBackend: this.featureSession ? 'wasm' : null, architecture: this.architecture}; }
    finally { input.dispose(); output?.dispose(); }
  }
  async _export({checkpoint} = {}) {
    if (this.running) throw new TrainingError('Pause training before exporting a model.', 'BUSY');
    const model = checkpoint || this.lastCheckpoint || await this.checkpoint(); validateCheckpoint(model);
    const {exportOnnx} = await import('../model/export-onnx.js');
    let backboneBytes;
    if (model.architecture === ARCHITECTURES.pretrained) backboneBytes = this.backboneBytes ? await verifyBackbone(this.backboneBytes) : await loadBackbone({baseUrl: this.baseUrl});
    return {bytes: exportOnnx(model, {backboneBytes}), mime: 'application/octet-stream', filename: 'local-yolo-model.onnx'};
  }
  async _dispose() { if (this.running) throw new TrainingError('Stop training before disposing its worker.', 'BUSY'); await this.releaseModel(); this.lastCheckpoint = null; return {disposed: true}; }
}

export const createEngine = options => new TrainerEngine(options);
