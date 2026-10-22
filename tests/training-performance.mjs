// Observability-only wrappers call the actual engine methods unchanged. This
// does not read or evaluate any final-test photograph and does not select a model.
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const root = fileURLToPath(new URL('../', import.meta.url)), out = path.join(root, 'test-results');
await mkdir(out, {recursive: true});
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const sourceFiles = ['src/training/engine.js', 'src/training/backend.js', 'src/training/model-ops.js', 'src/training/targets.js', 'src/training/worker-router.js', 'src/model/checkpoint-schema.js', 'public/models/yolo26n-features-128.onnx', 'package-lock.json'];
const bindings = async () => Object.fromEntries(await Promise.all(sourceFiles.map(async file => [file, sha(await readFile(path.join(root, file)))])));
const report = {startedAt: new Date().toISOString(), passed: false, scope: 'Actual local original PNG Blob training, 32 train and 8 validation, 20 epochs, WASM. Timing-only method wrappers; no test8 photographs or predictions.', sourceBefore: await bindings(), config: {mode: 'pretrained', backend: 'wasm', epochs: 20, batchSize: 4, learningRate: .003, augment: true, seed: 12345}, pageErrors: []};
let server, browser;
try {
  server = await createServer({root, configFile: path.join(root, 'vite.config.js'), optimizeDeps: {noDiscovery: true, include: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-wasm', '@tensorflow/tfjs-backend-webgl', '@tensorflow/tfjs-backend-webgpu', 'onnxruntime-web/wasm']}, server: {host: '0.0.0.0', port: 5189, strictPort: true, hmr: false}, clearScreen: false});
  await server.listen();
  const base = `http://127.0.0.1:5189${server.config.base}`;
  await writeFile(path.join(out, 'performance-harness.html'), '<!doctype html><meta charset="utf-8"><title>Actual Worker timing</title><p>Real Worker training performance measurement.</p>');
  await writeFile(path.join(out, 'performance-worker.js'), `
import {TrainerEngine} from '../src/training/engine.js';
import {createWorkerDispatcher} from '../src/training/worker-router.js';
const send = value => postMessage(value), timings = {initialization: [], preparation: [], featureExtraction: [], epochBoundaries: []};
class MeasuredEngine extends TrainerEngine {
  async initialize(...args) { const t = performance.now(); try { return await super.initialize(...args); } finally { timings.initialization.push(performance.now() - t); } }
  async prepareSamples(...args) { const t = performance.now(); try { return await super.prepareSamples(...args); } finally { timings.preparation.push(performance.now() - t); } }
  async features(...args) { const t = performance.now(); try { return await super.features(...args); } finally { timings.featureExtraction.push(performance.now() - t); } }
}
const engine = new MeasuredEngine({baseUrl: ${JSON.stringify(base)}, onProgress: value => send({type:'progress', ...value}), onCheckpoint: ({checkpoint}) => { const epoch = checkpoint.training.epoch; if (!timings.epochBoundaries.some(item => item.epoch === epoch)) timings.epochBoundaries.push({epoch, time: performance.now(), step: checkpoint.training.step}); }});
const dispatch = createWorkerDispatcher(engine, value => { if (value.type === 'result' && value.result?.evidence) value.result.observedTimings = timings; send(value); });
addEventListener('message', event => dispatch(event.data));
`);
  browser = await chromium.launch({channel: 'msedge', headless: true, args: ['--enable-gpu', '--enable-precise-memory-info']});
  report.browser = browser.version();
  const page = await browser.newPage(); page.on('pageerror', error => report.pageErrors.push(error.message));
  await page.goto(`${base}test-results/performance-harness.html`);
  const manifest = JSON.parse(await readFile(path.join(root, 'tests/fixtures/penn-fudan/manifest.json')));
  const samples = manifest.samples; assert.equal(samples.length, 40);
  report.dataset = {train: 32, validation: 8, inputSha256: samples.map(sample => ({id: sample.id, sha256: sample.sha256}))};
  report.measurements = await page.evaluate(async ({base, samples, classes, config}) => {
    const worker = new Worker(new URL('test-results/performance-worker.js', base), {type: 'module'}), pending = new Map(); let nextId = 0;
    const call = (type, payload = {}) => new Promise((resolve, reject) => {const id = ++nextId; pending.set(id, {resolve, reject}); worker.postMessage({id, type, payload});});
    worker.onmessage = ({data}) => {if (pending.has(data.id)) {const item = pending.get(data.id); pending.delete(data.id); if (data.type === 'error') item.reject(new Error(data.error.message)); else item.resolve(data.result);}};
    worker.onerror = event => {for (const item of pending.values()) item.reject(new Error(event.message)); pending.clear();};
    const probe = await call('probe', {mode: 'pretrained'}); if (!probe.backends.find(value => value.backend === 'wasm')?.supported) throw new Error('Actual WASM training unavailable.');
    const prepared = await Promise.all(samples.map(async sample => {
      const blob = await (await fetch(new URL(`tests/fixtures/penn-fudan/${sample.path}`, base))).blob();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())), byte => byte.toString(16).padStart(2, '0')).join('');
      if (digest !== sample.sha256) throw new Error('Photo digest mismatch.');
      return {...sample, split: sample.split === 'val' ? 'validation' : sample.split, blob};
    }));
    const heap = () => performance.memory ? {usedJSHeapSize: performance.memory.usedJSHeapSize, totalJSHeapSize: performance.memory.totalJSHeapSize, jsHeapSizeLimit: performance.memory.jsHeapSizeLimit} : null;
    const heaps = [{phase: 'before', ...heap()}], heartbeatIntervals = []; let previous = performance.now();
    const timer = setInterval(() => {const now = performance.now(); heartbeatIntervals.push(now - previous); previous = now; heaps.push({phase: 'during', ...heap()});}, 16);
    const start = performance.now(); let training;
    try { training = await call('train', {samples: prepared, classes, config}); } finally {clearInterval(timer);}
    const rpcElapsedMs = performance.now() - start; heaps.push({phase: 'after', ...heap()});
    await call('dispose'); worker.terminate();
    return {rpcElapsedMs, engineElapsedMs: training.evidence.elapsedMs, evidence: training.evidence, backend: training.backend.backend, timings: training.observedTimings, heartbeatIntervals, heaps, timerTargetMs: 16, memoryMeaning: 'Chromium performance.memory reports the main page JavaScript heap. This excludes Worker heaps, GPU memory, native ONNX/WASM memory and whole-machine RAM; snapshots are not a precise overall memory peak.', initializationMeaning: 'Actual TrainerEngine.initialize: backend validation/selection, variables, Adam and ORT WASM model fetch/verification/session creation. Backend probe already warm; model session first initialized in this run.', preparationMeaning: 'Actual prepareSamples: 40 PNG Blob decodes, Canvas high-quality letterbox, target validation, pixel hashing and cooperative yields. Original file fetching occurred before the measured training.', epochMeaning: 'Between the first checkpoint for successive completed epochs; includes genuine updates, validation and checkpoint work.'};
  }, {base, samples, classes: manifest.classes, config: report.config});
  const m = report.measurements, sum = values => values.reduce((a, b) => a + b, 0), sorted = values => [...values].sort((a, b) => a - b);
  const intervals = sorted(m.heartbeatIntervals), epochs = m.timings.epochBoundaries.slice(1).map((value, i) => value.time - m.timings.epochBoundaries[i].time);
  report.summary = {initializationMs: sum(m.timings.initialization), imagePreparationMs: sum(m.timings.preparation), featureExtractionCalls: m.timings.featureExtraction.length, featureExtractionTotalMs: sum(m.timings.featureExtraction), epochDurationsMs: epochs, meanEpochMs: sum(epochs) / epochs.length, trainingImagePresentationsPerSecondIncludingPreparationValidation: 32 * 20 * 1000 / m.engineElapsedMs, mainThreadHeartbeat: {samples: intervals.length, intervalMedianMs: intervals[Math.floor(intervals.length / 2)], intervalP95Ms: intervals[Math.min(intervals.length - 1, Math.floor(intervals.length * .95))], intervalMaxMs: intervals.at(-1)}, mainThreadHeap: {before: m.heaps[0], after: m.heaps.at(-1), maximumSampledUsedJSHeapSize: Math.max(...m.heaps.map(value => value.usedJSHeapSize || 0))}, unmeasured: ['Worker/native/WASM total memory', 'GPU VRAM', 'whole-system RAM', 'cold browser start and cold backend probe latency', 'physical iPhone/iPad/Safari performance']};
  assert.equal(m.evidence.steps, 160); assert(m.evidence.parametersChanged); assert.equal(epochs.length, 20); assert(intervals.length > 10); assert.equal(report.pageErrors.length, 0);
  report.sourceAfter = await bindings(); assert.deepEqual(report.sourceAfter, report.sourceBefore);
  report.passed = true; report.completedAt = new Date().toISOString();
  console.log('ACTUAL_TRAINING_PERFORMANCE', JSON.stringify(report.summary));
} catch (error) {report.error = {message: error.message, stack: error.stack}; throw error;}
finally {await writeFile(path.join(out, 'browser-training-performance.json'), JSON.stringify(report, null, 2)); await browser?.close(); await server?.close();}
