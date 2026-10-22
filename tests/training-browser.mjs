// Real isolated Chromium/Worker training. No browser APIs or training results
// are mocked. Penn-Fudan photographs remain local developer fixtures.
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const project = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(project, 'test-results');
const probeOnly = process.argv.includes('--probe-only');
const newHeadless = process.argv.includes('--new-headless');
const edge = process.argv.includes('--edge');
const accelerationOnly = process.argv.includes('--acceleration-only');
const forcedBackend = process.argv.find(value => value.startsWith('--backend='))?.slice(10);
if (forcedBackend) assert(['cpu', 'wasm', 'webgl', 'webgpu'].includes(forcedBackend));
const suffix = accelerationOnly ? 'browser-acceleration-training' : probeOnly ? edge ? 'browser-backend-edge' : newHeadless ? 'browser-backend-new-headless' : 'browser-backend-probes' : 'browser-real-training';
await mkdir(out, {recursive: true});
const report = {schemaVersion: 2, startedAt: new Date().toISOString(), scope: 'Actual browser Worker, original PNG Blob decoding, local training, checkpoint reload and ONNX execution', finalTestPolicy: 'Engineering replay after correcting Canvas scaling to the existing inference site high-quality setting. Architecture, focal-KL loss, 100 epochs, confidence .25, NMS .45 and matching IoU .5 remain frozen. The original first evaluation is retained in pre-canvas-high-v1; the second evaluation is not an untouched independent experiment. No final-test-based model selection or tuning.', preprocessingVersion: 'canvas-high-v1', phase: 'starting', passed: false, pageErrors: [], consoleErrors: [], nonGetRequests: [], externalRequests: [], servedResourceBindings: []};
const reportPath = path.join(out, `${suffix}.json`);
try { const prior = JSON.parse(await readFile(reportPath)); if (!prior.passed && prior.failedAt) await writeFile(path.join(out, `${suffix}-failed-${prior.failedAt.replace(/[:.]/g, '-')}.json`), JSON.stringify(prior, null, 2)); } catch (error) { if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error; }
const persist = () => writeFile(reportPath, JSON.stringify(report, null, 2));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const sourcePaths = ['src/training/engine.js', 'src/training/worker.js', 'src/training/worker-router.js', 'src/training/backend.js', 'src/training/model-ops.js', 'src/training/targets.js', 'src/model/backbone.js', 'src/model/checkpoint-schema.js', 'src/model/export-onnx.js', 'src/data/formats.js', 'package-lock.json', 'tests/fixtures/penn-fudan/manifest.json', 'public/models/yolo26n-features-128.onnx'];
const sourceBinding = async () => Object.fromEntries(await Promise.all(sourcePaths.map(async file => [file, digest(await readFile(path.join(project, file)))])));
report.sourceBindingBefore = await sourceBinding();
const responseBindings = [];
let server, browser;
try {
  server = await createServer({root: project, configFile: path.join(project, 'vite.config.js'), optimizeDeps: {noDiscovery: true, include: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-wasm', '@tensorflow/tfjs-backend-webgl', '@tensorflow/tfjs-backend-webgpu', 'onnxruntime-web/wasm', 'fflate']}, server: {host: '0.0.0.0', port: 5187, strictPort: true, hmr: false}, clearScreen: false});
  await server.listen();
  const base = server.config.base, rootUrl = `http://127.0.0.1:5187${base}`;
  const harness = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local YOLO Trainer — actual browser training verification</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#fafafa;color:#111}pre{white-space:pre-wrap}canvas{max-width:100%;height:auto;border:1px solid #aaa}a{display:inline-block;margin:10px}</style><h1>Actual browser training verification</h1><pre id="status">Starting real Worker…</pre><div id="preview"></div>';
  await writeFile(path.join(out, 'training-browser-harness.html'), harness);
  browser = await chromium.launch({headless: true, ...(edge ? {channel: 'msedge', args: ['--enable-gpu']} : newHeadless ? {channel: 'chromium', args: ['--enable-gpu']} : {})});
  report.launchMode = edge ? 'installed Microsoft Edge new headless; isolated temporary profile; --enable-gpu' : newHeadless ? 'chromium new headless; --enable-gpu' : 'Playwright default headless shell';
  report.browserVersion = browser.version();
  const page = await browser.newPage({viewport: {width: 1280, height: 1000}, acceptDownloads: true});
  page.on('pageerror', error => report.pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') report.consoleErrors.push(message.text());
    if (message.text().startsWith('TRAINER_QA_PROGRESS')) console.log(message.text());
  });
  page.on('request', request => {
    if (request.method() !== 'GET') report.nonGetRequests.push({method: request.method(), url: request.url()});
    if (/^https?:/.test(request.url()) && !request.url().startsWith('http://127.0.0.1:5187/')) report.externalRequests.push({method: request.method(), url: request.url()});
  });
  page.on('response', response => {
    const url = response.url();
    if (url.startsWith(rootUrl) && /\/(src|runtime|models)\//.test(url)) responseBindings.push((async () => {
      try { const bytes = await response.body(); report.servedResourceBindings.push({url: url.slice(rootUrl.length), status: response.status(), bytes: bytes.length, sha256: digest(bytes)}); }
      catch (error) { report.servedResourceBindings.push({url: url.slice(rootUrl.length), bindingError: error.message}); }
    })());
  });
  await page.goto(`${rootUrl}test-results/training-browser-harness.html`);
  await page.evaluate(async root => {
    const worker = new Worker(new URL('src/training/worker.js', root), {type: 'module'}), pending = new Map();
    const qa = window.__trainerQa = {root, worker, nextId: 0, pending, checkpoints: 0, latestProgress: null};
    qa.call = (type, payload = {}) => new Promise((resolve, reject) => {
      const id = ++qa.nextId, timer = setTimeout(() => {pending.delete(id); reject(new Error(`Worker ${type} timed out.`));}, type === 'train' ? 600000 : 120000);
      pending.set(id, {resolve, reject, timer}); worker.postMessage({id, type, payload});
    });
    worker.addEventListener('message', ({data}) => {
      if (data.type === 'progress') {
        qa.latestProgress = data; document.querySelector('#status').textContent = JSON.stringify(data, null, 2);
        if (data.state === 'training' && data.step % 100 === 0) console.log('TRAINER_QA_PROGRESS', JSON.stringify({step: data.step, epoch: data.epoch, loss: data.loss, elapsedMs: data.elapsedMs, backend: data.backend}));
      } else if (data.type === 'checkpoint') { qa.checkpoints++; qa.latestCheckpoint = data.checkpoint; }
      else if (pending.has(data.id)) {
        const item = pending.get(data.id); clearTimeout(item.timer); pending.delete(data.id);
        if (data.type === 'error') { const error = new Error(data.error.message); error.code = data.error.code; item.reject(error); } else item.resolve(data.result);
      }
    });
    worker.addEventListener('error', event => {for (const item of pending.values()) { clearTimeout(item.timer); item.reject(new Error(event.message)); } pending.clear();});
    qa.gpu = {available: Boolean(navigator.gpu)};
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      const info = adapter?.info || await adapter?.requestAdapterInfo?.();
      if (info) qa.gpu.adapter = {vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null};
    } catch (error) { qa.gpu.error = error.message; }
    try {
      const gl = document.createElement('canvas').getContext('webgl2');
      const extension = gl?.getExtension('WEBGL_debug_renderer_info');
      if (extension) qa.gpu.webglRenderer = gl.getParameter(extension.UNMASKED_RENDERER_WEBGL);
    } catch (error) { qa.gpu.webglError = error.message; }
  }, rootUrl);
  report.gpu = await page.evaluate(() => window.__trainerQa.gpu);
  const adapterText = JSON.stringify(report.gpu).toLowerCase();
  report.hardwareGpuConfirmed = Boolean(report.gpu.adapter && report.gpu.adapter.isFallbackAdapter !== true && /nvidia|blackwell|amd|intel|apple|qualcomm/.test(adapterText) && !/swiftshader|llvmpipe|software|microsoft basic|warp/.test(adapterText));
  report.phase = 'real-backend-gradient-probes'; await persist();
  report.workerMutualExclusion = await page.evaluate(async () => {
    const qa = window.__trainerQa, pending = qa.call('probe', {mode: 'pretrained'});
    let rejected = null;
    try { await qa.call('predict', {}); } catch (error) { rejected = error.code; }
    const result = await pending;
    if (rejected !== 'BUSY') throw new Error('A concurrent actual Worker operation was not rejected as BUSY.');
    return {concurrentPredictRejected: rejected, ownerProbeCompleted: result.backends.some(backend => backend.backend === 'cpu' && backend.supported)};
  });
  report.backendProbes = {};
  for (const mode of ['pretrained', 'scratch']) {
    report.backendProbes[mode] = await page.evaluate(mode => window.__trainerQa.call('probe', {mode, refresh: true}), mode);
    assert(report.backendProbes[mode].backends.find(backend => backend.backend === 'cpu')?.supported, `Actual ${mode} CPU training probe failed.`);
    console.log('ACTUAL_BACKEND_PROBES', JSON.stringify(report.backendProbes[mode])); await persist();
  }
  if (accelerationOnly) {
    assert(report.hardwareGpuConfirmed, 'Hardware GPU must be confirmed before claiming acceleration training.');
    report.accelerationRuns = [];
    for (const mode of ['pretrained', 'scratch']) for (const backend of ['webgl', 'webgpu']) {
      if (!report.backendProbes[mode].backends.find(value => value.backend === backend)?.supported) continue;
      const result = await page.evaluate(async ({mode, backend}) => {
        const qa = window.__trainerQa, samples = [];
        for (let i = 0; i < 4; i++) {
          const canvas = new OffscreenCanvas(128, 128), context = canvas.getContext('2d');
          context.fillStyle = `rgb(${20 + i},20,20)`; context.fillRect(0, 0, 128, 128);
          context.fillStyle = '#ff3030'; context.fillRect(20 + i * 12, 30 + i * 8, 18, 22);
          samples.push({id: `gpu-${i}`, split: 'train', blob: await canvas.convertToBlob({type: 'image/png'}), boxes: [{classId: 0, x: 20 + i * 12, y: 30 + i * 8, width: 18, height: 22}]});
        }
        const trained = await qa.call('train', {classes: ['red rectangle'], samples, config: {mode, backend, batchSize: 2, epochs: 3, seed: 12345, learningRate: .003, augment: false}});
        const before = await qa.call('predict', {...samples[0], confidence: .001});
        const after = await qa.call('predict', {...samples[0], checkpoint: trained.checkpoint, backend, confidence: .001});
        if (JSON.stringify(before.items) !== JSON.stringify(after.items)) throw new Error('GPU checkpoint reload changed predictions.');
        return {mode, backend, status: trained.status, evidence: trained.evidence, actualConfig: trained.checkpoint.training.config, backendProbe: trained.backend, reloadedPredictionCount: after.items.length, exactReload: true, scope: 'Four generated rectangle PNG Blobs; real hardware accelerated gradient/update and reload regression, not photographic accuracy evidence.'};
      }, {mode, backend});
      assert.equal(result.status, 'completed'); assert.equal(result.evidence.steps, 6); assert(result.evidence.parametersChanged); assert(result.evidence.firstGradient.finite && result.evidence.firstGradient.norm > 0);
      report.accelerationRuns.push(result); await persist();
    }
    assert(report.accelerationRuns.some(run => run.mode === 'pretrained' && run.backend === 'webgpu'), 'No complete hardware WebGPU training run passed.');
    report.passed = true; report.phase = 'completed-acceleration-training';
  }
  else if (probeOnly) { report.passed = true; report.phase = 'completed-probes-only'; }
  else {
    const manifest = JSON.parse(await readFile(path.join(project, 'tests/fixtures/penn-fudan/manifest.json')));
    assert.equal(manifest.samples.length, 40); assert.equal(manifest.samples.filter(s => s.split === 'train').length, 32); assert.equal(manifest.samples.filter(s => s.split === 'val').length, 8); assert.equal(manifest.samplesTest.length, 8);
    report.dataset = {source: manifest.source, splitPolicy: manifest.splitPolicy, counts: manifest.counts, manifestSha256: digest(await readFile(path.join(project, 'tests/fixtures/penn-fudan/manifest.json'))), finalTestIds: manifest.samplesTest.map(sample => sample.id)};
    const supported = report.backendProbes.pretrained.backends;
    report.selectedBackend = forcedBackend || (report.hardwareGpuConfirmed && supported.find(b => b.backend === 'webgpu')?.supported ? 'webgpu' : supported.find(b => b.backend === 'wasm')?.supported ? 'wasm' : 'cpu');
    assert(supported.find(backend => backend.backend === report.selectedBackend)?.supported);
    report.backendSelectionReason = forcedBackend ? 'Explicitly retain the original WASM reference backend for the preprocessing compatibility repair; acceleration is independently tested.' : 'Actual full-network training probe plus hardware identity.';
    report.config = {mode: 'pretrained', backend: report.selectedBackend, batchSize: 4, epochs: 100, learningRate: .003, augment: true, seed: 12345, preprocessingVersion: 'canvas-high-v1'};
    report.phase = 'loading-original-local-photographs'; await persist();
    await page.evaluate(async manifest => {
      const qa = window.__trainerQa; qa.manifest = manifest;
      qa.samples = await Promise.all(manifest.samples.map(async sample => {
        const response = await fetch(new URL(`tests/fixtures/penn-fudan/${sample.path}`, qa.root)); if (!response.ok) throw new Error(`Fixture failed: ${sample.id}`);
        const blob = await response.blob(), actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())), byte => byte.toString(16).padStart(2, '0')).join('');
        if (actual !== sample.sha256) throw new Error(`Real photograph SHA mismatch: ${sample.id}`);
        return {...sample, split: sample.split === 'val' ? 'validation' : sample.split, blob};
      }));
    }, manifest);
    report.phase = 'actual-worker-training'; await persist();
    const training = await page.evaluate(async config => {
      const qa = window.__trainerQa;
      qa.training = await qa.call('train', {samples: qa.samples, classes: qa.manifest.classes, config});
      return {status: qa.training.status, evidence: qa.training.evidence, metrics: qa.training.metrics, backend: qa.training.backend, history: qa.training.checkpoint.training.history, actualConfig: qa.training.checkpoint.training.config, checkpoints: qa.checkpoints};
    }, report.config);
    report.training = training;
    assert.equal(training.status, 'completed'); assert.equal(training.evidence.steps, 800); assert(training.evidence.parametersChanged); assert(training.evidence.firstGradient.finite && training.evidence.firstGradient.norm > 0); assert(training.evidence.lastLoss < training.evidence.firstLoss);
    await persist();
    // Exercise the same public checkpoint archive implementation used by UI.
    report.phase = 'checkpoint-save-reload'; await persist();
    report.reload = await page.evaluate(async () => {
      const qa = window.__trainerQa, formats = await import(new URL('src/data/formats.js', qa.root));
      qa.checkpointBlob = await formats.packCheckpoint(qa.training.checkpoint);
      qa.restoredCheckpoint = await formats.unpackCheckpoint(qa.checkpointBlob);
      const sample = qa.samples.find(sample => sample.split === 'validation');
      const before = await qa.call('predict', {...sample, confidence: .25});
      const after = await qa.call('predict', {...sample, checkpoint: qa.restoredCheckpoint, backend: qa.training.backend.backend, confidence: .25});
      qa.validationPreview = {sample, before, after};
      let maxDifference = 0;
      if (before.items.length !== after.items.length) throw new Error('Checkpoint reload changed the result count.');
      for (let i = 0; i < before.items.length; i++) {
        if (before.items[i].classId !== after.items[i].classId) throw new Error('Checkpoint reload changed a class.');
        for (const value of ['score']) maxDifference = Math.max(maxDifference, Math.abs(before.items[i][value] - after.items[i][value]));
        for (const key of ['x', 'y', 'width', 'height']) maxDifference = Math.max(maxDifference, Math.abs(before.items[i].box[key] - after.items[i].box[key]));
      }
      return {archiveBytes: qa.checkpointBlob.size, predictionCount: after.items.length, maxDifference, step: qa.restoredCheckpoint.training.step, exactAdamStateRetained: JSON.stringify(qa.restoredCheckpoint.optimizer.betaPowers) === JSON.stringify(qa.training.checkpoint.optimizer.betaPowers)};
    });
    assert(report.reload.maxDifference < .0002); assert(report.reload.exactAdamStateRetained); assert.equal(report.reload.step, 800);
    const download = async (kind, filename) => {
      const pending = page.waitForEvent('download');
      await page.evaluate(async ({kind, filename}) => {
        const qa = window.__trainerQa;
        const blob = kind === 'checkpoint' ? qa.checkpointBlob : new Blob([qa.exported.bytes], {type: 'application/octet-stream'});
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename; a.textContent = filename; document.body.append(a); a.click();
      }, {kind, filename});
      const received = await pending; await received.saveAs(path.join(out, filename));
    };
    await download('checkpoint', 'browser-trained-model.lymodel');
    await page.evaluate(async () => {const qa = window.__trainerQa; qa.exported = await qa.call('export', {checkpoint: qa.restoredCheckpoint});});
    await download('onnx', 'browser-trained-model.onnx');
    report.onnx = {bytes: (await readFile(path.join(out, 'browser-trained-model.onnx'))).length, sha256: digest(await readFile(path.join(out, 'browser-trained-model.onnx')))};
    report.phase = 'onnx-all-output-numeric-parity'; await persist();
    report.onnx.parity = await page.evaluate(async () => {
      const qa = window.__trainerQa, {tf} = await import(new URL('src/training/backend.js', qa.root)), {detectorForward} = await import(new URL('src/training/model-ops.js', qa.root));
      const {BACKBONE} = await import(new URL('src/model/backbone.js', qa.root)), {letterboxGeometry} = await import(new URL('src/training/targets.js', qa.root));
      const ort = await import(new URL('node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs', qa.root));
      ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = new URL('runtime/ort/', qa.root).href;
      await tf.setBackend('cpu'); await tf.ready();
      const sample = qa.validationPreview.sample, bitmap = await createImageBitmap(sample.blob, {imageOrientation: 'from-image'}), geometry = letterboxGeometry(bitmap.width, bitmap.height, 128);
      const canvas = new OffscreenCanvas(128, 128), context = canvas.getContext('2d'); context.fillStyle = '#727272'; context.fillRect(0, 0, 128, 128); context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high'; context.drawImage(bitmap, geometry.padX, geometry.padY, geometry.resizedWidth, geometry.resizedHeight); bitmap.close();
      const rgba = context.getImageData(0, 0, 128, 128).data, nchw = new Float32Array(3 * 128 * 128);
      for (let p = 0; p < 128 * 128; p++) for (let c = 0; c < 3; c++) nchw[c * 128 * 128 + p] = rgba[p * 4 + c] / 255;
      const backboneBytes = await (await fetch(new URL('models/yolo26n-features-128.onnx', qa.root))).arrayBuffer();
      const featureSession = await ort.InferenceSession.create(backboneBytes, {executionProviders: ['wasm']}), exportedSession = await ort.InferenceSession.create(qa.exported.bytes, {executionProviders: ['wasm']});
      const inputs = {images: new ort.Tensor('float32', nchw, [1, 3, 128, 128])};
      const feature = (await featureSession.run(inputs))[BACKBONE.output], nhwc = new Float32Array(feature.data.length);
      for (let p = 0; p < 256; p++) for (let c = 0; c < 128; c++) nhwc[p * 128 + c] = feature.data[c * 256 + p];
      const checkpoint = qa.restoredCheckpoint, weights = new Map(checkpoint.weights.map(weight => [weight.name, tf.tensor(weight.data, weight.shape)]));
      const input = tf.tensor4d(nhwc, [1, 16, 16, 128]);
      const tensor = tf.tidy(() => detectorForward(tf, name => weights.get(name), input, checkpoint.architecture, 128, checkpoint.classes.length)), logits = await tensor.data();
      const actual = (await exportedSession.run(inputs))[exportedSession.outputNames[0]], channels = 5 + checkpoint.classes.length, anchors = 512;
      const sigmoid = value => 1 / (1 + Math.exp(-value)), expected = new Float32Array((4 + checkpoint.classes.length) * anchors);
      for (let anchor = 0; anchor < anchors; anchor++) {
        const cell = Math.floor(anchor / 2), offset = anchor * channels;
        expected[anchor] = (sigmoid(logits[offset]) + cell % 16) * 8; expected[anchors + anchor] = (sigmoid(logits[offset + 1]) + Math.floor(cell / 16)) * 8;
        expected[2 * anchors + anchor] = sigmoid(logits[offset + 2]) * 128; expected[3 * anchors + anchor] = sigmoid(logits[offset + 3]) * 128;
        let max = -Infinity; for (let c = 0; c < checkpoint.classes.length; c++) max = Math.max(max, logits[offset + 5 + c]);
        let denominator = 0; for (let c = 0; c < checkpoint.classes.length; c++) denominator += Math.exp(logits[offset + 5 + c] - max);
        for (let c = 0; c < checkpoint.classes.length; c++) expected[(4 + c) * anchors + anchor] = sigmoid(logits[offset + 4]) * Math.exp(logits[offset + 5 + c] - max) / denominator;
      }
      let maxAbsoluteError = 0;
      if (actual.data.length !== expected.length || !actual.data.every(Number.isFinite)) throw new Error('ONNX output shape or values are invalid.');
      for (let i = 0; i < expected.length; i++) maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(expected[i] - actual.data[i]));
      tensor.dispose(); input.dispose(); for (const weight of weights.values()) weight.dispose(); await featureSession.release(); await exportedSession.release();
      return {comparedValues: expected.length, outputShape: actual.dims, maxAbsoluteError, tolerance: .0002, reference: 'actual browser TF.js CPU spatial-head output using real frozen features and exported trained weights'};
    });
    assert(report.onnx.parity.maxAbsoluteError < report.onnx.parity.tolerance);
    // Only after checkpoint/export gates, perform the one frozen final test.
    report.phase = 'frozen-independent-final-test'; report.finalTestStartedAt = new Date().toISOString(); await persist();
    report.finalTest = await page.evaluate(async () => {
      const qa = window.__trainerQa, {evaluateDetections} = await import(new URL('src/training/targets.js', qa.root));
      const samples = [], predictions = [];
      for (const sample of qa.manifest.samplesTest) {
        const response = await fetch(new URL(`tests/fixtures/penn-fudan/${sample.path}`, qa.root)); if (!response.ok) throw new Error('Independent test photo could not load.');
        const blob = await response.blob(), digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())), byte => byte.toString(16).padStart(2, '0')).join('');
        if (digest !== sample.sha256) throw new Error('Independent test photograph SHA mismatch.');
        const result = await qa.call('predict', {blob, confidence: .25});
        samples.push(sample); predictions.push({id: sample.id, sha256: digest, truth: sample.boxes, ...result});
        if (!qa.finalPreview) qa.finalPreview = {sample: {...sample, blob}, result};
      }
      return {confidence: .25, matchingIou: .5, modelSelectionUsedThesePhotographs: false, metrics: evaluateDetections(predictions.map(result => result.items), samples), predictions};
    });
    report.finalTestCompletedAt = new Date().toISOString(); await persist();
    await page.evaluate(async () => {
      const qa = window.__trainerQa, {sample, result} = qa.finalPreview, image = await createImageBitmap(sample.blob), canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext('2d'); context.drawImage(image, 0, 0); image.close();
      context.lineWidth = Math.max(2, canvas.width / 240); context.font = `${Math.max(14, canvas.width / 35)}px system-ui`;
      for (const item of result.items) { const box = item.box; context.strokeStyle = '#fff'; context.strokeRect(box.x, box.y, box.width, box.height); context.fillStyle = '#000'; const text = `${item.label} ${Math.round(item.score * 100)}%`, width = context.measureText(text).width; context.fillRect(box.x, Math.max(0, box.y - 24), width + 8, 26); context.fillStyle = '#fff'; context.fillText(text, box.x + 4, Math.max(20, box.y - 5)); }
      document.querySelector('#preview').append(canvas); document.querySelector('#status').textContent = JSON.stringify({backend: qa.training.backend, training: qa.training.evidence, validation: qa.training.metrics}, null, 2);
    });
    await page.screenshot({path: path.join(out, 'browser-trained-real-photo.png'), fullPage: true});
    await page.evaluate(async () => {await window.__trainerQa.call('dispose'); window.__trainerQa.worker.terminate();});
    assert.equal(report.pageErrors.length, 0); assert.equal(report.nonGetRequests.length, 0); assert.equal(report.externalRequests.length, 0);
    report.passed = true; report.phase = 'completed';
  }
  await Promise.allSettled(responseBindings);
  report.sourceBindingAfter = await sourceBinding();
  report.sourceFilesChangedDuringRun = sourcePaths.filter(file => report.sourceBindingBefore[file] !== report.sourceBindingAfter[file]);
  assert.equal(report.sourceFilesChangedDuringRun.length, 0, 'Source changed during the bound training run.');
  report.completedAt = new Date().toISOString(); await persist();
  console.log('BROWSER_TRAINING_RESULT', JSON.stringify({passed: report.passed, phase: report.phase, hardwareGpuConfirmed: report.hardwareGpuConfirmed, selectedBackend: report.selectedBackend, training: report.training?.evidence, validation: report.training?.metrics, finalTest: report.finalTest?.metrics, reload: report.reload, onnx: report.onnx, reportPath}));
} catch (error) {
  report.error = {name: error.name, message: error.message, stack: error.stack}; report.failedAt = new Date().toISOString(); await persist(); throw error;
} finally { await browser?.close(); await server?.close(); }
