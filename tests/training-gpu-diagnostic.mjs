// Actual operator isolation for a failing GPU gradient probe. No photos/test-set access.
import {createServer} from 'vite';
import {chromium} from 'playwright';
import {mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
await mkdir(path.join(root, 'test-results'), {recursive: true});
const server = await createServer({root, configFile: path.join(root, 'vite.config.js'), optimizeDeps: {noDiscovery: true, include: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-webgpu', '@tensorflow/tfjs-backend-webgl']}, server: {host: '0.0.0.0', port: 5188, strictPort: true, hmr: false}, clearScreen: false});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({channel: 'msedge', headless: true, args: ['--enable-gpu']});
  const page = await browser.newPage();
  await writeFile(path.join(root, 'test-results/gpu-diagnostic.html'), '<!doctype html><title>GPU operator diagnostic</title>');
  await page.goto(`http://127.0.0.1:5188${server.config.base}test-results/gpu-diagnostic.html`);
  const result = await page.evaluate(async base => {
    const {tf, probeBackend} = await import(new URL('src/training/backend.js', base));
    const {detectorForward, detectorLoss} = await import(new URL('src/training/model-ops.js', base));
    const production = await probeBackend('webgpu', {mode: 'pretrained', baseUrl: base, refresh: true});
    const output = {production, comparisons: []};
    const summarize = array => ({first: Array.from(array).slice(0, 6), finite: array.every(Number.isFinite), minimum: Math.min(...array), maximum: Math.max(...array)});
    const flatLoss = (raw, rawTargets) => {
      const logits = tf.reshape(raw, [-1, 7]), targets = tf.reshape(rawTargets, [-1, 7]);
      const object = tf.slice(targets, [0, 4], [-1, 1]), objectLogits = tf.slice(logits, [0, 4], [-1, 1]);
      const positive = tf.maximum(tf.sum(object), 1), negative = tf.sub(1, object), probability = tf.sigmoid(objectLogits), error = tf.add(tf.mul(object, tf.sub(1, probability)), tf.mul(negative, probability));
      const bce = tf.sub(tf.softplus(objectLogits), tf.mul(object, objectLogits)), alpha = tf.add(tf.mul(object, .25), tf.mul(negative, .75));
      const objectLoss = tf.div(tf.sum(tf.mul(tf.mul(alpha, tf.square(error)), bce)), positive);
      const truth = tf.slice(targets, [0, 0], [-1, 4]), boxLogits = tf.slice(logits, [0, 0], [-1, 4]), safe = tf.clipByValue(truth, 1e-7, 1 - 1e-7);
      const entropy = tf.neg(tf.add(tf.mul(safe, tf.log(safe)), tf.mul(tf.sub(1, safe), tf.log(tf.sub(1, safe)))));
      const boxes = tf.div(tf.sum(tf.mul(tf.maximum(tf.sub(tf.sub(tf.softplus(boxLogits), tf.mul(truth, boxLogits)), entropy), 0), object)), tf.mul(positive, 4));
      const classes = tf.div(tf.neg(tf.sum(tf.mul(tf.mul(tf.slice(targets, [0, 5], [-1, 2]), tf.logSoftmax(tf.slice(logits, [0, 5], [-1, 2]), -1)), object))), positive);
      return tf.addN([objectLoss, tf.mul(boxes, 5), classes]);
    };
    for (const backend of ['cpu', 'webgpu']) for (const variant of ['production-rank5', 'identical-loss-flattened-rank2']) {
      const values = new Map(), optimizer = tf.train.adam(.003);
      let loss, gradients;
      try {
        await tf.setBackend(backend); await tf.ready();
        for (const [name, shape, fill] of [['head/hidden/kernel', [128, 32], .02], ['head/hidden/bias', [32], .01], ['head/output/kernel', [32, 14], .02], ['head/output/bias', [14], .01]]) {
          const tensor = tf.fill(shape, fill), variable = tf.variable(tensor, true, `${backend}-${variant}-${name}`); tensor.dispose(); values.set(name, variable);
        }
        const logits = tf.tidy(() => detectorForward(tf, name => values.get(name), tf.ones([1, 16, 16, 128]), 'yolo26n-frozen-head-v1', 128, 2));
        const forward = summarize(await logits.data()); logits.dispose();
        ({value: loss, grads: gradients} = tf.variableGrads(() => tf.tidy(() => {
          const raw = detectorForward(tf, name => values.get(name), tf.ones([1, 16, 16, 128]), 'yolo26n-frozen-head-v1', 128, 2);
          const data = new Float32Array(512 * 7); data.set([.5, .5, .2, .2, 1, 1], 0); const targets = tf.tensor(data, [1, 16, 16, 2, 7]);
          return variant === 'production-rank5' ? detectorLoss(tf, raw, targets, 2) : flatLoss(raw, targets);
        }), [...values.values()]));
        const gradientStats = {}; for (const [name, tensor] of Object.entries(gradients)) gradientStats[name] = summarize(await tensor.data());
        const head = values.get('head/output/kernel'), before = Float32Array.from(await head.data()); optimizer.applyGradients(gradients); const after = await head.data();
        output.comparisons.push({backend, variant, forward, loss: (await loss.data())[0], gradientStats, maxParameterDelta: Math.max(...after.map((value, i) => Math.abs(value - before[i])))});
      } catch (error) { output.comparisons.push({backend, variant, error: error.message}); }
      finally { loss?.dispose(); if (gradients) Object.values(gradients).forEach(t => t.dispose()); optimizer.dispose(); values.forEach(t => t.dispose()); }
    }
    return output;
  }, `http://127.0.0.1:5188${server.config.base}`);
  await writeFile(path.join(root, 'test-results/gpu-op-diagnostic.json'), JSON.stringify({recordedAt: new Date().toISOString(), browser: browser.version(), ...result}, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally { await browser?.close(); await server.close(); }
