import test from 'node:test';
import assert from 'node:assert/strict';
import {tf} from '../src/training/backend.js';
import {detectorLoss} from '../src/training/model-ops.js';

// Preserve the original CPU focal/KL objective as the numerical reference.
// The production shape repair must not silently change its loss or gradients.
function referenceRankFive(logits, targets, classes) {
  const object = tf.slice(targets, [0, 0, 0, 0, 4], [-1, -1, -1, -1, 1]);
  const score = tf.slice(logits, [0, 0, 0, 0, 4], [-1, -1, -1, -1, 1]);
  const positives = tf.maximum(tf.sum(object), 1), negative = tf.sub(1, object);
  const bce = tf.sub(tf.softplus(score), tf.mul(object, score)), probability = tf.sigmoid(score);
  const error = tf.add(tf.mul(object, tf.sub(1, probability)), tf.mul(negative, probability));
  const focal = tf.div(tf.sum(tf.mul(tf.mul(tf.add(tf.mul(object, .25), tf.mul(negative, .75)), tf.square(error)), bce)), positives);
  const truth = tf.slice(targets, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 4]);
  const box = tf.slice(logits, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 4]);
  const safe = tf.clipByValue(truth, 1e-7, 1 - 1e-7);
  const entropy = tf.neg(tf.add(tf.mul(safe, tf.log(safe)), tf.mul(tf.sub(1, safe), tf.log(tf.sub(1, safe)))));
  const kl = tf.maximum(tf.sub(tf.sub(tf.softplus(box), tf.mul(truth, box)), entropy), 0);
  const boxLoss = tf.div(tf.sum(tf.mul(kl, object)), tf.mul(positives, 4));
  const labels = tf.slice(targets, [0, 0, 0, 0, 5], [-1, -1, -1, -1, classes]);
  const classLogits = tf.slice(logits, [0, 0, 0, 0, 5], [-1, -1, -1, -1, classes]);
  const classification = tf.div(tf.neg(tf.sum(tf.mul(tf.mul(labels, tf.logSoftmax(classLogits, -1)), object))), positives);
  return tf.addN([focal, tf.mul(boxLoss, 5), classification]);
}

test('rank-two GPU compatibility repair preserves the full original CPU loss and every gradient', async () => {
  await tf.setBackend('cpu'); await tf.ready();
  const shape = [2, 16, 16, 2, 8], length = shape.reduce((a, b) => a * b, 1);
  const values = Float32Array.from({length}, (_, i) => Math.sin(i * .371) * 4);
  const targets = new Float32Array(length);
  for (const cell of [0, 23, 94, 560, 873]) targets.set([.3, .8, .12, .25, 1, 0, 1, 0], cell * 8);
  const input = tf.tensor(values, shape), truth = tf.tensor(targets, shape);
  let expected, actual;
  try {
    expected = tf.valueAndGrad(x => tf.tidy(() => referenceRankFive(x, truth, 3)))(input);
    actual = tf.valueAndGrad(x => tf.tidy(() => detectorLoss(tf, x, truth, 3)))(input);
    assert(Math.abs(expected.value.dataSync()[0] - actual.value.dataSync()[0]) < 1e-6);
    const a = expected.grad.dataSync(), b = actual.grad.dataSync();
    assert.equal(a.length, length); assert.equal(b.length, length);
    let maximumError = 0; for (let i = 0; i < length; i++) maximumError = Math.max(maximumError, Math.abs(a[i] - b[i]));
    assert(maximumError < 1e-7, `Full CPU gradient changed: ${maximumError}`);
    console.log('GPU_SHAPE_REPAIR_CPU_EQUIVALENCE', JSON.stringify({values: length, loss: actual.value.dataSync()[0], maximumGradientDifference: maximumError}));
  } finally { input.dispose(); truth.dispose(); expected?.value.dispose(); expected?.grad.dispose(); actual?.value.dispose(); actual?.grad.dispose(); }
});
