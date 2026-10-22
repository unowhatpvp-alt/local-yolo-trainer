import test from 'node:test';
import assert from 'node:assert/strict';
import {createRng, shuffledIndices, letterboxGeometry, inputBoxes, encodeTargets, decodePredictions, validateBoxes, evaluateDetections} from '../src/training/targets.js';

test('non-square letterbox, horizontal flip, target assignment and inverse box mapping agree', () => {
  const geometry = letterboxGeometry(200, 100, 128), original = {classId: 0, x: 31.25, y: 18.75, width: 50, height: 25};
  assert.deepEqual([geometry.resizedWidth, geometry.resizedHeight, geometry.padX, geometry.padY], [128, 64, 0, 32]);
  const [box] = inputBoxes([original], geometry), [flipped] = inputBoxes([original], geometry, true);
  assert.deepEqual([box.x, box.y, box.width, box.height], [20, 44, 32, 16]);
  assert.equal(flipped.x, 76);
  const target = encodeTargets([[box]], 1), offset = ((6 * 16 + 4) * 2) * 6;
  assert.deepEqual(Array.from(target.data.slice(offset, offset + 6)), [.5, .5, .25, .125, 1, 1]);
  const logits = new Float32Array(16 * 16 * 2 * 6).fill(-20), logit = p => Math.log(p / (1 - p));
  logits.set([0, 0, logit(.25), logit(.125), 20, 0], offset);
  const [item] = decodePredictions(logits, ['object'], geometry, .25);
  for (const key of ['x', 'y', 'width', 'height']) assert(Math.abs(item.box[key] - original[key]) < .00001, key);
});

test('two targets in a cell occupy two slots; a third is rejected without silent data loss', () => {
  const boxes = [{classId: 0, x: 10, y: 10, width: 4, height: 4}, {classId: 1, x: 11, y: 11, width: 4, height: 4}];
  assert.equal(encodeTargets([boxes], 2).count, 2);
  assert.throws(() => encodeTargets([[...boxes, {...boxes[0]}]], 2), error => error.code === 'TARGET_COLLISION');
  assert.equal(boxes.length, 2);
  assert.throws(() => validateBoxes([{...boxes[0], x: NaN}], 128, 128, 2), error => error.code === 'INVALID_BOX');
  assert.throws(() => validateBoxes([{...boxes[0], classId: 2}], 128, 128, 2), error => error.code === 'INVALID_BOX');
  assert.equal(encodeTargets([[]], 2).count, 0);
});

test('seeded shuffle resumes exactly and detection metrics use held-out ground truth', () => {
  const rng = createRng(9876); shuffledIndices(8, rng); const state = rng.state;
  const expected = shuffledIndices(8, rng), resumed = createRng(1); resumed.state = state;
  assert.deepEqual(shuffledIndices(8, resumed), expected);
  const box = {x: 10, y: 10, width: 10, height: 10};
  const metric = evaluateDetections([[{classId: 0, score: .8, box}, {classId: 1, score: .7, box}]], [{boxes: [{...box, classId: 0}]}]);
  assert.equal(metric.truePositives, 1); assert.equal(metric.falsePositives, 1); assert.equal(metric.precision50, .5); assert.equal(metric.recall50, 1);
});
