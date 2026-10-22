import test from 'node:test';
import assert from 'node:assert/strict';
import {TrainerEngine} from '../src/training/engine.js';
import {createWorkerDispatcher} from '../src/training/worker-router.js';

test('real probe/train calls serialize; busy errors do not release the owner and pause still works', {timeout: 30000}, async () => {
  const messages = [], engine = new TrainerEngine(), dispatch = createWorkerDispatcher(engine, value => messages.push(value));
  try {
    const probing = dispatch({id: 1, type: 'probe', payload: {mode: 'scratch'}});
    await dispatch({id: 2, type: 'train', payload: {}});
    await dispatch({id: 3, type: 'dispose'});
    assert.equal(messages.find(message => message.id === 2).error.code, 'BUSY');
    assert.equal(messages.find(message => message.id === 3).error.code, 'BUSY');
    await assert.rejects(engine.predict({}), error => error.code === 'BUSY');
    await probing;
    assert.equal(messages.find(message => message.id === 1).type, 'result');
    engine.onProgress = event => { if (event.state === 'training') void dispatch({id: 5, type: 'pause'}); };
    const pixels = new Float32Array(128 * 128 * 3).fill(.2);
    const training = dispatch({id: 4, type: 'train', payload: {samples: [{id: 'one', width: 128, height: 128, pixels, boxes: [{classId: 0, x: 20, y: 20, width: 12, height: 12}]}], classes: ['object'], config: {mode: 'scratch', backend: 'cpu', epochs: 2, augment: false}}});
    await dispatch({id: 6, type: 'export'});
    assert.equal(messages.find(message => message.id === 6).error.code, 'BUSY');
    await training;
    assert.equal(messages.find(message => message.id === 5).type, 'result');
    assert.equal(messages.find(message => message.id === 4).result.status, 'paused');
    assert.equal(messages.find(message => message.id === 4).result.checkpoint.training.step, 1);
    await dispatch({id: 7, type: 'dispose'}); assert.equal(messages.find(message => message.id === 7).result.disposed, true);
  } finally { if (!engine.activeOperation) await engine.dispose(); }
});
