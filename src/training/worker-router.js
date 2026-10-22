import {TrainingError} from './targets.js';

/** One operation owns TF.js state. Pause/stop remain available during training. */
export function createWorkerDispatcher(engine, send) {
  let operationActive = null;
  return async function dispatch({id, type, payload = {}} = {}) {
    let ownsOperation = false;
    try {
      if (!['probe', 'train', 'pause', 'stop', 'predict', 'export', 'dispose'].includes(type)) throw new TrainingError('Unknown training worker command.', 'INVALID_COMMAND');
      if (!['pause', 'stop'].includes(type)) {
        if (operationActive) throw new TrainingError(`Wait for ${operationActive} before starting ${type}.`, 'BUSY');
        operationActive = type; ownsOperation = true;
      }
      const result = await engine[type](payload);
      send({id, type: 'result', result}, type === 'export' && result.bytes ? [result.bytes.buffer] : undefined);
    } catch (error) {
      send({id, type: 'error', error: {name: error.name || 'Error', message: error.message || String(error), code: error.code || 'TRAINING_ERROR'}});
    } finally { if (ownsOperation) operationActive = null; }
  };
}
