import {TrainerEngine} from './engine.js';
import {createWorkerDispatcher} from './worker-router.js';

const baseUrl = new URL(import.meta.env?.BASE_URL || './', self.location.origin).href;
const send = (value, transfer) => self.postMessage(value, transfer || []);
const engine = new TrainerEngine({baseUrl, onProgress: progress => send({type: 'progress', ...progress}), onCheckpoint: value => send({type: 'checkpoint', ...value})});
const dispatch = createWorkerDispatcher(engine, send);
self.addEventListener('message', event => dispatch(event.data));
