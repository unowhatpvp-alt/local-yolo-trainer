// Separate acceptance script: node --test tests/storage-browser.mjs
// Uses a fresh browser profile/origin, real IndexedDB transactions and real
// optimized checkpoint bytes. Injected failure paths are identified explicitly.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {chromium} from 'playwright';
import {TrainerEngine} from '../src/training/engine.js';
import {packCheckpoint} from '../src/data/formats.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const observedSources = ['package.json', 'package-lock.json', 'src/data/project.js', 'src/data/formats.js', 'src/data/storage.js', 'src/model/checkpoint-schema.js', 'src/model/backbone.js', 'src/training/engine.js', 'src/training/backend.js', 'src/training/model-ops.js', 'src/training/targets.js', 'public/models/yolo26n-features-128.onnx'];
async function sourceHashes() {
  return Object.fromEntries(await Promise.all(observedSources.map(async filename => [filename, createHash('sha256').update(await readFile(new URL('../' + filename, import.meta.url))).digest('hex')])));
}
async function actualCheckpointArchives() {
  const backboneBytes = await readFile(new URL('../public/models/yolo26n-features-128.onnx', import.meta.url));
  const engine = new TrainerEngine({backboneBytes}), results = [];
  const pixels = new Uint8Array(128 * 128 * 3).fill(30);
  for (let y = 20; y < 65; y++) for (let x = 30; x < 55; x++) pixels[(y * 128 + x) * 3] = 230;
  const samples = [{id: 'storage-gradient-fixture', width: 128, height: 128, pixels, split: 'train', boxes: [{id: 'person-box', classId: 0, x: 30, y: 20, width: 25, height: 45}]}];
  let checkpoint;
  try {
    for (const epochs of [1, 2, 3]) {
      const result = await engine.train({samples, classes: ['person'], checkpoint, config: {mode: 'pretrained', backend: 'cpu', epochs, batchSize: 1, augment: false, learningRate: .003, seed: 12345}});
      assert(result.evidence.parametersChanged); assert(result.evidence.maxGradientNorm > 0);
      checkpoint = result.checkpoint;
      assert.equal(checkpoint.training.step, epochs);
      results.push(Array.from(new Uint8Array(await (await packCheckpoint(checkpoint)).arrayBuffer())));
    }
    return results;
  } finally { await engine.dispose(); }
}

test('real IndexedDB commits, rollback, isolated restore and explicitly injected quota errors preserve data', {timeout: 120000}, async () => {
  const initialSourceHashes = await sourceHashes();
  const checkpointArchives = await actualCheckpointArchives();
  const base = '/local-yolo-trainer/';
  const vite = await createServer({root, configFile: fileURLToPath(new URL('../vite.config.js', import.meta.url)), logLevel: 'error', server: {host: '127.0.0.1', port: 0, strictPort: false}, plugins: [{
    name: 'storage-acceptance-harness',
    configureServer(server) { server.middlewares.use((request, response, next) => {
      if (!request.url?.endsWith('/storage-acceptance-harness')) return next();
      response.setHeader('Content-Type', 'text/html');
      response.end(`<!doctype html><html><meta charset="utf-8"><title>Isolated storage acceptance</title><body><p>Real IndexedDB acceptance harness</p><script type="module">
        import * as storage from '${base}src/data/storage.js';
        import * as project from '${base}src/data/project.js';
        import * as formats from '${base}src/data/formats.js';
        window.fixtureModules={storage,project,formats};
      </script></body></html>`);
    }); },
  }]});
  let browser;
  try {
    await vite.listen();
    const port = vite.httpServer.address().port;
    browser = await chromium.launch({headless: true});
    const context = await browser.newContext(), page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${port}${base}storage-acceptance-harness`);
    await page.waitForFunction(() => window.fixtureModules);
    const report = await page.evaluate(async archives => {
      const {storage, project: helpers, formats} = window.fixtureModules;
      const checkpoints = [];
      for (const archive of archives) checkpoints.push(await formats.unpackCheckpoint(new Blob([new Uint8Array(archive)], {type: 'application/zip'})));
      const source = helpers.createProject({title: 'Original persistent project', className: 'person'});
      const canvas = document.createElement('canvas'); canvas.width = 200; canvas.height = 100;
      const context = canvas.getContext('2d'); context.fillStyle = '#e8e4dc'; context.fillRect(0, 0, 200, 100); context.fillStyle = '#181818'; context.fillRect(20, 10, 40, 60);
      const image = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      source.images.push({id: crypto.randomUUID(), name: 'real-browser-canvas.png', mime: 'image/png', width: 200, height: 100, sha256: await helpers.sha256(image), boxes: [{id: 'original-box', classId: 0, x: 20, y: 10, width: 40, height: 60}], reviewed: true});
      helpers.assignSplits(source);
      const sourceId = source.id, imageId = source.images[0].id;
      await storage.saveImage(imageId, image, image); await storage.saveProject(source);
      await storage.saveCheckpoint(sourceId, checkpoints[0]); await storage.saveCheckpoint(sourceId, checkpoints[1]);
      const beforeCurrent = await storage.getCheckpoint(sourceId), beforePrevious = await storage.getPreviousCheckpoint(sourceId);
      const beforeProject = await storage.getProject(sourceId);
      if (beforeCurrent.training.step !== 2 || beforePrevious.training.step !== 1) throw new Error('Actual checkpoint rotation did not persist.');

      // Inject a duplicate-key operation at the final current write. The
      // ConstraintError and rollback are generated by the real IndexedDB
      // engine after the previous-checkpoint write has already been queued.
      const nativePut = IDBObjectStore.prototype.put;
      IDBObjectStore.prototype.put = function (value, key) {
        if (this.name === 'checkpoints' && key === sourceId + ':current') return this.add(value, key);
        return nativePut.apply(this, arguments);
      };
      let checkpointFailure;
      try { await storage.saveCheckpoint(sourceId, checkpoints[2]); }
      catch (error) { checkpointFailure = {name: error.name, message: error.message}; }
      finally { IDBObjectStore.prototype.put = nativePut; }
      if (!checkpointFailure) throw new Error('Injected actual duplicate-key transaction did not fail.');
      const afterCurrent = await storage.getCheckpoint(sourceId), afterPrevious = await storage.getPreviousCheckpoint(sourceId);
      const firstWeightHash = state => helpers.sha256(new Uint8Array(state.weights[0].data.buffer, state.weights[0].data.byteOffset, state.weights[0].data.byteLength));
      if (afterCurrent.training.step !== 2 || afterPrevious.training.step !== 1 || await firstWeightHash(afterCurrent) !== await firstWeightHash(beforeCurrent) || await firstWeightHash(afterPrevious) !== await firstWeightHash(beforePrevious)) throw new Error('Failed checkpoint transaction damaged a good current/previous checkpoint.');

      const archive = await formats.packProject(source, id => storage.getImage(id), afterCurrent), restored = await formats.unpackProject(archive);
      const thumbnails = new Map([...restored.images].map(([id, blob]) => [id, blob]));
      await storage.importProjectAtomic(restored.project, restored.images, thumbnails, restored.checkpoint);
      const readRestored = await storage.getProject(restored.project.id), readOriginal = await storage.getProject(sourceId);
      if (restored.project.id === sourceId || readRestored.images[0].id === imageId || JSON.stringify(readOriginal) !== JSON.stringify(beforeProject)) throw new Error('Backup import overwrote the original project or reused image IDs.');
      if (await helpers.sha256(await storage.getImage(readRestored.images[0].id)) !== source.images[0].sha256) throw new Error('Restored original image bytes changed.');
      if (JSON.stringify(readRestored.images[0].boxes) !== JSON.stringify(source.images[0].boxes) || await helpers.datasetDigest(readRestored) !== await helpers.datasetDigest(source)) throw new Error('Restored categories/annotations/dataset identity changed.');
      let duplicateImportRejected = false;
      try { await storage.importProjectAtomic(restored.project, restored.images, thumbnails, restored.checkpoint); } catch { duplicateImportRejected = true; }
      if (!duplicateImportRejected) throw new Error('Importing an existing project ID must fail instead of overwriting it.');

      const invalidProject = structuredClone(restored.project); invalidProject.id = crypto.randomUUID();
      const badCheckpoint = structuredClone(restored.checkpoint); delete badCheckpoint.optimizer.betaPowers;
      let directInvalidCheckpointRejected = false;
      try { await storage.importProjectAtomic(invalidProject, restored.images, thumbnails, badCheckpoint); } catch { directInvalidCheckpointRejected = true; }
      if (!directInvalidCheckpointRejected || await storage.getProject(invalidProject.id)) throw new Error('Direct import API stored an invalid training checkpoint.');
      const badImages = new Map(restored.images); badImages.set(restored.project.images[0].id, new Blob(['damaged'], {type: 'image/png'}));
      let directBadImageRejected = false;
      try { await storage.importProjectAtomic(invalidProject, badImages, thumbnails, restored.checkpoint); } catch (error) { directBadImageRejected = error.message === 'PROJECT_IMAGE_INTEGRITY'; }
      if (!directBadImageRejected || await storage.getProject(invalidProject.id)) throw new Error('Direct import API skipped original image SHA validation.');

      // Observe a real readwrite transaction completion before SaveQueue may
      // announce saved; this is not a mocked success callback.
      const order = [], nativeTransaction = IDBDatabase.prototype.transaction;
      IDBDatabase.prototype.transaction = function (stores, mode) {
        const transaction = nativeTransaction.apply(this, arguments);
        if (mode === 'readwrite' && [...transaction.objectStoreNames].includes('projects')) transaction.addEventListener('complete', () => order.push('actual-idb-oncomplete'));
        return transaction;
      };
      const states = [], queue = new storage.SaveQueue(value => storage.saveProject(value), state => { states.push(state); order.push(state); });
      source.title = 'Committed only after oncomplete'; queue.enqueue(source);
      const saving = queue.flush();
      if (!queue.dirty || states.includes('saved')) throw new Error('SaveQueue announced a successful commit synchronously.');
      await saving; IDBDatabase.prototype.transaction = nativeTransaction;
      if (queue.dirty || order.indexOf('actual-idb-oncomplete') < 0 || order.indexOf('saved') <= order.indexOf('actual-idb-oncomplete')) throw new Error('Saved was announced before the real IndexedDB transaction completed.');

      const beforeFailure = await storage.getProject(sourceId), quotaStates = [];
      const quotaQueue = new storage.SaveQueue(value => storage.saveProject(value), state => quotaStates.push(state));
      IDBObjectStore.prototype.put = function () { if (this.name === 'projects') throw new DOMException('Explicitly injected quota failure; not actual disk exhaustion.', 'QuotaExceededError'); return nativePut.apply(this, arguments); };
      source.title = 'Important edit retained after injected failure'; quotaQueue.enqueue(source);
      let quotaFailure;
      try { await quotaQueue.flush(); } catch (error) { quotaFailure = error.name; }
      finally { IDBObjectStore.prototype.put = nativePut; }
      if (quotaFailure !== 'QuotaExceededError' || !quotaQueue.dirty || quotaStates.includes('saved') || JSON.stringify(await storage.getProject(sourceId)) !== JSON.stringify(beforeFailure)) throw new Error('Injected quota error lost unsaved data or falsely announced saved.');
      await quotaQueue.flush();
      if (quotaQueue.dirty || quotaStates.at(-1) !== 'saved' || (await storage.getProject(sourceId)).title !== source.title) throw new Error('Real IndexedDB retry did not commit the retained edit.');

      const beforeRemoval = await storage.getProject(sourceId), nextProject = structuredClone(beforeRemoval);
      nextProject.images = nextProject.images.filter(item => item.id !== imageId); nextProject.updatedAt = new Date().toISOString();
      const nativeDelete = IDBObjectStore.prototype.delete;
      // Force a real native duplicate-key error after the new project metadata
      // is queued, proving that references and bytes roll back together.
      IDBObjectStore.prototype.delete = function (key) { if (this.name === 'images' && key === imageId) return this.add(image, key); return nativeDelete.apply(this, arguments); };
      let removalFailure;
      try { await storage.removeImageConfirmed(nextProject, imageId); } catch (error) { removalFailure = error.name; }
      finally { IDBObjectStore.prototype.delete = nativeDelete; }
      if (removalFailure !== 'ConstraintError' || JSON.stringify(await storage.getProject(sourceId)) !== JSON.stringify(beforeRemoval) || await helpers.sha256(await storage.getImage(imageId)) !== source.images[0].sha256 || !(await storage.getThumbnail(imageId) instanceof Blob)) throw new Error('Failed atomic removal lost a project reference, photo or thumbnail.');

      const staleProject = structuredClone(nextProject); staleProject.title = 'Unrelated stale change';
      let staleRemovalRejected = false;
      try { await storage.removeImageConfirmed(staleProject, imageId); } catch (error) { staleRemovalRejected = error.message === 'IMAGE_REMOVAL_CONFLICT'; }
      if (!staleRemovalRejected) throw new Error('Removing a picture must not silently replace unrelated project fields.');
      await storage.removeImageConfirmed(nextProject, imageId);
      if ((await storage.getProject(sourceId)).images.length !== 0 || await storage.getImage(imageId) !== undefined || await storage.getThumbnail(imageId) !== undefined) throw new Error('Confirmed removal left an orphan photo/thumbnail or a dangling project reference.');
      if ((await storage.getCheckpoint(sourceId)).training.step !== 2 || (await storage.getPreviousCheckpoint(sourceId)).training.step !== 1) throw new Error('Removing a photo damaged the saved model.');
      if (await helpers.sha256(await storage.getImage(readRestored.images[0].id)) !== readRestored.images[0].sha256) throw new Error('Removing the original project photo changed an independently restored project.');

      // Explicitly corrupt one stored timestamp through raw IDB, bypassing
      // the public validators, and verify healthy projects remain accessible.
      const database = await storage.openDatabase(), invalidRecord = structuredClone(nextProject);
      invalidRecord.id = 'deliberately-corrupt-date-fixture'; invalidRecord.updatedAt = null;
      await new Promise((resolve, reject) => { const transaction = database.transaction(['projects'], 'readwrite'); transaction.oncomplete = resolve; transaction.onabort = () => reject(transaction.error); transaction.objectStore('projects').put(invalidRecord); });
      const available = await storage.listProjects(), preservedInvalid = await storage.getProject(invalidRecord.id);
      if (available.length !== 2 || available.invalidProjects.length !== 1 || available.invalidProjects[0].id !== invalidRecord.id || preservedInvalid.updatedAt !== null) throw new Error('A corrupt stored timestamp blocked healthy projects or silently deleted the damaged record.');

      return {schemaVersion: 1, status: 'pass', timestamp: new Date().toISOString(), origin: location.origin, browser: navigator.userAgent,
        checkpointRotation: {status: 'pass', currentStep: 2, previousStep: 1, actualOptimizedCheckpoints: true},
        checkpointAtomicFailure: {status: 'pass', injectedOperation: 'Duplicate-key current write (native IDB.add) after previous write queued', actualError: checkpointFailure, currentAndPreviousRetained: true},
        backupRestore: {status: 'pass', originalId: sourceId, restoredId: readRestored.id, sourceUnchanged: true, exactImageSha256: source.images[0].sha256, boxesPreserved: true, digestStable: true, duplicateImportRejected, directInvalidCheckpointRejected, directBadImageRejected},
        commitOrdering: {status: 'pass', events: order},
        quotaFaultInjection: {status: 'pass', error: quotaFailure, states: quotaStates, dirtyRetained: true, actualRetryCommitted: true, limitation: 'QuotaExceededError was injected at IDB.put; this does not claim the physical disk or browser storage quota was full.'},
        atomicPictureRemoval: {status: 'pass', failure: removalFailure, referencesAndPhotoRetainedOnFailure: true, staleRemovalRejected, noOrphanPhotoOrThumbnailAfterSuccess: true, savedCurrentAndPreviousModelsKept: true, restoredProjectIndependent: true},
        corruptRecordIsolation: {status: 'pass', invalidRecordsSkipped: available.invalidProjects.length, healthyProjectsListed: available.length, invalidStoredRecordPreserved: true, injection: 'A timestamp was deliberately corrupted through raw IDB, bypassing the public import validator.'},
      };
    }, checkpointArchives);
    assert.equal(report.status, 'pass'); assert.deepEqual(errors, []);
    assert.deepEqual(await sourceHashes(), initialSourceHashes, 'Observed storage/training inputs must remain unchanged while this acceptance test runs.');
    await mkdir(new URL('../test-results/', import.meta.url), {recursive: true});
    await writeFile(new URL('../test-results/storage-browser.json', import.meta.url), JSON.stringify({...report, sourceHashes: initialSourceHashes, pageErrors: errors}, null, 2));
    console.log('ACTUAL_INDEXEDDB_ACCEPTANCE', JSON.stringify(report));
    await context.close();
  } finally { await browser?.close(); await vite.close(); }
});
