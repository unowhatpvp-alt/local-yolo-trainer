// Developer-only real-photograph fixtures. The application never downloads a
// user's media. The images/masks/ZIP live under ignored tests/fixtures/.
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {inflateSync} from 'node:zlib';
import {unzipSync} from 'fflate';

export const REAL_FIXTURE_SOURCE = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip';
export const REAL_FIXTURE_SOURCE_SHA256 = '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2';
const digest = input => createHash('sha256').update(input).digest('hex');

/** Lossless 8-bit PNG decoder for the official RGB photos/indexed masks. */
export function decodeFixturePng(input) {
  const bytes = Buffer.from(input);
  if (!bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error('Fixture is not a PNG.');
  let width, height, bitDepth, colorType, interlace, palette, ended = false;
  const data = [];
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const length = bytes.readUInt32BE(offset), type = bytes.toString('ascii', offset + 4, offset + 8), end = offset + 8 + length;
    if (end + 4 > bytes.length) throw new Error('Truncated PNG fixture.');
    const payload = bytes.subarray(offset + 8, end);
    if (type === 'IHDR') { width = payload.readUInt32BE(0); height = payload.readUInt32BE(4); bitDepth = payload[8]; colorType = payload[9]; interlace = payload[12]; }
    if (type === 'PLTE') palette = payload;
    if (type === 'IDAT') data.push(payload);
    if (type === 'IEND') { ended = true; break; }
    offset = end + 4;
  }
  if (!ended || bitDepth !== 8 || interlace !== 0 || !width || !height || width * height > 20_000_000 || ![0, 2, 3, 4, 6].includes(colorType)) throw new Error(`Unsupported fixture PNG layout ${width}x${height}, depth${bitDepth}, color${colorType}, interlace${interlace}.`);
  const channels = ({0: 1, 2: 3, 3: 1, 4: 2, 6: 4})[colorType], stride = width * channels;
  const inflated = inflateSync(Buffer.concat(data), {maxOutputLength: (stride + 1) * height});
  if (inflated.length !== (stride + 1) * height) throw new Error('PNG fixture byte count differs from dimensions.');
  const raw = new Uint8Array(stride * height);
  const paeth = (left, above, upperLeft) => {
    const prediction = left + above - upperLeft, a = Math.abs(prediction - left), b = Math.abs(prediction - above), c = Math.abs(prediction - upperLeft);
    return a <= b && a <= c ? left : b <= c ? above : upperLeft;
  };
  for (let y = 0; y < height; y++) {
    const filter = inflated[y * (stride + 1)];
    if (filter > 4) throw new Error('Invalid PNG fixture filter.');
    for (let x = 0; x < stride; x++) {
      const target = y * stride + x, left = x >= channels ? raw[target - channels] : 0, above = y ? raw[target - stride] : 0, upperLeft = y && x >= channels ? raw[target - stride - channels] : 0;
      const predictor = [0, left, above, Math.floor((left + above) / 2), paeth(left, above, upperLeft)][filter];
      raw[target] = (inflated[y * (stride + 1) + 1 + x] + predictor) & 255;
    }
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let index = 0; index < width * height; index++) for (let c = 0; c < 3; c++) rgb[index * 3 + c] = colorType === 3 ? palette[raw[index] * 3 + c] : raw[index * channels + (colorType === 0 || colorType === 4 ? 0 : c)];
  return {width, height, channels, colorType, raw, pixels: rgb};
}

export async function loadRealFixtureSamples(directory = new URL('../tests/fixtures/penn-fudan/', import.meta.url)) {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', directory), 'utf8'));
  const samples = [];
  for (const sample of manifest.samples) {
    const bytes = await readFile(new URL(sample.path, directory));
    if (digest(bytes) !== sample.sha256) throw new Error(`Real photograph fixture integrity failure: ${sample.id}`);
    const decoded = decodeFixturePng(bytes);
    samples.push({...sample, pixels: decoded.pixels});
  }
  return {manifest, samples};
}

/** Final test photos are deliberately separate from train/validation input. */
export async function loadFinalTestFixtureSamples(directory = new URL('../tests/fixtures/penn-fudan/', import.meta.url)) {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', directory), 'utf8'));
  if (!Array.isArray(manifest.samplesTest) || manifest.samplesTest.length !== 8) throw new Error('The separately frozen final test set is missing.');
  const samples = [];
  for (const sample of manifest.samplesTest) {
    const bytes = await readFile(new URL(sample.path, directory));
    if (digest(bytes) !== sample.sha256) throw new Error(`Final test photograph integrity failure: ${sample.id}`);
    samples.push({...sample, pixels: decodeFixturePng(bytes).pixels});
  }
  return {manifest, samples};
}

async function prepare() {
  const root = fileURLToPath(new URL('../tests/fixtures/penn-fudan/', import.meta.url));
  const checkedPath = relative => {
    if (path.isAbsolute(relative) || relative.includes('\\') || relative.split('/').includes('..')) throw new Error(`Unsafe fixture ZIP member ${relative}.`);
    const target = path.resolve(root, relative), relation = path.relative(root, target);
    if (!relation || relation.startsWith('..') || path.isAbsolute(relation)) throw new Error(`Fixture path leaves its intended directory: ${relative}.`);
    return target;
  };
  await mkdir(root, {recursive: true});
  const zipPath = checkedPath('PennFudanPed.zip');
  let archive;
  try { archive = new Uint8Array(await readFile(zipPath)); }
  catch {
    const response = await fetch(REAL_FIXTURE_SOURCE, {signal: AbortSignal.timeout(180000), credentials: 'omit'});
    if (!response.ok) throw new Error(`Official dataset download failed: HTTP ${response.status}.`);
    const length = Number(response.headers.get('content-length'));
    if (length > 80 * 1024 * 1024) throw new Error('Official dataset archive exceeded preparation limit.');
    archive = new Uint8Array(await response.arrayBuffer());
    if (archive.length > 80 * 1024 * 1024) throw new Error('Dataset archive exceeded preparation limit.');
    if (digest(archive) !== REAL_FIXTURE_SOURCE_SHA256) throw new Error('Official dataset download SHA-256 differs from the pinned validation source.');
    await writeFile(zipPath, archive);
  }
  if (archive.length !== 53723336 || digest(archive) !== REAL_FIXTURE_SOURCE_SHA256) throw new Error('Cached official dataset size/SHA-256 mismatch.');
  let uncompressed = 0;
  const entries = unzipSync(archive, {filter: member => {
    checkedPath(member.name.replace(/\/$/, '') || 'archive-root');
    uncompressed += member.originalSize;
    if (member.originalSize > 20 * 1024 * 1024 || uncompressed > 100 * 1024 * 1024) throw new Error('Dataset ZIP exceeded uncompressed preparation limits.');
    return /^PennFudanPed\/(?:PNGImages|PedMasks)\/(?:Penn|Fudan)Ped\d+(?:_mask)?\.png$/.test(member.name);
  }});
  const candidates = [], exclusions = [];
  for (const filename of Object.keys(entries).filter(name => name.includes('/PNGImages/')).sort()) {
    const id = path.posix.basename(filename, '.png'), maskPath = `PennFudanPed/PedMasks/${id}_mask.png`;
    if (!entries[maskPath]) throw new Error(`Official mask missing for ${id}.`);
    const photo = decodeFixturePng(entries[filename]), mask = decodeFixturePng(entries[maskPath]);
    if (photo.width !== mask.width || photo.height !== mask.height || ![0, 3].includes(mask.colorType)) throw new Error(`Mask dimensions/type disagree for ${id}.`);
    const objects = new Map();
    for (let y = 0; y < mask.height; y++) for (let x = 0; x < mask.width; x++) {
      const object = mask.raw[y * mask.width + x];
      if (!object) continue;
      const box = objects.get(object) || {left: x, top: y, right: x, bottom: y};
      box.left = Math.min(box.left, x); box.top = Math.min(box.top, y); box.right = Math.max(box.right, x); box.bottom = Math.max(box.bottom, y); objects.set(object, box);
    }
    const boxes = [...objects.entries()].sort(([a], [b]) => a - b).map(([object, box]) => ({id: `${id}-${object}`, classId: 0, x: box.left, y: box.top, width: box.right + 1 - box.left, height: box.bottom + 1 - box.top}));
    const ratio = Math.min(128 / photo.width, 128 / photo.height), rw = Math.round(photo.width * ratio), rh = Math.round(photo.height * ratio), counts = new Map();
    for (const box of boxes) {
      const x = Math.floor(((box.x + box.width / 2) * rw / photo.width + Math.floor((128 - rw) / 2)) / 8), y = Math.floor(((box.y + box.height / 2) * rh / photo.height + Math.floor((128 - rh) / 2)) / 8);
      const cell = `${x},${y}`; counts.set(cell, (counts.get(cell) || 0) + 1);
    }
    if (!boxes.length || [...counts.values()].some(count => count > 2)) { exclusions.push({id, reason: boxes.length ? 'Three or more actual object centers collide in a 128px two-slot cell.' : 'No labeled pedestrian.'}); continue; }
    const number = Number(id.match(/\d+$/)[0]), split = number <= (id.startsWith('Fudan') ? 48 : 64) ? 'train' : 'val';
    candidates.push({id, path: `images/${id}.png`, width: photo.width, height: photo.height, sha256: digest(entries[filename]), maskSha256: digest(entries[maskPath]), boxes, split, ranking: digest(Buffer.from(`penn-fudan-trainer-fixture-v1:${id}`)), originalPhotoPath: filename, originalMaskPath: maskPath});
  }
  const selected = [];
  for (const [split, count] of [['train', 32], ['val', 8]]) {
    const pool = candidates.filter(sample => sample.split === split).sort((a, b) => a.ranking.localeCompare(b.ranking));
    if (pool.length < count) throw new Error(`Not enough real ${split} samples.`);
    selected.push(...pool.slice(0, count));
  }
  const selectedIds = new Set(selected.map(sample => sample.id));
  const finalTest = candidates.filter(sample => sample.split === 'val' && !selectedIds.has(sample.id)).sort((a, b) => a.ranking.localeCompare(b.ranking)).slice(0, 8).map(sample => ({...sample, split: 'test'}));
  if (finalTest.length !== 8) throw new Error('Not enough independent final test photographs.');
  if (new Set([...selected, ...finalTest].map(sample => sample.sha256)).size !== selected.length + finalTest.length) throw new Error('A duplicate image SHA would cross train/validation/final-test sets.');
  await mkdir(checkedPath('images'), {recursive: true});
  for (const sample of [...selected, ...finalTest]) await writeFile(checkedPath(sample.path), entries[sample.originalPhotoPath]);
  let previous;
  try { previous = JSON.parse(await readFile(checkedPath('manifest.json'), 'utf8')); } catch { /* first preparation */ }
  const manifest = {
    schemaVersion: 1, dataset: 'Penn-Fudan pedestrian photographs, deterministic 40-image diagnostic subset', classes: ['person'],
    source: {url: REAL_FIXTURE_SOURCE, website: 'https://www.cis.upenn.edu/~jshi/ped_html/', tutorial: 'https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html', zipSha256: digest(archive), zipBytes: archive.length, license: 'No explicit SPDX license found on original public dataset page; retain author attribution, use only local developer validation and do not redistribute the photographs with this application.', authors: ['Liming Wang', 'Jianbo Shi', 'Gang Song', 'I-fan Shen'], publication: 'Object Detection Combining Recognition and Segmentation, ACCV 2007'},
    splitPolicy: 'Before model training: Fudan numeric IDs 1–48 / Penn 1–64 are train candidates; later IDs validation candidates. Rank each pool by SHA256(penn-fudan-trainer-fixture-v1:ID); select 32 train and 8 validation. No prediction-dependent selection. Image/SHA-disjoint; scene/person-identity independence cannot be established from dataset metadata.',
    annotationPolicy: 'For each nonzero mask instance ID, box is min pixel to max pixel + 1 in original PNG pixels. Include every labeled person; class ID zero means person in this detector, not background.',
    finalTestPolicy: 'Frozen after initial train/validation diagnostics, before deeper-feature architecture selection. Use the next eight unused validation-pool IDs in exactly the same SHA ranking. These eight original photos are never passed to train(); only final predict/evaluate after architecture/hyperparameters are fixed. The existing validation set may be used for model selection. No final-test prediction was used in selection.',
    finalTestFrozenAt: previous?.finalTestFrozenAt || new Date().toISOString(),
    exclusions, counts: {sourcePhotos: candidates.length + exclusions.length, selected: selected.length, train: 32, validation: 8, finalTest: 8, trainBoxes: selected.filter(sample => sample.split === 'train').reduce((sum, sample) => sum + sample.boxes.length, 0), validationBoxes: selected.filter(sample => sample.split === 'val').reduce((sum, sample) => sum + sample.boxes.length, 0), finalTestBoxes: finalTest.reduce((sum, sample) => sum + sample.boxes.length, 0)},
    limitations: ['Small diagnostic subset; not a published benchmark.', 'YOLO26n pretrained source includes COCO person; absence of overlap with pretraining cannot be proven.', 'The 128px/two-slot engine cannot represent every dense scene; all pre-training exclusions are recorded.'],
    samples: selected, samplesTest: finalTest,
  };
  await writeFile(checkedPath('manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify({path: root, source: manifest.source, counts: manifest.counts, exclusions}, null, 2));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await prepare();
