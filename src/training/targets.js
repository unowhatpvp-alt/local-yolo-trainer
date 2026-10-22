export class TrainingError extends Error {
  constructor(message, code = 'TRAINING_ERROR') { super(message); this.name = 'TrainingError'; this.code = code; }
}

export function validateClasses(classes) {
  if (!Array.isArray(classes) || !classes.length || classes.length > 32 || classes.some(c => typeof c !== 'string' || !c.trim() || c.length > 80)) {
    throw new TrainingError('Use 1–32 nonempty class names, each up to 80 characters.', 'INVALID_CLASSES');
  }
  if (new Set(classes.map(c => c.trim())).size !== classes.length) throw new TrainingError('Class names must be unique.', 'INVALID_CLASSES');
  return classes.map(c => c.trim());
}

export function createRng(seed = 12345) {
  let state = Number(seed) >>> 0 || 12345;
  return {
    next() { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; },
    get state() { return state >>> 0; },
    set state(value) { if (!Number.isInteger(value) || value <= 0 || value > 4294967295) throw new TrainingError('Invalid random state.', 'INVALID_CHECKPOINT'); state = value >>> 0; },
  };
}

export function shuffledIndices(count, rng) {
  const order = Array.from({length: count}, (_, i) => i);
  for (let i = count - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  return order;
}

export function letterboxGeometry(width, height, resolution = 128) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || width * height > 100_000_000) throw new TrainingError('Invalid or excessively large image dimensions.', 'INVALID_IMAGE');
  const ratio = Math.min(resolution / width, resolution / height);
  const resizedWidth = Math.max(1, Math.round(width * ratio));
  const resizedHeight = Math.max(1, Math.round(height * ratio));
  return {width, height, resolution, resizedWidth, resizedHeight, padX: Math.floor((resolution - resizedWidth) / 2), padY: Math.floor((resolution - resizedHeight) / 2), scaleX: resizedWidth / width, scaleY: resizedHeight / height};
}

export function validateBoxes(boxes, width, height, classCount) {
  if (!Array.isArray(boxes) || boxes.length > 512) throw new TrainingError('An image must contain at most 512 boxes.', 'INVALID_BOX');
  return boxes.map(box => {
    if (!box || !Number.isInteger(box.classId) || box.classId < 0 || box.classId >= classCount || ![box.x, box.y, box.width, box.height].every(Number.isFinite) || box.width <= 0 || box.height <= 0 || box.x < 0 || box.y < 0 || box.x + box.width > width + 0.01 || box.y + box.height > height + 0.01) throw new TrainingError('A box or its class is outside the source image.', 'INVALID_BOX');
    return {...box};
  });
}

export function inputBoxes(boxes, geometry, flip = false) {
  return boxes.map(box => {
    const width = box.width * geometry.scaleX, height = box.height * geometry.scaleY;
    let x = box.x * geometry.scaleX + geometry.padX;
    if (flip) x = geometry.resolution - x - width;
    return {...box, x, y: box.y * geometry.scaleY + geometry.padY, width, height};
  });
}

/** Two actual object assignments per spatial cell. Nothing is silently omitted. */
export function encodeTargets(boxSets, classCount, resolution = 128) {
  const grid = resolution / 8, slots = 2, channels = 5 + classCount;
  if (!Number.isInteger(grid) || grid < 1) throw new TrainingError('Resolution must be divisible by 8.', 'INVALID_CONFIG');
  const data = new Float32Array(boxSets.length * grid * grid * slots * channels);
  let count = 0;
  boxSets.forEach((boxes, batch) => {
    const occupied = new Uint8Array(grid * grid);
    validateBoxes(boxes, resolution, resolution, classCount).forEach(box => {
      const cx = (box.x + box.width / 2) / 8, cy = (box.y + box.height / 2) / 8;
      const gx = Math.min(grid - 1, Math.floor(cx)), gy = Math.min(grid - 1, Math.floor(cy));
      const cell = gy * grid + gx, slot = occupied[cell]++;
      if (slot >= slots) throw new TrainingError('Three objects have centers in one small grid cell. Move or resize the boxes, or use a future higher-resolution engine. All annotations were retained.', 'TARGET_COLLISION');
      const offset = ((batch * grid * grid + cell) * slots + slot) * channels;
      data.set([cx - gx, cy - gy, box.width / resolution, box.height / resolution, 1], offset);
      data[offset + 5 + box.classId] = 1; count++;
    });
  });
  return {data, shape: [boxSets.length, grid, grid, slots, channels], count};
}

export function boxIou(a, b) {
  const intersection = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return intersection / (a.width * a.height + b.width * b.height - intersection + 1e-9);
}

export function decodePredictions(logits, classes, geometry, confidence = .25, iou = .45) {
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) throw new TrainingError('Confidence must be between 0 and 1.', 'INVALID_CONFIG');
  const grid = geometry.resolution / 8, channels = 5 + classes.length;
  if (logits.length !== grid * grid * 2 * channels || !logits.every(Number.isFinite)) throw new TrainingError('The detector returned invalid values.', 'INVALID_OUTPUT');
  const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  const found = [];
  for (let gy = 0; gy < grid; gy++) for (let gx = 0; gx < grid; gx++) for (let slot = 0; slot < 2; slot++) {
    const offset = ((gy * grid + gx) * 2 + slot) * channels;
    let max = -Infinity, classId = 0;
    for (let c = 0; c < classes.length; c++) if (logits[offset + 5 + c] > max) { max = logits[offset + 5 + c]; classId = c; }
    let denominator = 0; for (let c = 0; c < classes.length; c++) denominator += Math.exp(logits[offset + 5 + c] - max);
    const score = sigmoid(logits[offset + 4]) / denominator;
    if (score < confidence) continue;
    const cx = (sigmoid(logits[offset]) + gx) * 8, cy = (sigmoid(logits[offset + 1]) + gy) * 8;
    const width = sigmoid(logits[offset + 2]) * geometry.resolution, height = sigmoid(logits[offset + 3]) * geometry.resolution;
    const x1 = Math.max(0, Math.min(geometry.width, (cx - width / 2 - geometry.padX) / geometry.scaleX));
    const y1 = Math.max(0, Math.min(geometry.height, (cy - height / 2 - geometry.padY) / geometry.scaleY));
    const x2 = Math.max(0, Math.min(geometry.width, (cx + width / 2 - geometry.padX) / geometry.scaleX));
    const y2 = Math.max(0, Math.min(geometry.height, (cy + height / 2 - geometry.padY) / geometry.scaleY));
    if (x2 > x1 && y2 > y1) found.push({classId, label: classes[classId], score, box: {x: x1, y: y1, width: x2 - x1, height: y2 - y1}});
  }
  const kept = [];
  for (const item of found.sort((a, b) => b.score - a.score)) {
    if (!kept.some(other => other.classId === item.classId && boxIou(other.box, item.box) > iou)) kept.push(item);
    if (kept.length === 100) break;
  }
  return kept;
}

export function evaluateDetections(predictions, samples, threshold = .5) {
  let tp = 0, fp = 0, fn = 0, matchedIou = 0;
  predictions.forEach((items, index) => {
    const boxes = samples[index].boxes, matched = new Set();
    for (const item of items) {
      let best = -1, bestIou = threshold;
      boxes.forEach((box, i) => { if (matched.has(i) || box.classId !== item.classId) return; const value = boxIou(box, item.box); if (value >= bestIou) { best = i; bestIou = value; } });
      if (best < 0) fp++; else { tp++; matched.add(best); matchedIou += bestIou; }
    }
    fn += boxes.length - matched.size;
  });
  return {images: samples.length, truePositives: tp, falsePositives: fp, falseNegatives: fn, precision50: tp + fp ? tp / (tp + fp) : null, recall50: tp + fn ? tp / (tp + fn) : null, meanMatchedIou: tp ? matchedIou / tp : null};
}
