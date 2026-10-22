export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export function normalizeBox(start, end, width, height) {
  const a = { x: clamp(start.x, 0, width), y: clamp(start.y, 0, height) };
  const b = { x: clamp(end.x, 0, width), y: clamp(end.y, 0, height) };
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), width: Math.abs(a.x - b.x), height: Math.abs(a.y - b.y) };
}

export function moveBox(box, delta, width, height) {
  return { ...box, x: clamp(box.x + delta.x, 0, width - box.width), y: clamp(box.y + delta.y, 0, height - box.height) };
}

export function resizeBox(box, handle, point, width, height, minSize = 1) {
  let left = box.x;
  let top = box.y;
  let right = box.x + box.width;
  let bottom = box.y + box.height;
  if (handle.includes('w')) left = clamp(point.x, 0, right - minSize);
  if (handle.includes('e')) right = clamp(point.x, left + minSize, width);
  if (handle.includes('n')) top = clamp(point.y, 0, bottom - minSize);
  if (handle.includes('s')) bottom = clamp(point.y, top + minSize, height);
  return { ...box, x: left, y: top, width: right - left, height: bottom - top };
}

export function validateBoxes(boxes, width, height, classCount) {
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) throw new TypeError('Image dimensions must be positive.');
  if (!Array.isArray(boxes)) throw new TypeError('Boxes must be an array.');
  const ids = new Set();
  return boxes.map((box) => {
    if (!box || typeof box.id !== 'string' || !box.id || ids.has(box.id)) throw new TypeError('Every box must have a unique ID.');
    ids.add(box.id);
    if (!Number.isInteger(box.classId) || box.classId < 0 || box.classId >= classCount) throw new TypeError('Box class is not in the project.');
    for (const key of ['x', 'y', 'width', 'height']) if (!Number.isFinite(box[key])) throw new TypeError('Box coordinates must be finite.');
    if (box.x < 0 || box.y < 0 || box.width <= 0 || box.height <= 0 || box.x + box.width > width + 1e-6 || box.y + box.height > height + 1e-6) throw new TypeError('Box is outside the original image.');
    return { ...box };
  });
}

export function zoomAt(view, nextZoom, anchor) {
  const zoom = clamp(nextZoom, 1, 12);
  const ratio = zoom / view.zoom;
  return { zoom, x: anchor.x - (anchor.x - view.x) * ratio, y: anchor.y - (anchor.y - view.y) * ratio };
}

export function pinchView(initial, current) {
  if (!(initial.distance > 0)) return { ...initial.view };
  const zoomed = zoomAt(initial.view, initial.view.zoom * current.distance / initial.distance, initial.center);
  return { ...zoomed, x: zoomed.x + current.center.x - initial.center.x, y: zoomed.y + current.center.y - initial.center.y };
}

export function createHistory(limit = 100) {
  let undo = [];
  let redo = [];
  const clone = (boxes) => boxes.map((box) => ({ ...box }));
  return {
    record(before, after) {
      if (JSON.stringify(before) === JSON.stringify(after)) return false;
      undo.push(clone(before));
      if (undo.length > limit) undo.shift();
      redo = [];
      return true;
    },
    undo(current) { if (!undo.length) return null; redo.push(clone(current)); return undo.pop(); },
    redo(current) { if (!redo.length) return null; undo.push(clone(current)); return redo.pop(); },
    reset() { undo = []; redo = []; },
    get canUndo() { return undo.length > 0; },
    get canRedo() { return redo.length > 0; },
  };
}
