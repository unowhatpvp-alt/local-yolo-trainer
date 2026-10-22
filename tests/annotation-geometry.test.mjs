import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBox, moveBox, resizeBox, validateBoxes, zoomAt, pinchView, createHistory } from '../src/ui/annotation-geometry.js';

test('reverse drawing and out-of-image pointers retain original-image pixels', () => {
  assert.deepEqual(normalizeBox({ x: 180, y: 120 }, { x: -20, y: 500 }, 400, 300), { x: 0, y: 120, width: 180, height: 180 });
});

test('move clamps each edge and preserves dimensions and class metadata', () => {
  const box = { id: 'box', classId: 2, x: 40, y: 30, width: 100, height: 80 };
  assert.deepEqual(moveBox(box, { x: 1000, y: -1000 }, 400, 300), { ...box, x: 300, y: 0 });
});

test('all eight resize handles preserve opposite edges and stay inside the original image', () => {
  const box = { id: 'b', classId: 0, x: 40, y: 30, width: 100, height: 80 };
  const expected = {
    n: [40, 10, 100, 100], s: [40, 30, 100, 170], w: [10, 30, 130, 80], e: [40, 30, 260, 80],
    nw: [10, 10, 130, 100], ne: [40, 10, 260, 100], sw: [10, 30, 130, 170], se: [40, 30, 260, 170],
  };
  for (const handle of Object.keys(expected)) {
    const next = resizeBox(box, handle, { x: handle.includes('w') ? 10 : 300, y: handle.includes('n') ? 10 : 200 }, 400, 300);
    assert.deepEqual([next.x, next.y, next.width, next.height], expected[handle], handle);
    assert.equal(next.id, 'b');
    assert.equal(next.classId, 0);
  }
  const crossed = resizeBox(box, 'nw', { x: 999, y: 999 }, 400, 300);
  assert.deepEqual([crossed.x, crossed.y, crossed.width, crossed.height], [139, 109, 1, 1]);
});

test('zoom and pinch keep the original-image anchor invariant', () => {
  const view = { x: -20, y: 10, zoom: 2 };
  const anchor = { x: 150, y: 80 };
  const next = zoomAt(view, 4, anchor);
  assert.equal((anchor.x - view.x) / view.zoom, (anchor.x - next.x) / next.zoom);
  assert.equal((anchor.y - view.y) / view.zoom, (anchor.y - next.y) / next.zoom);
  const pinched = pinchView({ view, center: anchor, distance: 100 }, { center: { x: 170, y: 110 }, distance: 200 });
  assert.deepEqual(pinched, { x: -170, y: -30, zoom: 4 });
  assert.equal(zoomAt(view, 100, anchor).zoom, 12);
  assert.equal(zoomAt(view, .001, anchor).zoom, 1);
});

test('annotation loading rejects invalid classes, duplicate IDs, nonfinite and out-of-bounds coordinates', () => {
  const valid = { id: 'a', classId: 0, x: .5, y: 2, width: 50.2, height: 20 };
  const loaded = validateBoxes([valid], 100, 80, 1);
  assert.deepEqual(loaded, [valid]);
  assert.notEqual(loaded[0], valid);
  for (const patch of [{ classId: 1 }, { x: NaN }, { x: -1 }, { width: 0 }, { width: Infinity }, { height: 100 }]) assert.throws(() => validateBoxes([{ ...valid, ...patch }], 100, 80, 1));
  assert.throws(() => validateBoxes([valid, valid], 100, 80, 1));
  assert.throws(() => validateBoxes([valid], 0, 80, 1));
});

test('undo/redo restores independent snapshots and a new edit invalidates only redo', () => {
  const history = createHistory();
  const first = [{ id: 'a', x: 1 }];
  const second = [{ id: 'a', x: 20 }];
  assert.equal(history.record(first, second), true);
  first[0].x = 99;
  assert.deepEqual(history.undo(second), [{ id: 'a', x: 1 }]);
  assert.equal(history.canRedo, true);
  assert.deepEqual(history.redo([{ id: 'a', x: 1 }]), second);
  assert.equal(history.record(second, second), false);
  history.undo(second);
  history.record([{ id: 'a', x: 1 }], [{ id: 'b', x: 2 }]);
  assert.equal(history.canRedo, false);
  history.reset();
  assert.equal(history.canUndo, false);
});
