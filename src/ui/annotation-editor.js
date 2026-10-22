import { tr } from './locale.js';
import { clamp, normalizeBox, moveBox, resizeBox, validateBoxes, zoomAt, pinchView, createHistory } from './annotation-geometry.js';

const SVG = 'http://www.w3.org/2000/svg';
const copy = (boxes) => boxes.map((box) => ({ ...box }));
const make = (tag, attributes = {}, text = '') => {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === 'className') element.className = value;
    else element.setAttribute(key, value);
  }
  element.textContent = text;
  return element;
};
const svgElement = (tag, attributes = {}) => {
  const element = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
};

/**
 * Image and boxes always use original, EXIF-oriented image pixels. SVG's inverse
 * screen matrix includes responsive fitting, page scroll, DPR, pan and zoom.
 * Every user edit is committed once, after pointerup; interrupted gestures revert.
 */
export function createAnnotationEditor({ root, onChange = () => {}, onStatus = () => {}, readonly = false }) {
  if (!(root instanceof HTMLElement)) throw new TypeError('An editor root element is required.');
  const history = createHistory();
  const events = new AbortController();
  let boxes = [];
  let classes = [];
  let activeClass = 0;
  let selected = null;
  let width = 1;
  let height = 1;
  let view = { x: 0, y: 0, zoom: 1 };
  let objectURL = null;
  let loadToken = 0;
  let destroyed = false;
  let loaded = false;
  let drag = null;
  let pinch = null;
  let panMode = false;
  let moveSelected = false;
  let spaceHeld = false;
  let pendingDelete = null;
  let deleteTimer;
  const pointers = new Map();
  root.classList.add('annotation-editor');
  const toolbar = make('div', { className: 'editor-toolbar', role: 'toolbar' });
  const tools = make('div', { className: 'editor-tools' });
  const viewTools = make('div', { className: 'editor-tools' });
  const button = (action, text) => {
    const node = make('button', { className: 'button compact quiet', type: 'button', 'data-action': action }, text);
    (['zoom-out', 'zoom-in', 'fit', 'pan'].includes(action) ? viewTools : tools).append(node);
    return node;
  };
  const undoButton = button('undo', '↶');
  const redoButton = button('redo', '↷');
  const deleteButton = button('delete', '');
  const moveButton = button('move-selected', '');
  const classSelect = make('select', { className: 'editor-class-select', 'data-action': 'class' });
  tools.append(classSelect);
  const panButton = button('pan', '');
  const zoomOutButton = button('zoom-out', '−');
  const zoomInButton = button('zoom-in', '+');
  const fitButton = button('fit', '');
  toolbar.append(tools, viewTools);
  const canvas = make('div', { className: 'annotation-canvas', tabindex: '0', role: 'group' });
  const svg = svgElement('svg', { viewBox: '0 0 1 1', preserveAspectRatio: 'xMidYMid meet', 'aria-hidden': 'true' });
  const world = svgElement('g');
  const image = svgElement('image', { x: '0', y: '0', preserveAspectRatio: 'none' });
  const overlays = svgElement('g');
  world.append(image, overlays);
  svg.append(world);
  canvas.append(svg);
  const footer = make('div', { className: 'editor-footer' });
  const hint = make('span', { className: 'editor-hint', id: `editor-hint-${crypto.randomUUID()}` });
  const count = make('span', { className: 'editor-count', 'aria-live': 'polite' });
  footer.append(hint, count);
  const list = make('div', { className: 'annotation-list', role: 'list' });
  root.replaceChildren(toolbar, canvas, footer, list);
  canvas.setAttribute('aria-describedby', hint.id);
  if (readonly) {
    root.classList.add('is-readonly');
    tools.hidden = true;
  }

  function listen(element, type, callback, options = {}) {
    element.addEventListener(type, callback, { ...options, signal: events.signal });
  }
  function status() {
    onStatus({ selectedId: selected, canUndo: history.canUndo, canRedo: history.canRedo, zoom: view.zoom, count: boxes.length });
  }
  function selectedBox() { return boxes.find((box) => box.id === selected); }
  function clearDelete() {
    pendingDelete = null;
    clearTimeout(deleteTimer);
  }
  function commit(before) {
    if (history.record(before, boxes)) onChange(copy(boxes));
    render();
    status();
  }
  function clientToSvg(clientX, clientY, node = world) {
    const matrix = node.getScreenCTM();
    if (!matrix) return { x: 0, y: 0 };
    const point = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
    return { x: point.x, y: point.y };
  }
  function screenUnit() {
    const matrix = world.getScreenCTM();
    return matrix ? 1 / Math.max(1e-9, Math.hypot(matrix.a, matrix.b)) : 1;
  }
  function setView(next) {
    view = next;
    // Pan is intentionally generous: an image can be inspected up to each edge.
    view.x = clamp(view.x, -width * view.zoom, width);
    view.y = clamp(view.y, -height * view.zoom, height);
    if (view.zoom === 1) view = { x: 0, y: 0, zoom: 1 };
    world.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.zoom})`);
    renderBoxes();
    refreshToolbar();
  }
  function zoomBy(factor, clientPoint) {
    if (!loaded) return;
    const rectangle = canvas.getBoundingClientRect();
    const anchor = clientPoint ? clientToSvg(clientPoint.x, clientPoint.y, svg) : clientToSvg(rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height / 2, svg);
    setView(zoomAt(view, view.zoom * factor, anchor));
    status();
  }
  function refreshToolbar() {
    undoButton.disabled = !loaded || !history.canUndo;
    redoButton.disabled = !loaded || !history.canRedo;
    deleteButton.disabled = !selectedBox();
    moveButton.hidden = !selectedBox();
    moveButton.textContent = moveSelected ? tr('拖动以移动', 'Drag to move') : tr('移动选框', 'Move box');
    moveButton.setAttribute('aria-pressed', String(moveSelected));
    deleteButton.classList.toggle('danger', pendingDelete === selected && !!selected);
    deleteButton.textContent = pendingDelete === selected && selected ? tr('确认删除', 'Confirm delete') : tr('删除', 'Delete');
    classSelect.disabled = !loaded || classes.length === 0;
    const expected = String(selectedBox()?.classId ?? activeClass);
    if (classSelect.value !== expected) classSelect.value = expected;
    panButton.textContent = panMode ? tr('移动画面中', 'Pan enabled') : tr('移动画面', 'Pan');
    panButton.setAttribute('aria-pressed', String(panMode));
    canvas.classList.toggle('is-panning', panMode || spaceHeld || moveSelected);
    zoomOutButton.disabled = !loaded || view.zoom <= 1;
    zoomInButton.disabled = !loaded || view.zoom >= 12;
    fitButton.disabled = !loaded;
    fitButton.textContent = `${Math.round(view.zoom * 100)}%`;
    count.textContent = tr(`${boxes.length} 个方框`, `${boxes.length} ${boxes.length === 1 ? 'box' : 'boxes'}`);
  }
  function renderBoxes() {
    const fragment = document.createDocumentFragment();
    const unit = screenUnit();
    for (const box of boxes) {
      const chosen = selected === box.id;
      const group = svgElement('g', { 'data-box-id': box.id, class: `annotation-box${chosen ? ' selected' : ''}` });
      group.append(svgElement('rect', { x: box.x, y: box.y, width: box.width, height: box.height, 'vector-effect': 'non-scaling-stroke', class: 'box-outline' }));
      const name = classes[box.classId] ?? String(box.classId);
      const score = Number.isFinite(box.score) ? ` ${Math.round(box.score * 100)}%` : Number.isFinite(box.confidence) ? ` ${Math.round(box.confidence * 100)}%` : '';
      const label = `${name}${score}`;
      const labelWidth = Math.max(40, Math.min(220, [...label].reduce((sum, char) => sum + (char.charCodeAt(0) > 255 ? 12 : 7), 12))) * unit;
      const labelY = box.y > 24 * unit ? box.y - 23 * unit : box.y;
      group.append(svgElement('rect', { x: box.x, y: labelY, width: labelWidth, height: 23 * unit, rx: 3 * unit, class: 'box-label-bg' }));
      const text = svgElement('text', { x: box.x + 5 * unit, y: labelY + 16 * unit, 'font-size': 12 * unit, class: 'box-label' });
      text.textContent = label;
      group.append(text);
      fragment.append(group);
      if (chosen && !readonly) {
        const positions = { n: [0.5, 0], s: [0.5, 1], w: [0, 0.5], e: [1, 0.5], nw: [0, 0], ne: [1, 0], sw: [0, 1], se: [1, 1] };
        for (const [handle, [dx, dy]] of Object.entries(positions)) {
          const x = box.x + box.width * dx;
          const y = box.y + box.height * dy;
          const hit = 44 * unit;
          const handleGroup = svgElement('g', { 'data-box-id': box.id, 'data-handle': handle, class: `resize-handle handle-${handle}` });
          handleGroup.append(svgElement('rect', { x: x - hit / 2, y: y - hit / 2, width: hit, height: hit, fill: 'transparent', 'pointer-events': 'all' }));
          handleGroup.append(svgElement('rect', { x: x - 4 * unit, y: y - 4 * unit, width: 8 * unit, height: 8 * unit, rx: 1.5 * unit, class: 'handle-visible', 'vector-effect': 'non-scaling-stroke', 'pointer-events': 'none' }));
          fragment.append(handleGroup);
        }
      }
    }
    overlays.replaceChildren(fragment);
  }
  function renderList() {
    const fragment = document.createDocumentFragment();
    for (const [index, box] of boxes.entries()) {
      const item = make('button', { type: 'button', className: 'annotation-item', 'data-select-box': box.id, 'aria-pressed': String(selected === box.id), role: 'listitem' });
      item.append(make('span', { className: 'annotation-item-number' }, String(index + 1)), make('span', {}, classes[box.classId] ?? String(box.classId)));
      const confidence = box.score ?? box.confidence;
      item.append(make('small', { className: 'muted' }, readonly && Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : `${Math.round(box.width)} × ${Math.round(box.height)}`));
      fragment.append(item);
    }
    list.replaceChildren(fragment);
    list.hidden = boxes.length === 0;
  }
  function render() { renderBoxes(); renderList(); refreshToolbar(); }
  function renderClassOptions() {
    classSelect.replaceChildren(...classes.map((name, index) => make('option', { value: index }, name)));
    refreshToolbar();
  }
  function translate() {
    toolbar.setAttribute('aria-label', tr('标注工具', 'Annotation tools'));
    canvas.setAttribute('aria-label', readonly ? tr('检测结果，可缩放查看', 'Detection results. Zoom to inspect.') : tr('图片标注画布。拖动画框，选中后可移动或调整。', 'Image annotation canvas. Drag to draw; select a box to move or resize.'));
    classSelect.setAttribute('aria-label', tr('选框类别／新方框类别', 'Selected or new box class'));
    for (const [node, zh, en] of [[undoButton, '撤销', 'Undo'], [redoButton, '重做', 'Redo'], [zoomInButton, '放大', 'Zoom in'], [zoomOutButton, '缩小', 'Zoom out'], [fitButton, '适合窗口', 'Fit image']]) {
      node.title = tr(zh, en);
      node.setAttribute('aria-label', tr(zh, en));
    }
    hint.textContent = readonly ? tr('滚轮或双指缩放，拖动画面查看。', 'Scroll or pinch to zoom. Drag to explore.') : tr('拖动画框 · 双指缩放和平移 · 选中方框后调整', 'Drag to draw · Pinch to zoom and pan · Select to edit');
    render();
  }
  function rollbackGesture() {
    if (drag?.before) boxes = copy(drag.before);
    drag = null;
    render();
  }
  function gestureMetrics() {
    const [a, b] = [...pointers.values()];
    if (!a || !b) return null;
    const center = clientToSvg((a.x + b.x) / 2, (a.y + b.y) / 2, svg);
    return { center, distance: Math.hypot(a.x - b.x, a.y - b.y) };
  }
  listen(canvas, 'pointerdown', (event) => {
    if (!loaded || (event.button !== 0 && event.button !== 1)) return;
    event.preventDefault();
    clearDelete();
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size >= 2) {
      rollbackGesture();
      const metrics = gestureMetrics();
      pinch = metrics ? { ...metrics, view: { ...view } } : null;
      return;
    }
    const point = clientToSvg(event.clientX, event.clientY);
    if (panMode || spaceHeld || readonly || event.button === 1) {
      drag = { type: 'pan', start: clientToSvg(event.clientX, event.clientY, svg), view: { ...view } };
      return;
    }
    if (moveSelected && selectedBox()) {
      drag = { type: 'move', start: point, original: { ...selectedBox() }, before: copy(boxes), clientStart: { x: event.clientX, y: event.clientY } };
      render();
      return;
    }
    const handleElement = event.target.closest?.('[data-handle]');
    const boxElement = event.target.closest?.('[data-box-id]');
    const hit = boxes.find((box) => box.id === boxElement?.dataset.boxId);
    if (hit) {
      selected = hit.id;
      drag = { type: handleElement ? 'resize' : 'move', handle: handleElement?.dataset.handle, start: point, original: { ...hit }, before: copy(boxes), clientStart: { x: event.clientX, y: event.clientY } };
    } else {
      selected = null;
      if (point.x < 0 || point.y < 0 || point.x > width || point.y > height || !classes.length) { render(); return; }
      const id = crypto.randomUUID();
      drag = { type: 'create', start: point, id, before: copy(boxes), clientStart: { x: event.clientX, y: event.clientY } };
    }
    render();
    status();
  });
  listen(canvas, 'pointermove', (event) => {
    if (!pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pinch && pointers.size >= 2) {
      const metrics = gestureMetrics();
      if (metrics) setView(pinchView(pinch, metrics));
      return;
    }
    if (!drag) return;
    event.preventDefault();
    if (drag.type === 'pan') {
      const point = clientToSvg(event.clientX, event.clientY, svg);
      setView({ ...drag.view, x: drag.view.x + point.x - drag.start.x, y: drag.view.y + point.y - drag.start.y });
      return;
    }
    const point = clientToSvg(event.clientX, event.clientY);
    if (drag.type === 'create') {
      if (Math.hypot(event.clientX - drag.clientStart.x, event.clientY - drag.clientStart.y) < 4) return;
      const rectangle = normalizeBox(drag.start, point, width, height);
      boxes = [...drag.before, { id: drag.id, classId: activeClass, ...rectangle }];
      selected = drag.id;
    } else {
      const handlePoint = {
        x: (drag.handle?.includes('w') ? drag.original.x : drag.original.x + drag.original.width) + point.x - drag.start.x,
        y: (drag.handle?.includes('n') ? drag.original.y : drag.original.y + drag.original.height) + point.y - drag.start.y,
      };
      const next = drag.type === 'move' ? moveBox(drag.original, { x: point.x - drag.start.x, y: point.y - drag.start.y }, width, height) : resizeBox(drag.original, drag.handle, handlePoint, width, height);
      boxes = boxes.map((box) => box.id === next.id ? next : box);
    }
    renderBoxes();
    refreshToolbar();
  });
  function pointerEnd(event, cancel) {
    if (!pointers.has(event.pointerId)) return;
    pointers.delete(event.pointerId);
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (pinch) {
      if (pointers.size < 2) pinch = null;
      drag = null;
      status();
      return;
    }
    if (cancel) { rollbackGesture(); return; }
    if (drag?.before) {
      const before = drag.before;
      boxes = boxes.filter((box) => box.width >= 1 && box.height >= 1);
      drag = null;
      if (!selectedBox()) selected = null;
      commit(before);
    } else { drag = null; status(); }
  }
  listen(canvas, 'pointerup', (event) => pointerEnd(event, false));
  listen(canvas, 'pointercancel', (event) => pointerEnd(event, true));
  listen(canvas, 'lostpointercapture', (event) => { if (pointers.has(event.pointerId)) pointerEnd(event, true); });
  listen(canvas, 'wheel', (event) => {
    if (!loaded) return;
    event.preventDefault();
    zoomBy(Math.exp(-clamp(event.deltaY, -100, 100) * 0.004), { x: event.clientX, y: event.clientY });
  }, { passive: false });
  function undo() {
    if (readonly) return;
    rollbackGesture();
    const previous = history.undo(boxes);
    if (previous) { boxes = previous; selected = null; clearDelete(); onChange(copy(boxes)); render(); status(); }
  }
  function redo() {
    if (readonly) return;
    rollbackGesture();
    const next = history.redo(boxes);
    if (next) { boxes = next; selected = null; clearDelete(); onChange(copy(boxes)); render(); status(); }
  }
  function requestDelete() {
    if (readonly || !selectedBox()) return;
    if (pendingDelete === selected) {
      const before = copy(boxes);
      boxes = boxes.filter((box) => box.id !== selected);
      selected = null;
      clearDelete();
      commit(before);
    } else {
      pendingDelete = selected;
      clearTimeout(deleteTimer);
      deleteTimer = setTimeout(() => { pendingDelete = null; refreshToolbar(); }, 4000);
      refreshToolbar();
    }
  }
  listen(toolbar, 'click', (event) => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (action !== 'delete') clearDelete();
    if (action === 'undo') undo();
    if (action === 'redo') redo();
    if (action === 'delete') requestDelete();
    if (action === 'move-selected') { moveSelected = !moveSelected; panMode = false; refreshToolbar(); }
    if (action === 'zoom-in') zoomBy(1.35);
    if (action === 'zoom-out') zoomBy(1 / 1.35);
    if (action === 'fit') { setView({ x: 0, y: 0, zoom: 1 }); status(); }
    if (action === 'pan') { panMode = !panMode; moveSelected = false; refreshToolbar(); }
  });
  listen(classSelect, 'change', () => {
    clearDelete();
    const classId = Number(classSelect.value);
    if (!Number.isInteger(classId) || classId < 0 || classId >= classes.length) return;
    activeClass = classId;
    if (selectedBox()) {
      const before = copy(boxes);
      boxes = boxes.map((box) => box.id === selected ? { ...box, classId } : box);
      commit(before);
    }
  });
  listen(list, 'click', (event) => {
    const id = event.target.closest('[data-select-box]')?.dataset.selectBox;
    if (!id) return;
    selected = id;
    clearDelete();
    render();
    status();
  });
  listen(root, 'keydown', (event) => {
    if (event.target.matches('input,select,textarea,[contenteditable="true"]')) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) redo(); else undo(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') { event.preventDefault(); redo(); return; }
    if (event.code === 'Space' && event.target === canvas) { event.preventDefault(); spaceHeld = true; refreshToolbar(); return; }
    if (event.key === 'Escape') { rollbackGesture(); selected = null; clearDelete(); render(); return; }
    if (event.key === 'Delete' || event.key === 'Backspace') { if (selected && !readonly) { event.preventDefault(); requestDelete(); } return; }
    const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (directions[event.key] && selectedBox() && !readonly) {
      event.preventDefault();
      const [dx, dy] = directions[event.key];
      const before = copy(boxes);
      const next = moveBox(selectedBox(), { x: dx * (event.shiftKey ? 10 : 1), y: dy * (event.shiftKey ? 10 : 1) }, width, height);
      boxes = boxes.map((box) => box.id === selected ? next : box);
      commit(before);
    }
  });
  listen(window, 'keyup', (event) => { if (event.code === 'Space') { spaceHeld = false; refreshToolbar(); } });
  listen(window, 'blur', () => { spaceHeld = false; pointers.clear(); pinch = null; rollbackGesture(); });
  listen(window, 'languagechange', translate);
  const observer = new ResizeObserver(() => { if (!destroyed) renderBoxes(); });
  observer.observe(canvas);
  translate();

  return {
    async load({ blob, width: nextWidth, height: nextHeight, boxes: nextBoxes = [], classes: nextClasses = [], activeClassIndex = 0 }) {
      if (destroyed) throw new Error('Editor is destroyed.');
      if (!(blob instanceof Blob)) throw new TypeError('An image Blob is required.');
      if (!Array.isArray(nextClasses) || nextClasses.some((name) => typeof name !== 'string' || !name.trim())) throw new TypeError('Class names must be non-empty strings.');
      const validated = validateBoxes(nextBoxes, nextWidth, nextHeight, nextClasses.length);
      const token = ++loadToken;
      const nextURL = URL.createObjectURL(blob);
      try {
        const probe = new Image();
        probe.src = nextURL;
        await probe.decode();
        if (destroyed || token !== loadToken) { URL.revokeObjectURL(nextURL); return; }
        if (probe.naturalWidth !== nextWidth || probe.naturalHeight !== nextHeight) throw new TypeError(tr('图片尺寸与原图方向不一致，请重新导入图片。', 'Image dimensions do not match the oriented image. Import the picture again.'));
      } catch (error) { URL.revokeObjectURL(nextURL); throw error; }
      rollbackGesture();
      clearDelete();
      pointers.clear();
      pinch = null;
      const previousURL = objectURL;
      objectURL = nextURL;
      image.setAttribute('href', nextURL);
      if (previousURL) URL.revokeObjectURL(previousURL);
      width = nextWidth;
      height = nextHeight;
      boxes = validated;
      classes = [...nextClasses];
      activeClass = clamp(Number.isInteger(activeClassIndex) ? activeClassIndex : 0, 0, Math.max(0, classes.length - 1));
      selected = null;
      moveSelected = false;
      loaded = true;
      image.setAttribute('width', width);
      image.setAttribute('height', height);
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      canvas.style.setProperty('--image-ratio', `${width} / ${height}`);
      history.reset();
      renderClassOptions();
      setView({ x: 0, y: 0, zoom: 1 });
      render();
      status();
    },
    setClasses(next) {
      if (!Array.isArray(next) || next.some((name) => typeof name !== 'string' || !name.trim())) throw new TypeError('Class names must be non-empty strings.');
      if (boxes.some((box) => box.classId >= next.length)) throw new TypeError('Existing box classes must be remapped before removing a class.');
      if (next.length < classes.length) history.reset();
      classes = [...next];
      activeClass = clamp(activeClass, 0, Math.max(0, classes.length - 1));
      renderClassOptions();
      render();
    },
    setActiveClass(index) {
      if (!Number.isInteger(index) || index < 0 || index >= classes.length) throw new RangeError('Active class is not in the project.');
      activeClass = index;
      selected = null;
      clearDelete();
      render();
    },
    undo, redo,
    zoomIn() { zoomBy(1.35); },
    zoomOut() { zoomBy(1 / 1.35); },
    fit() { setView({ x: 0, y: 0, zoom: 1 }); status(); },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      ++loadToken;
      events.abort();
      observer.disconnect();
      clearTimeout(deleteTimer);
      pointers.clear();
      if (objectURL) URL.revokeObjectURL(objectURL);
      objectURL = null;
      root.replaceChildren();
      root.classList.remove('annotation-editor', 'is-readonly');
    },
  };
}
