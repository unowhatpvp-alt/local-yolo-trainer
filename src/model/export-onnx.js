import {graph, model, node, tensor, valueInfo, readGraph, namedField} from './onnx-proto.js';
import {BACKBONE} from './backbone.js';

const equalShape = (actual, expected) => Array.isArray(actual) && actual.length === expected.length && actual.every((n, i) => n === expected[i]);
export function sameUpperPadding(size, kernel = 3, stride = 2) {
  const output = Math.ceil(size / stride), total = Math.max(0, (output - 1) * stride + kernel - size);
  return {before: Math.floor(total / 2), after: total - Math.floor(total / 2), output};
}

/** Exports a real, self-contained FP32 ONNX graph. Does not train any weights. */
export function exportOnnx(checkpoint, {backboneBytes} = {}) {
  if (checkpoint?.schemaVersion !== 1 || !['tiny-grid-v1', 'yolo26n-frozen-head-v1'].includes(checkpoint.architecture)) throw new Error('Unsupported Trainer checkpoint architecture/version. An inference ONNX is not a checkpoint.');
  const {architecture, resolution, classes, weights} = checkpoint;
  if (resolution !== 128 || checkpoint.slots !== 2) throw new Error('Export expects the verified resolution 128 and exactly two detection slots. Higher resolutions are not enabled.');
  if (!Array.isArray(classes) || !classes.length || classes.length > 256 || classes.some(name => typeof name !== 'string' || !name.trim()) || new Set(classes).size !== classes.length) throw new Error('Export requires 1–256 unique, nonempty class names in class-ID order.');
  if (!Array.isArray(weights) || new Set(weights.map(weight => weight.name)).size !== weights.length) throw new Error('Checkpoint has missing/duplicate weights.');
  const get = (name, shape) => {
    const weight = weights.find(item => item.name === name);
    if (!weight || weight.dtype !== 'float32' || !equalShape(weight.shape, shape) || !weight.data || weight.data.length !== shape.reduce((a, b) => a * b, 1) || Array.from(weight.data).some(value => !Number.isFinite(value))) throw new Error(`Invalid checkpoint weight ${name}; expected finite float32 [${shape}].`);
    return weight.data;
  };
  const grid = resolution / 8, anchors = grid * grid * 2, count = classes.length;
  const nodes = [], initializers = [];
  let features;
  if (architecture === 'yolo26n-frozen-head-v1') {
    if (!(backboneBytes instanceof Uint8Array)) throw new Error('Frozen-feature export requires verified pretrained ONNX bytes.');
    if (checkpoint.source?.sha256 !== BACKBONE.sha256 || checkpoint.source?.backboneId !== BACKBONE.id) throw new Error('Checkpoint pretrained source does not match the pinned backbone.');
    // The loader verifies the pinned SHA before this synchronous exporter runs.
    const source = readGraph(backboneBytes);
    if (namedField(source.inputs[0], 1) !== 'images' || namedField(source.outputs[0], 1) !== BACKBONE.output || source.nodes.length !== BACKBONE.nodeCount) throw new Error('Unexpected pretrained feature graph.');
    nodes.push(...source.nodes); initializers.push(...source.initializers);
    features = 'trainer/features_nhwc';
    nodes.push(node('Transpose', [BACKBONE.output], [features], {perm: [0, 2, 3, 1]}));
  } else {
    let previous = 'images', size = resolution, inChannels = 3;
    for (const [index, outChannels] of [8, 12, 16].entries()) {
      const layer = `conv${index + 1}`, kernel = get(`${layer}/kernel`, [3, 3, inChannels, outChannels]);
      const reordered = new Float32Array(kernel.length);
      // TensorFlow stores HWIO, while ONNX Conv consumes OIHW.
      for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) for (let input = 0; input < inChannels; input++) for (let output = 0; output < outChannels; output++) reordered[((output * inChannels + input) * 3 + y) * 3 + x] = kernel[((y * 3 + x) * inChannels + input) * outChannels + output];
      const kernelName = `trainer/${layer}/kernel`, biasName = `trainer/${layer}/bias`;
      initializers.push(tensor(kernelName, [outChannels, inChannels, 3, 3], reordered), tensor(biasName, [outChannels], get(`${layer}/bias`, [outChannels])));
      const pad = sameUpperPadding(size);
      const conv = `trainer/${layer}/conv`, activated = `trainer/${layer}/relu`;
      nodes.push(node('Conv', [previous, kernelName, biasName], [conv], {kernel_shape: [3, 3], strides: [2, 2], pads: [pad.before, pad.before, pad.after, pad.after]}), node('Relu', [conv], [activated]));
      previous = activated; size = pad.output; inChannels = outChannels;
    }
    features = 'trainer/features_nhwc';
    nodes.push(node('Transpose', [previous], [features], {perm: [0, 2, 3, 1]}));
  }
  const featureCount = architecture === 'tiny-grid-v1' ? 16 : 128;
  for (const [name, shape] of [['head/hidden/kernel', [featureCount, 32]], ['head/hidden/bias', [32]], ['head/output/kernel', [32, 2 * (5 + count)]], ['head/output/bias', [2 * (5 + count)]]]) initializers.push(tensor(`trainer/${name}`, shape, get(name, shape)));
  nodes.push(
    node('MatMul', [features, 'trainer/head/hidden/kernel'], ['trainer/hidden_linear']),
    node('Add', ['trainer/hidden_linear', 'trainer/head/hidden/bias'], ['trainer/hidden_biased']),
    node('Relu', ['trainer/hidden_biased'], ['trainer/hidden']),
    node('MatMul', ['trainer/hidden', 'trainer/head/output/kernel'], ['trainer/output_linear']),
    node('Add', ['trainer/output_linear', 'trainer/head/output/bias'], ['trainer/output_logits']),
  );
  initializers.push(tensor('trainer/flat_shape', [3], [1, anchors, 5 + count], 'int64'));
  nodes.push(node('Reshape', ['trainer/output_logits', 'trainer/flat_shape'], ['trainer/logits']));
  const slice = (name, start, end) => {
    for (const [suffix, value] of [['start', start], ['end', end], ['axis', 2]]) initializers.push(tensor(`${name}/${suffix}`, [1], [value], 'int64'));
    nodes.push(node('Slice', ['trainer/logits', `${name}/start`, `${name}/end`, `${name}/axis`], [name]));
  };
  slice('trainer/box_logits', 0, 4); slice('trainer/object_logits', 4, 5); slice('trainer/class_logits', 5, 5 + count);
  const offsets = new Float32Array(anchors * 4);
  for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) for (let slot = 0; slot < 2; slot++) {
    const offset = ((y * grid + x) * 2 + slot) * 4;
    offsets[offset] = x * 8; offsets[offset + 1] = y * 8;
  }
  initializers.push(tensor('trainer/box_scale', [1, 1, 4], [8, 8, resolution, resolution]), tensor('trainer/box_offset', [1, anchors, 4], offsets));
  nodes.push(
    node('Sigmoid', ['trainer/box_logits'], ['trainer/box_sigmoid']),
    node('Mul', ['trainer/box_sigmoid', 'trainer/box_scale'], ['trainer/box_scaled']),
    node('Add', ['trainer/box_scaled', 'trainer/box_offset'], ['trainer/boxes']),
    node('Sigmoid', ['trainer/object_logits'], ['trainer/objectness']),
    node('Softmax', ['trainer/class_logits'], ['trainer/class_probabilities'], {axis: 2}),
    node('Mul', ['trainer/class_probabilities', 'trainer/objectness'], ['trainer/class_scores']),
    node('Concat', ['trainer/boxes', 'trainer/class_scores'], ['trainer/predictions'], {axis: 2}),
    node('Transpose', ['trainer/predictions'], ['output0'], {perm: [0, 2, 1]}),
  );
  return model(graph({nodes, initializers, inputs: [valueInfo('images', [1, 3, resolution, resolution])], outputs: [valueInfo('output0', [1, 4 + count, anchors])]}), {
    task: 'detect', names: JSON.stringify(classes), imgsz: JSON.stringify([resolution, resolution]), end2end: 'false',
    architecture, training_scope: architecture === 'tiny-grid-v1' ? 'All tiny-grid-v1 parameters; not YOLO26 from scratch' : 'Frozen YOLO26n spatial features and trained new detection head; not full YOLO26 fine-tuning',
    trained_steps: Number.isInteger(checkpoint.training?.step) ? checkpoint.training.step : 0,
    backbone_sha256: architecture === 'tiny-grid-v1' ? 'none' : BACKBONE.sha256,
    preprocessing: 'RGB float32 /255; aspect-preserving letterbox filled with 114; stride8 two-slot grid',
    license: 'AGPL-3.0-only',
  });
}
