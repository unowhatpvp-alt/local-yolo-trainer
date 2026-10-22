// The subset of the official ONNX protobuf schema used by the Trainer.
// This writes standard ModelProto / GraphProto / TensorProto messages. It does
// not evaluate graph code or use a private file format renamed to .onnx.
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
export function concat(parts) {
  const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.length; }
  return result;
}
export function varint(value) {
  let integer = BigInt(value);
  if (integer < 0n) integer = BigInt.asUintN(64, integer);
  const result = [];
  do { const byte = Number(integer & 127n); integer >>= 7n; result.push(byte | (integer ? 128 : 0)); } while (integer);
  return new Uint8Array(result);
}
export const integer = (number, value) => concat([varint(number * 8), varint(value)]);
export function bytes(number, value) {
  const data = typeof value === 'string' ? textEncoder.encode(value) : value;
  return concat([varint(number * 8 + 2), varint(data.length), data]);
}
export const message = (number, fields) => bytes(number, concat(fields));
export const string = bytes;

export function fields(input) {
  const data = input instanceof Uint8Array ? input : new Uint8Array(input);
  let cursor = 0;
  const read = () => {
    let result = 0n, shift = 0n;
    for (let i = 0; i < 10; i++) {
      if (cursor >= data.length) throw new Error('Truncated ONNX protobuf integer.');
      const byte = data[cursor++]; result |= BigInt(byte & 127) << shift;
      if (!(byte & 128)) return result;
      shift += 7n;
    }
    throw new Error('Invalid ONNX protobuf integer.');
  };
  const result = [];
  while (cursor < data.length) {
    const start = cursor, tag = Number(read()), number = tag >>> 3, wire = tag & 7;
    if (!number) throw new Error('Invalid ONNX field number.');
    let value;
    if (wire === 0) value = read();
    else if (wire === 2) {
      const length = Number(read()), end = cursor + length;
      if (!Number.isSafeInteger(end) || end > data.length) throw new Error('Truncated ONNX field.');
      value = data.subarray(cursor, end); cursor = end;
    } else if (wire === 1 || wire === 5) {
      const end = cursor + (wire === 1 ? 8 : 4);
      if (end > data.length) throw new Error('Truncated ONNX fixed field.');
      value = data.subarray(cursor, end); cursor = end;
    } else throw new Error(`Unsupported ONNX wire type ${wire}.`);
    result.push({ number, wire, value, raw: data.subarray(start, cursor) });
  }
  return result;
}
export const decodeString = value => textDecoder.decode(value);
export const namedField = (data, number) => decodeString(fields(data).find(field => field.number === number)?.value || new Uint8Array());
export function attribute(name, value) {
  if (Array.isArray(value)) return concat([string(1, name), ...value.map(x => integer(8, x)), integer(20, 7)]);
  if (typeof value === 'number') return concat([string(1, name), integer(3, value), integer(20, 2)]);
  return concat([string(1, name), string(4, value), integer(20, 3)]);
}
export function node(op, inputs, outputs, attributes = {}, name = outputs[0]) {
  return concat([
    ...inputs.map(value => string(1, value)), ...outputs.map(value => string(2, value)),
    string(3, name), string(4, op),
    ...Object.entries(attributes).map(([key, value]) => bytes(5, attribute(key, value))),
  ]);
}
export function valueInfo(name, dims, elemType = 1) {
  const shape = dims.map(value => message(1, [typeof value === 'string' ? string(2, value) : integer(1, value)]));
  return concat([string(1, name), message(2, [message(1, [integer(1, elemType), message(2, shape)])])]);
}
export function tensor(name, dims, values, type = 'float32') {
  const count = dims.reduce((product, dimension) => product * dimension, 1);
  if (values.length !== count) throw new Error(`Tensor ${name} length ${values.length} does not match [${dims}].`);
  const raw = new Uint8Array(count * (type === 'int64' ? 8 : 4)), view = new DataView(raw.buffer);
  for (let i = 0; i < count; i++) {
    if (type === 'int64') view.setBigInt64(i * 8, BigInt(values[i]), true);
    else view.setFloat32(i * 4, values[i], true);
  }
  return concat([...dims.map(value => integer(1, value)), integer(2, type === 'int64' ? 7 : 1), string(8, name), bytes(9, raw)]);
}
export function graph({nodes, initializers, inputs, outputs, name = 'Local YOLO Trainer detector'}) {
  return concat([...nodes.map(value => bytes(1, value)), string(2, name), ...initializers.map(value => bytes(5, value)), ...inputs.map(value => bytes(11, value)), ...outputs.map(value => bytes(12, value))]);
}
export function model(graphBytes, metadata = {}, opset = 12) {
  return concat([
    integer(1, 8), string(2, 'Local YOLO Trainer Minimal'), string(3, '1.0.0'),
    integer(5, 1), string(6, 'Browser-trained spatial object detector; see metadata for exact architecture and scope.'),
    bytes(7, graphBytes), message(8, [string(1, ''), integer(2, opset)]),
    ...Object.entries(metadata).map(([key, value]) => message(14, [string(1, key), string(2, String(value))])),
  ]);
}
export function readGraph(modelBytes) {
  const graphField = fields(modelBytes).find(field => field.number === 7);
  if (!graphField) throw new Error('ONNX graph is missing.');
  const graphFields = fields(graphField.value);
  return {
    nodes: graphFields.filter(field => field.number === 1).map(field => field.value),
    initializers: graphFields.filter(field => field.number === 5).map(field => field.value),
    inputs: graphFields.filter(field => field.number === 11).map(field => field.value),
    outputs: graphFields.filter(field => field.number === 12).map(field => field.value),
  };
}
