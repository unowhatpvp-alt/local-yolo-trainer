// Reproduce the small frozen spatial feature extractor from an immutable,
// genuine pretrained YOLO26n ONNX. Never downloads or processes user images.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {readGraph, fields, decodeString, namedField, graph, model, valueInfo, tensor, node} from '../src/model/onnx-proto.js';
import {BACKBONE} from '../src/model/backbone.js';

const sourceUrl = 'https://huggingface.co/unileon-robotics/YOLO26-ONNX/resolve/e046fc5125ee6b579f1801f0b6df68a60b949c42/yolo26n.onnx';
const sourceSha256 = '4f44c33cea12d4b5785e5e7c0a20c13d0c62a77de607ce96a40d6036c7d7bbcf';
const feature = '/trainer/neck128';
const selectedFeatures = ['/model.16/cv1/act/Mul_output_0', '/model.16/cv2/act/Mul_output_0'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
let source;
const localPath = process.argv[2];
if(localPath)source=new Uint8Array(await readFile(localPath));
else {
  const response = await fetch(sourceUrl, {signal: AbortSignal.timeout(120000), credentials: 'omit'});
  if (!response.ok) throw new Error(`Pinned model download failed: HTTP ${response.status}`);
  source = new Uint8Array(await response.arrayBuffer());
}
if (source.length !== 9887916 || hash(source) !== sourceSha256) throw new Error('Pretrained source size/SHA-256 mismatch.');
const original = readGraph(source), needed = new Set(selectedFeatures), nodes = [];
for (let index = original.nodes.length - 1; index >= 0; index--) {
  const candidate = original.nodes[index], properties = fields(candidate);
  const outputs = properties.filter(field => field.number === 2).map(field => decodeString(field.value));
  if (!outputs.some(output => needed.has(output))) continue;
  nodes.unshift(candidate);
  for (const input of properties.filter(field => field.number === 1)) needed.add(decodeString(input.value));
}
if (nodes.length !== 214 || !needed.has('images')) throw new Error('Pinned neck graph does not match the expected image dependencies.');
const geometryChanges = [];
const initializers = original.initializers.filter(value => needed.has(namedField(value, 8))).map(value => {
  const name = namedField(value, 8);
  if (name === '/model.10/m/m.0/attn/Constant_output_0') {
    geometryChanges.push({name, original: [1, 2, 128, 300], resized: [1, 2, 128, 16]});
    return tensor(name, [4], [1, 2, 128, 16], 'int64');
  }
  if (name === '/model.10/m/m.0/attn/Constant_3_output_0') {
    geometryChanges.push({name, original: [1, 128, 15, 20], resized: [1, 128, 4, 4]});
    return tensor(name, [4], [1, 128, 4, 4], 'int64');
  }
  return value;
});
if (geometryChanges.length !== 2) throw new Error('The expected attention reshape constants were not found.');
nodes.push(node('Concat', selectedFeatures, [feature], {axis: 1}));
// Preserve the exact candidate bytes whose numerical and real-photo results
// selected this extractor. Its historical diagnostic-purpose string remains
// untouched; current scope/source/geometry are documented in manifest.json.
const result = model(graph({nodes, initializers, inputs: [valueInfo('images', [1, 3, 128, 128])], outputs: [valueInfo(feature, ['B', 'C', 'H', 'W'])]}), {purpose: 'Work-directory feasibility probe only. No released artifact changed.'});
if (result.length !== BACKBONE.bytes || hash(result) !== BACKBONE.sha256) throw new Error('Rebuilt neck feature artifact differs from the pinned verified candidate.');
const directory = new URL('../public/models/', import.meta.url);
await mkdir(directory, {recursive: true});
const manifest = {
  schemaVersion: 1, id: BACKBONE.id, filename: 'yolo26n-features-128.onnx',
  bytes: result.length, sha256: hash(result), nodeCount: nodes.length, initializerCount: initializers.length,
  input: 'images', inputShape: [1, 3, 128, 128], output: feature, outputShape: [1, 128, 16, 16],
  sourceUrl, sourceSha256, sourceBytes: source.length,
  sourceRevision: 'e046fc5125ee6b579f1801f0b6df68a60b949c42', license: 'AGPL-3.0', opset: 12,
  selectedFeatures, geometryChanges,
  scope: 'Frozen pretrained YOLO26n neck features concatenated to 128 channels. The Trainer learns a new spatial detection head, not the full YOLO26 network.',
  priorBackboneRejected: {id: 'yolo26n-features-stride8-v1', sha256: 'd07f0d496799d77361cb621a67c86b3ceaa856eb0007d1a1c8deb1383b14a897'},
};
await writeFile(new URL('yolo26n-features-128.onnx', directory), result);
await writeFile(new URL('manifest.json', directory), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
