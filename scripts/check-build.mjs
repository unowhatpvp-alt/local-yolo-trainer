// Static artifact checks complement actual browser/training acceptance. They
// do not claim that a correctly packaged model predicts accurately.
import {readFile, readdir, lstat, mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {isBuiltin} from 'node:module';
import {parseAst} from 'rollup/parseAst';
import {BACKBONE, BACKBONE_ASSET_PATH} from '../src/model/backbone.js';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export const RUNTIME_FILES = [
  ['runtime/ort/ort-wasm-simd-threaded.mjs', 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs'],
  ['runtime/ort/ort-wasm-simd-threaded.wasm', 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'],
  ...['tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm', 'tfjs-backend-wasm-threaded-simd.wasm'].map(name => [`runtime/tfjs/${name}`, `node_modules/@tensorflow/tfjs-backend-wasm/dist/${name}`]),
  ...['ONNX_RUNTIME_LICENSE.txt', 'ONNX_RUNTIME_THIRD_PARTY_NOTICES.txt', 'TENSORFLOW_JS_LICENSE.txt', 'FFLATE_LICENSE.txt'].map(name => [`runtime/${name}`, `third_party/${name}`]),
];

function normalizedBase(base) {
  if (typeof base !== 'string' || !base.startsWith('/') || !base.endsWith('/') || base.includes('\\') || /[?#]/.test(base) || base.split('/').slice(1, -1).some(part => !part || part === '.' || part === '..')) throw new Error('Base must be an absolute static pathname with a trailing slash, for example /local-yolo-trainer/.');
  return base;
}
export async function checkBuild({directory = path.join(projectRoot, 'dist'), sourceRoot = projectRoot, base = process.env.TRAINER_BASE || '/local-yolo-trainer/'} = {}) {
  const report = {schemaVersion: 1, startedAt: new Date().toISOString(), base, status: 'fail', checks: [], errors: [], files: []};
  let root;
  const errors = report.errors, buffers = new Map();
  const fail = message => errors.push(message);
  try { base = normalizedBase(base); root = path.resolve(directory); }
  catch (error) { fail(error.message); return report; }
  const walk = async relative => {
    const absolute = path.join(root, relative);
    const entries = await readdir(absolute, {withFileTypes: true});
    for (const entry of entries) {
      const filename = relative ? `${relative}/${entry.name}` : entry.name;
      if (entry.isSymbolicLink()) { fail(`Symlink is not allowed in a static artifact: ${filename}`); continue; }
      if (entry.isDirectory()) await walk(filename);
      else if (entry.isFile()) {
        const value = await readFile(path.join(root, filename));
        buffers.set(filename, value); report.files.push({path: filename, bytes: value.length, sha256: digest(value)});
      }
    }
  };
  try { if (!(await lstat(root)).isDirectory()) throw new Error('not a directory'); await walk(''); }
  catch (error) { fail(`Cannot read build directory: ${error.message}`); return report; }
  report.files.sort((a, b) => a.path.localeCompare(b.path));
  if (!buffers.has('index.html')) fail('Missing production index.html.');
  if (!buffers.has('.nojekyll')) fail('Missing .nojekyll for GitHub Pages.');
  const developmentFiles = [...buffers.keys()].filter(name => /^(?:src|scripts|tests|test-results|node_modules|docs)\//.test(name) || /\.(?:lyproj|lymodel|log)$/.test(name));
  if (developmentFiles.length) fail(`Development fixtures/reports entered the production artifact: ${developmentFiles.join(', ')}`);
  report.checks.push({name: 'production-files', files: buffers.size, developmentFiles});

  let manifest;
  try {
    manifest = JSON.parse(buffers.get('models/manifest.json')?.toString('utf8') || 'null');
    const model = buffers.get(BACKBONE_ASSET_PATH);
    if (!model || model.length !== BACKBONE.bytes || digest(model) !== BACKBONE.sha256) fail('Bundled feature model does not match the pinned byte count/SHA-256.');
    if (manifest?.id !== BACKBONE.id || manifest?.sha256 !== BACKBONE.sha256 || manifest?.bytes !== BACKBONE.bytes || `models/${manifest?.filename}` !== BACKBONE_ASSET_PATH) fail('Feature manifest does not match the compiled model identity/path.');
    const unexpected = [...buffers.keys()].filter(name => name.endsWith('.onnx') && name !== BACKBONE_ASSET_PATH);
    if (unexpected.length) fail(`Unexpected unverified models in production: ${unexpected.join(', ')}`);
  } catch (error) { fail(`Invalid feature manifest: ${error.message}`); }
  report.checks.push({name: 'pinned-model', id: BACKBONE.id, path: BACKBONE_ASSET_PATH, sha256: BACKBONE.sha256});

  const wasmHashes = new Set();
  for (const [destination, source] of RUNTIME_FILES) {
    const value = buffers.get(destination);
    if (!value?.length) { fail(`Missing or empty runtime/notice: ${destination}`); continue; }
    try { const expected = digest(await readFile(path.join(sourceRoot, source))); if (destination.endsWith('.wasm')) wasmHashes.add(expected); if (digest(value) !== expected) fail(`Runtime/notice differs from its installed source: ${destination}`); }
    catch { fail(`Runtime source is unavailable for verification: ${source}`); }
  }
  for (const [filename, value] of buffers) if (filename.endsWith('.wasm')) {
    if (value.length < 8 || !value.subarray(0, 8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) fail(`Invalid WASM module header: ${filename}`);
    if (!wasmHashes.has(digest(value))) fail(`Unrecognized/changed runtime WASM bytes: ${filename}`);
  }
  report.checks.push({name: 'same-origin-runtime-and-notices', requiredFiles: RUNTIME_FILES.map(([filename]) => filename), wasmModules: [...buffers.keys()].filter(name => name.endsWith('.wasm')).length});

  let references = 0, localRuntimeDependencies = 0;
  const reference = (raw, from) => {
    if (!raw || /^(?:data:|blob:|#)/.test(raw)) return;
    references++;
    try {
      const origin = 'https://static-trainer.invalid', target = new URL(raw, `${origin}${base}${from}`);
      if (target.origin !== origin) throw new Error('runtime resource must be same-origin');
      if (!target.pathname.startsWith(base)) throw new Error(`resource escapes deployment base ${base}`);
      const relative = decodeURIComponent(target.pathname.slice(base.length)), resolved = path.resolve(root, relative), relation = path.relative(root, resolved);
      if (path.isAbsolute(relative) || relation.startsWith('..') || path.isAbsolute(relation)) throw new Error('resource leaves build directory');
      if (!buffers.has(relative)) throw new Error(`missing ${relative}`);
    } catch (error) { fail(`Invalid resource in ${from}: ${raw} (${error.message})`); }
  };
  const privateUrl = /https?:\/\/(?:localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|\[::1\])(?=[:/"'\s]|$)/i;
  const machinePath = /(?:\b[A-Za-z]:[\\/](?:Users|Windows|Program Files)[\\/]|file:\/\/(?:\/[A-Za-z]:|\/Users\/|\/home\/)|^\/(?:Users|home)\/)/;
  for (const [filename, value] of buffers) {
    if (!/\.(?:html|css|js|mjs|json)$/.test(filename)) continue;
    const text = value.toString('utf8');
    if (privateUrl.test(text)) { localRuntimeDependencies++; fail(`Hard-coded local/private network URL in production text: ${filename}`); }
    if (!/\.(?:js|mjs)$/.test(filename) && machinePath.test(text)) { localRuntimeDependencies++; fail(`Machine-specific filesystem path in production text: ${filename}`); }
    if (filename.endsWith('.html')) {
      if (/\/(?:@vite|src\/app\.js)/.test(text)) fail(`Development entry point in production HTML: ${filename}`);
      const tags = text.match(/<(?:script|link|img|source|video|audio|iframe)\b[^>]*>/gi) || [];
      for (const tag of tags) for (const match of tag.matchAll(/\b(?:src|href|poster)\s*=\s*(["'])(.*?)\1/gi)) reference(match[2], filename);
    } else if (/\.(?:js|mjs)$/.test(filename)) {
      // Reuse Vite's installed Rollup parser. Regex would mistake minified
      // switch-case strings or vendor code examples for actual imports.
      try {
        const pending = [parseAst(text)];
        while (pending.length) {
          const item = pending.pop();
          if (!item || typeof item !== 'object') continue;
          if (item.type === 'Literal' && typeof item.value === 'string' && item.value !== '/home/web_user' && machinePath.test(item.value)) { localRuntimeDependencies++; fail(`Machine-specific filesystem path in production JavaScript: ${filename}`); }
          if (['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration', 'ImportExpression'].includes(item.type) && typeof item.source?.value === 'string') {
            const specifier = item.source.value;
            if (/^(?:\.|\/)/.test(specifier)) reference(specifier, filename);
            else if (!(item.type === 'ImportExpression' && isBuiltin(specifier))) fail(`Unbundled import in ${filename}: ${specifier}`);
          }
          if (item.type === 'NewExpression' && item.callee?.name === 'URL' && typeof item.arguments?.[0]?.value === 'string' && item.arguments[1]?.object?.type === 'MetaProperty' && item.arguments[1]?.property?.name === 'url') reference(item.arguments[0].value, filename);
          for (const value of Object.values(item)) if (value && typeof value === 'object') { if (Array.isArray(value)) pending.push(...value); else pending.push(value); }
        }
      } catch (error) { fail(`Cannot parse production JavaScript ${filename}: ${error.message}`); }
    } else if (filename.endsWith('.css')) for (const match of text.matchAll(/url\(\s*(["']?)([^)'"\s]+)\1\s*\)/g)) reference(match[2], filename);
  }
  if (![...buffers.keys()].some(name => /(?:^|\/)worker[^/]*\.js$/.test(name))) fail('Missing emitted training Worker JavaScript.');
  report.checks.push({name: 'subpath-resources-and-portability', references, base, localRuntimeDependencies, note: 'Generic vendor file:// scheme handling, conditional dynamic Node builtins, Emscripten virtual /home/web_user and public provenance URLs are not host-machine runtime dependencies.'});
  report.completedAt = new Date().toISOString(); report.status = errors.length ? 'fail' : 'pass';
  report.totalBytes = report.files.reduce((sum, file) => sum + file.bytes, 0);
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), options = {};
  for (let index = 0; index < args.length; index++) {
    if (args[index] === '--dir') options.directory = path.resolve(args[++index]);
    else if (args[index] === '--base') options.base = args[++index];
    else throw new Error(`Unknown check-build option: ${args[index]}`);
  }
  const report = await checkBuild(options);
  await mkdir(path.join(projectRoot, 'test-results'), {recursive: true});
  await writeFile(path.join(projectRoot, 'test-results/build-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({status: report.status, base: report.base, files: report.files.length, bytes: report.totalBytes, checks: report.checks, errors: report.errors}, null, 2));
  if (report.status !== 'pass') process.exitCode = 1;
}
