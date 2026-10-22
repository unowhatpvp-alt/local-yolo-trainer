export const BACKBONE_ASSET_PATH = 'models/yolo26n-features-128.onnx';
export const BACKBONE_ID = 'yolo26n-neck128-stride8-v2';
export const BACKBONE_SHA256 = 'a2d8f60b3f6a3f39fc1b4d2f0791c40f6e8a3497a7e01a940c3f6a68cbe57b99';
export const BACKBONE = Object.freeze({
  id: BACKBONE_ID, sha256: BACKBONE_SHA256, bytes: 6100763, nodeCount: 215, initializerCount: 121,
  output: '/trainer/neck128', input: 'images',
  inputShape: [1, 3, 128, 128], outputShape: [1, 128, 16, 16],
  sourceSha256: '4f44c33cea12d4b5785e5e7c0a20c13d0c62a77de607ce96a40d6036c7d7bbcf',
});
export async function verifyBackbone(input) {
  const value = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (value.byteLength !== BACKBONE.bytes) throw new Error('Pretrained feature file size mismatch.');
  const digest = await crypto.subtle.digest('SHA-256', value);
  const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  if (hex !== BACKBONE.sha256) throw new Error('Pretrained feature SHA-256 mismatch.');
  return value;
}
export async function loadBackbone({baseUrl, signal} = {}) {
  const context = typeof location !== 'undefined' ? location.href : undefined;
  const base = baseUrl ? new URL(baseUrl, context).href : (context ? new URL(import.meta.env?.BASE_URL || './', context).href : null);
  if (!base) throw new Error('A same-origin base URL is required to load the pretrained feature asset.');
  const url = new URL(BACKBONE_ASSET_PATH, base);
  if (typeof location !== 'undefined' && url.origin !== location.origin) throw new Error('Pretrained feature assets must be served from the application origin.');
  const response = await fetch(url, {signal, credentials: 'same-origin'});
  if (!response.ok) throw new Error(`Pretrained feature download failed: HTTP ${response.status}`);
  return verifyBackbone(new Uint8Array(await response.arrayBuffer()));
}
