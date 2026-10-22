import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';

// ZIP64 advertises a sentinel compressed length without its required extra
// field. Vulnerable parsers loop before the application's member filter runs.
function malformedZip64() {
  const name = new TextEncoder().encode('manifest.json');
  const data = new TextEncoder().encode('{}');
  const central = 30 + name.length + data.length;
  const centralSize = 46 + name.length;
  const zip64 = central + centralSize;
  const bytes = new Uint8Array(zip64 + 56 + 20 + 22);
  const view = new DataView(bytes.buffer);
  const u16 = (offset, value) => view.setUint16(offset, value, true);
  const u32 = (offset, value) => view.setUint32(offset, value, true);
  const u64 = (offset, value) => view.setBigUint64(offset, BigInt(value), true);
  u32(0, 0x04034b50); u16(4, 45); u32(18, 2); u32(22, 2); u16(26, name.length);
  bytes.set(name, 30); bytes.set(data, 30 + name.length);
  u32(central, 0x02014b50); u16(central + 4, 45); u16(central + 6, 45);
  u32(central + 20, 0xffffffff); u32(central + 24, 2); u16(central + 28, name.length);
  bytes.set(name, central + 46);
  u32(zip64, 0x06064b50); u64(zip64 + 4, 44); u16(zip64 + 12, 45); u16(zip64 + 14, 45);
  u64(zip64 + 24, 1); u64(zip64 + 32, 1); u64(zip64 + 40, centralSize); u64(zip64 + 48, central);
  const locator = zip64 + 56;
  u32(locator, 0x07064b50); u64(locator + 8, zip64); u32(locator + 16, 1);
  const end = locator + 20;
  u32(end, 0x06054b50); u16(end + 8, 1); u16(end + 10, 1);
  u32(end + 12, centralSize); u32(end + 16, central);
  return bytes;
}

test('project and checkpoint imports reject malformed ZIP64 without hanging', () => {
  const source = new URL('../src/data/formats.js', import.meta.url).href;
  const payload = Buffer.from(malformedZip64()).toString('base64');
  // A process timeout is necessary: a synchronous parser loop cannot be
  // interrupted by Promise.race or the test runner's JavaScript timer.
  const script = `import assert from 'node:assert/strict';
    import {unpackProject,unpackCheckpoint} from ${JSON.stringify(source)};
    const blob=new Blob([Buffer.from(${JSON.stringify(payload)},'base64')]);
    for(const unpack of [unpackProject,unpackCheckpoint]) {
      await assert.rejects(unpack(blob), error=>error.code===13 && /invalid zip data/i.test(error.message));
    }
    console.log('both imports rejected malformed ZIP64');`;
  const result = spawnSync(process.execPath, ['--input-type=module', '--eval', script], {encoding: 'utf8', timeout: 5000});
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /both imports rejected malformed ZIP64/);
});
