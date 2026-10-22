// Compare an actual JSON download from the unchanged inference website with
// the Trainer's recorded prediction for exactly the same image/model.
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const [expectedPath,actualPath,modelPath,reportPath]=process.argv.slice(2);
if(!reportPath)throw new Error('Usage: node scripts/compare-inference-result.mjs expected.json actual.json model.onnx report.json');
const expected=JSON.parse(await readFile(expectedPath)),actual=JSON.parse(await readFile(actualPath));
const model=await readFile(modelPath),modelSha256=createHash('sha256').update(model).digest('hex');
assert.equal(modelSha256,expected.onnx.sha256,'Compare the exact recorded model');
assert.equal(createHash('sha256').update(await readFile(expected.input.relativePath)).digest('hex'),expected.input.sha256,'Compare the exact recorded source picture');
assert.equal(actual.task,'detect');assert.equal(actual.sourceWidth,expected.input.width);assert.equal(actual.sourceHeight,expected.input.height);
assert.equal(actual.items.length,expected.expectedItems.length);
let maximumCoordinateError=0,maximumScoreError=0;
for(let i=0;i<actual.items.length;i++){
 const a=actual.items[i],b=expected.expectedItems[i];assert.equal(a.classId,b.classId);assert.equal(a.label,b.label);
 maximumScoreError=Math.max(maximumScoreError,Math.abs(a.score-b.score));
 for(const key of ['x','y','width','height'])maximumCoordinateError=Math.max(maximumCoordinateError,Math.abs(a.box[key]-b.box[key]));
}
assert.ok(maximumCoordinateError<.001,'Original-image coordinates must agree within .001 pixels');
assert.ok(maximumScoreError<.00001,'Confidence scores must agree within .00001');
const report={schemaVersion:1,status:'pass',checkedAt:new Date().toISOString(),website:'https://unowhatpvp-alt.github.io/local-yolo/',inferenceSourceCommit:'435a299fbe60c8e0fe4570c9e0cc8f944f37ecbc',modelSha256,input:expected.input,preprocessingVersion:expected.preprocessingVersion,confidence:expected.confidence,nmsIou:expected.nmsIou,items:actual.items.length,maximumCoordinateError,maximumScoreError,coordinateTolerance:.001,scoreTolerance:.00001,actualBackend:actual.backend,latencyMs:actual.latency,scope:'Real manual browser import of the exported ONNX, original image inference and JSON download. Numerical compatibility only; not a new independent accuracy evaluation.'};
await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
