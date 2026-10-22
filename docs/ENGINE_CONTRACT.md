# Trainer engine and persistence contract

This browser application is independent of the deployed inference repository. Its source repository is [unowhatpvp-alt/local-yolo-trainer](https://github.com/unowhatpvp-alt/local-yolo-trainer). This document specifies interfaces; it is not itself evidence that a device, backend or release passed acceptance. Only the website is in scope; there is no Python service or desktop application.

## Supported architectures

- `yolo26n-frozen-head-v1`: real immutable YOLO26n neck features, frozen; a genuinely trained new spatial detection head. This is explicitly **not** full YOLO26 fine-tuning. The current backbone is `yolo26n-neck128-stride8-v2`, SHA-256 `a2d8f60b3f6a3f39fc1b4d2f0791c40f6e8a3497a7e01a940c3f6a68cbe57b99`; it concatenates the native model.16 cv1/cv2 feature maps to 128 channels. Earlier shallow-feature checkpoints are incompatible and rejected. [Exact provenance](MODEL_PROVENANCE.md).
- `tiny-grid-v1`: small randomly initialized convolutional detector; all its parameters are trained. This is explicitly **not** YOLO26 from scratch.
- Import/continue only our versioned, validated checkpoint. An inference ONNX is not a training checkpoint.
- Full YOLO26 end-to-end/from-scratch is unsupported unless separate actual proof becomes available.

## Common model and tensor specification

Only resolution **128 × 128** is enabled. RGB input is divided by 255 after exact 114-filled aspect-preserving letterbox. The versioned production Blob preprocessing is **`preprocessingVersion='canvas-high-v1'`**: orient the image, round resized dimensions, floor the leading padding, enable Canvas image smoothing and set `imageSmoothingQuality='high'`. This matches the existing inference tool. Checkpoints without this version, including earlier low-smoothing development checkpoints, are rejected rather than silently changing their input distribution. The typed-pixel path is restricted to already canonical 128 × 128 developer fixtures. It rejects source pixels requiring resizing, so Node nearest-neighbour resampling cannot be mistaken for browser Blob acceptance.

Batch 1 is the default. Batch 2/4/8 must pass a real forward/loss/gradient/update probe for the requested architecture/backend; auto conservatively resolves to 1. Frozen ONNX feature extraction runs one image at a time in WASM, and the head can train on a batch of these features. Higher resolutions, full YOLO26 fine-tuning and unverified backend combinations are not enabled by an API-presence check.

Grid stride is 8, grid=resolution/8, with TWO slots per spatial cell. A third same-cell target is a validation error, never silently dropped. The frozen feature session emits NCHW `[1,128,16,16]`, converted to NHWC for the head. Scratch has three kernel3/stride2/SAME_UPPER Conv+ReLU layers with channels 8,12,16; scratch features are NHWC `[B,G,G,16]`. Class IDs are zero-based and contiguous, with 1–32 nonempty names of at most 80 characters.

Shared detection head: per-cell MatMul(F,32)+bias+ReLU, MatMul(32,2*(5+C))+bias. Engine raw logits `[B,G,G,2,5+C]`: tx,ty,tw,th,objectness,class logits. Decode cx=(sigmoid(tx)+gx)*8, cy analogous; width/height=sigmoid(tw/th)*resolution. Class scores=sigmoid(objectness)*softmax(classes). One class softmax is 1.

The enabled objective is **`lossVersion='focal-kl-v1'`**: focal binary cross-entropy for objectness, bounded box-parameter KL divergence and class cross-entropy. GPU-sensitive elementwise operations are computed on semantically equivalent flattened tensors to avoid a verified rank-five backend arithmetic defect. The logical output/target layout above remains unchanged. Diagnostics from the rejected implementation are kept separately; capability probes execute the actual current detector graph.

Initial objectness bias -4; width/height bias -1.4; other head biases zero. Seeded initialization. Class IDs are contiguous indices. Source image boxes use **original oriented-image pixels** `{id,classId,x,y,width,height}`. All training augmentation must transform boxes correspondingly.

Checkpoint structure:
```
{schemaVersion:1, architecture, resolution, slots:2, classes:[string],
 weights:[{name,shape,dtype,data:Float32Array|Int32Array}],
 optimizer:{type:'adam',learningRate,weights:[{name,shape,dtype,data}],betaPowers:[number,number]},
 training:{epoch,batchOffset,step,rngState,datasetHash,history,config,
           epochOrder,epochLossSum,epochBatches,datasetRevision?},
 source:{backboneId,sha256}}
```
Stable variable names: scratch `conv1/kernel`, `conv1/bias` ... `conv3/*`; common `head/hidden/kernel`, `head/hidden/bias`, `head/output/kernel`, `head/output/bias`. Conv kernel HWIO; dense kernel [in,out]. Serialization must not detach tensors or overwrite a good checkpoint after errors.

Adam uses beta1=.9, beta2=.999 and epsilon=1e-7. Its ordered state is scalar int32 `iter` with shape `[]`, followed by every parameter's `/m` and `/v` float32 tensors. At step zero only `iter` exists. `iter` must equal `training.step`. The actual float32 beta accumulators are saved as `optimizer.betaPowers`; rebuilding them with a mathematical power can produce different rounding from uninterrupted training, so restoration assigns the exact saved values. `src/model/checkpoint-schema.js` validates shapes, state, configuration, preprocessing/loss versions and source without importing a training runtime. Archive validation additionally verifies binary offsets, payload SHA-256, typed arrays and size limits.

Worker commands `{id,type,payload}`: `probe`, `train`, `pause`, `stop`, `predict`, `export`, `dispose`. Result `{id,type:'result',result}`; error `{id,type:'error',error:{name,message,code}}`; progress `{type:'progress',runId,...}`; checkpoint `{type:'checkpoint',runId,checkpoint}`. Training accepts `{samples:[{id,blob,width,height,boxes,split}],classes,config,checkpoint?,additionalEpochs?}`. Predict accepts `{blob,checkpoint?,confidence:.25}`. Export returns valid ONNX bytes.

`train`, `probe`, `load`, `predict`, `export` and `dispose` share an exclusive-operation mutex; overlapping commands fail explicitly with `BUSY`. Pause/stop requests bypass that mutex and are observed between real updates. They do not dispose tensors during an active update. A hidden document requests pause; closing the browser does not continue training. Worker crashes reject pending requests and retain the last committed project/checkpoint.

Sample `split='val'` and `split='validation'` are equivalent; `train` or an omitted split is training. Other split values are rejected. Final test images stay outside a training request. Decoded duplicate content cannot cross train/validation. Dataset hashes exclude project/image/box UUIDs, so restoring a project under new IDs does not reset training. Changed pixels, boxes, classes or splits restart dataset progress and are reported explicitly.

For a completed run, the UI sends `additionalEpochs` rather than calculating a new total from stale project metadata. The worker checks the decoded dataset first, then sets total epochs to its current dataset epoch plus the requested additional amount, with a 1–1000 limit. A paused run retains its saved epoch/order/batch offset and original total target. Adam state and random state survive either continuation.

Config defaults are `{mode:'pretrained',resolution:128,batchSize:1,epochs:30,learningRate:.003,augment:true,seed:12345,backend:'auto',lossVersion:'focal-kl-v1',preprocessingVersion:'canvas-high-v1'}`. UI mode also accepts `scratch` or `checkpoint`; stored checkpoint config resolves mode to its actual architecture. Probe and record the real backend; unsupported combinations are disabled with reasons. Progress, timings, loss, gradients and weight hashes come from actual operations in a Worker.

Export function `exportOnnx(checkpoint,{backboneBytes?}) -> Uint8Array`; model proto has one NCHW float32 image input and one output `[1,4+C,G*G*2]`, cxcywh input-pixel coordinates and per-class scores, with no extra objectness channel. Metadata includes `task=detect`, JSON `names`, `imgsz=[resolution,resolution]`, `end2end=false` and accurate architecture/training scope. TF SAME_UPPER padding matches explicit ONNX pads. Frozen export includes the exact verified neck bytes' original weights; scratch kernels are reordered HWIO→OIHW. Successful packaging alone does not establish image accuracy; actual saved-model/browser/inference-tool results must be checked separately.

Project model for UI/storage: `{schemaVersion:1,id,title,mode:'single'|'multi',classes:[string],images:[{id,name,mime,width,height,sha256,boxes,reviewed,split}],config,createdAt,updatedAt}`; image blobs stored independently by image ID in IndexedDB. Controller supplies blobs to worker. No image bytes/local file paths are sent to external services.

Project timestamps are canonical UTC ISO strings. Invalid records are isolated from normal listing without being deleted. Photo removal atomically updates the project and deletes only that photo/thumbnail; saved models remain available. Confirmed project removal deletes its own records. Backup import validates original image hashes and creates new project/image IDs without overwriting an existing project. Save success is announced only after the actual IndexedDB transaction commits. Quota/private-mode failure keeps unsaved state available for retry or backup.

Annotation component: `createAnnotationEditor({root,onChange,onStatus?,readonly?})`; API `load({blob,width,height,boxes,classes,activeClassIndex})`, `setClasses`, `setActiveClass`, `undo`, `redo`, `zoomIn`, `zoomOut`, `fit`, `destroy`. Change callback receives boxes. All UI text bilingual; locale module exports `tr(zh,en)`, `language()`, `setLanguage(value)` and dispatches `languagechange`. Preferences only may use localStorage.
