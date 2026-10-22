# Model sources, training scope and export compatibility

The Trainer implements two explicitly named detector architectures. Neither is presented as complete YOLO26 training. A downloaded inference ONNX is not a training checkpoint.

| Architecture | Parameters updated by browser training | Inference export |
|---|---|---|
| `yolo26n-frozen-head-v1` | A new two-layer spatial detection head; the actual pretrained YOLO26n feature extractor remains frozen | Frozen feature graph with original pretrained tensors, followed by the learned head and standard decoded detection output |
| `tiny-grid-v1` | All three convolutional layers and the two-layer detection head, initialized from a seeded random distribution | The same learned convolution/head parameters, reordered for ONNX where necessary |

Full YOLO26 fine-tuning and full YOLO26 from-scratch training are unsupported. The default resolution is 128, stride is 8, and there are two detection slots per cell. A dataset with three objects assigned to the same cell is rejected instead of silently losing an annotation. Lower resolution and these small architectures limit practical accuracy; training success does not imply reliable generalization.

## Immutable pretrained source

- Upstream architecture/checkpoints: [Ultralytics YOLO26](https://huggingface.co/Ultralytics/YOLO26).
- Community ONNX export: [unileon-robotics/YOLO26-ONNX](https://huggingface.co/unileon-robotics/YOLO26-ONNX/tree/e046fc5125ee6b579f1801f0b6df68a60b949c42), immutable revision `e046fc5125ee6b579f1801f0b6df68a60b949c42`.
- Exact source: [`yolo26n.onnx`](https://huggingface.co/unileon-robotics/YOLO26-ONNX/resolve/e046fc5125ee6b579f1801f0b6df68a60b949c42/yolo26n.onnx), 9,887,916 bytes.
- Source SHA-256: `4f44c33cea12d4b5785e5e7c0a20c13d0c62a77de607ce96a40d6036c7d7bbcf`.
- Source/model license: AGPL-3.0. Application source is AGPL-3.0-only. Retain notices when redistributing the feature artifact or a detector containing it. Dependencies retain their individual licenses.

`scripts/prepare-backbone.mjs` verifies the complete original file by byte count and SHA-256, follows dependencies backward from both `/model.16/cv1/act/Mul_output_0` and `/model.16/cv2/act/Mul_output_0`, and retains 214 original nodes. A single channel-axis Concat joins these two 64-channel spatial feature maps. The resulting **215-node / 121-initializer** feature graph has a fixed `[1,3,128,128]` RGB float32 input and a runtime-verified `[1,128,16,16]` output named `/trainer/neck128`. Its symbolic ONNX output dimension labels are retained from the exact tested candidate; the manifest records actual measured dimensions.

All learned parameters remain unchanged. Two non-learned attention reshape initializers are adapted from the original 480×640 input geometry to 128×128: `[1,2,128,300]` becomes `[1,2,128,16]`, and `[1,128,15,20]` becomes `[1,128,4,4]`. The other 119 initializers and all original retained nodes are copied byte for byte. Without these geometry changes, actual ORT session creation failed; with them, real inference produced finite nonzero features. The removed parts are the original final detection head, unused later paths and stale shape annotations.

Bundled artifact: `public/models/yolo26n-features-128.onnx`, **6,100,763 bytes**, SHA-256 `a2d8f60b3f6a3f39fc1b4d2f0791c40f6e8a3497a7e01a940c3f6a68cbe57b99`, backbone ID **`yolo26n-neck128-stride8-v2`**. The generic asset filename remains the same; identity is established by the new ID and exact SHA. Its source, measured shapes, two geometry changes and integrity information are in `public/models/manifest.json`. The runtime loader verifies byte count and native SHA-256 before use, fetching the asset from the application's own origin. User images are not part of this download.

The exact candidate bytes selected by real-photo validation were adopted without rewriting its historical diagnostic-purpose metadata string. That string records the earlier experiment; current source, training scope and identity are specified here and in the manifest. Rebuilding is deterministic and must match the adopted candidate's complete SHA-256 before it is written.

An earlier 305,064-byte shallow `/model.4` feature prototype used ID `yolo26n-features-stride8-v1` and SHA `d07f0d496799d77361cb621a67c86b3ceaa856eb0007d1a1c8deb1383b14a897`. Real-photograph validation exposed excessive false positives, prompting the deeper spatial-feature comparison. Its results remain diagnostic evidence rather than being relabeled successful. Shared checkpoint validation and the ONNX exporter **reject that previous ID/hash**: the same head tensor shape does not make differently trained feature representations interchangeable.

To reproduce with an existing source file:

```sh
node scripts/prepare-backbone.mjs /absolute/path/to/yolo26n.onnx
```

Without the argument, the script first tries the previous inference checkout's ignored test fixture, then downloads only the immutable public model if that fixture is absent. This development preparation is not required of end users; the website bundles the verified feature artifact.

## Export format

`src/model/onnx-proto.js` contains a small standard protobuf encoder/parser limited to the messages needed here. The schema field numbers follow the [official ONNX protobuf schema](https://github.com/onnx/onnx/blob/v1.19.1/onnx/onnx.proto). It does not require Python, execute model metadata, or disguise a checkpoint as an ONNX file.

`exportOnnx(checkpoint, {backboneBytes})` writes a real self-contained FP32 ONNX model with a single RGB NCHW `[1,3,R,R]` input and a single `[1,4+C,(R/8)*(R/8)*2]` output. Only the actually validated resolution **R=128** is enabled; higher resolutions are rejected explicitly. Preprocessing is aspect-preserving letterbox filled with RGB114, followed by RGB/255. Decoded channels contain center-x/center-y/width/height in input pixels and objectness-times-class-probability scores. There is no extra objectness output channel. Metadata includes exact class names, `task=detect`, `end2end=false`, input shape, actual architecture, declared training steps and honest training scope.

Scratch convolution kernels are transposed from TensorFlow HWIO to ONNX OIHW. Stride-two padding uses the exact [SAME_UPPER rule](https://onnx.ai/onnx/operators/onnx__Conv.html): even-sized inputs with a 3×3 kernel receive zero leading and one trailing pixel, rather than symmetric padding. Dense weights remain `[input,output]` for ONNX MatMul. Slot order is row, column, slot. Softmax is applied only to class logits on their final axis, followed by multiplication with sigmoid objectness; this is compatible with [opset-12 Softmax semantics](https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax-11).

The existing Local YOLO inference application's local-ONNX importer accepts this declared decoded layout. Actual end-to-end use of trained exported models must be verified separately; matching a signature alone does not prove accuracy or training success.

## What the automated ONNX tests prove

`node --test tests/onnx-export.test.mjs` executes the actual bundled feature graph and actual exported detectors in ONNX Runtime Web/WASM. It compares exported predictions against an independent TensorFlow.js CPU forward path, including single-class and multiclass heads. The absolute tolerance is 0.0002 in input-pixel coordinates/score units; assertions are not skipped or replaced by simulated runtime results.

These tests deliberately use deterministic **untrained numerical fixtures** to isolate export arithmetic, tensor ordering, SAME_UPPER padding and metadata. They are not evidence of successful optimization or accuracy. The separate real training tests and browser acceptance evidence must demonstrate nonzero gradients, updated weights, held-out evaluation, checkpoint resume and saved-model inference before the product is accepted.

Two further export regressions execute four actual Adam updates for each architecture and compare the saved learned weights' TensorFlow predictions with the exported ORT/WASM result. They prove learned-parameter serialization. Their generated rectangles remain synthetic diagnostics, not a real-world accuracy claim.

`src/model/checkpoint-schema.js` validates the same architecture, exact weight shapes, ordered Adam moments, scalar int32 iteration, exact saved beta powers, training configuration, dataset state and pinned backbone source for both the worker and project archive path. It imports no TensorFlow or ONNX runtime, so project validation does not initialize a main-thread training engine. Archive validation adds typed-array and payload-size limits. The exported ONNX remains an inference file; it cannot be used to resume optimizer state.

## Independent real-photograph validation fixtures

`scripts/prepare-real-fixtures.mjs` downloads the original [Penn-Fudan pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/), also used by the [official TorchVision detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). The complete original ZIP is pinned at 53,723,336 bytes and SHA-256 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`. The source page names Liming Wang, Jianbo Shi, Gang Song and I-fan Shen, *Object Detection Combining Recognition and Segmentation*, ACCV 2007. It does not state an explicit SPDX license; the preparation records that absence rather than inventing a permissive license. Photos/masks/archive stay in ignored local test fixtures and are not redistributed in the production site.

The fixed diagnostic subset contains 32 training photos / 81 labeled people and eight held-out photos / 21 labeled people. Every box is independently derived from the original instance mask using inclusive foreground pixels and an exclusive right/bottom edge. IDs and SHA-256 digests of each original photo/mask, all boxes, source attribution, selection rule and limitations are saved in `tests/fixtures/penn-fudan/manifest.json`. All 170 source images were checked against the 128-pixel/two-slot assignment limit: none were excluded. No original photo SHA is shared by train and validation.

The split is fixed before inference: Fudan IDs 1–48 and Penn IDs 1–64 are training candidates; later numbered images are validation candidates. Each pool is ranked by `SHA256("penn-fudan-trainer-fixture-v1:" + imageID)`, taking 32 and eight respectively. Selection does not depend on model predictions. The eight validation images can inform architecture/hyperparameter choice and must therefore not be described as an untouched final test.

Before deeper-feature model selection, a separate **eight-photo / 20-person final test set** was frozen at `2026-10-21T23:38:37.778Z`, taking the next eight unused images in the original validation pool's SHA ranking. It lives in the manifest's separate `samplesTest` array and is never passed to training; `loadFinalTestFixtureSamples()` is a distinct reader for final prediction/evaluation only. Its IDs are PennPed00096, FudanPed00049, PennPed00089, PennPed00094, FudanPed00065, FudanPed00051, FudanPed00063 and FudanPed00052. All 48 selected photos have distinct original-file SHA digests. The manifest captures the freeze time and every image/mask digest.

Number blocks and exact image hashes establish image separation; the metadata does not establish independent scenes or person identities. Absence of overlap with the YOLO26n model's original pretraining cannot be proved. This is a small engineering diagnostic and not a published accuracy benchmark. Real training results, including failures and final-test performance, must be reported separately.

```sh
node scripts/prepare-real-fixtures.mjs
```

The developer-only script uses project Node dependencies and a restricted lossless PNG decoder, not Python or an installed training service. It validates every ZIP member path before writing and bounds compressed/uncompressed size. `loadRealFixtureSamples()` supplies the original RGB photographs and mask-derived boxes to Node training tests. Browser acceptance uses the same original PNGs through ordinary Blob image decoding.
