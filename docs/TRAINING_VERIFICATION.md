# Training verification

This records actual local tests of the browser Trainer. It distinguishes successful execution, numerical correctness and photographic detection quality. It does not certify every browser/device or claim full YOLO26 training. UI and storage acceptance are covered in [UI verification](UI_VERIFICATION.md); tensor formats and checkpoint rules are specified in the [engine contract](ENGINE_CONTRACT.md).

## What the three training choices actually do

| UI choice | Actual implementation | Verified scope |
| --- | --- | --- |
| Pretrained features | Frozen real YOLO26n neck, `yolo26n-neck128-stride8-v2`, plus a new trainable spatial detection head | Actual loss, gradients and Adam updates; original photographic PNG Blob training; checkpoint reload; executed ONNX export |
| Small detector from scratch | Randomly initialized three-convolution `tiny-grid-v1`; all convolution and head weights train | Actual CPU and hardware GPU updates, data-dependent convolution weights, checkpoint reload and ONNX numerical parity; no natural-photo accuracy claim for this architecture |
| Continue a checkpoint | Restore one of the two architectures, every parameter, Adam moments/iteration/beta accumulators, sample order and RNG | CPU paused/resumed training equals uninterrupted training exactly; restored project UUIDs do not change the dataset identity; additional epochs are resolved after checking actual data |

Continuing a checkpoint is a third workflow, not a third architecture. An inference-only ONNX cannot resume training. The frozen-feature choice is **not full YOLO26 fine-tuning**, and the small scratch detector is **not YOLO26 from scratch**. Only 128 × 128 is enabled. Two target slots share each stride-eight cell; a third target produces an explicit error rather than disappearing from training.

## Executed backend matrix

| Architecture | CPU | WASM | WebGL | WebGPU |
| --- | --- | --- | --- | --- |
| Frozen YOLO26n features + trained head | Passed real forward/loss/gradients/update, training and reload | Passed; authoritative 100-epoch photograph run uses this backend | Passed actual six-step hardware training and reload | Passed actual six-step hardware training and reload |
| Small scratch detector | Passed full convolution gradients/update and data-dependent training | **Unsupported:** `Conv2DBackpropFilter` is not registered; explicitly disabled | Passed actual six-step hardware training and reload | Passed actual six-step hardware training and reload |
| Continue checkpoint | Uses the saved architecture; exact interrupted CPU continuation verified | Saved frozen-feature checkpoint reload verified | Six-step GPU-trained checkpoints reload with identical predictions | Six-step GPU-trained checkpoints reload with identical predictions |

Frozen ONNX feature extraction remains WASM even when the head trains on a GPU. GPU results came from an isolated Microsoft Edge 155.0.4283.62 headless profile on Windows, with a real **NVIDIA RTX 5080 / Blackwell** adapter, `isFallbackAdapter=false`, and the matching ANGLE renderer. They are not software-renderer results. [Complete acceleration evidence](evidence/browser-acceleration-training.json) records each probe, whole-gradient CPU comparison, actual backend, updated weight hashes and reload result.

The tests cover the concrete configurations recorded in their JSON. CPU scratch probes include batches 1/2/4/8. Hardware smoke training uses batch 2; the 100-epoch photograph reference uses WASM batch 4. This is not a claim that every backend/batch/class-count combination was tested on every device. Runtime selection probes the requested complete detector graph, including gradients and Adam, before enabling it. GPU checkpoint prediction reload does not establish bit-for-bit optimizer continuation across different backends.

## Regression tests and numerical correctness

`node --test tests/*.test.mjs` completed **53/53 tests, zero failures and zero skips** after the final engine/schema fixes. This is an observed local result, not a statement that an uninspected remote CI run passed. The suite includes:

- actual box learning on a new synthetic image, plus two tiny objects in one cell retaining distinct classes and noncollapsed boxes;
- all scratch convolution kernels changing in response to different source pixels;
- pause after two real updates and resume to nine total updates, matching uninterrupted weights, complete Adam state and RNG exactly;
- original-image letterbox/inverse coordinates, augmentation/targets, collision rejection and validation exclusion;
- `validation`/`val` compatibility, changed-data continuation, additional-epoch bounds and retained valid checkpoints after errors;
- rejected incomplete/corrupt checkpoints, including missing exact epoch state, incompatible backbone/loss/preprocessing and forged optimizer state;
- real overlapping probe/train/predict/export/dispose rejection, while pause remains available between updates;
- actual Node WASM assets and head updates, with the separate real scratch-gradient limitation retained;
- executed ONNX Runtime WASM against TensorFlow.js for both architectures, including learned weights and one/three-class numerical fixtures.

The 100-epoch browser run exported a real 6,129,877-byte ONNX, SHA-256 `a0984b956a69b41535bc458ff721db01e67eb6905a6e92a7123f613eca437e61`. Its 2,560 decoded output values matched the browser TensorFlow.js CPU reference with maximum absolute error **0.0000228882**, below the fixed **0.0002** tolerance. Saving through the application's checkpoint archive, unpacking it and reloading the Worker changed predictions by **0** and retained the exact Adam beta accumulators. These are separate gates from photograph accuracy.

The deployed inference tool also imported this exported model and ran the same original image. The separate [cross-tool round-trip record](evidence/inference-site-roundtrip.json) records the actual comparison: maximum coordinate difference **1.12e-5 original-image pixels**, maximum confidence difference **2.49e-8**. The inference repository was not modified for this test.

## Photographs, selection and the necessary preprocessing replay

The reference uses original [Penn-Fudan pedestrian photographs and instance masks](https://www.cis.upenn.edu/~jshi/ped_html/). Boxes are derived from the masks. The fixed split contains **32 training images / 81 people**, **8 validation images / 21 people**, and **8 final-test images / 20 people**. Photo hashes are checked before execution, with no duplicate image hashes across splits. Final-test photos are never passed to `train`. No source photos are redistributed in this repository's evidence archive.

The original eight-image final test was frozen before prediction. Validation, not final test, was used to compare the shallow feature baseline, focal loss changes and deeper feature candidates. The selected reference profile was then frozen: real neck features, focal/KL objective, 128 pixels, batch 4, 100 epochs, learning rate 0.003, augmentation enabled, seed 12345, confidence 0.25, NMS IoU 0.45 and matching IoU 0.5.

The first browser run used Canvas's default low-quality scaling. A real import into the existing inference site exposed a compatibility defect: that site explicitly uses `imageSmoothingQuality='high'`, so the same image/model produced materially different scores and boxes. Trainer now explicitly uses high-quality Canvas letterboxing and requires `preprocessingVersion='canvas-high-v1'`. Earlier low-scaling development checkpoints are rejected. Test-only typed pixels must already be canonical 128 × 128; Node nearest-neighbor resizing cannot impersonate browser preprocessing.

The corrected run repeated the **same WASM backend and frozen training profile**, followed by a necessary final-test replay. The original report is preserved intact. No final-test-based model, loss, threshold or epoch tuning took place. The replay is **not a second untouched independent experiment**.

| Run | Validation: TP / FP / FN | Final test: TP / FP / FN | Final precision / recall at IoU 0.5 | Mean matched IoU |
| --- | --- | --- | --- | --- |
| First browser run, low scaling; superseded for compatibility | 14 / 4 / 7 | 15 / 0 / 5 | 1.0000 / 0.7500 | 0.78775 |
| Corrected high-scaling engineering replay | 13 / 4 / 8 | **15 / 1 / 5** | **0.9375 / 0.7500** | 0.76568 |

The corrected run performed **800 real Adam updates**, loss **1.7214696 → 0.0940875**, initial gradient norm **2.1813015**, and changed the weight hash from `43645a32937d64bafbd6ec5c4ecaaf8684aa2a014eaecf606817c9107f402770` to `3957987e275d07416036c3fed321437415ac497326714a8ec342d45f054c8e55`. Engine elapsed time was 8,117.2 ms on this machine.

The [corrected full report](evidence/browser-real-training.json) contains every final-test prediction and ground-truth box. The [first full report](evidence/pre-canvas-high-v1/browser-real-training.json) is retained separately. These small development samples demonstrate genuine local learning and expose remaining misses/false positives; they are not mAP, a general-purpose pedestrian benchmark or proof that the pretrained backbone never saw similar imagery.

Earlier unsuccessful diagnostics remain in the [evidence index](evidence/training-evidence-index.json): the first shallow baseline had 14 TP, **267 FP**, 7 FN on validation; other shallow losses and deeper candidates are retained with their actual histories and predictions. They are historical Node pixel-path experiments, not the current canonical browser acceptance run. Their scripts, source-binding coverage and preprocessing differ; current commands are not promised to recreate every historical experiment byte for byte.

## GPU arithmetic defect and repair

Actual hardware testing caught a failure that an API check or “nonzero gradients” assertion would have missed. The original rank-five loss arithmetic returned CPU loss **498.210083**, but one real WebGPU execution returned finite, incorrect loss **0.75** and incorrect gradients while still changing parameters. The [original diagnostic](evidence/gpu-op-diagnostic.json) keeps both failures and successful reference calculations.

The production repair reshapes the same cells/channels into rank-two views before elementwise loss operations; it does not change the detector, loss formulas, targets or export layout. A CPU regression compares all **8,192 logit gradients** with the original rank-five objective: loss difference **0**, maximum gradient difference **0**. Current capability probes additionally compare the complete loss and every model-parameter gradient with an actual CPU reference; a finite but wrong result is rejected.

On the measured GPU, complete-gradient relative errors were below **6.6e-6**, compared with a fixed **0.002** limit. Four additional real Worker PNG-Blob runs—both architectures on WebGL and WebGPU—each performed six Adam steps, changed their weight hash and reloaded with identical predictions. These generated rectangles validate execution and persistence, not natural-photo quality or GPU speedup.

## Measured performance and its boundaries

The separate [performance run](evidence/browser-training-performance.json) used the same 32 training / 8 validation photos, WASM, batch 4 and 20 epochs. It did not load or evaluate final-test photos. Timing-only wrappers call the actual engine methods unchanged; source hashes match before and after.

| Measurement | Actual result | Scope |
| --- | --- | --- |
| Model initialization | 214.5 ms | Backend already probed; includes parameter/Adam creation and real ORT model verification/loading/session creation |
| 40-image preparation | 109.1 ms | Original PNG decode, Canvas high scaling, targets, pixel hashing and cooperative yields; file fetch occurred beforehand |
| Frozen feature extraction | 72 calls, 382.4 ms total | Actual cache misses, including training flips and validation |
| Completed epoch duration | Mean 88.685 ms; all 20 values retained | Measured checkpoint boundaries; includes updates, validation and checkpoint work |
| Training throughput | 297.38 image presentations/second | 640 training presentations divided by whole engine time, including preparation/validation; not distinct photographs/second |
| Main-thread 16 ms heartbeat | 134 samples; median 16 ms, P95 17 ms, max 17.7 ms | Actual timer during Worker training on this machine |
| Main-page JS heap | 14,861,482 → 15,093,938 bytes; maximum sampled 15,093,938 | Chromium `performance.memory`, precise-memory flag; main page only |

Worker/native/WASM total memory, GPU VRAM, whole-system RAM and cold browser/backend startup were **not measured**. The JS heap samples cannot substitute for any of those values. There is no claimed speedup against another engine, no physical-phone performance result and no universal latency guarantee.

## Source binding, archive integrity and reproduction

The [index](evidence/training-evidence-index.json) lists the exact byte length and SHA-256 for ten complete JSON reports. They are archived verbatim, with the relative `pre-canvas-high-v1/` directory retained. The archive contains JSON evidence only: no photographs, screenshots, checkpoints, trained model files or absolute local user paths. Hashes prove file integrity, not by themselves that a claim is correct.

The corrected photograph and GPU runs record source hashes before/after, zero changes during execution, and served-module/model/runtime response hashes. The measured photograph `engine.js` SHA is `768cb14235792d2089184ce94319b9878a7354eac1d9c9396b9697f3fdf4fa4f`; the repaired shared `model-ops.js` SHA is `f59c75126d8afcf70cf5d596c9203810578575f64464944ffc88ed963983bbe7`. Dataset manifest SHA is `928c548f4e88654962431d3220969d59155bfcbe591e6c7a12b6c8445dc0ea50`; neck asset SHA is `a2d8f60b3f6a3f39fc1b4d2f0791c40f6e8a3497a7e01a940c3f6a68cbe57b99`.

After those runs, three compatibility/validation changes were made: reject noncanonical typed-pixel test resizing, preserve Node's installed WASM asset lookup, and require complete epoch continuation fields. The old browser reports **do not claim to have executed that later source**. Their `subsequentCompatibilityOnlyChanges` fields identify the later files/hashes. The 53-test run and separately source-bound performance run cover the later source; the latter records `engine.js` SHA `7b2f3e7858856dea71882879b8201dd5561735007edd184a47564379641c6704`. The original low-scaling run has an explicitly labeled post-run hash capture, rather than pretending to have before/after binding.

From the repository root, using the Node version/dependencies recorded in the project:

```sh
npm ci
node scripts/prepare-assets.mjs
node --test tests/*.test.mjs
npm run build

# Download local developer fixtures; original photos stay outside distribution.
node scripts/prepare-real-fixtures.mjs

# On Windows with Microsoft Edge installed; isolated temporary browser profiles.
node tests/training-browser.mjs --edge --probe-only
node tests/training-browser.mjs --edge --acceleration-only
node tests/training-performance.mjs

# Explicit full engineering replay, including the already-used test8 evaluation.
# This is a reproduction action, not a fresh independent test set.
node tests/training-browser.mjs --edge --backend=wasm
```

GPU reproduction requires real compatible hardware; a missing GPU or unsupported kernel must remain reported as unsupported, not simulated. The default browser harness uses installed Playwright Chromium when `--edge` is omitted. Fixture setup requires network access to the original dataset. Backend, browser and timing differences can change floating-point results and performance; they do not justify weakening numerical/quality assertions.

To verify the committed report archive independently:

```sh
node --input-type=module -e "import fs from 'node:fs'; import crypto from 'node:crypto'; import assert from 'node:assert/strict'; const base='docs/evidence/'; const index=JSON.parse(fs.readFileSync(base+'training-evidence-index.json')); for(const item of index.reports){const bytes=fs.readFileSync(base+item.file); assert.equal(bytes.length,item.bytes); assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),item.sha256);} console.log(index.reports.length+' report hashes verified');"
```

Physical iPhone Safari, iPad Safari, iOS PWA installation/background behavior, physical mobile GPU limits and production GitHub Actions/Pages release state require their own evidence. Desktop viewport emulation and a successful local build do not certify those conditions.
