# UI verification

The interface is implemented and the browser workflow below passed against a production static build. This is a record of a specific tested candidate, rather than a claim that every browser, device, training mode, or later source revision has passed.

## Product scope

The primary flow is **Pictures → Mark objects → Teach model → Test & save**. The black-and-white interface keeps technical settings in an expandable section. It provides Chinese/English, light/dark/system appearance, desktop and small-screen layouts, local project management, and explicit save/error states.

The annotation editor provides multiple boxes and classes, original oriented-image coordinates, creation, movement, eight resize handles, class changes, deletion confirmation, undo/redo, keyboard controls, zoom/pan, and two-finger gestures. It uses Blob URLs and releases them when pictures or views are replaced. The shared editor also displays model predictions in read-only mode.

No additional essential UI feature was identified in this review. Model quality, backend qualification, and physical-device qualification are separate acceptance questions.

## Executed browser checks

The final focused run completed on Windows with Playwright Chromium on **2026-10-22, 00:05:59–00:06:45 UTC**: **4 browser tests passed, 0 failed, 0 skipped**.

| Area | Actual evidence |
| --- | --- |
| Annotation editor | Real pointer drawing, all eight SVG resize handles, movement, category changes, undo/redo, zoomed original-pixel coordinates, deletion confirmation, and Blob URL release |
| Touch interaction | Trusted Chromium touch input draws boxes; two-finger zoom does not create accidental annotations; interrupted gestures restore the saved boxes |
| Shared navigation | Both languages and themes fit the tested sizes; branding, header actions, and step navigation do not overlap |
| Actual application | Upload, duplicate detection, single/multiple classes, automatic save and reload, training, inference, model downloads, and full project recovery |

Actual application layout checks used these six CSS viewports:

| Layout | Width × height |
| --- | --- |
| Small phone portrait | 375 × 650 |
| Phone portrait | 390 × 844 |
| Phone landscape | 844 × 390 |
| Tablet portrait | 768 × 1024 |
| Tablet landscape | 1024 × 768 |
| Desktop | 1280 × 900 |

Five rendered application states—empty upload, gallery, annotation, training settings, and model testing—were checked at all six sizes in Chinese/English and light/dark: **120 combinations**. Project and help dialogs were additionally checked at all six sizes in the current Chinese/light state: **12 combinations**, for **132 total**.

These checks assert actual rendered content, horizontal containment, visible controls, and a minimum 44 CSS px touch target for enabled buttons and form interaction areas. A 0.001 px arithmetic tolerance handles browser rectangle rounding; the CSS target remains 44 px. Normal vertical page scrolling is intentional.

## End-to-end results

- Uploaded generated pictures through the real file chooser; duplicate detection preserved the existing pictures. Project switching and navigation were observed disabled during image processing.
- Created and reloaded two classes and multiple accurately positioned boxes. Fast category selection after picture loading retained the chosen class.
- Cancelled removal preserved the picture. Confirmed removal after visiting training survived reload; read-only IndexedDB checks found no remaining original, thumbnail, or project reference for the removed image.
- An invalid epoch input reported an error while the last valid project remained exportable.
- A real CPU `tiny-grid-v1` run performed two parameter updates, produced finite loss and a nonzero gradient, changed the weight SHA, and displayed completed progress from the actual run.
- Continuing its checkpoint advanced the actual step count from **2 to 4**. The resumed starting weight SHA matched the previous ending SHA, then changed again.
- Tested a picture, downloaded a checkpoint and ONNX, exported a complete project, created a separate project, imported the backup, and ran inference again. Images, both classes, annotations, and the checkpoint were retained. Actual held-out validation metrics were restored from checkpoint history unchanged.
- The actual application run recorded **0 page script errors** and **0 non-GET/HEAD requests** carrying user data.

These deliberately tiny, generated examples verify computation and interaction. They do **not** establish useful detection accuracy or generalisation. The test picture returned zero detections, and the retained validation record includes missed targets; neither result was replaced with a success animation. This UI run does not qualify all training modes or backends.

## Candidate identity and reproduction

The aggregate source SHA before and after the run was identical:

```text
ad347a691ecc5d4a9cd7dbd1c5ceb9e21b252acc3b149afaf40d0d338e597fec
```

It covers 27 files: all files under `src/`, plus `index.html`, `vite.config.js`, `package-lock.json`, and the model manifest. Sorted relative paths are hashed as `path + NUL + exact file bytes + NUL`; individual file hashes are recorded in the generated report. This fingerprint is byte-sensitive, including line endings.

The tested build's HTML SHA was:

```text
7ea2b88894ffc069afb9c66579a99e2839e0f5a4befe8787453fffdc79310620
```

Later Node-only resource-path or checkpoint-validation changes are outside that candidate fingerprint. Run the checks again on the desired revision and use its CI artifacts for current-source acceptance.

```sh
npm ci
npx playwright install chromium
npm run build
node --test tests/annotation-browser.mjs tests/app-ui-browser.mjs
```

`npm run test:browser` also includes the separate storage regression suite. The focused application test builds its own candidate under `test-results/ui/build/`, serves the GitHub Pages subpath through a temporary static preview, and closes its browser/server in `finally`. It does not depend on a development hot-reload session.

Generated evidence is written to `test-results/ui/actual-app-ui.json`; earlier failures remain under `test-results/ui/history/`. These are local/CI artifacts, not committed source. The report includes source hashes, the build HTML hash, all layout records, real training records, busy-state observations, and post-removal storage checks.

## Screenshots and limits

[Desktop upload screenshot](images/trainer-desktop.png) · [Mobile-viewport annotation screenshot](images/trainer-annotation-mobile.png)

The screenshots use generated examples to demonstrate interaction, not model accuracy. 使用生成示例展示交互，不代表检测精度。

Phone and tablet sizes, touch injection, and device pixel ratios were tested in **desktop Chromium**. They are not physical-device training evidence. Real Android browsers, iPhone Safari, iOS browser suspension/virtual-keyboard behaviour, physical tablet gestures, assistive-technology combinations, and device-specific memory/GPU limits remain unverified by this run. Native mobile performance must be measured on those devices.

The static build and subpath loaded locally. A local pass does not by itself establish a published GitHub Pages deployment or an online-device pass. The existing inference website was outside this UI task and was not modified.
