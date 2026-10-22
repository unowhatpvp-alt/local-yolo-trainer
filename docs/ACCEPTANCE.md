# V1 acceptance and release gates

The agreed scope is the user's complete Local YOLO Trainer Minimal chapters 1–20. This is an independent website. The two unspecified future Local YOLO products and desktop software are outside this release.

## Confirmed scope

| Requirement | Implementation and evidence |
| --- | --- |
| Local browser operation | Static Vite application, a real training Worker, same-origin frameworks and feature model; no training API or image upload endpoint |
| Minimal workflow | Pictures → Mark objects → Teach model → Test & save; advanced training settings are collapsed |
| Batches, orientation, duplicates, errors | Sequential import, original oriented resolution, thumbnails, bounded gallery, duplicate hashes, real EXIF/transparent/corrupt image browser tests |
| Single/multiple categories | Reusable names, accurate original-pixel boxes, move/eight resize handles/delete/class changes/history; safe mode switching |
| Mouse/touch/responsive design | Real pointer/touch gesture tests; six desktop-emulated viewports, two languages and themes; 132 actual-page/dialog layout checks |
| Actual pretrained training | Immutable YOLO26n neck features plus a trainable spatial detection head; original-photo browser training and checkpoint/ONNX proof |
| From scratch | Randomly initialized three-convolution detector, all weights trainable; independent data-dependent gradient/update and reload proof |
| Continue training | Full Adam/RNG/sample-order state, exact paused CPU continuation, additional-epoch semantics and changed-dataset handling |
| Dataset correctness | Reviewed-negative support, validation isolation, duplicate-pixel leakage rejection, transformed horizontal-flip boxes; unsupported collisions fail explicitly |
| Real progress/recovery | Worker progress, loss, elapsed/estimated time, pause/stop, periodic checkpoints; closing the browser does not continue training |
| Data protection | Actual IndexedDB transaction-commit ordering, atomic checkpoint rotation, atomic confirmed image removal, quota fault injection and successful retry |
| Backup/restore | Complete project archive imports with fresh UUIDs, no overwrite; image hashes/classes/boxes/checkpoints preserved |
| Export/interoperability | Valid ONNX numerical parity and actual import into the unchanged inference website; YOLO dataset archive and resumable model archive |
| GitHub Pages | Subpath build verification, hash navigation, pinned model and WASM integrity, notices, HTTPS; verification jobs precede deployment |

The default pretrained mode is **not full YOLO26 fine-tuning**. The scratch mode is **not YOLO26 from scratch**. These limits are visible in the interface and README. The application does not promise useful recognition from one picture or from every tiny dataset.

## Verification boundaries

See [training verification](TRAINING_VERIFICATION.md), [UI verification](UI_VERIFICATION.md), and the accompanying numerical evidence. Tests distinguish generated shape regressions, natural-photo diagnostics, actual hardware acceleration, and desktop mobile emulation.

Physical Android phones, iPhone/iPad Safari, iOS PWA, Firefox, and independently installed Google Chrome have not been qualified in this environment. Desktop Chromium and installed Microsoft Edge are the directly tested browsers. Feature detection and an actual architecture-specific gradient/numerical check choose a usable backend on each device. WASM scratch convolution gradients are unsupported by the pinned framework and are disabled with the real reason.

The original-photo test set was first evaluated before an engineering fix to match the inference site's high-quality Canvas scaling. That first report is preserved. A necessary replay after the fix uses unchanged architecture, loss, hyperparameters and thresholds; its result is not presented as a fresh independent benchmark. No model selection used those final eight photos.

No physical full-disk condition was induced. QuotaExceededError was explicitly injected into the real IndexedDB write path. Atomic-transaction rollback was tested using a real native duplicate-key ConstraintError. Main-page JavaScript heap measurements are not total Worker/WASM memory, system RAM, or GPU VRAM.

## Release procedure

The release dependency review is recorded in [dependency review](DEPENDENCY_REVIEW.md). Malformed ZIP64 project/checkpoint imports have a bounded-process regression test; the directly reachable archive-parser issue is fixed by the pinned fflate patch.

The [verification workflow](../.github/workflows/pages.yml) installs the lockfile, runs unit tests, builds and validates the exact static artifact, runs real browser UI/storage/media tests, then runs original-photo Worker training, checkpoint reload and ONNX numerical parity. Pull requests cannot deploy. A main-branch deployment depends on verification succeeding.

Current CI and deployment status is available in [GitHub Actions](https://github.com/unowhatpvp-alt/local-yolo-trainer/actions). A written report alone is not a successful deployment. The released site must additionally be opened and exercised after deployment, including same-origin runtime/model loading and a real train/save/test cycle.

No source photographs or local project data are included in the static artifact. Downloaded developer fixtures and historical experimental models remain ignored. Published numerical reports record limitations and source hashes; a historical passing report does not establish that later modified files were tested.

## Recovery

This is the first independent Trainer release; there is no earlier stable Trainer website to roll back to. Keep a known-good release commit and its source archive. To recover a later regression, create a normal revert or a new branch from that known-good commit, run verification, and redeploy. Do not force-push or reset existing history.

Project backups (`.lyproj`) contain pictures, annotations, configuration and the compatible checkpoint; import creates a separate project. A `.lymodel` saves training state without source pictures. Browser profile/origin storage is not a permanent backup.

The existing inference repository remains separate and unchanged at `435a299fbe60c8e0fe4570c9e0cc8f944f37ecbc`. Its earlier stable backup is `backup/minimal-v1-2cea0eb` at `2cea0ebebd3399a77a2d649c44755e54ca0e16b0`. Trainer recovery must not overwrite that site.
