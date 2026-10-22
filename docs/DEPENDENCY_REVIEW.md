# Dependency advisory review

Reviewed on **2026-10-22 UTC** (2026-10-21 in America/New_York). This is a review of the pinned dependencies and actual browser build, not a claim that every dependency is free of vulnerabilities.

The initial review covers PR 1 head `fecfda5a2748a7b299edf059d6777d5e0cbd4ef5`. Its successful CI still reported four moderate package findings. Those findings represented **two root advisories**, rather than four independent defects. After the compatible fflate patch below, a fresh official npm audit reported **three moderate package findings, one root advisory, zero high and zero critical findings**. The audit exits with status 1 while those three findings remain; they have not been hidden or reclassified.

| Installed package at initial review | Relationship | Initial audit finding |
| --- | --- | --- |
| `fflate@0.8.2` | Direct production dependency | Malformed ZIP64 header parsing can fail to terminate |
| `@tensorflow/tfjs@4.22.0` | Direct production dependency | Propagated finding from its declared `argparse` dependency |
| `argparse@1.0.10` | Indirect production dependency | Propagated finding from `sprintf-js` |
| `sprintf-js@1.0.3` | Indirect production dependency | Attacker-controlled numeric precision format can throw an uncaught RangeError |

These are production dependency declarations in the lockfile, not development-only packages. Inclusion in an installed production dependency tree does not establish inclusion in the deployed browser JavaScript; that distinction is checked below.

## fflate: fixed before release

[GHSA-px8p-9vwx-vf98 / CVE-2026-45820](https://github.com/advisories/GHSA-px8p-9vwx-vf98) identifies affected 0.8.x versions before 0.8.3. A ZIP64 sentinel compressed size without the required extra field can leave the old extra-field search unbounded. The [upstream fix](https://github.com/101arrowz/fflate/commit/e6d5e6e1076892f72770ac732d83c81da9f3316e) adds an explicit extra-field boundary check; [release 0.8.3](https://github.com/101arrowz/fflate/releases/tag/v0.8.3) includes the fix.

This was an actual release blocker for this application. `src/data/formats.js` imports fflate's asynchronous `unzip` for user-supplied `.lyproj` and `.lymodel` files. The old asynchronous and synchronous APIs share central-directory parsing. Parsing the vulnerable header happens before the application's path/size filter, so existing archive filters cannot prevent the loop. An imported malicious file could freeze that browser page. This application has no archive-processing backend; this finding does not establish a server compromise or data exfiltration.

The dependency is now pinned to **`fflate@0.8.3`**. It is a patch upgrade with the existing zip/unzip interface retained; the application is not downgraded and no audit-force command is used. Its updated upstream MIT notice must accompany the build.

A locally constructed 202-byte malformed ZIP64 archive uses a central-directory compressed-size sentinel `0xffffffff` and zero extra-field length, followed by valid ZIP64 end/locator records. Both actual public entry points, `unpackProject` and `unpackCheckpoint`, rejected it with `invalid zip data`, code 13, under 0.8.3. The regression runs in a bounded child process: a JavaScript promise timeout in the same thread cannot stop an old synchronous parser loop. After the dependency change, all **54 unit tests**, **8 browser tests**, the production build and static artifact checks passed locally, including valid checkpoint/project round trips and browser storage/import regressions. The old installed version was not deliberately hung inside the application process. Remote CI and publication remain separate checks.

## sprintf-js: installed, not included in this website

[GHSA-hp3w-g68c-fv3c / CVE-2026-97058](https://github.com/advisories/GHSA-hp3w-g68c-fv3c) covers `sprintf-js` through 1.1.3. Its trigger requires control of a format string with an out-of-range numeric precision. The advisory lists **no patched version**. An official npm registry query during this review returned `latest: 1.1.3` and versions ending at 1.1.3; **1.1.4 was not available**.

The exact installed path is:

```text
local-yolo-trainer-minimal
└─ @tensorflow/tfjs@4.22.0
   └─ argparse@1.0.10
      └─ sprintf-js@1.0.3
```

A small caught diagnostic against the installed 1.0.3 implementation confirmed that `sprintf('%.10000f', 1.5)` throws `RangeError`. The package is vulnerable when used with an uncontrolled format string. We did not claim that installing it is safe in every application.

For this deployment, the evidence is narrower and concrete:

- `src/training/backend.js` imports TensorFlow.js through its browser ESM entry. `node_modules/@tensorflow/tfjs/dist/index.js` imports the TensorFlow subpackages and registers gradients/chained operations; it does not import argparse or sprintf-js.
- The separate installed `tfjs-custom-module` Node CLI uses `yargs`. It is not invoked by the app, build, or test scripts and is not a browser entry. We do not assume that argparse is necessarily used merely because a CLI and a dependency are present in the same package.
- Searches of the actual emitted `dist/assets` JavaScript found no argparse/sprintf module markers. More decisively, a fresh **in-memory production Rollup build**, with `write:false` and `emptyOutDir:false`, inspected every emitted chunk's `modules` list in both the main and Worker builds. It inspected **18 page modules and 1,704 Worker modules**: zero argparse modules, zero sprintf-js modules, and zero custom-module CLI modules. The Worker inventory did contain the expected `@tensorflow/tfjs/dist/index.js`; the page inventory did not.
- No production code exposes a formatting endpoint or imports either affected module. The site serves static browser assets rather than the installed `node_modules` tree.

On this evidence, the remaining single root advisory is **not a release blocker for this specific static browser artifact**. This is a scoped reachability assessment, not suppression of npm's result, proof about unrelated Node use, or a promise about future bundles. New imports, dependency versions, build settings, server-side tools, or user-controlled formatting require a fresh review.

## Reproduction and follow-up

These checks are read-only with respect to package manifests and the lockfile:

```sh
npm audit --json
npm view sprintf-js versions dist-tags --json
rg -n -uuu 'argparse|sprintf|ArgumentParser' node_modules/@tensorflow/tfjs/dist/index.js node_modules/@tensorflow/tfjs/dist/tools/custom_module/cli.js
rg -l -uuu 'argparse|sprintf|\[sprintf\]' dist/assets
```

The module inventory used Vite's programmatic `build` with `build: {write:false, emptyOutDir:false}`. A `generateBundle` diagnostic collected `Object.keys(chunk.modules)` separately for the page and `worker.plugins` build and checked the path segments `node_modules/argparse/`, `node_modules/sprintf-js/` and `custom_module`. No source, runtime files or learned model tensors were changed by that inventory.

Keep the TensorFlow.js 4.22.0 set aligned. npm's suggested downgrade to TensorFlow.js 2.1.0 is a major compatibility change, not a validated security patch for this engine. Do not use `npm audit fix --force`, override to an unpublished version, or replace argparse across major versions solely to silence the report. Prefer a genuine upstream fix or a compatible TensorFlow release that removes the unused dependency, then rerun backend/gradient, checkpoint, ONNX and browser training regressions and repeat the actual bundle inventory. Record the remaining advisory honestly until it is removed.
