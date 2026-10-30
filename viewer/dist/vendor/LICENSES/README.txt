Vendored third-party code (no CDN at run time). Both are MIT-licensed; the licence files are next to this note.

@sparkjsdev/spark 2.4.0   (World Labs Technologies, MIT)   npm tarball sha1 465990d1aa43c0222c56eb6ed235494019ec8382
  vendor/spark.module.js   = package/dist/spark.module.js   (WASM inline)
three 0.180.0             (three.js authors, MIT)           npm tarball sha1 b930cabfb524f6d36bf63e874b4d866888b50487
  vendor/three.module.js   = package/build/three.module.js
  vendor/three.core.js     = package/build/three.core.js     (imported by three.module.js)
  vendor/addons/postprocessing/Pass.js = package/examples/jsm/postprocessing/Pass.js   (imported by Spark)

sha256
  5ae86ab35d4ceb655db486dcc336299775bb314c280d7e5b39f64e4539f8e0e9  spark.module.js
  c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a  three.module.js
  eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20  three.core.js
  444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5  addons/postprocessing/Pass.js

Fetched on 2026-10-30 with `npm pack @sparkjsdev/spark@2.4.0 three@0.180.0` into a scratch folder; files copied unmodified.
