# Inkling Pocket

On-device AI agent for iPad Air M4, running Inkling 2 Mini (abliterated, MLX 12GB build) with a custom engine.

- Design: `docs/Design.md` · Spec: `docs/Spec.md` · Plan: `docs/Plan.md`
- Builds: every push runs GitHub Actions and uploads an unsigned `InklingPocket.ipa` (artifact `InklingPocket-ipa`).
- Install: import the IPA into LiveContainer. Model files are never in this repo; import the model folder in the app.

## Install on the iPad (LiveContainer)

1. Open the latest green run under **Actions** and download the `InklingPocket-ipa` artifact (a zip with `InklingPocket.ipa`).
2. In LiveContainer, tap **+** and pick `InklingPocket.ipa`. LiveContainer signs it; no Apple ID goes into this app.
3. Keep iRAM+ (the memory-cap patch) enabled for Inkling Pocket in LiveContainer.

## Import the model

1. Download `orcarouter/Inkling-2-Mini-abliterated-MLX` (the 12 GB build) into a folder in Files.
2. In the app: **Import** → pick that folder. It is checked, copied and re-laid out for fast loading (needs about 12 GB free).
   "Use the folder where it is" skips the copy but loads slower and uses more memory.
3. **Settings** shows the model status. **Benchmark** → *Run self-test* checks the fast engine against the plain one; *Run benchmark* reports tokens per second and memory. *Copy results* puts the text on the clipboard.

## Credits

See `THIRD_PARTY_NOTICES.md`. This product includes software developed by UncleCode (https://x.com/unclecode) as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).

