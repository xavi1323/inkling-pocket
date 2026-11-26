# Engine notes

## Zero-copy weights (Task 15 spike, mlx-swift 0.33.0)

- `MLXArray(rawPointer:_:dtype:finalizer:)` (public) calls mlx-c `mlx_array_new_data_managed_payload`,
  which builds `mlx::core::array(data, shape, dtype, deleter)`.
- That constructor calls `allocator::make_buffer(ptr, nbytes)`. On Metal this is
  `device->newBuffer(ptr, size, options, nullptr)` (= `newBufferWithBytesNoCopy`). It returns nil unless the
  pointer is page-aligned and the length is a whole number of pages; then MLX falls back to
  `malloc + copy` and calls the deleter at once.
- Safetensors tensors are not page-aligned, so wrapping the original files copies every tensor
  (≈10.9 GB of language-model weights → too much for the iPad).

**Chosen approach:** on import, `ModelImporter` rewrites the shards with `WeightRepacker`: every tensor starts on a
16 KB boundary and owns whole pages (padding ≈ 15 MB total), vision-tower tensors are dropped (text only), and the
header carries `__metadata__.inkling_alignment = 16384`. `MappedSafetensors` mmaps the shards
(`PROT_READ`, `MAP_PRIVATE`) and wraps each tensor's whole pages as a 1-D array, then slices/reshapes to the real
shape (a view, no copy). The pages are clean and file-backed, so iOS can drop them under pressure and fault them
back from flash.

Consequences:
- "Use the folder in place" skips the repack, so MLX copies weights — only for testing; the copy/repack mode is the
  default.
- Whether Metal-wrapped file pages count against the app's memory limit is unknown until measured on the iPad
  (Benchmark shows footprint vs mapped bytes).
- The fast path (Task 20) can also bind the same mapping as `MTLBuffer`s for custom kernels (`pointer(_:)`).

## Real model facts (from config.json, checked on the PC)

- 48 layer visits: 0–5, 6–17 ×3, 18–23. Global (non-sliding) layers: 5, 11, 17, 23.
- Experts: gate_proj 2 or 3-bit (group 128), up_proj 3-bit (group 128), down_proj 3-bit (group 128 or 64) or 4-bit.
- Attention projections (q, k, v, o, r, gate) 8-bit group 64; embed_tokens and lm_head 6-bit group 64.
- Default (unlisted) quantization: 4-bit group 64.
- Weight names start with `language_model.model.` / `language_model.lm_head`; `vision_tower.*` is unused.
