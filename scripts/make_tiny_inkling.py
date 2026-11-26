"""Build a tiny random-weight inkling2_loop text model and its golden outputs.

Same structure as Inkling 2 Mini (looped core with per-visit caches, MoE, sinks, short convs, sliding + global
attention, attention residuals, message controllers, mixed quantization), shrunk to < 5 MB.
Run on CPU with mlx 0.32.4 (the MLX core inside mlx-swift 0.33.0; 0.32.5 changed CPU quantized matmul rounding)
+ mlx-vlm at commit d998a6207b7dc613b52618646dbea13a4799f7ca:

    python -I scripts/make_tiny_inkling.py InklingKit/Tests/InklingEngineTests/Fixtures/tiny-inkling
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path, tree_unflatten

from mlx_vlm.models.inkling2.config import TextConfig
from mlx_vlm.models.inkling2.language import LanguageModel

OUT = sys.argv[1]
assert mx.__version__ == "0.32.4", f"use mlx 0.32.4, found {mx.__version__}"
mx.set_default_device(mx.cpu)
mx.random.seed(0)

text = dict(
    model_type="inkling2_loop_text", hidden_size=64, num_hidden_layers=8, vocab_size=512, unpadded_vocab_size=500,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
    sliding_window_size=8, local_layer_ids=[0, 2, 3, 4, 5, 6], d_rel=16, rel_extent=16,
    log_scaling_n_floor=4, log_scaling_alpha=0.1, sconv_kernel_size=4, intermediate_size=32,
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=0, route_scale=8.0,
    n_layers_per_attn_res_block=2, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, final_logit_softcapping=30.0,
    rms_norm_eps=1e-6, use_gate_bias=True,
    loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=4, is_varloop=False),
)
config = TextConfig.from_dict(text)
model = LanguageModel(config)

F32_NAMES = ("gate_weight", "e_score_correction_bias", "query", "message_queries")


def randomize(path, value):
    if not isinstance(value, mx.array) or not mx.issubdtype(value.dtype, mx.floating):
        return value
    leaf = path.split(".")[-1]
    if "norm" in path or leaf in ("q_scale", "rel_scale"):
        new = 1.0 + 0.1 * mx.random.normal(value.shape)
    elif leaf == "e_score_correction_bias":
        new = 0.05 * mx.random.normal(value.shape)
    elif value.ndim >= 2 and value.shape[-1] >= 8:
        # ~unit gain per projection keeps the random model from amplifying last-bit rounding differences.
        new = (0.6 / value.shape[-1] ** 0.5) * mx.random.normal(value.shape)
    else:
        new = 0.2 * mx.random.normal(value.shape)
    keep_f32 = any(leaf == n or path.endswith(n) for n in F32_NAMES)
    return new.astype(mx.float32 if keep_f32 else mx.bfloat16)


model.update(tree_map_with_path(randomize, model.parameters()))

quant = {"group_size": 32, "bits": 4}
for i in range(config.num_hidden_layers):
    base = f"model.layers.{i}"
    quant[f"{base}.mlp.switch_mlp.gate_proj"] = {"group_size": 32, "bits": 2 if i % 2 else 3, "mode": "affine"}
    quant[f"{base}.mlp.switch_mlp.up_proj"] = {"group_size": 32, "bits": 3, "mode": "affine"}
    quant[f"{base}.mlp.switch_mlp.down_proj"] = {"group_size": 32, "bits": 4 if i == 7 else 3, "mode": "affine"}
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "r_proj", "gate_proj"):
        quant[f"{base}.self_attn.{p}"] = {"group_size": 32, "bits": 8, "mode": "affine"}
quant["model.embed_tokens"] = {"group_size": 32, "bits": 6, "mode": "affine"}
quant["lm_head"] = {"group_size": 32, "bits": 6, "mode": "affine"}


def predicate(path, module):
    spec = quant.get(path)
    return dict(spec) if isinstance(spec, dict) else False


nn.quantize(model, class_predicate=predicate)
mx.eval(model.parameters())

weights = {f"language_model.{k}": v for k, v in tree_flatten(model.parameters())}
os.makedirs(OUT, exist_ok=True)
mx.save_safetensors(os.path.join(OUT, "model.safetensors"), weights, metadata={"format": "mlx"})
cfg = {
    "model_type": "inkling2",
    "eos_token_id": 499,
    "text_config": text,
    "quantization": {
        **{k: v for k, v in quant.items() if k in ("group_size", "bits")},
        **{f"language_model.{k}": v for k, v in quant.items() if isinstance(v, dict)},
    },
}
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(cfg, f, indent=1)

# Golden outputs: 12 teacher-forced tokens through a fresh cache, then 8 greedy steps.
tokens = [1, 17, 42, 99, 256, 3, 499, 7, 77, 128, 311, 64]
cache = model.make_cache()
logits = model(mx.array([tokens]), cache=cache).logits.astype(mx.float32)
mx.eval(logits)
greedy = []
nxt = int(mx.argmax(logits[0, -1]).item())
for _ in range(8):
    greedy.append(nxt)
    out = model(mx.array([[nxt]]), cache=cache).logits.astype(mx.float32)
    nxt = int(mx.argmax(out[0, -1]).item())
golden = {
    "tokens": tokens,
    "vocab": logits.shape[-1],
    "logits": [[round(float(x), 6) for x in row] for row in logits[0].tolist()],
    "greedy": greedy,
    "visits": len(cache),
    "mlx_version": mx.__version__,
}
with open(os.path.join(OUT, "golden.json"), "w") as f:
    json.dump(golden, f)
print("weights", sum(v.nbytes for v in weights.values()), "bytes; visits", len(cache), "greedy", greedy)
