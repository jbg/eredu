"""Independent NumPy FP32 K2 oracle. Run with Python + NumPy; no eredu imports.

Parameter values use the deterministic numeric-backend fixture seed. Equations
follow the pinned publisher model; packed bank names describe fixture storage.
"""
import json
import math
from pathlib import Path
import numpy as np


def weight(name, shape, norm=False):
    h = 2166136261
    for byte in name.encode():
        h = ((h ^ byte) * 16777619) & 0xffffffff
    result = []
    for i in range(math.prod(shape)):
        x = (h + i * 747796405) & 0xffffffff
        n = i % 23
        x = ((x << n) | (x >> ((32 - n) % 32))) & 0xffffffff
        centered = np.float32(x % 2001 - 1000)
        result.append(np.float32(1) + centered / np.float32(50000) if norm else centered / np.float32(25000))
    return np.array(result, dtype=np.float32).reshape(shape)


def linear(x, root, width):
    return x @ weight(root + ".weight", [width, x.shape[-1]]).T


def norm(x, root, groups, eps):
    y = x.reshape(*x.shape[:-1], groups, -1)
    y = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + np.float32(eps))
    return y.reshape(x.shape) * weight(root + ".weight", [x.shape[-1]], True)


def silu(x):
    return x / (np.float32(1) + np.exp(-x))


def dense(x, root, width):
    return linear(silu(linear(x, root + ".gate_proj", width)) * linear(x, root + ".up_proj", width), root + ".down_proj", x.shape[-1])


def route(x, root, count, top, normalized, config):
    logits = linear(x, root, count)
    if config["router_score_func"] == "sigmoid":
        scores = np.float32(1) / (np.float32(1) + np.exp(-logits))
    else:
        scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
    selection = scores + weight(root + ".bias", [count], True) if config["moe_gate_bias"] else scores
    ids = np.argsort(-selection, axis=-1, kind="stable")[:, :top]
    selected = np.take_along_axis(scores, ids, axis=-1)
    coefficients = selected / selected.sum(axis=-1, keepdims=True) if normalized else selected.copy()
    coefficients *= np.float32(config["router_scaling_factor"])
    return ids, selected, coefficients


def rotary(x, config):
    head = config["head_dim"]
    dims = config.get("rope_head_dim", head)
    half = dims // 2
    inv = np.float32(1) / np.power(np.float32(config["rope_theta"]), np.arange(half, dtype=np.float32) * np.float32(2 / dims))
    rope = config["rope_parameters"]
    amplitude = np.float32(1)
    if rope["rope_type"] == "yarn":
        low = max(0, math.floor(dims * math.log(rope["original_max_position_embeddings"] / (rope["beta_fast"] * 2 * math.pi)) / (2 * math.log(config["rope_theta"]))))
        high = min(dims - 1, math.ceil(dims * math.log(rope["original_max_position_embeddings"] / (rope["beta_slow"] * 2 * math.pi)) / (2 * math.log(config["rope_theta"]))))
        ramp = np.clip((np.arange(half, dtype=np.float32) - low) / np.float32(high - low if high != low else .001), 0, 1)
        inv = inv * (1 - ramp) + inv / np.float32(rope["factor"]) * ramp
        amplitude = np.float32(rope["attention_factor"])
    angles = np.arange(x.shape[0], dtype=np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(angles)[:, None, :] * amplitude, np.sin(angles)[:, None, :] * amplitude
    result = x.copy()
    first, second = x[:, :, :half], x[:, :, head // 2:head // 2 + half]
    result[:, :, :half] = first * cos - second * sin
    result[:, :, head // 2:head // 2 + half] = second * cos + first * sin
    return result


def run(config):
    tokens = [1, 3, 2, 4, 5, 6, 7]
    hidden, head = config["hidden_size"], config["head_dim"]
    qh, kh = config["num_attention_heads"], config["num_key_value_heads"]
    x = weight("model.embed_tokens.weight", [config["vocab_size"], hidden])[tokens]
    routes = {}
    for layer in range(config["num_hidden_layers"]):
        root = f"model.layers.{layer}"
        sparse = config["num_experts"] > 0 and layer not in config["mlp_only_layers"] and (layer + 1) % config["decoder_sparse_step"] == 0
        a = norm(x, root + ".input_layernorm", config["layernorm_num_groups"], config["rms_norm_eps"])
        q, k = linear(a, root + ".self_attn.q_proj", qh * head), linear(a, root + ".self_attn.k_proj", kh * head)
        if config["query_key_norm"]:
            q, k = norm(q, root + ".self_attn.q_norm", qh, config["rms_norm_eps"]), norm(k, root + ".self_attn.k_norm", kh, config["rms_norm_eps"])
        if sparse and config["mova_num_experts"]:
            n, top = config["mova_num_experts"], config["mova_num_experts_per_tok"]
            ids, scores, coeff = route(a, root + ".self_attn.v_router", n, top, top > 1, config)
            routes[f"{layer}:1"] = dict(ids=ids.tolist(), scores=scores.tolist(), coefficients=coeff.tolist())
            bank = weight(root + ".self_attn.v_experts.weight", [n, kh * head, hidden])
            v = np.stack([sum(silu(bank[e] @ a[t]) * coeff[t, j] for j, e in enumerate(ids[t])) for t in range(len(tokens))])
        else:
            v = linear(a, root + ".self_attn.v_proj", kh * head)
        q, k = rotary(q.reshape(-1, qh, head), config), rotary(k.reshape(-1, kh, head), config)
        v = v.reshape(-1, kh, head)
        attended = np.empty_like(q)
        for t in range(len(tokens)):
            for h in range(qh):
                kv = h // (qh // kh)
                scores = k[:t + 1, kv] @ q[t, h] / np.float32(math.sqrt(head))
                prob = np.exp(scores - scores.max())
                prob /= prob.sum()
                attended[t, h] = prob @ v[:t + 1, kv]
        attended = attended.reshape(-1, qh * head)
        if config["attention_gate_func"]:
            gate = linear(a, root + ".self_attn.gate_proj", qh * head)
            attended *= np.logaddexp(np.float32(0), np.float32(math.log(2)) * gate) / np.float32(math.log(2))
        x = x + linear(attended, root + ".self_attn.o_proj", hidden)
        a = norm(x, root + ".post_attention_layernorm", config["layernorm_num_groups"], config["rms_norm_eps"])
        if sparse:
            n, width, top = config["num_experts"], config["moe_intermediate_size"], config["num_experts_per_tok"]
            ids, scores, coeff = route(a, root + ".mlp.gate", n, top, config["norm_topk_prob"], config)
            routes[f"{layer}:0"] = dict(ids=ids.tolist(), scores=scores.tolist(), coefficients=coeff.tolist())
            gu = weight(root + ".mlp.experts.gate_up_proj", [n, 2 * width, hidden])
            down = weight(root + ".mlp.experts.down_proj", [n, hidden, width])
            y = np.stack([sum((down[e] @ (silu(gu[e, :width] @ a[t]) * (gu[e, width:] @ a[t]))) * coeff[t, j] for j, e in enumerate(ids[t])) for t in range(len(tokens))])
            if config["num_shared_experts"]:
                y += dense(a, root + ".mlp.shared_experts", width * config["num_shared_experts"])
        else:
            y = dense(a, root + ".mlp", config["intermediate_size"])
        x = x + y
    logits = linear(norm(x, "model.norm", config["layernorm_num_groups"], config["rms_norm_eps"]), "lm_head", config["vocab_size"])
    return dict(config=config, tokens=tokens, logits=logits.tolist(), routes=routes)


def main():
    base = dict(model_type="k2_horizon", hidden_size=8, num_hidden_layers=3, intermediate_size=12,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, rms_norm_eps=1e-5,
        vocab_size=17, max_position_embeddings=256, rope_theta=1000, layernorm_num_groups=2,
        query_key_norm=True, attention_gate_func="softplus", num_experts=5, num_experts_per_tok=2,
        moe_intermediate_size=6, num_shared_experts=2, mlp_only_layers=[0], decoder_sparse_step=1,
        norm_topk_prob=False, moe_gate_bias=True, router_score_func="sigmoid", router_scaling_factor=1.7,
        mova_num_experts=3, mova_num_experts_per_tok=2, rope_head_dim=2,
        rope_parameters=dict(rope_type="yarn",factor=4,original_max_position_embeddings=16,beta_fast=8,beta_slow=1,attention_factor=1.35))
    cases = {}
    for name in ["dense", "grouped", "moe", "mova"]:
        c = base.copy()
        if name != "mova":
            c.update(mova_num_experts=0,mova_num_experts_per_tok=0,rope_head_dim=4)
        if name in ["dense", "grouped"]:
            c.update(num_experts=0,num_experts_per_tok=0,num_shared_experts=0)
        if name == "dense":
            c.update(query_key_norm=False,layernorm_num_groups=1,attention_gate_func=None)
        cases[name] = run(c)
    Path(__file__).with_name("reference.json").write_text(json.dumps(cases, indent=2) + "\n")

if __name__ == "__main__":
    main()
