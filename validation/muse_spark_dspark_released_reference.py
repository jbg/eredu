"""Released DSpark weights, accepted target captures and actual target embeddings.

The pinned vLLM block/model, normalization, Markov, confidence and output-transform
definitions execute unchanged. CPU PyTorch projection/attention replace native
serving kernels. This is neural component evidence, not a speculative session.
"""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

import torch
from torch import nn

from muse_spark_dspark_reference_source import (
    definitions, rope, head_norm, BODY_SHA256, REVISION, SOURCE_SHA256,
    record, SOURCES, HEAD_SHA256, ParallelLMHead, ReplicatedLinear,
)

ASSISTANT_REVISION = "6f7aed1cac8d8b1c02a902988876172b39e49fc0"
TARGET_REVISION = "4239e4db19376904f2e82c167e78f70abb5455fe"


@lru_cache(maxsize=3)
def header(path):
    with path.open("rb") as source:
        length = int.from_bytes(source.read(8), "little")
        if length > 16 << 20 or length > path.stat().st_size-8:
            raise ValueError("invalid SafeTensors header extent")
        return 8+length, json.loads(source.read(length))


def read_tensor(path, name, row=None):
    base, tensors = header(path)
    tensor = tensors[name]
    shape, (first, end) = tensor["shape"], tensor["data_offsets"]
    dtype = {"BF16": torch.bfloat16, "I64": torch.int64}[tensor["dtype"]]
    if end-first != math.prod(shape)*torch.empty((), dtype=dtype).element_size():
        raise ValueError("tensor byte extent differs from geometry")
    if row is not None:
        if not 0 <= row < shape[0]:
            raise ValueError("embedding row outside vocabulary")
        stride = (end-first)//shape[0]
        first, end = first+row*stride, first+(row+1)*stride
        shape = [1, *shape[1:]]
    with path.open("rb") as source:
        source.seek(base+first)
        data = bytearray(source.read(end-first))
    if len(data) != end-first:
        raise ValueError("truncated tensor payload")
    return torch.frombuffer(data, dtype=dtype).reshape(shape)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def emit(output, name, value):
    data = value.detach().float().contiguous().numpy().tobytes()
    (output / name).write_bytes(data)
    return dict(file=name, sha256=hashlib.sha256(data).hexdigest(),
                shape=list(value.shape), dtype=str(value.dtype).split(".")[-1])


def read_trace(root, record):
    path = root / record["file"]
    if digest(path) != record["sha256"]:
        raise ValueError("target trace digest differs")
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).reshape(record["shape"])


class Attention(nn.Module):
    def __init__(self, c, norm):
        super().__init__()
        self.c = c
        self.pre_qkv_proj_norm = norm(c.hidden_size, eps=c.rms_norm_eps, zero_centered_gamma=True)
        for name, width in [("q_proj", c.num_attention_heads*c.head_dim),
                            ("k_proj", c.num_key_value_heads*c.head_dim),
                            ("v_proj", c.num_key_value_heads*c.head_dim),
                            ("o_gate_proj", c.hidden_size), ("o_proj", c.hidden_size)]:
            setattr(self, name, nn.Linear(c.hidden_size, width, bias=False, device="meta"))

    def project_kv(self, hidden, positions):
        c = self.c
        k = self.k_proj(hidden).reshape(-1, c.num_key_value_heads, c.head_dim).transpose(0, 1)
        v = self.v_proj(hidden).reshape(-1, c.num_key_value_heads, c.head_dim).transpose(0, 1)
        return rope(head_norm(k), positions), v

    def forward(self, positions, hidden_states):
        c = self.c
        q = self.q_proj(hidden_states).reshape(-1, c.num_attention_heads, c.head_dim).transpose(0, 1)
        q = rope(head_norm(q), positions)
        k, v = self.project_kv(hidden_states, positions)
        repeats = c.num_attention_heads // c.num_key_value_heads
        self.keys = torch.cat((self.committed[0], k), 1)
        self.values = torch.cat((self.committed[1], v), 1)
        k = self.keys.repeat_interleave(repeats, 0)
        v = self.values.repeat_interleave(repeats, 0)
        key_positions = torch.arange(self.context_start, int(positions[-1])+1)
        allowed = (key_positions[None] <= positions[:, None]) & (positions[:, None]-key_positions[None] < c.sliding_window)
        scale = (43.7840518911 / math.sqrt(c.head_dim)) / math.sqrt(c.head_dim)
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        attended = (torch.softmax(scores.masked_fill(~allowed[None], float("-inf")), -1) @ v.float()).to(v.dtype)
        self.queries, self.attended = q, attended
        normalized = head_norm(attended.float()).transpose(0, 1).reshape(-1, c.hidden_size)
        gated = (normalized * torch.sigmoid(self.o_gate_proj(hidden_states).float())).to(hidden_states.dtype)
        return self.o_proj(gated)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False, device="meta")
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False, device="meta")
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False, device="meta")

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("serving", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if torch.__version__ != "2.9.0":
        raise ValueError("requires torch==2.9.0")
    torch.set_num_threads(8)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("released execution outputs belong outside tracked source")
    c = SimpleNamespace(**json.loads((args.checkpoint / "config.json").read_text()))
    if digest(args.checkpoint / "config.json") != "588298a4cd6e5179467656644c33d29c31f33f4c951f5c26d2b43aaa390b9af7":
        raise ValueError("released assistant configuration differs")
    if (c.hidden_size, c.num_hidden_layers, c.head_dim, c.num_attention_heads,
        c.num_key_value_heads, c.block_size) != (8192, 5, 64, 128, 16, 7):
        raise ValueError("expected pinned released DSpark geometry")
    proof = json.loads((args.checkpoint / "payload-evidence.json").read_text())
    if proof["revision"] != ASSISTANT_REVISION or not proof["all_weight_payload_sha256_verified"]:
        raise ValueError("expected pinned DSpark payload proof")
    for file in proof["files"]:
        if digest(args.checkpoint / file["file"]) != file["sha256"]:
            raise ValueError("released assistant payload digest differs")
    target_proof = args.target / "payload-evidence.json"
    capture = json.loads((args.captures / "reference-evidence.json").read_text())
    if capture["revision"] != TARGET_REVISION or digest(target_proof) != capture["payload_proof_sha256"]:
        raise ValueError("target payload lineage differs from accepted captures")
    for name, sha in SOURCES.items():
        if digest(args.serving / name) != sha:
            raise ValueError(f"serving mechanism source differs: {name}")
    common = args.serving / "vllm/models/muse_spark/common"
    namespace = dict(torch=torch, nn=nn, F=torch.nn.functional, math=math,
                     MuseSpark1xDecoderLayer=nn.Module, DFlashQwen3Model=nn.Module, VllmConfig=object)
    definitions(common / "layers.py", SOURCE_SHA256, {
        "calculate_gain_center", "MuseSpark1xRMSNorm", "MuseSpark1xRMSNormWithInputScale",
        "MuseSpark1xResidualConnection", "MuseSpark1xNormPreservingResidualConnection"}, namespace)
    definitions(common / "dspark.py", BODY_SHA256,
                {"MuseSparkDSparkDecoderLayer", "MuseSparkDSparkModel"}, namespace)
    norm = namespace["MuseSpark1xRMSNorm"]
    norm.supports_fused_kernel = lambda self, x: False
    model = namespace["MuseSparkDSparkModel"].__new__(namespace["MuseSparkDSparkModel"])
    nn.Module.__init__(model)
    model.default_dtype, model.config = torch.bfloat16, SimpleNamespace(residual_dtype=torch.float32)
    model.fc = nn.Linear(c.hidden_size*c.num_target_layers, c.hidden_size, bias=False, device="meta")
    model.hidden_norm = norm(c.hidden_size, eps=c.rms_norm_eps, zero_centered_gamma=True)
    model.norm = norm(c.hidden_size, eps=c.rms_norm_eps, zero_centered_gamma=True,
                      residual_id=-1, output_norm_gain_center_type="zero")
    layers = []
    for _ in range(c.num_hidden_layers):
        layer = namespace["MuseSparkDSparkDecoderLayer"].__new__(namespace["MuseSparkDSparkDecoderLayer"])
        nn.Module.__init__(layer)
        layer.residual_dtype, layer.default_dtype = torch.float32, torch.bfloat16
        layer.self_attn = Attention(c, norm)
        layer.pre_ffn_norm = norm(c.hidden_size, eps=c.rms_norm_eps, zero_centered_gamma=True)
        layer.post_attn_norm = norm(c.hidden_size, eps=c.rms_post_norm_eps, has_weight=False)
        layer.post_ffn_norm = namespace["MuseSpark1xRMSNormWithInputScale"](
            c.hidden_size, eps=c.rms_post_norm_eps, zero_centered_gamma=True)
        layer.feed_forward = MLP(c)
        layer.add_attn = namespace["MuseSpark1xNormPreservingResidualConnection"](c.hidden_size)
        layer.add_ffn = namespace["MuseSpark1xNormPreservingResidualConnection"](c.hidden_size)
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    def trace_layer(layer, inputs, output):
        layer.published_input, layer.published_output = inputs[1], output[0]
    for layer in layers:
        layer.register_forward_hook(trace_layer)
    heads = args.serving / "vllm/model_executor/models/qwen3_dspark.py"
    definitions(heads, HEAD_SHA256, {"DSparkMarkovHead", "DSparkConfidenceHead"}, namespace := dict(
        torch=torch, nn=nn, ParallelLMHead=ParallelLMHead, ReplicatedLinear=ReplicatedLinear,
        maybe_prefix=lambda prefix, name: f"{prefix}.{name}", QuantizationConfig=object, LogitsProcessor=object))
    tree = ast.parse((common / "dspark.py").read_bytes())
    transform = next(m for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MuseSparkDSparkForCausalLM"
                     for m in n.body if isinstance(m, ast.FunctionDef) and m.name == "apply_output_logit_transform")
    exec(compile(ast.Module(body=[transform], type_ignores=[]), str(common), "exec"), namespace)
    with torch.device("meta"):
        markov = namespace["DSparkMarkovHead"](c.vocab_size, c.draft_vocab_size, c.markov_rank, "markov_head")
        confidence = namespace["DSparkConfidenceHead"](c.hidden_size+c.markov_rank, "confidence_head", bias=True)
    source = args.checkpoint / "model.safetensors"
    requested = (set(model.state_dict()) | {"markov_head."+n for n in markov.state_dict()}
                 | {"confidence_head."+n for n in confidence.state_dict()} | {"lm_head.weight", "d2t"})
    if set(header(source)[1])-{"__metadata__"} != requested:
        raise ValueError("released assistant has missing or unconsumed tensors")
    model.load_state_dict({name: read_tensor(source, name) for name in model.state_dict()}, strict=True, assign=True)
    markov.load_state_dict({name: read_tensor(source, "markov_head."+name) for name in markov.state_dict()}, strict=True, assign=True)
    confidence.load_state_dict({name: read_tensor(source, "confidence_head."+name).float() for name in confidence.state_dict()}, strict=True, assign=True)
    lm_head, mapping = read_tensor(source, "lm_head.weight"), read_tensor(source, "d2t")
    columns = torch.arange(c.draft_vocab_size) + mapping
    if columns.unique().numel() != c.vocab_size or columns.min() != 0 or columns.max() != c.vocab_size-1:
        raise ValueError("released full-vocabulary map is not a bijection")
    model._context_pre_norms = tuple(layer.self_attn.pre_qkv_proj_norm for layer in layers)
    model._stacked_kv_weight = torch.stack([torch.cat((layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight), 0) for layer in layers])
    target_index = json.loads((args.target / "model.safetensors.index.json").read_text())["weight_map"]
    embed_name = "model.language_model.embed_tokens.weight"
    embedding_file = args.target / target_index[embed_name]
    target_files = {r["file"]: r for r in json.loads(target_proof.read_text())["files"]}
    retained_header = json.loads((args.target / (embedding_file.name+".header.json")).read_text())
    if (embedding_file.stat().st_size != target_files[embedding_file.name]["bytes"]
        or embedding_file.stat().st_mtime_ns > target_proof.stat().st_mtime_ns):
        raise ValueError("verified target embedding source changed")
    with embedding_file.open("rb") as source:
        if hashlib.sha256(source.read(retained_header["header_bytes"])).hexdigest() != retained_header["header_sha256"]:
            raise ValueError("target embedding header differs from admitted source")
    def embedding(ids):
        return torch.cat([read_tensor(embedding_file, embed_name, i) for i in ids])
    outputs, prefixes, end = [], [], 0
    for step, target_step in enumerate(capture["steps"]):
        selected = [next(r for r in capture["layer_traces"] if r["step"] == step and r["layer"] == layer)
                    for layer in c.target_layer_ids]
        pending = torch.cat([read_trace(args.captures, r) for r in selected], -1)[0]
        prefixes.append(pending)
        end += pending.shape[0]
        encoded_input = model.fc(torch.cat(prefixes).to(torch.bfloat16))
        projected = model._project_muse_context_kv(encoded_input)
        encoded = model.hidden_norm(encoded_input)
        start = max(0, end-c.sliding_window)
        context_layers = []
        for i, layer in enumerate(layers):
            raw_k, raw_v = projected[i, start:end].split(c.num_key_value_heads*c.head_dim, -1)
            k = rope(head_norm(raw_k.reshape(-1, c.num_key_value_heads, c.head_dim).transpose(0, 1)), torch.arange(start, end))
            v = raw_v.reshape(-1, c.num_key_value_heads, c.head_dim).transpose(0, 1)
            layer.self_attn.committed, layer.self_attn.context_start = (k, v), start
            context_layers.append(dict(keys=emit(args.output, f"step-{step}-kv-{i}-k.f32", k[None]),
                                       values=emit(args.output, f"step-{step}-kv-{i}-v.f32", v[None])))
        noise = embedding([target_step["input_ids"][-1]]+[c.mask_token_id]*(c.block_size-1))
        hidden = model(input_ids=torch.zeros(c.block_size, dtype=torch.long),
                       positions=torch.arange(end, end+c.block_size), input_embeds=noise)
        if step == 0:
            attention = layers[0].self_attn
            # This released head detects scaling Q before its FP32 dot product:
            # the small F32 difference crosses a BF16 store boundary.
            q, k, v = attention.queries[4:5][None], attention.keys[0:1][None], attention.values[0:1][None]
            mask = torch.arange(end+c.block_size)[None] <= torch.arange(end, end+c.block_size)[:, None]
            scale = (43.7840518911/math.sqrt(c.head_dim))/math.sqrt(c.head_dim)
            scores = ((q.float() @ k.float().transpose(-2, -1))*scale).masked_fill(~mask, float("-inf"))
            float_output = torch.softmax(scores, -1) @ v.float()
            output = float_output.to(torch.bfloat16)
            with_sink = torch.softmax(torch.cat((scores, torch.zeros((*scores.shape[:-1], 1))), -1), -1)[..., :-1]
            case = dict(dtype="bfloat16", queries=record(q), keys=record(k), values=record(v), mask=record(mask[None, None]),
                        scale=scale, sinks=record(torch.zeros(1)), output=record(output),
                        float_output=record(float_output), float_sink_output=record(with_sink @ v.float()),
                        sink_output=record((with_sink @ v.float()).to(torch.bfloat16)))
            fixture = dict(repository=proof["repository"], revision=ASSISTANT_REVISION, serving_revision=REVISION,
                           source_sha256=BODY_SHA256, target_capture_evidence_sha256=digest(args.captures / "reference-evidence.json"),
                           runner_sha256=digest(Path(__file__)), step=0, layer=0, query_head=4, kv_head=0, attention=[case])
            (args.output / "released-attention.json").write_text(json.dumps(fixture, indent=2)+"\n")
        traces = []
        for i, layer in enumerate(layers):
            traces.append({name: emit(args.output, f"step-{step}-layer-{i}-{name.replace('.', '-')}.f32", value[None])
                           for name, value in [("input", layer.published_input),
                              ("attention.queries", layer.self_attn.queries),
                              ("attention.attended_heads", layer.self_attn.attended),
                              ("output", layer.published_output)]})
        raw = nn.functional.linear(hidden.to(lm_head.dtype), lm_head)
        rows, previous = [], target_step["input_ids"][-1]
        for row in range(c.block_size):
            embedded = markov.embed(torch.tensor([previous]))
            bias = markov.bias(embedded, lambda module, x: nn.functional.linear(x, module.weight))
            transformed = namespace["apply_output_logit_transform"](SimpleNamespace(
                output_multiplier=(c.hidden_size/c.metap["base_width"])**-0.5,
                output_soft_cap_temp=c.output_soft_cap_temp), raw[row:row+1]+bias)
            scores = torch.full_like(transformed, float("-inf"))
            scores[:, columns] = transformed
            conf = torch.sigmoid(confidence(hidden[row:row+1], embedded))
            top1 = int(scores.argmax(-1))
            rows.append(dict(previous=previous, top1=top1,
                             scores=emit(args.output, f"step-{step}-row-{row}.f32", scores),
                             confidence=emit(args.output, f"step-{step}-confidence-{row}.f32", conf)))
            previous = top1
        outputs.append(dict(end=end, pending=emit(args.output, f"step-{step}-pending.f32", pending[None]),
                            embeddings=emit(args.output, f"step-{step}-embeddings.f32", noise[None]),
                            encoded=emit(args.output, f"step-{step}-encoded.f32", encoded[start:end][None]),
                            layers=context_layers, hidden=emit(args.output, f"step-{step}-hidden.f32", hidden[None]),
                            embedding_ids=[target_step["input_ids"][-1]]+[c.mask_token_id]*(c.block_size-1), traces=traces, rows=rows))
        print(f"released DSpark step {step}: context={end}, proposals={c.block_size}, top1={[r['top1'] for r in rows]}", flush=True)
    report = dict(schema_version=1, repository=proof["repository"], revision=ASSISTANT_REVISION,
                  serving_revision=REVISION, body_sha256=BODY_SHA256, normalization_sha256=SOURCE_SHA256,
                  heads_sha256=HEAD_SHA256, runner_sha256=digest(Path(__file__)), torch_version=torch.__version__,
                  config_sha256=digest(args.checkpoint / "config.json"), payload_proof_sha256=digest(args.checkpoint / "payload-evidence.json"),
                  target_revision=TARGET_REVISION, target_capture_evidence_sha256=digest(args.captures / "reference-evidence.json"),
                  inputs="accepted full target layer outputs; actual BF16 last-input-token and mask embedding rows",
                  comparison_kind="released neural body/head composition; no sampling, verification, control or distributed session claim",
                  native_comparison_verified=False, elapsed_seconds=time.monotonic()-started, steps=outputs)
    (args.output / "reference-evidence.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
