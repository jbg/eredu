"""Released DFlash2 BF16 body/selector against pinned SGLang eager equations.

Published block, convolution and selector definitions execute unchanged; CPU
PyTorch projections, typed RMS/RoPE and FP32 attention replace serving kernels.
Inputs reuse accepted target captures and actual target embedding/head weights.
This establishes neural component evidence, not speculative session behavior.
"""
import argparse
import ast
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
from torch import nn

REVISION = "4cca7fcd3e644f22a61c7a6e6b57bb89b568d62a"
SOURCE_SHA256 = "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff"
from muse_spark_dspark_released_reference import (
    TARGET_REVISION, digest, emit, header, read_tensor, read_trace,
)

ASSISTANT_REVISION = "46c90f412549c472941a68382d91a99e5e6372a0"


class RMS(nn.Module):
    reference_forward = None
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device="meta"))
        self.hidden_size, self.variance_epsilon = width, eps
        self.override_orig_dtype = self.variance_size_override = None
        self.fp32_residual = self.cast_x_before_out_mul = False

    def forward(self, x, residual=None):
        return self.reference_forward(x, residual)


def rope(x, positions, base):
    frequency = base ** (-torch.arange(0, x.shape[-1], 2).float() / x.shape[-1])
    phase = positions.float()[:, None] * frequency[None]
    # Execute the serving CPU eager helper, including its typed table/products.
    return rope.reference_apply(x.transpose(0, 1), phase.cos(), phase.sin(), True).transpose(0, 1)


class Attention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = c = config
        for name, width in [("q_proj", c.num_attention_heads*c.head_dim),
                            ("k_proj", c.num_key_value_heads*c.head_dim),
                            ("v_proj", c.num_key_value_heads*c.head_dim)]:
            setattr(self, name, nn.Linear(c.hidden_size, width, bias=False, device="meta"))
        self.o_proj = nn.Linear(c.num_attention_heads*c.head_dim, c.hidden_size, bias=False, device="meta")
        self.q_norm, self.k_norm = RMS(c.head_dim, c.rms_norm_eps), RMS(c.head_dim, c.rms_norm_eps)

    def project_context(self, x, positions):
        c = self.config
        k = self.k_norm(self.k_proj(x).reshape(-1, c.num_key_value_heads, c.head_dim)).transpose(0, 1)
        v = self.v_proj(x).reshape(-1, c.num_key_value_heads, c.head_dim).transpose(0, 1)
        return rope(k, positions, c.rope_theta), v

    def forward(self, positions, hidden_states, forward_batch):
        c = self.config
        q = self.q_norm(self.q_proj(hidden_states).reshape(-1, c.num_attention_heads, c.head_dim)).transpose(0, 1)
        q = rope(q, positions, c.rope_theta)
        k, v = self.project_context(hidden_states, positions)
        k = torch.cat((self.committed[0], k), 1).repeat_interleave(c.num_attention_heads//c.num_key_value_heads, 0)
        v = torch.cat((self.committed[1], v), 1).repeat_interleave(c.num_attention_heads//c.num_key_value_heads, 0)
        key_positions = torch.arange(self.context_start, int(positions[-1])+1)
        allowed = (key_positions[None] >= self.context_end) | (positions[:, None]-key_positions[None] < c.sliding_window)
        scores = ((q.float() @ k.float().transpose(-2, -1)) * c.head_dim**-0.5).masked_fill(~allowed[None], float("-inf"))
        attended = (torch.softmax(scores, -1) @ v.float()).to(hidden_states.dtype)
        self.queries, self.attended = q, attended
        return self.o_proj(attended.transpose(0, 1).reshape(-1, c.num_attention_heads*c.head_dim))


class MLP(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device="meta")
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device="meta")
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device="meta")

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def checked_target_file(root, name, proof):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    path = root / index[name]
    row = next(r for r in proof["files"] if r["file"] == path.name)
    if path.stat().st_size != row["bytes"] or path.stat().st_mtime_ns > (root / "payload-evidence.json").stat().st_mtime_ns:
        raise ValueError("verified target component source changed")
    retained = json.loads((root / (path.name+".header.json")).read_text())
    import hashlib
    with path.open("rb") as source:
        if hashlib.sha256(source.read(retained["header_bytes"])).hexdigest() != retained["header_sha256"]:
            raise ValueError("target component header differs from accepted source")
    return path


def build_model(c, d, namespace):
    model = nn.Module()
    with torch.device("meta"):
        model.fc = nn.Linear(c.hidden_size*len(d.target_layer_ids), c.hidden_size, bias=False)
        model.hidden_norm, model.norm = RMS(c.hidden_size, c.rms_norm_eps), RMS(c.hidden_size, c.rms_norm_eps)
        conv = lambda: namespace["DFlashGroupedConv"](c.hidden_size, d.block_size, d.conv_kernel_size, d.conv_group_size)
        model.layers = nn.ModuleList(namespace["DFlashDecoderLayer"](c, i, attention_conv=conv(), mlp_conv=conv())
                                     for i in range(c.num_hidden_layers))
        model.candidate_selector = namespace["CandidateSelector"](hidden_size=c.hidden_size, vocab_size=c.vocab_size,
                                      state_rank=d.selector_rank, top_k=d.selector_top_k)
    return model




def audit_native(args, reference):
    """Compare retained native publications; execute no replacement neural body.

    The original native report remains immutable, including its failure against
    the unspecified CPU fallback tie order. Every actual binary must match that
    report's hash, and teacher-forced predecessors must be identical.
    """
    native_path = args.audit_native / "native-comparison.json"
    native = json.loads(native_path.read_text())
    origin = json.loads((args.native_reference / "reference-evidence.json").read_text())
    if native["reference_evidence_sha256"] != digest(args.native_reference / "reference-evidence.json"):
        raise ValueError("native output lineage differs from its original reference")
    if native["payload_proof_sha256"] != reference["payload_proof_sha256"] or len(native["steps"]) != 4:
        raise ValueError("native released source or complete update inventory differs")
    atol, rtol = native["atol"], native["rtol"]
    records, passed = [], True
    for actual_step, expected, original in zip(native["steps"], reference["steps"], origin["steps"], strict=True):
        expected_records = [expected["encoded"], expected["hidden"], expected["raw"]]
        expected_records += [r for layer in expected["layers"] for r in layer.values()]
        expected_records += [r for layer in expected["traces"] for r in layer.values()]
        expected_records += [r["scores"] for r in expected["rows"]]
        actual_records = {r["file"]: r for r in actual_step["comparisons"]}
        if len(actual_records) != 40 or {r["file"] for r in expected_records} != set(actual_records):
            raise ValueError("native publication inventory is incomplete or duplicated")
        comparisons = []
        for record in expected_records:
            actual_record = actual_records[record["file"]]
            if actual_record["shape"] != record["shape"]:
                raise ValueError("native publication geometry differs")
            a = read_trace(args.audit_native, actual_record).flatten()
            b = read_trace(args.output, record).flatten()
            if (a.numel() != math.prod(record["shape"]) or a.numel() != b.numel()
                or torch.isnan(a).any() or torch.isnan(b).any() or torch.isposinf(a).any() or torch.isposinf(b).any()):
                raise ValueError("native publication extent or values differ")
            finite = torch.isfinite(a) & torch.isfinite(b)
            support = int((torch.isneginf(a) != torch.isneginf(b)).sum())
            delta = (a[finite].double()-b[finite].double()).abs()
            outside = int((delta > atol+rtol*b[finite].double().abs()).sum())+support
            passed &= outside == 0
            comparisons.append(dict(file=record["file"], shape=record["shape"], sha256=actual_record["sha256"],
                reference_sha256=record["sha256"], max_abs=float(delta.max()) if delta.numel() else 0.,
                rms=float(delta.square().mean().sqrt()) if delta.numel() else 0., support_mismatches=support, outside_tolerance=outside))
        candidates = [i for batch in expected["candidate_ids"] for row in batch for i in row]
        if actual_step["candidate_ids"] != candidates:
            raise ValueError("native candidate IDs differ from the explicit portable tie policy")
        top1, fallback_rows = [], []
        for row, old in zip(expected["rows"], original["rows"], strict=True):
            if row["previous"] != old["previous"]:
                raise ValueError("native conditional predecessor differs; fresh neural scoring is required")
            values = read_trace(args.audit_native, actual_records[row["scores"]["file"]])
            best = int(values.argmax(-1))
            passed &= best == row["top1"]
            top1.append(best)
            fallback = read_trace(args.output, row["fallback_scores"])
            shared = torch.isfinite(values) & torch.isfinite(fallback)
            fallback_rows.append(dict(previous=row["previous"], native_top1=best, fallback_top1=row["fallback_top1"],
                support_mismatches=int((torch.isneginf(values) != torch.isneginf(fallback)).sum()),
                max_shared_abs=float((values[shared]-fallback[shared]).abs().max())))
        passed &= actual_step["top1"] == top1
        records.append(dict(step=actual_step["step"], top1=top1, candidate_ids_verified=True,
            original_predecessors_verified=True, fallback_membership_differences=expected["fallback_membership_differences"],
            fallback_ordered_id_differences=expected["fallback_ordered_id_differences"],
            published_cpu_fallback_comparisons=fallback_rows, comparisons=comparisons))
    report = dict(schema_version=1, comparison_verified=passed, repository=reference["repository"], revision=reference["revision"],
        native_report_sha256=digest(native_path), native_original_reference_sha256=digest(args.native_reference / "reference-evidence.json"),
        reference_evidence_sha256=digest(args.output / "reference-evidence.json"), runner_sha256=digest(Path(__file__)),
        atol=atol, rtol=rtol, selected_candidate_policy=reference["topk"],
        published_cpu_fallback_candidate_identity_verified=False, retained_native_binaries_verified=True,
        neural_reexecution=False, scope="retained full released native neural publications compared with the explicit portable tie policy; original CPU-fallback failure preserved", steps=records)
    (args.output / "native-comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    if not passed:
        raise ValueError("retained native neural comparison differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "target", "captures", "serving", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--audit-native", type=Path, help="audit immutable native publications against the explicit candidate tie policy")
    parser.add_argument("--native-reference", type=Path, help="original reference used by that native run")
    args = parser.parse_args()
    if torch.__version__ != "2.9.0":
        raise ValueError("requires torch==2.9.0")
    torch.set_num_threads(8)
    started = time.monotonic()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("released execution outputs belong outside tracked source")
    args.output.mkdir(parents=True, exist_ok=True)
    pinned = Path(__file__).parent / "fixtures/muse-spark/dflash2-config.json"
    if digest(args.checkpoint / "config.json") != digest(pinned):
        raise ValueError("released assistant configuration differs")
    c = SimpleNamespace(**json.loads(pinned.read_text()))
    d = SimpleNamespace(**c.dflash_config)
    proof = json.loads((args.checkpoint / "payload-evidence.json").read_text())
    if proof["revision"] != ASSISTANT_REVISION or not proof["all_weight_payload_sha256_verified"]:
        raise ValueError("requires pinned assistant payload proof")
    for file in proof["files"]:
        if digest(args.checkpoint / file["file"]) != file["sha256"]:
            raise ValueError("released assistant payload digest differs")
    capture = json.loads((args.captures / "reference-evidence.json").read_text())
    if capture["revision"] != TARGET_REVISION or digest(args.target / "payload-evidence.json") != capture["payload_proof_sha256"]:
        raise ValueError("target capture payload lineage differs")
    source = args.serving / "python/sglang/srt/models/dflash.py"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("pinned serving source differs")
    mechanisms = {
        "python/sglang/srt/layers/layernorm.py": "98072d836e0c77312058521ee1015c80177b2c370da7408fd066eec61015ff21",
        "python/sglang/srt/layers/rotary_embedding/base.py": "bf024728ee2932583ea609f852d9bd29b75005584c4ec36d46bb686273eb0da8",
        "python/sglang/srt/layers/rotary_embedding/utils.py": "828a50285218e40c3429d2f4e75b1d051e3a89d831a7ab005e8c2609278aeddc",
    }
    for name, sha in mechanisms.items():
        if digest(args.serving / name) != sha:
            raise ValueError("pinned serving operator differs")
    norm_tree = ast.parse((args.serving / "python/sglang/srt/layers/layernorm.py").read_bytes())
    norm_class = next(n for n in norm_tree.body if isinstance(n, ast.ClassDef) and n.name == "RMSNorm")
    norm_forward = next(n for n in norm_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward_native")
    ns = dict(torch=torch, Optional=Optional, Union=__import__("typing").Union, Tuple=Tuple, nn=nn)
    exec(compile(ast.Module(body=[norm_forward], type_ignores=[]), str(source), "exec"), ns)
    RMS.reference_forward = ns["forward_native"]
    rotary_tree = ast.parse((args.serving / "python/sglang/srt/layers/rotary_embedding/utils.py").read_bytes())
    apply = next(n for n in rotary_tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_rotary_emb")
    exec(compile(ast.Module(body=[apply], type_ignores=[]), str(source), "exec"), ns)
    rope.reference_apply = ns["apply_rotary_emb"]
    tree = ast.parse(source.read_bytes())
    names = {"_grouped_conv", "DFlashGroupedConv", "DFlashDecoderLayer", "_score_edges", "_follow_maps", "CandidateSelector"}
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if len(definitions) != len(names):
        raise ValueError("incomplete pinned serving definitions")
    for node in definitions:
        node.decorator_list = []
    namespace = dict(torch=torch, nn=nn, F=torch.nn.functional, RMSNorm=RMS, DFlashAttention=Attention,
                     DFlashMLP=MLP, Optional=Optional, Tuple=Tuple, ForwardBatch=object,
                     _flashinfer_top_k=None, logger=SimpleNamespace(warning=lambda *args: None))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlash2DraftModel")
    transform = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_transform_unary_logits")
    exec(compile(ast.Module(body=[transform], type_ignores=[]), str(source), "exec"), namespace)
    model = build_model(c, d, namespace)
    payload = args.checkpoint / "model.safetensors"
    if set(model.state_dict()) != set(header(payload)[1]) - {"__metadata__"}:
        raise ValueError("released assistant has missing or unconsumed parameters")
    model.load_state_dict({name: read_tensor(payload, name) for name in model.state_dict()}, strict=True, assign=True)
    def publish(layer, inputs, output):
        layer.published_input = inputs[1] if inputs[3] is None else inputs[1] + inputs[3]
        layer.published_output = output[0] + output[1]
    for layer in model.layers:
        layer.register_forward_hook(publish)
    target_proof = json.loads((args.target / "payload-evidence.json").read_text())
    embed_name = "model.language_model.embed_tokens.weight"
    embedding_file = checked_target_file(args.target, embed_name, target_proof)
    head_file = checked_target_file(args.target, "lm_head.weight", target_proof)
    lm_head = read_tensor(head_file, "lm_head.weight")
    outputs, prefixes, end = [], [], 0
    for step, target_step in enumerate(capture["steps"]):
        selected = [next(r for r in capture["layer_traces"] if r["step"] == step and r["layer"] == layer)
                    for layer in d.target_layer_ids]
        pending = torch.cat([read_trace(args.captures, r) for r in selected], -1)[0]
        prefixes.append(pending)
        end += pending.shape[0]
        encoded = model.hidden_norm(model.fc(torch.cat(prefixes).to(torch.bfloat16)))
        start = max(0, end-c.sliding_window)
        contexts = []
        for i, layer in enumerate(model.layers):
            a = layer.self_attn
            a.committed = a.project_context(encoded[start:end], torch.arange(start, end))
            a.context_start, a.context_end = start, end
            contexts.append(dict(keys=emit(args.output, f"step-{step}-kv-{i}-k.f32", a.committed[0][None]),
                                 values=emit(args.output, f"step-{step}-kv-{i}-v.f32", a.committed[1][None])))
        ids = [target_step["input_ids"][-1]] + [d.mask_token_id]*(d.block_size-1)
        noise = torch.cat([read_tensor(embedding_file, embed_name, i) for i in ids])
        hidden, residual = noise, None
        for layer in model.layers:
            hidden, residual = layer(torch.arange(end, end+d.block_size), hidden, None, residual)
        hidden, _ = model.norm(hidden, residual)
        hidden = hidden[None, 1:]
        raw = nn.functional.linear(hidden.to(lm_head.dtype), lm_head)
        # The portable ranked_topk contract breaks ties by ascending token ID.
        # Keep the published eager fallback separately; torch.topk does not
        # guarantee tie order and can choose another member at the cutoff.
        fallback_values, fallback_ids = torch.topk(raw, d.selector_top_k, dim=-1)
        candidates = raw.argsort(dim=-1, descending=True, stable=True)[..., :d.selector_top_k]
        selected = raw.gather(-1, candidates)
        if not torch.equal(selected, fallback_values):
            raise ValueError("stable selection changes a non-tied candidate score")
        boundary_ties = [(row == row.topk(d.selector_top_k).values[-1]).sum().item() for row in raw[0]]
        unary = namespace["_transform_unary_logits"](SimpleNamespace(draft_config=d), selected)
        fallback_unary = namespace["_transform_unary_logits"](SimpleNamespace(draft_config=d), fallback_values)
        selector = model.candidate_selector
        anchor = torch.tensor([ids[0]])
        lattice = selector.build_lattice(candidate_ids=candidates, unary_logits=unary, hidden_states=hidden, anchor_token_ids=anchor)
        rows, predecessor, predecessor_index = [], ids[0], 0
        for row in range(d.block_size-1):
            conditional = lattice[:, row, predecessor_index].float()
            dense = torch.full((1, c.vocab_size), float("-inf"))
            dense.scatter_(1, candidates[:, row], conditional)
            best = int(dense.argmax(-1))
            # Use the same actual predecessor for the fallback comparison, even
            # if its preceding candidate support differs from the chosen policy.
            projected = selector.hidden_projection(hidden[:, row])
            fallback_conditional = fallback_unary[:, row] + torch.einsum("br,bcr->bc",
                selector.predecessor_codebook[torch.tensor([predecessor])]*projected,
                selector.successor_codebook[fallback_ids[:, row]])
            fallback_dense = torch.full_like(dense, float("-inf"))
            fallback_dense.scatter_(1, fallback_ids[:, row], fallback_conditional.float())
            rows.append(dict(previous=predecessor, top1=best, scores=emit(args.output, f"step-{step}-row-{row}.f32", dense),
                fallback_top1=int(fallback_dense.argmax(-1)), fallback_scores=emit(args.output, f"step-{step}-fallback-row-{row}.f32", fallback_dense)))
            predecessor_index = int((candidates[0, row] == best).nonzero()[0])
            predecessor = best
        traces = [{name: emit(args.output, f"step-{step}-layer-{i}-{name.replace('.', '-')}.f32", value[None])
                   for name, value in [("input", layer.published_input), ("attention.queries", layer.self_attn.queries),
                                       ("attention.attended_heads", layer.self_attn.attended), ("output", layer.published_output)]}
                  for i, layer in enumerate(model.layers)]
        outputs.append(dict(end=end, pending=emit(args.output, f"step-{step}-pending.f32", pending[None]),
            embeddings=emit(args.output, f"step-{step}-embeddings.f32", noise[None]), encoded=emit(args.output, f"step-{step}-encoded.f32", encoded[start:end][None]),
            layers=contexts, hidden=emit(args.output, f"step-{step}-hidden.f32", hidden), raw=emit(args.output, f"step-{step}-raw.f32", raw),
            candidate_ids=candidates.tolist(), fallback_candidate_ids=fallback_ids.tolist(),
            fallback_membership_differences=sum(len(set(a)-set(b)) for a, b in zip(candidates[0].tolist(), fallback_ids[0].tolist())),
            fallback_ordered_id_differences=int((candidates != fallback_ids).sum()),
            selected_raw_values_identical_to_fallback=True,
            boundary_tie_counts=boundary_ties, embedding_ids=ids, traces=traces, rows=rows))
        print(f"released DFlash2 step {step}: context={end}, top1={[r['top1'] for r in rows]}, boundary_ties={boundary_ties}", flush=True)
    report = dict(schema_version=1, repository=proof["repository"], revision=ASSISTANT_REVISION, serving_revision=REVISION,
        source_sha256=SOURCE_SHA256, operator_source_sha256=mechanisms, runner_sha256=digest(Path(__file__)), torch_version=torch.__version__,
        config_sha256=digest(args.checkpoint / "config.json"), payload_proof_sha256=digest(args.checkpoint / "payload-evidence.json"),
        target_revision=TARGET_REVISION, target_capture_evidence_sha256=digest(args.captures / "reference-evidence.json"),
        target_payload_proof_sha256=digest(args.target / "payload-evidence.json"),
        target_head=dict(name="lm_head.weight", file=head_file.name, shape=list(lm_head.shape), dtype="bfloat16",
                        header=json.loads((args.target / (head_file.name+".header.json")).read_text())["header_sha256"]),
        comparison_kind="released BF16 neural body/selector composition; no speculative session claim",
        topk="Eredu portable ranked_topk: descending raw score, ascending token ID; independent stable torch.argsort; published torch.topk fallback IDs and same-predecessor distributions retained separately",
        elapsed_seconds=time.monotonic()-started, steps=outputs)
    (args.output / "reference-evidence.json").write_text(json.dumps(report, indent=2)+"\n")
    if args.audit_native:
        if not args.native_reference:
            raise ValueError("native audit requires its original reference")
        audit_native(args, report)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
