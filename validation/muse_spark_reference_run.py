"""Run pinned Transformers text or multimodal prefill and cached decoding over official NVFP4.

Dense text parameters stay resident. Expert banks remain packed on disk; the
unchanged eager Transformers expert forward indexes a validation-only accessor
which independently decodes one expert with pinned NVIDIA ModelOpt. No full bank
or checkpoint is expanded. Outputs are evidence for comparison, not an Eredu
parity claim. Python 3.11, torch==2.9.0, numpy==2.3.4 are required.
"""

import argparse
import ast
import hashlib
import json
import mmap
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from muse_spark_reference_source import load_reference, REVISION as TRANSFORMERS_REVISION
from nvfp4_reference import load_reference as load_nvfp4, REVISION as MODELOPT_REVISION


REPOSITORY = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
REVISION = "4239e4db19376904f2e82c167e78f70abb5455fe"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
          "F8_E4M3": torch.float8_e4m3fn, "F8_E4M3FN": torch.float8_e4m3fn}


class Source:
    def __init__(self, directory, decoder, preflight):
        self.directory = directory
        evidence = json.loads((directory / "evidence.json").read_text())
        if (evidence["repository"], evidence["revision"]) != (REPOSITORY, REVISION):
            raise ValueError("reference execution requires the pinned official NVFP4 source")
        self.headers, self.tensors = {}, {}
        for item in evidence["headers"]:
            header = json.loads((directory / (item["file"] + ".header.json")).read_text())
            if any(header[key] != item[key] for key in item):
                raise ValueError("retained header differs from its retrieval evidence")
            self.headers[item["file"]] = header
            for name, metadata in header["tensors"].items():
                if name in self.tensors:
                    raise ValueError("duplicate source tensor")
                self.tensors[name] = (header, metadata)
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        if {key: header["file"] for key, (header, _) in self.tensors.items()} != index:
            raise ValueError("tensor catalog differs from the pinned shard index")
        if not preflight:
            proof_path = directory / "payload-evidence.json"
            proof = json.loads(proof_path.read_text())
            if (proof["repository"], proof["revision"]) != (REPOSITORY, REVISION) or not proof["all_weight_payload_sha256_verified"]:
                raise ValueError("complete official payload verification is required")
            files = {item["file"]: item for item in proof["files"]}
            inventory = json.loads((Path(__file__).parent / "fixtures/muse-spark/inventory.json").read_text())
            pinned = next(item for item in inventory["artifacts"] if item["repository"] == REPOSITORY and item["revision"] == REVISION)
            if files.keys() != self.headers.keys() or not all(item["verified"] for item in files.values()):
                raise ValueError("payload verification does not cover the complete catalog")
            for name, header in self.headers.items():
                actual = (directory / name).stat()
                if files[name]["sha256"] != pinned["files"][name]["sha256"]:
                    raise ValueError("payload verification differs from the frozen publisher hash")
                if files[name]["bytes"] != header["bytes"] or actual.st_size != header["bytes"] or actual.st_mtime_ns > proof_path.stat().st_mtime_ns:
                    raise ValueError("verified payload length changed")
            self.payload_proof_sha256 = hashlib.sha256((directory / "payload-evidence.json").read_bytes()).hexdigest()
        self.decoder = decoder
        self.checked = set()
        self.read_bytes = 0
        self.maximum_decoded_expert_bytes = 0

    def read(self, name, expert=None):
        header, metadata = self.tensors[name]
        path = self.directory / header["file"]
        if header["file"] not in self.checked:
            with path.open("rb") as source:
                actual = source.read(header["header_bytes"])
            if hashlib.sha256(actual).hexdigest() != header["header_sha256"]:
                raise ValueError("local payload header differs from its pinned source")
            self.checked.add(header["file"])
        shape = metadata["shape"]
        first, end = metadata["data_offsets"]
        if expert is not None:
            if not shape or not 0 <= expert < shape[0] or (end - first) % shape[0]:
                raise ValueError("expert selection is outside its source bank")
            length = (end - first) // shape[0]
            first += expert * length
            end = first + length
            shape = shape[1:]
        start = header["header_bytes"] + first
        length = end - first
        aligned = start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        # Clone before closing the mapping; live tensor storage never borrows it.
        with path.open("rb") as source, mmap.mmap(source.fileno(), start - aligned + length,
                access=mmap.ACCESS_READ, offset=aligned) as mapped:
            borrowed = torch.frombuffer(mapped, dtype=DTYPES[metadata["dtype"]], offset=start - aligned)
            result = borrowed.clone().reshape(shape)
            del borrowed
        self.read_bytes += length
        return result

    def expert(self, name, index, expected_shape):
        _, metadata = self.tensors[name]
        if metadata["dtype"] == "BF16":
            result = self.read(name, index)
        elif metadata["dtype"] == "U8":
            packed = self.read(name, index)
            scales = self.read(name + "_weight_scale", index)
            globals_ = self.read(name + "_weight_scale_2", index).flatten()
            rows, half_columns = packed.shape
            if scales.shape != (rows, half_columns // 8) or globals_.numel() not in (1, 2) or rows % globals_.numel():
                raise ValueError("NVFP4 expert companions have inconsistent geometry")
            if not torch.isfinite(globals_).all() or not (globals_ > 0).all():
                raise ValueError("NVFP4 global scales must be finite and positive")
            scale_values = scales.float()
            if not torch.isfinite(scale_values).all() or not (scale_values >= 0).all():
                raise ValueError("NVFP4 block scales must be finite and nonnegative")
            global_rows = globals_.repeat_interleave(rows // globals_.numel()).reshape(rows, 1)
            # NVIDIA's pinned eager implementation supplies the actual decoding.
            result = self.decoder(packed).dequantize(dtype=torch.bfloat16, scale=scales,
                double_scale=global_rows, block_sizes={-1: 16})
            self.maximum_decoded_expert_bytes = max(self.maximum_decoded_expert_bytes, result.numel() * 2)
        else:
            raise ValueError("unsupported expert encoding in the official reference source")
        if tuple(result.shape) != tuple(expected_shape):
            raise ValueError("decoded expert geometry differs from the pinned Transformers module")
        return result


class ExpertAccessor:
    """Parameter-free lazy indexing seam; the reference expert equation is unchanged."""
    def __init__(self, source, name, shape):
        self.source, self.name, self.shape = source, name, tuple(shape)

    def __getitem__(self, index):
        return self.source.expert(self.name, int(index), self.shape[1:])


class Cache:
    def __init__(self):
        self.layers = {}

    def get_seq_length(self):
        return self.layers[0][0].shape[2] if 0 in self.layers else 0

    def update(self, keys, values, layer):
        if layer in self.layers:
            previous = self.layers[layer]
            keys, values = torch.cat((previous[0], keys), dim=2), torch.cat((previous[1], values), dim=2)
        self.layers[layer] = (keys, values)
        return keys, values


def bind(module, name, value):
    path, _, leaf = name.rpartition(".")
    owner = module.get_submodule(path) if path else module
    if leaf in owner._parameters:
        owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    else:
        owner._buffers[leaf] = value


def masks(start, length, window):
    query = torch.arange(start, start + length)[:, None]
    key = torch.arange(start + length)[None, :]
    causal = key <= query
    def additive(visible):
        return torch.zeros(visible.shape, dtype=torch.bfloat16).masked_fill(~visible, float("-inf"))[None, None]
    return {"full_attention": additive(causal), "sliding_attention": additive(causal & (key > query - window))}



def multimodal_reference(args, namespace, raw, text_model, source):
    """Bind the unchanged parent model, including its actual scatter and norms."""
    from muse_spark_reference_source import VISION_UTILS_SHA256
    if hashlib.sha256(args.vision_utils.read_bytes()).hexdigest() != VISION_UTILS_SHA256:
        raise ValueError("vision helpers differ from the pinned reference")
    helpers = {"get_vision_cu_seqlens", "get_vision_position_ids", "get_vision_window_index",
               "_interpolation_axis_taps_weights", "get_vision_interpolation_indices_and_weights"}
    definitions = [node for node in ast.parse(args.vision_utils.read_bytes()).body
                   if isinstance(node, ast.FunctionDef) and node.name in helpers]
    if len(definitions) != len(helpers):
        raise ValueError("incomplete vision helper inventory")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(args.vision_utils), "exec"), namespace)
    def output(**kwargs):
        for name in ("hidden_states", "attentions", "router_logits"):
            kwargs.setdefault(name, None)
        return SimpleNamespace(**kwargs)
    namespace.update(is_flash_attention_requested=lambda config: False,
                     BaseModelOutputWithPooling=output, BaseModelOutput=output,
                     MoeModelOutputWithPast=output, MuseSparkModelOutputWithPast=output)
    config = SimpleNamespace(**{key: value for key, value in raw.items()
                                if key not in ("text_config", "vision_config", "audio_config")})
    config.text_config = text_model.config
    config.vision_config = SimpleNamespace(**raw["vision_config"], _attn_implementation="eager")
    config.audio_config = SimpleNamespace(**raw["audio_config"])
    cadence = 1
    for patch in config.audio_config.patch_sizes:
        cadence *= patch
    if cadence != 1920:
        raise ValueError("released waveform cadence differs")
    config.audio_config.samples_per_token = cadence
    # The pinned configuration class supplies the omitted video-token default.
    configuration_sha256 = "c955ee47b396d4a5a5be8457ee85b4ac0639705beea18960eb06910d2d6e4e1b"
    configuration = args.configuration.read_bytes()
    if hashlib.sha256(configuration).hexdigest() != configuration_sha256:
        raise ValueError("configuration source differs from the pinned reference")
    config_class = next(node for node in ast.parse(configuration).body
                       if isinstance(node, ast.ClassDef) and node.name == "MuseSparkConfig")
    video_default = next(ast.literal_eval(node.value) for node in config_class.body
                         if isinstance(node, ast.AnnAssign) and node.target.id == "video_token_id")
    config.video_token_id = raw.get("video_token_id", video_default)
    def construct(child):
        if child is config.text_config:
            return text_model
        name = "MuseSparkVisionModel" if child is config.vision_config else "MuseSparkAudioModel"
        return namespace[name](child)
    namespace["AutoModel"] = SimpleNamespace(from_config=construct)
    with torch.device("meta"):
        parent = namespace["MuseSparkModel"](config).eval()
    parent.vision_tower.rotary_emb = namespace["MuseSparkVisionRotaryEmbedding"](config.vision_config)
    for name in list(parent.state_dict()):
        if not name.startswith("language_model."):
            value = source.read("model." + name)
            if value.dtype != torch.bfloat16:
                raise ValueError("released media parameters must retain BF16 storage precision")
            bind(parent, name, value)
    width = 3 * config.vision_config.patch_temporal * config.vision_config.patch_size ** 2
    pixels = 0.8 * torch.cos(torch.arange(16 * width, dtype=torch.float32).reshape(16, width) * 0.021 + 0.3)
    positions = torch.arange(3840, dtype=torch.float32)
    waveform = (0.3 * torch.sin(positions * 0.023) + 0.17 * torch.cos(positions * 0.071))[None]
    def save(name, tensor):
        data = tensor.float().contiguous().numpy().astype("<f4").tobytes()
        (args.output / name).write_bytes(data)
        return dict(file=name, shape=list(tensor.shape), sha256=hashlib.sha256(data).hexdigest())
    media = dict(pixels=save("pixels.f32", pixels), waveform=save("waveform.f32", waveform),
                 grid=[[1, 2, 4], [2, 2, 2]], sample_rate_hz=24000, samples_per_token=cadence,
                 vision_utils_sha256=VISION_UTILS_SHA256, configuration_sha256=configuration_sha256,
                 parts=[{"kind": kind, "tokens": count} for kind, count in
                        [("text", 2), ("image", 2), ("text", 1), ("audio", 2), ("text", 1), ("video", 2)]])
    kwargs = dict(pixel_values=pixels[:8], image_grid_thw=torch.tensor([[1, 2, 4]]),
                  pixel_values_videos=pixels[8:], video_grid_thw=torch.tensor([[2, 2, 2]]),
                  input_values=waveform.to(torch.bfloat16))
    tokens = [200000, 1234, config.image_token_id, config.image_token_id, 5678,
              config.audio_token_id, config.audio_token_id, 21, config.video_token_id, config.video_token_id]
    return parent, media, kwargs, tokens

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("transformers", type=Path)
    parser.add_argument("modelopt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", action="store_true", help="validate meta geometry without opening payloads")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--trace-layers", action="store_true", help="retain bounded decoder outputs for numerical diagnosis")
    parser.add_argument("--multimodal", action="store_true", help="exercise the actual parent image/audio/video scatter")
    parser.add_argument("--vision-utils", type=Path)
    parser.add_argument("--configuration", type=Path)
    args = parser.parse_args()
    if args.multimodal and (args.vision_utils is None or args.configuration is None or args.preflight):
        parser.error("multimodal execution requires --vision-utils and --configuration, without --preflight")
    if args.threads <= 0:
        raise ValueError("reference CPU thread count must be positive")
    torch.set_num_threads(args.threads)
    extra_names = {"MuseSparkTextModel", "MuseSparkTextNormedEmbedding", "MuseSparkTextRotaryEmbedding"}
    if args.multimodal:
        extra_names.update({"MuseSparkModel", "MuseSparkVisionPatchEmbedder", "MuseSparkVisionAttention",
            "MuseSparkVisionMLP", "MuseSparkVisionEncoderLayer", "MuseSparkVisionRotaryEmbedding",
            "MuseSparkVisionModel", "apply_rotary_pos_emb_vision", "get_vision_pixel_shuffle_index"})
    namespace, logit_code = load_reference(args.transformers, extra_names)
    # Loading precision is part of the reference contract. These modules are
    # explicitly kept in FP32 by Transformers even when their serialized gates
    # are BF16. Other gains retain storage precision for the centered-gain sum.
    pretrained = next(node for node in ast.parse(args.transformers.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "MuseSparkPreTrainedModel")
    strict = next(ast.literal_eval(node.value) for node in pretrained.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "_keep_in_fp32_modules_strict"
                          for target in node.targets))
    def keep_fp32(name):
        return any("." + pattern + "." in "." + name + "." for pattern in strict)
    value = json.loads((args.source / "config.json").read_text())
    raw_config = value
    config = SimpleNamespace(**value["text_config"], _attn_implementation="eager")
    if config.rope_parameters != {"rope_theta": 500000.0, "rope_type": "default"}:
        raise ValueError("this released-checkpoint validation uses the pinned default rotary policy")
    source = Source(args.source, load_nvfp4(args.modelopt), args.preflight)
    with torch.device("meta"):
        model = namespace["MuseSparkTextModel"](config).eval()
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    banks = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, namespace["MuseSparkExperts"]):
            for projection in ("gate_up_proj", "down_proj"):
                parameter = module._parameters.pop(projection)
                key = f"model.language_model.{name}.{projection}"
                _, metadata = source.tensors[key]
                expected = list(parameter.shape)
                if metadata["dtype"] == "U8":
                    expected[-1] //= 2
                if metadata["shape"] != expected:
                    raise ValueError("source expert layout differs from the reference module")
                setattr(module, projection, ExpertAccessor(source, key, parameter.shape))
                banks += 1
    names = list(model.state_dict())
    for name in names:
        if "model.language_model." + name not in source.tensors:
            raise ValueError("reference module parameter is absent from the official source: " + name)
    print("reference meta geometry validated", "expert banks", banks, "dense text tensors", len(names), flush=True)
    if args.preflight:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    for index, name in enumerate(names):
        value = source.read("model.language_model." + name)
        bind(model, name, value.float() if keep_fp32(name) else value)
        if index % 100 == 0:
            print("dense tensors bound", index + 1, "of", len(names), flush=True)
    model.rotary_emb = namespace["MuseSparkTextRotaryEmbedding"](config)
    head.weight = torch.nn.Parameter(source.read("lm_head.weight"), requires_grad=False)
    trace_step = [0]
    traces = []
    if args.trace_layers:
        for index, layer in enumerate(model.layers):
            def trace(module, inputs, output, index=index):
                value = output.float().contiguous()
                data = value.numpy().astype("<f4").tobytes()
                filename = f"step-{trace_step[0]}-layer-{index}.f32"
                (args.output / filename).write_bytes(data)
                traces.append(dict(step=trace_step[0], layer=index, shape=list(value.shape),
                    file=filename, sha256=hashlib.sha256(data).hexdigest()))
            layer.register_forward_hook(trace)
    for index, layer in enumerate(model.layers):
        if (index + 1) % 8 == 0 or index + 1 == len(model.layers):
            def progress(module, inputs, output, index=index):
                print("reference decoder layer completed", index, flush=True)
            layer.register_forward_hook(progress)
    execution_model = model
    media, media_kwargs = None, {}
    prompt = [200000, 1234, 5678]
    if args.multimodal:
        execution_model, media, media_kwargs, prompt = multimodal_reference(args, namespace, raw_config, model, source)
    cache = Cache()
    report = {"repository": REPOSITORY, "revision": REVISION,
        "transformers_revision": TRANSFORMERS_REVISION, "modelopt_revision": MODELOPT_REVISION,
        "transformers_source_sha256": hashlib.sha256(args.transformers.read_bytes()).hexdigest(),
        "modelopt_source_sha256": hashlib.sha256(args.modelopt.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256((Path(__file__).parent / "muse_spark_reference_source.py").read_bytes()).hexdigest(),
        "comparison_kind": "teacher-forced text prefill and cached scoring of the same decoded NVFP4 artifact",
        "strict_fp32_modules": strict,
        "payload_proof_sha256": source.payload_proof_sha256, "torch_version": torch.__version__,
        "expert_access": "selected expert only, independent ModelOpt decode",
        "eredu_comparison_verified": False, "steps": []}
    if media is not None:
        report["media"] = media
        report["comparison_kind"] = "teacher-forced image/audio/video prefill and cached scoring of the same decoded NVFP4 artifact"
    for step, tokens in enumerate([prompt, [20], [21], [22]]):
        trace_step[0] = step
        before = cache.get_seq_length()
        result = execution_model(**(media_kwargs if step == 0 else {}), input_ids=torch.tensor([tokens]), attention_mask=masks(before, len(tokens), config.sliding_window),
            past_key_values=cache, use_cache=True)
        logits = head(result.last_hidden_state)
        local = {"logits": logits, "torch": torch, "self": SimpleNamespace(config=SimpleNamespace(text_config=config))}
        exec(logit_code, local)
        logits = local["logits"].float().contiguous()
        if not torch.isfinite(logits).all():
            raise ValueError("reference logits contain nonfinite values")
        data = logits.numpy().astype("<f4").tobytes()
        filename = f"step-{step}.f32"
        (args.output / filename).write_bytes(data)
        record = {"input_ids": tokens, "cache_before": before, "cache_after": cache.get_seq_length(),
            "shape": list(logits.shape), "file": filename, "sha256": hashlib.sha256(data).hexdigest(),
            "top1": logits.argmax(-1).tolist()}
        report["steps"].append(record)
        print("reference step", step, "cache", cache.get_seq_length(), "top1", record["top1"], flush=True)
    report.update(read_bytes=source.read_bytes, maximum_decoded_expert_bytes=source.maximum_decoded_expert_bytes,
                  elapsed_seconds=time.monotonic() - start_time)
    if args.trace_layers:
        report["layer_traces"] = traces
    (args.output / "reference-evidence.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
