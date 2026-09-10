"""Pinned local publisher reference: prefill and four teacher-forced decode steps.

The artifact directory must have been downloaded at an immutable revision and
verified against its manifest before running this script. No network is used.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--capture-routes", action="store_true")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="Plain text encoded with the artifact tokenizer")
    prompts.add_argument("--chat-request", type=Path, help="JSON containing messages, tools and template_options")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="Feed the previous argmax into each cached step")
    args = parser.parse_args()
    if args.decode_steps < 0:
        parser.error("--decode-steps must be nonnegative")
    torch.set_num_threads(args.threads)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.artifact, trust_remote_code=True, local_files_only=True,
        dtype=getattr(torch, args.dtype), attn_implementation="eager",
    ).eval().to(args.device)
    tokenizer = None
    rendered_prompt = None
    prefix = [1, 3, 2]
    if args.prompt is not None or args.chat_request is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.artifact, trust_remote_code=True, local_files_only=True)
        if args.chat_request is not None:
            request = json.loads(args.chat_request.read_text())
            rendered_prompt = tokenizer.apply_chat_template(
                request["messages"], tools=request.get("tools"), tokenize=False,
                add_generation_prompt=True, **request.get("template_options", {}))
            prefix = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        else:
            rendered_prompt = args.prompt
            prefix = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not prefix:
        raise ValueError("The prefill input must not be empty")
    inputs = list(prefix)
    generated = []
    fixture = {}
    if args.capture_layers:
        def hook(index, field):
            def capture(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                fixture[f"layers.{step}.{index}.{field}"] = value.float().contiguous()
            return capture
        for index, layer in enumerate(model.model.layers):
            layer.register_forward_hook(hook(index, "output"))
            layer.input_layernorm.register_forward_hook(hook(index, "input_norm"))
            layer.self_attn.register_forward_hook(hook(index, "attention"))
            if index < 6:
                layer.post_attention_layernorm.register_forward_hook(hook(index, "post_attention_norm"))
                layer.mlp.register_forward_hook(hook(index, "mlp"))
                for field in ["q_proj", "k_proj", "v_proj", "gate_proj"]:
                    if hasattr(layer.self_attn, field):
                        getattr(layer.self_attn, field).register_forward_hook(hook(index, field))
    if args.capture_routes:
        def routes(index, bank):
            def capture(module, positional, keywords, output):
                hidden = keywords.get("hidden_states", positional[0] if positional else None)
                hidden = hidden.reshape(-1, hidden.shape[-1])
                router = module.v_router if bank == 1 else module.gate
                logits = torch.nn.functional.linear(hidden, router.weight)
                scores = (torch.softmax(logits, dim=-1, dtype=torch.float32)
                          if module.router_score_func == "softmax" else torch.sigmoid(logits.float()))
                choices = scores if router.bias is None else scores + router.bias.float()
                top = module.num_experts_per_tok if bank == 1 else module.top_k
                ids = torch.topk(choices, top, dim=-1).indices
                selected = torch.gather(scores, -1, ids)
                coefficients = selected
                if (top > 1 if bank == 1 else module.norm_topk_prob):
                    coefficients = coefficients / coefficients.sum(dim=-1, keepdim=True)
                coefficients = (coefficients * module.router_scaling_factor).to(hidden.dtype)
                prefix = f"routes.{step}.{index}.{bank}"
                for field, value in [("input", hidden), ("logits", logits), ("ids", ids.to(torch.int32)),
                                     ("scores", selected), ("coefficients", coefficients)]:
                    fixture[f"{prefix}.{field}"] = value.contiguous() if field == "ids" else value.float().contiguous()
            return capture
        for index, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, "v_router"):
                layer.self_attn.register_forward_hook(routes(index, 1), with_kwargs=True)
            if hasattr(layer.mlp, "experts"):
                layer.mlp.register_forward_hook(routes(index, 0), with_kwargs=True)
    cache = None
    with torch.inference_mode():
        for step in range(args.decode_steps + 1):
            ids = prefix if step == 0 else [generated[-1] if args.greedy else step + 3]
            if step:
                inputs.extend(ids)
            result = model(torch.tensor([ids], device=args.device), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            fixture[f"logits.{step}"] = result.logits.float().contiguous()
            generated.append(int(result.logits[0, -1].argmax()))
            if args.stop_at_eos and tokenizer is not None and generated[-1] == tokenizer.eos_token_id:
                break
    fixture["input_ids"] = torch.tensor(inputs, dtype=torch.int32)
    save_file({name: value.cpu() for name, value in fixture.items()}, str(args.output))
    source = args.artifact / "modeling_k2_horizon.py"
    report = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "device": args.device, "threads": args.threads, "dtype": args.dtype,
        "attention": "eager", "inputs": inputs,
        "prefill_tokens": len(prefix), "greedy": args.greedy,
        "requested_decode_steps": args.decode_steps, "stop_at_eos": args.stop_at_eos,
        "rendered_prompt": rendered_prompt, "argmax_tokens": generated,
        "generated_text": tokenizer.decode(generated) if tokenizer is not None else None,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
