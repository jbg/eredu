#!/usr/bin/env python3
"""Compare a chat_probe report with unchanged publisher Transformers code.

Uses the probe's exact prompt IDs and explicit decoding settings. Reports the
closing-token rank and the ordinary model argmax before each sampling decision.
Keep reports and Hugging Face caches outside the source tree.
"""
import argparse
import hashlib
import json
import pathlib
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    report = json.loads(args.probe.read_text())
    path = report["checkpoint"]
    # Load the released fast tokenizer directly. The checkpoint's legacy=False
    # LlamaTokenizer class otherwise requests a slow SentencePiece conversion,
    # even when tokenizer.json is available. No model or template code changes.
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        report["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    assert prompt == report["prompt"], "publisher/Eredu rendered prompts differ"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert input_ids == report["prompt_ids"], "publisher/Eredu prompt token IDs differ"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="eager",
    ).to(args.device).eval()
    close = tokenizer.convert_tokens_to_ids("</think>")
    rows = []

    class Inspect(transformers.LogitsProcessor):
        def __call__(self, ids, scores):
            logits = scores[0].float()
            rows.append({
                "step": len(rows), "argmax": logits.argmax().item(),
                "close_logit": logits[close].item(),
                "close_rank": int((logits > logits[close]).sum()) + 1,
            })
            return scores

    settings = dict(
        max_new_tokens=report["max_new_tokens"],
        do_sample=report["temperature"] > 0,
        repetition_penalty=report["repetition_penalty"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if settings["do_sample"]:
        settings.update(temperature=report["temperature"], top_k=report["top_k"],
                        top_p=report["top_p"], min_p=report["min_p"])
    torch.manual_seed(report["seed"])
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([input_ids], device=args.device),
            logits_processor=transformers.LogitsProcessorList([Inspect()]), **settings,
        )[0, len(input_ids):].tolist()
    elapsed = time.monotonic() - start
    # Independently rebuild the whole prefix at the closing prediction. This
    # checks the cached decode at the precise point the old constraint diverged.
    close_step = next((i for i, token in enumerate(output) if token == close), None)
    uncached = None
    if close_step is not None:
        with torch.inference_mode():
            logits = model(torch.tensor([input_ids + output[:close_step]], device=args.device),
                           use_cache=False).logits[0, -1].float()
        uncached = {"step":close_step, "argmax":logits.argmax().item(),
                    "close_logit":logits[close].item(),
                    "close_rank":int((logits > logits[close]).sum()) + 1}
    result = dict(
        torch=torch.__version__, transformers=transformers.__version__,
        device=args.device, dtype=args.dtype, attention="eager", settings=settings,
        prompt=prompt, prompt_ids=input_ids, token_ids=output,
        decoded=tokenizer.decode(output, skip_special_tokens=False),
        rows=rows, uncached_closing_prediction=uncached, elapsed_seconds=elapsed,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in pathlib.Path(path).glob("*.py")},
        exact_match=output == report["token_ids"],
        first_divergence=next((i for i, (a, b) in enumerate(zip(output, report["token_ids"])) if a != b), None),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["decoded"], flush=True)
    print(f"{len(output)} tokens; {elapsed:.2f}s; exact match={result['exact_match']}", flush=True)


if __name__ == "__main__":
    main()
