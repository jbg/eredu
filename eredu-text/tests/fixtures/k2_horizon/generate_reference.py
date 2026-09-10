"""Generate exact publisher Jinja/Transformers fixtures from verified local artifacts.

Usage: python generate_reference.py /private/tmp/eredu-k2-horizon
No network requests or model weights are needed.
"""
import argparse
import hashlib
import json
import platform
import struct
import subprocess
from pathlib import Path

import jinja2
import transformers


def gguf_metadata(path):
    with path.open("rb") as f:
        def unpack(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        def string():
            return f.read(unpack("Q")).decode()
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = unpack("I"), unpack("Q")
                return [value(element) for _ in range(count)]
            return unpack({0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}[kind])
        assert f.read(4) == b"GGUF" and unpack("I") == 3
        unpack("Q")  # Tensor descriptors are unnecessary for tokenizer validation.
        count = unpack("Q")
        result = {}
        for _ in range(count):
            key = string()
            result[key] = value(unpack("I"))
        return result


def cases():
    user = {"role": "user", "content": "Cafe\u0301 x\u200cy\u200dz 1234567\nمرحبا 世界"}
    yield "plain", [user], [], {}, True
    for effort, field in [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")]:
        for generation_prompt in [False, True]:
            yield f"history-{effort}-{generation_prompt}", [
                {"role": "system", "content": "Be precise."}, user,
                {"role": "assistant", field: "Check the evidence.", "content": "Confirmed."},
                {"role": "user", "content": "Continue."},
            ], [], {"reasoning_effort": effort}, generation_prompt
    tool = {"type": "function", "function": {
        "name": "lookup", "description": "Look up a record.", "parameters": {
            "type": "object", "properties": {
                "query": {"$ref": "#/$defs/Query", "description": "Search text."},
                "count": {"type": "integer"},
                "filters": {"type": "array", "items": {"type": "string"}},
                "mixed": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            }, "required": ["query", "count"],
            "$defs": {"Query": {"type": "string", "description": "A query."}},
        },
    }}
    for presentation in ["markdown", "json", "xml"]:
        for call_format in ["json", "xml", "xml_typed"]:
            yield f"tools-{presentation}-{call_format}", [user, {
                "role": "assistant", "think_fast": "Use the lookup tool.", "content": "",
                "tool_calls": [{"type": "function", "id": "call_0", "function": {
                    "name": "lookup", "arguments": {
                        "query": "a < b & café", "count": 2,
                        "filters": ["active", "recent"], "mixed": 7,
                    },
                }}],
            }, {"role": "tool", "tool_call_id": "call_0", "content": {"found": True, "count": 2}},
            {"role": "user", "content": "Explain."}], [tool], {
                "tool_presentation_format": presentation, "tool_call_format": call_format,
                "reasoning_effort": "low",
            }, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    llama_tokenize = args.root / "llama-build/bin/llama-tokenize"
    if not llama_tokenize.is_file():
        parser.error("build the pinned publisher llama-tokenize before generating GGUF fixtures")
    output = Path(__file__).parent
    templates = {}
    fixtures = []
    for family, gguf_file in [("dense", "K2-Horizon-1B-BF16.gguf"), ("mova", "K2-Horizon-36B-BF16.gguf")]:
        artifact = args.root / "artifacts" / family
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(artifact, local_files_only=True)
        metadata = gguf_metadata(args.root / "artifacts" / f"{family}-gguf" / gguf_file)
        # GGUF retains the corresponding publisher vocabulary indices.
        assert all(tokenizer.convert_ids_to_tokens(i) == token for i, token in enumerate(metadata["tokenizer.ggml.tokens"]))
        for source, template in [("safetensors", (artifact / "chat_template.jinja").read_text()), ("gguf", metadata["tokenizer.chat_template"])]:
            name = f"{family}-{source}"
            templates[name] = template
            (output / f"{name}.jinja").write_text(template)
            for case, messages, tools, kwargs, generation_prompt in cases():
                rendered = tokenizer.apply_chat_template(messages, tools=tools, chat_template=template,
                    tokenize=False, add_generation_prompt=generation_prompt, **kwargs)
                if source == "gguf":
                    command = [str(llama_tokenize), "-m", str(args.root / "artifacts" / f"{family}-gguf" / gguf_file),
                        "--stdin", "--ids", "--no-bos", "--no-escape"]
                    result = subprocess.run(command, input=rendered, text=True, capture_output=True, check=True)
                    ids = json.loads(result.stdout)
                else:
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                fixtures.append(dict(name=f"{name}-{case}", template=name, messages=messages,
                    tools=tools, kwargs=kwargs, add_generation_prompt=generation_prompt,
                    template_defaults=tokenizer.special_tokens_map,
                    rendered=rendered, token_ids=ids))
    (output / "reference.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
    provenance = dict(python=platform.python_version(), transformers=transformers.__version__,
        jinja2=jinja2.__version__, template_sha256={name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in templates.items()}, tokenization_reference={"safetensors":"Publisher Transformers tokenizer",
        "gguf":"MBZUAI-IFM/llama.cpp 35999d101cf2233fc54f09c3c8d599da7303ce02 llama-tokenize --stdin --ids --no-bos --no-escape"},
        llama_tokenize_sha256=hashlib.sha256(llama_tokenize.read_bytes()).hexdigest())
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
