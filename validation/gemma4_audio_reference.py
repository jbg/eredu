"""Export Gemma 4 audio-conditioned prefill and cached greedy logits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("pcm", type=Path, help="Mono 16 kHz little-endian FP32 samples")
    parser.add_argument("output", type=Path, help="Output SafeTensors file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text", "text": "Describe this sound in one short sentence."},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    config = json.loads((args.checkpoint / "processor_config.json").read_text())
    extract = Gemma4AudioFeatureExtractor(**config["feature_extractor"])
    pcm = np.fromfile(args.pcm, dtype="<f4")
    features = extract([pcm], sampling_rate=16_000, return_tensors="pt")
    replacement = Gemma4Processor.replace_audio_token(tokenizer, features, 0)
    inputs = tokenizer(
        prompt.replace(tokenizer.audio_token, replacement, 1),
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs.update(features)

    torch.set_num_threads(8)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    text_ids = [token for token in inputs["input_ids"][0].tolist() if token != model.config.audio_token_id]
    scores, tokens = [], []
    with torch.inference_mode():
        value = model(**inputs, use_cache=True, logits_to_keep=1)
        for step in range(8):
            row = value.logits[0, -1].float().cpu().numpy()
            scores.append(row)
            token = int(row.argmax())
            tokens.append(token)
            if step < 7:
                value = model(
                    input_ids=torch.tensor([[token]]),
                    past_key_values=value.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"prefill.logits": np.array(scores[:1]), "decode.logits": np.array(scores[1:])}, args.output)
    report = {
        "input": {"token_ids": text_ids},
        "output": {"fed_token_ids": tokens[:-1], "tensor_file": args.output.name},
        "reference": {
            "implementation": "transformers",
            "device": "cpu",
            "dtype": "bfloat16",
            "greedy_token_ids": tokens,
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
