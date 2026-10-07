#!/usr/bin/env python3
"""Hash-checked NVIDIA agent sampler with supplied uniforms, without NeMo imports.

Only torch.multinomial is replaced: it records the original method's final
probabilities and selects their inverse CDF using a supplied uniform. This
isolates sampling policy from backend RNG algorithms. No model weights execute.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
SOURCE = "nemo/collections/speechlm2/inference/model_wrappers/model_factory.py"
SHA256 = "14d791a65e6885a444fe48e437750a8540b6ba2d01fcd77339cdb29da0b0dc42"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = (args.checkout / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256
    cls = next(x for x in ast.parse(source).body
               if isinstance(x, ast.ClassDef) and x.name == "ModelInterface")
    method = next(x for x in cls.body
                  if isinstance(x, ast.FunctionDef) and x.name == "_sample_text_token")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), SOURCE, "exec"), namespace)
    sample = namespace["_sample_text_token"]
    logits = [-1.3, .35, -.4, 2.8, -.9, 1.7, .65, -2.2,
              1.05, -.15, .2, -1.7, .1, -.55, .85, -2.8]

    def case(name, temperature=1., top_p=1., repetition_penalty=1.,
             history=None, current_step=None, uniform=.73, values=None):
        history = history if history is not None else [3, 3, 4, 1, 12, 2, 5, 15]
        return dict(name=name, temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty, history=history,
                    current_step=len(history) if current_step is None else current_step,
                    uniform=uniform, logits=values if values is not None else logits)

    special_logits = logits.copy()
    special_logits[12] = 3.2
    cases = [
        case("streaming_greedy", temperature=0.),
        case("zero_temperature_ignores_filters", temperature=0., top_p=.61, repetition_penalty=1.8),
        case("identity_greedy"),
        case("special_bypass", temperature=.7, top_p=.65,
             repetition_penalty=2., values=special_logits),
        case("penalty_sign_unique_special_exclusion", temperature=.8,
             repetition_penalty=1.8),
        case("nucleus_crossing", temperature=.9, top_p=.7, repetition_penalty=1.4),
        case("nucleus_single_still_draws", top_p=.01),
        case("history_prefix", temperature=.8, repetition_penalty=1.8, current_step=2),
        case("empty_history_step", temperature=.8, repetition_penalty=1.8, current_step=0),
        case("effective_forced_feedback", temperature=.8, repetition_penalty=1.8,
             history=[5, 5, 1, 12, 2]),
    ]
    original_multinomial = torch.multinomial
    maximum_delta = 0.

    def execute(item, dtype):
        draws = []

        def supplied_multinomial(probs, num_samples):
            assert num_samples == 1 and probs.ndim == 1
            draws.append(probs.tolist())
            cdf = torch.cumsum(probs, dim=0)
            token = torch.searchsorted(cdf, torch.tensor(item["uniform"], dtype=dtype), right=True)
            return token.clamp(max=len(probs) - 1).reshape(1)

        torch.multinomial = supplied_multinomial
        try:
            policy = SimpleNamespace(special_token_ids={1, 2, 12},
                                     temperature=item["temperature"], top_p=item["top_p"],
                                     repetition_penalty=item["repetition_penalty"])
            # Float64 runs widen the exact Float32 inputs for precision evidence.
            values = torch.tensor([item["logits"]], dtype=torch.float32).to(dtype)
            token = sample(policy, values, torch.tensor([item["history"]]), item["current_step"])
            assert len(draws) <= 1
            return dict(token=token.item(), draws=len(draws), probabilities=draws[0] if draws else None)
        finally:
            torch.multinomial = original_multinomial

    for item in cases:
        item["original_greedy"] = torch.tensor(item["logits"]).argmax().item()
        item["expected"] = execute(item, torch.float32)
        widened = execute(item, torch.float64)
        assert widened["token"] == item["expected"]["token"]
        if widened["probabilities"] is not None:
            maximum_delta = max(maximum_delta, max(abs(a - b) for a, b in
                zip(widened["probabilities"], item["expected"]["probabilities"])))
    fixture = dict(provenance=dict(revision=REVISION, source=SOURCE, sha256=SHA256,
                                  torch=torch.__version__, device="cpu",
                                  stochastic_boundary="multinomial replaced by supplied-uniform inverse CDF"),
                   tolerance=dict(absolute=2e-7, relative=2e-6,
                                  float32_float64_max_absolute=maximum_delta),
                   special_token_ids=[1, 2, 12], cases=cases)
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
