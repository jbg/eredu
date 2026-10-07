#!/usr/bin/env python3
"""Tiny CPU oracle executing pinned NVIDIA VoiceChat RNNT methods.

Requires torch and a local checkout of NVIDIA-NeMo/Speech at REVISION. No Eredu
code or checkpoint configuration is imported. The four upstream files are
hash-checked, then their original method ASTs are executed with tiny torch
modules and disabled adapters/punctuation. This avoids importing the entire
NeMo training stack. The fixture is component evidence with synthetic nonzero
weights, not released-checkpoint inference or a full NeMo installation test.
"""

import argparse
import ast
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
FILES = {
    "nemo/collections/asr/modules/rnnt.py":
        "b9d4925fe5c7c48da8330f48a57126688850fd96dd28438a3f918439ca44cb93",
    "nemo/collections/asr/modules/rnnt_abstract.py":
        "e4373e707acf0435c3511cd1e4457ff96c48e29e15d47047aec08ec544aea4f3",
    "nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py":
        "b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a",
    "nemo/collections/common/parts/rnn.py":
        "70a7068702a0fbfb0b3a32368e256b38256a1d91cfbbca0bbbfaa4faead0a73c",
}


def methods(checkout, relative, class_name, names):
    path = checkout / relative
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == FILES[relative], path
    parsed = ast.parse(source, filename=str(path))
    cls = next(node for node in parsed.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = dict(torch=torch, Optional=Optional, List=List, Tuple=Tuple,
                     logging=logging)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def values(tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--perception-fixture", type=Path,
                        help="Use the pinned perception fixture's six raw streaming outputs")
    args = parser.parse_args()
    perception = None
    if args.perception_fixture:
        perception = json.loads(args.perception_fixture.read_text())
        assert perception["provenance"]["revision"] == REVISION
    encoder_hidden = 4 if perception else 3
    for relative, expected in FILES.items():
        assert hashlib.sha256((args.checkout / relative).read_bytes()).hexdigest() == expected, relative
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    # No randomized parameters remain: every entry is overwritten below.
    predictor_methods = methods(args.checkout, next(iter(FILES)), "RNNTDecoder", ["predict"])
    joint_methods = methods(args.checkout, next(iter(FILES)), "RNNTJoint",
                            ["project_encoder", "project_prednet", "joint_after_projection"])
    joint_methods.update(methods(args.checkout, list(FILES)[1], "AbstractRNNTJoint", ["joint"]))
    wrapper_path = list(FILES)[2]
    wrapper_ast = ast.parse((args.checkout / wrapper_path).read_bytes())
    wrapper_class = next(node.name for node in wrapper_ast.body
                         if isinstance(node, ast.ClassDef) and any(
                             isinstance(member, ast.FunctionDef) and member.name == "_rnnt_step"
                             for member in node.body))
    wrapper_methods = methods(args.checkout, wrapper_path, wrapper_class,
                              ["_rnnt_init_state", "_rnnt_step"])

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pred_hidden = 2
            self.random_state_sampling = False
            # Released normalization_mode=None selects LSTMDropout; in eval
            # its only active operation is this torch LSTM (no projection).
            self.prediction = torch.nn.ModuleDict({
                "embed": torch.nn.Embedding(4, 2, padding_idx=3),
                "dec_rnn": torch.nn.LSTM(2, 2, num_layers=2, dropout=0.2),
            })

        def is_adapter_available(self):
            return False

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Linear(encoder_hidden, 3)
            self.pred = torch.nn.Linear(2, 3)
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                                  torch.nn.Linear(3, 4))
            self.masking_prob = 0.0
            self.preserve_memory = False
            self.log_softmax = None
            self.temperature = 1.0

        def is_adapter_available(self):
            return False

    class Wrapper:
        pass

    for cls, extracted in [(Predictor, predictor_methods), (Joint, joint_methods),
                           (Wrapper, wrapper_methods)]:
        for name, method in extracted.items():
            setattr(cls, name, method)

    predictor, joint = Predictor().eval(), Joint().eval()
    with torch.no_grad():
        for ordinal, (_, parameter) in enumerate(predictor.named_parameters()):
            parameter.copy_(torch.tensor([
                0.31 * math.sin((index + 1) * 0.71 + ordinal * 1.17)
                for index in range(parameter.numel())]).reshape(parameter.shape))
        predictor.prediction["embed"].weight[3].zero_()
        if perception:
            # Center the nonconstant acoustic coordinate so this short stream
            # contains both emitted labels and blank frames, with all input
            # coordinates and recurrent feedback participating.
            joint.enc.weight.copy_(torch.tensor([
                [.07, 20., -.03, .02], [.11, -20., .05, -.02], [.03, -.08, .6, .04]]))
            joint.enc.bias.copy_(torch.tensor([-22.36, 22.37, .1]))
        else:
            joint.enc.weight.copy_(torch.tensor([[1., .07, -.03], [-1., .11, .05], [.03, -.08, .6]]))
            joint.enc.bias.copy_(torch.tensor([.04, -.03, .1]))
        joint.pred.weight.copy_(torch.tensor([[.4, -.3], [-.2, .5], [.1, .25]]))
        joint.pred.bias.copy_(torch.tensor([.03, -.02, .01]))
        joint.joint_net[2].weight.copy_(torch.tensor([
            [1.2, -.5, .2], [-.3, .8, -.2], [-.4, -.3, -.7], [-.2, 1.4, .1]]))
        joint.joint_net[2].bias.copy_(torch.tensor([.01, -.04, -.06, .03]))

    def parameter_map():
        result = {}
        for prefix, model in [("stt_model.rnnt_decoder", predictor), ("stt_model.rnnt_joint", joint)]:
            for name, parameter in model.named_parameters():
                if "dec_rnn." in name:
                    name = name.replace("dec_rnn.", "dec_rnn.lstm.")
                result[prefix + "." + name] = {"shape": list(parameter.shape), "values": values(parameter)}
        return result

    wrapper = Wrapper()
    wrapper.model = SimpleNamespace(stt_model=SimpleNamespace(
        _rnnt_decoder=predictor, _rnnt_joint=joint, _rnnt_blank_id=3))
    wrapper._rnnt_max_symbols = 3
    wrapper._rnnt_punct_ids = []
    wrapper._rnnt_punct_ids_set = set()
    wrapper._rnnt_punct_bias_min_blanks = 5
    calls = []

    def capture_joint(_module, inputs, output):
        calls.append({"hidden_sum": values(inputs[0]), "logits": values(output)})

    joint.joint_net.register_forward_hook(capture_joint)
    predictor_steps = []
    with torch.inference_mode():
        hidden = None
        for token in [None, 0, 2, 1]:
            out, hidden = predictor.predict(
                y=None if token is None else torch.tensor([[token]]), state=hidden,
                add_sos=token is None, batch_size=1)
            predictor_steps.append({"token": token, "output_shape": list(out.shape),
                                    "output": values(out), "h": values(hidden[0]), "c": values(hidden[1])})

    def run_frames(frames):
        state = wrapper._rnnt_init_state(1, "cpu")
        records = []
        with torch.inference_mode():
            for frame in frames:
                calls.clear()
                previous_count = len(state["y_sequence"])
                state, is_blank = wrapper._rnnt_step(torch.tensor([frame]), state)
                ended_blank = max(range(4), key=lambda token: calls[-1]["logits"][token]) == 3
                emitted = state["y_sequence"][previous_count:]
                records.append({"encoder": frame, "emitted": state["y_sequence"][previous_count:],
                    "first_blank": bool(is_blank[0]), "prediction": values(state["pred_out"]),
                    "ended_blank": ended_blank,
                    "hit_symbol_limit": len(emitted) == wrapper._rnnt_max_symbols and not ended_blank,
                    "h": values(state["pred_hidden"][0]), "c": values(state["pred_hidden"][1]),
                    "joint_calls": list(calls), "blank_count": int(state["blank_count"][0]),
                    "nonblank_total": int(state["nonblank_total"][0]),
                    "rolling_density": float(state["rolling_density"][0])})
        return records

    frames = ([step["encoder_output"]["values"] for step in perception["perception"]["steps"]]
              if perception else [[-.8, .2, .3], [1., -.4, .2], [1., .1, -.3], [-1., -.2, .4]])
    records = run_frames(frames)
    # Find a stable frame at which predictor feedback changes a nonblank to
    # blank before the cap. The selected input itself is retained in the fixture.
    for index in ([] if perception else range(-2000, 2001)):
        candidate = [index / 10000., .13, -.27]
        trial = run_frames([candidate])
        if 0 < len(trial[0]["emitted"]) < wrapper._rnnt_max_symbols:
            records.extend(trial)
            break
    else:
        if not perception:
            raise RuntimeError("fixture lacks a nonblank-to-blank transition")
    if perception:
        assert any(record["emitted"] for record in records)
        assert any(record["first_blank"] for record in records)

    # Exercise both LSTM bias vectors, all four gates and nonzero recurrent
    # states independently of zero-SOS priming. PyTorch uses i,f,g,o gate order.
    sequence = torch.tensor([[[.23, -.41]], [[-.17, .38]], [[.07, .19]]])
    initial_h = torch.tensor([[[.12, -.08]], [[.19, .04]]])
    initial_c = torch.tensor([[[-.11, .09]], [[.06, -.14]]])
    with torch.inference_mode():
        sequence_output, (final_h, final_c) = predictor.prediction["dec_rnn"](
            sequence, (initial_h, initial_c))
        double_lstm = copy.deepcopy(predictor.prediction["dec_rnn"]).double()
        double_output, (double_h, double_c) = double_lstm(
            sequence.double(), (initial_h.double(), initial_c.double()))
    precision_delta = max(float((left.double() - right).abs().max()) for left, right in [
        (sequence_output, double_output), (final_h, double_h), (final_c, double_c)])
    original_parameters = parameter_map()
    predictor.double()
    joint.double()
    double_records = run_frames(frames)
    if not perception:
        double_records += run_frames([records[-1]["encoder"]])
    for f32, f64 in zip(records, double_records):
        assert f32["emitted"] == f64["emitted"]
        for name in ["prediction", "h", "c"]:
            precision_delta = max(precision_delta, max(abs(a - b) for a, b in zip(f32[name], f64[name])))
        for left, right in zip(f32["joint_calls"], f64["joint_calls"]):
            precision_delta = max(precision_delta, max(abs(a - b) for a, b in zip(left["logits"], right["logits"])))
    # Set before any Eredu comparison. 2e-6 absolute / 2e-5 relative leaves
    # room for different f32 reduction kernels while remaining far below a
    # wrong gate, missing bias or single-versus-double SOS error.
    assert precision_delta < 2e-6

    fixture = {
        "provenance": {"repository": "https://github.com/NVIDIA-NeMo/Speech", "revision": REVISION,
                       "source_sha256": FILES, "torch_version": torch.__version__, "dtype": "float32",
                       "oracle": "unmodified extracted NVIDIA methods with tiny torch modules",
                       "weights": "synthetic deterministic nonzero component parameters",
                       "scope": "batch=1; punctuation bias disabled; no released-weight or end-to-end parity"},
        "tolerance": {"absolute": 2e-6, "relative": 2e-5,
                      "f32_vs_f64_max_absolute": precision_delta,
                      "basis": "same f32 weights/inputs widened to f64; pinned methods preserve all decisions; tolerance set before Eredu comparison"},
        "geometry": {"encoder_hidden": encoder_hidden, "prediction_hidden": 2, "prediction_layers": 2,
                     "joint_hidden": 3, "vocab_size": 3, "blank_id": 3, "max_symbols": 3},
        "parameters": original_parameters, "predictor_steps": predictor_steps,
        "lstm_sequence": {"input_shape": list(sequence.shape), "input": values(sequence),
                          "initial_h": values(initial_h), "initial_c": values(initial_c),
                          "output": values(sequence_output), "h": values(final_h), "c": values(final_c)},
        "streaming_frames": records[:len(frames)],
        "feedback_terminates_frame": records[len(frames):],
    }
    if perception:
        fixture["provenance"]["perception_fixture_sha256"] = hashlib.sha256(
            args.perception_fixture.read_bytes()).hexdigest()
        fixture["provenance"]["input"] = "unprojected streaming outputs of perception-nvidia.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
