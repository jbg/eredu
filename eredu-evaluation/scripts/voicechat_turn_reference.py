#!/usr/bin/env python3
"""Execute the pinned NVIDIA host turn method without loading NeMo or weights.

The original method is extracted with ast and executed unchanged. The small
scalar/vector facade implements only the tensor indexing and comparisons used
by that method; RNNT counters are explicit inputs, not a decoder simulation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
SOURCE = "nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py"
SHA256 = "b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a"
POLICY = dict(force_turn_taking=True, force_turn_taking_threshold=40,
              rnnt_eou_frames=40, rnnt_bou_frames=40,
              asr_eou_first_turn=40, asr_min_speech_frames_first_turn=2,
              rnnt_self_play_suppression=False, max_agent_response_sec=0,
              tts_text_token_ratio_cap=16.0, tts_text_token_min=5)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        value = self.values[index]
        return Vector(value) if isinstance(index, slice) else Scalar(value)

    def __setitem__(self, index, value):
        self.values[index] = value

    def __eq__(self, value):
        return Vector(x == value for x in self.values)

    def any(self):
        return any(self.values)

    def __len__(self):
        return len(self.values)


class Matrix:
    def __init__(self, values):
        self.rows = [Vector(row) for row in values]

    def size(self, axis):
        assert axis == 0
        return len(self.rows)

    def __getitem__(self, index):
        row, column = index
        return self.rows[row][column]

    def __setitem__(self, index, value):
        row, column = index
        self.rows[row][column] = value


def cases():
    base = dict(blank_count=0, nonblank_consec=0, nonblank_total=0,
                speech_confirmed=False, agent_speaking=False,
                first_turn=True, user_first_turn=True,
                rolling_density=0.0, post_eos_fired=False)

    def case(name, history=None, raw_agent=12, **updates):
        return dict(name=name, history=[12] * 45 if history is None else history,
                    raw_agent=raw_agent, state=base | updates)

    return [
        case("confirmed_first_user_eou", blank_count=40, speech_confirmed=True,
             nonblank_total=2, rolling_density=.02),
        case("eou_one_frame_early", blank_count=39, speech_confirmed=True,
             nonblank_total=2, rolling_density=.02),
        case("first_fallback_without_silence", history=[12] * 50,
             nonblank_consec=2, nonblank_total=2, rolling_density=.2),
        case("barge_in_threshold", history=[1] + [12] * 39,
             nonblank_consec=40, nonblank_total=40, agent_speaking=True,
             first_turn=False, rolling_density=.4),
        case("barge_in_one_frame_early", history=[1] + [12] * 38,
             nonblank_consec=39, nonblank_total=39, agent_speaking=True,
             first_turn=False, rolling_density=.4),
        case("older_eos_wins_newer_bos", history=[2, 1] + [12] * 38,
             nonblank_total=5, speech_confirmed=True, agent_speaking=True,
             first_turn=False, user_first_turn=False, rolling_density=.4),
        case("prior_eos_wins_current_bos", history=[2] + [12] * 39,
             raw_agent=1, nonblank_total=5, speech_confirmed=True,
             first_turn=False, user_first_turn=False, rolling_density=.4),
        case("first_user_minimum_one", nonblank_consec=1, nonblank_total=1,
             rolling_density=.1),
        case("first_user_minimum_two", nonblank_consec=2, nonblank_total=2,
             rolling_density=.2),
        case("later_user_minimum_three", nonblank_consec=3, nonblank_total=3,
             first_turn=False, user_first_turn=False, rolling_density=.3),
        case("low_density_defers_confirmation", nonblank_consec=5, nonblank_total=5,
             first_turn=False, user_first_turn=False, rolling_density=.1),
        case("low_density_six_confirms", nonblank_consec=6, nonblank_total=6,
             first_turn=False, user_first_turn=False, rolling_density=.1),
        case("silence_clears_unconfirmed_total", blank_count=10, nonblank_total=2,
             first_turn=False, user_first_turn=False, rolling_density=.2),
        case("silence_retains_confirmed_total", blank_count=10, nonblank_total=2,
             speech_confirmed=True, first_turn=False,
             user_first_turn=False, rolling_density=.2),
        case("bos_just_outside_history", history=[1] + [12] * 40,
             blank_count=40, nonblank_total=3, speech_confirmed=True,
             first_turn=False, user_first_turn=False, rolling_density=.2),
        case("bos_at_history_boundary", history=[1] + [12] * 39,
             blank_count=40, nonblank_total=3, speech_confirmed=True,
             first_turn=False, user_first_turn=False, rolling_density=.2),
        case("current_bos_resets_user_counters", raw_agent=1,
             nonblank_total=3, speech_confirmed=True, rolling_density=.2),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source_file.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError("NVIDIA wrapper source does not match the pinned hash")
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef)
               and x.name == "NemotronVoicechatInferenceWrapper")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == "_apply_rnnt_turn_taking")
    namespace = dict(torch=SimpleNamespace(Tensor=Matrix),
                     logging=SimpleNamespace(info=lambda *args: None,
                                             debug=lambda *args: None))
    exec(compile(ast.Module(body=[method], type_ignores=[]), SOURCE, "exec"), namespace)
    apply = namespace[method.name]
    rows = cases()
    for row in rows:
        state = {key: Vector([value]) for key, value in row["state"].items()}
        owner = SimpleNamespace(model_cfg=POLICY,
                                model=SimpleNamespace(stt_model=SimpleNamespace(
                                    text_bos_id=1, text_eos_id=2, text_pad_id=12)),
                                _redirect_tokens_queue=[], _max_agent_response_frames=0,
                                _agent_eos_just_fired=False, _post_tc_bos_exempt=False)
        text = Matrix([row["history"] + [row["raw_agent"]]])
        apply(owner, len(row["history"]), text, Vector([row["state"]["blank_count"] > 0]), state)
        row["expected"] = dict(
            agent_token=text[0, len(row["history"])].item(),
            state={key: state[key][0].item() for key in row["state"]},
            agent_eos_just_fired=owner._agent_eos_just_fired)
    fixture = dict(provenance=dict(revision=REVISION, source=SOURCE, sha256=SHA256,
                                  method=method.name, execution="original extracted Python method",
                                  tensor_boundary="scalar/vector indexing only; supplied RNNT counters"),
                   controls=dict(bos=1, eos=2, pad=12), policy=POLICY, cases=rows)
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} original-method cases to {args.output}")


if __name__ == "__main__":
    main()
