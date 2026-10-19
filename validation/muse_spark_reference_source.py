"""Evaluate exact classes from a pinned Transformers file on CPU.

The reference file is downloaded outside the checkout. Named eager classes and
their helpers execute with explicit parameters, inert documentation/output hooks
and a pretrained-init stand-in; no installed Transformers code is substituted.
Requires Python 3.11, torch==2.9.0 and numpy==2.3.4.
"""

import argparse
import ast
import __future__
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from functools import cached_property

import torch


REVISION = "b0cdc61e607f39403e248b11c35aef4edfa7d216"
SOURCE_SHA256 = "250b582762a72b3fccc0b272c560edd69653a22e83d091bf9958ce245c52a97a"


def load_reference(reference, extra_names=()):
    """Execute hash-verified eager definitions; forward equations remain unchanged."""
    source = reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("reference file does not match the pinned Transformers source")
    if torch.__version__ != "2.9.0":
        raise ValueError("reference execution requires torch==2.9.0")
    names = {"MuseSparkTextInputScaledRMSNorm", "MuseSparkResidualGate", "MuseSparkTopkRouter",
             "MuseSparkExperts", "MuseSparkRMSNorm", "MuseSparkTextCenteredRMSNorm",
             "MuseSparkTextMLP", "MuseSparkTextAttention", "MuseSparkTextDecoderLayer", "MuseSparkMoE",
             "repeat_kv", "eager_attention_forward", "apply_rotary_pos_emb", "rotate_half",
             "MuseSparkAudioConv1dCacheLayer", "MuseSparkAudioConv1dPaddingCache",
             "MuseSparkAudioCausalConv1d", "MuseSparkAudioPatchEmbedding", "MuseSparkAudioMLP",
             "MuseSparkAudioConvBlock", "MuseSparkAudioLayer", "MuseSparkAudioModel"}
    names.add("MuseSparkVisionAdapter")
    names.update(extra_names)
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(definitions) != len(names):
        raise ValueError("reference class inventory differs")
    class ReferenceModule(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def get_input_embeddings(self):
            return self.embed_tokens

        def post_init(self):
            # The reference caller binds the admitted checkpoint parameters.
            pass

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "MuseSparkTextConfig": SimpleNamespace,
                 "ACT2FN": {"sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu,
                            "gelu": torch.nn.functional.gelu},
                 "GradientCheckpointingLayer": torch.nn.Module,
                 "ALL_ATTENTION_FUNCTIONS": SimpleNamespace(get_interface=lambda name, default: default),
                 "use_kernelized_func": lambda func: lambda cls: cls,
                 "use_kernel_forward_from_hub": lambda name: lambda func: func,
                 "MuseSparkPreTrainedModel": ReferenceModule, "cached_property": cached_property,
                 "auto_docstring": lambda cls=None, **kwargs: cls if cls is not None else lambda cls: cls,
                 "merge_with_config_defaults": lambda func: func,
                 "capture_outputs": lambda func: func,
                 "can_return_tuple": lambda func: func,
                 "BaseModelOutput": lambda **kwargs: SimpleNamespace(**kwargs),
                 "torch_compilable_check": check, "is_torchdynamo_compiling": lambda: False,
                 "OutputRecorder": lambda *args, **kwargs: None,
                 "dynamic_rope_update": lambda func: func,
                 "MoeModelOutputWithPast": lambda **kwargs: SimpleNamespace(**kwargs),
                 # Select the class's eager forward, with no external kernel override.
                 "use_experts_implementation": lambda **kwargs: lambda cls: cls}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reference), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    generation = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "MuseSparkForConditionalGeneration")
    forward = next(node for node in generation.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    logit_steps = [node for node in forward.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "logits" and "lm_head" not in ast.unparse(node.value)]
    if len(logit_steps) != 4:
        raise ValueError("reference logit transform stage inventory differs")
    logit_code = compile(ast.Module(body=logit_steps, type_ignores=[]), str(reference), "exec")
    return namespace, logit_code

# Released vision helper source required by full multimodal acceptance.
VISION_UTILS_SHA256 = "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356"
