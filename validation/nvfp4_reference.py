"""Generate bounded NVFP4 evidence with NVIDIA's pinned eager CPU decoder.

Requires the pinned torch==2.9.0 environment. The optional official source
directory contains the authenticated headers from muse_spark_inventory.py.
Only 44 packed-weight/scale bytes are fetched for the released example.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from muse_spark_inventory import credential, read_bounded


REVISION = "f0daaa241ac6644e9bedd19a1af40c4a2aa8ca04"
SOURCE_SHA256 = "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e"


class ReferenceStorage:
    def __init__(self, packed):
        self._quantized_data = packed
        self.metadata = {"dtype": torch.float32, "shape": [packed.shape[0], packed.shape[1] * 2]}


def load_reference(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("NVIDIA source does not match its pinned digest")
    names = {"NVFP4QTensor", "_cast_per_block_scale_to_fp8", "e2m1_values", "e2m1_bounds"}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = {"torch": torch, "BaseQuantizedTensor": ReferenceStorage, "E4M3_MAX": 448.0}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["NVFP4QTensor"]
