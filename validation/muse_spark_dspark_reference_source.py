"""Pinned serving-source adapters for whole-assistant reference acceptance."""
import ast
import hashlib
import torch
from torch import nn

REVISION = "6cc692b5a87a3f33c7fc5431d8add0640cb032f6"


SOURCE_SHA256 = "0721fdde4a3f3cf4355f3ec46976409193645848938f321f86bd7b6e39341b55"


def record(value):
    return dict(shape=list(value.shape), values=value.float().flatten().tolist())


SOURCES = {
    "vllm/model_executor/layers/rotary_embedding/base.py":
        "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5",
    "csrc/pos_encoding_kernels.cu":
        "23b62c68866ad41385cc3bb6156648f76e87b86b15b33dbd9a65f530d97d8e8c",
    "vllm/model_executor/layers/activation.py":
        "bebeddbd4997f49f54471ee31ac42b22c3c11dfe25244e55d20c36f390489c05",
    "vllm/third_party/flash_linear_attention/ops/fused_norm_gate.py":
        "cb50b1826c93a619aaea568b8601e29bad3745f5a41c14cbebfe2208710eb08d",
    "csrc/activation_kernels.cu":
        "4e0a96e587fe808661ecef8f66038b58e00f6ba17d779f430be0a19beac7e673",
}


BODY_SHA256 = "53fc17a1be255fe9983d47e257fa9d7f5781e441bc63de9cdbe220d4e963b4dd"


def definitions(path, expected, names, namespace):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("reference source hash differs")
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError("reference definition inventory differs")
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def rope(x, positions):
    frequency = 500000. ** (-torch.arange(0, x.shape[-1], 2).float() / x.shape[-1])
    phase = positions.float()[:, None] * frequency[None]
    cos, sin = phase.cos()[None], phase.sin()[None]
    a, b = x.chunk(2, -1)
    if x.dtype == torch.bfloat16:
        cos, sin = cos.to(x.dtype).float(), sin.to(x.dtype).float()
        a, b = a.float(), b.float()
    return torch.cat((a*cos - b*sin, b*cos + a*sin), -1).to(x.dtype)


def head_norm(x):
    if x.dtype == torch.bfloat16:
        work = x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


HEAD_SHA256 = "7be21c094a49520c41f418aceea71b338d5e15b14b3ddd052e6d6f7b7ac8ae79"


class ParallelLMHead(nn.Module):
    def __init__(self, vocab, width, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab, width))


class ReplicatedLinear(nn.Linear):
    def __init__(self, input_dim, output_dim, bias=False, params_dtype=None, **kwargs):
        super().__init__(input_dim, output_dim, bias=bias, dtype=params_dtype)
