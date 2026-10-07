#!/usr/bin/env python3
"""Execute pinned NVIDIA feature methods with CPU Torch/librosa, without Eredu.

Produces a compact synthetic-component fixture, not released-weight validation.
"""
import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Tuple
import numpy as np
import librosa
import torch
from torch import nn

REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
SOURCE = "nemo/collections/asr/parts/preprocessing/features.py"
SHA256 = "db0ba4b9f3592ed35d4e6e70e095c9d3e12d07df0810d0d183f26e9fe1f7a0ca"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.checkout / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256
    selected = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {"FilterbankFeatures", "normalize_batch"}]
    namespace = dict(torch=torch, nn=nn, math=math, random=random, logging=logging,
                     librosa=librosa, CONSTANT=1e-5)
    exec(compile(ast.Module(body=selected, type_ignores=[]), SOURCE, "exec"), namespace)
    processor = namespace["FilterbankFeatures"](
        sample_rate=16000, n_window_size=400, n_window_stride=160, window="hann",
        normalize="NA", n_fft=512, preemph=0.97, nfilt=128, lowfreq=0,
        highfreq=8000, log=True, log_zero_guard_type="add", log_zero_guard_value=2**-24,
        dither=0, pad_to=0, exact_pad=False, mag_power=2.0, mel_norm="slaney").eval()
    # Integer arithmetic gives exactly representable broadband input in both runtimes.
    pcm = torch.tensor([(((i * 73 + i * i * 19) % 2048) - 1024) / 8192
                        for i in range(6400)], dtype=torch.float32)
    cases = []
    max_precision_error = 0.0
    # Includes the first reflected row, live right edge, rollover left edge, and
    # the cached wrapper's 17-column (9 cached + 8 new) selection after rollover.
    for name, start, end, first, stop in [
        ("initial", 0, 1280, 0, 1),
        ("second_tick", 0, 2560, 0, 9),
        ("finite_right_edge", 0, 2560, 15, 17),
        ("rollover_left_edge", 1280, 6400, 0, 2),
        ("rollover_selection", 1280, 6400, 9, 26),
    ]:
        x = pcm[start:end][None]
        lengths = torch.tensor([end-start])
        values, _ = processor(x, lengths)
        high, _ = processor(x.double(), lengths)
        high = high.float()
        max_precision_error = max(max_precision_error, (values-high).abs().max().item())
        cases.append(dict(name=name, start_sample=start, end_sample=end,
                          first_frame=first, end_frame=stop,
                          values=values[0, :, first:stop].T.contiguous().flatten().tolist()))
    # Fixed from independent F32/F64 precision evidence before Eredu comparison.
    tolerance = 4e-4
    assert max_precision_error < tolerance / 4, max_precision_error
    cache_path = "nemo/collections/speechlm2/inference/model_wrappers/perception_cache.py"
    cache_hash = "96308c2057296a291ef517b413638400669cd8d94807cf135e6dc7021c72b9d9"
    cache_source = (args.checkout / cache_path).read_bytes()
    assert hashlib.sha256(cache_source).hexdigest() == cache_hash
    manager = next(n for n in ast.parse(cache_source).body
                   if isinstance(n, ast.ClassDef) and n.name == "PerceptionCacheManager")
    step = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    namespace.update(time=time, Tuple=Tuple, PerceptionCacheState=SimpleNamespace)
    exec(compile(ast.Module(body=[step], type_ignores=[]), cache_path, "exec"), namespace)
    schedule = []
    class RecordingEncoder:
        _cfg = SimpleNamespace(att_context_size=[70, 0])
        def cache_aware_stream_step(self, **kw):
            mel = kw["processed_signal"]
            schedule.append(dict(frames=mel.shape[-1],
                                 drop_extra_pre_encoded=kw["drop_extra_pre_encoded"],
                                 values=mel[0].T.contiguous().flatten().tolist()))
            return (mel, kw["processed_signal_length"], None, None, None)
    perception = SimpleNamespace(encoder=RecordingEncoder(),
        modality_adapter=lambda audio_signal, length: (audio_signal, length), proj=lambda x: x)
    def preprocessor(input_signal, length):
        return processor(input_signal, length)
    owner = SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(perception=perception)),
        device="cpu", preprocessor=preprocessor, pad_and_drop_preencoded=False,
        input_features=128, use_cudagraph=False, streaming_cfg=SimpleNamespace(
            chunk_size=[1, 8], shift_size=[1, 8], pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=2))
    state = SimpleNamespace(cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None)
    synchronize = torch.cuda.synchronize
    try:
        torch.cuda.synchronize = lambda: None  # CPU execution; numerical/slicing methods unchanged.
        for tick in range(4):
            end = (tick+1)*1280
            start = max(0, end-3*1280)
            namespace["step"](owner, pcm[start:end][None], tick, 1, state)
        default_schedule = schedule
        schedule = []
        owner.pad_and_drop_preencoded = True
        for tick in range(4):
            end = (tick+1)*1280
            start = max(0, end-3*1280)
            namespace["step"](owner, pcm[start:end][None], tick, 1, state)
        padded_schedule = schedule
        schedule = []
        owner.pad_and_drop_preencoded = False
        bf16_pcm = pcm.to(torch.bfloat16)
        bf16_precision_error = 0.0
        for tick in range(4):
            end = (tick+1)*1280
            start = max(0, end-3*1280)
            namespace["step"](owner, bf16_pcm[start:end][None], tick, 1, state)
            x = bf16_pcm[start:end][None].float()
            lengths = torch.tensor([end-start])
            normal, _ = processor(x, lengths)
            high, _ = processor(x.double(), lengths)
            bf16_precision_error = max(bf16_precision_error, (normal-high.float()).abs().max().item())
        bf16_schedule = schedule
        assert bf16_precision_error < tolerance / 4, bf16_precision_error
    finally:
        torch.cuda.synchronize = synchronize
    output = dict(cache_source_path=cache_path, cache_source_sha256=cache_hash,
                  cadence_samples=1280, buffer_frames=3, schedule=default_schedule,
                  padded_schedule=padded_schedule, bf16_schedule=bf16_schedule,
                  bf16_ingress_precision_max_abs=bf16_precision_error,
                  reference_revision=REVISION, source_path=SOURCE, source_sha256=SHA256,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  librosa_version=librosa.__version__, precision_max_abs=max_precision_error,
                  absolute_tolerance=tolerance, waveform_rule="((i*73+i*i*19)%2048-1024)/8192",
                  cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(f"Wrote {args.output}; F32/F64 max absolute error={max_precision_error:.8g}; tolerance={tolerance}")


if __name__ == "__main__":
    main()
