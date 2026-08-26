#!/usr/bin/env python3
"""Export greedy parity tensors from NVIDIA's released PersonaPlex 7B-v1.

This generator never creates a synthetic model or configuration. It requires:

* a Rust-loadable artifact directory containing the strict
  ``{"model_type":"personaplex","version":"7b-v1"}`` configuration;
* the released PyTorch SafeTensors checkpoint (one file or a shard directory);
* NVIDIA's PersonaPlex reference source files; and
* Kyutai Moshi's ``gating.py`` and ``rope.py`` source files.

Example:

  python eredu-backend-mlx/validation/personaplex_torch_fixture.py \
    --artifact-dir /models/personaplex-7b-v1 \
    --checkpoint /models/personaplex-7b-v1 \
    --personaplex-source /src/personaplex \
    --moshi-source /src/kyutai-moshi/moshi \
    --output /fixtures/personaplex-7b-v1-greedy.safetensors

Keep ``--output`` outside ``--artifact-dir``. PersonaPlex checkpoints are
directory-loaded, so a fixture beside model shards would be mistaken for model
weights by any strict loader.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tempfile
from pathlib import Path


PERSONAPLEX_DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
RELEASED_TEXT_CARD = 32_000
RELEASED_AUDIO_CARD = 2_048
RELEASED_TOTAL_CODEBOOKS = 16
RELEASED_DEPTH_CODEBOOKS = 16
RELEASED_GENERATED_CODEBOOKS = 8
RELEASED_DIM = 4_096
RELEASED_FEED_FORWARD = 16_896
RELEASED_HEADS = 32
RELEASED_LAYERS = 32
RELEASED_CONTEXT = 3_000
RELEASED_DEPTH_DIM = 1_024
RELEASED_DEPTH_FEED_FORWARD = 4_224
RELEASED_DEPTH_HEADS = 16
RELEASED_DEPTH_LAYERS = 6
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a released PersonaPlex 7B-v1 PyTorch parity fixture."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Rust-loadable released PersonaPlex 7B-v1 artifact directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Released SafeTensors file or shard directory inside artifact-dir",
    )
    parser.add_argument(
        "--personaplex-source",
        type=Path,
        required=True,
        help="Directory containing personaplex_lm.py, personaplex_transformer.py, and personaplex_streaming.py",
    )
    parser.add_argument(
        "--moshi-source",
        type=Path,
        required=True,
        help="Kyutai Moshi checkout/package root containing modules/gating.py and modules/rope.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parity SafeTensors path outside artifact-dir",
    )
    parser.add_argument("--steps", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=314_159)
    parser.add_argument(
        "--require-torch-version",
        help="Fail unless the Python torch package has this exact version",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device used for the released model (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve(strict=True)
    personaplex_source = args.personaplex_source.resolve(strict=True)
    moshi_source = args.moshi_source.resolve(strict=True)
    checkpoint_files = validate_released_inputs(
        artifact_dir,
        args.checkpoint.resolve(strict=True),
        personaplex_source,
        moshi_source,
        args.output.resolve(strict=False),
    )

    # Heavy imports follow argument and artifact validation. `--help` and
    # rejected-profile checks therefore work without a PyTorch environment.
    import numpy as np
    import torch
    from safetensors.torch import save_file

    torch_version = importlib.metadata.version("torch")
    if (
        args.require_torch_version is not None
        and torch_version != args.require_torch_version
    ):
        raise SystemExit(
            f"torch {args.require_torch_version} is required, but {torch_version} is installed"
        )

    with tempfile.TemporaryDirectory(prefix="personaplex-7b-v1-ref-") as package_root:
        package_root = Path(package_root)
        build_reference_package(
            package_root, personaplex_source, moshi_source
        )
        sys.path.insert(0, str(package_root))
        from moshi.models.lm import LMGen, LMModel  # type: ignore

        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        # Build on `meta` so loading the released 7B state does not first
        # allocate a second full random model on CPU/GPU.
        with torch.device("meta"):
            model = released_model(LMModel)
        load_released_state(model, checkpoint_files, device)
        model.eval()

        tensors = teacher_forced_tensors(
            model, args.batch_size, args.steps, args.seed + 2, device, np, torch
        )

        rng = np.random.default_rng(args.seed + 1)
        input_audio = torch.tensor(
            rng.integers(
                0,
                RELEASED_AUDIO_CARD,
                size=(args.batch_size, 8, args.steps),
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=device,
        )

        gen = LMGen(
            model,
            device=device,
            use_sampling=False,
            temp=0.0,
            temp_text=0.0,
            check=True,
            report_loss=False,
            return_logits=False,
        )
        sampled = []
        emitted = []
        emitted_steps = []
        with torch.inference_mode(), gen.streaming(args.batch_size):
            for step in range(args.steps):
                old_offset = gen._streaming_state.offset
                out = gen.step(input_tokens=input_audio[:, :, step : step + 1])
                if old_offset > 0:
                    target_position = old_offset % gen._streaming_state.cache.shape[2]
                    # Rust exposes the text decision followed by the eight
                    # generated-side audio decisions. The other eight depth
                    # slots are occupied by the input-side stream.
                    sampled.append(
                        gen._streaming_state.cache[
                            :, : 1 + RELEASED_GENERATED_CODEBOOKS, target_position
                        ].to(torch.int32)
                    )
                if out is not None:
                    emitted.append(
                        out[:, 1 : 1 + RELEASED_GENERATED_CODEBOOKS, 0].to(
                            torch.int32
                        )
                    )
                    emitted_steps.append(step)

        if sampled:
            expected_sampled = torch.stack(sampled, dim=2)
        else:
            expected_sampled = torch.zeros(
                (args.batch_size, 1 + RELEASED_GENERATED_CODEBOOKS, 0),
                dtype=torch.int32,
                device=device,
            )
        if emitted:
            expected_output_audio = torch.stack(emitted, dim=2)
        else:
            expected_output_audio = torch.zeros(
                (args.batch_size, RELEASED_GENERATED_CODEBOOKS, 0),
                dtype=torch.int32,
                device=device,
            )

        tensors.update({
            "generation.input_audio": input_audio.to(device="cpu", dtype=torch.int32).contiguous(),
            "generation.expected_sampled": expected_sampled.to(device="cpu").contiguous(),
            "generation.expected_emitted_steps": torch.tensor(
                emitted_steps, dtype=torch.int32, device="cpu"
            ),
            "generation.expected_output_audio": expected_output_audio.to(device="cpu").contiguous(),
        })
        tensors.update(
            prompt_parity_tensors(
                model, LMGen, args.batch_size, args.seed + 3, device, np, torch
            )
        )
        metadata = {
            "profile": "personaplex-7b-v1",
            "reference": "nvidia-personaplex-pytorch",
            "seed": str(args.seed),
            "steps": str(args.steps),
            "batch_size": str(args.batch_size),
            "checkpoint_sha256": checkpoint_digest(checkpoint_files),
            "reference_source_sha256": checkpoint_digest(
                reference_source_files(personaplex_source, moshi_source)
            ),
            "torch_version": torch_version,
            "safetensors_version": importlib.metadata.version("safetensors"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, args.output, metadata=metadata)
        print(
            f"wrote {len(tensors)} released PersonaPlex 7B-v1 fixture tensors "
            f"to {args.output}"
        )


def validate_released_inputs(
    artifact_dir: Path,
    checkpoint: Path,
    personaplex_source: Path,
    moshi_source: Path,
    output: Path,
) -> list[Path]:
    config_path = artifact_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"released artifact is missing {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    expected = {"model_type": "personaplex", "version": "7b-v1"}
    if config != expected:
        raise SystemExit(
            "released PersonaPlex config must be exactly "
            f"{expected!r}; got {config!r}. Synthetic/alternate profiles are not admitted."
        )

    checkpoint_files = safetensor_files(checkpoint)
    artifact_checkpoint_files = safetensor_files(artifact_dir)
    if {path.resolve() for path in checkpoint_files} != {
        path.resolve() for path in artifact_checkpoint_files
    }:
        raise SystemExit(
            "--checkpoint must select exactly the SafeTensors files that the Rust "
            "artifact directory loader will consume"
        )
    for path in checkpoint_files:
        try:
            path.resolve().relative_to(artifact_dir)
        except ValueError as error:
            raise SystemExit(
                f"checkpoint shard {path} is outside artifact-dir {artifact_dir}; "
                "the Rust parity command must load the same released files"
            ) from error
    try:
        output.relative_to(artifact_dir)
    except ValueError:
        pass
    else:
        raise SystemExit(
            "--output must be outside --artifact-dir because PersonaPlex loads all "
            "SafeTensors shards from that directory"
        )

    reference_source_files(personaplex_source, moshi_source)
    return checkpoint_files


def safetensor_files(checkpoint: Path) -> list[Path]:
    if checkpoint.is_file():
        if checkpoint.suffix != ".safetensors":
            raise SystemExit(f"checkpoint file must end in .safetensors: {checkpoint}")
        return [checkpoint]
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint path is neither a file nor directory: {checkpoint}")

    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as index_file:
            payload = json.load(index_file)
        names = sorted(set(payload.get("weight_map", {}).values()))
        if not names:
            raise SystemExit(f"SafeTensors index has no weight_map entries: {index}")
        files = [checkpoint / name for name in names]
    else:
        files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no SafeTensors checkpoint files found in {checkpoint}")
    for path in files:
        require_file(path, "checkpoint shard")
    return files


def released_model(lm_model_cls):
    return lm_model_cls(
        delays=PERSONAPLEX_DELAYS,
        n_q=RELEASED_TOTAL_CODEBOOKS,
        dep_q=RELEASED_DEPTH_CODEBOOKS,
        card=RELEASED_AUDIO_CARD,
        text_card=RELEASED_TEXT_CARD,
        dim=RELEASED_DIM,
        num_heads=RELEASED_HEADS,
        hidden_scale=RELEASED_FEED_FORWARD / RELEASED_DIM,
        norm="rms_norm_f32",
        norm_emb=False,
        bias_proj=False,
        depformer_dim=RELEASED_DEPTH_DIM,
        depformer_dim_feedforward=RELEASED_DEPTH_FEED_FORWARD,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_pos_emb="none",
        existing_text_padding_id=3,
        context=RELEASED_CONTEXT,
        num_layers=RELEASED_LAYERS,
        causal=True,
        positional_embedding="rope",
        max_period=10_000,
        gating="silu",
        depformer_num_heads=RELEASED_DEPTH_HEADS,
        depformer_num_layers=RELEASED_DEPTH_LAYERS,
        depformer_max_period=10_000,
        depformer_gating="silu",
    )


def teacher_forced_tensors(
    model, batch_size: int, steps: int, seed: int, device, np, torch
):
    """Emit frozen temporal/layer/head/depth observations in Rust fixture layout."""
    rng = np.random.default_rng(seed)
    text = torch.tensor(
        rng.integers(
            0,
            RELEASED_TEXT_CARD,
            size=(batch_size, 1, steps),
            dtype=np.int64,
        ),
        dtype=torch.long,
        device=device,
    )
    audio = torch.tensor(
        rng.integers(
            0,
            RELEASED_AUDIO_CARD,
            size=(batch_size, RELEASED_TOTAL_CODEBOOKS, steps),
            dtype=np.int64,
        ),
        dtype=torch.long,
        device=device,
    )
    codes = torch.cat([text, audio], dim=1)
    initial = model._get_initial_token().expand(batch_size, -1, -1)
    delayed_streams = []
    for codebook, delay in enumerate(model.delays):
        delayed = codes[:, codebook].roll(delay, dims=1)
        if delay > 0:
            delayed[:, :delay] = initial[:, codebook]
        delayed_streams.append(delayed)
    delayed = torch.stack(delayed_streams, dim=1)
    pattern = torch.cat([initial, delayed], dim=2)
    temporal_codes = pattern[:, :, :-1]
    depth_inputs = pattern[:, : model.dep_q, 1:]

    layer_outputs = [None] * len(model.transformer.layers)
    handles = []
    for layer_index, layer in enumerate(model.transformer.layers):
        def capture(_module, _args, output, index=layer_index):
            layer_outputs[index] = output.detach()

        handles.append(layer.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            temporal_input = model.embed_codes(temporal_codes)
            temporal, text_logits = model.forward_embeddings(temporal_input)
            audio_logits = model.forward_depformer_training(pattern[:, :, 1:], temporal)
    finally:
        for handle in handles:
            handle.remove()

    tensors = {
        "input.text": temporal_codes[:, :1].permute(2, 0, 1).to(
            device="cpu", dtype=torch.int32
        ).contiguous(),
        "input.audio": temporal_codes[:, 1:].permute(2, 0, 1).to(
            device="cpu", dtype=torch.int32
        ).contiguous(),
        "input.depth": depth_inputs.permute(2, 0, 1).to(
            device="cpu", dtype=torch.int32
        ).contiguous(),
    }
    for step in range(steps):
        tensors[f"expected.{step}.temporal_input"] = temporal_input[
            :, step : step + 1
        ].to(device="cpu").contiguous()
        for layer_index, output in enumerate(layer_outputs):
            if output is None:
                raise RuntimeError(f"temporal layer {layer_index} produced no observation")
            tensors[f"expected.{step}.temporal_layer.{layer_index}"] = output[
                :, step : step + 1
            ].to(device="cpu").contiguous()
        tensors[f"expected.{step}.text_logits"] = text_logits[
            :, 0, step : step + 1
        ].to(device="cpu").contiguous()
        for slice_index in range(model.dep_q):
            tensors[f"expected.{step}.audio_logits.{slice_index}"] = audio_logits[
                :, slice_index, step : step + 1
            ].to(device="cpu").contiguous()
    return tensors


def prompt_parity_tensors(
    model, lm_gen_cls, batch_size: int, seed: int, device, np, torch
):
    """Run voice, text, mixed, and flush frames through the released LMGen."""
    rng = np.random.default_rng(seed)
    voice = rng.integers(
        0,
        RELEASED_AUDIO_CARD,
        size=(batch_size, RELEASED_GENERATED_CODEBOOKS),
        dtype=np.int64,
    )
    mixed_user = rng.integers(
        0,
        RELEASED_AUDIO_CARD,
        size=(batch_size, RELEASED_GENERATED_CODEBOOKS),
        dtype=np.int64,
    )
    mixed_agent = rng.integers(
        0,
        RELEASED_AUDIO_CARD,
        size=(batch_size, RELEASED_GENERATED_CODEBOOKS),
        dtype=np.int64,
    )
    sine = np.broadcast_to(SINE_TOKENS, (batch_size, RELEASED_GENERATED_CODEBOOKS))
    silence = np.broadcast_to(SILENCE_TOKENS, (batch_size, RELEASED_GENERATED_CODEBOOKS))
    user = np.stack([sine, sine, mixed_user, sine, sine], axis=2)
    agent = np.stack([voice, silence, mixed_agent, silence, silence], axis=2)
    text = np.broadcast_to(
        np.array([3, 101, 202, 3, 3], dtype=np.int64),
        (batch_size, 5),
    )[:, None, :]
    user_tensor = torch.tensor(user, dtype=torch.long, device=device)
    agent_tensor = torch.tensor(agent, dtype=torch.long, device=device)
    text_tensor = torch.tensor(text, dtype=torch.long, device=device)

    gen = lm_gen_cls(
        model,
        device=device,
        use_sampling=False,
        temp=0.0,
        temp_text=0.0,
        check=True,
        report_loss=False,
        return_logits=False,
    )
    sampled = []
    emitted = []
    emitted_steps = []
    with torch.inference_mode(), gen.streaming(batch_size):
        for step in range(user_tensor.shape[2]):
            old_offset = gen._streaming_state.offset
            out = gen.step(
                input_tokens=user_tensor[:, :, step : step + 1],
                moshi_tokens=agent_tensor[:, :, step : step + 1],
                text_token=text_tensor[:, 0, step],
            )
            if old_offset > 0:
                target_position = old_offset % gen._streaming_state.cache.shape[2]
                sampled.append(
                    gen._streaming_state.cache[
                        :, : 1 + RELEASED_GENERATED_CODEBOOKS, target_position
                    ].to(torch.int32)
                )
            if out is not None:
                emitted.append(
                    out[:, 1 : 1 + RELEASED_GENERATED_CODEBOOKS, 0].to(torch.int32)
                )
                emitted_steps.append(step)

    return {
        "prompt.user_audio": user_tensor.to(device="cpu", dtype=torch.int32).contiguous(),
        "prompt.agent_audio": agent_tensor.to(device="cpu", dtype=torch.int32).contiguous(),
        "prompt.text": text_tensor.to(device="cpu", dtype=torch.int32).contiguous(),
        "prompt.expected_sampled": torch.stack(sampled, dim=2).to(device="cpu").contiguous(),
        "prompt.expected_emitted_steps": torch.tensor(
            emitted_steps, dtype=torch.int32, device="cpu"
        ),
        "prompt.expected_output_audio": torch.stack(emitted, dim=2).to(
            device="cpu"
        ).contiguous(),
    }


def load_released_state(model, checkpoint_files: list[Path], device) -> None:
    from safetensors.torch import load_file

    expected = set(model.state_dict())
    observed: set[str] = set()
    for path in checkpoint_files:
        shard = load_file(str(path), device=str(device))
        duplicate = observed.intersection(shard)
        if duplicate:
            raise SystemExit(
                f"checkpoint repeats {len(duplicate)} state keys; first: {sorted(duplicate)[0]}"
            )
        unexpected = set(shard).difference(expected)
        if unexpected:
            raise SystemExit(
                f"checkpoint has {len(unexpected)} unexpected state keys; first: {sorted(unexpected)[0]}"
            )
        model.load_state_dict(shard, strict=False, assign=True)
        observed.update(shard)
    missing = expected.difference(observed)
    if missing:
        raise SystemExit(
            f"checkpoint is missing {len(missing)} released state keys; first: {sorted(missing)[0]}"
        )
    unresolved = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if unresolved:
        raise SystemExit(
            f"released state left {len(unresolved)} tensors on meta; first: {unresolved[0]}"
        )


def checkpoint_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_reference_package(
    package_root: Path, personaplex_source: Path, moshi_source: Path
) -> None:
    moshi = package_root / "moshi"
    models = moshi / "models"
    modules = moshi / "modules"
    utils = moshi / "utils"
    tqdm_pkg = package_root / "tqdm"
    for path in [models, modules, utils, tqdm_pkg]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (moshi / "__init__.py").write_text("", encoding="utf-8")

    shutil.copyfile(personaplex_source / "personaplex_lm.py", models / "lm.py")
    shutil.copyfile(
        personaplex_source / "personaplex_transformer.py", modules / "transformer.py"
    )
    shutil.copyfile(
        personaplex_source / "personaplex_streaming.py", modules / "streaming.py"
    )
    shutil.copyfile(resolve_moshi_module(moshi_source, "gating.py"), modules / "gating.py")
    write_rope_module(
        modules / "rope.py", resolve_moshi_module(moshi_source, "rope.py")
    )
    write_compile_module(utils / "compile.py")
    write_sampling_module(utils / "sampling.py")
    (package_root / "sphn.py").write_text(
        "def read(*args, **kwargs): raise RuntimeError('sphn.read is unavailable in parity fixture generation')\n"
        "def resample(*args, **kwargs): raise RuntimeError('sphn.resample is unavailable in parity fixture generation')\n",
        encoding="utf-8",
    )
    (tqdm_pkg / "auto.py").write_text(
        "def tqdm(iterable=None, *args, **kwargs):\n"
        "    return iterable if iterable is not None else []\n",
        encoding="utf-8",
    )


def resolve_moshi_module(source: Path, name: str) -> Path:
    candidates = [
        source / "modules" / name,
        source / "moshi" / "modules" / name,
        source / "moshi" / "moshi" / "modules" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"Moshi reference source is missing {name}; checked {rendered}")


def reference_source_files(personaplex_source: Path, moshi_source: Path) -> list[Path]:
    files = [
        personaplex_source / "personaplex_lm.py",
        personaplex_source / "personaplex_transformer.py",
        personaplex_source / "personaplex_streaming.py",
        resolve_moshi_module(moshi_source, "gating.py"),
        resolve_moshi_module(moshi_source, "rope.py"),
    ]
    for path in files[:3]:
        require_file(path, "PersonaPlex reference source")
    return files


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is missing: {path}")


def write_compile_module(path: Path) -> None:
    path.write_text(
        "from contextlib import contextmanager\n\n"
        "def torch_compile_lazy(fn):\n"
        "    return fn\n\n"
        "@contextmanager\n"
        "def no_compile():\n"
        "    yield\n\n"
        "class CUDAGraphed:\n"
        "    def __init__(self, fn, disable=True):\n"
        "        self.fn = fn\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self.fn(*args, **kwargs)\n",
        encoding="utf-8",
    )


def write_sampling_module(path: Path) -> None:
    path.write_text(
        "def sample_token(logits, use_sampling, temp, top_k):\n"
        "    return logits.argmax(dim=-1)\n",
        encoding="utf-8",
    )


def write_rope_module(path: Path, source_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    signature = "def __init__(self, interleave: bool, max_period: float = 10000.0):"
    if signature in source:
        source = source.replace(
            signature,
            "def __init__(self, interleave: bool = True, max_period: float = 10000.0):",
        )
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
