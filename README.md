# SafeMLX

> **eredu** *noun, Basque* — “model; pattern.”

SafeMLX is an unofficial Rust workspace for building
[MLX](https://github.com/ml-explore/mlx) applications and
running local machine-learning models without a Python runtime. It combines a
safe array API, model loading and generation, bounded checkpoint readers, and
neural audio codecs.

Apple silicon with Metal is the primary platform. The core crates also support
CPU and NVIDIA CUDA builds on x86-64 Linux and Windows.

SafeMLX is independent of Apple and is not an official MLX project.

## Choose a crate

| Crate | Use it for |
| --- | --- |
| [`safemlx`](safemlx/) | Arrays, operations, transforms, neural-network modules, optimizers, streams, and distributed MLX |
| [`eredu`](eredu/) | Loading and running supported text, multimodal, and realtime speech models |
| [`eredu-core`](eredu-core/) | Backend-neutral LM contracts, scheduling, validation, and portable schemas |
| [`eredu-gguf`](eredu-gguf/) | Bounded, framework-independent GGUF reading, writing, and tensor conversion |
| [`safemlx-codec`](safemlx-codec/) | Neural audio codecs, including Mimi |
| [`eredu-text`](eredu-text/) | Tokenizer and chat-template utilities |
| [`safemlx-sys`](safemlx-sys/) | Low-level MLX C bindings and native build integration |

The workspace also contains procedural-macro crates, integration tests, and a
[`eredu` command-line example](examples/eredu-cli/).

## Quick start

The minimum supported Rust version is 1.89. Add the core crate to your project:

```toml
[dependencies]
safemlx = "0.1.3"
```

MLX operations are lazy. Build an array graph on a stream, then evaluate it
before reading values on the host:

```rust
use safemlx::{array, Device, DeviceType, Stream};

let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let left = array!([1.0, 2.0, 3.0]);
let right = array!([4.0, 5.0, 6.0]);
let sum = left.add(&right, &stream)?.into_evaluated()?;

assert_eq!(sum.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
# Ok::<(), safemlx::error::Exception>(())
```

To try a supported local language model:

```sh
cargo run --release -p eredu-cli -- \
  --model /path/to/model \
  "Write a Rust function that adds two integers."
```

The CLI accepts local Hugging Face-style model directories, GGUF files, and
model identifiers already present in the local Hugging Face cache. It does not
download models.

## Documentation

- [Documentation index](doc/README.md)
- [Model and checkpoint support](doc/model-support.md)
- [Language-model backend architecture](doc/backend-architecture.md)
- [Model loading, quantization, and memory](doc/model-loading.md)
- [Parallel execution](doc/parallel-execution.md)
- [Platform setup](doc/platforms.md)
- [CLI guide](examples/eredu-cli/README.md)

All published crates are pre-1.0, so applications should expect API changes
between minor releases.

## License

The workspace crates are available under MIT or Apache-2.0 unless a crate or
vendored component states otherwise. See [`LICENSE-MIT`](LICENSE-MIT),
[`LICENSE-APACHE`](LICENSE-APACHE), and the notices inside vendored source
trees.
