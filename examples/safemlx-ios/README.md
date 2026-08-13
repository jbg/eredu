# SafeMLX iOS demo

A minimal SwiftUI application that downloads MLX-format SafeTensors model
repositories from Hugging Face, keeps them in an app-local cache, and streams
SafeMLX generation to the screen.

The Xcode project is generated rather than committed. The application has one
Swift package dependency:
[`huggingface/swift-huggingface`](https://github.com/huggingface/swift-huggingface),
which provides repository snapshots, resumable downloads, progress, and the
Python-compatible Hugging Face cache layout.

## Prerequisites

- Xcode 26 or newer with the Metal Toolchain component
- Rust 1.89 or newer
- CMake
- [XcodeGen](https://github.com/yonaskolb/XcodeGen)

Install the Metal toolchain if necessary:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Generate and run

From this directory:

```sh
./scripts/bootstrap.sh
open SafeMLXDemo.xcodeproj
```

Select your development team and an attached iPhone, then run the
`SafeMLXDemo` scheme. The generated project and native build products remain
ignored by Git.

The default repository is
`mlx-community/LFM2.5-1.2B-Instruct-4bit`. A smaller first-device smoke test is
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

Enter a dedicated MLX-format Hugging Face model repository. The downloader
intentionally fetches model configuration, tokenizer assets, chat templates,
and SafeTensors weights while excluding unrelated PyTorch weights and GGUF
variants that may coexist in a repository. Public repositories need no token.
After generation, the status line reports model load time, time to first token,
and decode throughput in tokens per second.

## Architecture

- `ModelStore` owns the Hugging Face cache index and the screen state.
- `SafeMLXEngine` wraps a small C ABI exposed by the Rust `staticlib` crate.
- Each loaded model lives on one dedicated Rust thread. Loading and generation
  therefore preserve SafeMLX's thread-affine runtime state while Swift remains
  asynchronous.
- The bridge uses checkpoint generation defaults, falling back to SafeMLX's
  defaults and a 256-token limit when the checkpoint declares no limit.
- Model weights are fully resident. This is the normal fast path for the small
  mobile models this demo targets.

The Hugging Face cache is under the application's `Library/Application Support`
directory and is excluded from device backups. The app discovers complete model
snapshots from that cache at launch rather than relying on persisted absolute
sandbox paths. Existing downloads made by earlier demo builds are migrated from
`Library/Caches` on first launch. Deleting a model in the UI removes its
repository cache.
