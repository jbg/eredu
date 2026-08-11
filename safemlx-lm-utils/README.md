# safemlx-lm-utils

`safemlx-lm-utils` contains tokenizer and chat-template utilities for Rust MLX
language-model runtimes. It supports structured messages, system roles, tool
metadata, Jinja chat templates, and the tokenizer backends used by
[`safemlx-lm`](../safemlx-lm/).

Most applications should depend on `safemlx-lm`, which exposes these
capabilities through its higher-level loading and generation APIs. Use this
crate directly when integrating the tokenizer or template layer into another
runtime.

```toml
[dependencies]
safemlx-lm-utils = "0.1"
```

Default features enable the Oniguruma and fast SentencePiece-compatible
tokenizer paths. Consult the crate features in `Cargo.toml` when a smaller or
more portable build is required.

## License

Licensed under either Apache-2.0 or MIT.
