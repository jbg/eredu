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

The portable pure-Rust `fancy-regex` tokenizer engine is always available, so
`--no-default-features` is a valid configuration. Default features additionally
enable Oniguruma (which the tokenizer selects when both engines are present)
and the fast SentencePiece-compatible path.

## License

Licensed under either Apache-2.0 or MIT.
