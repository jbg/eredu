# eredu-text

`eredu-text` contains backend-neutral tokenizer and chat-template
utilities for Rust language-model runtimes. It supports structured messages,
system roles, tool metadata, Jinja chat templates, GGUF tokenizer metadata,
Kimi-style tiktoken vocabularies, and the tokenizer backends used by
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu).

Most applications should depend on `eredu`, which exposes these
capabilities through its higher-level loading and generation APIs. Use this
crate directly when integrating the tokenizer or template layer into another
runtime. It does not depend on a concrete backend or native accelerator
runtime.

The portable pure-Rust `fancy-regex` tokenizer engine is always available, so
`--no-default-features` is a valid configuration. Default features additionally
enable Oniguruma (which the tokenizer selects when both engines are present)
and the fast SentencePiece-compatible path.

## License

Licensed under either Apache-2.0 or MIT.
