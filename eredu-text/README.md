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
enable Oniguruma for general regex profiles and the fast SentencePiece-compatible
path. Recognized tokenizer expressions use one shared compiled workspace source
for ordinary and admitted execution. With `onig`, those programs preserve the
pinned Oniguruma semantics, including its word-class differences.

Chat rendering preserves JSON object insertion order. Its `tojson` filter uses
Hugging Face's Python JSON defaults, including spaces after commas and colons,
literal Unicode, and no HTML escaping. Templates can override `ensure_ascii`,
`indent`, `separators`, and `sort_keys`. These details affect prompt tokens and
model behavior, particularly when tool schemas are rendered into the prompt.

## Admitted tokenizer sources

The original source and ID workers support deterministic BPE, WordLevel and
Unigram; ordered NFC, Prepend, Lowercase and literal Replace; and repeated or
nested sequences of admitted ByteLevel, Digits, Metaspace, Whitespace and exact
Split profiles. Metaspace decoding and the selected byte/fallback decoder forms
share their ordinary component workers. Source compilation checks the complete
configuration and returns typed profile refusals for other programs.

See [composition coverage and limits](../doc/bounded-text-processing.md)
for independent-reference, funding-refusal and public-generation evidence,
including the scoped census of published tokenizer configurations. These tests
do not establish arbitrary regex, decoder, stochastic Unigram or training bounds.

## License

Licensed under either Apache-2.0 or MIT.
