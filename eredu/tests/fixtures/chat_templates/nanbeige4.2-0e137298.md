# Nanbeige 4.2 template and tokenizer fixture

Source: `Nanbeige/Nanbeige4.2-3B`, revision
`0e137298720f7241e83b8aabecc4263dcc7d84b3` (Apache-2.0).

`nanbeige4.2-0e137298.jinja` is the unmodified `chat_template` string from
the released `tokenizer_config.json`, including its leading and trailing newlines.

`nanbeige4.2-0e137298-tokenizer.json` retains the released added tokens and IDs,
special flags, Metaspace pre-tokenizer, normalization, byte-fallback decoder and
post-processor. To keep this portable fixture small, its BPE vocabulary contains
only ASCII characters, the Metaspace marker, byte-fallback tokens and the added
markers; merges are empty. Added markers also appear in the sparse base vocabulary
so deserialization retains their original IDs. This fixture tests token identity,
decoding, whitespace and constraints; it does not reproduce model tokenization.
The native tests and `chat_probe` use the complete released tokenizer.

Original file SHA-256:

| File | SHA-256 |
| --- | --- |
| `tokenizer.json` | `1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f` |
| `tokenizer_config.json` | `3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518` |
