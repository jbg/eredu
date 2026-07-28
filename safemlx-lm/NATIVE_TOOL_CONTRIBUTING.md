# Contributing native tool formats

Native tool support is fail-closed and fixture-driven. A model family name or
recognizable marker is not sufficient evidence for a registration.

## Add an audited fixture

1. Pin the authoritative repository and immutable revision.
2. Copy the selected template body byte for byte into
   `tests/fixtures/chat_templates/`. Preserve upstream whitespace. If the
   repository file needs a terminator that is not part of the template body,
   document and remove that terminator in the signature test.
3. Add its source, revision, selected-template name, whitespace notes, and any
   shared-body evidence to `tests/fixtures/chat_templates/README.md`.
4. Add a golden rendering test containing tools, prior calls and results,
   parallel calls where supported, reasoning controls, and both generation
   prompt modes.

Do not silently update a fixture in place. A changed body is a new audited
signature and should retain old registration only while that exact released
body remains intentionally supported.

## Add a registration

Add one exact signature constant and one registry entry in `src/chat.rs`.
Registration identity should name the wire format and include a short revision
or signature suffix. Extend the registry uniqueness test and the
cross-dialect golden matrix. Tests must prove that a one-byte modification is
unsupported and that duplicate matching signatures are ambiguous.

Architecture metadata, model IDs, repository names, template markers, and
regex guesses must not participate in profile selection.

## Prefer declarative features

Use `DeclarativeDialectSpec` when the response can be described by exact
envelopes, delimited reasoning/text channels, JSON object or list payloads,
fixed JSON function fields, named JSON arguments, structural-object quoting,
call separators, and finite protocol call limits.

For every new declarative feature:

1. Validate contradictory or overlapping configuration in `validate`.
2. Generate both the constraint grammar and incremental parser behavior from
   the same field.
3. Add valid, malformed, incomplete, overlapping-stop, and structural-token
   cases.
4. Run the golden response through every UTF-8-safe byte split and every
   tokenizer-piece boundary.
5. Assert canonical semantic events and that protocol syntax is never emitted
   as visible text.

Keep schema validation in `tool_constraints.rs`; do not embed
architecture-specific argument regexes in public APIs.

## Add a custom dialect

Use a private `FormatDialect` implementation only when the released surface
cannot be represented declaratively. The implementation must provide:

- exact generation-prompt behavior and reasoning control;
- a constraint configuration generated from request tools;
- an exact optional auto-activation trigger;
- required structural special tokens and profile stop sequences;
- an independent incremental parser producing only canonical
  `SemanticEvent`s.

Add exhaustive split-boundary tests, malformed transitions, incomplete calls,
parallel indexing, Unicode, nested values, and stop overlap. Then register the
custom implementation by exact template signature. Never expose the dialect,
parser, constraint engine, structural regex, or llguidance types publicly.

## Runtime and checkpoint validation

Extend ordinary/canonical-MTP/lookahead-MTP semantic equivalence and
single/multi-request scheduler-interleaving equivalence tests. If stream
placement is relevant, add ignored Metal tests for CPU-draft/GPU-target and
same-GPU split streams.

Add an ignored real-checkpoint smoke test keyed by a documented environment
variable. It must load the checkpoint, prepare one valid tool request, assert
the expected registered profile prefix, and fail clearly when the checkpoint
path is missing. Never download checkpoints from a test.

Before committing, run:

```sh
CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/mlx-lm" cargo fmt --all -- --check
CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/mlx-lm" cargo clippy --locked -p safemlx-lm --all-targets -- -D warnings
CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/mlx-lm" cargo test --locked -p safemlx-lm --lib
CARGO_BUILD_BUILD_DIR="$HOME/Library/Caches/cargo-build/mlx-lm" cargo +1.88.0 check --locked -p safemlx-lm --all-targets -p safemlx-lm-cli
```

Metal-dependent ignored tests must run on a Metal host outside a restricted
sandbox.
