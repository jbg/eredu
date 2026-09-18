# Chat session consolidation inventory

The pre-change inventory found two consumers of the retained tokenizer/template
source compiler: `LoadedModel::prepare_chat` and a separate text-only managed-chat
request/start/generate route. The latter skipped tool declaration compilation and
started the plain cursor through private text-chat adapters. It represented no
separate native execution capability and has been removed.

The canonical result is one retained `PreparedChat`, prepared before rendering
with its original profile/declarations, followed by `PreparedChatRequest` and
`start_prepared_chat`. Manual `advance` and uninterrupted `run` share the same
committed cursor. Original source compilation, cancellation, named templates,
default/caller bindings, capacity, empty-input refusal, stops and owner retirement
remain covered. Speculation consumes the same semantic preparation.

Differences preserved at the canonical boundary:

- The canonical request retains `skip_special_tokens`, defaulting to true, and
  passes it to the same semantic decoder.
- Canonical literal stops borrow `String` entries rather than `&str` entries;
  tests own their stop declarations at the caller boundary.
- Canonical output retains token IDs and emits semantic text events. Text parity
  assertions belong on those events; output ownership assertions retain actual
  token storage rather than manufacturing a second owned output string.
- Tools/reasoning pass through the actual prepared policy. The migrated tests
  distinguish invalid declarations/history from supported policy while
  preserving early refusal and no-submission assertions.
- The older allocating prepared-chat request and driver have also been removed.
  `start_prepared_chat(...).run(...)` provides uninterrupted completion.

Consumers found before edits: private `original_chat.rs` tests, public managed,
profile and released-template fixtures, and the native managed-plain lifecycle
chat fixture. Portable facade/backend conformance consumers have another owner.
Completed: the old managed chat request/start/generate route and private
text-only startup adapters are removed. Literal output is an explicit
`PreparedChatOutputMode::Text` policy on the canonical request, preparation and
cursor; `Semantic` remains the default. Text eligibility uses the retained
profile's actual support checks, and both policies use the same original decoder
with the requested skip-special-token behavior. The source compiler remains.

The final portable original-chat filter passes 23 tests; one existing external
reference oracle is ignored. This includes exact named-template selection,
cancellation/refusal, invalid EOS rejection, source/copy retirement, caller stops,
manual/run parity, and released profile fixtures. Shared runtime token-domain
checks pass six tests, including overlapping added-token IDs: the logical domain
uses actual consistent IDs while retaining the full admitted backing capacity.
The native lifecycle fixture was migrated but its hardware validation is tracked
by the native owner.
