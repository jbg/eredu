# Reference registry allocation inventory

Cold JSON Schema compilation builds a referencing registry before it creates its
keyword compilation context. The selected dependency is referencing 0.52.1,
archive SHA-256
`d38a014525040cdc9893361b7419bcf1f43b7ba7055eabaf6d91d6b75caa8b3f`.
Its canonical registry builder, crawl/index pipeline and resolver must own their
prospective allocation facts; a quote of the completed registry is insufficient.

The first audit identified these producers:

- Registry builder pending maps and URI keys, default retriever ownership, and
  lifetime-conversion `collect` calls that rebuild pending maps unnecessarily.
- Resource-document and known-resource maps; crawl queues, deferred-reference
  vectors, visited sets, pointer strings, anchors and stored-resource owners.
- URI normalization/resolution caches, cloned cache keys, shared resolved URIs,
  registry clones, dynamic-scope list nodes and copied vocabulary sets.
- URI/parser/reference diagnostic strings and boxed external error causes.
- Lazy meta-schema/default URI initialization reached during cold construction.

The URI worker itself delegates to fluent-uri 0.4.1. Its normalizer allocates an
output String and a path String; its resolver allocates an output String. Those
allocations happen before referencing receives a result. Narrow hooks therefore
belong inside that dependency's existing normalize/resolve workers as well.
Copies of completed URI values do not cover these earlier producers.

The intended policy is a borrowed neutral allocation callback and fixed refusal
marker, propagated through the same ordinary registry, URI and resolver workers.
Ordinary entry points supply an unenforced policy. The schema compiler retains
its original funding owner through compilation, published validators and terminal
errors; a stack adapter borrows that owner for the registry's lifetime. There is
no backend dependency, separate resolver, successful fallback after refusal, or
caller-supplied byte estimate.

The map storage needs the pinned hashbrown prospective layout query before each
actual table growth. Vector/string growth must reserve the complete replacement
request before mutation; fixed boxes and Arc bodies must be charged before they
exist. URI and registry failures retain the original semantic errors when
funding permits their diagnostics, while funding refusal uses a fixed marker.
The implemented path and evidence follow below.

## Implemented producer path

`referencing` 0.52.1 is the selected workspace fork, with the workspace unsafe-code
prohibition. The selected fluent-uri 0.4.1 dependency retains its own existing
safe-source boundary. Both original archives, file hashes, VCS metadata and
licenses are retained. The registry pending/known/document/visited collections
now use the pinned hashbrown tables, whose prospective layout query owns the
actual bucket/control allocation calculation. Inline index maps promote through
that same table producer, reserving before replacing the inline entries.

A borrowed `Allocation` policy flows from registry construction through URI
cache growth, indexing, pointer and anchor storage, deferred traversal, vocabulary
copies, resolver ownership and dynamic-scope nodes. Original APIs use explicit
`Unenforced` over the same workers. Typed fallible entry points preserve refusal;
failed semantic diagnostics admit their strings before copying them. Existing
shared owners are cloned without charging a second allocation. Callers retain
their original account through every returned value and error.

The URI normalizer/resolver now admit their original output/path buffers and
reached replacements, including IPv6 output that grows beyond the input length.
Owned URI conversion moves the normalized string instead of parsing and copying
it again. Fragment replacement reserves before mutation; refusal leaves the
original URI intact. JSON pointer unescaping, percent decoding and diagnostics
use the same prospective producers. Pointer and anchor strings retain their
paid String storage without an extra shrink-to-box conversion.

Removed surfaces include the pending-map lifetime-conversion rebuilds, the
separate static-registry index construction path, unused deep registry/builder/
cache/index clones, and private unfunded pointer-construction wrappers. Ordinary
and admitted registry preparation use one crawl/index pipeline. Borrowed and
owned source storage remain distinct ownership mechanisms. The recursion spill
threshold is 32 frames; this changes when existing paid work-list storage is
used, without rejecting deeper schemas. Reference targets outside recognized
schema keywords now use a paid depth-first work list too.

Bundled meta-schemas are parsed from their unchanged static bytes into the actual
source owner's Value storage, through the shared serde JSON producer. Registry
imports do not adopt lazy global parsed values. The public ordinary meta-schema
constants use the same parser with `Unenforced`. `source_for_draft` exposes the
immutable primary bytes to the schema compiler without parsing or allocation.

External retrieval has an explicit prospective producer contract. An enforced
opaque callback is refused before invocation; the default no-fetch retriever
has a fixed, allocation-free error. An application can implement the funded
retrieval contract. Asynchronous external retrieval still has opaque boxed
future and join storage, so that specific enforced operation returns a fixed
unqualified error before constructing it. Self-contained asynchronous registry
preparation shares the paid producers. This is a transport callback limitation,
not a limitation of built-in schema/tool declarations.

## Validation

With the actual Rust 1.98 compiler and incremental compilation disabled:

- `cargo test -p referencing --all-features --lib`: 148 tests passed. Coverage
  includes upstream synchronous/asynchronous resolution, every reached allocation
  refusal for borrowed and owned source construction, URI/pointer/anchor error
  paths, inline-map promotion rollback, deep deferred traversal, a 5,000-level
  reference target, and refusal before an opaque callback is entered.
- The bundled 2020-12 registry fixture made 982 prospective requests totaling
  92,023 cumulative requested bytes. All 982 refusal positions returned the
  original fixed allocation marker and made no later request. This is cumulative
  reservation accounting, not a claim about retained heap or total process RSS.
- Fluent URI unit/integration suites pass in default, `net`, and minimal
  `alloc,impl-error` configurations. New tests refuse each reached normalization,
  resolution, owned-copy and fragment-growth producer.
- The independent URI probe verifies all 31 original archive file hashes and
  compares 11,760 URI/IRI normalization, resolution and semantic diagnostic
  results against the pristine 0.4.1 library. All match exactly. The oracle is
  linked only into the validation executable.

Reproduce the independent comparison after building the fluent URI tests:

```sh
RUSTC="$(rustup which rustc)" bash third-party/fluent-uri-0.4.1/validation/run-reference.sh \
  /path/to/pristine/fluent-uri-0.4.1 /tmp/uri-reference \
  /path/to/fluent-uri-test-target/debug/deps
```

These dependency checks establish the reached registry/URI producers. The schema
compiler and facade must still propagate their retained source account through
these APIs; passing these tests alone does not establish a complete cold grammar
or public generation bound.
