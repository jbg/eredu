# Unicode and class allocation inventory

The opening inventory records the pre-consolidation state. The implementation
and validation results below describe its completed replacement.

The syntax parser has one existing HIR class and interval implementation. Its
ordinary constructors, normalization, case folding, unions/intersections,
differences, negation, byte/Unicode conversion and Unicode property lookup still
allocate without consulting the caller's retained parser funding. Canonical
methods will retain these algorithms and accept the shared borrowed `Allocator`;
ordinary entry points will call those workers with `Unenforced`.

Every reached vector growth, owned class copy, string normalization and conversion
must be reserved before allocation. Case-fold table unavailability stays distinct
from allocation refusal. No fallback to an ordinary constructor follows refusal.
Append-based interval operations must roll back their provisional suffix before
returning failure, preserving the original canonical class. Symmetric difference sequences completed
intersection/union/difference operations and preserves a canonical intermediate
class if a later stage refuses. Canonicalization
will sort without auxiliary allocation and merge/truncate in place, removing its
hidden stable-sort and append destination allocations for both policies.

This slice owns `hir/interval.rs`, Unicode property producers, and the existing
Class/ClassUnicode/ClassBytes range methods. HIR smart constructors, translator,
properties and iterative teardown are owned by the tokenizer slice.


Implemented the shared fallible workers, including range copies, literal/ASCII
conversions, normalization scratch, static Unicode property expansion, age unions,
case folding and all set operations. Errors preserve the first fixed allocation
refusal, independently of missing Unicode tables. Private unused ordinary interval
wrappers were removed; public ordinary class methods use the same workers.

Validation on 2026-09-18: direct qualified rustc test binaries passed all 157 tests
with every Unicode feature, all 152 tests with no features, and all 153 tests
with only general-category tables. New tests reject
every reached allocation in set/property/case-fold construction, verify no callback
after refusal, verify canonical retained failure state and compare all 256 byte
memberships against independent set equations. Existing upstream translator,
class and Unicode tests passed in both profiles. Logs are
`/tmp/native-unicode-tests.log` and `/tmp/native-unicode-tests-minimal.log`.
