# Controlled record construction

Controlled delivery uses one custody-preserving `ControlledGenerationRecord`
family. Logical trace quotas and admission of its actual host storage remain
separate obligations.

## Canonical construction

The `record` worker accepts the actual `HostMetadataFunding`, reserves fixed
controls and each destination before construction, and retains that payer until
the final payload and shared shell retire. Fixed typed refusals retain the
account without allocating a diagnostic. Owned snapshot and branch fields move
into the record. Ordinary and enforced callers use this worker with an explicit
policy.

`PromptRecord::from_tokens` admits its shape, part, segment and token vectors
and shared attribution owner before construction. Core supplies the exact
shared-shell fact. Existing prepared/media attribution can be aliased only from
its genuine retained source. An alias grants no new source authority.

Serialized `CapturedStep` records require an explicit transaction outcome.
Low-level capture can explicitly report `Untracked`; committed and aborted
forwards preserve their distinct outcomes. Outcome-less older records are
rejected, and owned/shared records use the same serialization contract.

## Ownership boundaries

The facade owns record assembly, while core owns attribution layout and closed
retirement. The record producer does not certify allocation of event payloads
received by move, delivery context construction, semantic-prefix accumulation,
or snapshot continuation construction. Those producers need their own original
funding before invoking the record worker. Logical trace quotas remain separate
from host admission. The existing snapshot copy transaction remains separately
accounted and is not replaced by a record-family adapter.

The control/session owner supplies the actual source account to delivery.

## Implemented and checked

`record` returns a fixed
`RecordConstructionError`. Each context/instrumentation string is admitted
before a fallible exact reserve, followed by the exact shared record shell.
Owned snapshot and branch fields move into the record. `PromptRecord::from_tokens`
admits shape, part, segment and ID vectors, the fixed fingerprint destination,
the closed authority shell and core attribution publication before construction.
The ordinary fingerprint helper and prepaid destination use the same hash/hex
worker. Core supplies its own publication and hashing control facts.

The delivery owner passes the compiled chat's actual account;
refusal is embedded directly in the public controlled error instead of creating
an unpriced diagnostic Box. Three focused tests pass, exercising every reached
reservation cutoff, exact attribution/hash/range values, failure and alias
retirement, snapshot-field pointer preservation and diagnostic serialization.
Command: `cargo test -p eredu --no-default-features --lib --offline
api::control::records::tests::`. Log: `/private/tmp/contracts-record-producer-tests2.log`.

Moved event payloads must already retain their original production custody;
this worker admits only its actual new destinations. Media attribution remains
a separate source-derived producer in the same record family.
