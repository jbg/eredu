# eredu-checkpoint

`eredu-checkpoint` defines backend-neutral checkpoint contracts for Eredu. It
describes stored tensor encodings, logical tensor recipes, checkpoint schemas,
selection plans, and encoded tensor leases without allocating backend tensors
or executing accelerator operations.

The crate includes portable SafeTensors and GGUF validation, stable
filesystem-backed artifact fingerprinting, and storage interfaces. Its
canonical SafeTensors shard discovery requires every index
mapping to exactly match the referenced shard headers and confines resolved
payloads to the checkpoint access root. Neutral stores use the same canonical
index parsing and path admission, but validate and buffer a payload shard only
when a caller requests one. On first access, the complete opened shard header
must exactly match every index mapping for that shard; unopened remote-only
shards remain unread. Artifact inspection and conversion tooling use strict
discovery. Architecture crates use these types to declare checkpoint intent;
backend implementations decide how selected values are materialized and where
they reside.

`PreparedCheckpointSource` binds the exact admitted catalog, shard identities,
provenance, selections, and bounded lease geometry to the source used for later
payload reads. Preparation records each admitted file's identity and inspects
only its metadata and header; it does not retain a descriptor per shard. The
snapshot includes canonical path and Unix change time (plus device and inode);
targets whose standard metadata API exposes no equivalent change counter
retain the strongest available length and timestamp metadata. After recipe and
destination preflight, only a requested shard is reopened and validated, then
its exact selected ranges are admitted by matching bounded reads. Immutable
cached bytes protect published leases from later path or in-place substitution.

Most applications should use
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Use this crate directly
when implementing checkpoint tooling, an architecture integration, or a
concrete backend.

## License

Licensed under either Apache-2.0 or MIT.
