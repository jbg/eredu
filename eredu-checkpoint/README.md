# eredu-checkpoint

`eredu-checkpoint` defines backend-neutral checkpoint contracts for Eredu. It
describes stored tensor encodings, logical tensor recipes, checkpoint schemas,
selection plans, and encoded tensor leases without allocating backend tensors
or executing accelerator operations.

The crate includes portable SafeTensors and GGUF validation and storage
interfaces. Its canonical SafeTensors shard discovery requires every index
mapping to exactly match the referenced shard headers and confines resolved
payloads to the checkpoint access root. Neutral stores use the same canonical
index parsing and path admission, but validate and buffer a payload shard only
when a caller requests one; selective distributed loads therefore do not read
remote-only shards. Artifact inspection and conversion tooling use strict
discovery. Architecture crates use these types to declare checkpoint intent;
backend implementations decide how selected values are materialized and where
they reside.

Most applications should use
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Use this crate directly
when implementing checkpoint tooling, an architecture integration, or a
concrete backend.

## License

Licensed under either Apache-2.0 or MIT.
