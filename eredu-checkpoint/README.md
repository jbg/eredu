# eredu-checkpoint

`eredu-checkpoint` defines backend-neutral checkpoint contracts for Eredu. It
describes stored tensor encodings, logical tensor recipes, checkpoint schemas,
selection plans, and encoded tensor leases without allocating backend tensors
or executing accelerator operations.

The crate includes portable SafeTensors and GGUF validation and storage
interfaces. Its canonical SafeTensors shard discovery rejects malformed index
mappings and confines resolved payloads to the checkpoint access root; artifact
inspection, stores, and conversion tooling share that facility. Architecture
crates use these types to declare checkpoint intent; backend implementations
decide how selected values are materialized and where they reside.

Most applications should use
[`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Use this crate directly
when implementing checkpoint tooling, an architecture integration, or a
concrete backend.

## License

Licensed under either Apache-2.0 or MIT.
