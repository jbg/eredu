# Eredu collections

A dependency-free, `no_std` foundation for the safe ordered AVL map shared by
portable source construction and the local JSON fork. Eredu owns this map's
implementation; upstream JSON archive provenance and licenses are recorded
under `third-party`.

`Map::try_insert_with` calls the supplied policy with the exact new node layout
before `Box::new`. Existing-key replacement, rotations, removals and iterators
allocate no backing. Keys, values, diagnostics, traversal controls and the
lifetime of their funding remain the consumer's responsibility. Ordinary
insertion uses the same worker with an infallible callback. The map uses no
unsafe code, admission accounts, native resources or higher-layer dependencies.

AVL height bounds recursion and fixed iterator frontiers by the addressable
node population. Mutable and owning iterators use fixed safe storage; their
concrete control sizes are visible to callers through the iterator types.
