# Eredu collections

A dependency-free, `no_std` foundation for the safe ordered AVL map used by
portable source construction. Eredu owns its implementation and prospective
node-allocation contract.

`Map::try_insert_with` calls the supplied policy with the exact new node layout
before `Box::new`. Existing-key replacement, rotations, removals and iterators
allocate no backing. Keys, values, diagnostics, traversal controls and the
lifetime of their funding remain the consumer's responsibility. Ordinary
insertion uses the same worker with an infallible callback. The map uses no
unsafe code, admission accounts, native resources or higher-layer dependencies.

AVL height bounds recursion and fixed iterator frontiers by the addressable
node population. Mutable and owning iterators use fixed safe storage; their
concrete control sizes are visible to callers through the iterator types.
