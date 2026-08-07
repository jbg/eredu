//! Architecture-independent distributed execution infrastructure.

/// Cartesian communicator, transport, generation, and consensus contexts.
pub mod cartesian;
/// Expert assignment, routing, and exchange mechanics.
pub mod expert;
/// Architecture-neutral tensor-parallel planning and execution contexts.
pub mod parallel;
/// Runtime topology, placement planning, and selective checkpoint loading.
pub mod topology;
