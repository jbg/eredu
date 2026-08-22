//! Architecture-neutral distributed MLX model composition.

/// Routed-expert execution shared by distributed stages.
pub(crate) mod expert;
/// Executable distributed-stage adapters for TP, PP, EP, and Cartesian layouts.
pub mod pipeline;
