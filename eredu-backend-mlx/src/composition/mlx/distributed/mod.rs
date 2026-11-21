//! Architecture-neutral distributed MLX model composition.

/// Routed-expert execution shared by distributed stages.
#[cfg(feature = "distributed")]
pub(crate) mod expert;
