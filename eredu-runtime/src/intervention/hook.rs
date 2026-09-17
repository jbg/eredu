//! Shared declaration-order selection for activation hooks.
use super::*;
#[derive(Clone, Copy)]
/// Shared prospective decision for one declaration at an actual activation hook.
pub enum ActivationHook {
    /// Another hook, routed provider, or explicitly inactive operation.
    Unrelated,
    /// First occurrence of this declared dense activation.
    Active,
    /// The same active declaration was already attempted.
    Repeated,
}
/// Select a declared action without allocating or changing its outcome.
pub fn activation_hook(
    operation: &InterventionOperation,
    point: &InterventionPoint,
    outcome: &InterventionOutcome,
    path: &str,
) -> ActivationHook {
    if operation.target != path
        || point.routing.is_some()
        || point.routed_units.is_some()
        || *outcome == InterventionOutcome::Inactive
    {
        ActivationHook::Unrelated
    } else if *outcome == InterventionOutcome::Missing {
        ActivationHook::Active
    } else {
        ActivationHook::Repeated
    }
}
