//! Closed prospective restrictions preserve an admitted controller baseline.
use super::*;

/// A controller whose pending choice only restricts its admitted decision.
///
/// Implementations must preserve source identities, semantic state and committed
/// history. Clearing a choice restores the same admitted baseline. Workspace
/// declarations must cover both original and restricted masks simultaneously;
/// preparation uses the supplied actual funding before any allocation. These
/// operations cannot alter sampling configuration, RNG, or mutable model state.
pub trait ProspectiveTokenController: TokenFilterController {
    /// Typed refusal preserving any independently retained preparation storage.
    type ChoiceError: std::error::Error + 'static;
    /// Validates and stages one restriction; failure preserves the prior choice.
    fn stage_choice(
        &mut self,
        token: u32,
        funding: &HostMetadataFunding,
    ) -> Result<(), Self::ChoiceError>;
    /// Restores the admitted decision without changing its semantic predicate.
    fn clear_choice(&mut self) -> bool;
}

/// A completed and drained boundary permitting only prospective restrictions.
/// Unlike arbitrary controller mutation, this cannot broaden or replace the
/// admitted source policy and therefore preserves its preparation identity.
pub struct TextTokenChoiceBoundary<'a, C: ProspectiveTokenController> {
    controller: &'a mut C,
}
impl<C: ProspectiveTokenController> TextTokenChoiceBoundary<'_, C> {
    /// Stages a validated one-token restriction through the controller's worker.
    pub fn force_next(
        &mut self,
        token: u32,
        funding: &HostMetadataFunding,
    ) -> Result<(), C::ChoiceError> {
        self.controller.stage_choice(token, funding)
    }
    /// Removes a restriction, restoring the same admitted source decision.
    pub fn clear_forced(&mut self) -> bool {
        self.controller.clear_choice()
    }
}
impl<B: TextGenerationBackend, C: ProspectiveTokenController> ControlledTextGeneration<'_, B, C> {
    /// Checks the same completed/drained boundary used for original state copies.
    /// This grants no general controller mutation or additional work allowance.
    pub fn token_choice_boundary(
        &mut self,
    ) -> Result<TextTokenChoiceBoundary<'_, C>, TextContinuationError<B::Error, C::Error>> {
        self.snapshot_source()?;
        Ok(TextTokenChoiceBoundary {
            controller: &mut self.inner.controller,
        })
    }
}
