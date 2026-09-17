//! Actual source queries; logical snapshot budgets do not pay host constructors.
use super::*;
use crate::HostPreparationAuthority;
use std::mem::size_of;

pub(super) fn sampler<E: SpeculativeExecutor, S: SpeculativeSampling>(
    source: &S,
    target: Option<&S::RandomState>,
    draft: Option<&S::DraftRandomness>,
    executor: &E,
    context: E::Context<'_>,
) -> Result<HostPreparationAuthority, SpeculativeControlError> {
    let parts = [
        source.control_snapshot_metadata_bytes(target, draft),
        Some(size_of::<(
            S,
            Option<S::RandomState>,
            Option<S::DraftRandomness>,
        )>()),
        Some(size_of::<
            Result<
                (S, Option<S::RandomState>, Option<S::DraftRandomness>),
                SpeculativeControlError,
            >,
        >()),
        Some(size_of::<(
            Option<&S::RandomState>,
            Option<&S::DraftRandomness>,
        )>()),
        Some(size_of::<HostPreparationAuthority>()),
    ];
    admit(parts, executor, context)
}
pub(super) fn constraint<E: SpeculativeExecutor, C: SpeculativeConstraint>(
    source: &C,
    executor: &E,
    context: E::Context<'_>,
) -> Result<HostPreparationAuthority, SpeculativeControlError> {
    admit(
        [
            source.control_snapshot_metadata_bytes(),
            Some(size_of::<C>()),
            Some(size_of::<Result<C, SpeculativeControlError>>()),
        ],
        executor,
        context,
    )
}
fn admit<E: SpeculativeExecutor, const N: usize>(
    parts: [Option<usize>; N],
    executor: &E,
    context: E::Context<'_>,
) -> Result<HostPreparationAuthority, SpeculativeControlError> {
    let bytes = parts
        .into_iter()
        .try_fold(size_of::<[Option<usize>; N]>(), |sum, part| {
            sum.checked_add(part?)
        })
        .and_then(|sum| {
            sum.checked_add(size_of::<
                Result<HostPreparationAuthority, SpeculativeControlError>,
            >())
        });
    // The ordinary hook ignores unknown queries. Original funding rejects them
    // before entering an unqualified sampler Clone or semantic fork callback.
    executor
        .driver_host_metadata(bytes, context)
        .map_err(|cause| {
            SpeculativeControlError::backend_with_retained(cause, E::take_retained_failure)
        })
}
