//! Paid immutable destination rows for the existing controlled record worker.
use super::*;
use eredu_core::{SpeculativeBuffer, SpeculativeValues};

pub(super) fn collect<E: SpeculativeExecutor, T>(
    values: impl ExactSizeIterator<Item = T>,
    executor: &E,
    context: E::Context<'_>,
    controls: usize,
) -> Result<SpeculativeValues<T>, SpeculativeControlError> {
    SpeculativeValues::collect_with_metadata(values, executor, context, controls)
        .map_err(|e| SpeculativeControlError::driver_with_retained(e, E::take_retained_failure))
}
pub(super) fn freeze<E: SpeculativeExecutor, T>(
    values: SpeculativeBuffer<T>,
    executor: &E,
    context: E::Context<'_>,
) -> Result<SpeculativeValues<T>, SpeculativeControlError> {
    SpeculativeValues::from_driver_buffer(values, executor, context, 0)
        .map_err(|e| SpeculativeControlError::driver_with_retained(e, E::take_retained_failure))
}
pub(super) fn push<E: SpeculativeExecutor, T>(
    values: &mut SpeculativeBuffer<T>,
    value: T,
    executor: &E,
    context: E::Context<'_>,
) -> Result<(), SpeculativeControlError> {
    if values.len() == values.capacity() {
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(1))
            .ok_or(ExecutionControlError::Overflow)?;
        let mut destination = executor.driver_buffer(capacity, context).map_err(|e| {
            SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
        })?;
        destination
            .try_extend(std::mem::take(values))
            .map_err(|_| SpeculativeControlError::Invalid("record destination capacity"))?;
        *values = destination;
    }
    values
        .try_push(value)
        .map_err(|_| SpeculativeControlError::Invalid("record destination capacity"))
}
