//! Actual integer values enter through the retained request's initialized input
//! factory; empty metadata uses an explicit empty slice of its paid unit seed.
use super::*;
use safemlx::{Array, PreparedInputPlan};

impl OriginalParallelControlProjection {
    pub(crate) fn prepare_expert_route_input(&self, values: &[i32], completed: &eredu_core::ErasedSharedStorageOwner,
        context: &Group, stream: &Stream) -> Result<Array, Error> {
        self.with_context(context, |active| {
            let c = &self.custody;
            reserve(&c.funding, &[
                size_of::<(&Self, &[i32], &Group, &Stream)>(),
                size_of::<[usize; 2]>(), size_of::<[[i32; 2]; 3]>(),
                size_of::<[Array; 1]>(), size_of::<PreparedInputPlan<'_>>(),
                size_of::<Result<PreparedInputPlan<'_>, safemlx::PreparedInputCause>>(),
                size_of::<Result<Array, Error>>(), failure_control_bytes().ok_or_else(overflow)?,
                Array::static_slice_control_bytes().ok_or_else(overflow)?,
            ])?;
            if active.is_none() {
                return Err(control_error(ControlCause::Identity, &c.source, &c.funding));
            }
            let retained = self.upgrade().map_err(|cause| Error::with_original_control_source(cause, false))?;
            let owner = retained.owner();
            if !completed.downcast_ref::<super::peer_counts::table::PeerCountTable>()
                .is_some_and(|table| table.belongs_to(owner, c)) {
                return Err(control_error(ControlCause::Identity, &c.source, &c.funding));
            }
            let input = owner.request.source.agreement_inputs()
                .ok_or_else(|| control_error(ControlCause::Identity, &c.source, &c.funding))?;
            let shape = [values.len().max(1), 1];
            let plan = if values.is_empty() {
                input.runtime().zeros(safemlx::Dtype::Int32, &shape)
            } else { input.runtime().i32(values, &shape) }
                .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
            // Publish the original immutable birth before any numerical or
            // sparse-capture alias can escape into the caller.
            let value = super::sampling::token::construct(owner, plan)?;
            if values.is_empty() {
                value.try_slice(&[0, 0], &[0, 1], &[1, 1], stream)
                    .map_err(|cause| failure(Cause::Native(cause), &c.source, &c.funding))
            } else { Ok(value) }
        })?
    }
}
