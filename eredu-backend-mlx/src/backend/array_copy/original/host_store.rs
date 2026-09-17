//! Finite Host destinations of the same independently admitted copy program.
use super::*;
use safemlx::{
    ImmutableHostTransferBuffer, PreparedHostCopyDestination, PreparedHostTransferPlan,
    PreparedInputArena, PreparedInputRuntime,
};
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Population {
    pub count: usize,
    pub backing: usize,
    pub compaction: usize,
    pub direct: usize,
    pub controls: usize,
}
impl Population {
    pub fn add(&mut self, plan: &SavedHostCopyPlan) -> Result<(), OriginalCopyCause> {
        *self = Self {
            count: self
                .count
                .checked_add(1)
                .ok_or(OriginalCopyCause::Overflow)?,
            backing: self
                .backing
                .checked_add(plan.backing)
                .ok_or(OriginalCopyCause::Overflow)?,
            compaction: self
                .compaction
                .checked_add(plan.logical)
                .ok_or(OriginalCopyCause::Overflow)?,
            direct: self
                .direct
                .checked_add(plan.direct)
                .ok_or(OriginalCopyCause::Overflow)?,
            controls: self
                .controls
                .checked_add(plan.controls)
                .ok_or(OriginalCopyCause::Overflow)?,
        };
        Ok(())
    }
    fn take(&mut self, plan: &SavedHostCopyPlan) -> Result<(), OriginalCopyCause> {
        // Every field is checked before the paired finite debit. A failed native
        // construction never restores this slot or recovers source capacity.
        *self = Self {
            count: self
                .count
                .checked_sub(1)
                .ok_or(OriginalCopyCause::Capacity)?,
            backing: self
                .backing
                .checked_sub(plan.backing)
                .ok_or(OriginalCopyCause::Capacity)?,
            compaction: self
                .compaction
                .checked_sub(plan.logical)
                .ok_or(OriginalCopyCause::Capacity)?,
            direct: self
                .direct
                .checked_sub(plan.direct)
                .ok_or(OriginalCopyCause::Capacity)?,
            controls: self
                .controls
                .checked_sub(plan.controls)
                .ok_or(OriginalCopyCause::Capacity)?,
        };
        Ok(())
    }
    pub fn empty(self) -> bool {
        self.count == 0
            && self.backing == 0
            && self.compaction == 0
            && self.direct == 0
            && self.controls == 0
    }
}

/// Source-derived constructor declaration; it grants no source or native work.
/// The copied source binder retains its actual canonical row and operand order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SavedHostCopyPlan {
    shape: [i32; 4],
    dtype: Dtype,
    metadata: usize,
    backing: usize,
    logical: usize,
    direct: usize,
    controls: usize,
}
impl SavedHostCopyPlan {
    pub(crate) fn from_array(
        source: &Array,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, OriginalCopyCause> {
        let descriptor = source.try_descriptor()?;
        if descriptor.facts().allocation().is_none() {
            return Err(OriginalCopyCause::UnsettledSource);
        }
        Self::inspect(descriptor.shape(), descriptor.facts().dtype(), environment)
    }
    pub(crate) fn from_host(
        source: &ImmutableHostTransferBuffer,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, OriginalCopyCause> {
        let descriptor = source.try_fixed_descriptor::<4>()?;
        if descriptor.policy() != safemlx::HostTransferPolicy::Transfer
            || descriptor.storage_kind() != safemlx::HostTransferStorageKind::MetalShared
        {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        Self::inspect(descriptor.shape(), descriptor.dtype(), environment)
    }
    fn inspect(
        shape: &[i32],
        dtype: Dtype,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, OriginalCopyCause> {
        let shape: [i32; 4] = shape
            .try_into()
            .map_err(|_| OriginalCopyCause::UnknownLayout)?;
        if shape.iter().any(|n| *n <= 0)
            || !matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let runtime = environment.input_runtime()?;
        let plan = PreparedHostTransferPlan::new(&runtime, &shape, dtype, 0)?;
        let (native, direct) = PreparedHostCopyDestination::original_layout(4, dtype)
            .ok_or(OriginalCopyCause::UnknownLayout)?;
        let arena = PreparedInputArena::layout::<WorkspaceCopyRetention>(plan.metadata_bytes())?;
        let parts = [
            native,
            plan.control_bytes().ok_or(OriginalCopyCause::Overflow)?,
            arena.total_bytes().ok_or(OriginalCopyCause::Overflow)?,
            PreparedHostCopyDestination::control_bytes().ok_or(OriginalCopyCause::Overflow)?,
            WorkspaceContext::metadata_arc_bytes::<ImmutableHostTransferBuffer>()
                .ok_or(OriginalCopyCause::Overflow)?,
            Array::descriptor_control_bytes().ok_or(OriginalCopyCause::Overflow)?,
            OriginalBufferBudget::inspection_control_bytes().ok_or(OriginalCopyCause::Overflow)?,
            OriginalScopeObserver::control_bytes().ok_or(OriginalCopyCause::Overflow)?,
            size_of::<(&OriginalScopeObserver, &Array)>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<SavedHostCopyPlan, OriginalCopyCause>>(),
            size_of::<Result<(), OriginalCopyFailure>>(),
            size_of::<(&PreparedSavedHostCopy, Option<&Array>)>(),
            size_of::<(&Array, &OriginalCopyEnvironment<'_>)>(),
            size_of::<(&ImmutableHostTransferBuffer, &OriginalCopyEnvironment<'_>)>(),
            size_of::<(&[i32], Dtype, &OriginalCopyEnvironment<'_>)>(),
            safemlx::HostTransferDescriptor::<4>::control_bytes()
                .ok_or(OriginalCopyCause::Overflow)?,
            size_of::<Self>(),
            size_of::<PreparedSavedHostCopy>(),
            size_of::<Population>(),
            size_of::<(
                PreparedHostTransferPlan<'_>,
                PreparedInputRuntime,
                PreparedInputArena,
            )>(),
            size_of::<(
                Option<OriginalScopeObserver>,
                Result<(), OriginalCopyCause>,
                Result<ImmutableHostTransferBuffer, safemlx::PreparedHostCopyError>,
            )>(),
            size_of::<Result<PreparedSavedHostCopy, OriginalCopyFailure>>(),
            size_of::<(
                &Array,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut PreparedSavedHostCopy,
            )>(),
            BackendFailure::source_retention_peak_bytes::<OriginalCopyFailure>()
                .ok_or(OriginalCopyCause::Overflow)?,
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(OriginalCopyCause::Overflow)?;
        Ok(Self {
            shape,
            dtype,
            metadata: plan.metadata_bytes(),
            backing: plan.backing_bytes(),
            logical: plan.logical_bytes(),
            direct,
            controls,
        })
    }
}

/// Native roots/events retire before the completed buffer and its exact copy
/// account. This whole prefix belongs in the existing saved-copy Recovery owner.
pub(crate) struct PreparedSavedHostCopy {
    destination: PreparedHostCopyDestination,
    completed: Option<Arc<ImmutableHostTransferBuffer>>,
    plan: SavedHostCopyPlan,
    budget: OriginalBufferBudget,
    attempted: bool,
    custody: WorkspaceCopyRetention,
}
impl PreparedOriginalCopy {
    pub(crate) fn prepare_host_destination(
        &mut self,
        plan: SavedHostCopyPlan,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<PreparedSavedHostCopy, OriginalCopyFailure> {
        let custody = self.custody.clone();
        let result = (|| {
            custody
                .validate_pool(environment.pool())
                .map_err(|_| OriginalCopyCause::ForeignPool)?;
            self.host_stores.take(&plan)?;
            let runtime = environment.input_runtime()?;
            let native = PreparedHostTransferPlan::new(&runtime, &plan.shape, plan.dtype, 0)?;
            if native.metadata_bytes() != plan.metadata
                || native.backing_bytes() != plan.backing
                || native.logical_bytes() != plan.logical
            {
                return Err(OriginalCopyCause::UnknownLayout);
            }
            let quota = PreparedSubmissionGraphQuota::try_new(plan.metadata, custody.clone())
                .map_err(|e| e.into_parts().0)?;
            let arena = PreparedInputArena::try_allocate(quota).map_err(|e| e.into_parts().0)?;
            let destination = native.construct_copy_destination(&arena)?;
            Ok(PreparedSavedHostCopy {
                destination,
                completed: None,
                plan,
                budget: self.buffer.clone(),
                attempted: false,
                custody: custody.clone(),
            })
        })();
        result.map_err(|cause| OriginalCopyFailure {
            cause,
            _custody: custody,
        })
    }
}
impl PreparedSavedHostCopy {
    /// Requires an actual allocation born under this copy's native budget;
    /// another original Model, copy, or byte-equivalent source is refused.
    pub(crate) fn run(
        &mut self,
        copied: &Array,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<(), OriginalCopyFailure> {
        let result = (|| -> Result<(), OriginalCopyCause> {
            if self.attempted {
                return Err(OriginalCopyCause::UnknownLayout);
            }
            let observer =
                OriginalScopeObserver::try_current()?.ok_or(OriginalCopyCause::UnknownLayout)?;
            observer.validate_completed_array(copied)?;
            let witness = self
                .budget
                .inspect_array(copied)?
                .ok_or(OriginalCopyCause::UnknownLayout)?;
            let descriptor = copied.try_descriptor()?;
            if descriptor.shape() != self.plan.shape
                || descriptor.facts().dtype() != self.plan.dtype
                || descriptor.facts().allocation() != Some(witness.allocation())
            {
                return Err(OriginalCopyCause::UnknownLayout);
            }
            drop(descriptor);
            drop(witness);
            {
                let retained = roots
                    .try_borrow()
                    .map_err(|_| OriginalCopyCause::UnknownLayout)?;
                if retained.len() == retained.capacity() {
                    return Err(OriginalCopyCause::Capacity);
                }
            }
            self.attempted = true;
            let result = self
                .destination
                .submit(copied, stream, &observer)
                .map(|_| ());
            // A failed native submission can still create its completion root.
            if let Some(root) = self.destination.output() {
                roots.borrow_mut().push(root.try_clone_handle()?);
            }
            result?;
            self.destination.synchronize()?;
            // The wait settles the actual copy event; the shared completed-only
            // observer then detaches that matching event from its ArrayDesc.
            // Cold storage inspection intentionally does neither operation and
            // must continue refusing evaluated descriptors with a live event.
            // Keep the destination/root prefix intact if validation refuses.
            observer.validate_completed_array(
                self.destination
                    .output()
                    .ok_or(OriginalCopyCause::UnknownLayout)?,
            )?;
            let buffer = self.destination.take_completed()?;
            let descriptor = buffer.try_fixed_descriptor::<4>()?;
            if descriptor.shape() != self.plan.shape
                || descriptor.dtype() != self.plan.dtype
                || descriptor.allocation().bytes() != self.plan.backing
            {
                return Err(OriginalCopyCause::UnknownLayout);
            }
            self.completed = Some(Arc::new(buffer));
            Ok(())
        })();
        result.map_err(|cause| OriginalCopyFailure {
            cause,
            _custody: self.custody.clone(),
        })
    }
    /// Same completed native output for the existing final copy root union.
    /// Borrowing it grants no source provenance or reuse of this destination.
    pub(crate) fn completed_output(&self) -> Option<&Array> {
        self.completed.as_ref()?;
        self.destination.output()
    }
    /// Closed completed-copy witness. The terminal publisher must match both
    /// the actual immutable allocation and this exact copy's buffer budget.
    pub(crate) fn completed_for(
        &self,
        budget: &OriginalBufferBudget,
    ) -> Option<&Arc<ImmutableHostTransferBuffer>> {
        self.budget.same_budget(budget).then_some(())?;
        self.completed.as_ref()
    }
}
