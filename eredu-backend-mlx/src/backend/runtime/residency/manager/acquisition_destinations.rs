//! Cold metadata tickets from the actual window's canonical closure.
use super::*;
use crate::backend::runtime::checkpoint::store::{
    PreparedSourceAcquisitionFailure, PreparedSourceAcquisitions,
};
use eredu_checkpoint::{
    recipe::RecipeSourceVisitor,
    store::{ReadPolicy, RetainedCheckpointSource, TensorReadRequest, TensorSelection},
};
use eredu_runtime::{residency::ResidencyClosureSlot, working_memory::OriginalTextControlGuard};
use std::alloc::Layout;

pub(crate) struct SelectedSourceOccurrence {
    pub(crate) plan: eredu_checkpoint::store::SelectedGgufConversionPlan,
    pub(crate) multiplicity: usize,
    pub(crate) acquisition: eredu_checkpoint::store::SelectedGgufAcquisitionStorage,
}
struct Descriptor {
    source: RetainedCheckpointSource,
    key: String,
    selection: TensorSelection,
    multiplicity: usize,
}
fn overflow() -> ResidencyError {
    ResidencyError::ArithmeticOverflow {
        context: "prepared source acquisition population",
    }
}
impl Descriptor {
    fn backing_bytes(&self) -> Option<usize> {
        let selection = match &self.selection {
            TensorSelection::Indices { indices, .. } => {
                Layout::array::<usize>(indices.capacity()).ok()?.size()
            }
            TensorSelection::Contiguous { shape, .. } => {
                Layout::array::<usize>(shape.capacity()).ok()?.size()
            }
            _ => 0,
        };
        self.key.capacity().checked_add(selection)
    }
}
impl ResidencyManager {
    fn source_descriptors<E: From<ResidencyError>>(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        reserve: impl FnOnce(std::collections::TryReserveError) -> E,
    ) -> Result<(Vec<Descriptor>, usize), E> {
        // Only immutable declaration/source aliases are copied while locked.
        // No source routing, validation, acquisition or user callback occurs here.
        let result = {
            let state = self.inner.state.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
                std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
            })?;
            let closure = state
                .control
                .operation_closure(roots, scratch)
                .map_err(ResidencyError::OperationClosure)?;
            let selected = || {
                closure
                    .units()
                    .filter(|unit| self.inner.sources.prepared_host(unit.id()).is_none())
            };
            // Original host catalogs already retain the final immutable buffers
            // built and filled by their source constructor. No encoded source
            // acquisition is performed by the later host-to-device copy worker.
            let (rows, count) = selected()
                .try_fold((0usize, 0usize), |(rows, count), unit| {
                    let population =
                        super::operation_population::MaterializationPopulation::bindings(
                            unit.bindings(),
                        )?;
                    Some((
                        rows.checked_add(population.pending_weights)?,
                        count
                            .checked_add(population.recipe_pending_weight_bound.checked_mul(4)?)?,
                    ))
                })
                .ok_or_else(overflow)?;
            Layout::array::<Descriptor>(rows).map_err(|_| overflow())?;
            let mut descriptors = Vec::new();
            descriptors.try_reserve_exact(rows).map_err(reserve)?;
            for unit in selected() {
                let source = self
                    .inner
                    .sources
                    .retained(unit.id())
                    .ok_or(ResidencyError::OriginalOperationDomain)?;
                let mut visitor = Descriptors {
                    rows: &mut descriptors,
                    source,
                    multiplicity: 4,
                };
                for binding in unit.bindings().iter().filter(|binding| !binding.is_alias()) {
                    if let Some(recipe) = binding.recipe() {
                        recipe.visit_sources(&mut visitor)?;
                    } else {
                        visitor.source(binding.checkpoint_key(), binding.selection())?;
                    }
                }
            }
            (descriptors, count)
        };
        Ok(result)
    }

    /// Cold metadata query over the same source-occurrence traversal. All
    /// manager loans end before wrapper authorization or provider callbacks.
    pub(crate) fn source_conversion_plans(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<(Vec<SelectedSourceOccurrence>, usize), crate::backend::Error> {
        use crate::backend::Error;
        let (descriptors, _) = self
            .source_descriptors::<Error>(roots, scratch, |cause| Error::Other(Box::new(cause)))?;
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(descriptors.len())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let temporary = descriptors
            .iter()
            .try_fold(
                Layout::array::<Descriptor>(descriptors.capacity())
                    .map_err(|_| overflow())?
                    .size(),
                |sum, descriptor| sum.checked_add(descriptor.backing_bytes()?),
            )
            .ok_or_else(overflow)?;
        for descriptor in &descriptors {
            let plan = eredu_checkpoint::store::SelectedGgufConversionPlan::query_retained(
                descriptor.source.clone(),
                TensorReadRequest {
                    key: descriptor.key.clone(),
                    selection: descriptor.selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                },
            )
            .map_err(|cause| Error::Other(Box::new(cause)))?
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?;
            let acquisition = plan
                .acquisition_storage()
                .map_err(|cause| Error::Other(Box::new(cause)))?
                .ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))?;
            plans.push(SelectedSourceOccurrence {
                plan,
                multiplicity: descriptor.multiplicity,
                acquisition,
            });
        }
        Ok((plans, temporary))
    }

    /// Existing accepted reservation precedes these metadata allocations. The
    /// guard alone does not establish the still-unqualified payload peak in Q.
    #[cfg(test)]
    pub(crate) fn prepare_source_acquisitions(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        custody: OriginalTextControlGuard,
    ) -> Result<PreparedSourceAcquisitions, ResidencyError> {
        let (descriptors, count) = self.source_descriptors(roots, scratch, |cause| {
            ResidencyError::from(CheckpointMaterializationError::from(
                PreparedSourceAcquisitionFailure::reserve(cause, &custody),
            ))
        })?;
        // The complete mutex guard is gone before any actual source callback.
        let temporary_bytes = descriptors
            .iter()
            .try_fold(
                Layout::array::<Descriptor>(descriptors.capacity())
                    .map_err(|_| overflow())?
                    .size(),
                |bytes, row| bytes.checked_add(row.backing_bytes()?),
            )
            .ok_or_else(overflow)?;
        let mut bank = PreparedSourceAcquisitions::new(count, custody)
            .map_err(CheckpointMaterializationError::from)?;
        for descriptor in &descriptors {
            for _ in 0..descriptor.multiplicity {
                bank.prepare_next(
                    descriptor.source.clone(),
                    TensorReadRequest {
                        key: descriptor.key.clone(),
                        selection: descriptor.selection.clone(),
                        policy: ReadPolicy::RequireBounded,
                    },
                )
                .map_err(CheckpointMaterializationError::from)?;
            }
        }
        bank.seal().map_err(CheckpointMaterializationError::from)?;
        bank.record_preparation_temporary_bytes(temporary_bytes);
        // Descriptor Vec/key/selection storage overlaps ticket construction,
        // then retires; it is neither retained cache storage nor a payload bound.
        drop(descriptors);
        Ok(bank)
    }
}
struct Descriptors<'a> {
    rows: &'a mut Vec<Descriptor>,
    source: &'a RetainedCheckpointSource,
    multiplicity: usize,
}
impl RecipeSourceVisitor for Descriptors<'_> {
    type Error = ResidencyError;
    fn source(&mut self, key: &str, selection: &TensorSelection) -> Result<(), Self::Error> {
        self.rows.push(Descriptor {
            source: self.source.clone(),
            key: key.to_owned(),
            selection: selection.clone(),
            multiplicity: self.multiplicity,
        });
        Ok(())
    }
    fn enter_join(&mut self) -> Result<(), Self::Error> {
        self.multiplicity = self.multiplicity.checked_mul(2).ok_or_else(overflow)?;
        Ok(())
    }
    fn leave_join(&mut self) {
        self.multiplicity /= 2;
    }
}
