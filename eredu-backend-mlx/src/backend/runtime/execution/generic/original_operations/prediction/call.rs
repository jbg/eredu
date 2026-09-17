//! One exact original acquire, equation and nested completion transaction.
use super::*;
use eredu_architectures::prediction_extension::PreparedPredictionInvocationRoots;
use safemlx::Array;

type RootIter<'a> = std::iter::Map<std::slice::Iter<'a, MlxTensor>, fn(&MlxTensor) -> &Array>;
fn native(value: &MlxTensor) -> &Array {
    value.as_ref()
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CallFailure {
    #[source]
    cause: Error,
    custody: OriginalOperationMetadataCustody,
}
fn failure(cause: Error, bank: &Inner) -> Error {
    retained_error(cause, bank.role.budget_custody().into())
}
pub(super) fn retained_error(cause: Error, custody: OriginalOperationMetadataCustody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(CallFailure { cause, custody }),
        false,
    )
}
pub(super) fn preparation_error_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<CallFailure>(),
        size_of::<OriginalOperationMetadataCustody>(),
        size_of::<Result<PreparedPredictionModuleBank, Error>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<CallFailure>()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
pub(super) fn call_control_bytes(rows: usize) -> Option<usize> {
    let frames = [
        size_of::<CallSlot>(),
        size_of::<Option<CallSlot>>(),
        size_of::<CallFailure>(),
        size_of::<storage::PreparedResidencyAttempt>(),
        size_of::<Result<storage::PreparedResidencyAttempt, Error>>(),
        size_of::<Result<ResidentTransfer, Error>>(),
        size_of::<ResidentTransfer>(),
        size_of::<PreparedUnit<Vec<MlxTensor>>>(),
        size_of::<MlxUnitLease<Vec<MlxTensor>>>(),
        size_of::<Vec<MlxTensor>>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<(
            &PredictionModuleProjection,
            &ResidencyManager,
            &OffloadUnitId,
            &Stream,
        )>(),
        size_of::<(&mut NestedRoots, &Stream)>(),
        size_of::<(&Vec<MlxTensor>, &OriginalScopeObserver)>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), safemlx::error::Exception>>(),
        size_of::<usize>(),
        size_of::<usize>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<CallFailure>()?,
        eredu_nn::Error::retained_source_control_bytes::<Error>()?,
        NestedRoots::submission_control_bytes::<RootIter<'_>>()?,
        crate::backend::runtime::checkpoint::binding::original_parameter_binding_control_bytes(
            rows,
        )?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

impl PredictionModuleProjection {
    /// The actual module supplies its retained manager and physical ID. Source,
    /// stream, current Scope and quoted call position are checked before checkout.
    pub(crate) fn invoke<O, F>(
        &self,
        manager: &ResidencyManager,
        id: &OffloadUnitId,
        stream: &Stream,
        should_evict: bool,
        operation: F,
    ) -> Result<O, Error>
    where
        F: FnOnce(
            &ResidentUnitLease,
            usize,
            &mut dyn PreparedPredictionInvocationRoots<MlxTensor>,
        ) -> (Result<O, Error>, Vec<MlxTensor>),
    {
        let bank = self.value.upgrade().ok_or(Error::PrefillScopeUnavailable)?;
        let frames = [
            size_of::<F>(),
            size_of::<O>(),
            size_of::<Result<O, Error>>(),
            size_of::<(Result<O, Error>, Vec<MlxTensor>)>(),
            size_of::<(
                &ResidentUnitLease,
                usize,
                &mut dyn PreparedPredictionInvocationRoots<MlxTensor>,
            )>(),
        ];
        bank.plan
            .funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let result = (|| {
            let observer = OriginalScopeObserver::require_current()?;
            if !bank.active.get()
                || !bank.selected_stream.matches_source(stream)
                || !bank
                    .observer
                    .try_borrow()
                    .map_err(|_| Error::PrefillScopeReentrant)?
                    .as_ref()
                    .is_some_and(|value| value.same_scope(&observer))
            {
                return Err(identity().at_speculative_stage("prediction module call source or scope"));
            }
            manager
                .validate_supplementary_source(&bank.plan.source)
                .map_err(|_| identity().at_speculative_stage("prediction module retained manager"))?;
            let index = bank.started.get();
            let call = *bank.plan.calls.get(index).ok_or_else(identity)?;
            let request = bank
                .requests
                .get(call.source_ordinal)
                .ok_or_else(identity)?;
            if &request.0 != id {
                return Err(identity().at_speculative_stage("prediction module call source or scope"));
            }
            // Every irreversible checkout follows source validation. RefCell
            // loans end before user equations can enter a nested shared module.
            bank.started.set(index.checked_add(1).ok_or_else(overflow)?);
            let mut slot = bank
                .slots
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(identity)?;
            let (prepared, mut attempt) = {
                let mut storage = bank
                    .prepared
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?;
                let unit = storage
                    .units
                    .checkout()
                    .map_err(|_| Error::PrefillScopeUnavailable)?;
                let attempt = storage.checkout_residency(index)?;
                (unit, attempt)
            };
            let transfer = {
                let mut storage = bank
                    .prepared
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?;
                let mut slots = storage.residency(&mut attempt, None);
                let transfer = manager.acquire_many_with_original_transfer(
                    std::slice::from_ref(request),
                    MemoryTier::Device,
                    &mut slots,
                    &observer,
                )?;
                transfer.order_after_original(stream, slots.observations, &observer)?;
                transfer
            };
            let mut lease = MlxUnitLease::from_prepared(
                prepared,
                MlxModule::new(Vec::new()),
                MlxUnitTransfer::Ordinary {
                    _transfer: transfer,
                },
                observer.clone(),
            );
            let (outcome, mut values) = operation(
                lease.population_parts().1,
                bank.plan.population.unprepared.transfer.binding_rows,
                &mut slot.roots,
            );
            // A root-copy refusal retains the already cloned prefix in its
            // source. Move that prefix into the SAME lease before any return.
            slot.roots.recover_prefix(&mut values)?;
            lease.population_parts().0.inner = values;
            let append = slot
                .roots
                .append_validations(&mut lease.population_parts().0.inner, &observer);
            if let Err(cause) = append {
                return outcome.and_then(|_| Err(cause));
            }
            if bank.completed.get() != call.completion {
                return outcome.and_then(|_| Err(identity()));
            }
            // The module's exact source recipe selects synchronous completion.
            // Keep its host bank through routed CPU callbacks as well as GPU
            // work, then use the same unit/transfer recovery and retirement.
            lease.complete_original(|values, observer| {
                slot.completion.complete(
                    values.iter().map(native as fn(&MlxTensor) -> &Array),
                    observer,
                    stream,
                ).map_err(Error::from)
            })?;
            let validation =
                crate::backend::nn::tensor::validate_active_original_token_validations(&observer)
                    .map_err(Error::from);
            bank.completed
                .set(call.completion.checked_add(1).ok_or_else(overflow)?);
            let outcome = validation.and(outcome);
            if should_evict {
                let eviction = manager
                    .evict_original_supplementary(
                        id,
                        MemoryTier::Device,
                        &bank.plan.source,
                        &bank.role.budget_custody().into(),
                        &observer,
                    )
                    .map_err(Error::from);
                if outcome.is_ok() {
                    eviction?;
                }
            }
            outcome
        })();
        if result.is_err() {
            bank.active.set(false);
        }
        result.map_err(|cause| failure(cause, &bank))
    }
}
