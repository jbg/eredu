//! Original validation storage is derived only from the retained native recipe.
use super::*;
use crate::backend::{
    error::Error,
    nn::workspace::{AutoregressiveEquationRecipe, EmbeddedEquationRecipe, ResidentNativeRecipe},
};
use eredu_runtime::working_memory::{
    InferenceTextStep, InferenceWorkspaceSpan, OriginalSpeculativeRole, OriginalTextControlGuard,
    OriginalTextMetadataCustody, WorkingMemoryError,
};
use std::{alloc::Layout, mem::size_of};

// Closed metadata ownership. The full role never enters native backing; each
// array keeps only the account alias installed by the native buffer producer.
#[derive(Clone, Debug)]
pub(super) enum TokenValidationCustody {
    Operation(eredu_runtime::working_memory::OriginalOperationMetadataCustody),
    Text(OriginalTextMetadataCustody),
    Speculative(OriginalSpeculativeRole),
    Embedded(eredu_runtime::working_memory::OriginalEmbeddedSpeculativeRole),
    External(eredu_runtime::working_memory::OriginalExternalSpeculativeRole),
}

impl From<OriginalTextMetadataCustody> for TokenValidationCustody {
    fn from(value: OriginalTextMetadataCustody) -> Self {
        Self::Text(value)
    }
}

pub(crate) struct PreparedTokenValidations(
    pub(super) TokenValidationBatch,
    pub(super) PreparedGroupedOutputs,
);

#[derive(Default)]
pub(crate) enum TokenValidationIngress {
    #[default]
    Ordinary,
    Original(Option<PreparedTokenValidations>),
}
impl TokenValidationIngress {
    pub(crate) fn realtime_control_bytes(recipe:crate::backend::nn::workspace::ResidentCompletionRecipe)
        ->Result<u64,Error> {Self::speculative_completion_control_bytes(recipe)}
    /// Finite root collector for one original frame. The accepted native scope
    /// is still required by begin; custody alone cannot enter or replace it.
    pub(crate) fn prepare_realtime(recipe:crate::backend::nn::workspace::ResidentCompletionRecipe,
        custody:eredu_runtime::working_memory::OriginalRealtimeBudgetCustody,
        funding:&eredu_nn::workspace::WorkspaceMetadataFunding)->Result<Self,Error> {
        funding.reserve_metadata(usize::try_from(Self::realtime_control_bytes(recipe)?)
            .map_err(|_|Error::PrefillControl(WorkingMemoryError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let source:eredu_runtime::working_memory::OriginalHostSourceCustody=custody.clone().into();
        source.validate_account(None).map_err(Error::PrefillControl)?;
        Self::prepare_limits(recipe.validation_roots,recipe.grouped_outputs,
            TokenValidationCustody::Operation(custody.into()))
    }

    pub(crate) fn prepare(
        recipe: &ResidentNativeRecipe,
        step: &InferenceTextStep,
        prefill: bool,
        controls: &OriginalTextControlGuard,
    ) -> Result<Self, Error> {
        if recipe.plan().geometry() != step.request().geometry() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        controls
            .validate_reservation(
                step.request()
                    .memory_reservation()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,
            )
            .map_err(Error::PrefillControl)?;
        let capacity = if prefill {
            prefill_capacity(recipe)?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?
        } else {
            recipe.completion_for_step(step)?.validation_roots
        };
        let grouped = if prefill {
            prefill_grouped_storage(recipe)?
        } else {
            recipe.completion_for_step(step)?.grouped_outputs
        };
        Self::prepare_limits(
            capacity,
            grouped,
            TokenValidationCustody::Text(controls.metadata_custody()),
        )
    }

    pub(crate) fn speculative_control_bytes(
        recipe: &AutoregressiveEquationRecipe,
    ) -> Result<u64, Error> {
        let completion = recipe.single_equation_completion()?;
        Self::speculative_completion_control_bytes(completion)
    }
    pub(crate) fn speculative_span_control_bytes(
        recipe: &AutoregressiveEquationRecipe,
        ordinal: usize,
    ) -> Result<u64, Error> {
        Self::speculative_completion_control_bytes(recipe.equation_completion(ordinal)?)
    }
    pub(super) fn speculative_completion_control_bytes(
        completion: crate::backend::nn::workspace::ResidentCompletionRecipe,
    ) -> Result<u64, Error> {
        control_bytes(completion.validation_roots)
            .and_then(|n| {
                n.checked_add(u64::try_from(completion.grouped_outputs.control_bytes()?).ok()?)
            })
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))
    }
    pub(crate) fn prepare_speculative(
        recipe: &AutoregressiveEquationRecipe,
        role: &OriginalSpeculativeRole,
    ) -> Result<Self, Error> {
        role.validate_plan(recipe.plan())
            .map_err(Error::PrefillControl)?;
        role.validate_invocation(recipe.invocation())
            .map_err(Error::PrefillControl)?;
        Self::speculative_control_bytes(recipe)?;
        let completion = recipe.single_equation_completion()?;
        Self::prepare_limits(
            completion.validation_roots,
            completion.grouped_outputs,
            TokenValidationCustody::Speculative(role.clone()),
        )
    }
    pub(crate) fn prepare_speculative_span(
        recipe: &AutoregressiveEquationRecipe,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
    ) -> Result<Self, Error> {
        span.validate_plan(recipe.plan()).map_err(Error::PrefillControl)?;
        span.role().validate_invocation(recipe.invocation()).map_err(Error::PrefillControl)?;
        Self::speculative_span_control_bytes(recipe, span.ordinal())?;
        let completion = recipe.equation_completion(span.ordinal())?;
        Self::prepare_limits(
            completion.validation_roots,
            completion.grouped_outputs,
            TokenValidationCustody::Speculative(span.role().clone()),
        )
    }
    pub(crate) fn embedded_control_bytes(
        recipe: &EmbeddedEquationRecipe,
    ) -> Result<u64, Error> {
        Self::speculative_completion_control_bytes(recipe.equation_completion()?)
    }
    pub(crate) fn prepare_embedded(
        recipe: &EmbeddedEquationRecipe,
        role: &eredu_runtime::working_memory::OriginalEmbeddedSpeculativeRole,
    ) -> Result<Self, Error> {
        role.validate_plan(recipe.plan()).map_err(Error::PrefillControl)?;
        role.validate_invocation(recipe.workspace().invocation()).map_err(Error::PrefillControl)?;
        Self::embedded_control_bytes(recipe)?;
        let completion = recipe.equation_completion()?;
        Self::prepare_limits(
            completion.validation_roots,
            completion.grouped_outputs,
            TokenValidationCustody::Embedded(role.clone()),
        )
    }
    pub(crate) fn external_control_bytes(recipe: &ResidentNativeRecipe) -> Result<u64, Error> {
        if recipe.records().len() != 1 { return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)); }
        Self::speculative_completion_control_bytes(recipe.external_equation_completion()?)
    }
    pub(crate) fn prepare_external(recipe: &ResidentNativeRecipe,
        role: &eredu_runtime::working_memory::OriginalExternalSpeculativeRole,
    ) -> Result<Self, Error> {
        role.validate_plan(recipe.plan()).map_err(Error::PrefillControl)?;
        Self::external_control_bytes(recipe)?;
        let completion = recipe.external_equation_completion()?;
        Self::prepare_limits(completion.validation_roots, completion.grouped_outputs,
            TokenValidationCustody::External(role.clone()))
    }
    fn prepare_limits(
        capacity: usize,
        grouped: GroupedOutputStorage,
        custody: TokenValidationCustody,
    ) -> Result<Self, Error> {
        if control_bytes(capacity).is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        let active = TOKEN_VALIDATION_SCOPE.with(|slot| {
            slot.try_borrow()
                .map(|slot| slot.is_some())
                .map_err(|_| Error::PrefillScopeReentrant)
        })?;
        if active {
            return Err(Error::PrefillScopeReentrant);
        }
        Self::prepare_storage(capacity, grouped, custody)
    }
    pub(super) fn prepare_storage(capacity: usize, grouped: GroupedOutputStorage,
        custody: TokenValidationCustody) -> Result<Self, Error> {
        if control_bytes(capacity).is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        let mut validations = Vec::new();
        validations.try_reserve_exact(capacity).map_err(|cause| {
            Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(cause))
        })?;
        if validations.capacity() != capacity {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch).at_speculative_stage("token validation destination capacity"));
        }
        let grouped = PreparedGroupedOutputs::prepare(grouped, custody.clone())?;
        Ok(Self::Original(Some(PreparedTokenValidations(
            TokenValidationBatch {
                validations,
                _original: Some(custody),
            },
            grouped,
        ))))
    }

    pub(crate) fn begin(&mut self) -> Result<TokenValidationScope, Error> {
        match self {
            Self::Ordinary => TokenValidationScope::begin().map_err(Into::into),
            Self::Original(slot) => TokenValidationScope::begin_prepared(
                slot.take().ok_or(Error::PrefillScopeUnavailable)?,
            )
            .map_err(Into::into),
        }
    }
}

fn prefill_capacity(recipe: &ResidentNativeRecipe) -> Result<Option<usize>, Error> {
    let mut count = 0usize;
    let mut found = false;
    for row in recipe.records() {
        if matches!(row.span(), InferenceWorkspaceSpan::Prefill(_)) {
            found = true;
            let Some(n) = row.validation_roots() else {
                return Ok(None);
            };
            count = count
                .checked_add(n)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        }
    }
    Ok(found.then_some(count))
}

fn prefill_grouped_storage(recipe: &ResidentNativeRecipe) -> Result<GroupedOutputStorage, Error> {
    recipe
        .records()
        .iter()
        .filter(|row| matches!(row.span(), InferenceWorkspaceSpan::Prefill(_)))
        .try_fold(GroupedOutputStorage::default(), |limits, row| {
            limits
                .merge(row.grouped_outputs())
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
        })
}

/// Every selected submission may retain its own completed batch concurrently.
/// Sum one whole-prefill batch and each actual decode batch, with no arena bytes.
pub(crate) fn request_control_bytes(recipe: &ResidentNativeRecipe) -> Result<Option<u64>, Error> {
    let mut bytes = 0u64;
    if recipe
        .records()
        .iter()
        .any(|row| matches!(row.span(), InferenceWorkspaceSpan::Prefill(_)))
    {
        let Some(prefill) = prefill_capacity(recipe)? else {
            return Ok(None);
        };
        let Some(part) = control_bytes(prefill) else {
            return Ok(None);
        };
        let Some(grouped) = prefill_grouped_storage(recipe)?.control_bytes() else {
            return Ok(None);
        };
        bytes = part
            .checked_add(
                u64::try_from(grouped)
                    .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    }
    for row in recipe.records() {
        if matches!(row.span(), InferenceWorkspaceSpan::Decode { .. }) {
            let Some(count) = row.validation_roots() else {
                return Ok(None);
            };
            let Some(part) = control_bytes(count) else {
                return Ok(None);
            };
            let Some(grouped) = row.grouped_outputs().control_bytes() else {
                return Ok(None);
            };
            let part = part
                .checked_add(
                    u64::try_from(grouped)
                        .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
                )
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            bytes = bytes
                .checked_add(part)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        }
    }
    Ok(Some(bytes))
}

fn control_bytes(capacity: usize) -> Option<u64> {
    // Reuse the owning managed-storage qualification already required by the
    // original native bank. This grants no independent bytes or attempts.
    crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes(
    )?;
    let vector = Layout::array::<TokenValidation>(capacity).ok()?.size();
    let bytes = [
        size_of::<TokenValidationBatch>(),
        size_of::<PreparedTokenValidations>(),
        size_of::<TokenValidationIngress>(),
        size_of::<ActiveTokenValidations>(),
        size_of::<TokenValidationScope>(),
        size_of::<TokenValidationCustody>(),
        size_of::<Option<ActiveTokenValidations>>(),
        size_of::<Option<(ActiveTokenValidations, Option<Exception>)>>(),
        size_of::<std::cell::RefMut<'static, Option<ActiveTokenValidations>>>(),
        size_of::<Result<TokenValidationScope, Exception>>(),
        size_of::<Result<TokenValidationIngress, Error>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<Option<TokenValidation>, (TokenValidation, Exception)>>(),
        size_of::<Option<TokenValidation>>(),
        size_of::<TokenValidation>(),
        size_of::<safemlx::EvaluatedArray<'_>>(),
        size_of::<Result<safemlx::EvaluatedArray<'_>, Exception>>(),
        size_of::<Result<&[bool], safemlx::error::AsSliceError>>(),
        size_of::<std::slice::Iter<'_, TokenValidation>>(),
        size_of::<std::collections::TryReserveError>(),
        // The original grouped-ID worker normalizes, retains its predicate,
        // then creates safe indices. Its transport and TLS read loan are
        // separate from the retained TokenValidation vector above.
        size_of::<[safemlx::Array; 4]>(),
        size_of::<[Option<safemlx::Array>; 2]>(),
        size_of::<Result<Option<safemlx::Array>, Exception>>(),
        size_of::<std::cell::Ref<'static, Option<ActiveTokenValidations>>>(),
        size_of::<
            Result<std::cell::Ref<'static, Option<ActiveTokenValidations>>, std::cell::BorrowError>,
        >(),
    ]
    .into_iter()
    .try_fold(vector, usize::checked_add)?;
    let bytes = bytes
        .checked_add(safemlx::OriginalScopeObserver::control_bytes()?)?
        .checked_add(safemlx::original_scoped_evaluation_control_bytes()?)?
        .checked_add(Exception::retained_source_control_bytes::<
            OriginalValidationFailure,
        >()?)?;
    u64::try_from(bytes).ok()
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(super) fn grouped_test_control_bytes(capacity: usize) -> Option<u64> {
    control_bytes(capacity)
}
