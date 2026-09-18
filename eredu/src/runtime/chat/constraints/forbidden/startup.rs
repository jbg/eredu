//! Selected trigger and retained tokenizer into a funded forbidden source.
use super::super::{selection::Selection, GenerationRuntimePlan};
use super::*;
use eredu_runtime::working_memory::{OriginalChatBackend, OriginalForbiddenSource};

impl ConstraintController {
    /// Borrow the actual original source after the same controller constructor.
    pub(crate) fn original_forbidden_inputs(&self) -> Option<&OriginalForbiddenSource> {
        match &self.runtime {
            ConstraintRuntime::PreparedForbidden { original, .. } => Some(original),
            _ => None,
        }
    }

    /// Builds the ToolChoice::None branch from the retained tokenizer and the
    /// selected policy trigger. Startup never serializes or reconstructs a tokenizer.
    pub(crate) fn from_original_forbidden_generation_plan<B: OriginalChatBackend>(
        runtime: &eredu_core::ModelRuntime<B>,
        tokenizer: &eredu_runtime::working_memory::OriginalTokenizer,
        plan: &GenerationRuntimePlan,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, ForbiddenSourceError> {
        Self::from_original_forbidden_generation_plan_with(
            plan,
            validity,
            capacity,
            funding,
            |trigger| {
                Ok(B::compile_original_forbidden_tokenizer_source(
                    runtime, tokenizer, trigger,
                )?)
            },
        )
    }
    pub(in super::super) fn from_original_forbidden_generation_plan_with<F>(
        plan: &GenerationRuntimePlan,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &HostMetadataFunding,
        compile: F,
    ) -> Result<Self, ForbiddenSourceError>
    where
        F: FnOnce(&[u8]) -> Result<OriginalForbiddenSource, Cause>,
    {
        let retain = |cause| ForbiddenSourceError {
            cause,
            funding: funding.clone(),
        };
        let parts = [
            controls().ok_or_else(|| retain(Cause::Overflow))?,
            Selection::control_bytes().ok_or_else(|| retain(Cause::Overflow))?,
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<F>(),
            size_of::<Result<Self, ForbiddenSourceError>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<OriginalForbiddenSource, Cause>>(),
            size_of::<
                Result<
                    OriginalForbiddenSource,
                    eredu_runtime::working_memory::OriginalForbiddenSourceError,
                >,
            >(),
            size_of::<(
                &GenerationRuntimePlan,
                SharedTokenFilter,
                usize,
                &HostMetadataFunding,
            )>(),
            PlainControllerHistory::copy_metadata_bytes(capacity)
                .ok_or_else(|| retain(Cause::Overflow))?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .ok_or_else(|| retain(Cause::Overflow))?,
        ];
        funding
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or_else(|| retain(Cause::Overflow))?,
            )
            .map_err(|cause| retain(Cause::Funding(cause)))?;
        let result = (|| -> Result<Self, Cause> {
            let Selection::Forbidden(trigger) = Selection::from_plan(plan)? else {
                return Err(Cause::GrammarSource);
            };
            let original = compile(trigger)?;
            let authority = HostPreparationAuthority::retain(funding.clone());
            let committed_tokens =
                PlainControllerHistory::default().copy_prepared(capacity, authority.clone())?;
            Ok(Self {
                runtime: ConstraintRuntime::PreparedForbidden {
                    inputs: original.inputs().clone(),
                    pending: TriggerPrefix::default(),
                    original,
                },
                committed_tokens,
                validity,
                authority,
                preparation: None,
            })
        })();
        result.map_err(retain)
    }
}
