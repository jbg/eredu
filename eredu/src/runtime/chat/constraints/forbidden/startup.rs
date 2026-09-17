//! Actual prepared-plan trigger and historical tokenizer into original source.
use super::super::{GenerationRuntimePlan, selection::Selection};
use super::*;
use eredu_runtime::working_memory::{OriginalChatBackend, OriginalForbiddenSource};

impl ConstraintController {
    /// Borrow the actual original source after the same controller constructor.
    pub(crate) fn original_forbidden_inputs(&self) -> Option<&OriginalForbiddenSource> {
        match &self.runtime {
            ConstraintRuntime::PreparedForbidden { original, .. } => original.as_ref(),
            _ => None,
        }
    }

    /// Builds the same ToolChoice::None branch from its exact prepared recipe.
    /// The historical tokenizer is compiled through the existing original root
    /// worker; current loaded tokenizer changes cannot substitute another vocab.
    pub(crate) fn from_original_forbidden_generation_plan<B: OriginalChatBackend>(
        runtime: &eredu_core::ModelRuntime<B>,
        plan: &GenerationRuntimePlan,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, ForbiddenSourceError> {
        Self::from_original_generation_plan_with(
            plan,
            validity,
            capacity,
            funding,
            |json, trigger| {
                let tokenizer_plan =
                    eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(json)?;
                let tokenizer = B::compile_original_tokenizer(runtime, tokenizer_plan)?;
                Ok(B::compile_original_forbidden_tokenizer_source(
                    runtime, &tokenizer, trigger,
                )?)
            },
        )
    }
    pub(in super::super) fn from_original_generation_plan_with<F>(
        plan: &GenerationRuntimePlan,
        validity: SharedTokenFilter,
        capacity: usize,
        funding: &WorkspaceMetadataFunding,
        compile: F,
    ) -> Result<Self, ForbiddenSourceError>
    where
        F: FnOnce(&[u8], &[u8]) -> Result<OriginalForbiddenSource, Cause>,
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
            size_of::<eredu_runtime::working_memory::OriginalTokenizer>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::OriginalTokenizer,
                    eredu_core::BackendFailure,
                >,
            >(),
            size_of::<eredu_text::tokenizer_storage::TokenizerPlan<'_>>(),
            size_of::<
                Result<
                    eredu_text::tokenizer_storage::TokenizerPlan<'_>,
                    eredu_text::tokenizer_storage::TokenizerSourceError,
                >,
            >(),
            size_of::<(
                &GenerationRuntimePlan,
                SharedTokenFilter,
                usize,
                &WorkspaceMetadataFunding,
            )>(),
            PlainControllerHistory::copy_metadata_bytes(capacity)
                .ok_or_else(|| retain(Cause::Overflow))?,
            HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
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
            let json = plan
                .generation_constraint()
                .inner
                .recipe
                .tokenizer_object_json()
                .ok_or(Cause::TokenizerRecipe)?;
            let original = compile(json, trigger)?;
            let authority = HostPreparationAuthority::retain(funding.clone());
            let committed_tokens =
                PlainControllerHistory::default().copy_prepared(capacity, authority.clone())?;
            Ok(Self {
                runtime: ConstraintRuntime::PreparedForbidden {
                    inputs: original.inputs().clone(),
                    pending: TriggerPrefix::default(),
                    original: Some(original),
                },
                committed_tokens,
                validity,
                authority,
            })
        })();
        result.map_err(retain)
    }
}
