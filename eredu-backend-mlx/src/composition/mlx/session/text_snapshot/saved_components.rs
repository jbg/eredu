//! An immutable decoder/sampler/input pair, separate from installable state.

use super::*;
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::{
    CopiedTextComponents, CopiedTextComponentsOwner, PreparedTextComponentsCopy,
};
use eredu_runtime::execution_control::SamplingCopyPolicy;

/// Independently saved components from one quiescent continuation. There is no
/// public constructor, decoder extraction or exchange operation for this owner.
/// Funded data copying does not authorize its conversion into a runnable pair.
pub struct MlxSavedTextComponents {
    native: CopiedTextComponentsOwner,
    sampling: MlxSavedSamplingState,
}

fn unknown() -> Error {
    Error::Other(Box::new(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    ))
}

impl MlxSavedTextComponents {
    /// The fresh core resume driver accepts only the actual funded saved pair.
    /// This borrows immutable data; its copy account grants no new run authority.
    pub(in crate::composition::mlx::session) fn funded_source(
        &self,
    ) -> Result<&CopiedTextComponentsOwner, Error> {
        self.funded_source_fixed().ok_or_else(unknown)
    }

    /// Immutable paired source without constructing an error before host
    /// admission. This loan supplies neither a copy grant nor resume authority.
    pub(in crate::composition::mlx::session) fn funded_source_fixed(
        &self,
    ) -> Option<&CopiedTextComponentsOwner> {
        Some(&self.native)
    }

    /// Query the existing source-bound host constructor plan without cloning
    /// payloads, entering native work, or allocating a diagnostic before H0.
    pub(super) fn original_resume_preparation_bytes(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        config: eredu_core::TextGenerationConfig,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        use crate::composition::mlx::session::model_session::saved_array_copy::decoder::PreparedSavedTextResumeQuote;
        use eredu_runtime::working_memory::WorkingMemoryError;
        let source = self
            .funded_source_fixed()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        PreparedSavedTextResumeQuote::validate_resume_options(source, config, options)?;
        let bytes =
            PreparedSavedTextResumeQuote::known_source_preparation_component_bytes(runtime, source)
                .map_err(|cause| cause.into_memory())?;
        let parts = [
            bytes,
            std::mem::size_of::<eredu_core::OriginalTextResumeOptions<'_>>(),
            std::mem::size_of::<eredu_core::SamplingStateFacts>(),
            std::mem::size_of::<
                Result<
                    eredu_runtime::execution_control::ValidatedSamplingOverride,
                    eredu_core::SamplingOverrideError<Error>,
                >,
            >(),
            std::mem::size_of::<Option<&CopiedTextComponentsOwner>>(),
            std::mem::size_of::<Result<u64, WorkingMemoryError>>(),
            std::mem::size_of::<Result<Option<u64>, WorkingMemoryError>>(),
            std::mem::size_of::<Result<u64, std::num::TryFromIntError>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }

    pub(super) fn original_resume_estimate(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        config: eredu_core::TextGenerationConfig,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Option<SnapshotEstimate> {
        model_session::saved_array_copy::decoder::PreparedSavedTextResumeQuote::validate_resume_options(
            self.funded_source_fixed()?, config.clone(), options).ok()?;
        cold_estimate::resume(
            self.funded_source_fixed()?
                .logical_resume_source(runtime, config)?,
        )
    }

    /// Both stable views retain the same sealed pair and its one account. No
    /// independent component can be supplied to this private constructor.
    fn funded(pair: CopiedTextComponents) -> Self {
        let pair = CopiedTextComponentsOwner::new(pair);
        Self {
            native: pair.clone(),
            sampling: MlxSavedSamplingState::paired(pair),
        }
    }

    pub(in crate::composition::mlx::session) fn funded_control_bytes() -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<CopiedTextComponentsOwner>(),
            std::mem::size_of::<Result<Self, Error>>(),
            MlxSavedSamplingState::paired_control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(super) fn capture_generation(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        state: &MlxTextGenerationState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        policy: SamplingCopyPolicy,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, Error> {
        let SamplingCopyPolicy::Bounded(limits) = policy;
        use crate::composition::mlx::session::model_session::saved_array_copy::{
            capture::PreparedCaptureCopy, decoder::cold_source,
        };
        // Logical reservation and this exact constructor's H are already held.
        // Copy the drained host checkpoint before native work; it cannot change
        // the live ledger or carry a native execution permission.
        let capture = PreparedCaptureCopy::inspect(state)
            .and_then(|plan| plan.map(|plan| plan.construct(host)).transpose())
            .map_err(|cause| cold_source::retain_failure(cause, host))?;
        let prepared = PreparedTextComponentsCopy::prepare_with_input_and_capture(
            runtime,
            &state.sampling,
            pending,
            host,
            capture.as_ref(),
        )?;
        Ok(Self::funded(prepared.copy(runtime, limits)?))
    }

    pub(super) fn sampling(&self) -> &MlxSavedSamplingState {
        &self.sampling
    }

    pub(super) fn validate(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        self.native.validate_resume_origin(runtime)
    }

    pub(super) fn native_growth(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        input_tokens: u64,
    ) -> Result<Option<u64>, Error> {
        self.validate(runtime)?;
        Ok(self.native.logical_continuation_growth(input_tokens))
    }
}
