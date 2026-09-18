//! An immutable decoder/sampler/input pair, separate from installable state.

use super::*;
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::{
    CopiedTextComponents, CopiedTextComponentsOwner, PreparedTextComponentsCopy,
};
use eredu_core::execution_control::NativeTextStateBackend;
use eredu_runtime::execution_control::SamplingCopyPolicy;

type Pending = Option<PendingTextInput<MlxModelInput, MlxTextToken>>;

/// Independently saved components from one quiescent continuation. There is no
/// public constructor, decoder extraction or exchange operation for this owner.
/// Funded data copying does not authorize its conversion into a runnable pair.
pub struct MlxSavedTextComponents {
    native: SavedNative,
    sampling: MlxSavedSamplingState,
}

enum SavedNative {
    Unquoted(MlxNativeTextState),
    Funded(CopiedTextComponentsOwner),
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
        match &self.native {
            SavedNative::Funded(pair) => Some(pair),
            SavedNative::Unquoted(_) => None,
        }
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
            std::mem::size_of::<Result<eredu_runtime::execution_control::ValidatedSamplingOverride, eredu_core::SamplingOverrideError<Error>>>(),
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
            self.funded_source_fixed()?, config, options).ok()?;
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
            native: SavedNative::Funded(pair.clone()),
            sampling: MlxSavedSamplingState::paired(pair),
        }
    }

    pub(in crate::composition::mlx::session) fn funded_control_bytes() -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<SavedNative>(),
            std::mem::size_of::<Result<Self, Error>>(),
            MlxSavedSamplingState::paired_control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(super) fn capture(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        sampling: &MlxTextSamplingState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        policy: SamplingCopyPolicy,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, Error> {
        match policy {
            SamplingCopyPolicy::Unquoted => {
                // Each legacy worker keeps its existing unquoted ownership and
                // recovery. Whole facade snapshots retain their separate host guard.
                let sampling = MlxSavedSamplingState::capture(runtime, sampling, pending, policy)?;
                let native = MlxBackend::capture_native_text_state(runtime)?;
                Ok(Self {
                    native: SavedNative::Unquoted(native),
                    sampling,
                })
            }
            SamplingCopyPolicy::Bounded(limits) => {
                let prepared = PreparedTextComponentsCopy::prepare_with_input(
                    runtime, sampling, pending, host,
                )?;
                Ok(Self::funded(prepared.copy(runtime, limits)?))
            }
        }
    }

    pub(super) fn capture_generation(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        state: &MlxTextGenerationState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        policy: SamplingCopyPolicy,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, Error> {
        let SamplingCopyPolicy::Bounded(limits) = policy else {
            return Self::capture(runtime, &state.sampling, pending, policy, Some(host));
        };
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
            Some(host),
            capture.as_ref(),
        )?;
        Ok(Self::funded(prepared.copy(runtime, limits)?))
    }

    pub(super) fn copy(
        &self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        policy: SamplingCopyPolicy,
    ) -> Result<Self, Error> {
        match (&self.native, policy) {
            (SavedNative::Unquoted(native), SamplingCopyPolicy::Unquoted) => {
                self.validate(runtime)?;
                let sampling = self.sampling.copy(runtime, policy)?;
                let native = MlxBackend::copy_native_text_state(runtime, native)?;
                Ok(Self {
                    native: SavedNative::Unquoted(native),
                    sampling,
                })
            }
            (SavedNative::Funded(pair), SamplingCopyPolicy::Bounded(limits)) => {
                let prepared = PreparedTextComponentsCopy::prepare_saved(runtime, pair)?;
                Ok(Self::funded(prepared.copy(runtime, limits)?))
            }
            // The requested copy policy cannot relabel a source or remove its
            // funded custody. Each accepted bounded copy creates a fresh pair.
            _ => Err(unknown()),
        }
    }

    pub(super) fn sampling(&self) -> &MlxSavedSamplingState {
        &self.sampling
    }

    pub(super) fn validate(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        match &self.native {
            SavedNative::Unquoted(native) => {
                MlxBackend::validate_native_text_state(runtime, native)
            }
            SavedNative::Funded(pair) => {
                // A saved data source need not match the live historical frontier.
                // Preparation checks source association and current destination;
                // actual copy admission revalidates origin health atomically.
                // Its temporary destination lease is released before returning.
                drop(PreparedTextComponentsCopy::prepare_saved(runtime, pair)?);
                Ok(())
            }
        }
    }

    pub(super) fn estimate(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        match &self.native {
            SavedNative::Unquoted(native) => {
                self.validate(runtime)?;
                let native = MlxBackend::estimate_native_text_state(runtime, Some(native))?;
                let sampling = self.sampling.estimate(runtime)?;
                Ok(native.zip(sampling).and_then(|(native, sampling)| {
                    Some(SnapshotEstimate {
                        retained_bytes: native
                            .retained_bytes
                            .checked_add(sampling.retained_bytes)?,
                        copy_bytes: native.copy_bytes.checked_add(sampling.copy_bytes)?,
                    })
                }))
            }
            SavedNative::Funded(pair) => {
                let prepared = PreparedTextComponentsCopy::prepare_saved(runtime, pair)?;
                // Both views already share this aggregate. Adding the sampling
                // estimate would count its decoder/host custody a second time.
                Ok(Some(SnapshotEstimate {
                    retained_bytes: pair.bytes(),
                    copy_bytes: prepared.required_bytes(),
                }))
            }
        }
    }

    pub(super) fn native_growth(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        input_tokens: u64,
    ) -> Result<Option<u64>, Error> {
        match &self.native {
            SavedNative::Unquoted(native) => {
                MlxBackend::estimate_native_text_growth(runtime, native, input_tokens)
            }
            SavedNative::Funded(_) => {
                self.validate(runtime)?;
                // A new runnable decoder and its physical growth need a fresh
                // authority path; frozen copy custody is insufficient evidence.
                Ok(None)
            }
        }
    }

    pub(super) fn prepare_resume(
        &self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(MlxNativeTextState, MlxTextSamplingState, Pending), Error> {
        let native = match &self.native {
            SavedNative::Unquoted(native) => native,
            SavedNative::Funded(pair) => {
                pair.validate_resume_origin(runtime)?;
                // Matching origin is necessary but supplies neither a new
                // finite request nor an installed fresh decoder/controller.
                return Err(unknown());
            }
        };
        self.validate(runtime)?;
        let (sampling, pending) = self.sampling.prepare_resume(runtime)?;
        let native = MlxBackend::copy_native_text_state(runtime, native)?;
        Ok((native, sampling, pending))
    }
}
