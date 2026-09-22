//! Borrowed saved values for logical destination estimation only.
use super::*;
use crate::backend::runtime::cache::state::PreparedResidentDenseCopy;
use crate::composition::mlx::session::model_session::pending_prompt::PreparedPendingPrompt;

/// Fixed source rejection shared by cold estimation and admitted resume planning.
#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session) enum ResumeOriginError {
    #[error(transparent)]
    Memory(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error(transparent)]
    Authority(#[from] eredu_core::SessionAuthorityError),
    #[error("resume source authority is borrowed")]
    Borrowed(#[from] std::cell::BorrowError),
    #[error(transparent)]
    Control(#[from] eredu_runtime::replicated_session::PreparedControlBindingError),
    #[error("executable does not expose a fixed saved-source origin check")]
    Unavailable,
}

impl ResumeOriginError {
    pub(in crate::composition::mlx::session) fn into_memory(
        self,
    ) -> eredu_runtime::working_memory::WorkingMemoryError {
        use eredu_runtime::working_memory::WorkingMemoryError;
        match self {
            Self::Memory(error) => error,
            Self::Control(
                eredu_runtime::replicated_session::PreparedControlBindingError::ForeignOrigin,
            ) => WorkingMemoryError::IdentityMismatch,
            Self::Authority(_) | Self::Borrowed(_) | Self::Control(_) | Self::Unavailable => {
                WorkingMemoryError::UnknownBound
            }
        }
    }
}

pub(in crate::composition::mlx::session) struct LogicalResumeSource<'a> {
    pub(in crate::composition::mlx::session) decoder: PreparedResidentDenseCopy<'a>,
    pub(in crate::composition::mlx::session) history_bytes: u64,
    pub(in crate::composition::mlx::session) control_input:
        Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
    pub(in crate::composition::mlx::session) key: Option<&'a Array>,
    pub(in crate::composition::mlx::session) pending: Option<PreparedPendingPrompt<'a>>,
    pub(in crate::composition::mlx::session) media:
        Option<&'a crate::composition::mlx::CompletedOriginalModelInput>,
    pub(in crate::composition::mlx::session) terminal: bool,
}
impl CopiedTextComponents {
    pub(in crate::composition::mlx::session) fn resume_origin_control_bytes() -> Option<usize> {
        std::mem::size_of::<ResumeOriginError>()
            .checked_add(std::mem::size_of::<Result<(), ResumeOriginError>>())?
            .checked_add(std::mem::size_of::<
                std::cell::Ref<'static, eredu_core::SessionAuthority>,
            >())
    }

    pub(in crate::composition::mlx::session) fn validate_resume_origin_fixed(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(), ResumeOriginError> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let session = runtime.session();
        if session.poison.get() {
            return Err(WorkingMemoryError::ExecutionFenced.into());
        }
        if !runtime
            .backend()
            .matches_prepared_target(&session.payload.target)
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        session.authority.try_borrow()?.require_idle()?;
        if !runtime
            .backend()
            .memory_ledger()
            .same_ledger(self.sampling.arrays.custody.pool())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        session
            .payload
            .model
            .erased()
            .validate_resident_control_origin_fixed(&self.decoder.origin)
            .ok_or(ResumeOriginError::Unavailable)??;
        if !session.parameter_epoch_matches(
            self.sampling
                .parameter_epoch
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        ) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(media) = self.sampling.pending_media() {
            let captured = &self.sampling.arrays.source;
            media.geometry()?;
            if self.sampling.arrays.pending.is_some()
                || !media.semantics().binding().matches_snapshot(
                    &captured.execution,
                    &captured.revision,
                    captured.frontier,
                )
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
        Ok(())
    }

    pub(in crate::composition::mlx::session) fn logical_resume_source(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        config: eredu_core::TextGenerationConfig,
    ) -> Option<LogicalResumeSource<'_>> {
        self.validate_resume_origin_fixed(runtime).ok()?;
        // These plans borrow only. Configuration validation does not spend the
        // old sampler account or include future history growth in copy work.
        let _ = self
            .sampling
            .sampler
            .borrow_funded()
            .prepare_resume(config.clone())
            .ok()?;
        let history = self.sampling.sampler.as_sampler().prepare_copy().ok()?;
        Some(LogicalResumeSource {
            decoder: self
                .decoder
                .native
                .prepare_copy_fixed()
                .ok()?
                .into_dense_fixed()
                .ok()?,
            history_bytes: history.history_bytes(),
            control_input: self.decoder.input.as_ref().map(AsRef::as_ref),
            key: self.sampling.arrays.key.as_ref(),
            pending: if config.sampling().max_new_tokens == Some(0) {
                None
            } else {
                self.sampling.prepare_pending_tokens().ok()?
            },
            media: (config.sampling().max_new_tokens != Some(0))
                .then(|| self.sampling.pending_media().map(|media| media.packet()))
                .flatten(),
            terminal: config.sampling().max_new_tokens == Some(0),
        })
    }
}
