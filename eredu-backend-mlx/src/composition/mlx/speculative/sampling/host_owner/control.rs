//! Paid immutable prospective-choice replacement; no native work or RNG advance.
use super::*;
use eredu_core::speculative::SpeculativeOutputError;
use eredu_nn::workspace::WorkspaceMetadataFunding;
use eredu_runtime::execution_control::TokenChoiceError;
use eredu_runtime::TokenDomain;

fn host_failure(cause: WorkspaceMetadataFundingError) -> SpeculativeControlError {
    SpeculativeControlError::Output(SpeculativeOutputError::HostFunding(cause))
}
fn unsupported() -> SpeculativeControlError {
    SpeculativeControlError::Unsupported("sampler has no paid plain choice producer")
}
fn choice_failure(
    cause: PreparedControllerError,
    host: &HostPreparationAuthority,
) -> SpeculativeControlError {
    match cause {
        TokenChoiceError::InvalidToken(token) => SpeculativeControlError::InvalidToken(token),
        TokenChoiceError::Forbidden(token) => SpeculativeControlError::ForbiddenToken(token),
        TokenChoiceError::AlreadyPending => SpeculativeControlError::PendingToken,
        TokenChoiceError::UnexpectedCommit { .. } => {
            SpeculativeControlError::Invalid("inconsistent forced choice")
        }
        cause => SpeculativeControlError::Backend(controller_failure(cause, host)),
    }
}
fn prepare_host<S, E, L>(
    funding: &WorkspaceMetadataFunding,
    copy_bytes: usize,
) -> Result<HostPreparationAuthority, SpeculativeControlError> {
    let parts = [
        controls::<S, E, L>(copy_bytes)
            .ok_or_else(|| host_failure(WorkspaceMetadataFundingError::Overflow))?,
        HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
            .ok_or_else(|| host_failure(WorkspaceMetadataFundingError::Overflow))?,
        size_of::<Option<PreparedSpeculativeController<'_, S>>>(),
        size_of::<Result<S, PreparedControllerError>>(),
        size_of::<Result<(), PreparedControllerError>>(),
        size_of::<Result<(), SpeculativeControlError>>(),
        size_of::<Result<bool, SpeculativeControlError>>(),
        size_of::<Result<HostPreparationAuthority, SpeculativeControlError>>(),
        size_of::<TokenDomain>(),
        size_of::<u32>(),
        size_of::<usize>(),
        size_of::<bool>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or_else(|| host_failure(WorkspaceMetadataFundingError::Overflow))?;
    funding.reserve_metadata(bytes).map_err(host_failure)?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}
impl<S, E, L> MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    pub(in crate::composition::mlx::speculative::sampling) fn force_original(
        &mut self,
        token: u32,
        domain: TokenDomain,
        position: usize,
    ) -> Result<(), SpeculativeControlError> {
        if self.inner.source().prepared_grammar_controller().is_some() {
            return self.force_grammar_original(token, domain, position);
        }
        let funding = self.inner.metadata_funding().ok_or_else(unsupported)?;
        let plan = self
            .inner
            .source()
            .prepared_controller()
            .ok_or_else(unsupported)?;
        // The closed original source was authenticated when prepared. Copying
        // rechecks the exact retained filter/history; no foreign source is supplied.
        let host = prepare_host::<S, E, L>(funding, plan.copy_metadata_bytes())?;
        plan.validate_force(token, domain)
            .map_err(|cause| choice_failure(cause, &host))?;
        let source = plan
            .force(token, domain, position, host.clone())
            .map_err(|cause| choice_failure(cause, &host))?;
        let replacement = own_original(
            source,
            host,
            Some(funding.clone()),
            true,
            self.inner.snapshot_context().cloned(),
        );
        self.inner = replacement;
        Ok(())
    }
    pub(in crate::composition::mlx::speculative::sampling) fn clear_original_forced(
        &mut self,
    ) -> Result<bool, SpeculativeControlError> {
        if self.inner.source().prepared_grammar_controller().is_some() {
            return self.clear_grammar_original();
        }
        if self.inner.source().control_pending_forced().is_none() {
            return Ok(false);
        }
        let funding = self.inner.metadata_funding().ok_or_else(unsupported)?;
        let plan = self
            .inner
            .source()
            .prepared_controller()
            .ok_or_else(unsupported)?;
        let host = prepare_host::<S, E, L>(funding, plan.copy_metadata_bytes())?;
        let source = plan
            .clear(host.clone())
            .map_err(|cause| choice_failure(cause, &host))?;
        let replacement = own_original(
            source,
            host,
            Some(funding.clone()),
            true,
            self.inner.snapshot_context().cloned(),
        );
        self.inner = replacement;
        Ok(true)
    }
}
