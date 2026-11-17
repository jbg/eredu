//! Prepared input and retained-prefill policy for the selected VL session.

use std::num::NonZeroUsize;

use eredu_core::{capture::CaptureRequestShape, BackendFailure, CapabilityError};
use eredu_nn::{Error, Tensor};
use eredu_runtime::{
    execution_control::SessionState, PreparedInputCacheIdentity, PreparedInputInspector,
    PreparedModelInput,
};

use super::{VisionPromptAdmission, VisionPromptPreparation};
use crate::media_plan::PreparedInputPartPlan;

/// Owns the admitted prompt semantics and mutable decoder state of a VL session.
pub struct VlSessionDriver<S> {
    admission: VisionPromptAdmission,
    state: SessionState<S>,
}

impl<S> VlSessionDriver<S> {
    /// Pairs the selected prompt admission with its initial decoder state.
    pub fn new(admission: VisionPromptAdmission, state: S) -> Self {
        Self {
            admission,
            state: SessionState::new(state),
        }
    }

    /// Borrows the decoder state owner for observation or snapshot capture.
    pub fn state(&self) -> &SessionState<S> {
        &self.state
    }

    /// Lends the decoder state owner to execution or state exchange.
    pub fn state_mut(&mut self) -> &mut SessionState<S> {
        &mut self.state
    }

    /// Prepares expanded prompt tokens and retains the exact input identity.
    pub fn prepare<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<VisionPromptPreparation<T>, Error> {
        self.admission
            .admit_with_identity(input, inspector, identity)
    }

    /// Reports the expanded decoder tokens.
    pub fn input_tokens<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<Vec<u32>, BackendFailure> {
        self.prepare(input, inspector, identity)
            .map_err(BackendFailure::from_error)?
            .token_ids()
            .iter()
            .map(|&token| u32::try_from(token).map_err(BackendFailure::from_error))
            .collect()
    }

    /// Validates the prepared prefill request before execution or agreement.
    pub fn prepare_prefill<T: Tensor>(
        &self,
        input: Result<PreparedModelInput<T>, BackendFailure>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
        max_tokens: usize,
        capture: Option<CaptureRequestShape>,
    ) -> Result<(VisionPromptPreparation<T>, NonZeroUsize), BackendFailure> {
        let prompt = self
            .prepare(&input?, inspector, identity)
            .map_err(BackendFailure::from_error)?;
        prepare_vl_prefill_request(prompt, max_tokens, capture).map_err(BackendFailure::from_error)
    }

    /// Projects input admission onto the public per-part preparation report.
    pub fn input_plans<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError> {
        Ok(self
            .admission
            .admit_composite(input, inspector)?
            .parts()
            .iter()
            .cloned()
            .map(Into::into)
            .collect())
    }
}

/// Exact request, step, or snapshot rejected by VL session policy.
#[derive(Debug, thiserror::Error)]
pub enum VlSessionPolicyError {
    /// A prefill advance requires a positive decoder chunk size.
    #[error("VL prefill chunk size must be positive")]
    EmptyChunk,
    /// Prefill batches and extents must fit the admitted capture request.
    #[error(transparent)]
    Capture(#[from] eredu_core::capture::CaptureError),
}

/// Validates a prefill request before any rank advances its installed state.
pub fn prepare_vl_prefill_request<T: Tensor>(
    preparation: VisionPromptPreparation<T>,
    max_tokens: usize,
    geometry: Option<CaptureRequestShape>,
) -> Result<(VisionPromptPreparation<T>, NonZeroUsize), VlSessionPolicyError> {
    if let Some(geometry) = geometry {
        geometry.validate_prefill(1, preparation.token_ids().len() as u64)?;
    }
    let chunk = NonZeroUsize::new(max_tokens).ok_or(VlSessionPolicyError::EmptyChunk)?;
    Ok((preparation, chunk))
}
