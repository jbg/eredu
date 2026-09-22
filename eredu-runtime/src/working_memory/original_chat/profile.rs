//! Source-qualified host preparation for facade-owned behavioral profile probes.
use super::OriginalChatTemplate;
use crate::working_memory::OriginalTokenizer;
use crate::working_memory::{InferenceExecutionIdentity, MemoryLedger};
use eredu_core::{HostPreparationAuthority, TokenInputRejection};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Fixed refusal before a probe source or render is constructed.
#[derive(Debug, thiserror::Error)]
pub enum OriginalChatProfileError {
    /// Invalid physical-domain configuration before source preparation.
    #[error(transparent)]
    Limits(#[from] eredu_core::MemoryDomainError),
    /// The backend/source pair is unavailable or belongs to another domain.
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    /// The actual cumulative host account refused the next producer.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}

/// Actual original J/C sources and one cumulative preparation account. This
/// certifies neither a recognized protocol nor a render/encoding/native role.
#[derive(Debug, Clone)]
pub struct OriginalChatProfilePreparation {
    template: OriginalChatTemplate,
    tokenizer: OriginalTokenizer,
    execution: InferenceExecutionIdentity,
    funding: HostMetadataFunding,
}
impl OriginalChatProfilePreparation {
    /// Start the existing host account for the backend's actual selected
    /// execution. Source identity is checked before any account is created.
    pub fn new(
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        execution: &InferenceExecutionIdentity,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, OriginalChatProfileError> {
        template
            .validate_pool(tokenizer.pool())
            .map_err(|_| TokenInputRejection::IdentityMismatch)?;
        let funding = tokenizer
            .pool()
            .prepare_workspace_metadata(execution, capacity)?;
        let parts = [
            size_of::<Self>(),
            size_of::<OriginalChatProfileError>(),
            size_of::<Result<Self, OriginalChatProfileError>>(),
            size_of::<(
                &OriginalChatTemplate,
                &OriginalTokenizer,
                &InferenceExecutionIdentity,
                u64,
            )>(),
            size_of::<HostMetadataFunding>(),
            size_of::<InferenceExecutionIdentity>(),
        ];
        funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(HostMetadataFundingError::Overflow)?,
        )?;
        Ok(Self {
            template: template.clone(),
            tokenizer: tokenizer.clone(),
            execution: execution.clone(),
            funding,
        })
    }
    /// Authenticate actual source identity before constructing probe inputs.
    pub fn has_sources(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
    ) -> bool {
        self.template.same_source(template) && self.tokenizer.same_source(tokenizer)
    }
    /// Validate the real retained source domain without native work.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), TokenInputRejection> {
        self.template
            .validate_pool(pool)
            .and_then(|()| self.tokenizer.validate_pool(pool))
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    pub(super) fn validate_semantic_preparation(
        &self,
        preparation: &crate::working_memory::PreparedSemanticSource,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        if !self.tokenizer.same_source(preparation.tokenizer()) {
            return Err(crate::working_memory::WorkingMemoryError::IdentityMismatch);
        }
        preparation.validate(self.tokenizer.pool(), &self.execution)
    }
    /// Source-qualified account for facade declaration and grammar producers.
    /// Retained products must keep this account through their own retirement.
    /// It grants no native allocation or submission authority.
    pub fn metadata_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    /// The exact tokenizer authenticated when this preparation was created.
    pub fn tokenizer(&self) -> &OriginalTokenizer {
        &self.tokenizer
    }
    /// Reserve named source/control producers before birth. Every returned host
    /// owner retains this exact account through its actual payload retirement.
    /// The reservation supplies no tensor, submission or arbitrary-object grant.
    pub fn reserve_controls(
        &self,
        bytes: usize,
    ) -> Result<HostPreparationAuthority, OriginalChatProfileError> {
        let parts = [
            bytes,
            size_of::<(&Self, usize)>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<HostPreparationAuthority, OriginalChatProfileError>>(),
            HostPreparationAuthority::retention_bytes::<Self>()
                .ok_or(HostMetadataFundingError::Overflow)?,
        ];
        self.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(HostMetadataFundingError::Overflow)?,
        )?;
        Ok(HostPreparationAuthority::retain(self.clone()))
    }
}
