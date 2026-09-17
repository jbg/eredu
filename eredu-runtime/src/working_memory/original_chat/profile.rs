//! Source-qualified host preparation for facade-owned behavioral profile probes.
use super::OriginalChatTemplate;
use crate::working_memory::OriginalTokenizer;
use crate::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
use eredu_core::{HostPreparationAuthority, TokenInputRejection};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Fixed refusal before a probe source or render is constructed.
#[derive(Debug, thiserror::Error)]
pub enum OriginalChatProfileError {
    /// The backend/source pair is unavailable or belongs to another domain.
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    /// The actual cumulative host account refused the next producer.
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
}

/// Actual original J/C sources and one cumulative preparation account. This
/// certifies neither a recognized protocol nor a render/encoding/native role.
#[derive(Debug, Clone)]
pub struct OriginalChatProfilePreparation {
    template: OriginalChatTemplate,
    tokenizer: OriginalTokenizer,
    funding: WorkspaceMetadataFunding,
}
impl OriginalChatProfilePreparation {
    /// Start the existing host account for the backend's actual selected
    /// execution. Source identity is checked before any account is created.
    pub fn new(
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        execution: &InferenceExecutionIdentity,
        capacity: u64,
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
            size_of::<WorkspaceMetadataFunding>(),
        ];
        funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataFundingError::Overflow)?,
        )?;
        Ok(Self {
            template: template.clone(),
            tokenizer: tokenizer.clone(),
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
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), TokenInputRejection> {
        self.template
            .validate_pool(pool)
            .and_then(|()| self.tokenizer.validate_pool(pool))
            .map_err(|_| TokenInputRejection::IdentityMismatch)
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
                .ok_or(WorkspaceMetadataFundingError::Overflow)?,
        ];
        self.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataFundingError::Overflow)?,
        )?;
        Ok(HostPreparationAuthority::retain(self.clone()))
    }
}
