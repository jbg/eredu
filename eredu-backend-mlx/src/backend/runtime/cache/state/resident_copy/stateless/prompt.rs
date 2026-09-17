//! Fresh absent destination using the original prompt claim; no layer table.

use super::*;
use eredu_runtime::working_memory::{
    InferencePreparationStage, InferencePromptCompletion, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope,
};

/// Exact borrowed absence plus the sole original prompt completion. It owns no
/// table, host charge or numerical payload. Native source/copy custody belongs
/// to the separately returned scope and the enclosing recovery owner.
pub(in crate::backend::runtime::cache::state::resident_copy) struct InitializedStatelessPrompt<'a> {
    source: PreparedStatelessPoolingCopy<'a>,
    completion: InferencePromptCompletion,
}

/// Actual fresh None/None state with empty inference retention. No old request,
/// transaction, native completion or readiness is inherited from the source.
pub(in crate::backend::runtime::cache::state::resident_copy) struct PreparedStatelessPrompt {
    state: MlxPoolingAttentionState,
    completion: InferencePromptCompletion,
}

impl<'a> PreparedStatelessPoolingCopy<'a> {
    /// Opens only the original request's native scope. The caller supplies the
    /// complete source pins and owns all quoted key/pending work and recovery.
    pub(in crate::backend::runtime::cache::state::resident_copy) fn construct_prompt(
        &self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<StorageIdentity>,
    ) -> Result<(InitializedStatelessPrompt<'a>, WorkingMemoryFundingScope), Error> {
        // Current actual absent DeviceState is None/None. Preserve that exact
        // semantic shape rather than discarding any future optional layout.
        if self.shared_layout().is_some() {
            return Err(mismatch());
        }
        let (completion, native) = stage
            .construct_without_decoder(funding, complete_source)
            .map_err(|error| Error::Other(Box::new(error)))?;
        Ok((
            InitializedStatelessPrompt {
                source: *self,
                completion,
            },
            native,
        ))
    }

    /// Rechecks the exact borrowed source before producing an independent
    /// absent state. This executes no native work or host payload allocation.
    pub(in crate::backend::runtime::cache::state::resident_copy) fn copy_prompt(
        self,
        initialized: InitializedStatelessPrompt<'a>,
    ) -> Result<PreparedStatelessPrompt, Error> {
        if !self.same_source(&initialized.source) || self.shared_layout().is_some() {
            return Err(mismatch());
        }
        Ok(PreparedStatelessPrompt {
            state: MlxPoolingAttentionState::stateless(),
            completion: initialized.completion,
        })
    }
}

impl PreparedStatelessPrompt {
    /// Moves the actual absent state and original finish-only remainder. This
    /// confers neither origin binding nor native settlement; an eventual shared
    /// driver must still validate selected NoState control/quote semantics.
    pub(in crate::backend::runtime::cache::state::resident_copy) fn into_parts(
        self,
    ) -> (MlxPoolingAttentionState, InferencePromptCompletion) {
        (self.state, self.completion)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
