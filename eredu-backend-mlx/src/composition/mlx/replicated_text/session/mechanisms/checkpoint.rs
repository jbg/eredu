//! One checkpoint payload, preserving the caller's proven copy semantics.
use super::*;
use eredu_runtime::working_memory::{InferenceRetention, InferenceStateRetention};
pub(crate) struct StateCheckpoint<S> {
    state: S,
    independent: bool,
}
impl<S> StateCheckpoint<S> {
    pub(super) fn ordinary(state: S) -> Self {
        Self {
            state,
            independent: false,
        }
    }
    /// The private caller supplies the settled, independently admitted copy of
    /// this exact lane. Rollback consumes it; it must never clone native work.
    pub(in crate::composition::mlx::replicated_text::session) fn independent(state: S) -> Self {
        Self {
            state,
            independent: true,
        }
    }
}
impl<S: InferenceStateRetention> InferenceStateRetention for StateCheckpoint<S> {
    fn inference_retention(&self) -> &InferenceRetention {
        self.state.inference_retention()
    }
    fn inference_retention_mut(&mut self) -> &mut InferenceRetention {
        self.state.inference_retention_mut()
    }
    fn retain_inference(&mut self, request: &eredu_runtime::working_memory::InferenceRequest) {
        self.state.retain_inference(request);
    }
}
impl<S: MlxStateMechanisms> StateCheckpoint<S> {
    pub(super) fn restore(self, state: &mut S, stream: &Stream) -> Result<(), Error> {
        if self.independent {
            if state.optional_layout() != self.state.optional_layout() {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            *state = self.state;
            Ok(())
        } else {
            state
                .restore_checkpoint(&self.state, stream)
                .map_err(Into::into)
        }
    }
}
