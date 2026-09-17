//! A single external assistant equation reduced by the existing native worker.
use super::*;
impl ResidentRecipeRecorder {
    pub(crate) fn finish_external_operation(
        self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<ResidentNativeRecipe, Error> {
        if self.geometry.max_output_tokens != 0
            || self.geometry.prefill_chunk_positions != self.geometry.input_positions
            || self.records.len() != 1
        {
            return Err(self.metadata_error(
                "external assistant recipe differs from its single actual invocation",
            ));
        }
        self.finish_with_sampling(plan, Some(0))
    }
}

impl ResidentNativeRecipe {
    pub(crate) fn bind_external_target_completion(
        &mut self, invocation: eredu_runtime::speculative::external_occurrence::ExternalInvocation,
        roots: Vec<usize>,
    ) -> Result<(), crate::backend::error::Error> {
        use eredu_runtime::speculative::external_occurrence::ExternalInvocationKind;
        if !matches!(invocation.kind(), ExternalInvocationKind::TargetPrefill | ExternalInvocationKind::TargetVerification)
            || self.plan().geometry() != invocation.geometry() || self.records.len() != 1
            || roots.len() != 1 || roots[0] == 0 || self.records[0].prediction_roots.is_some()
        {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
            ));
        }
        self.records[0].prediction_roots = Some(roots);
        Ok(())
    }

    pub(crate) fn external_equation_completion(&self) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        if self.records.len() != 1 {
            return Err(crate::backend::error::Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.equation_completion(0)
    }
    pub(crate) fn external_equation_domains(&self, preparation_graph_bytes: usize)
        -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), crate::backend::error::Error> {
        if self.records.len() != 1 {
            return Err(crate::backend::error::Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.equation_domains(0, preparation_graph_bytes)
    }
}
