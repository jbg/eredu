//! One actual target or prediction equation; source binding and admission follow.
use super::*;
use eredu_runtime::speculative::embedded_occurrence::EmbeddedInvocationWorkspace;

/// Retains the exact architecture equation descriptor and the shared native
/// reduction. It supplies no parameter epoch, state identity or execution grant.
pub(crate) struct EmbeddedEquationRecipe {
    workspace: EmbeddedInvocationWorkspace,
    recipe: ResidentNativeRecipe,
}
impl EmbeddedEquationRecipe {
    pub(crate) fn take_missing_operation_detail(&mut self) -> Option<(usize, String)> {
        self.recipe.records[0].take_missing_operation_detail()
    }
    pub(crate) fn workspace(&self) -> EmbeddedInvocationWorkspace {
        self.workspace
    }
    pub(crate) fn plan(&self) -> &InferenceSpanWorkspacePlan {
        self.recipe.plan()
    }
    pub(crate) fn native_recipe(&self) -> &ResidentNativeRecipe {
        &self.recipe
    }
    pub(crate) fn with_native_recipe<T>(
        &mut self,
        run: impl FnOnce(&mut ResidentNativeRecipe) -> T,
    ) -> T {
        run(&mut self.recipe)
    }
    pub(crate) fn record(&self) -> &ResidentSpanRecipe {
        // The move-out constructor admits exactly one actual invocation.
        &self.recipe.records[0]
    }
    pub(crate) fn bind_neural_boundaries(
        &mut self,
        per_forward: usize,
        consumers: usize,
    ) -> Result<(), crate::backend::error::Error> {
        self.recipe
            .bind_neural_boundaries(self.workspace.geometry(), per_forward, consumers)
    }
    /// Exact attempted module/state completion root counts, in invocation order.
    /// The caller moves the original quote-funded vector; it is never cloned.
    pub(crate) fn bind_prediction_boundaries(
        &mut self,
        roots: Vec<usize>,
        consumers: usize,
    ) -> Result<(), crate::backend::error::Error> {
        self.recipe
            .bind_prediction_boundaries(self.workspace.geometry(), roots, consumers)
    }
    /// One explicit shared target-session completion, independent of selected
    /// module/group boundaries. Reuses their exact root-distribution reducer.
    pub(crate) fn bind_target_completion(
        &mut self,
        roots: Vec<usize>,
    ) -> Result<(), crate::backend::error::Error> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let target = EmbeddedInvocationWorkspace::target_with_readout(
            self.workspace.invocation(),
            self.workspace.geometry().output,
        )
        .map_err(|_| {
            crate::backend::error::Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
        })?;
        if target != self.workspace
            || roots.len() != 1
            || roots[0] == 0
            || self.recipe.records[0].prediction_roots.is_some()
        {
            return Err(crate::backend::error::Error::PrefillControl(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.recipe.records[0].prediction_roots = Some(roots);
        Ok(())
    }
    pub(crate) fn matches_prediction_boundaries(&self, roots: &[usize], consumers: usize) -> bool {
        self.recipe
            .matches_prediction_boundaries(self.workspace.geometry(), roots.len(), consumers)
            && self.recipe.records[0].prediction_roots.as_deref() == Some(roots)
    }
    /// Borrowed exact attempted-root distribution; no second inventory is made.
    pub(crate) fn prediction_boundary_roots(&self) -> Option<&[usize]> {
        self.recipe.records[0].prediction_roots.as_deref()
    }
    pub(crate) fn matches_neural_boundaries(&self, submissions: usize, consumers: usize) -> bool {
        self.recipe
            .matches_neural_boundaries(self.workspace.geometry(), submissions, consumers)
    }
    pub(crate) fn equation_completion(
        &self,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        self.recipe.equation_completion(0)
    }
    /// Reduces the actual trace with the same Graph, Record and completion
    /// worker as independent speculation. Input/copy/sampling preparation has
    /// its own contribution; a caller cannot infer it from this descriptor.
    pub(crate) fn equation_domains(
        &self,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), crate::backend::error::Error>
    {
        self.recipe.equation_domains(0, preparation_graph_bytes)
    }
}
impl ResidentRecipeRecorder {
    pub(crate) fn finish_embedded(
        self,
        plan: &InferenceSpanWorkspacePlan,
        workspace: EmbeddedInvocationWorkspace,
    ) -> Result<EmbeddedEquationRecipe, Error> {
        if self.geometry != workspace.geometry()
            || self.geometry.max_output_tokens != 0
            || self.geometry.prefill_chunk_positions != self.geometry.input_positions
            || self.records.len() != 1
        {
            return Err(self.metadata_error("embedded recipe differs from its actual invocation"));
        }
        if let Some(context) = &self.context {
            let controls = [
                std::mem::size_of::<EmbeddedEquationRecipe>(),
                std::mem::size_of::<Result<EmbeddedEquationRecipe, Error>>(),
                std::mem::size_of::<(&InferenceSpanWorkspacePlan, EmbeddedInvocationWorkspace)>(),
            ];
            context.charge_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?;
        }
        // Keep target and prediction demand exactly as traced. A state-only
        // target may still retain a hidden capture; a fused proposal can have a
        // prediction-local frontier and width different from its target input.
        let recipe = self.finish_with_sampling(plan, Some(0))?;
        Ok(EmbeddedEquationRecipe { workspace, recipe })
    }
}
