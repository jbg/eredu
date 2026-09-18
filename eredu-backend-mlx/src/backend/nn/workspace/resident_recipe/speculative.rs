//! Equation-only independent invocation; never a complete generation recipe.
use super::*;
use eredu_runtime::speculative::autoregressive::AutoregressiveInvocation;

/// The same exact equation reduction, with no invented sampler phase or
/// conversion into the ordinary full-generation recipe. The enclosing source
/// owner must keep the actual model/lane binding and supply copies, sampling,
/// role-bound completion and cumulative occurrence admission before execution.
pub(crate) struct AutoregressiveEquationRecipe {
    invocation: AutoregressiveInvocation,
    recipe: ResidentNativeRecipe,
    input: Option<safemlx::OriginalPromptInputFacts>,
    io_bound: bool,
}
impl AutoregressiveEquationRecipe {
    /// Borrow the retained native reduction for the selected operation bank.
    pub(crate) fn native_recipe(&self) -> &ResidentNativeRecipe { &self.recipe }
    pub(crate) fn with_native_recipe<T>(
        &mut self, run: impl FnOnce(&mut ResidentNativeRecipe) -> T,
    ) -> T { run(&mut self.recipe) }
    pub(crate) fn invocation(&self) -> AutoregressiveInvocation {
        self.invocation
    }
    pub(crate) fn plan(&self) -> &InferenceSpanWorkspacePlan {
        self.recipe.plan()
    }
    pub(crate) fn records(&self) -> &[ResidentSpanRecipe] {
        self.recipe.records()
    }
    /// Selected resident group submissions reuse the same equation reducer;
    /// neither the wrapper nor its caller substitutes a text-step request.
    pub(crate) fn bind_neural_boundaries(
        &mut self,
        geometry: InferenceGeometry,
        per_forward: usize,
        consumers: usize,
    ) -> Result<(), crate::backend::error::Error> {
        self.recipe
            .bind_neural_boundaries(geometry, per_forward, consumers)
    }
    pub(crate) fn matches_neural_boundaries(
        &self,
        geometry: InferenceGeometry,
        submissions: usize,
        consumers: usize,
    ) -> bool {
        self.recipe
            .matches_neural_boundaries(geometry, submissions, consumers)
    }
    pub(crate) fn bind_io(
        &mut self,
        readout: Option<&super::speculative_io::AutoregressiveReadoutRecipe>,
        input: Option<safemlx::OriginalPromptInputFacts>,
    ) -> Result<(), crate::backend::error::Error> {
        if self.io_bound
            || input.is_some_and(|input| input.elements() != self.invocation.positions())
        {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        super::speculative_io::bind(&mut self.recipe, readout)?;
        self.input = input;
        self.io_bound = true;
        Ok(())
    }
    pub(crate) fn bind_prefill_inputs(
        &mut self,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        mut trace: impl FnMut(&eredu_runtime::prefill::PrefillChunk) -> Result<WorkspaceTraceReport, crate::backend::error::Error>,
    ) -> Result<(), crate::backend::error::Error> {
        if self.io_bound || self.invocation.execution_pass() != eredu_runtime::ExpertPass::Prefill {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        // Failed enrichment cannot be reused. No native role is admitted until
        // every actual row and source-view trace has completed this cold worker.
        self.io_bound = true;
        let geometry = self.plan().geometry();
        for row in &mut self.recipe.records {
            let InferenceWorkspaceSpan::Prefill(chunk) = &row.span else {
                return Err(crate::backend::error::Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            };
            let report = trace(chunk)?;
            super::speculative_io::bind_prefill_input(row, &report, geometry, mechanism, context)?;
        }
        Ok(())
    }
    pub(crate) fn equation_completion(
        &self,
        ordinal: usize,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        self.recipe.equation_completion(ordinal)
    }
    pub(crate) fn equation_domains(
        &self,
        ordinal: usize,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), crate::backend::error::Error> {
        self.recipe.equation_domains(ordinal, preparation_graph_bytes)
    }
    pub(crate) fn single_equation_completion(
        &self,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        if self.recipe.records.len() != 1 {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.recipe.completion_for(|_| true)
    }
    /// One actual equation span's existing completion/arena reduction. Input,
    /// saved-copy and sampler producers are separate compiler contributions.
    /// A multi-span invocation must install the corresponding bank per span;
    /// it cannot reuse this single recorded graph for a later frontier.
    pub(crate) fn single_equation_domains(
        &self,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), crate::backend::error::Error>
    {
        let unknown = || {
            crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )
        };
        if self.recipe.records.len() != 1 || self.recipe.unqualified_host_owner().is_some() {
            return Err(unknown());
        }
        self.equation_domains(0, self.input.map_or(0, |input| input.graph_bytes()))
    }

    /// Exact verification output frontier. The shared driver requests one
    /// completion only for Verification, after its full-sequence decoder call.
    /// The entire recorded DAG is retained, including state-update side roots;
    /// this query is not a Graph/Record quota or a role-acceptance witness.
    pub(crate) fn verification_completion(
        &self,
    ) -> Result<Option<ResidentCompletionRecipe>, crate::backend::error::Error> {
        if self.invocation.pass()
            != eredu_runtime::speculative::autoregressive::AutoregressivePass::Verification
        {
            return Ok(None);
        }
        if self.recipe.records.len() != 1 {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.recipe.completion_for(|_| true).map(Some)
    }
}
impl ResidentRecipeRecorder {
    pub(crate) fn finish_autoregressive(
        self,
        plan: &InferenceSpanWorkspacePlan,
        invocation: AutoregressiveInvocation,
    ) -> Result<AutoregressiveEquationRecipe, Error> {
        let expected_output = match invocation.pass() {
            eredu_runtime::speculative::autoregressive::AutoregressivePass::TargetPrefill => {
                eredu_core::OutputDemand::LastPosition
            }
            eredu_runtime::speculative::autoregressive::AutoregressivePass::DraftPrefill => {
                eredu_core::OutputDemand::StateOnly
            }
            _ => eredu_core::OutputDemand::Sequence,
        };
        let decode = invocation.execution_pass() == eredu_runtime::ExpertPass::Decode;
        if self.geometry.max_output_tokens != 0
            || u64::try_from(invocation.positions()).ok() != Some(self.geometry.input_positions)
            || self.geometry.output != expected_output
            || (decode
                && (self.geometry.prefill_chunk_positions != self.geometry.input_positions
                    || self.records.len() != 1))
        {
            return Err(
                self.metadata_error("independent recipe differs from its actual invocation")
            );
        }
        if let Some(context) = &self.context {
            let controls = [
                std::mem::size_of::<AutoregressiveEquationRecipe>(),
                std::mem::size_of::<Result<AutoregressiveEquationRecipe, Error>>(),
                std::mem::size_of::<(&InferenceSpanWorkspacePlan, AutoregressiveInvocation)>(),
            ];
            context.charge_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?;
        }
        // Reuse the same source-plan comparison and move-out constructor. Zero
        // means this distinct producer observed no sampling, not a fake bound.
        let recipe = self.finish_with_sampling(plan, Some(0))?;
        Ok(AutoregressiveEquationRecipe {
            invocation,
            recipe,
            input: None,
            io_bound: false,
        })
    }
}

// Shared native reduction for exact AR and Embedded equation wrappers.
impl ResidentNativeRecipe {
    pub(super) fn equation_completion(
        &self,
        ordinal: usize,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        let row = self.records.get(ordinal).ok_or(
            crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ),
        )?;
        self.completion_for(|span| span == &row.span)
    }
    pub(super) fn equation_domains(
        &self,
        ordinal: usize,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), crate::backend::error::Error> {
        let unknown = || crate::backend::error::Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        );
        let row = self.records.get(ordinal).ok_or_else(unknown)?;
        if row.unqualified_kernel_owner.is_some() || !self.sampling.rows.is_empty()
            || self.resume_copy.is_some()
            || ((self.host_copies.is_some() || self.host_transfers.is_some())
                && !self.has_host_copy_recipe()) {
            return Err(unknown());
        }
        let rows = std::slice::from_ref(row);
        let completion = self.equation_completion(ordinal)?;
        let graph = self.graph_storage_requirement_with_rows(preparation_graph_bytes, rows)?;
        let record = self.record_storage_requirement_with_rows(rows)?;
        Ok((
            completion,
            usize::try_from(graph.full_capacity.ok_or_else(unknown)?).map_err(|_| unknown())?,
            usize::try_from(record.full_capacity.ok_or_else(unknown)?).map_err(|_| unknown())?,
            self.kernel_attempts_with_rows(rows).ok_or_else(unknown)?,
            graph.control_bytes().ok_or_else(unknown)?,
        ))
    }
}
