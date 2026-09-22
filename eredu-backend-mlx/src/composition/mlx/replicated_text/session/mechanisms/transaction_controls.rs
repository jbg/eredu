//! Exact selected-session control callbacks for independent prediction rows.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::control::speculative::{
    PreparedSpeculativeControl, SpeculativeTransactionQuote, SpeculativeTransactionVisitor,
};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataAllocation};
use eredu_runtime::working_memory::InferenceWorkspaceSpan;
impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    pub(crate) fn prepare_session_transaction_controls<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        source: Option<&PreparedSpeculativeControl>,
        transactional: bool,
        requires_sequence: &[bool],
        additional_controls: usize,
        funding: &HostMetadataFunding,
    ) -> Result<(usize, Vec<Option<SpeculativeTransactionQuote>>), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        funding
            .reserve_metadata(std::mem::size_of::<(
                &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
                Option<&PreparedSpeculativeControl>,
                bool,
                &[bool],
                usize,
                Vec<Option<SpeculativeTransactionQuote>>,
                Result<(usize, Vec<Option<SpeculativeTransactionQuote>>), Error>,
                eredu_runtime::replicated_session::SessionTransactionControlPlan,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        let mut quotes = funding
            .metadata_vec(recipe.records().len())
            .map_err(Error::Neural)?;
        if requires_sequence.len() != recipe.records().len() {
            return Err(Error::PrefillScopeUnavailable);
        }
        for (ordinal, row) in recipe.records().iter().enumerate() {
            let demand = match row.span() {
                InferenceWorkspaceSpan::Prefill(chunk) => chunk.output,
                InferenceWorkspaceSpan::Decode { .. } => eredu_core::OutputDemand::Sequence,
                InferenceWorkspaceSpan::Sampling(_) => return Err(Error::PrefillScopeUnavailable),
            };
            let plan = session.transaction_control_plan(
                demand,
                transactional,
                requires_sequence[ordinal],
                true,
            );
            if plan.occurrences().next().is_none() {
                quotes.push(None);
                continue;
            }
            let source = source.ok_or(Error::PrefillScopeUnavailable)?;
            let mut visitor = SpeculativeTransactionVisitor::new(source, plan.clone(), funding)?;
            if !session.visit_transaction_control_callbacks(&plan, &mut visitor)? {
                return Err(Error::PrefillScopeUnavailable);
            }
            quotes.push(Some(visitor.finish(0, funding)?));
        }
        Ok((additional_controls, quotes))
    }
}
