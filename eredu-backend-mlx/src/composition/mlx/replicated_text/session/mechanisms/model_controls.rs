//! Model control callbacks from the actual recorded equation and session strategy.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::control::speculative::{
    PreparedSpeculativeControl, SpeculativeModelControlQuote, SpeculativeModelControlVisitor,
};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataAllocation};
impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    fn prepare_session_model_control_plan<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        plan: &eredu_runtime::replicated_session::SessionModelControlPlan,
        source: Option<&PreparedSpeculativeControl>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<SpeculativeModelControlQuote>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        funding.reserve_metadata(std::mem::size_of::<(
            &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
            &eredu_runtime::replicated_session::SessionModelControlPlan,
            Option<&PreparedSpeculativeControl>,
            &HostMetadataFunding,
            Result<Option<SpeculativeModelControlQuote>, Error>,
        )>())?;
        let source = match source {
            Some(source) => source,
            None if plan.occurrences().is_empty() => return Ok(None),
            None => return Err(Error::PrefillScopeUnavailable),
        };
        let mut visitor = SpeculativeModelControlVisitor::new(source, plan.clone(), funding)?;
        if !session.visit_model_control_callbacks(plan, &mut visitor)? {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(Some(visitor.finish(funding)?))
    }
    pub(crate) fn prepare_session_embedded_model_control<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        recipe: &crate::backend::nn::workspace::ResidentNativeRecipe,
        source: Option<&PreparedSpeculativeControl>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<SpeculativeModelControlQuote>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        funding.reserve_metadata(std::mem::size_of::<(
            &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
            &crate::backend::nn::workspace::ResidentNativeRecipe,
            Option<&PreparedSpeculativeControl>,
            &HostMetadataFunding,
            Result<Option<SpeculativeModelControlQuote>, Error>,
        )>())?;
        let [row] = recipe.records() else {
            return Err(Error::PrefillScopeUnavailable);
        };
        Self::prepare_session_model_control_plan(
            session,
            row.model_controls().ok_or(Error::PrefillScopeUnavailable)?,
            source,
            funding,
        )
    }
    pub(crate) fn prepare_session_model_controls<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        source: Option<&PreparedSpeculativeControl>,
        funding: &HostMetadataFunding,
    ) -> Result<Vec<Option<SpeculativeModelControlQuote>>, Error>
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
                &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
                Option<&PreparedSpeculativeControl>,
                &HostMetadataFunding,
                Vec<Option<SpeculativeModelControlQuote>>,
                Result<Vec<Option<SpeculativeModelControlQuote>>, Error>,
                eredu_runtime::replicated_session::SessionModelControlPlan,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        let mut quotes = funding
            .metadata_vec(recipe.records().len())
            .map_err(Error::Neural)?;
        for row in recipe.records() {
            quotes.push(Self::prepare_session_model_control_plan(
                session,
                row.model_controls().ok_or(Error::PrefillScopeUnavailable)?,
                source,
                funding,
            )?);
        }
        Ok(quotes)
    }
}
