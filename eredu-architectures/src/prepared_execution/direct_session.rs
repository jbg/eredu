//! Direct session construction from opaque prepared architecture ownership.

use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{
    LayeredArchitecture, ReplicatedTextSession, ReplicatedTextSessionMechanisms, SubmissionBackend,
};

use super::{PreparedCompositeSessionFacts, PreparedTextSessionFacts};
use crate::{
    composite_execution::PreparedCompositeArchitecture,
    replicated_text::{PreparedCompositeTextArchitecture, PreparedReplicatedTextArchitecture},
};

/// Constructs an ordinary text session without exposing module/contract assembly to adapters.
#[allow(clippy::type_complexity)]
pub fn construct_selected_text_session<B, A, M>(
    prepared: PreparedReplicatedTextArchitecture<A>,
    mechanisms: M,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(ReplicatedTextSession<A, B, M>, PreparedTextSessionFacts), String>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let facts = PreparedTextSessionFacts::from_prepared(&prepared);
    let mut modules = prepared.into_modules();
    let session = eredu_runtime::construct_replicated_text_session(
        modules.take_architecture(),
        modules.take_source_architecture(),
        modules.take_contract(),
        mechanisms,
        context,
    )
    .map_err(|error| error.to_string())?;
    Ok((session, facts))
}

/// Constructs direct composite execution with its exact retained ingress admission.
#[allow(clippy::type_complexity)]
pub fn construct_selected_composite_session<B, A, Admission, M>(
    prepared: PreparedCompositeTextArchitecture<A, Admission>,
    mechanisms: M,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (
        ReplicatedTextSession<PreparedCompositeArchitecture<A>, B, M>,
        PreparedCompositeSessionFacts<Admission>,
    ),
    String,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<PreparedCompositeArchitecture<A>, B>,
    PreparedCompositeArchitecture<A>: LayeredArchitecture<B, M::State>,
    <PreparedCompositeArchitecture<A> as LayeredArchitecture<B, M::State>>::Error:
        std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let facts = PreparedTextSessionFacts::from_parts(
        prepared.prompt_cache_identity().clone(),
        prepared.capability_estimate().clone(),
        prepared.effective_model_type().to_owned(),
        prepared.selected().residency(),
    );
    let (architecture, source, contract, processor, admission) = prepared.into_parts();
    let session = eredu_runtime::construct_replicated_text_session(
        architecture,
        source,
        contract,
        mechanisms,
        context,
    )
    .map_err(|error| error.to_string())?;
    Ok((
        session,
        PreparedCompositeSessionFacts::new(facts, processor, admission),
    ))
}
