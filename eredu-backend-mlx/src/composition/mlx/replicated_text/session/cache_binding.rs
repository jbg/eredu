//! Lexical cache serialization and agreement loans around the shared driver.
use super::*;
use crate::backend::runtime::cache::residency::PromptCacheMaterialization;
use crate::backend::runtime::distributed::topology::original_source::control::cache::{
    CacheControlOwner, CacheControlProjection,
};
use eredu_runtime::cache::PromptCachePersistenceFunding;

type Session<A, S, D> =
    ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>;
type Previous = (
    Option<CacheControlProjection>,
    Option<PromptCachePersistenceFunding>,
    Option<PromptCacheMaterialization>,
);

fn restore<A, S>(mechanisms: &mut MlxReplicatedTextMechanisms<A, S>, previous: Previous)
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    mechanisms.cache_control = previous.0;
    mechanisms.prompt_cache_context = previous.1;
    mechanisms.prompt_cache_materialization = previous.2;
}

/// All loans are restored by the existing neutral binding guard on every exit,
/// including a fenced error or unwind. This creates no native authority.
pub(super) fn run<A, S, D, T, F>(
    session: &mut Session<A, S, D>,
    control: Option<&CacheControlOwner>,
    persistence: &PromptCachePersistenceFunding,
    materialization: Option<&PromptCacheMaterialization>,
    operation: F,
) -> Result<T, Error>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    F: FnOnce(&mut Session<A, S, D>) -> Result<T, Error>,
{
    let context = persistence.context();
    context
        .charge_metadata(std::mem::size_of::<(
            &mut Session<A, S, D>,
            Option<&CacheControlOwner>,
            &PromptCachePersistenceFunding,
            Option<&PromptCacheMaterialization>,
            F,
            T,
            Result<T, Error>,
            Previous,
        )>())
        .map_err(|cause| Error::Neural(cause.into()))?;
    let funding = context
        .metadata_funding()
        .ok_or(Error::PrefillScopeUnavailable)?;
    if let Some(materialization) = materialization {
        materialization.validate(session.inference_execution_identity(), persistence)?;
    }
    let installed = control.map(CacheControlOwner::install).transpose()?;
    if let Some((_, projection)) = &installed {
        projection.validate_execution(session.inference_execution_identity())?;
    }
    let (activation, projection) = match installed {
        Some((activation, projection)) => (Some(activation), Some(projection)),
        None => (None, None),
    };
    let result = session
        .with_execution_mechanism_binding(
            &funding,
            |mechanisms| {
                // Installation is checked before replacing either source.
                if mechanisms.cache_control.is_some()
                    || mechanisms.prompt_cache_context.is_some()
                    || mechanisms.prompt_cache_materialization.is_some()
                {
                    return Err(Error::PrefillScopeReentrant);
                }
                Ok((
                    std::mem::replace(&mut mechanisms.cache_control, projection),
                    mechanisms.prompt_cache_context.replace(persistence.clone()),
                    std::mem::replace(
                        &mut mechanisms.prompt_cache_materialization,
                        materialization.cloned(),
                    ),
                ))
            },
            restore::<A, S>,
            operation,
        )
        .map_err(|cause| Error::Neural(funding.metadata_source(cause)))
        .and_then(|result| result)
        .and_then(|result| result);
    let finished = control
        .map(|control| control.finish(result.is_ok()))
        .transpose();
    drop(activation);
    match (result, finished) {
        (Err(cause), _) | (Ok(_), Err(cause)) => Err(cause),
        (Ok(value), Ok(_)) => Ok(value),
    }
}

/// Wraps the unchanged cache failure under the actual construction account.
pub(super) fn failure(
    funding: &PromptCachePersistenceFunding,
    error: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    match funding.context().metadata_funding() {
        Some(account) => Error::Neural(account.metadata_source(error)),
        None => Error::PrefillScopeUnavailable,
    }
}
