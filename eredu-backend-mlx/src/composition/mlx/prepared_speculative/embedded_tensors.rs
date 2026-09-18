//! Owned outer numerical tensors; execution stays in the shared numerical compiler.
use super::*;
use eredu_architectures::speculative_execution::EmbeddedPredictionTensor;
use eredu_runtime::working_memory::WorkingMemoryError;
fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
pub(super) fn tokens(
    tokens: &[u32],
    context: SpeculativeExecutionStreams<'_>,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    if context.embedded_invocation().is_some() {
        return Err(invalid());
    }
    let (sources, environment) = context.original_numerical().ok_or_else(invalid)?;
    let preparation = context.original_cache_preparation().ok_or_else(invalid)?;
    super::super::speculative::prepared_prediction_token_ids(
        tokens,
        sources,
        environment,
        preparation,
    )
}
pub(super) fn token_range(
    value: &EmbeddedPredictionTensor<MlxTensor>,
    start: usize,
    end: usize,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<EmbeddedPredictionTensor<MlxTensor>, Error> {
    if context.embedded_invocation().is_some() {
        return Err(invalid());
    }
    let (sources, environment) = context.original_numerical().ok_or_else(invalid)?;
    let preparation = context.original_cache_preparation().ok_or_else(invalid)?;
    super::super::speculative::prepared_prediction_tensor_range(
        value,
        start,
        end,
        true,
        value.evidence(),
        Some(value),
        sources,
        environment,
        preparation,
    )
}

pub(super) fn with_sources<R>(
    additional: &[&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    context: SpeculativeExecutionStreams<'_>,
    run: impl for<'scope> FnOnce(SpeculativeExecutionStreams<'scope>) -> Result<R, Error>,
) -> Result<R, Error> {
    let (sources, _) = context.original_numerical().ok_or_else(invalid)?;
    let funding = sources.metadata_funding();
    let existing = context.embedded_tensor_sources();
    let count = existing
        .len()
        .checked_add(additional.len())
        .ok_or_else(invalid)?;
    let controls = [
        std::mem::size_of::<
            Vec<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        >(),
        std::mem::size_of_val(&run),
        std::mem::size_of::<Result<R, Error>>(),
        std::mem::size_of::<
            std::slice::Iter<
                '_,
                &eredu_architectures::speculative_execution::PreparedEmbeddedEvidence,
            >,
        >(),
    ];
    funding
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or_else(invalid)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let mut joined = funding.metadata_vec(count).map_err(Error::from)?;
    joined.extend_from_slice(existing);
    joined.extend_from_slice(additional);
    run(context.with_embedded_tensor_sources(&joined)?)
}


pub(super) fn capture_range(value:&MlxTensor,start:usize,end:usize,
    evidence:Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
    parent:Option<&EmbeddedPredictionTensor<MlxTensor>>,context:SpeculativeExecutionStreams<'_>)
    ->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    if context.embedded_invocation().is_some(){return Err(invalid());}
    let (sources,environment)=context.original_numerical().ok_or_else(invalid)?;
    let preparation=context.original_cache_preparation().ok_or_else(invalid)?;
    super::super::speculative::prepared_prediction_tensor_range(value,start,end,false,evidence,parent,sources,environment,preparation)
}
pub(super) fn concatenate(left:&EmbeddedPredictionTensor<MlxTensor>,right:&EmbeddedPredictionTensor<MlxTensor>,
    context:SpeculativeExecutionStreams<'_>,ordinary_callback_bytes:usize)->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    if context.embedded_invocation().is_some(){return Err(invalid());}
    let (sources,environment)=context.original_numerical().ok_or_else(invalid)?;
    let preparation=context.original_cache_preparation().ok_or_else(invalid)?;
    sources.metadata_funding().reserve_metadata(ordinary_callback_bytes).map_err(Error::WorkspacePlanning)?;
    super::super::speculative::prepared_prediction_tensor_concatenate(left,right,sources,environment,preparation)
}
/// Sharing immutable completed data spends no new native-copy allowance. Exact
/// request/stream/backing validation precedes the cheap owner-reference clone.
pub(super) fn share(value:&EmbeddedPredictionTensor<MlxTensor>,context:SpeculativeExecutionStreams<'_>)
    ->Result<EmbeddedPredictionTensor<MlxTensor>,Error>{
    if context.embedded_invocation().is_some(){return Err(invalid());}
    let (sources,environment)=context.original_numerical().ok_or_else(invalid)?;
    let result=(||{
        sources.validate_environment(environment)?;
        let funding=sources.metadata_funding();
        let parts=[crate::backend::runtime::cache::state::CompletedResidentSource::array_source_control_bytes().ok_or_else(invalid)?,
            std::mem::size_of::<EmbeddedPredictionTensor<MlxTensor>>(),
            std::mem::size_of::<Result<EmbeddedPredictionTensor<MlxTensor>,Error>>()];
        funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
            .ok_or_else(invalid)?).map_err(Error::WorkspacePlanning)?;
        let evidence=value.evidence().ok_or_else(invalid)?;
        if let Some(completed)=super::super::speculative::completed_tensor_source(evidence){
            completed.validate_request_source(sources.request(),environment.stream(),funding)?;
            let _=completed.array_source_account(value.as_array(),funding)?;
        }else if let Some(registered)=super::super::speculative::registered_tensor_source(evidence){
            registered.validate(sources.request(),environment.stream(),funding)?;
            registered.validate_array(value.as_array(),funding)?;
        }else{return Err(invalid());}
        Ok(value.clone())
    })();
    result.map_err(|cause|sources.retain_error(cause))
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
