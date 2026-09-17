//! Existing native vocabulary lookup with retained original error transports.
use super::*;
use std::mem::{size_of,size_of_val};

pub(crate) fn control_bytes()->Option<usize> {
    let parts=[size_of::<[Array;12]>(),size_of::<[Result<Array,safemlx::error::Exception>;12]>(),
        size_of::<Result<MlxTensor,ComputeError>>(),size_of::<Result<(),eredu_nn::EmbeddingValidationError>>(),
        size_of::<Result<eredu_nn::BalancedVocabularyWidths,eredu_nn::VocabularyRangeError>>(),
        size_of::<eredu_nn::BalancedVocabularyWidths>(),size_of::<Option<i32>>(),size_of::<[i32;2]>(),
        size_of::<(&mut MlxEmbedding,&MlxTensor,EmbeddingLookupPolicy,&Group,&Stream)>(),
        size_of::<(&VocabularyParallelRange,&Group)>(),size_of::<Option<&VocabularyParallelRange>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
pub(super) fn lookup(embedding:&mut MlxEmbedding,input:&MlxTensor,policy:EmbeddingLookupPolicy,
    parallel:&Group,context:&Stream)->Result<MlxTensor,ComputeError> {
    if let Some(funding)=parallel.model_funding() {
        funding.reserve_metadata(control_bytes().ok_or_else(||parallel.model_source_missing())?)
            .map_err(eredu_nn::workspace::WorkspaceMetadataError::Funding)?;
    }
    policy.validate_fixed().map_err(|cause|parallel.model_embedding_error(cause))?;
    let range=embedding.vocabulary_range.as_ref().ok_or_else(||parallel.model_source_missing())?;
    range.balanced_peer_widths_plan(parallel.size(),parallel.rank())
        .map_err(|cause|parallel.model_vocabulary_error(cause))?;
    let start=i32::try_from(range.local.start).map_err(|_|parallel.model_source_missing())?;
    let end=i32::try_from(range.local.end).map_err(|_|parallel.model_source_missing())?;
    let sentinel=match policy {EmbeddingLookupPolicy::Strict=>None,EmbeddingLookupPolicy::ZeroSentinel(n)=>Some(n)};
    let native=|result:Result<Array,safemlx::error::Exception>|result.map_err(|cause|parallel.model_native_error(cause));
    let input=native(validate_token_domain(input.as_array(),embedding.vocabulary,sentinel,context))?;
    let start=native(Array::try_from_int(start))?;
    let end=native(Array::try_from_int(end))?;
    let valid=native(native(input.ge(&start,context))?.logical_and(&native(input.lt(&end,context))?,context))?;
    let local=native(input.subtract(&start,context))?;
    let safe=native(safemlx::ops::r#where(&valid,&local,native(Array::try_from_int(0))?,context))?;
    let value=native(embedding.module.forward(&safe,context))?;
    let mask=native(valid.expand_dims(-1,context))?;
    let zero_value=native(safemlx::ops::zeros_like(&value,context))?;
    let value=native(safemlx::ops::r#where(&mask,&value,&zero_value,context))?;
    parallel.sum_model(&value,context).map(MlxTensor::from_array)
}
