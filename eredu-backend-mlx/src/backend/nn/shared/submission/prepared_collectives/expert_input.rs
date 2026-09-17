//! Lexical completed-ID loan under an explicitly retained model context.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataFunding;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("expert route source is not a selected rank-two integer tensor")]
    Geometry,
    #[error("expert route source readout funding: {0}")]
    Funding(#[source] eredu_nn::workspace::WorkspaceMetadataFundingError),
    #[error("expert route source readout allocation: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("expert route source has no active original observer")]
    Observer,
    #[error("expert route source completion: {0}")]
    Completion(#[source] Error),
    #[error("expert route source validation: {0}")]
    Native(#[source] safemlx::error::Exception),
    #[error("expert route source storage: {0}")]
    Read(#[source] safemlx::error::AsSliceError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    funding: WorkspaceMetadataFunding,
}
fn retain(cause: Cause, funding: &WorkspaceMetadataFunding) -> Error {
    Error::Other(Box::new(Failure { cause, funding: funding.clone() }))
}

pub(in crate::backend::nn::shared::submission) fn with_indices<T,E,F>(value:&MlxTensor,context:&Group,stream:&Stream,run:F)
    ->Result<Result<T,E>,Error>
where F:for<'loan> FnOnce(Option<(&'loan[i32],&'loan WorkspaceMetadataFunding)>)->Result<T,E> {
    MlxNeuralBackend::with_parallel_control_context(context, |prepared| {
        let Some((_, funding)) = prepared else { return Ok(run(None)); };
        let controls = [
            size_of::<(&MlxTensor,&Group,&Stream)>(), size_of::<F>(),
            size_of::<Result<T,E>>(), size_of::<Result<Result<T,E>,Error>>(),
            size_of::<Failure>(), size_of::<Cause>(), size_of::<Box<Failure>>(),
            size_of::<Error>(), size_of::<WorkspaceMetadataFunding>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<Result<Option<safemlx::OriginalScopeObserver>,safemlx::error::Exception>>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),
            size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
            size_of::<Option<Array>>(), size_of::<Result<Array,safemlx::error::Exception>>(),
            size_of::<Vec<i32>>(), size_of::<Result<(),std::collections::TryReserveError>>(),
            size_of::<Result<(),eredu_nn::workspace::WorkspaceMetadataFundingError>>(),
            size_of::<(usize,usize,Option<usize>)>(),
            size_of::<Option<(&[i32],&WorkspaceMetadataFunding)>>(),
        ];
        let bytes = controls.iter().copied().try_fold(size_of_val(&controls),usize::checked_add)
            .and_then(|bytes|bytes.checked_add(safemlx::OriginalScopeObserver::control_bytes()?))
            .and_then(|bytes|bytes.checked_add(safemlx::EvaluatedArray::iteration_control_bytes::<i32>()?
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<u32>()?)));
        let Some(bytes) = bytes else { return Err(Error::Neural(context.model_source_missing())); };
        // This reserves the fixed failure object before any source query or work.
        if funding.reserve_metadata(bytes).is_err() {
            return Err(Error::Neural(context.model_source_missing()));
        }
        charge(funding,1).map_err(|cause|retain(Cause::Completion(cause),funding))?;
        let input = value.as_array();
        if !matches!(input.dtype(), safemlx::Dtype::Int32 | safemlx::Dtype::Uint32)
            || input.ndim()!=2 || input.dim(1)<=0
        {
            return Err(retain(Cause::Geometry,funding));
        }
        let count = input.size();
        let bytes = count.checked_mul(size_of::<i32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(||retain(Cause::Geometry,funding))?;
        funding.reserve_metadata(bytes)
            .map_err(|cause|retain(Cause::Funding(cause),funding))?;
        let mut indices = Vec::new();
        indices.try_reserve_exact(count)
            .map_err(|cause|retain(Cause::Allocation(cause),funding))?;
        let observer = safemlx::OriginalScopeObserver::try_current()
            .map_err(|cause|retain(Cause::Native(cause),funding))?
            .ok_or_else(||retain(Cause::Observer,funding))?;
        // Consume the actual integer descriptor through the existing completed
        // signed-stride reader. Native U32 -> I32 conversion preserves the low
        // 32 bits; the same host cast avoids constructing an extra parent DAG.
        let array=input;
        // Same model-owned dependency worker used by tensor and pipeline waves.
        // A failed completion keeps its existing native custody in the source.
        crate::backend::runtime::cache::complete_values([array],stream)
            .map_err(|cause|retain(Cause::Completion(cause.into()),funding))?;
        let evaluated = array.completed_in_original_scope(&observer)
            .map_err(|cause|retain(Cause::Native(cause),funding))?;
        if input.dtype()==safemlx::Dtype::Int32 {
            let values=evaluated.try_iter::<i32>().map_err(|cause|retain(Cause::Read(cause),funding))?;
            if values.len()!=count{return Err(retain(Cause::Geometry,funding));}
            indices.extend(values);
        } else {
            let values=evaluated.try_iter::<u32>().map_err(|cause|retain(Cause::Read(cause),funding))?;
            if values.len()!=count{return Err(retain(Cause::Geometry,funding));}
            indices.extend(values.map(|value|value as i32));
        }
        Ok(run(Some((&indices,funding))))
    })?
}
