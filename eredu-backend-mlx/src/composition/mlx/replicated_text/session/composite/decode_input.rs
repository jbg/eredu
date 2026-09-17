//! Same selected text admission for cached decode, without unused cache hashing.
use super::*;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::{
    PreparedInputInspector, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
};

type Prepared<P> = (
    PreparedModelInput<MlxTensor>,
    eredu_architectures::media_plan::AdmittedCompositeInput<P>,
    Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
);

pub(super) fn prepare<A>(
    admission: &A::AdmissionConfig,
    processor: &eredu_runtime::SelectedProcessorExecution,
    tokens: &Array,
    metadata: Option<&WorkspaceContext>,
) -> Result<Prepared<A::InputPartPlan>, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    if let Some(metadata) = metadata {
        metadata
            .charge_metadata(std::mem::size_of::<(
                Prepared<A::InputPartPlan>,
                Result<Prepared<A::InputPartPlan>, Error>,
                PreparedInputPart<MlxTensor>,
                Vec<PreparedInputPart<MlxTensor>>,
                input::MlxTensorInputInspector,
                Option<&WorkspaceContext>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
    }
    // This is the existing text-input branch, not a processor reselection.
    eredu_architectures::media_plan::validate_selected_part(
        processor,
        eredu_core::InputModality::Text,
        eredu_core::InputPayloadKind::TokenIds,
    )
    .map_err(|cause| match metadata {
        Some(metadata) => Error::Neural(metadata.metadata_source(cause)),
        None => Error::ArchitectureModel(cause.to_string()),
    })?;
    input::validate_token_ids_with_diagnostic(tokens, |message| match metadata {
        Some(metadata) => Error::Neural(metadata.metadata_error(message)),
        None => Error::Exception(Exception::custom(message.to_string())),
    })?;
    // Reserve the only row before constructing its owning tensor alias. The
    // actual clone is the input clone observed by the same Workspace equation.
    let mut parts = match metadata {
        Some(metadata) => metadata.metadata_vec(1).map_err(Error::Neural)?,
        None => Vec::with_capacity(1),
    };
    let part = PreparedInputPart::new(
        eredu_core::InputModality::Text,
        PreparedInputPayload::TokenIds(MlxTensor::from_array(tokens.try_clone_handle()?)),
        [],
    )
    .map_err(|cause| match metadata {
        Some(metadata) => Error::Neural(metadata.metadata_source(cause)),
        None => Error::ArchitectureModel(cause.to_string()),
    })?;
    parts.push(part);
    let inspector = input::MlxTensorInputInspector;
    let prepared = match metadata {
        Some(metadata) => PreparedModelInput::new_with_metadata(parts, metadata, |tensor| {
            inspector.identity_with_metadata(tensor, metadata)
        })
        .map_err(Error::Neural)?,
        None => PreparedModelInput::new(parts, |tensor| inspector.identity(tensor))
            .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?,
    };
    let admitted = match metadata {
        Some(metadata) => {
            A::admit_prepared_input_with_metadata(admission, &prepared, &inspector, metadata)
                .map_err(Error::Neural)?
        }
        None => A::admit_prepared_input(admission, &prepared, &inspector)
            .map_err(|cause| Error::Other(Box::new(cause)))?,
    };
    #[cfg(test)]
    input::record_original_semantic_preparation();
    // Decode preserves the committed prefix identity in the session; its new
    // single-token input identity was always discarded by this caller.
    Ok((prepared, admitted, None))
}

/// The saved Decode source is a single token-only row. Its committed media
/// identity already lives in the copied session; constructing this pending row
/// must neither hash a replacement prefix nor re-enter encoder preparation.
pub(super) fn prepare_continuation<A>(
    admission: &A::AdmissionConfig,
    processor: &eredu_runtime::SelectedProcessorExecution,
    input: input::ModelInput<'_>,
    metadata: &WorkspaceContext,
) -> Result<Prepared<A::InputPartPlan>, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    metadata
        .charge_metadata(std::mem::size_of::<(
            input::ModelInput<'_>,
            Result<Prepared<A::InputPartPlan>, Error>,
            Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        )>())
        .map_err(|cause| Error::Neural(cause.into()))?;
    let [part] = input.parts else {
        return Err(Error::Neural(metadata.metadata_error(format_args!(
            "saved decoder continuation requires one token-only input part",
        ))));
    };
    let (eredu_core::InputModality::Text, input::InputPayload::TokenIds(tokens)) =
        (part.modality(), part.payload())
    else {
        return Err(Error::Neural(metadata.metadata_error(format_args!(
            "saved decoder continuation requires one token-only input part",
        ))));
    };
    if !part.metadata().is_empty() || !part.extents().is_empty() {
        return Err(Error::Neural(metadata.metadata_error(format_args!(
            "saved decoder continuation cannot carry new media metadata",
        ))));
    }
    let mut prepared = prepare::<A>(admission, processor, tokens, Some(metadata))?;
    if input
        .cache_identity()
        .is_some_and(|identity| identity.prepared() != prepared.0.identity())
    {
        return Err(Error::Neural(metadata.metadata_error(format_args!(
            "prepared-input cache identity differs from the submitted tensors",
        ))));
    }
    // Managed prompts carry a closed shared owner whenever they carry identity.
    // No ordinary/raw identity is manufactured at this private continuation entry.
    if input.cache_identity().is_some() && input.shared_cache_identity().is_none() {
        return Err(Error::Neural(metadata.metadata_error(format_args!(
            "saved decoder continuation lost its shared cache identity owner",
        ))));
    }
    prepared.2 = input.shared_cache_identity().cloned();
    Ok(prepared)
}
