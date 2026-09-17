//! Import only an actual published copy, retaining its independent Q/H.
use super::*;
use crate::backend::{array_copy::RegisteredArrayCopy, OriginalCopyEnvironment};
use eredu_runtime::working_memory::WorkingMemoryError;

pub(crate) fn token_input(
    copy: RegisteredArrayCopy,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
) -> Result<OriginalNumericalValue, Error> {
    value_input(copy,Meaning::TokenIds,sources,environment)
}
/// Receives only an actual completed registered destination. The closed
/// caller selects meaning; descriptor validation remains shared here.
pub(super) fn value_input(copy:RegisteredArrayCopy,meaning:Meaning,
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,
)->Result<OriginalNumericalValue,Error>{
    let funding = sources.metadata_funding();
    let result = (|| {
        sources.validate_environment(environment)?;
        let frames = [
            size_of::<RegisteredArrayCopy>(),size_of::<Meaning>(),
            size_of::<(RegisteredArrayCopy,Meaning,&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>(),
            size_of::<OriginalSpeculativeRegisteredSource>(),
            size_of::<Result<OriginalSpeculativeRegisteredSource, WorkingMemoryError>>(),
            size_of::<Value>(),
            size_of::<Result<OriginalNumericalValue, Error>>(),
            value_control_bytes().ok_or_else(model::overflow)?,
            Array::descriptor_control_bytes().ok_or_else(model::overflow)?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(model::overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        copy.validate_completed_stream(environment.stream(), funding)?;
        {
            let descriptor = copy
                .array()
                .try_descriptor()
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let valid=match meaning {
                Meaning::TokenIds=>matches!(descriptor.shape(),[1,n] if *n>0)
                    && matches!(descriptor.facts().dtype(),safemlx::Dtype::Int32|safemlx::Dtype::Uint32),
                Meaning::RandomKey=>descriptor.shape()==[2] && descriptor.facts().dtype()==safemlx::Dtype::Uint32,
                Meaning::Logits=>(matches!(descriptor.shape(),[n] if *n>0)
                    || matches!(descriptor.shape(),[1,n] if *n>0)
                    || matches!(descriptor.shape(),[1,positions,n] if *positions>0 && *n>0))
                    && descriptor.facts().dtype()==safemlx::Dtype::Float32,
                _=>false,
            };
            if !valid || descriptor.facts().allocation().is_none()
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        let provenance = sources
            .request()
            .bind_registered_copy_source(copy.copy_retention())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        // This constructor copies only the already completed stream identity;
        // its own deferred native handle retains the exact registered account.
        let stream =
            model::prepare_registered_stream(provenance.clone(), environment.stream(), funding)?;
        // All fallible preparation precedes this move. Payload stays before
        // registered account/source and host custody on every return path.
        let (array, custody) = copy.into_parts();
        Ok(OriginalNumericalValue(
            Some(Rc::new(Value {
                array,
                original_budget: None,
                stream: ValueStream::Embedded(stream),
                meaning,
                provenance: Provenance::Registered(provenance),
                funding: funding.clone(),
                _copy: Some(custody),
                _snapshot_host: None,
                _readout_source: None,
            })),
            None,
        ))
    })();
    result.map_err(|cause| sources.retain_error(cause))
}
