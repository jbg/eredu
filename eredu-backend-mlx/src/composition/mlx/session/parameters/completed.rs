//! Query already resident, completed parameters without constructing a graph.
use super::*;
use eredu_core::HostPreparationAuthority;
use safemlx::{error::CompletedReadbackError, EvaluatedArray};
use std::mem::size_of;

#[derive(Debug, thiserror::Error)]
enum ReadCause {
    #[error(transparent)]
    Environment(#[from] crate::backend::OriginalCopyEnvironmentError),
    #[error(transparent)]
    Source(#[from] CompletedReadbackError),
    #[error("parameter output allocation failed: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("completed parameter destination capacity differs from its preparation")]
    Capacity,
    #[error("more than one loaded parameter slot has the selected identity")]
    Identity,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ReadFailure {
    #[source]
    cause: ReadCause,
    _host: HostPreparationAuthority,
}
fn failure(cause: ReadCause, host: &HostPreparationAuthority) -> ParameterError {
    BackendFailure::new(
        BackendFailureKind::Other,
        ReadFailure {
            cause,
            _host: host.clone(),
        },
    )
    .into()
}

pub(super) fn environment_failure(
    cause: crate::backend::OriginalCopyEnvironmentError,
    host: &HostPreparationAuthority,
) -> ParameterError {
    failure(cause.into(), host)
}

pub(super) fn control_bytes() -> Option<usize> {
    let reader = [
        EvaluatedArray::completed_region_readback_control_bytes::<f32, f32>()?,
        EvaluatedArray::completed_region_readback_control_bytes::<half::f16, f32>()?,
        EvaluatedArray::completed_region_readback_control_bytes::<half::bf16, f32>()?,
    ]
    .into_iter()
    .max()?;
    Array::completed_borrow_control_bytes()?
        .checked_add(reader)?
        .checked_add(size_of::<Reader<'_>>())?
        .checked_add(size_of::<ReadFailure>())?
        .checked_add(size_of::<Result<Vec<f32>, ParameterError>>())?
        .checked_add(size_of::<Result<(), std::collections::TryReserveError>>())?
        .checked_add(BackendFailure::source_retention_peak_bytes::<ReadFailure>()?)
}

struct Reader<'a> {
    parameter: &'a str,
    region: &'a ParameterRegion,
    host: &'a HostPreparationAuthority,
    output: Option<Result<Vec<f32>, ParameterError>>,
}
impl ParameterSlotVisitor<MlxTensor> for Reader<'_> {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &MlxTensor) {
        if metadata.id().as_str() != self.parameter {
            return;
        }
        if self.output.is_some() {
            self.output = Some(Err(failure(ReadCause::Identity, self.host)));
            return;
        }
        let array = value.as_array();
        if !matches!(
            array.dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ) {
            return;
        }
        self.output = Some((|| {
            let source = array
                .try_completed()
                .map_err(|cause| failure(cause.into(), self.host))?;
            let count = usize::try_from(elements(&self.region.shape)?)
                .map_err(|_| ParameterError::Overflow)?;
            let mut output = Vec::new();
            output
                .try_reserve_exact(count)
                .map_err(|cause| failure(ReadCause::Allocation(cause), self.host))?;
            if output.capacity() != count {
                return Err(failure(ReadCause::Capacity, self.host));
            }
            output.resize(count, 0.);
            let read = match array.dtype() {
                Dtype::Float32 => source.try_map_region_into::<f32, f32>(
                    &self.region.starts,
                    &self.region.shape,
                    &mut output,
                    |v| v,
                ),
                Dtype::Float16 => source.try_map_region_into::<half::f16, f32>(
                    &self.region.starts,
                    &self.region.shape,
                    &mut output,
                    |v| v.to_f32(),
                ),
                Dtype::Bfloat16 => source.try_map_region_into::<half::bf16, f32>(
                    &self.region.starts,
                    &self.region.shape,
                    &mut output,
                    |v| v.to_f32(),
                ),
                _ => unreachable!("validated floating source"),
            };
            read.map_err(|cause| failure(cause.into(), self.host))?;
            Ok(output)
        })());
    }
}

impl MlxModelSession {
    pub(super) fn query_completed_parameter(
        &mut self,
        parameter: &str,
        region: &ParameterRegion,
        host: &HostPreparationAuthority,
    ) -> Result<Option<Vec<f32>>, ParameterError> {
        self.original_model_source()
            .map_err(|cause| super::failure(Error::text_admission(cause)))?;
        let mut reader = Reader {
            parameter,
            region,
            host,
            output: None,
        };
        // The existing resident traversal neither acquires a residency unit nor
        // rebuilds a module. The actual matching slot remains borrowed through
        // the synchronous read; no tensor handle or source ownership escapes.
        self.payload
            .get_mut()
            .expect("exclusive idle parameter source")
            .model
            .erased_mut()
            .visit_loaded_parameters(&mut reader);
        reader.output.transpose()
    }
}
