//! Native tensor representations and conversions used by portable text execution.

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        runtime::{cache::MlxCacheBackend, media::input},
        MlxDistributedSession,
    },
    MlxTensor,
};
use eredu_core::BackendFailure;
use safemlx::{error::Exception, Array, Stream};

/// Concrete representations supplied by MLX for the portable executable.
#[doc(hidden)]
pub struct MlxTextValues;

impl eredu_core::backend::NativeTextValues for MlxTextValues {
    type InputTensor = Array;
    type Decode = Array;
    type Scores = Array;
    type Tensor = MlxTensor;
    type InvocationTensor = Array;
    type ObserverMapping = crate::composition::observer_mapping::ArrayToTensor;
    type Context = Stream;
    type Error = Error;
    type StateError = Exception;
    type Group = crate::backend::runtime::distributed::Group;
    type InputInspector = input::MlxInputInspector;

    fn snapshot_retirement() -> Option<Box<dyn std::any::Any>> {
        Some(Box::new(crate::backend::ordinary_retirement::ReclaimOnDrop))
    }

    fn input_inspector() -> Self::InputInspector {
        input::MlxInputInspector
    }

    fn read_token_ids(tokens: &Array, _: &Stream) -> Result<Vec<u32>, Error> {
        input::host_token_ids(tokens)
    }

    type Neural = MlxNeuralBackend;
    type Cache = MlxCacheBackend;
    type Communication = MlxDistributedSession;
    type TensorInputInspector = input::MlxTensorInputInspector;
    type TensorObserver = crate::composition::observer_mapping::TensorToArray;

    fn sampling_group(
        communication: &MlxDistributedSession,
        group: eredu_core::CollectiveGroupId,
    ) -> Option<&Self::Group> {
        communication.selected_group(group)
    }

    fn authority(
        communication: &MlxDistributedSession,
    ) -> &eredu_core::communication::CommunicationAuthority {
        communication.authority()
    }
    fn input_tensor(value: &Array) -> MlxTensor {
        MlxTensor::from_array(value.clone())
    }
    fn tensor_input_inspector() -> Self::TensorInputInspector {
        input::MlxTensorInputInspector
    }
    fn decode_tensor(tokens: &Array) -> MlxTensor {
        Self::input_tensor(tokens)
    }
    fn scores(value: MlxTensor) -> Array {
        value.into_array()
    }
    fn concatenate_token_parts(parts: &[&Array], stream: &Stream) -> Result<MlxTensor, Error> {
        input::concatenate_token_parts(parts, stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
    fn prompt_tokens(value: &Array) -> Array {
        value.clone()
    }
    fn decode_prompt_tokens(value: &Array) -> Array {
        value.clone()
    }
    fn state_error(error: BackendFailure) -> Exception {
        Exception::from_source(error)
    }
    fn tensor_bytes(value: &MlxTensor) -> u64 {
        value.as_array().nbytes() as u64
    }
    fn decoder_token_ids(value: &MlxTensor) -> Result<Vec<u32>, Error> {
        input::host_token_ids(value.as_array())
    }

    type Materializer = super::MlxEmbeddedPredictionMaterializer;
    type Embedded = super::prepared_speculative::MlxEmbeddedPredictionMechanisms;

    fn slice_tokens(
        tokens: &MlxTensor,
        range: std::ops::Range<usize>,
        context: &Stream,
    ) -> Result<MlxTensor, Exception> {
        use safemlx::ops::indexing::TryIndexOp;
        let start = i32::try_from(range.start).map_err(Exception::from_source)?;
        let end = i32::try_from(range.end).map_err(Exception::from_source)?;
        tokens
            .as_array()
            .try_index_device((.., start..end), context)
            .map(MlxTensor::from_array)
    }

    fn settle_input(value: &MlxTensor) -> Result<(), Exception> {
        value.as_array().evaluated().map(|_| ())
    }

    fn input_failure(error: BackendFailure) -> Exception {
        Exception::from_source(error)
    }

    fn observe_forward() {}

    fn observe_publication() {}

    fn read_fixed_values(value: &Array) -> Result<Vec<f32>, BackendFailure> {
        read_numeric_state(value).map_err(BackendFailure::from_error)
    }
}

// Numeric conformance reports include integer media positions as well as
// floating cache values. Reject integers that cannot be represented exactly.
fn read_numeric_state(array: &Array) -> Result<Vec<f32>, Exception> {
    // Host copies read physical storage; pooling tails can be strided views.
    thread_local! {
        static COPY_STREAM: Stream = Stream::new_with_device(&safemlx::Device::new(
            safemlx::DeviceType::Cpu, 0,
        ));
    }
    let contiguous = COPY_STREAM.with(|stream| {
        let array = match array.dtype() {
            safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 => {
                array.as_dtype(safemlx::Dtype::Float32, stream)?
            }
            _ => array.clone(),
        };
        array.contiguous(false, stream)
    })?;
    let evaluated = contiguous.evaluated()?;
    match contiguous.dtype() {
        safemlx::Dtype::Float32 => evaluated
            .try_to_vec::<f32>()
            .map_err(|error| Exception::custom(error.to_string())),
        safemlx::Dtype::Int32 => evaluated
            .try_to_vec::<i32>()
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_iter()
            .map(|value| {
                let converted = value as f32;
                if f64::from(converted) == f64::from(value) {
                    Ok(converted)
                } else {
                    Err(Exception::custom(
                        "integer state is not exactly representable in the numeric state report",
                    ))
                }
            })
            .collect(),
        dtype => Err(Exception::custom(format!(
            "unsupported numeric state dtype: {dtype:?}"
        ))),
    }
}
