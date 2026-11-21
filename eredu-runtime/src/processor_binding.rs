//! Native lowering and observation of portable processor products.

use eredu_core::communication::pipeline::NativePipelineWorld;
use eredu_core::input::{
    MediaInputLowering, OptionalProcessorMechanism, PreparedInputInspector, ProcessorMechanisms,
    ProcessorPreparationScope,
};
use eredu_core::realtime_mechanisms::RealtimeStartupMechanisms;
use eredu_core::SubmissionBackend;

use eredu_core::{
    checkpoint::TensorDtype, InputTensorIdentity, PreparedInputError, TokenizedMultimodalRequest,
};
use eredu_nn::{
    DistributedNeuralBackend, NeuralBackend, RetainedTensorRoots, Tensor,
    TensorParallelGroupedNeuralBackend,
};

use crate::{
    observe_and_intervene, processor_resources::ProcessorResourceError, ActivationObserver,
    ComponentRankMechanisms, LocalComponentMechanisms, RealtimeGraphMechanisms,
};
use eredu_core::sampling::SamplingBackend;
use eredu_core::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};

/// Signals failure across the erased tokenizer callback while its caller retains the error value.
#[derive(Debug, thiserror::Error)]
#[error("processor text encoding failed")]
pub struct ProcessorTextError;

/// A selected portable processor executable parameterized by its neutral tensor mechanisms.
pub trait PreparedProcessorProgram<T, M: std::fmt::Display>: std::fmt::Debug + Send + Sync {
    /// Executes host preparation, native lowering and post-intervention checks.
    fn execute(
        &self,
        request: &TokenizedMultimodalRequest,
        mechanisms: &mut dyn ProcessorMechanisms<Tensor = T, Error = M>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, ProcessorTextError>,
        observer: &mut dyn ActivationObserver<T, ProcessorExecutionError<ProcessorTextError, M>>,
    ) -> Result<PreparedModelInput<T>, ProcessorExecutionError<ProcessorTextError, M>>;
}

/// Executes an erased processor while retaining the caller's exact tokenizer failure.
pub fn execute_processor<T, M, E>(
    processor: &dyn PreparedProcessorProgram<T, M>,
    request: &TokenizedMultimodalRequest,
    mechanisms: &mut dyn ProcessorMechanisms<Tensor = T, Error = M>,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    observer: &mut dyn ActivationObserver<T, ProcessorExecutionError<ProcessorTextError, M>>,
) -> Result<PreparedModelInput<T>, ProcessorExecutionError<E, M>>
where
    M: std::fmt::Display,
    E: std::fmt::Display,
{
    let mut text_error = None;
    let result = processor.execute(
        request,
        mechanisms,
        &mut |text| {
            if text_error.is_some() {
                return Err(ProcessorTextError);
            }
            encode_text(text).map_err(|error| {
                text_error = Some(error);
                ProcessorTextError
            })
        },
        observer,
    );
    result.map_err(|error| match error {
        ProcessorExecutionError::Text(_) => match text_error {
            Some(error) => ProcessorExecutionError::Text(error),
            None => ProcessorExecutionError::Plan(
                "processor reported text failure without an encoder error".into(),
            ),
        },
        ProcessorExecutionError::Plan(error) => ProcessorExecutionError::Plan(error),
        ProcessorExecutionError::Admission(error) => ProcessorExecutionError::Admission(error),
        ProcessorExecutionError::Resources(error) => ProcessorExecutionError::Resources(error),
        ProcessorExecutionError::Mechanism(error) => ProcessorExecutionError::Mechanism(error),
        ProcessorExecutionError::Prepared(error) => ProcessorExecutionError::Prepared(error),
    })
}

/// Executes the selected program and identifies the exact input before settling native work.
pub fn prepare_media<T, M, E>(
    processor: &dyn PreparedProcessorProgram<T, M::Error>,
    request: &TokenizedMultimodalRequest,
    mut scope: M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<
    (PreparedModelInput<T>, crate::PreparedInputCacheIdentity),
    eredu_core::MultimodalPreparationFailure<M::Error, E>,
>
where
    M: ProcessorPreparationScope<Tensor = T>,
    M::Error: From<eredu_core::BackendFailure>,
    E: std::error::Error + 'static,
{
    use eredu_core::{BackendFailure, MultimodalPreparationFailure as Failure};
    let result = execute_processor(
        processor,
        request,
        &mut scope,
        encode_text,
        &mut crate::NoopObserver,
    )
    .map_err(|error| match error {
        ProcessorExecutionError::Text(error) => Failure::Text(error),
        ProcessorExecutionError::Admission(error) => Failure::Capability(error),
        ProcessorExecutionError::Mechanism(error) => Failure::Backend(error),
        ProcessorExecutionError::Resources(error) => {
            Failure::Backend(BackendFailure::from_error(error).into())
        }
        ProcessorExecutionError::Plan(error) => {
            Failure::Backend(BackendFailure::from_error(std::io::Error::other(error)).into())
        }
        ProcessorExecutionError::Prepared(error) => {
            Failure::Backend(BackendFailure::from_error(error).into())
        }
    })
    .and_then(|prepared| {
        let identity = crate::PreparedInputCacheIdentity::new(
            prepared.identity().clone(),
            request.semantic_content_fingerprint(),
        )
        .map_err(|error| Failure::Backend(BackendFailure::from_error(error).into()))?;
        Ok((prepared, identity))
    });
    let completed = scope.finish();
    let prepared = result?;
    completed.map_err(Failure::Backend)?;
    Ok(prepared)
}

/// Runs media processing for a layered graph and validates the converted tensor representation.
pub fn prepare_graph_input<T, M, E>(
    lowering: &M,
    processor: &impl PreparedProcessorProgram<M::ProcessorTensor, M::ProcessorError>,
    request: &TokenizedMultimodalRequest,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<(PreparedModelInput<T>, crate::PreparedInputCacheIdentity), eredu_core::BackendFailure>
where
    M: MediaInputLowering<T>,
    E: std::error::Error + Send + Sync + 'static,
{
    use eredu_core::BackendFailure;
    let scope = lowering.begin().map_err(BackendFailure::from_error)?;
    let (prepared, identity) = prepare_media(processor, request, scope, encode_text)
        .map_err(BackendFailure::from_error)?;
    for part in prepared.identity().parts() {
        let tensor = part.payload();
        part.payload_kind()
            .validate_decoder_tensor(tensor.dtype(), tensor.shape())
            .map_err(BackendFailure::from_error)?;
    }
    let values = prepared
        .wire_values()
        .into_iter()
        .map(|value| lowering.convert_tensor(value))
        .collect();
    let input = PreparedModelInput::from_identity_wire_values(
        prepared.identity().clone(),
        values,
        |value| lowering.inspector().identity(value),
    )
    .map_err(BackendFailure::from_error)?;
    Ok((input, identity))
}

/// Portable composition for finite media-conditioned execution over a layered graph.
///
/// Implemented on the context opened by the neutral realtime factory. Portable
/// execution selects components and transport routes before using these mechanisms.
pub trait MediaGraphMechanisms {
    /// Neural, distributed, and submission operators used by the graph.
    type Neural: TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + SubmissionBackend<Executor = <<Self::Neural as NeuralBackend>::Tensor as Tensor>::Context>
        + 'static;
    /// Native transport for a selected distributed route.
    type Transport: 'static;
    /// State allocation, parameter binding, and sampling on the selected device.
    type Graph: RealtimeGraphMechanisms<
            Self::Neural,
            State: RetainedTensorRoots<<Self::Neural as NeuralBackend>::Tensor>,
            Components: LocalComponentMechanisms<
                Self::Neural,
                <Self::Graph as RealtimeGraphMechanisms<Self::Neural>>::State,
            > + ComponentRankMechanisms<
                Self::Neural,
                <Self::Graph as RealtimeGraphMechanisms<Self::Neural>>::State,
                Self::Transport,
            >,
            Sampling: SamplingBackend<
                RandomState = <Self::Graph as RealtimeStartupMechanisms>::Random,
            >,
        > + RealtimeStartupMechanisms<Random: Clone>
        + Clone
        + 'static;

    /// Opens the graph's component and state mechanisms on the selected device.
    fn graph(&self) -> Self::Graph;

    /// Supplies native media lowering for an already selected processor.
    fn media_lowering(&self) -> impl MediaInputLowering<<Self::Neural as NeuralBackend>::Tensor>;

    /// Opens the native world and constructs portable rank placement from observed identity.
    fn open_pipeline(
        &self,
    ) -> Result<(crate::PlacementRank, Self::Transport), eredu_core::BackendFailure>;
}

impl<M> MediaGraphMechanisms for M
where
    M: eredu_core::realtime_mechanisms::MediaGraphNative,
    M::Graph: RealtimeGraphMechanisms<
            M::Neural,
            State: RetainedTensorRoots<<M::Neural as NeuralBackend>::Tensor>,
            Components: LocalComponentMechanisms<
                M::Neural,
                <M::Graph as RealtimeGraphMechanisms<M::Neural>>::State,
            > + ComponentRankMechanisms<
                M::Neural,
                <M::Graph as RealtimeGraphMechanisms<M::Neural>>::State,
                <M::World as NativePipelineWorld>::Transport,
            >,
            Sampling: SamplingBackend<
                RandomState = <M::Graph as RealtimeStartupMechanisms>::Random,
            >,
        > + RealtimeStartupMechanisms<Random: Clone>
        + Clone
        + 'static,
    <M::World as NativePipelineWorld>::Transport: 'static,
{
    type Neural = M::Neural;
    type Transport = <M::World as NativePipelineWorld>::Transport;
    type Graph = M::Graph;

    fn graph(&self) -> Self::Graph {
        M::graph(self)
    }

    fn media_lowering(&self) -> impl MediaInputLowering<<Self::Neural as NeuralBackend>::Tensor> {
        M::media_lowering(self)
    }

    fn open_pipeline(
        &self,
    ) -> Result<(crate::PlacementRank, Self::Transport), eredu_core::BackendFailure> {
        let world = M::pipeline_world(self).open_world()?;
        let (size, rank) = M::World::identity(&world);
        let placement = crate::PlacementRank::new(size, rank)
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok((placement, world))
    }
}

/// Typed host buffers emitted by portable media processing.
#[derive(Debug, Clone, PartialEq)]
pub enum HostTensor {
    /// Token IDs.
    U32 {
        /// Contiguous row-major values.
        values: Vec<u32>,
        /// Exact tensor dimensions.
        shape: Vec<usize>,
    },
    /// Floating-point features.
    F32 {
        /// Contiguous row-major values.
        values: Vec<f32>,
        /// Exact tensor dimensions.
        shape: Vec<usize>,
    },
    /// Signed metadata.
    I32 {
        /// Contiguous row-major values.
        values: Vec<i32>,
        /// Exact tensor dimensions.
        shape: Vec<usize>,
    },
    /// Boolean metadata.
    Bool {
        /// Contiguous row-major values.
        values: Vec<bool>,
        /// Exact tensor dimensions.
        shape: Vec<usize>,
    },
}

impl HostTensor {
    /// Computes the geometry and dtype of the host buffer.
    pub fn identity(&self) -> Result<InputTensorIdentity, PreparedInputError> {
        let (dtype, shape) = match self {
            Self::U32 { shape, .. } => (TensorDtype::U32, shape),
            Self::F32 { shape, .. } => (TensorDtype::F32, shape),
            Self::I32 { shape, .. } => (TensorDtype::I32, shape),
            Self::Bool { shape, .. } => (TensorDtype::Bool, shape),
        };
        InputTensorIdentity::new(dtype, shape.clone())
    }
}

fn lower_host_tensor<M, E>(
    tensor: &HostTensor,
    mechanisms: &mut M,
) -> Result<M::Tensor, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms + ?Sized,
    E: std::fmt::Display,
{
    match tensor {
        HostTensor::U32 { values, shape } => mechanisms
            .tensor_u32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::F32 { values, shape } => mechanisms
            .tensor_f32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::I32 { values, shape } => mechanisms
            .tensor_i32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::Bool { values, shape } => mechanisms
            .tensor_bool(values, shape)
            .map_err(optional_mechanism_error),
    }
}

/// Lowers completed portable host products into owned native tensors.
pub fn lower_host_input<M, E>(
    host: PreparedModelInput<HostTensor>,
    mechanisms: &mut M,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms + ?Sized,
    E: std::fmt::Display,
{
    let mut parts = Vec::with_capacity(host.len());
    for part in host.parts() {
        let payload = match part.payload() {
            PreparedInputPayload::TokenIds(value) => {
                PreparedInputPayload::TokenIds(lower_host_tensor(value, mechanisms)?)
            }
            PreparedInputPayload::Tensor(value) => {
                PreparedInputPayload::Tensor(lower_host_tensor(value, mechanisms)?)
            }
            PreparedInputPayload::Embeddings(value) => {
                PreparedInputPayload::Embeddings(lower_host_tensor(value, mechanisms)?)
            }
        };
        let metadata = part
            .metadata()
            .iter()
            .map(|(key, tensor)| lower_host_tensor(tensor, mechanisms).map(|tensor| (*key, tensor)))
            .collect::<Result<Vec<_>, _>>()?;
        parts.push(
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                metadata,
                part.extents().iter().copied(),
            )
            .map_err(ProcessorExecutionError::Prepared)?,
        );
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.inspector().identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

/// Failure from architecture processor semantics, text encoding, or a mechanism.
#[derive(Debug, thiserror::Error)]
pub enum ProcessorExecutionError<E, M>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    /// Architecture processor policy rejected the request.
    #[error("processor plan rejected input: {0}")]
    Plan(String),
    /// Architecture rejected final host products before native allocation.
    #[error(transparent)]
    Admission(eredu_core::CapabilityError),
    /// Request buffer budget or arithmetic failed before media execution.
    #[error(transparent)]
    Resources(#[from] ProcessorResourceError),
    /// Facade tokenizer callback failed.
    #[error("processor text encoding failed: {0}")]
    Text(E),
    /// Generic host-media or tensor mechanism failed.
    #[error("processor mechanism failed: {0}")]
    Mechanism(M),
    /// Prepared-part or identity construction failed.
    #[error("prepared input is invalid: {0}")]
    Prepared(PreparedInputError),
}

/// Translates optional native primitive failure into the portable processor error.
pub fn optional_mechanism_error<E: std::fmt::Display, M: std::fmt::Display>(
    error: OptionalProcessorMechanism<M>,
) -> ProcessorExecutionError<E, M> {
    match error {
        OptionalProcessorMechanism::Unavailable(name) => {
            ProcessorExecutionError::Plan(format!("required {name} mechanism is unavailable"))
        }
        OptionalProcessorMechanism::Backend(error) => ProcessorExecutionError::Mechanism(error),
    }
}

/// Applies processor observations and interventions and recomputes exact input identity.
pub fn observe_processor_output<M, E, O>(
    prepared: &PreparedModelInput<M::Tensor>,
    mechanisms: &M,
    observer: &mut O,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms + ?Sized,
    M::Tensor: Clone,
    E: std::fmt::Display,
    O: ActivationObserver<M::Tensor, ProcessorExecutionError<E, M::Error>> + ?Sized,
{
    let mut parts = Vec::with_capacity(prepared.len());
    for (index, part) in prepared.parts().iter().enumerate() {
        let path = format!(
            "{}.{}",
            eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH,
            index
        );
        let payload_value = observe_and_intervene(observer, &path, part.payload().value())?;
        let payload = match part.payload() {
            PreparedInputPayload::TokenIds(_) => PreparedInputPayload::TokenIds(payload_value),
            PreparedInputPayload::Tensor(_) => PreparedInputPayload::Tensor(payload_value),
            PreparedInputPayload::Embeddings(_) => PreparedInputPayload::Embeddings(payload_value),
        };
        let metadata = part
            .metadata()
            .iter()
            .map(|(key, value)| {
                let path = format!(
                    "{}.{}.metadata.{key:?}",
                    eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH,
                    index
                );
                observe_and_intervene(observer, &path, value).map(|value| (*key, value))
            })
            .collect::<Result<Vec<_>, _>>()?;
        parts.push(
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                metadata,
                part.extents().iter().copied(),
            )
            .map_err(ProcessorExecutionError::Prepared)?,
        );
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.inspector().identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}
