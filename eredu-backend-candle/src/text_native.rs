//! Native inputs and eager submission for prepared text sessions.

use std::{any::Any, convert::Infallible, rc::Rc};

use candle_core::DType;
use eredu_core::{
    backend::{
        local_text::NoTextTransport, CausalTextGeneration, NativeTextValues, SharedTextNative,
        TextSessionNative, TextTransactionMechanisms,
    },
    checkpoint::TensorDtype,
    communication::{CommunicationAuthority, CommunicationSessionIdentity},
    eager_completion::EagerCompletion,
    input::{InputTensorIdentity, PlainTextTokens, PreparedInputError, PreparedInputInspector},
    observation::{ObserverTensorMapping, TensorObservation, TensorObservationData},
    speculative::SpeculativeQueueMechanisms,
    BackendFailure, BackendFailureKind, CapabilityError, CollectiveGroupId, Submission,
    SubmissionLease,
};
use eredu_nn::Tensor;

use crate::{
    text::invalid, CandleCacheStorage, CandleContext, CandleNeuralBackend, CandleTensor,
    CandleTextScope, CandleToken,
};

/// Prepared text tensor representations.
pub struct CandleTextValues;
/// Metadata and bounded integer reads for prepared tensors.
pub struct CandleInputInspector;
/// Identity mapping between native and observer tensor handles.
pub struct CandleObserver<I>(std::marker::PhantomData<fn() -> I>);

trait ObservationError: std::error::Error + Send + Sync + 'static {
    fn from_observation(error: BackendFailure) -> Self;
}

impl ObservationError for BackendFailure {
    fn from_observation(error: BackendFailure) -> Self {
        error
    }
}

impl ObservationError for eredu_nn::Error {
    fn from_observation(error: BackendFailure) -> Self {
        Self::backend_source(error)
    }
}

impl<I> ObserverTensorMapping for CandleObserver<I>
where
    I: ObservationError,
{
    type InvocationTensor = CandleTensor;
    type ObserverTensor = CandleTensor;
    type InvocationError = I;
    type ObserverError = BackendFailure;

    fn borrow_tensor(value: &CandleTensor) -> &CandleTensor {
        value
    }
    fn borrow_invocation_tensor(value: &CandleTensor) -> &CandleTensor {
        value
    }
    fn generated_tensor(value: CandleTensor) -> CandleTensor {
        value
    }
    fn replacement_tensor(value: CandleTensor) -> CandleTensor {
        value
    }
    fn invocation_error(error: BackendFailure) -> I {
        I::from_observation(error)
    }
    fn observer_error(error: I) -> BackendFailure {
        BackendFailure::from_error(error)
    }
}

impl PreparedInputInspector<CandleTensor> for CandleInputInspector {
    fn identity(&self, tensor: &CandleTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        let shape = tensor.value.dims().to_vec();
        let dtype = match tensor.value.dtype() {
            DType::I32 => TensorDtype::I32,
            DType::U32 => TensorDtype::U32,
            DType::I64 => TensorDtype::I64,
            DType::U8 if tensor.boolean => TensorDtype::Bool,
            DType::U8 => TensorDtype::U8,
            DType::F32 => TensorDtype::F32,
            DType::F16 => TensorDtype::F16,
            DType::BF16 => TensorDtype::Bf16,
            DType::F64 => TensorDtype::F64,
            dtype => TensorDtype::Encoded(format!("{dtype:?}")),
        };
        InputTensorIdentity::new(dtype, shape)
    }

    fn i32_values(&self, tensor: &CandleTensor) -> Result<Vec<i32>, CapabilityError> {
        tensor
            .to_i32_vec(&CandleContext::cpu())
            .map_err(|error| CapabilityError::Observation(error.to_string()))
    }

    fn bool_values(&self, tensor: &CandleTensor) -> Result<Vec<bool>, CapabilityError> {
        if !tensor.boolean {
            return Err(CapabilityError::Observation(
                "expected Boolean metadata".into(),
            ));
        }
        tensor
            .value
            .flatten_all()
            .and_then(|value| value.to_vec1::<u8>())
            .map(|values| values.into_iter().map(|value| value != 0).collect())
            .map_err(|error| CapabilityError::Observation(error.to_string()))
    }
}

fn token_span(
    tokens: &CandleTensor,
    start: usize,
    end: usize,
) -> Result<CandleTensor, BackendFailure> {
    let count = end
        .checked_sub(start)
        .ok_or_else(|| invalid("invalid token span"))?;
    CandleTensor::wrap(tokens.value.narrow(1, start, count)).map_err(BackendFailure::from_error)
}

impl PlainTextTokens<CandleTensor> for CandleContext {
    fn materialize_tokens(&self, ids: &[u32]) -> Result<CandleTensor, BackendFailure> {
        self.tokens(ids)
    }
    fn token_identity(tokens: &CandleTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        CandleInputInspector.identity(tokens)
    }
    fn token_count(tokens: &CandleTensor) -> Option<usize> {
        let [1, count] = *tokens.value.dims() else {
            return None;
        };
        Some(count)
    }
    fn split_tokens(
        &self,
        tokens: &CandleTensor,
        at: std::num::NonZeroUsize,
    ) -> Result<(CandleTensor, CandleTensor), BackendFailure> {
        let count = Self::token_count(tokens).ok_or_else(|| invalid("invalid token shape"))?;
        Ok((
            token_span(tokens, 0, at.get())?,
            token_span(tokens, at.get(), count)?,
        ))
    }
}

impl SpeculativeQueueMechanisms for CandleContext {
    type Queue = Self;
    type Root = CandleTensor;
    type Handoff = ();
    type Error = BackendFailure;
    fn same_queue(left: &Self, right: &Self) -> bool {
        left.same_context(right)
    }
    fn same_device(left: &Self, right: &Self) -> Result<bool, BackendFailure> {
        Ok(left.device.location() == right.device.location())
    }
    fn order_roots<'a>(
        roots: impl IntoIterator<Item = &'a CandleTensor>,
        consumer: &Self,
    ) -> Result<(), BackendFailure> {
        for root in roots {
            consumer.validate_tensor_device(root)?;
        }
        consumer.synchronize()
    }
    fn invalid_assignment(message: String) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::InvalidInput,
            std::io::Error::other(message),
        )
    }
}

impl NativeTextValues for CandleTextValues {
    type InputTensor = CandleTensor;
    type Decode = CandleTensor;
    type Scores = CandleTensor;
    type Tensor = CandleTensor;
    type InvocationTensor = CandleTensor;
    type ObserverMapping = CandleObserver<BackendFailure>;
    type Context = CandleContext;
    type Error = BackendFailure;
    type StateError = BackendFailure;
    type Group = Infallible;
    type InputInspector = CandleInputInspector;
    type TensorInputInspector = CandleInputInspector;
    type TensorObserver = CandleObserver<eredu_nn::Error>;
    type Neural = CandleNeuralBackend;
    type Cache = CandleCacheStorage;
    type Communication = NoTextTransport;
    type Materializer = crate::prediction::CandlePredictionMaterializer;
    type Embedded = crate::prediction::CandlePredictionOperators;

    fn input_inspector() -> Self::InputInspector {
        CandleInputInspector
    }
    fn tensor_input_inspector() -> Self::TensorInputInspector {
        CandleInputInspector
    }
    fn read_token_ids(
        tokens: &CandleTensor,
        _: &CandleContext,
    ) -> Result<Vec<u32>, BackendFailure> {
        CandleInputInspector.token_ids(tokens)
    }
    fn input_tensor(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }
    fn decode_tensor(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }
    fn scores(value: CandleTensor) -> CandleTensor {
        value
    }
    fn prompt_tokens(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }
    fn decode_prompt_tokens(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }
    fn concatenate_token_parts(
        parts: &[&CandleTensor],
        _: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let parts: Vec<_> = parts.iter().map(|part| &part.value).collect();
        CandleTensor::wrap(candle_core::Tensor::cat(&parts, 1)).map_err(BackendFailure::from_error)
    }
    fn sampling_group(transport: &NoTextTransport, _: CollectiveGroupId) -> Option<&Infallible> {
        match *transport {}
    }
    fn authority(transport: &NoTextTransport) -> &CommunicationAuthority {
        match *transport {}
    }
    fn state_error(error: BackendFailure) -> BackendFailure {
        error
    }
    fn tensor_bytes(value: &CandleTensor) -> u64 {
        (value.value.elem_count() as u64).saturating_mul(value.value.dtype().size_in_bytes() as u64)
    }
    fn decoder_token_ids(value: &CandleTensor) -> Result<Vec<u32>, BackendFailure> {
        CandleInputInspector.token_ids(value)
    }
    fn settle_input(value: &CandleTensor) -> Result<(), BackendFailure> {
        CandleContext::settle_roots([value])
    }
    fn input_failure(error: BackendFailure) -> BackendFailure {
        error
    }
}

impl TextSessionNative<CandleContext> for CandleContext {
    type Generation = CausalTextGeneration;
    type Target = CandleContext;
    type Communication = NoTextTransport;
    type TextValues = CandleTextValues;
    type Retained = CandleContext;
    type ParameterValue = CandleTensor;
    type Retired<R: 'static> = Box<R>;
    type Output = Option<CandleTensor>;
    type Completion = EagerCompletion<BackendFailure, SubmissionLease>;
    type Scope<'a> = CandleTextScope;
    type SampledDecode = u32;
    type ControlTransport = NoTextTransport;

    fn check_native_health(&self) -> Result<(), BackendFailure> {
        eredu_core::session_authority::SessionRetirement::refresh_native_session(self)
    }
    fn validate_target(backend: &Self, target: &Self) -> Result<(), BackendFailure> {
        backend.check_native_health()?;
        target.check_native_health()?;
        if !backend.device.same_device(&target.device) {
            return Err(invalid(
                "bound target belongs to a different Candle context",
            ));
        }
        Ok(())
    }
    fn attach_native(
        target: Self,
        _: Option<NoTextTransport>,
    ) -> Result<(Self, Self), BackendFailure> {
        target.check_native_health()?;
        Ok((target.clone(), target))
    }
    fn retain_resources<R: 'static>(resources: R) -> Box<R> {
        Box::new(resources)
    }
    fn begin(&self, lease: SubmissionLease, _: bool) -> Result<CandleTextScope, BackendFailure> {
        self.open_transaction(lease)
    }
    fn decode_token_id(token: u32, backend: &Self) -> Result<CandleTensor, BackendFailure> {
        backend.tokens(&[token])
    }
    fn decode_sample(token: u32, backend: &Self) -> Result<CandleTensor, BackendFailure> {
        Self::decode_token_id(token, backend)
    }
    fn busy() -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::Busy,
            std::io::Error::other("text resources are retained"),
        )
    }
    fn context(backend: &Self) -> &Self {
        backend
    }
    type LocalCollector<'a> = crate::capture::CandleCapture<'a>;

    fn local_capture<'a>(
        context: &'a Self,
        domain: Option<eredu_core::capture::CaptureTokenDomain<'a>>,
    ) -> Option<Self::LocalCollector<'a>> {
        context
            .placement()
            .supports_local_text_extensions()
            .then(|| crate::capture::CandleCapture::new(context, domain))
    }

    fn local_intervention(
        &self,
    ) -> Option<(
        eredu_core::intervention::InterventionMechanisms,
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
    )> {
        use eredu_core::capture::SpeculativeCapturePrimitives;
        self.placement().supports_local_text_extensions().then(|| {
            (
                CandleTextValues::intervention_mechanisms(),
                CandleTextValues::intervention_estimator(),
            )
        })
    }

    fn supports_host_refinement(&self) -> bool {
        self.placement().supports_local_text_extensions()
    }

    fn host_refinement_canvas(ids: &[u32], context: &Self) -> Result<CandleTensor, BackendFailure> {
        context.tokens(ids)
    }

    fn host_refinement_format(scores: &CandleTensor) -> Option<eredu_nn::TensorElementType> {
        match scores.element_type()? {
            kind @ (eredu_nn::TensorElementType::F32
            | eredu_nn::TensorElementType::F16
            | eredu_nn::TensorElementType::Bf16) => Some(kind),
            _ => None,
        }
    }

    fn host_refinement_feedback(
        scores: &[f32],
        shape: &[i32],
        element: eredu_nn::TensorElementType,
        context: &Self,
    ) -> Result<CandleTensor, BackendFailure> {
        context.check_healthy()?;
        let source = CandleTensor::from_f32_slice(scores, shape, context)
            .map_err(BackendFailure::from_error)?;
        let feedback = source
            .cast_float(element, context)
            .map_err(BackendFailure::from_error);
        if let Err(error) = context.synchronize() {
            context.quarantine((source, feedback.ok()));
            return Err(error);
        }
        feedback
    }

    fn host_refinement_scores(
        scores: &CandleTensor,
        shape: [usize; 2],
        context: &Self,
    ) -> Result<Vec<f32>, BackendFailure> {
        let [positions, vocabulary] = shape;
        if scores.value.dims() != [positions, vocabulary]
            && scores.value.dims() != [1, positions, vocabulary]
        {
            return Err(crate::text::invalid(
                "native scores differ from refinement canvas",
            ));
        }
        scores
            .to_f32_vec(context)
            .map_err(BackendFailure::from_error)
    }

    fn estimate_local_capture(
        source: &[u64],
        selection: &eredu_core::capture::CaptureSelection,
        slice: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        crate::capture::estimate_local_capture(source, selection, slice)
    }

    fn publish_scores(scores: CandleTensor, publish: bool) -> Self::Output {
        publish.then_some(scores)
    }
    fn validate_native_target(
        &self,
        retained: &Self,
        backend: &Self,
    ) -> Result<(), BackendFailure> {
        self.check_native_health()?;
        Self::validate_target(backend, retained)
    }
    fn check_transport_active(_: &Self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_queues(backend: &Self) -> Result<(), BackendFailure> {
        backend.synchronize()
    }
    fn control_transport(_: &Self) -> Option<&NoTextTransport> {
        None
    }
    fn control_transport_identity(transport: &NoTextTransport) -> CommunicationSessionIdentity {
        match *transport {}
    }
    fn parameter_control_supported(transport: &NoTextTransport) -> bool {
        match *transport {}
    }
    fn preparation_admission(
        transport: &NoTextTransport,
        _: usize,
    ) -> Result<Option<eredu_core::run_preparation::TextPreparationAdmission>, BackendFailure> {
        match *transport {}
    }
    fn output_scores(output: &Self::Output) -> Option<&CandleTensor> {
        output.as_ref()
    }
    fn read_observation_tensors(
        &self,
        lease: SubmissionLease,
        root: Rc<dyn Any>,
        backend: &Self,
        tensors: Vec<CandleTensor>,
    ) -> Result<Vec<TensorObservation>, BackendFailure> {
        let result = tensors
            .iter()
            .map(|tensor| {
                backend.validate_tensor_device(tensor)?;
                let values = tensor
                    .to_f32_vec(backend)
                    .map_err(BackendFailure::from_error)?;
                TensorObservation::new(
                    tensor.value.dims().to_vec(),
                    TensorObservationData::F32(values),
                )
                .map_err(BackendFailure::from_error)
            })
            .collect();
        if let Err(error) = backend.synchronize() {
            backend.quarantine((root, tensors, lease));
            return Err(error);
        }
        result
    }
}

impl SharedTextNative for CandleContext {
    type Native = Self;
    fn shared_queue(&self) -> &Self {
        self
    }
    fn shared_token_id(&self, token: &CandleToken) -> Result<u32, BackendFailure> {
        Ok(token.0)
    }
    fn shared_progress_value() -> CandleTensor {
        CandleTensor::from_f32_slice(&[0.], &[1], &Self::cpu()).expect("CPU progress marker")
    }
    fn finish_shared_progress(
        submission: Submission<CandleTensor, Self::TextCompletion>,
    ) -> Result<Self::TextCompletion, BackendFailure> {
        Ok(submission.completion)
    }
}

/// Tensor allocations on the installed session device for the portable media processor.
#[cfg(any(feature = "image", feature = "audio"))]
pub struct CandleProcessorScope(crate::device::CandleSettlement<Vec<CandleTensor>>);

#[cfg(any(feature = "image", feature = "audio"))]
impl eredu_core::input::ProcessorMechanisms for CandleProcessorScope {
    type Tensor = CandleTensor;
    type Error = BackendFailure;

    fn inspector(&self) -> &dyn PreparedInputInspector<CandleTensor> {
        &CandleInputInspector
    }

    fn tensor_u32(
        &mut self,
        values: &[u32],
        shape: &[usize],
    ) -> Result<CandleTensor, BackendFailure> {
        let values = values
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BackendFailure::from_error)?;
        self.tensor_i32(&values, shape)
    }

    fn tensor_f32(
        &mut self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<CandleTensor, BackendFailure> {
        let shape = processor_shape(shape)?;
        let tensor = CandleTensor::from_f32_slice(values, &shape, &self.0.queue)
            .map_err(BackendFailure::from_error)?;
        self.0.resources_mut().push(tensor.clone());
        Ok(tensor)
    }

    fn tensor_i32(
        &mut self,
        values: &[i32],
        shape: &[usize],
    ) -> Result<CandleTensor, BackendFailure> {
        let shape = processor_shape(shape)?;
        let tensor = CandleTensor::from_i32_slice(values, &shape, &self.0.queue)
            .map_err(BackendFailure::from_error)?;
        self.0.resources_mut().push(tensor.clone());
        Ok(tensor)
    }
    #[cfg(feature = "audio")]
    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<CandleTensor, eredu_core::input::OptionalProcessorMechanism<BackendFailure>> {
        use eredu_core::input::OptionalProcessorMechanism;
        let bytes: Vec<_> = values.iter().copied().map(u8::from).collect();
        let mut tensor = CandleTensor::wrap(candle_core::Tensor::from_vec(
            bytes,
            shape,
            &self.0.queue.device,
        ))
        .map_err(BackendFailure::from_error)
        .map_err(OptionalProcessorMechanism::Backend)?;
        tensor.boolean = true;
        self.0.resources_mut().push(tensor.clone());
        Ok(tensor)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
fn processor_shape(shape: &[usize]) -> Result<Vec<i32>, BackendFailure> {
    shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(BackendFailure::from_error)
}

#[cfg(any(feature = "image", feature = "audio"))]
impl eredu_core::input::ProcessorPreparationScope for CandleProcessorScope {
    fn finish(self) -> Result<(), BackendFailure> {
        self.0.finish().map(drop)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl eredu_core::backend::TextMediaNative<CandleContext> for CandleContext {
    type ProcessorScope = CandleProcessorScope;

    fn begin_processor(&self) -> Result<Self::ProcessorScope, BackendFailure> {
        self.check_healthy()?;
        Ok(CandleProcessorScope(crate::device::CandleSettlement::new(
            self,
            Vec::new(),
        )))
    }
}

impl eredu_core::execution_control::TextSnapshotNative for CandleContext {
    fn prompt_tensor_identity(tensor: &CandleTensor) -> Option<InputTensorIdentity> {
        CandleInputInspector.identity(tensor).ok()
    }

    fn prompt_tensor_bytes(tensor: &CandleTensor) -> Option<u64> {
        <CandleCacheStorage as eredu_core::cache::mechanisms::CacheTensorStorage>::snapshot_tensor_costs()?
            .bytes(CandleTextValues::tensor_bytes(tensor), tensor.shape().len())
    }

    fn token_bytes(_: &CandleToken) -> Option<u64> {
        Some(std::mem::size_of::<CandleToken>() as u64)
    }

    fn copy_token(_: &Self, token: &CandleToken) -> Result<CandleToken, BackendFailure> {
        Ok(*token)
    }

    fn copy_prompt_tensor(
        backend: &Self,
        value: &CandleTensor,
    ) -> Result<CandleTensor, BackendFailure> {
        value.compact(backend).map_err(BackendFailure::from_error)
    }

    fn complete_prompt_tensors<'a>(
        roots: impl Iterator<Item = &'a CandleTensor>,
    ) -> Result<(), BackendFailure> {
        CandleContext::settle_roots(roots)
    }
}

impl eredu_core::checkpoint_mechanisms::NativeTextWeights for CandleTextValues {
    type Parameters = CandleNeuralBackend;

    fn text_output_row(
        output: CandleTensor,
        index: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        output
            .index(
                &[eredu_nn::Index::Full, eredu_nn::Index::At(index)],
                context,
            )
            .map_err(BackendFailure::from_error)
    }

    fn complete_text_output(
        output: &CandleTensor,
        state_roots: &[&CandleTensor],
        queue: &CandleContext,
    ) -> Result<(), BackendFailure> {
        let roots = std::iter::once(output.clone())
            .chain(state_roots.iter().map(|root| (*root).clone()))
            .collect::<Vec<_>>();
        crate::device::CandleSettlement::new(queue, roots)
            .finish()
            .map(drop)
    }

    fn text_parameter_memory(
        queue: &CandleContext,
    ) -> Result<Option<eredu_core::capability::ParameterMemoryFacts>, BackendFailure> {
        Ok(Some(crate::memory::placement_memory(queue)))
    }
}

impl eredu_core::checkpoint_mechanisms::TextComponentResources for CandleContext {
    type Values = CandleTextValues;
    type Components = Self;

    fn component_execution(&self) -> &Self {
        self
    }

    fn component_weights(&self) -> &Self {
        self
    }

    fn text_components(&self) -> Self {
        self.clone()
    }
}

impl eredu_core::checkpoint_mechanisms::TextPublicationResources for CandleContext {
    type Values = CandleTextValues;
    type BankStorage = crate::residency::CandleWeightResidency;

    fn publication_context(&self) -> &Self {
        self
    }
}
