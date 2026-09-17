use super::*;
use crate::backend::managed_memory::NativeMemoryRetention;
use eredu_runtime::working_memory::{InferenceRetention, WorkingMemoryReservation};
use std::sync::Arc;

/// Payload-free authority suitable for native descriptor/backing attachment.
/// It must never retain a session, observer, array, or source allocation.
#[derive(Clone)]
pub(super) struct ArrayObserverAllocationAuthority(Arc<ObserverAllocationRetention>);

struct ObserverAllocationRetention {
    _memory: NativeMemoryRetention,
    _reservations: Vec<WorkingMemoryReservation>,
}

impl ArrayObserverAllocationAuthority {
    pub(super) fn new(memory: NativeMemoryRetention, inference: &InferenceRetention) -> Self {
        Self(Arc::new(ObserverAllocationRetention {
            _memory: memory,
            _reservations: inference
                .requests()
                .filter_map(|request| request.memory_reservation().cloned())
                .collect(),
        }))
    }

    fn retain(&self, array: &Array) -> Result<(), Error> {
        array
            .retain_deferred_allocation_owner(Arc::clone(&self.0))
            .map_err(|failure| {
                let (error, _unattached) = failure.into_parts();
                Error::from(error)
            })
    }
}

pub(super) struct ArrayObserverAdapter<'a, O: ?Sized> {
    pub(super) inner: &'a mut O,
    pub(super) routed_path: Option<String>,
    pub(super) routed_invocation_active: bool,
    // Covers values crossing this adapter, including returned replacements.
    // Callbacks remain responsible for other independent arrays they create
    // and retain without returning them through this interface.
    pub(super) allocation_authority: Option<ArrayObserverAllocationAuthority>,
}

impl<O: ?Sized> ArrayObserverAdapter<'_, O> {
    fn retain_arrays<'a>(&self, arrays: impl IntoIterator<Item = &'a Array>) -> Result<(), Error> {
        if let Some(authority) = &self.allocation_authority {
            for array in arrays {
                authority.retain(array)?;
            }
        }
        Ok(())
    }

    fn retain_routed_batch(
        &self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.retain_arrays([
            batch.units.values,
            batch.units.group_indices,
            batch.units.selection_indices,
            batch.units.token_indices,
            batch.units.coefficients,
            batch.source_groups,
        ])
        .map_err(eredu_nn::Error::backend_source)
    }
}

pub(super) struct InspectionCollector<'a> {
    request: &'a ObservationRequest,
    values: Vec<(String, MlxTensor)>,
}

impl<'a> InspectionCollector<'a> {
    pub(super) fn new(request: &'a ObservationRequest) -> Self {
        Self {
            request,
            values: Vec::new(),
        }
    }

    pub(super) fn capture(&mut self, path: &str, value: &MlxTensor) {
        if self.request.matches(path) {
            self.values.push((path.into(), value.clone()));
        }
    }

    pub(super) fn materialize(self, stream: &Stream) -> Result<ObservationSet, Error> {
        let mut observations = ObservationSet::new();
        for (path, value) in self.values {
            observations
                .insert(
                    path,
                    ObservationValue::Tensor(observe_tensor(&value, stream)?),
                )
                .map_err(Error::observation)?;
        }
        Ok(observations)
    }
}

impl RuntimeActivationObserver<MlxTensor, Error> for InspectionCollector<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Error> {
        self.capture(path, value);
        Ok(())
    }

    fn observe_generated(
        &mut self,
        path: &str,
        _: &MlxTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, Error>,
    ) -> Result<(), Error> {
        if self.request.matches(path) {
            self.capture(path, &generate()?);
        }
        Ok(())
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), Error> {
        routing.for_each_tensor(|path, value| self.capture(&path, value));
        Ok(())
    }
}

pub(super) fn observe_tensor(
    value: &MlxTensor,
    stream: &Stream,
) -> Result<TensorObservation, Error> {
    #[cfg(test)]
    super::bounded_capture::record_host_read(value.as_array().size());
    let shape = value
        .shape()
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Error::ArchitectureModel(format!(
                    "observed tensor has negative dimension {dimension}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.contains(&0) {
        let data = match value.as_array().dtype() {
            Dtype::Bool => TensorObservationData::Bool(Vec::new()),
            Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 => {
                TensorObservationData::U64(Vec::new())
            }
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => {
                TensorObservationData::I64(Vec::new())
            }
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 => {
                TensorObservationData::F32(Vec::new())
            }
            Dtype::Complex64 => {
                return Err(Error::ArchitectureModel(
                    "complex activation observation is unsupported".into(),
                ));
            }
        };
        return TensorObservation::new(shape, data).map_err(Error::observation);
    }
    // Preserve the admitted payload capacity even for one-element results;
    // collecting a generic iterator may otherwise use Vec's minimum growth size.
    macro_rules! mapped_values {
        ($ty:ty, $map:expr) => {{
            let evaluated = value.as_array().evaluated()?;
            let values = evaluated
                .try_iter::<$ty>()
                .map_err(eredu_nn::Error::backend_source)?;
            let mut output = Vec::with_capacity(values.len());
            output.extend(values.map($map));
            output
        }};
    }
    let data = match value.as_array().dtype() {
        Dtype::Bool => TensorObservationData::Bool(
            value
                .as_array()
                .evaluated()?
                .try_to_vec::<bool>()
                .map_err(eredu_nn::Error::backend_source)?,
        ),
        Dtype::Uint8 => TensorObservationData::U64(mapped_values!(u8, u64::from)),
        Dtype::Uint16 => TensorObservationData::U64(mapped_values!(u16, u64::from)),
        Dtype::Uint32 => TensorObservationData::U64(mapped_values!(u32, u64::from)),
        Dtype::Uint64 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .try_to_vec::<u64>()
                .map_err(eredu_nn::Error::backend_source)?,
        ),
        Dtype::Int8 => TensorObservationData::I64(mapped_values!(i8, i64::from)),
        Dtype::Int16 => TensorObservationData::I64(mapped_values!(i16, i64::from)),
        Dtype::Int32 => TensorObservationData::I64(mapped_values!(i32, i64::from)),
        Dtype::Int64 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .try_to_vec::<i64>()
                .map_err(eredu_nn::Error::backend_source)?,
        ),
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 => {
            TensorObservationData::F32(value.to_f32_vec(stream)?)
        }
        Dtype::Complex64 => {
            return Err(Error::ArchitectureModel(
                "complex activation observation is unsupported".into(),
            ));
        }
    };
    TensorObservation::new(shape, data).map_err(Error::observation)
}

impl<O> RuntimeActivationObserver<Array, Error> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Error> + ?Sized,
{
    fn requires_prepared_traversal(&self) -> bool {
        self.inner.requires_prepared_traversal()
    }
    fn requires_sequence_readout(&self) -> bool {
        self.inner.requires_sequence_readout()
    }
    fn original_speculative_capture(&self) -> Option<eredu_runtime::capture::OriginalSpeculativeCaptureInvocation<'_>> { self.inner.original_speculative_capture() }
    fn retain_original_speculative_capture(&mut self, capture: eredu_core::speculative::SpeculativeActivationCapture) -> Result<(), eredu_runtime::capture::CaptureProtocolError> { self.inner.retain_original_speculative_capture(capture) }
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedPrefillCapture<'_>> {
        self.inner.admitted_prefill_capture()
    }
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedCaptureContinuation<'_>> {
        self.inner.admitted_capture_continuation()
    }
    fn ordinary_prefill_capture(&self) -> Option<&eredu_runtime::capture::OrdinaryPrefillCapture> {
        self.inner.ordinary_prefill_capture()
    }
    fn supports_prefill_spans(&self) -> bool {
        self.inner.supports_prefill_spans()
    }
    fn supports_prefill_context(&self) -> bool {
        self.inner.supports_prefill_context()
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), Error> {
        self.inner.begin_prefill_context(frontier)
    }

    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<Array>>, Error> {
        let Some(observer) = self.inner.routed_unit_observer(path)? else {
            self.routed_path = None;
            self.routed_invocation_active = false;
            return Ok(None);
        };
        self.routed_invocation_active = observer.invocation_active();
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(path.into());
        }
        Ok(Some(self))
    }
    fn transactional(&self) -> bool {
        self.inner.transactional()
    }
    fn begin_prefill_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Error> {
        self.inner.begin_prefill_chunk(chunk)
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.inner.finish_prefill(committed);
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.inner.requires_prefill_opening_state()
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
        opening: &eredu_runtime::inspection::PrefillOpeningState<'_, Array>,
    ) -> Result<Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>, Error> {
        opening.with_tensor_adapter(MlxTensor::ref_cast, |opening| {
            self.inner
                .prepare_prefill_chunk_with_opening(context, opening)
        })
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>, Error> {
        self.inner.prepare_prefill_chunk_retention(context)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        settled: eredu_runtime::inspection::SettledPrefillChunkRetention,
    ) -> Result<(), Error> {
        self.inner.retire_prefill_chunk_retention(settled)
    }

    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.inner.prepare_transaction(epoch, pass)
    }
    fn coordinate_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), Error> {
        self.inner.coordinate_transaction(epoch)
    }
    fn complete_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), Error> {
        self.inner.complete_transaction(epoch)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.inner.finish_transaction(epoch, committed)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Error> {
        self.inner.routing_control(path, rows)
    }

    fn routing_unmodified_interest(&self, path: &str) -> eredu_runtime::RoutingUnmodifiedInterest {
        self.inner.routing_unmodified_interest(path)
    }
    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: eredu_runtime::RoutingDecision<'_, Array>,
    ) -> Result<(), Error> {
        match self.inner.routing_unmodified_interest(path) {
            eredu_runtime::RoutingUnmodifiedInterest::None => return Ok(()),
            eredu_runtime::RoutingUnmodifiedInterest::Metadata => (),
            eredu_runtime::RoutingUnmodifiedInterest::Values => {
                self.retain_arrays([effective.ids, effective.coefficients])?;
            }
        }
        self.inner.routing_unmodified(
            path,
            eredu_runtime::RoutingDecision {
                ids: MlxTensor::ref_cast(effective.ids),
                coefficients: MlxTensor::ref_cast(effective.coefficients),
            },
        )
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, Array>>,
        effective: eredu_runtime::RoutingDecision<'_, Array>,
    ) -> Result<(), Error> {
        if let Some(original) = &original {
            self.retain_arrays([original.ids, original.coefficients])?;
        }
        self.retain_arrays([effective.ids, effective.coefficients])?;
        self.inner.routing_applied(
            path,
            original.map(|value| eredu_runtime::RoutingDecision {
                ids: MlxTensor::ref_cast(value.ids),
                coefficients: MlxTensor::ref_cast(value.coefficients),
            }),
            eredu_runtime::RoutingDecision {
                ids: MlxTensor::ref_cast(effective.ids),
                coefficients: MlxTensor::ref_cast(effective.coefficients),
            },
        )
    }

    fn routing_failed(&mut self, path: &str, message: &str) {
        self.inner.routing_failed(path, message);
    }

    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Error> {
        self.retain_arrays([value])?;
        self.inner.observe(path, MlxTensor::ref_cast(value))
    }
    fn observe_replica(&mut self, path: &str, value: &Array) -> Result<(), Error> {
        self.retain_arrays([value])?;
        self.inner.observe_replica(path, MlxTensor::ref_cast(value))
    }

    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<Array, Error>,
    ) -> Result<(), Error> {
        self.retain_arrays([prototype])?;
        let authority = self.allocation_authority.clone();
        self.inner
            .observe_generated(path, MlxTensor::ref_cast(prototype), source, &mut || {
                let generated = generate()?;
                if let Some(authority) = &authority {
                    authority.retain(&generated)?;
                }
                Ok(MlxTensor::from_array(generated))
            })
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<Array, Error>,
    ) -> Result<(), Error> {
        self.retain_arrays([prototype])?;
        let authority = self.allocation_authority.clone();
        let mut protected = AuthorityGeneratedFactory { factory, authority };
        let mut mapped = eredu_nn::MappedGeneratedTensorFactory::new(
            &mut protected,
            MlxTensor::ref_cast,
            MlxTensor::from_array,
            std::convert::identity::<Error>,
            |_: &Error| Error::Other(Box::new(eredu_nn::GeneratedTensorRetentionSignal)),
        );
        self.inner.observe_generated_retained(
            path,
            MlxTensor::ref_cast(prototype),
            source,
            &mut mapped,
        )
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Error> {
        self.retain_arrays([value])?;
        let replacement = self
            .inner
            .intervene(path, MlxTensor::ref_cast(value))
            .map(|replacement| replacement.map(MlxTensor::into_array))?;
        self.retain_arrays(replacement.iter())?;
        Ok(replacement)
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), Error> {
        self.retain_arrays(
            [
                Some(routing.selected_experts),
                Some(routing.selected_scores),
                Some(routing.coefficients),
                Some(routing.routed_output),
                routing.local_routed_output,
                routing.reduced_routed_output,
                routing.shared_output,
                routing.combined_output,
            ]
            .into_iter()
            .flatten(),
        )?;
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: MlxTensor::ref_cast(routing.selected_experts),
                selected_scores: MlxTensor::ref_cast(routing.selected_scores),
                coefficients: MlxTensor::ref_cast(routing.coefficients),
                routed_output: MlxTensor::ref_cast(routing.routed_output),
                local_routed_output: routing.local_routed_output.map(MlxTensor::ref_cast),
                reduced_routed_output: routing.reduced_routed_output.map(MlxTensor::ref_cast),
                shared_output: routing.shared_output.map(MlxTensor::ref_cast),
                combined_output: routing.combined_output.map(MlxTensor::ref_cast),
                expert_count: routing.expert_count,
            })
    }
}

impl<O: RuntimeActivationObserver<MlxTensor, Error> + ?Sized>
    eredu_runtime::RoutedUnitObserver<Array> for ArrayObserverAdapter<'_, O>
{
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.retain_arrays([invocation.input])
            .map_err(eredu_nn::Error::backend_source)?;
        self.routed_invocation_active = true;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer.begin_invocation(&eredu_runtime::RoutedUnitInvocation {
                input: MlxTensor::ref_cast(invocation.input),
                origins: invocation.origins,
                unit_coordinates: invocation.unit_coordinates,
            }),
            None => Ok(()),
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.routed_invocation_active = false;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer.finish_invocation(success),
            None => Ok(()),
        }
    }
    fn invocation_active(&self) -> bool {
        self.routed_invocation_active
    }
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.retain_routed_batch(batch)?;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer.observe(&batch.map_tensors(MlxTensor::ref_cast)),
            None => Ok(()),
        }
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, Array>,
    ) -> Result<Option<Array>, eredu_nn::Error> {
        self.retain_routed_batch(batch)?;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        let replacement = match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer
                .intervene(&batch.map_tensors(MlxTensor::ref_cast))
                .map(|value| value.map(MlxTensor::into_array)),
            None => Ok(None),
        }?;
        self.retain_arrays(replacement.iter())
            .map_err(eredu_nn::Error::backend_source)?;
        Ok(replacement)
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.retain_routed_batch(batch)?;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer.observe_effective(&batch.map_tensors(MlxTensor::ref_cast)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "observation/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "observation/memory_tests.rs"]
mod memory_tests;

// No source or numerical payload is cloned by this adapter. The actual native
// owner is attached before a created root reaches the downstream retention sink.
struct AuthorityGeneratedFactory<'a> {
    factory: &'a mut dyn eredu_nn::RetainedGeneratedTensorFactory<Array, Error>,
    authority: Option<ArrayObserverAllocationAuthority>,
}
impl eredu_nn::RetainedGeneratedTensorFactory<Array, Error> for AuthorityGeneratedFactory<'_> {
    fn program(&self) -> eredu_nn::GeneratedTensorProgram<'_> {
        self.factory.program()
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(eredu_nn::GeneratedTensorSourceRole, &Array) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let authority = &self.authority;
        self.factory.visit_sources(&mut |role, value| {
            if let Some(authority) = authority {
                authority.retain(value)?;
            }
            retain(role, value)
        })
    }
    fn generate(
        &mut self,
        retain: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<Array, Error> {
        let authority = &self.authority;
        self.factory.generate(&mut |value| {
            if let Some(authority) = authority {
                authority.retain(value)?;
            }
            retain(value)
        })
    }
}

#[cfg(test)]
#[path = "observation/prefill_tests.rs"]
mod prefill_tests;
