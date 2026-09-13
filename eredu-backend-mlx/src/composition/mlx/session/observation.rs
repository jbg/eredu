use super::*;

pub(super) struct ArrayObserverAdapter<'a, O: ?Sized> {
    pub(super) inner: &'a mut O,
    pub(super) routed_path: Option<String>,
    pub(super) routed_invocation_active: bool,
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
                ))
            }
        };
        return TensorObservation::new(shape, data).map_err(Error::observation);
    }
    let data = match value.as_array().dtype() {
        Dtype::Bool => {
            TensorObservationData::Bool(value.as_array().evaluated()?.as_slice::<bool>().to_vec())
        }
        Dtype::Uint8 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u8>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint16 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u16>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint32 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u32>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint64 => {
            TensorObservationData::U64(value.as_array().evaluated()?.as_slice::<u64>().to_vec())
        }
        Dtype::Int8 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i8>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int16 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i16>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int32 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i32>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int64 => {
            TensorObservationData::I64(value.as_array().evaluated()?.as_slice::<i64>().to_vec())
        }
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 => {
            TensorObservationData::F32(value.to_f32_vec(stream)?)
        }
        Dtype::Complex64 => {
            return Err(Error::ArchitectureModel(
                "complex activation observation is unsupported".into(),
            ))
        }
    };
    TensorObservation::new(shape, data).map_err(Error::observation)
}

impl<O> RuntimeActivationObserver<Array, Error> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Error> + ?Sized,
{
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

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, Array>>,
        effective: eredu_runtime::RoutingDecision<'_, Array>,
    ) -> Result<(), Error> {
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
        self.inner.observe(path, MlxTensor::ref_cast(value))
    }
    fn observe_replica(&mut self, path: &str, value: &Array) -> Result<(), Error> {
        self.inner.observe_replica(path, MlxTensor::ref_cast(value))
    }

    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<Array, Error>,
    ) -> Result<(), Error> {
        self.inner
            .observe_generated(path, MlxTensor::ref_cast(prototype), source, &mut || {
                generate().map(MlxTensor::from_array)
            })
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Error> {
        self.inner
            .intervene(path, MlxTensor::ref_cast(value))
            .map(|replacement| replacement.map(MlxTensor::into_array))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), Error> {
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
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing array routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(eredu_nn::Error::backend_source)?
        {
            Some(observer) => observer
                .intervene(&batch.map_tensors(MlxTensor::ref_cast))
                .map(|value| value.map(MlxTensor::into_array)),
            None => Ok(None),
        }
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
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
