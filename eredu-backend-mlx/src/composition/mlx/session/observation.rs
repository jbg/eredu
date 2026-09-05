use super::*;

pub(super) struct ArrayObserverAdapter<'a, O: ?Sized> {
    pub(super) inner: &'a mut O,
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
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(observations)
    }
}

impl RuntimeActivationObserver<MlxTensor, Exception> for InspectionCollector<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Exception> {
        self.capture(path, value);
        Ok(())
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), Exception> {
        let root = format!("{}.routing", routing.path);
        self.capture(
            &format!("{root}.selected_experts"),
            routing.selected_experts,
        );
        self.capture(&format!("{root}.selected_scores"), routing.selected_scores);
        self.capture(&format!("{root}.coefficients"), routing.coefficients);
        self.capture(&format!("{root}.routed_output"), routing.routed_output);
        if let Some(value) = routing.local_routed_output {
            self.capture(&format!("{root}.local_routed_output"), value);
        }
        if let Some(value) = routing.reduced_routed_output {
            self.capture(&format!("{root}.reduced_routed_output"), value);
        }
        if let Some(value) = routing.shared_output {
            self.capture(&format!("{root}.shared_output"), value);
        }
        if let Some(value) = routing.combined_output {
            self.capture(&format!("{root}.combined_output"), value);
        }
        Ok(())
    }
}

pub(super) fn observe_tensor(
    value: &MlxTensor,
    stream: &Stream,
) -> Result<TensorObservation, Error> {
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
    TensorObservation::new(shape, data).map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl<O> RuntimeActivationObserver<Array, Exception> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Exception> + ?Sized,
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Exception> {
        self.inner.observe(path, MlxTensor::ref_cast(value))
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
        self.inner
            .intervene(path, MlxTensor::ref_cast(value))
            .map(|replacement| replacement.map(MlxTensor::into_array))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), Exception> {
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
