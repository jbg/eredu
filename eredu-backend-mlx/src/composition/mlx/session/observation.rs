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
        return TensorObservation::new(shape, data)
            .map_err(|error| Error::ArchitectureModel(error.to_string()));
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
    TensorObservation::new(shape, data).map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl<O> RuntimeActivationObserver<Array, Exception> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Exception> + ?Sized,
{
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Exception> {
        self.inner.routing_control(path, rows)
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, Array>>,
        effective: eredu_runtime::RoutingDecision<'_, Array>,
    ) -> Result<(), Exception> {
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
