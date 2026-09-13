use super::*;
use eredu_nn::{GroupedUnitBatch, GroupedUnitError, GroupedUnitObserver};

struct NativeUnitObserver<'a> {
    inner: &'a mut dyn GroupedUnitObserver<MlxTensor>,
    failure: Option<ComputeError>,
}

impl common::grouped::NativeGroupedUnitObserver for NativeUnitObserver<'_> {
    fn apply(&mut self, batch: &GroupedUnitBatch<'_, Array>) -> Result<Array, Exception> {
        let result = (|| {
            let batch = GroupedUnitBatch {
                values: MlxTensor::ref_cast(batch.values),
                group_indices: MlxTensor::ref_cast(batch.group_indices),
                selection_indices: MlxTensor::ref_cast(batch.selection_indices),
                token_indices: MlxTensor::ref_cast(batch.token_indices),
                coefficients: MlxTensor::ref_cast(batch.coefficients),
                token_offset: batch.token_offset,
                total_token_count: batch.total_token_count,
                group_count: batch.group_count,
            };
            self.inner.observe(&batch)?;
            let replacement = self.inner.intervene(&batch)?;
            let effective = replacement.as_ref().unwrap_or(batch.values);
            if effective.shape() != batch.values.shape() {
                return Err(ComputeError::backend_source(
                    GroupedUnitError::ReplacementShape {
                        expected: batch.values.shape().to_vec(),
                        actual: effective.shape().to_vec(),
                    },
                ));
            }
            if effective.as_array().dtype() != batch.values.as_array().dtype() {
                return Err(ComputeError::backend_source(
                    GroupedUnitError::ReplacementDtype,
                ));
            }
            self.inner
                .observe_effective(&batch.with_values(effective))?;
            Ok(effective.as_array().clone())
        })();
        result.map_err(|error: ComputeError| {
            let native = Exception::custom(error.to_string());
            self.failure = Some(error);
            native
        })
    }
}

fn with_observer<R>(
    observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    execute: impl FnOnce(
        Option<&mut dyn common::grouped::NativeGroupedUnitObserver>,
    ) -> Result<R, Exception>,
) -> Result<R, ComputeError> {
    let mut adapter = observer.map(|inner| NativeUnitObserver {
        inner,
        failure: None,
    });
    let result = execute(
        adapter
            .as_mut()
            .map(|adapter| adapter as &mut dyn common::grouped::NativeGroupedUnitObserver),
    );
    match adapter.and_then(|adapter| adapter.failure) {
        Some(error) => Err(error),
        None => compute(result),
    }
}

impl MlxGroupedGatedProduct {
    pub(super) fn forward_units(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                context,
                observer,
            )
        })?;
        compute_tensor(output.reshape(input.shape(), context))
    }

    pub(super) fn forward_units_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_tensor_parallel_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                partitions,
                context,
                observer,
            )
        })?;
        reshape_partial(output, input.shape(), context)
    }
}

impl MlxGroupedRelu2 {
    pub(super) fn forward_units(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Relu2,
            context,
        )?;
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                context,
                observer,
            )
        })?;
        compute_tensor(output.reshape(input.shape(), context))
    }

    pub(super) fn forward_units_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Relu2,
            context,
        )?;
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_tensor_parallel_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                partitions,
                context,
                observer,
            )
        })?;
        reshape_partial(output, input.shape(), context)
    }
}

fn reshape_partial(
    output: TensorParallelGroupedOutput<Array>,
    shape: &[i32],
    context: &Stream,
) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
    let (reducible, post_reduce) = output.into_parts();
    Ok(TensorParallelGroupedOutput::new(
        compute_tensor(reducible.reshape(shape, context))?,
        post_reduce
            .map(|bias| compute_tensor(bias.reshape(shape, context)))
            .transpose()?,
    ))
}
