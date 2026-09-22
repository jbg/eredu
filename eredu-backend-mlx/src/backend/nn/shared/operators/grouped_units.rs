use super::*;
use crate::backend::nn::tensor::PreparedGroupedUnitError;
use eredu_nn::{GroupedUnitBatch, GroupedUnitError, GroupedUnitObserver};

struct NativeUnitObserver<'a> {
    inner: &'a mut dyn GroupedUnitObserver<MlxTensor>,
    failure: Option<ComputeError>,
    shape_error: Option<PreparedGroupedUnitError>,
    original: bool,
    ordinary: Option<eredu_core::HostPreparationAuthority>,
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
                return Err(match self.shape_error.take() {
                    Some(prepared) => prepared
                        .shape(batch.values.shape(), effective.shape())
                        .map_err(observation_transport::native)?,
                    None if self.ordinary.is_some() => {
                        ComputeError::backend_retained_source(OrdinaryGroupedShapeFailure {
                            expected: OrdinaryShape::new(batch.values.shape()),
                            actual: OrdinaryShape::new(effective.shape()),
                            _host: self
                                .ordinary
                                .as_ref()
                                .expect("checked ordinary custody")
                                .clone(),
                        })
                    }
                    None => {
                        ComputeError::backend_retained_source(GroupedUnitError::ReplacementShape {
                            expected: batch.values.shape().to_vec(),
                            actual: effective.shape().to_vec(),
                        })
                    }
                });
            }
            if effective.as_array().dtype() != batch.values.as_array().dtype() {
                return Err(match self.shape_error.take() {
                    Some(prepared) => prepared.dtype(),
                    None => {
                        ComputeError::backend_retained_source(GroupedUnitError::ReplacementDtype)
                    }
                });
            }
            self.inner
                .observe_effective(&batch.with_values(effective))?;
            // Transfer an owned replacement after the final borrowed callback.
            // The no-intervention branch shares the original, as Workspace does.
            Ok(match replacement {
                Some(effective) => effective.into(),
                None => batch.values.as_array().clone(),
            })
        })();
        result.map_err(|error: ComputeError| {
            let (error, native) = if self.original {
                let error = observation_transport::callback(error);
                let native = observation_transport::signal(&error);
                (error, native)
            } else if let Some(host) = &self.ordinary {
                let native = Exception::from_retained_source(OrdinaryGroupedSignal {
                    _host: host.clone(),
                });
                let error = ComputeError::backend_retained_source(OrdinaryGroupedFailure {
                    cause: error,
                    _host: host.clone(),
                });
                (error, native)
            } else {
                let native = Exception::custom(error.to_string());
                (error, native)
            };
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
    let mut adapter = match observer {
        Some(inner) => {
            let shape_error =
                PreparedGroupedUnitError::take_current().map_err(observation_transport::native)?;
            let ordinary = if shape_error.is_none() {
                crate::backend::nn::shared::current_ordinary_execution_owner()
                    .map_err(ComputeError::backend_retained_source)?
                    .map(|owner| owner.host().clone())
            } else {
                None
            };
            Some(NativeUnitObserver {
                inner,
                ordinary,
                failure: None,
                original: shape_error.is_some(),
                shape_error,
            })
        }
        None => None,
    };
    let result = execute(
        adapter
            .as_mut()
            .map(|adapter| adapter as &mut dyn common::grouped::NativeGroupedUnitObserver),
    );
    match adapter.and_then(|adapter| adapter.failure) {
        Some(error) => Err(error),
        None => super::super::selected_linear::original::Transport::new(true).compute(result),
    }
}

impl MlxGroupedGatedProduct {
    /// One actual TP forwarding frame, including its reducible/post-bias
    /// carriers and the existing source-preserving result transport.
    pub(crate) fn original_tensor_parallel_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            super::super::selected_linear::original::control_bytes()?,
            size_of::<[Array; 4]>(),
            size_of::<TensorParallelGroupedOutput<Array>>(),
            size_of::<TensorParallelGroupedOutput<MlxTensor>>(),
            size_of::<Result<TensorParallelGroupedOutput<Array>, Exception>>(),
            size_of::<Result<TensorParallelGroupedOutput<Array>, ComputeError>>(),
            size_of::<Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError>>(),
            size_of::<(Array, Option<Array>)>(),
            size_of::<Result<Option<MlxTensor>, ComputeError>>(),
            size_of::<Option<&mut dyn common::grouped::NativeGroupedUnitObserver>>(),
            size_of::<(&MlxTensor, &GroupSelection<MlxTensor>, usize, &Stream)>(),
            size_of::<(&[i32], &Stream)>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    pub(crate) fn original_unit_observation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            super::super::selected_linear::original::control_bytes()?,
            observation_transport::grouped_callback_control_bytes()?,
            size_of::<NativeUnitObserver<'_>>(),
            size_of::<Option<NativeUnitObserver<'_>>>(),
            size_of::<GroupedUnitBatch<'_, MlxTensor>>(),
            size_of::<GroupedUnitBatch<'_, Array>>(),
            size_of::<Option<MlxTensor>>(),
            size_of::<Result<Option<MlxTensor>, ComputeError>>(),
            size_of::<Result<Array, ComputeError>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<Option<ComputeError>>(),
            size_of::<Option<&Array>>(),
            size_of::<&mut dyn GroupedUnitObserver<MlxTensor>>(),
            size_of::<Option<&mut dyn common::grouped::NativeGroupedUnitObserver>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    pub(super) fn forward_units(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        if observer.is_none() {
            return self.forward_grouped(input, selections, context);
        }
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
        let transport = super::super::selected_linear::original::Transport::new(true);
        let input = input.as_array();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                context,
                observer,
            )
        })?;
        transport.tensor(output.reshape(input.shape(), context))
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
        let transport = super::super::selected_linear::original::Transport::new(true);
        let input = input.as_array();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
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
        reshape_partial(output, input.shape(), context, &transport)
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
        let transport = super::super::selected_linear::original::Transport::new(true);
        let input = input.as_array();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = with_observer(observer, |observer| {
            self.module.forward_with_unit_observer(
                &flattened,
                selections.group_indices().as_array(),
                selections.coefficients().as_array(),
                context,
                observer,
            )
        })?;
        transport.tensor(output.reshape(input.shape(), context))
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
        reshape_partial(
            output,
            input.shape(),
            context,
            &super::super::selected_linear::original::Transport::new(false),
        )
    }
}

fn reshape_partial(
    output: TensorParallelGroupedOutput<Array>,
    shape: &[i32],
    context: &Stream,
    transport: &super::super::selected_linear::original::Transport,
) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
    let (reducible, post_reduce) = output.into_parts();
    Ok(TensorParallelGroupedOutput::new(
        transport.tensor(reducible.reshape(shape, context))?,
        post_reduce
            .map(|bias| transport.tensor(bias.reshape(shape, context)))
            .transpose()?,
    ))
}

/// A malformed replacement can carry an arbitrary rank. The funded error keeps
/// complete small shapes and a fixed prefix plus full rank for larger ones.
#[derive(Debug)]
struct OrdinaryShape {
    axes: [i32; 4],
    rank: usize,
}
impl OrdinaryShape {
    fn new(shape: &[i32]) -> Self {
        let mut axes = [0; 4];
        let count = shape.len().min(axes.len());
        axes[..count].copy_from_slice(&shape[..count]);
        Self {
            axes,
            rank: shape.len(),
        }
    }
}
impl std::fmt::Display for OrdinaryShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.axes[..self.rank.min(4)], f)?;
        if self.rank > 4 {
            write!(f, " (first 4 axes of rank {})", self.rank)?;
        }
        Ok(())
    }
}
#[derive(Debug, thiserror::Error)]
#[error("grouped unit replacement shape differs: expected {expected}, actual {actual}")]
struct OrdinaryGroupedShapeFailure {
    expected: OrdinaryShape,
    actual: OrdinaryShape,
    _host: eredu_core::HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OrdinaryGroupedFailure {
    #[source]
    cause: ComputeError,
    _host: eredu_core::HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("grouped unit observer failed; the enclosing adapter retains its cause")]
struct OrdinaryGroupedSignal {
    _host: eredu_core::HostPreparationAuthority,
}

impl MlxGroupedGatedProduct {
    /// Fixed controls of the actual ordinary callback adapter. Callback-owned
    /// observation/replacement payloads are separate producers. The bounded
    /// replacement-shape error never formats or copies an arbitrary-rank Vec.
    pub(crate) fn ordinary_unit_observation_control_bytes(rank: usize) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        if rank > 4 {
            return None;
        }
        let controls = [
            Self::ordinary_tensor_parallel_control_bytes()?,
            Array::ordinary_clone_control_bytes()?,
            ComputeError::retained_source_construction_bytes::<OrdinaryGroupedShapeFailure>()?,
            ComputeError::retained_source_construction_bytes::<GroupedUnitError>()?,
            ComputeError::retained_source_construction_bytes::<OrdinaryGroupedFailure>()?,
            Exception::retained_source_control_bytes::<OrdinaryGroupedSignal>()?,
            size_of::<NativeUnitObserver<'_>>(),
            size_of::<Option<NativeUnitObserver<'_>>>(),
            size_of::<GroupedUnitBatch<'_, MlxTensor>>().checked_mul(2)?,
            size_of::<GroupedUnitBatch<'_, Array>>(),
            size_of::<Option<MlxTensor>>(),
            size_of::<Result<Option<MlxTensor>, ComputeError>>(),
            size_of::<Result<Array, ComputeError>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<Option<ComputeError>>(),
            size_of::<Option<&Array>>(),
            size_of::<&mut dyn GroupedUnitObserver<MlxTensor>>(),
            size_of::<Option<&mut dyn common::grouped::NativeGroupedUnitObserver>>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<OrdinaryShape>().checked_mul(2)?,
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Ordinary TP/result transports of the existing selected-linear wrapper.
    /// Its Original observer lookup returns None; no Original guard is created.
    pub(crate) fn ordinary_tensor_parallel_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            ComputeError::retained_source_construction_bytes::<Exception>()?,
            size_of::<super::super::selected_linear::original::Transport>(),
            size_of::<Result<Option<safemlx::OriginalScopeObserver>, Exception>>(),
            size_of::<Option<Exception>>(),
            // The try_current C out-record is one opaque pointer; no observer
            // owner is constructed on the authenticated ordinary branch.
            size_of::<*const ()>(),
            size_of::<[Array; 4]>(),
            size_of::<TensorParallelGroupedOutput<Array>>(),
            size_of::<TensorParallelGroupedOutput<MlxTensor>>(),
            size_of::<Result<TensorParallelGroupedOutput<Array>, Exception>>(),
            size_of::<Result<TensorParallelGroupedOutput<Array>, ComputeError>>(),
            size_of::<Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError>>(),
            size_of::<(Array, Option<Array>)>(),
            size_of::<Result<Option<MlxTensor>, ComputeError>>(),
            size_of::<Option<&mut dyn common::grouped::NativeGroupedUnitObserver>>(),
            size_of::<(&MlxTensor, &GroupSelection<MlxTensor>, usize, &Stream)>(),
            size_of::<(&[i32], &Stream)>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
}

#[cfg(test)]
impl MlxGroupedGatedProduct {
    pub(crate) fn test_ordinary_unit_observer(
        observer: &mut dyn GroupedUnitObserver<MlxTensor>,
        batch: &GroupedUnitBatch<'_, Array>,
    ) -> Result<Array, ComputeError> {
        with_observer(Some(observer), |adapter| {
            adapter.expect("supplied observer").apply(batch)
        })
    }
}
