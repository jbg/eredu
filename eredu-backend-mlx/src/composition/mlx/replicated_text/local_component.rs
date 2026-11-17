//! MLX mechanisms for one bound local layerwise component.

use std::marker::PhantomData;

use eredu_core::{
    residency::{ParameterConversionRetentionTrimReport, ParameterConversionTrimError},
    BackendFailure,
};
use eredu_runtime::{
    ActivationObserver, BoundLocalComponent, DenseDiskStreamReport, LayeredArchitecture,
    LayerwiseRuntime, PreparedLocalComponent, ResidencyReport, SelectedStateRealization,
    StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxHybridState,
            execution::generic::{
                prepare_layerwise_policy_from_bindings, MlxSelectedLayerwisePolicy,
            },
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

/// Native binding of an architecture-provided local layerwise component.
pub struct MlxLocalComponent<A>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    runtime: LayerwiseRuntime<
        A,
        MlxNeuralBackend,
        MlxHybridState,
        MlxSelectedLayerwisePolicy<A::Unit, ()>,
    >,
    state_layout: StateLayout,
    selected_state: Option<SelectedStateRealization>,
    stream: Stream,
}

impl<A> MlxLocalComponent<A>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::Unit: 'static,
{
    /// Materializes the exact retained local component resources.
    pub fn bind(
        prepare: impl FnOnce() -> Result<PreparedLocalComponent<A>, BackendFailure>,
        weights: &Stream,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        let prepared =
            detached_preparation(Vec::<Array>::new(), prepare, BackendFailure::from_error)?;
        detached_preparation(
            Vec::<Array>::new(),
            || {
                let (mut architecture, component, policy) = prepared.into_parts();
                let eredu_runtime::LocalComponentPolicy {
                    residency,
                    state,
                    conversion_retention,
                    conversion_eligibility,
                } = policy;
                let state_layout = component.declaration().state_layout().clone();
                let (_, addresses, source, units, pinned) = component.into_parts();
                let (policy, _) = prepare_layerwise_policy_from_bindings(
                    source,
                    &mut architecture,
                    (),
                    PhantomData::<MlxHybridState>,
                    residency,
                    stream,
                    weights,
                    |_| false,
                    addresses,
                    pinned.into_values().flatten().collect(),
                    units,
                )
                .map_err(BackendFailure::from_error)?;
                policy
                    .residency_manager()
                    .configure_parameter_conversion_retention(
                        conversion_retention,
                        conversion_eligibility,
                    )
                    .map_err(BackendFailure::from_error)?;
                let policy = if residency.is_fully_resident() {
                    MlxSelectedLayerwisePolicy::resident(
                        policy
                            .into_resident(&architecture, stream, PhantomData::<MlxHybridState>)
                            .map_err(BackendFailure::from_error)?,
                    )
                } else {
                    MlxSelectedLayerwisePolicy::bounded(policy)
                };
                Ok(Self {
                    runtime: LayerwiseRuntime::new(architecture, policy),
                    state_layout,
                    selected_state: state,
                    stream: stream.clone(),
                })
            },
            BackendFailure::from_error,
        )
    }
}

impl<A> BoundLocalComponent<A, MlxNeuralBackend, MlxHybridState> for MlxLocalComponent<A>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::Unit: 'static,
{
    fn execute(
        &mut self,
        input: A::Input<'_>,
        state: &mut MlxHybridState,
        observer: &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        self.runtime
            .forward_with_observer(input, state, &self.stream, observer)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn execute_parallel(
        &mut self,
        input: A::Input<'_>,
        state: &mut MlxHybridState,
        parallel: &<MlxNeuralBackend as eredu_nn::NeuralBackend>::ParallelContext,
    ) -> Result<MlxTensor, eredu_nn::Error>
    where
        A: eredu_runtime::ParallelLayeredArchitecture<MlxNeuralBackend, MlxHybridState>,
    {
        self.runtime
            .forward_parallel(input, state, parallel, &self.stream)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    fn readout(
        &mut self,
        project: impl FnOnce(&mut A, &Stream) -> Result<MlxTensor, eredu_nn::Error>,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        project(self.runtime.architecture_mut(), &self.stream)
    }

    fn settle<'a>(values: impl Iterator<Item = &'a MlxTensor>) -> Result<(), BackendFailure> {
        safemlx::transforms::eval(values.map(MlxTensor::as_array))
            .map_err(BackendFailure::from_error)
    }

    fn fresh_state(&self) -> Result<MlxHybridState, BackendFailure> {
        if let Some(selected) = &self.selected_state {
            let manager =
                super::selected_state_manager(selected).map_err(BackendFailure::from_error)?;
            MlxHybridState::from_selected(selected, manager, None)
                .map_err(BackendFailure::from_error)
        } else {
            MlxHybridState::device(self.state_layout.clone(), &[])
                .map_err(BackendFailure::from_error)
        }
    }

    fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        let policy = self.runtime.policy();
        Ok((
            policy
                .residency_report()
                .map_err(BackendFailure::from_error)?,
            policy
                .dense_stream_report()
                .map_err(BackendFailure::from_error)?,
        ))
    }

    fn trim_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.runtime.policy().trim_parameter_conversions()
    }
}

impl<A> eredu_runtime::LayeredInvocation<MlxNeuralBackend, MlxHybridState> for MlxLocalComponent<A>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    type Architecture = A;
    type Error =
        eredu_runtime::LayerwiseRuntimeError<eredu_nn::Error, crate::backend::error::Error>;

    fn architecture_mut(&mut self) -> &mut A {
        self.runtime.architecture_mut()
    }

    fn invoke_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut MlxHybridState,
        context: &Stream,
    ) -> Result<(MlxTensor, A::ForwardContext), Self::Error> {
        self.runtime.invoke_with_context(input, state, context)
    }
}

impl<A> eredu_runtime::ParallelLayeredInvocation<MlxNeuralBackend, MlxHybridState>
    for MlxLocalComponent<A>
where
    A: eredu_runtime::ParallelLayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
        Error = eredu_nn::Error,
    >,
    A::Unit: 'static,
{
    fn invoke_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut MlxHybridState,
        parallel: &<MlxNeuralBackend as eredu_nn::NeuralBackend>::ParallelContext,
        context: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        self.runtime
            .forward_parallel(input, state, parallel, context)
    }
}
