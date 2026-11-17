//! Binding contract for one local layerwise component.

use std::{borrow::Borrow, sync::Arc};

use eredu_core::{
    residency::{
        ParameterConversionRetentionEligibility, ParameterConversionRetentionPolicy,
        ParameterConversionRetentionTrimReport, ParameterConversionTrimError,
    },
    BackendFailure,
};
use eredu_nn::NeuralBackend;

use crate::{
    ActivationObserver, ComponentAdapter, ComponentDefinition, DenseDiskStreamReport,
    LayerWeightResidency, LayeredArchitecture, PreparedComponent, ResidencyReport, RuntimeState,
    SelectedStateRealization,
};

/// Exact resources and policies for an architecture-constructed local component.
pub struct PreparedLocalComponent<A> {
    architecture: A,
    component: PreparedComponent<Arc<ComponentDefinition>>,
    policy: LocalComponentPolicy,
}

/// Selected realization of the component's parameters and state.
pub struct LocalComponentPolicy {
    /// Selected parameter residency.
    pub residency: LayerWeightResidency,
    /// State realization selected for this component, if one is retained.
    pub state: Option<SelectedStateRealization>,
    /// Optional conversion retention selected by the load request.
    pub conversion_retention: Option<ParameterConversionRetentionPolicy>,
    /// Architecture-selected eligibility for retaining converted parameters.
    pub conversion_eligibility: ParameterConversionRetentionEligibility,
}

impl<D, K, R> PreparedLocalComponent<ComponentAdapter<D, K, R>>
where
    D: Borrow<ComponentDefinition>,
{
    /// Pairs a constructed architecture with its exact retained parameter declaration.
    pub fn new(
        architecture: ComponentAdapter<D, K, R>,
        component: PreparedComponent<Arc<ComponentDefinition>>,
        policy: LocalComponentPolicy,
    ) -> Result<Self, eredu_nn::Error> {
        if architecture.definition() != component.declaration().as_ref() {
            return Err(eredu_nn::Error::backend(
                "local component bindings differ from the constructed definition",
            ));
        }
        Ok(Self {
            architecture,
            component,
            policy,
        })
    }
}

impl<A> PreparedLocalComponent<A> {
    /// Transfers the validated architecture, resources, and selection to a binder.
    pub fn into_parts(
        self,
    ) -> (
        A,
        PreparedComponent<Arc<ComponentDefinition>>,
        LocalComponentPolicy,
    ) {
        (self.architecture, self.component, self.policy)
    }
}

/// Native mechanisms supplied by a bound layerwise component.
pub trait BoundLocalComponent<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Advances one architecture input with the supplied observer.
    fn execute(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        observer: &mut dyn ActivationObserver<B::Tensor, A::Error>,
    ) -> Result<B::Tensor, A::Error>;

    /// Advances one rank-local input using its matching tensor collective.
    fn execute_parallel(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        parallel: &B::ParallelContext,
    ) -> Result<B::Tensor, A::Error>
    where
        A: crate::ParallelLayeredArchitecture<B, S>;

    /// Borrows the retained state layout.
    fn state_layout(&self) -> &crate::StateLayout;

    /// Runs an architecture-owned readout with the bound weights and execution context.
    fn readout(
        &mut self,
        project: impl FnOnce(
            &mut A,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
    ) -> Result<B::Tensor, A::Error>;

    /// Completes tensors retained at a prefill cursor boundary.
    fn settle<'a>(values: impl Iterator<Item = &'a B::Tensor>) -> Result<(), BackendFailure>
    where
        B::Tensor: 'a;

    /// Creates independent state with the selected storage layout.
    fn fresh_state(&self) -> Result<S, BackendFailure>;

    /// Reports combined parameter residency and optional dense streaming.
    fn residency(&self)
        -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure>;

    /// Releases optional converted parameter retention.
    fn trim_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError>;
}

/// Binds local component resources without knowing their model family.
pub trait LocalComponentMechanisms<B: NeuralBackend, S: RuntimeState<B>> {
    /// Bound component for any compatible layerwise architecture.
    type Local<A>: BoundLocalComponent<A, B, S>
        + crate::LayeredInvocation<B, S, Architecture = A>
        + 'static
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;

    /// Realizes the supplied local component and retains its selected state policy.
    fn bind_local<A>(
        &self,
        prepare: impl FnOnce() -> Result<PreparedLocalComponent<A>, BackendFailure>,
    ) -> Result<Self::Local<A>, BackendFailure>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;
}
