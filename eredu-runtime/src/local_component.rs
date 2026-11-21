//! Binding contract for one local layerwise component.

use eredu_core::backend::LocalComponentStorage;
use eredu_core::checkpoint_mechanisms::unit_binding::{
    ComponentPreparationScope, LocalComponentBinding,
};

mod driver;
pub use driver::{LocalComponentDriver, LocalParameterRetention};

use std::{borrow::Borrow, sync::Arc};

use eredu_core::{
    residency::{
        ParameterConversionRetentionEligibility, ParameterConversionRetentionPolicy,
        ParameterConversionRetentionTrimReport, ParameterConversionTrimError,
    },
    BackendFailure, SubmissionBackend,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    ActivationObserver, ComponentAdapter, ComponentDefinition, DenseDiskStreamReport,
    LayerWeightResidency, LayeredArchitecture, ParameterUnitMechanisms, PreparedComponent,
    ResidencyReport, RuntimeState, SelectedStateRealization,
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
    /// Constructs resident units above native materialization and binds the selected driver.
    pub fn bind<B, S, M>(
        self,
        mechanisms: &M,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<M::Local<A>, BackendFailure>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        M: LocalComponentMechanisms<B, S>,
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static,
    {
        let scope = mechanisms.begin_local_preparation()?;
        let result = (|| {
            let (mut architecture, prepared, mut policy) = self.into_parts();
            let state = policy.state.take();
            let layout = prepared.declaration().state_layout().clone();
            let residency = policy.residency;
            let invocations = prepared.invocations().clone();
            let storage = mechanisms.prepare_local_storage(&mut architecture, prepared, policy)?;
            let units = residency
                .is_fully_resident()
                .then(|| {
                    crate::build_resident_units::<B, S, A>(&architecture, &invocations, context)
                })
                .transpose()?;
            mechanisms.finish_local_binding(architecture, storage, units, layout, state)
        })();
        scope.finish(result)
    }

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
    /// Scope covering native work issued during architecture construction.
    type PreparationScope: ComponentPreparationScope;

    /// Opens recovery before the caller prepares the local architecture.
    fn begin_local_preparation(&self) -> Result<Self::PreparationScope, BackendFailure>;

    /// Bound component for any compatible layerwise architecture.
    type Local<A>: BoundLocalComponent<A, B, S>
        + crate::LayeredInvocation<
            B,
            S,
            Architecture = A,
            Error: std::error::Error + Send + Sync + 'static,
        > + 'static
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;

    /// Native storage of the admitted local component.
    type Storage;

    /// Binds pinned parameters and prepares the selected unit storage.
    fn prepare_local_storage<A>(
        &self,
        architecture: &mut A,
        prepared: PreparedComponent<Arc<ComponentDefinition>>,
        policy: LocalComponentPolicy,
    ) -> Result<Self::Storage, BackendFailure>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;

    /// Populates supplied resident units, or adopts prepared bounded storage.
    fn finish_local_binding<A>(
        &self,
        architecture: A,
        storage: Self::Storage,
        resident: Option<Vec<A::Unit>>,
        layout: crate::StateLayout,
        state: Option<SelectedStateRealization>,
    ) -> Result<Self::Local<A>, BackendFailure>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;
}

impl<B, S, M> LocalComponentMechanisms<B, S> for M
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + LocalComponentStorage
        + 'static,
    B::OwnedExecutor: 'static,
    S: crate::state::PortableTextState<B, B::CacheStorage> + 'static,
    M: LocalComponentBinding<B> + ParameterUnitMechanisms<B>,
{
    type PreparationScope = M::Scope;
    type Storage = crate::ParameterUnitStorage<B, M>;

    type Local<A>
        = LocalComponentDriver<A, B, S, M::UnitPolicy<A::Unit>>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;

    fn begin_local_preparation(&self) -> Result<Self::PreparationScope, BackendFailure> {
        self.begin_unit_binding()
    }

    fn prepare_local_storage<A>(
        &self,
        architecture: &mut A,
        prepared: PreparedComponent<Arc<ComponentDefinition>>,
        policy: LocalComponentPolicy,
    ) -> Result<Self::Storage, BackendFailure>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static,
    {
        let (_, invocations, source, units, pinned) = prepared.into_parts();
        let (storage, _) = crate::parameter_units::materialize_component_storage::<B, S, A, M>(
            self,
            architecture,
            source,
            &invocations,
            policy.residency,
            pinned.into_values().flatten().collect(),
            units,
            Some((policy.conversion_retention, policy.conversion_eligibility)),
        )?;
        Ok(storage)
    }

    fn finish_local_binding<A>(
        &self,
        architecture: A,
        storage: Self::Storage,
        resident: Option<Vec<A::Unit>>,
        layout: crate::StateLayout,
        state: Option<SelectedStateRealization>,
    ) -> Result<Self::Local<A>, BackendFailure>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static,
    {
        use crate::bounded_execution::units::{BindBoundedPolicy as _, BindResidentPolicy as _};
        let policy = match resident {
            Some(units) => {
                M::UnitPolicy::<A::Unit>::bind_resident(storage, units, self.unit_executor())?
            }
            None => M::UnitPolicy::<A::Unit>::bind_bounded(storage),
        };
        Ok(LocalComponentDriver::new(
            architecture,
            policy,
            layout,
            state,
            self.local_executor(),
        ))
    }
}
