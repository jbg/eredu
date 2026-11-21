//! Binding for independently leased layered components with caller-owned state.

use eredu_core::SubmissionBackend;

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use crate::{
    DenseDiskStreamReport, ExecutionUnitAddressMap, LayerWeightResidency, LayeredArchitecture,
    LayeredInvocation, LayeredTensorCohort, LayerwiseModelMetadata, LayerwisePolicy,
    LayerwiseRuntime, LayerwiseRuntimeError, ParallelLayeredArchitecture, ParameterUnitMechanisms,
    ResidencyReport, StatelessComponentState, UnitPolicyResidency,
};
use eredu_core::residency::parameter_binding::WeightBinding;

/// An independently leased component and its optional tensor cohort.
pub struct BoundLayerwiseComponent<A, B, P, C, S>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: crate::RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    runtime: LayerwiseRuntime<A, B, S, P>,
    cohort: Option<C>,
}

impl<A, B, P, C, S> BoundLayerwiseComponent<A, B, P, C, S>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: crate::RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::error::Error + Send + Sync + 'static,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Binds prepared unit recipes using the architecture's physical unit factory.
    pub fn bind<H>(
        architecture: A,
        source: SharedCheckpointSource,
        invocations: ExecutionUnitAddressMap,
        units: Vec<Vec<WeightBinding>>,
        residency: LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
        mechanisms: &H,
        cohort: Option<C>,
    ) -> Result<(Self, LayerwiseModelMetadata), BackendFailure>
    where
        H: ParameterUnitMechanisms<B, UnitPolicy<A::Unit> = P>,
        A::Unit: Parameterized<B::Tensor> + 'static,
    {
        let (policy, metadata) = crate::bind_parameter_units(
            mechanisms,
            source,
            invocations.clone(),
            units,
            residency,
            || crate::build_resident_units::<B, S, A>(&architecture, &invocations, context),
        )?;
        Ok((
            Self {
                runtime: LayerwiseRuntime::new(architecture, policy),
                cohort,
            },
            metadata,
        ))
    }

    /// Borrows the exact architecture that was bound.
    pub fn architecture(&self) -> &A {
        self.runtime.architecture()
    }

    /// Borrows the retained tensor cohort for invocation admission.
    pub fn cohort(&self) -> Option<&C> {
        self.cohort.as_ref()
    }

    /// Borrows the runtime and cohort for architecture-owned guarded operations.
    pub fn parts_mut(&mut self) -> (&mut LayerwiseRuntime<A, B, S, P>, Option<&C>) {
        (&mut self.runtime, self.cohort.as_ref())
    }

    /// Certifies immutable parameter results for the unsharded component.
    pub fn immutable_parameters_identity(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Option<&eredu_core::resources::ResourceIdentity>
    where
        P: crate::ImmutableParameterPolicy<B>,
    {
        if self.cohort.is_none() {
            self.runtime.policy().immutable_parameters_identity(context)
        } else {
            None
        }
    }

    /// Advances caller-owned state and returns the invocation's completed readouts.
    pub fn invoke_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        A::Error: From<BackendFailure>,
        C: LayeredTensorCohort<B>,
    {
        if let Some(cohort) = &self.cohort {
            self.runtime
                .forward_parallel_with_unit_executor_and_traversal_hook(
                    input,
                    state,
                    cohort.parallel_context(),
                    context,
                    |architecture,
                     group,
                     index,
                     unit,
                     hidden,
                     state,
                     forward,
                     parallel,
                     context| {
                        architecture.forward_unit_parallel(
                            group, index, unit, hidden, state, forward, parallel, context,
                        )
                    },
                    &mut cohort.traversal_hook(),
                )
        } else {
            self.runtime.invoke_with_context(input, state, context)
        }
    }

    /// Invokes the bound weights with compatible caller-owned state.
    pub fn forward_with_state<'a, S2>(
        &mut self,
        input: <A as LayeredArchitecture<B, S2>>::Input<'a>,
        state: &mut S2,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<<A as LayeredArchitecture<B, S>>::Error, P::Error>>
    where
        S2: crate::RuntimeState<B>,
        A: ParallelLayeredArchitecture<B, S2>
            + LayeredArchitecture<
                B,
                S2,
                Unit = <A as LayeredArchitecture<B, S>>::Unit,
                Error = <A as LayeredArchitecture<B, S>>::Error,
            >,
        <A as LayeredArchitecture<B, S>>::Error: From<BackendFailure>,
        C: LayeredTensorCohort<B>,
    {
        if let Some(cohort) = &self.cohort {
            self.runtime.forward_parallel_with_state_hook(
                input,
                state,
                cohort.parallel_context(),
                context,
                &mut cohort.traversal_hook(),
            )
        } else {
            self.runtime.forward_with_state(input, state, context)
        }
    }
}

impl<A, B, P, C, S> UnitPolicyResidency for BoundLayerwiseComponent<A, B, P, C, S>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: crate::RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit> + UnitPolicyResidency,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn residency(&self) -> Result<ResidencyReport, BackendFailure> {
        self.runtime.policy().residency()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.runtime.policy().dense_stream_report()
    }
}

/// An independently leased component with no mutable architecture state.
pub type BoundStatelessComponent<A, B, P, C> =
    BoundLayerwiseComponent<A, B, P, C, StatelessComponentState>;

impl<A, B, P, C> BoundStatelessComponent<A, B, P, C>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, StatelessComponentState>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::error::Error + Send + Sync + 'static,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Executes a phase locally or agrees the tensor traversal before collectives.
    pub fn invoke<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, StatelessComponentState>,
        A::Error: From<BackendFailure>,
        C: LayeredTensorCohort<B>,
    {
        let mut state = StatelessComponentState::new()?;
        self.invoke_with_context(input, &mut state, context)
    }
}
