//! The selected text route retains its checkpoint and native storage operators.

use eredu_core::checkpoint_mechanisms::bounded_quantization::BoundedQuantizationBackend;
use eredu_core::checkpoint_mechanisms::unit_binding::ParameterStorageBinding;

use std::marker::PhantomData;

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    checkpoint_mechanisms::{NativeTextWeights, TextComponentResources},
    BackendFailure, SubmissionBackend,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    replicated_session::{ParameterMemoryFacts, ReplicatedTextPrimitives, TextStateStorage},
    state::PortableTextState,
    ParameterUnitMechanisms, ParameterUnitStorage,
};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Portable text parameter binding for an exact source and state profile.
pub struct TextRouteWeights<B, S, W, F> {
    source: SharedCheckpointSource,
    storage: W,
    operations: PhantomData<fn() -> (B, S, F)>,
}

impl<B, S, W, F> TextRouteWeights<B, S, W, F> {
    /// Retains the exact source and physical parameter binding.
    pub(in crate::executable_binding) fn new(source: SharedCheckpointSource, storage: W) -> Self {
        Self {
            source,
            storage,
            operations: PhantomData,
        }
    }
}

pub(in crate::executable_binding) type Weights<B, S, N> = TextRouteWeights<
    B,
    S,
    <N as TextComponentResources>::Components,
    <N as TextComponentResources>::Values,
>;

impl<B, S, W, F> TextStateStorage<B> for TextRouteWeights<B, S, W, F>
where
    B: SubmissionBackend<Executor = Context<B>>,
    Context<B>: Sized,
    F: NativeTextWeights<Neural = B, Tensor = B::Tensor, Context = Context<B>>
        + super::super::TextPredictionPrimitives,
    S: PortableTextState<B, F::Cache>,
{
    type CacheStorage = F::Cache;
    type Owner = S;
    type NativeError = F::Error;

    fn state_failure(error: BackendFailure) -> Self::NativeError {
        error.into()
    }
}

impl<B, S, W, F> ReplicatedTextPrimitives<B> for TextRouteWeights<B, S, W, F>
where
    B: SubmissionBackend<Executor = Context<B>>,
    Context<B>: Sized,
    F: NativeTextWeights<Neural = B, Tensor = B::Tensor, Context = Context<B>>
        + super::super::TextPredictionPrimitives,
    S: PortableTextState<B, F::Cache>,
    W: ParameterUnitMechanisms<B, UnitError = F::Error>,
{
    type Parameters = F::Parameters;
    type UnitStorage = ParameterUnitStorage<B, Self>;
    type WeightStorage = W::ResidencyMechanisms;
    type UnitBinding<'a>
        = &'a Self
    where
        Self: 'a;

    fn take_parameter_storage(&mut self) -> Result<Self::UnitBinding<'_>, Self::Error> {
        Ok(self)
    }

    fn parameter_source(&self) -> Option<SharedCheckpointSource> {
        Some(self.source.clone())
    }

    fn parameter_memory_facts(&self) -> Result<Option<ParameterMemoryFacts>, Self::Error> {
        F::text_parameter_memory(self.storage.unit_executor())
    }

    fn index_text_output(
        &mut self,
        output: B::Tensor,
        index: i32,
        queue: &Context<B>,
    ) -> Result<B::Tensor, Self::Error> {
        F::text_output_row(output, index, queue)
    }

    fn complete(
        &mut self,
        output: &B::Tensor,
        roots: &[&B::Tensor],
        queue: &Context<B>,
    ) -> Result<(), Self::Error> {
        F::complete_text_output(output, roots, queue)
    }
}

impl<B, S, W, F> ParameterStorageBinding<B> for TextRouteWeights<B, S, W, F>
where
    B: NeuralBackend,
    W: ParameterStorageBinding<B>,
{
    type UnitOperations = W::UnitOperations;
    type ImmutableParameters = W::ImmutableParameters;
    type ResidencyMechanisms = W::ResidencyMechanisms;
    type Parameters = W::Parameters;

    fn unit_quantizer(&self) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure> {
        self.storage.unit_quantizer()
    }

    fn unit_host_capacity(
        &self,
        bindings: &[eredu_core::residency::parameter_binding::WeightBinding],
    ) -> Result<u64, BackendFailure> {
        self.storage.unit_host_capacity(bindings)
    }

    fn unit_residency_context(
        &self,
        distinct_transfer_stream: bool,
    ) -> Result<
        <Self::ResidencyMechanisms as eredu_core::residency::ResidencyStorageMechanisms>::Context,
        BackendFailure,
    > {
        self.storage
            .unit_residency_context(distinct_transfer_stream)
    }

    fn retain_parameter(
        value: &<<<Self::ResidencyMechanisms as eredu_core::residency::ResidencyStorageMechanisms>::Storage as eredu_core::residency::ResidencyStorage>::Copy as eredu_core::residency::ResidencyLeaseStorage>::DeviceValue,
    ) -> B::Tensor {
        W::retain_parameter(value)
    }

    fn unit_operations(&self) -> Self::UnitOperations {
        self.storage.unit_operations()
    }
}
