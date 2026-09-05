//! MLX mechanisms and generic binding for replicated text composition.

use std::{
    collections::BTreeMap,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use eredu_checkpoint::{store::CheckpointSource, LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology, StateResidencyClass,
};
use eredu_nn::{NeuralBackend, Parameterized, PoolingAttentionCache};
use eredu_runtime::{
    BackendMechanismCapabilities, CacheResidencyPolicy, CacheResidencyReport,
    DenseDiskStreamReport, GroupedOperationRequirement, LayerRuntimeState, LayeredArchitecture,
    ReplicatedTextArchitecture, ReplicatedTextMaterializationTask, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ReplicatedTextSession, ReplicatedTextSessionMechanisms,
    ResidencyReport, RuntimeState, SelectedReplicatedTextRealization, SelectedStateRealization,
    StateComponentMechanism, StateComponentPlacement, StateMechanismCapabilities,
    TransactionalPromptCacheMechanisms, WeightBinding, WeightLoweringCapability,
    WeightLoweringDescriptor, WeightLoweringKind, WeightResidencyMechanism,
};

type MlxDirectPartitionExecutor<A> =
    eredu_architectures::partitioned_execution::DirectPartitionExecutor<
        A,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<A, MlxHybridState>,
    >;

type MlxDirectPartitionStrategy<A> = eredu_runtime::PartitionedTextExecution<
    MlxDirectPartitionExecutor<A>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::NoBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

type MlxSharedAddressableBank =
    crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank;
type MlxEmbeddedPredictionObservers =
    eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
        MlxTensor,
        Array,
        Exception,
    >;
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Dtype,
    Stream,
};

#[derive(Clone, Copy, Default)]
struct MlxPartitionTensorAllocator;

impl eredu_architectures::partitioned_execution::PartitionTensorAllocator<MlxNeuralBackend>
    for MlxPartitionTensorAllocator
{
    fn tensor_to_wire(
        &mut self,
        tensor: MlxTensor,
        logical_dtype: eredu_runtime::BoundaryTensorDtype,
        activation_dtype: eredu_runtime::PipelineActivationDtype,
        context: &Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        let dtype = mlx_boundary_dtype(logical_dtype, activation_dtype)?;
        let source = tensor.as_array().dtype();
        let valid = match logical_dtype {
            eredu_runtime::BoundaryTensorDtype::Activation => {
                matches!(source, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
            }
            eredu_runtime::BoundaryTensorDtype::Uint32 => source == Dtype::Uint32,
            eredu_runtime::BoundaryTensorDtype::Int32 => source == Dtype::Int32,
            _ => false,
        };
        if !valid {
            return Err(eredu_nn::Error::backend(
                "MLX pipeline source tensor does not match its logical boundary dtype",
            ));
        }
        tensor
            .as_array()
            .as_dtype(dtype, context)
            .map(MlxTensor::from_array)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        logical_dtype: eredu_runtime::BoundaryTensorDtype,
        activation_dtype: eredu_runtime::PipelineActivationDtype,
        context: &Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        let dtype = mlx_boundary_dtype(logical_dtype, activation_dtype)?;
        safemlx::ops::zeros_dtype(shape, dtype, context)
            .map(MlxTensor::from_array)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

fn mlx_boundary_dtype(
    logical: eredu_runtime::BoundaryTensorDtype,
    activation: eredu_runtime::PipelineActivationDtype,
) -> Result<Dtype, eredu_nn::Error> {
    match logical {
        eredu_runtime::BoundaryTensorDtype::Activation => mlx_pipeline_activation_dtype(activation),
        eredu_runtime::BoundaryTensorDtype::Uint32 => Ok(Dtype::Uint32),
        eredu_runtime::BoundaryTensorDtype::Int32 => Ok(Dtype::Int32),
        _ => Err(eredu_nn::Error::backend(
            "MLX pipeline boundary uses an unsupported logical dtype",
        )),
    }
}

fn mlx_pipeline_activation_dtype(
    dtype: eredu_runtime::PipelineActivationDtype,
) -> Result<Dtype, eredu_nn::Error> {
    match dtype {
        eredu_runtime::PipelineActivationDtype::Float16 => Ok(Dtype::Float16),
        eredu_runtime::PipelineActivationDtype::Bfloat16 => Ok(Dtype::Bfloat16),
        eredu_runtime::PipelineActivationDtype::Float32 => Ok(Dtype::Float32),
        _ => Err(eredu_nn::Error::backend(
            "unsupported MLX pipeline wire dtype",
        )),
    }
}

type MlxPipelinePartitionExecutor<A, S> =
    eredu_architectures::partitioned_execution::PipelinePartitionExecutor<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxPartitionTensorAllocator,
    >;

type MlxPipelinePartitionStrategy<A, S> = eredu_runtime::PartitionedTextExecution<
    MlxPipelinePartitionExecutor<A, S>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::OpaqueBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

#[cfg(test)]
use eredu_runtime::PagedCacheOptions;

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        nn::tensor::{active_token_validation_arrays, validate_active_token_validations},
        runtime::{
            cache::{
                residency::{
                    load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
                },
                state::{
                    MlxHybridState, MlxKeyValueState, MlxPoolingAttentionCache,
                    MlxPoolingAttentionState, MlxPoolingAttentionStateFactory,
                },
            },
            execution::generic::{
                MlxLayerwisePolicy, MlxResidentPolicy, MlxResidentUnit, MlxSelectiveUnitPopulator,
                MlxUnitLease, MlxUnitPopulator,
            },
            media::input,
        },
    },
    native_quantization::NativeQuantizationFormat,
    MlxTensor,
};

use crate::backend::runtime::execution::{
    generic::prepare_layerwise_policy_from_bindings,
    layerwise::{
        quantize_exact_replicated_text_tasks, shard_addressable_member_bindings,
        shard_layer_bindings,
    },
};
use crate::backend::{
    nn::shared::neutral_parameter_refs,
    runtime::checkpoint::binding::build_exact_replicated_text_bindings,
};

use eredu_architectures::composite_execution::{
    CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_architectures::replicated_text::{
    CompositeTextArchitectureVisitor, PreparedCompositeTextArchitecture,
    PreparedReplicatedTextArchitecture, PreparedRoutedCompositeTextArchitecture,
    ReplicatedTextArchitectureVisitor, ReplicatedTextProfileDispatcher,
};

enum MlxSelectedLayerwisePolicyInner<U, P> {
    Resident(MlxResidentPolicy<U>),
    Bounded {
        policy: MlxLayerwisePolicy<U, P>,
        local_addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    },
}

struct MlxSelectedLayerwisePolicy<U, P> {
    inner: Arc<Mutex<MlxSelectedLayerwisePolicyInner<U, P>>>,
}

type MlxArchitectureLayerwisePolicy<A, S> = MlxSelectedLayerwisePolicy<
    <A as LayeredArchitecture<MlxNeuralBackend, S>>::Unit,
    MlxSelectiveUnitPopulator,
>;

impl<U, P> Clone for MlxSelectedLayerwisePolicy<U, P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<U, P> MlxSelectedLayerwisePolicy<U, P> {
    fn resident(policy: MlxResidentPolicy<U>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Resident(
                policy,
            ))),
        }
    }

    fn bounded(
        policy: MlxLayerwisePolicy<U, P>,
        layout: &eredu_runtime::ExecutionUnitLayout,
    ) -> Self {
        let local_addresses = (0..layout.len())
            .map(|ordinal| {
                layout
                    .address(ordinal)
                    .expect("validated execution layout covers every ordinal")
            })
            .collect();
        Self {
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Bounded {
                policy,
                local_addresses,
            })),
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.residency_report(),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.residency_report(),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(_) => Ok(None),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.dense_stream_report(),
        }
    }
}

enum MlxSelectedUnitLease<U> {
    Resident(MlxResidentUnit<U>),
    Bounded(MlxUnitLease<U>),
}

impl<U> Deref for MlxSelectedUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U> DerefMut for MlxSelectedUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U, P> eredu_runtime::LayerwisePolicy<MlxNeuralBackend, U> for MlxSelectedLayerwisePolicy<U, P>
where
    U: eredu_nn::Parameterized<MlxTensor>,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxSelectedUnitLease<U>;
    type Error = Error;

    fn begin(&mut self, initial: &MlxTensor, context: &Stream) -> Result<(), Self::Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.begin(initial, context),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.begin(initial, context)
            }
        }
    }

    fn abort(
        &mut self,
        active: Option<(usize, eredu_runtime::ExecutionUnitAddress, Self::Lease)>,
        context: &Stream,
    ) {
        let Ok(mut policy) = self.inner.lock() else {
            drop(active);
            return;
        };
        match (&mut *policy, active) {
            (
                MlxSelectedLayerwisePolicyInner::Resident(policy),
                Some((ordinal, address, MlxSelectedUnitLease::Resident(lease))),
            ) => policy.abort(Some((ordinal, address, lease)), context),
            (
                MlxSelectedLayerwisePolicyInner::Bounded {
                    policy,
                    local_addresses,
                },
                Some((ordinal, address, MlxSelectedUnitLease::Bounded(lease))),
            ) => policy.abort(
                Some((
                    ordinal,
                    local_addresses.get(ordinal).copied().unwrap_or(address),
                    lease,
                )),
                context,
            ),
            (MlxSelectedLayerwisePolicyInner::Resident(policy), None) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Bounded { policy, .. }, None) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Resident(policy), Some(_)) => {
                policy.abort(None, context)
            }
            (MlxSelectedLayerwisePolicyInner::Bounded { policy, .. }, Some(_)) => {
                policy.abort(None, context)
            }
        }
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        build: F,
        context: &Stream,
    ) -> Result<Self::Lease, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
    {
        let mut policy = self.inner.lock().map_err(|_| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "selected layerwise policy lock was poisoned".into(),
            ))
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy
                .acquire(ordinal, address, build, context)
                .map(MlxSelectedUnitLease::Resident),
            MlxSelectedLayerwisePolicyInner::Bounded {
                policy,
                local_addresses,
            } => {
                let local = local_addresses.get(ordinal).copied().ok_or_else(|| {
                    eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(format!(
                        "bounded partition unit ordinal {ordinal} has no local slot"
                    )))
                })?;
                if local.group() != address.group() {
                    return Err(eredu_runtime::LayerwiseAcquireError::Policy(
                        Error::ArchitectureModel(format!(
                            "bounded partition unit group {} differs from local group {}",
                            address.group(),
                            local.group()
                        )),
                    ));
                }
                policy.acquire(ordinal, local, build, context).map(|lease| {
                    #[cfg(test)]
                    super::path_instrumentation::bounded_unit_acquisition();
                    MlxSelectedUnitLease::Bounded(lease)
                })
            }
        }
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a MlxTensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &Stream,
    ) -> Result<(), Self::Error>
    where
        MlxTensor: 'a,
        StateValues: Iterator<Item = &'a MlxTensor>,
        ContextValues: Iterator<Item = &'a MlxTensor>,
    {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match (&mut *policy, lease) {
            (
                MlxSelectedLayerwisePolicyInner::Resident(policy),
                MlxSelectedUnitLease::Resident(lease),
            ) => policy.complete(
                ordinal,
                address,
                lease,
                output,
                state_values,
                context_values,
                context,
            ),
            (
                MlxSelectedLayerwisePolicyInner::Bounded {
                    policy,
                    local_addresses,
                },
                MlxSelectedUnitLease::Bounded(lease),
            ) => {
                let local = local_addresses.get(ordinal).copied().ok_or_else(|| {
                    Error::ArchitectureModel(format!(
                        "bounded partition unit ordinal {ordinal} has no local slot"
                    ))
                })?;
                if local.group() != address.group() {
                    return Err(Error::ArchitectureModel(format!(
                        "bounded partition unit group {} differs from local group {}",
                        address.group(),
                        local.group()
                    )));
                }
                policy.complete(
                    ordinal,
                    local,
                    lease,
                    output,
                    state_values,
                    context_values,
                    context,
                )
            }
            _ => Err(Error::ArchitectureModel(
                "selected layerwise policy received a lease from another residency".into(),
            )),
        }
    }

    fn finish(&mut self, output: &MlxTensor, context: &Stream) -> Result<(), Self::Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.finish(output, context),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.finish(output, context)
            }
        }
    }
}

/// Reports the exact MLX mechanisms applicable to one neutral requirement set.
///
/// The report is derived only from source encodings, executable formats, and
/// implemented backend facilities. It does not receive architecture identity.
pub(crate) const GROUPED_OPERATION_CAPABILITIES: [GroupedOperationRequirement; 4] = [
    GroupedOperationRequirement::GatedProduct,
    GroupedOperationRequirement::GatedProductTensorParallelPartial,
    GroupedOperationRequirement::Relu2,
    GroupedOperationRequirement::Relu2TensorParallelPartial,
];

pub(crate) fn capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
) -> BackendMechanismCapabilities {
    let mut weight_lowerings = Vec::new();
    for parameter in requirements
        .parameters()
        .iter()
        .chain(requirements.auxiliary_parameters())
    {
        if !parameter.has_lowering_source() {
            continue;
        }
        let requested = request
            .quantization()
            .and_then(|requested| parameter.transform_target(requested).ok().flatten())
            .map(|target| target.executable());
        for executable in std::iter::once(parameter.native_executable()).chain(requested) {
            let descriptor = parameter
                .lowering_descriptor(executable)
                .expect("validated replicated parameter forms a lowering query");
            let direct_is_semantically_valid = supports_direct(&descriptor)
                && !(parameter.role() == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    && matches!(
                        descriptor.source(),
                        SourceTensorEncoding::Safetensors(StoredDtype::U8)
                    )
                    && executable == LinearFormat::Dense);
            let kind =
                if executable == parameter.native_executable() && direct_is_semantically_valid {
                    Some(WeightLoweringKind::Direct)
                } else if supports_transform(&descriptor) {
                    Some(WeightLoweringKind::Transform)
                } else {
                    None
                };
            if let Some(kind) = kind {
                let capability = WeightLoweringCapability::new(descriptor, kind);
                if !weight_lowerings.contains(&capability) {
                    weight_lowerings.push(capability);
                }
            }
        }
    }
    let state =
        StateMechanismCapabilities::new((0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .filter_map(move |component| {
                    let paged = match component.residency() {
                        StateResidencyClass::SealablePaged => StateComponentPlacement::Paged,
                        StateResidencyClass::AlwaysDeviceMutable
                        | StateResidencyClass::LayerScopedOffloadable => {
                            StateComponentPlacement::Device
                        }
                    };
                    mlx_supports_state_component(component).then(|| {
                        StateComponentMechanism::new(
                            layer,
                            component.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(paged),
                        )
                    })
                })
        }))
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(matches!(request.state(), CacheResidencyPolicy::Paged(_)))
        .with_observation_retention(true);
    BackendMechanismCapabilities::new(
        MlxNeuralBackend::OPERATOR_CAPABILITIES,
        weight_lowerings,
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ],
        state,
    )
    .with_session(eredu_core::SessionCapabilities::new(true, true, true))
    .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
    .with_indexed_movement(true)
    .with_addressable_storage(
        eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX).with_tiers(
            eredu_runtime::AddressableStorageTiers::new(true, true, true),
        ),
    )
    .with_prompt_cache(true)
    .with_exact_completion(true)
}

fn mlx_supports_state_component(component: &eredu_core::cache::StateComponentPolicy) -> bool {
    use eredu_core::cache::{StateTensorDimension, StateTensorDtype};

    !component.shape().is_empty()
        && component.shape().iter().all(|dimension| match dimension {
            StateTensorDimension::Fixed(value)
            | StateTensorDimension::PrefixTokensDiv(value)
            | StateTensorDimension::PrefixTokensRem(value) => value.get() > 0,
            StateTensorDimension::Batch | StateTensorDimension::PrefixTokens => true,
            StateTensorDimension::Scalar => component.shape().len() == 1,
        })
        && matches!(
            component.dtype(),
            StateTensorDtype::Floating
                | StateTensorDtype::Float32
                | StateTensorDtype::Int32
                | StateTensorDtype::Uint32
        )
}

#[cfg(test)]
type StatePresenceSnapshot = Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;

#[cfg(test)]
type FixedNumericStateSnapshot = Vec<(
    usize,
    eredu_core::cache::StateTensorRole,
    Vec<i32>,
    Vec<f32>,
)>;

#[cfg(test)]
type RetainedNumericStateSnapshot = Vec<(Vec<i32>, Vec<f32>)>;

#[cfg(test)]
type CheckpointRestoreProbe = (
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    Vec<f32>,
);

trait MlxParameterBankTelemetry {
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    >;
}

impl MlxParameterBankTelemetry for eredu_runtime::DirectReplicatedTextExecution {
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<
        eredu_architectures::PlannedResidentGatedProduct,
    >
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<eredu_architectures::PlannedResidentRelu2>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl<E, G, R, I, T, U, V> MlxParameterBankTelemetry
    for eredu_runtime::PartitionedTextExecution<E, G, R, I, T, U, V>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

type MlxAddressableGated = eredu_architectures::PlannedAddressableGatedProduct<
    MlxNeuralBackend,
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank,
    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
>;
type MlxAddressableRelu2 = eredu_architectures::PlannedAddressableRelu2<
    MlxNeuralBackend,
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank,
    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
>;

macro_rules! addressable_bank_telemetry {
    ($provider:ty) => {
        impl MlxParameterBankTelemetry
            for eredu_runtime::RoutedReplicatedTextExecution<$provider>
        {
            fn parameter_bank_report(
                &self,
            ) -> Result<
                Option<
                    crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport,
                >,
                Error,
            > {
                self.provider()
                    .bank_report()
                    .map(Some)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            }
        }
    };
}

addressable_bank_telemetry!(MlxAddressableGated);
addressable_bank_telemetry!(MlxAddressableRelu2);

trait ErasedPredictionTargetState: std::any::Any {
    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception>;
    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
    fn offset(&self) -> i32;
}

impl<S> ErasedPredictionTargetState for S
where
    S: MlxStateMechanisms + 'static,
{
    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception> {
        self.deep_checkpoint()
            .map(|state| Box::new(state) as Box<dyn ErasedPredictionTargetState>)
    }

    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let checkpoint = checkpoint
            .as_any()
            .downcast_ref::<S>()
            .ok_or_else(|| Exception::custom("prediction target checkpoint state type changed"))?;
        self.restore_checkpoint(checkpoint, stream)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn offset(&self) -> i32 {
        MlxStateMechanisms::offset(self)
    }
}

/// Opaque MLX storage for one ordinary target lane.
///
/// Neutral prediction membership and transaction metadata live in
/// `EmbeddedPredictionCache`; this wrapper supplies only native clone,
/// restore, type transfer, and frontier inspection mechanisms.
pub(crate) struct MlxPredictionTargetState(Option<Box<dyn ErasedPredictionTargetState>>);

impl MlxPredictionTargetState {
    pub(crate) fn new<S: MlxStateMechanisms + 'static>(state: S) -> Self {
        Self(Some(Box::new(state)))
    }

    fn is<S: 'static>(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|state| state.as_ref().as_any().is::<S>())
    }

    fn take_state<S: 'static>(&mut self) -> Result<S, Error> {
        Ok(*self
            .0
            .take()
            .ok_or_else(|| {
                Error::ArchitectureModel("prediction target state is already active".into())
            })?
            .into_any()
            .downcast::<S>()
            .expect("prediction target state type checked before transfer"))
    }

    fn restore_state<S: MlxStateMechanisms + 'static>(&mut self, state: S) {
        self.0 = Some(Box::new(state));
    }

    pub(crate) fn deep_clone(&self) -> Result<Self, Exception> {
        self.0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?
            .deep_clone_box()
            .map(|state| Self(Some(state)))
    }

    pub(crate) fn restore(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        let current = self
            .0
            .as_mut()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?;
        let checkpoint = checkpoint
            .0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target checkpoint is active"))?;
        current.restore_box(checkpoint.as_ref(), stream)
    }

    pub(crate) fn generation(&self) -> Result<u64, Error> {
        let state = self.0.as_ref().ok_or_else(|| {
            Error::ArchitectureModel("prediction target state is already active".into())
        })?;
        u64::try_from(state.offset())
            .map_err(|_| Error::ArchitectureModel("target capture generation is negative".into()))
    }
}

struct ExactPredictionCaptureObserver {
    paths: Vec<String>,
    values: std::rc::Rc<std::cell::RefCell<Vec<Option<MlxTensor>>>>,
}

impl ExactPredictionCaptureObserver {
    fn new(paths: Vec<String>) -> Result<Self, Error> {
        if paths.is_empty() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture declares no target paths".into(),
            ));
        }
        let unique = paths.iter().collect::<std::collections::BTreeSet<_>>();
        if unique.len() != paths.len() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture paths are not unique".into(),
            ));
        }
        let values = std::rc::Rc::new(std::cell::RefCell::new(vec![None; paths.len()]));
        Ok(Self { paths, values })
    }
}

impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>
    for ExactPredictionCaptureObserver
{
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        if let Some(index) = self.paths.iter().position(|expected| expected == path) {
            if self.values.borrow_mut()[index]
                .replace(value.clone())
                .is_some()
            {
                return Err(eredu_nn::Error::backend(format!(
                    "external-assistant target reached capture path {path} more than once"
                )));
            }
        }
        Ok(())
    }
}

struct CompositePredictionTargetOperation<'a> {
    operation: ExternalPredictionTargetOperation<'a, MlxTensor>,
}

impl<A>
    eredu_runtime::PredictionTargetOperation<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
    > for CompositePredictionTargetOperation<'_>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = MlxTensor;

    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _state: &mut MlxHybridState,
        parallel: Option<&<MlxNeuralBackend as NeuralBackend>::ParallelContext>,
        context: &Stream,
    ) -> Result<Self::Output, eredu_nn::Error> {
        if parallel.is_some() {
            return Err(eredu_nn::Error::backend(
                "external assistant target operations are unavailable under tensor parallelism",
            ));
        }
        architecture
            .inner_mut()
            .external_prediction_target_operation(self.operation, context)?
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "architecture does not implement the selected external target operation",
                )
            })
    }
}

pub(crate) struct MlxEmbeddedPredictionMaterializer;

pub(crate) type MaterializedEmbeddedPrediction =
    eredu_architectures::prediction_extension::MaterializedPredictionExtension<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    >;

fn materialize_prepared_prediction_unit<M>(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    store: &dyn CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<crate::backend::nn::shared::MlxModule<M>, Error>
where
    M: Parameterized<MlxTensor>,
{
    use crate::backend::runtime::checkpoint::binding::{
        build_exact_replicated_text_bindings, materialize_module_bindings,
        populate_module_from_arrays_excluding,
    };

    let (source, mut local, selected_tasks) = prepared.into_parts();
    let task_refs = selected_tasks.iter().collect::<Vec<_>>();
    let bindings = build_exact_replicated_text_bindings(
        &source,
        store,
        &task_refs,
        &std::collections::BTreeSet::new(),
    )?;
    let bindings = match layout {
        Some(layout) => shard_unmaterialized_bindings(
            bindings,
            store,
            layout,
            &std::collections::BTreeSet::new(),
        )?,
        None => bindings,
    };
    let arrays = materialize_module_bindings(store, &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut local, &arrays, |_| false)?;
    Ok(crate::backend::nn::shared::MlxModule::new(local))
}

pub(crate) fn materialize_prediction_extension(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionExtension<
        MlxNeuralBackend,
    >,
    store: &dyn CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedEmbeddedPrediction, Error> {
    let mut context = MlxPredictionMaterializationContext {
        store,
        stream,
        weights_stream,
    };
    prepared.materialize::<MlxEmbeddedPredictionMaterializer>(&mut context)
}

pub(crate) struct MlxPredictionMaterializationContext<'a> {
    store: &'a dyn CheckpointSource,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::prediction_extension::PredictionExtensionMaterializer<MlxNeuralBackend>
    for MlxEmbeddedPredictionMaterializer
{
    type Error = Error;
    type Module<M> = crate::backend::nn::shared::MlxModule<M>;
    type PoolingState = crate::backend::runtime::cache::state::MlxPoolingAttentionCache;
    type SequentialState = crate::backend::runtime::cache::kv::CompressedLatentCache;
    type ModelState = MlxHybridState;
    type Context<'a> = MlxPredictionMaterializationContext<'a>;

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<MlxTensor>,
    {
        materialize_prepared_prediction_unit(
            prepared,
            layout,
            context.store,
            context.stream,
            context.weights_stream,
        )
    }

    fn pooling_state(
        _context: &mut Self::Context<'_>,
        ordinal: usize,
        policy: eredu_core::cache::LayerCachePolicy,
    ) -> Result<Self::PoolingState, Self::Error> {
        Ok(Self::PoolingState::resident_from_policy(ordinal, &policy)?)
    }

    fn model_state(
        _context: &mut Self::Context<'_>,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self::ModelState, Self::Error> {
        Ok(MlxHybridState::device(layout)?)
    }

    fn sequential_state() -> Self::SequentialState {
        Self::SequentialState::new()
    }
}

impl eredu_architectures::prediction_extension::PredictionModelState<MlxNeuralBackend>
    for MlxHybridState
{
    type LayerState = crate::backend::runtime::cache::state::MlxHybridLayerState;

    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.layers_mut()
    }
}

struct SelectedPrediction<P> {
    extension: P,
    selected: eredu_runtime::SelectedSpeculativeRealization,
}

struct NoSelectedPrediction;

trait ReplicatedPredictionCapability<A, S, D>: Sized
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>>;

    fn present() -> bool;
}

impl<A, S, D> ReplicatedPredictionCapability<A, S, D> for NoSelectedPrediction
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    fn lend(
        _: &mut CompletedReplicatedText<A, S, D, Self>,
        _: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }

    fn present() -> bool {
        false
    }
}

pub(crate) trait ErasedExternalPredictionExecutable: 'static {
    fn prepare_external_prediction_target_cache(
        &mut self,
    ) -> Result<MlxPredictionTargetState, Error>;
    fn prefill_external_prediction_target(
        &mut self,
        input: input::ModelInput<'_>,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn verify_external_prediction_target(
        &mut self,
        tokens: &MlxTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, MlxTensor>,
    ) -> Result<MlxTensor, Error>;
}

/// Backend-private erased operations for a paired architecture and mutable state.
pub(crate) trait ErasedReplicatedTextExecutable {
    fn effective_model_type(&self) -> &str;
    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate;
    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency;
    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot;
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception>;
    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error>;
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport>;
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    >;
    fn has_partition_control(&self) -> bool {
        false
    }
    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        None
    }
    fn partition_public_output(&self) -> bool {
        true
    }
    fn with_embedded_prediction(
        &mut self,
        _continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }
    fn has_embedded_prediction(&self) -> bool {
        false
    }
    fn install_embedded_prediction_observers(
        &mut self,
        _observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        false
    }
    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut (dyn ErasedExternalPredictionExecutable + 'static)> {
        None
    }
    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity;
    fn reset_cache(&mut self) -> Result<(), Exception>;
    fn reset_cache_distributed(&mut self) -> Result<(), Error>;
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        let _ = input_identity;
        self.load_prompt_cache(directory, expected, prefix_token_ids)
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error>;
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error>;
    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
    fn decode_with_observer(
        &mut self,
        tokens: &Array,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
}

pub(crate) fn prepared_composite_input(
    input: input::ModelInput<'_>,
) -> Result<eredu_runtime::PreparedModelInput<MlxTensor>, Error> {
    use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedInputPayload};

    input::validate(input)?;
    let parts = input
        .parts
        .iter()
        .map(|part| {
            let payload = match part.payload() {
                input::InputPayload::TokenIds(value) => {
                    PreparedInputPayload::TokenIds(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Tensor(value) => {
                    PreparedInputPayload::Tensor(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Embeddings(value) => {
                    PreparedInputPayload::Embeddings(MlxTensor::from_array(value.clone()))
                }
                _ => {
                    return Err(eredu_core::PreparedInputError::BackendTensorIdentity(
                        "MLX prepared input contains an unknown payload kind".into(),
                    ))
                }
            };
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                part.metadata()
                    .iter()
                    .map(|(key, value)| (*key, MlxTensor::from_array(value.clone()))),
                part.extents().iter().copied(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let inspector = input::MlxTensorInputInspector;
    eredu_runtime::PreparedModelInput::new(parts, |tensor| inspector.identity(tensor))
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(crate) trait MlxStateMechanisms: LayerRuntimeState<MlxNeuralBackend> + Sized {
    fn offset(&self) -> i32;
    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error>;
    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn retained_arrays(&self) -> Vec<&Array>;
    fn deep_checkpoint(&self) -> Result<Self, Exception>;
    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception>;
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception>;
    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;
    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    >;
    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception>;
}

fn fork_mlx_prediction_target_state<S: MlxStateMechanisms>(
    state: &S,
    stream: &Stream,
) -> Result<S, Error> {
    state
        .fork_prediction_target_state(stream)
        .map_err(Into::into)
}

fn selected_state_manager(
    selected: &SelectedStateRealization,
) -> Result<Option<CacheResidencyManager>, Error> {
    let needs_paging = selected
        .components()
        .iter()
        .any(|component| component.placement() == StateComponentPlacement::Paged);
    match (needs_paging, selected.policy()) {
        (_, CacheResidencyPolicy::Paged(options)) => CacheResidencyManager::new(options.clone())
            .map(Some)
            .map_err(|error| Error::Parallel(error.to_string())),
        (false, CacheResidencyPolicy::Device) => Ok(None),
        (true, CacheResidencyPolicy::Device) => Err(Error::Parallel(
            "selected paged state component has no paging policy".into(),
        )),
    }
}

impl MlxStateMechanisms for MlxKeyValueState {
    fn offset(&self) -> i32 {
        MlxKeyValueState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        super::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxKeyValueState::save_prompt_cache(
            self,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxKeyValueState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxKeyValueState::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxKeyValueState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxKeyValueState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .map(|layer| (eredu_nn::AttentionCache::offset(layer), Vec::new()))
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        Ok(Vec::new())
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        Ok(Vec::new())
    }
}

impl MlxStateMechanisms for MlxHybridState {
    fn offset(&self) -> i32 {
        MlxHybridState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        super::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxHybridState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        state.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxHybridState::save_prompt_cache(self, destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxHybridState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxHybridState::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxHybridState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxHybridState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.semantic_snapshot()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        self.fixed_numeric_snapshot()
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_numeric_snapshot()
    }
}

impl MlxStateMechanisms for MlxPoolingAttentionState {
    fn offset(&self) -> i32 {
        self.as_ref().first().map_or(0, |layer| layer.offset())
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        super::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        match manager {
            Some(manager) => MlxPoolingAttentionStateFactory::paged(
                selected.layout().clone(),
                manager,
                global_layer_start,
                0,
                rank,
            ),
            None => MlxPoolingAttentionStateFactory::device(selected.layout().clone()),
        }
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let prefix = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?;
        let mut state = MlxPoolingAttentionStateFactory::paged(
            selected.layout().clone(),
            manager,
            identity.global_layer_start(),
            prefix,
            expected.topology().cache_rank_identity(),
        )?;
        let mut tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        for (layer, cache) in state.as_mut().iter_mut().enumerate() {
            let processed = prefix
                .checked_add(identity.layer_prefix_offsets()[layer])
                .ok_or_else(|| Error::Parallel("prompt-cache layer offset overflowed".into()))?;
            cache.restore_prompt_cache_state(
                identity.global_layer_start() + layer,
                &mut tensors,
                processed,
            )?;
        }
        if !tensors.is_empty() {
            return Err(Error::Parallel(
                "prompt cache contains unexpected state tensors".into(),
            ));
        }
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let mut manager = None;
        for layer in self.as_mut() {
            layer.finalize()?;
            manager.get_or_insert_with(|| layer.residency_manager().cloned());
        }
        let fixed = self
            .as_ref()
            .iter()
            .enumerate()
            .flat_map(|(layer, cache)| cache.prompt_cache_state_arrays(layer))
            .collect::<Vec<_>>();
        manager
            .flatten()
            .ok_or_else(|| Error::Parallel("prompt-cache persistence requires paged state".into()))?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &fixed, options)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(CacheResidencyManager::report)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.as_ref()
            .iter()
            .flat_map(MlxPoolingAttentionCache::retained_arrays)
            .collect()
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        eredu_runtime::DeviceState::create(self.layout().clone(), |layer, _| {
            self.as_ref()[layer].deep_clone_state()
        })
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxPoolingAttentionStateFactory::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        if self.layout() != checkpoint.layout() || self.as_ref().len() != checkpoint.as_ref().len()
        {
            return Err(Exception::custom(
                "pooling-attention checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.as_mut().iter_mut().zip(checkpoint.as_ref()) {
            PoolingAttentionCache::restore(current, previous, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let present = layer
                    .prompt_cache_state_arrays(index)
                    .into_iter()
                    .map(|state| state.role)
                    .collect::<std::collections::BTreeSet<_>>();
                let components = self
                    .layout()
                    .components(index)
                    .expect("pooling state layout contains each realized layer")
                    .iter()
                    .filter_map(|component| match component.role() {
                        eredu_core::cache::StateComponentRole::Fixed(role) => {
                            Some((role, present.contains(&role)))
                        }
                        _ => None,
                    })
                    .collect();
                (PoolingAttentionCache::offset(layer), components)
            })
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        let mut snapshot = Vec::new();
        for (layer, cache) in self.as_ref().iter().enumerate() {
            for state in cache.prompt_cache_state_arrays(layer) {
                let evaluated = state.array.evaluated()?;
                snapshot.push((
                    layer,
                    state.role,
                    state.array.shape().to_vec(),
                    evaluated.as_slice::<f32>().to_vec(),
                ));
            }
        }
        Ok(snapshot)
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_arrays()
            .into_iter()
            .map(|array| {
                let evaluated = array.evaluated()?;
                Ok((array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec()))
            })
            .collect()
    }
}
struct MlxExecutionReport {
    residency: ResidencyReport,
    dense: Option<DenseDiskStreamReport>,
}

struct MlxStateReport {
    residency: Option<CacheResidencyReport>,
    #[cfg(test)]
    presence: StatePresenceSnapshot,
    #[cfg(test)]
    fixed_numeric: FixedNumericStateSnapshot,
    #[cfg(test)]
    retained_numeric: RetainedNumericStateSnapshot,
}

struct MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    store: Arc<dyn CheckpointSource>,
    prepared_bindings: Option<PreparedExactBindings>,
    resident_report: Option<ResidencyReport>,
    materialization: Arc<std::sync::Mutex<Option<eredu_runtime::WeightMaterializationReport>>>,
    stream: Stream,
    weights_stream: Stream,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    source_parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    ignored_checkpoint_sources: std::collections::BTreeSet<String>,
    state_rank: Option<eredu_core::cache::CacheRankIdentity>,
    state_global_layer_start: usize,
    state: PhantomData<fn() -> (A, S)>,
}

struct PreparedExactBindings {
    layout: eredu_runtime::ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    unit_bindings: Vec<Vec<WeightBinding>>,
    excluded_parameters: std::collections::BTreeSet<String>,
    local_parameters: Option<std::collections::BTreeSet<String>>,
}

fn prompt_cache_storage_directory(root: &Path, topology: &PromptCacheTopology) -> PathBuf {
    let coordinate = |axis: Option<(usize, usize)>| {
        axis.map_or_else(|| "x".to_owned(), |(_, rank)| rank.to_string())
    };
    if topology.cache_rank_identity().is_none() {
        root.to_path_buf()
    } else {
        root.join(format!(
            "rank-p{}-t{}-e{}",
            coordinate(topology.stage()),
            coordinate(topology.shard()),
            coordinate(topology.addressable())
        ))
    }
}

struct MlxPromptCacheSaveTransaction {
    publication: eredu_runtime::ReversiblePromptCachePublication,
    manifest: PromptCacheManifest,
}

type ExactTaskPartitions<'a> = (
    Vec<&'a ReplicatedTextMaterializationTask>,
    Vec<Vec<&'a ReplicatedTextMaterializationTask>>,
);

fn partition_local_materialization_tasks<'a>(
    tasks: &'a [ReplicatedTextMaterializationTask],
    global_layout: &eredu_runtime::ExecutionUnitLayout,
    addresses: &[eredu_runtime::ExecutionUnitAddress],
) -> Result<ExactTaskPartitions<'a>, Error> {
    if addresses.is_empty() {
        return Err(Error::ArchitectureModel(
            "local partition has no selected execution units".into(),
        ));
    }
    let mut static_tasks = Vec::new();
    let mut unit_tasks = vec![Vec::new(); addresses.len()];
    for task in tasks {
        match task.owner() {
            eredu_runtime::ReplicatedTextParameterOwner::StaticRole(_) => static_tasks.push(task),
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                let local = addresses
                    .iter()
                    .position(|address| {
                        global_layout
                            .group_id(address.group())
                            .is_some_and(|id| id.as_str() == group)
                            && address.index() == *unit
                    })
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "local task {:?} has no owned global unit {group}.{unit}",
                            task.name()
                        ))
                    })?;
                unit_tasks[local].push(task);
            }
            _ => {
                return Err(Error::ArchitectureModel(format!(
                    "local task {:?} has an unsupported owner",
                    task.name()
                )))
            }
        }
    }
    let consumed = static_tasks.len() + unit_tasks.iter().map(Vec::len).sum::<usize>();
    if consumed != tasks.len() {
        return Err(Error::ArchitectureModel(
            "local partition tasks were not consumed exactly once".into(),
        ));
    }
    Ok((static_tasks, unit_tasks))
}

fn partition_materialization_tasks<'a>(
    tasks: &'a [ReplicatedTextMaterializationTask],
    layout: &eredu_runtime::ExecutionUnitLayout,
) -> Result<ExactTaskPartitions<'a>, Error> {
    let mut static_tasks = Vec::new();
    let mut unit_tasks = vec![Vec::new(); layout.len()];
    for task in tasks {
        match task.owner() {
            eredu_runtime::ReplicatedTextParameterOwner::StaticRole(_) => {
                static_tasks.push(task);
            }
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                let group_index = (0..layout.group_count())
                    .find(|index| {
                        layout
                            .group_id(*index)
                            .is_some_and(|id| id.as_str() == group)
                    })
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "exact task {:?} names unknown execution group {group:?}",
                            task.name()
                        ))
                    })?;
                let ordinal = layout.ordinal(group_index, *unit).ok_or_else(|| {
                    Error::ArchitectureModel(format!(
                        "exact task {:?} names unknown unit {unit} in group {group:?}",
                        task.name()
                    ))
                })?;
                unit_tasks[ordinal].push(task);
            }
            _ => {
                return Err(Error::ArchitectureModel(format!(
                    "exact task {:?} has an unsupported parameter owner",
                    task.name()
                )))
            }
        }
    }
    let consumed = static_tasks.len() + unit_tasks.iter().map(Vec::len).sum::<usize>();
    if consumed != tasks.len() {
        return Err(Error::ArchitectureModel(
            "exact materialization tasks were not partitioned exactly once".into(),
        ));
    }
    Ok((static_tasks, unit_tasks))
}

fn locally_materialized_outputs(
    tasks: &[ReplicatedTextMaterializationTask],
) -> std::collections::BTreeSet<String> {
    tasks
        .iter()
        .filter(|task| {
            matches!(
                task.lowering(),
                WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
            )
        })
        .flat_map(|task| {
            std::iter::once(task.name().to_owned()).chain(
                task.output_companions()
                    .iter()
                    .map(|companion| companion.name().to_owned()),
            )
        })
        .collect()
}

fn shard_unmaterialized_bindings(
    bindings: Vec<WeightBinding>,
    store: &dyn CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
    locally_materialized: &std::collections::BTreeSet<String>,
) -> Result<Vec<WeightBinding>, Error> {
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if locally_materialized.contains(binding.name()) {
            output.push(binding);
        } else {
            output.extend(shard_layer_bindings(vec![binding], store, layout)?);
        }
    }
    Ok(output)
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    fn new(
        store: Arc<dyn CheckpointSource>,
        materialization: Arc<std::sync::Mutex<Option<eredu_runtime::WeightMaterializationReport>>>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Self {
        Self {
            store,
            prepared_bindings: None,
            resident_report: None,
            materialization,
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            parallel_layout: None,
            source_parallel_layout: None,
            ignored_checkpoint_sources: std::collections::BTreeSet::new(),
            state_rank: None,
            state_global_layer_start: 0,
            state: PhantomData,
        }
    }

    fn set_parallel_layout(&mut self, layout: eredu_runtime::LocalModelLayout) {
        self.parallel_layout = Some(layout);
    }

    fn set_source_parallel_layout(&mut self, layout: Option<eredu_runtime::LocalModelLayout>) {
        self.source_parallel_layout = layout;
    }

    fn set_ignored_checkpoint_sources(&mut self, sources: std::collections::BTreeSet<String>) {
        self.ignored_checkpoint_sources = sources;
    }

    fn set_state_partition(
        &mut self,
        rank: eredu_core::cache::CacheRankIdentity,
        global_layer_start: usize,
    ) {
        self.state_rank = Some(rank);
        self.state_global_layer_start = global_layer_start;
    }

    fn apply_selected_transforms(
        &mut self,
        target_architecture: &A,
        target_units: &[A::Unit],
        source_architecture: Option<&A>,
        source_units: Option<&[A::Unit]>,
        source_layout: Option<&eredu_runtime::LocalModelLayout>,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<(), Error> {
        let transform_tasks = tasks
            .iter()
            .filter(|task| {
                matches!(
                    task.lowering(),
                    WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
                )
            })
            .collect::<Vec<_>>();
        if transform_tasks.is_empty() {
            if source_architecture.is_some() || source_units.is_some() {
                return Err(Error::Quantization(
                    "selected source architecture has no materialization tasks".into(),
                ));
            }
            return Ok(());
        }
        let source = source_architecture.ok_or_else(|| {
            Error::Quantization("selected transform tasks have no source architecture".into())
        })?;
        let source_units = source_units.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected source architecture has no neutral materialization units".into(),
            )
        })?;
        if source_units.len() != target_units.len() {
            return Err(Error::Quantization(
                "selected materialization tasks changed the local execution-unit cardinality"
                    .into(),
            ));
        }
        let mut task_groups = Vec::<(
            eredu_checkpoint::WeightQuantization,
            Vec<&ReplicatedTextMaterializationTask>,
        )>::new();
        for task in transform_tasks {
            let format = task.executable().weight_quantization().ok_or_else(|| {
                Error::Quantization(format!(
                    "selected materialization task {:?} has no packed output format",
                    task.name()
                ))
            })?;
            match task_groups
                .iter_mut()
                .find(|(selected, _)| *selected == format)
            {
                Some((_, tasks)) => tasks.push(task),
                None => task_groups.push((format, vec![task])),
            }
        }
        let source_static = source.static_modules();
        let target_static = target_architecture.static_modules();
        let mut combined = eredu_runtime::WeightMaterializationReport::default();
        for (quantization, exact_tasks) in task_groups {
            #[cfg(test)]
            super::path_instrumentation::materialization();
            let (store, report) = quantize_exact_replicated_text_tasks(
                Arc::clone(&self.store),
                source_static,
                target_static,
                source_units,
                target_units,
                source_layout,
                quantization,
                &exact_tasks,
                &self.stream,
            )?;
            self.store = store;
            combined.admitted_working_set_bytes = combined
                .admitted_working_set_bytes
                .max(report.admitted_working_set_bytes);
            combined.transformed_weights += report.transformed_weights;
            combined.source_tiles += report.source_tiles;
            combined.peak_in_flight_tiles = combined
                .peak_in_flight_tiles
                .max(report.peak_in_flight_tiles);
            combined.source_bytes_read += report.source_bytes_read;
            combined.output_bytes += report.output_bytes;
            combined.peak_planned_working_set_bytes = combined
                .peak_planned_working_set_bytes
                .max(report.peak_planned_working_set_bytes);
            combined.largest_source_tile_bytes = combined
                .largest_source_tile_bytes
                .max(report.largest_source_tile_bytes);
            combined.largest_output_tile_bytes = combined
                .largest_output_tile_bytes
                .max(report.largest_output_tile_bytes);
        }
        *self.materialization.lock().map_err(|_| {
            Error::ArchitectureModel("materialization report lock was poisoned".into())
        })? = Some(combined);
        Ok(())
    }

    fn prepare_local_partition_materialization(
        &mut self,
        architecture: &A,
        source_architecture: Option<&A>,
        global_layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
        units: &[A::Unit],
        source_units: Option<&[A::Unit]>,
        source_layout: Option<&eredu_runtime::LocalModelLayout>,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<(), Error> {
        self.prepare_local_partition_materialization_with_addressable_parameters(
            architecture,
            source_architecture,
            global_layout,
            addresses,
            units,
            source_units,
            source_layout,
            tasks,
            &std::collections::BTreeSet::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_local_partition_materialization_with_addressable_parameters(
        &mut self,
        architecture: &A,
        source_architecture: Option<&A>,
        global_layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
        units: &[A::Unit],
        source_units: Option<&[A::Unit]>,
        source_layout: Option<&eredu_runtime::LocalModelLayout>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &std::collections::BTreeSet<String>,
    ) -> Result<(), Error> {
        if addresses.len() != units.len() || addresses.is_empty() {
            return Err(Error::ArchitectureModel(
                "local partition addresses and constructed units differ".into(),
            ));
        }
        let (static_tasks, unit_tasks) =
            partition_local_materialization_tasks(tasks, global_layout, addresses)?;
        self.apply_selected_transforms(
            architecture,
            units,
            source_architecture,
            source_units,
            source_layout,
            tasks,
        )?;
        let locally_materialized = locally_materialized_outputs(tasks);
        let selected_static_parameters = static_tasks
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name().to_owned()).chain(
                    task.output_companions()
                        .iter()
                        .map(|companion| companion.name().to_owned()),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        // Composite family modules may retain lazy handles for pinned roles
        // owned by another pipeline rank. The architecture-selected tasks are
        // the sole ownership authority: expose precisely those targets to the
        // exact binder and keep every selected target mandatory.
        let mut excluded_parameters = neutral_parameter_refs(architecture.static_modules(), false)
            .flatten()
            .into_keys()
            .map(|name| name.as_ref().to_owned())
            .filter(|name| !selected_static_parameters.contains(name))
            .collect::<std::collections::BTreeSet<_>>();
        excluded_parameters.extend(addressable_parameters.iter().cloned());
        let mut static_bindings = build_exact_replicated_text_bindings(
            architecture.static_modules(),
            self.store.as_ref(),
            &static_tasks,
            &excluded_parameters,
        )?;
        #[cfg(test)]
        super::path_instrumentation::local_static_materialization(
            static_bindings.len(),
            excluded_parameters.len(),
        );
        let mut unit_bindings = units
            .iter()
            .zip(&unit_tasks)
            .map(|(unit, tasks)| {
                #[cfg(test)]
                super::path_instrumentation::unit_construction();
                build_exact_replicated_text_bindings(
                    unit,
                    self.store.as_ref(),
                    tasks,
                    addressable_parameters,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(layout) = &self.parallel_layout {
            static_bindings = shard_unmaterialized_bindings(
                static_bindings,
                self.store.as_ref(),
                layout,
                &locally_materialized,
            )?;
            unit_bindings = unit_bindings
                .into_iter()
                .map(|bindings| {
                    shard_unmaterialized_bindings(
                        bindings,
                        self.store.as_ref(),
                        layout,
                        &locally_materialized,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        let graph = architecture
            .execution_graph()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let counts = (0..graph.groups().len())
            .map(|group| {
                addresses
                    .iter()
                    .filter(|address| address.group() == group)
                    .count()
            })
            .collect::<Vec<_>>();
        let layout = eredu_runtime::ExecutionUnitLayout::new(&graph, counts)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.prepared_bindings = Some(PreparedExactBindings {
            layout,
            static_bindings,
            unit_bindings,
            excluded_parameters,
            local_parameters: Some(
                tasks
                    .iter()
                    .flat_map(|task| {
                        std::iter::once(task.name().to_owned()).chain(
                            task.output_companions()
                                .iter()
                                .map(|companion| companion.name().to_owned()),
                        )
                    })
                    .collect(),
            ),
        });
        Ok(())
    }

    fn take_prepared_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
    ) -> Result<
        (
            MlxLayerwisePolicy<A::Unit, MlxSelectiveUnitPopulator>,
            eredu_runtime::ExecutionUnitLayout,
        ),
        Error,
    > {
        let prepared = self.prepared_bindings.take().ok_or_else(|| {
            Error::ArchitectureModel("execution policy requested before materialization".into())
        })?;
        let layout = prepared.layout.clone();
        let mut ignored_sources = self.ignored_checkpoint_sources.clone();
        for parameter in selected
            .requirements()
            .parameters()
            .iter()
            .filter(|parameter| {
                prepared.excluded_parameters.contains(parameter.name())
                    || prepared
                        .local_parameters
                        .as_ref()
                        .is_some_and(|local| !local.contains(parameter.name()))
            })
        {
            ignored_sources.extend(parameter.sources().iter().cloned());
            if let Some(recipe) = selected
                .requirements()
                .derived_recipes()
                .get(parameter.name())
            {
                ignored_sources.extend(recipe.source_keys().into_iter().map(str::to_owned));
            }
        }
        let (policy, _) = prepare_layerwise_policy_from_bindings(
            Arc::clone(&self.store),
            architecture,
            MlxSelectiveUnitPopulator::new(prepared.excluded_parameters.clone()),
            PhantomData::<S>,
            selected.residency(),
            &self.stream,
            &self.weights_stream,
            move |key| ignored_sources.contains(key),
            prepared.layout,
            prepared.static_bindings,
            prepared.unit_bindings,
        )?;
        Ok((policy, layout))
    }
}

impl<A, S> ReplicatedTextSessionMechanisms<A, MlxNeuralBackend>
    for MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Error: std::fmt::Display,
{
    type State = S;
    type PolicyError = Error;
    type ResidentPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type BoundedPolicy = MlxArchitectureLayerwisePolicy<A, S>;
    type StateCheckpoint = S;
    type StateReport = MlxStateReport;
    type ExecutionReport = MlxExecutionReport;
    type Error = Error;

    fn prepare_materialization(
        &mut self,
        architecture: &mut A,
        target_layout: &eredu_runtime::ExecutionUnitLayout,
        target_units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &[String],
        _context: &Stream,
    ) -> Result<(), Self::Error> {
        if target_units.len() != target_layout.len() {
            return Err(Error::ArchitectureModel(
                "neutral target unit set differs from the selected execution layout".into(),
            ));
        }
        let source_layout = self.source_parallel_layout.clone();
        self.apply_selected_transforms(
            architecture,
            target_units,
            source_architecture.as_deref(),
            source_units.as_deref(),
            source_layout.as_ref(),
            tasks,
        )?;

        let (static_tasks, unit_tasks) = partition_materialization_tasks(tasks, target_layout)?;
        let locally_materialized = locally_materialized_outputs(tasks);
        let addressable_parameters = addressable_parameters
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut static_bindings = build_exact_replicated_text_bindings(
            architecture.static_modules(),
            self.store.as_ref(),
            &static_tasks,
            &addressable_parameters,
        )?;
        if let Some(layout) = &self.parallel_layout {
            static_bindings = shard_unmaterialized_bindings(
                static_bindings,
                self.store.as_ref(),
                layout,
                &locally_materialized,
            )?;
        }
        let mut unit_bindings = target_units
            .iter()
            .zip(&unit_tasks)
            .enumerate()
            .map(|(ordinal, (unit, tasks))| {
                #[cfg(test)]
                super::path_instrumentation::unit_construction();
                build_exact_replicated_text_bindings(
                    unit,
                    self.store.as_ref(),
                    tasks,
                    &addressable_parameters,
                )
                .map_err(|error| {
                    Error::ArchitectureModel(format!(
                        "execution unit {ordinal} exact bindings failed: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(layout) = &self.parallel_layout {
            unit_bindings = unit_bindings
                .into_iter()
                .map(|bindings| {
                    shard_unmaterialized_bindings(
                        bindings,
                        self.store.as_ref(),
                        layout,
                        &locally_materialized,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
        }

        self.prepared_bindings = Some(PreparedExactBindings {
            layout: target_layout.clone(),
            static_bindings,
            unit_bindings,
            excluded_parameters: addressable_parameters,
            local_parameters: None,
        });
        Ok(())
    }

    fn realize_state(
        &mut self,
        selected: &SelectedStateRealization,
        _context: &Stream,
    ) -> Result<S, Error> {
        S::realize(selected, self.state_rank, self.state_global_layer_start)
    }

    fn resident_policy(
        &mut self,
        architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedReplicatedTextRealization,
        context: &Stream,
    ) -> Result<Self::ResidentPolicy, Self::Error> {
        let (policy, _) = self.take_prepared_policy(architecture, selected)?;
        let resident = policy.into_resident_units(units, context)?;
        self.resident_report = Some(resident.residency_report()?);
        Ok(MlxSelectedLayerwisePolicy::resident(resident))
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
        _context: &Stream,
    ) -> Result<Self::BoundedPolicy, Self::Error> {
        self.take_prepared_policy(architecture, selected)
            .map(|(policy, layout)| MlxSelectedLayerwisePolicy::bounded(policy, &layout))
    }

    fn index_text_output(
        &mut self,
        output: MlxTensor,
        sequence_index: i32,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        output
            .as_array()
            .try_index_device((.., sequence_index, ..), context)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn checkpoint_state(&mut self, state: &S, _context: &Stream) -> Result<S, Error> {
        state.deep_checkpoint().map_err(Into::into)
    }

    fn restore_state(
        &mut self,
        state: &mut S,
        checkpoint: S,
        context: &Stream,
    ) -> Result<(), Error> {
        state
            .restore_checkpoint(&checkpoint, context)
            .map_err(Into::into)
    }

    fn fork_prediction_target_state(
        &mut self,
        state: &S,
        _selected: &SelectedStateRealization,
        context: &Stream,
    ) -> Result<S, Error> {
        fork_mlx_prediction_target_state(state, context)
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        selected: &SelectedStateRealization,
        context: &Stream,
    ) -> Result<(S, PromptCacheManifest), Error> {
        let directory = prompt_cache_storage_directory(directory, expected.topology());
        S::load_prompt_cache(
            selected,
            &directory,
            expected,
            identity,
            prefix_token_ids,
            context,
        )
    }

    fn save_prompt_cache(
        &mut self,
        state: &mut S,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _context: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let destination = prompt_cache_storage_directory(destination, descriptor.topology());
        S::save_prompt_cache(state, &destination, descriptor, prefix_token_ids, options)
    }

    fn state_report(&self, state: &S) -> Result<Self::StateReport, Error> {
        Ok(MlxStateReport {
            residency: state.residency_report()?,
            #[cfg(test)]
            presence: S::state_snapshot(state),
            #[cfg(test)]
            fixed_numeric: S::fixed_numeric_snapshot(state)?,
            #[cfg(test)]
            retained_numeric: S::retained_numeric_snapshot(state)?,
        })
    }

    fn execution_report(
        &self,
        _residency: eredu_runtime::LayerWeightResidency,
        bounded: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Error> {
        match bounded {
            Some(policy) => Ok(MlxExecutionReport {
                residency: policy.residency_report()?,
                dense: policy.dense_stream_report()?,
            }),
            None => Ok(MlxExecutionReport {
                residency: self.resident_report.clone().ok_or_else(|| {
                    Error::ArchitectureModel("resident report was not captured".into())
                })?,
                dense: None,
            }),
        }
    }

    fn complete(&mut self, output: &MlxTensor, state: &S, _context: &Stream) -> Result<(), Error> {
        #[cfg(test)]
        super::path_instrumentation::completion();
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            std::iter::once(output.as_array())
                .chain(state.retained_arrays())
                .chain(token_validations.iter()),
        )?
        .synchronize()?;
        validate_active_token_validations().map_err(Into::into)
    }
}

impl<A, S> TransactionalPromptCacheMechanisms<A, MlxNeuralBackend>
    for MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Error: std::fmt::Display,
{
    type PromptCacheSaveTransaction = MlxPromptCacheSaveTransaction;

    fn prepare_prompt_cache_save(
        &mut self,
        state: &mut S,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _context: &Stream,
    ) -> Result<Self::PromptCacheSaveTransaction, Error> {
        let destination = prompt_cache_storage_directory(destination, descriptor.topology());
        let publication = eredu_runtime::ReversiblePromptCachePublication::begin(
            &destination,
            options.replace_existing(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let staging_options =
            PromptCacheOptions::new(options.application_namespace().map(str::to_owned), false)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let manifest = S::save_prompt_cache(
            state,
            publication.staging_destination(),
            descriptor,
            prefix_token_ids,
            &staging_options,
        )?;
        Ok(MlxPromptCacheSaveTransaction {
            publication,
            manifest,
        })
    }

    fn prepared_prompt_cache_manifest(
        transaction: &Self::PromptCacheSaveTransaction,
    ) -> &PromptCacheManifest {
        &transaction.manifest
    }

    fn publish_prompt_cache_save(
        &mut self,
        transaction: &mut Self::PromptCacheSaveTransaction,
    ) -> Result<(), Error> {
        transaction
            .publication
            .publish()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn commit_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        transaction.publication.commit().unwrap_or_else(|error| {
            panic!("committed prompt-cache publication cleanup failed: {error}")
        });
    }

    fn rollback_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        transaction
            .publication
            .rollback()
            .unwrap_or_else(|error| panic!("prompt-cache publication rollback failed: {error}"));
    }
}

struct CompletedReplicatedText<
    A,
    S,
    D = eredu_runtime::DirectReplicatedTextExecution,
    P = NoSelectedPrediction,
> where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    session: ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    prediction: P,
    embedded_prediction_observers: MlxEmbeddedPredictionObservers,
    parameter_bank:
        Option<crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank>,
    #[cfg(test)]
    selected_residency: eredu_runtime::LayerWeightResidency,
    partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
    partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
    partition_sampling_rank: Option<usize>,
    partition_public_output: bool,
    stream: Stream,
}

impl<A, S> CompletedReplicatedText<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn new(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        let selected_residency = prepared.selected().residency();
        let prompt_cache_identity = prepared.prompt_cache_identity().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let mut modules = prepared.into_modules();
        let architecture = modules.take_architecture();
        let source_architecture = modules.take_source_architecture();
        let contract = modules.take_contract();
        let materialization = Arc::new(std::sync::Mutex::new(None));
        let mechanisms = MlxReplicatedTextMechanisms::new(
            store,
            Arc::clone(&materialization),
            stream,
            weights_stream,
        );
        #[cfg(test)]
        super::path_instrumentation::constructor();
        let session = eredu_runtime::construct_replicated_text_session(
            architecture,
            source_architecture,
            contract,
            mechanisms,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let materialization = materialization
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("materialization report lock was poisoned".into())
            })?
            .clone();
        Ok(Self {
            session,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            parameter_bank: None,
            #[cfg(test)]
            selected_residency,
            partition_sampling_group: None,
            partition_communication_authority: None,
            partition_sampling_rank: None,
            partition_public_output: true,
            stream: stream.clone(),
        })
    }
}

impl<A, S, D> CompletedReplicatedText<A, S, D>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    fn from_session(
        session: ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        prompt_cache_identity: PromptCacheModelIdentity,
        capability_estimate: eredu_architectures::capability::CapabilityEstimate,
        effective_model_type: String,
        materialization: Option<eredu_runtime::WeightMaterializationReport>,
        selected_residency: eredu_runtime::LayerWeightResidency,
        partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
        partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
        partition_sampling_rank: Option<usize>,
        partition_public_output: bool,
        stream: &Stream,
    ) -> Self {
        #[cfg(not(test))]
        let _ = selected_residency;
        Self {
            session,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            parameter_bank: None,
            #[cfg(test)]
            selected_residency,
            stream: stream.clone(),
            partition_sampling_group,
            partition_communication_authority,
            partition_sampling_rank,
            partition_public_output,
        }
    }

    fn with_parameter_bank(
        mut self,
        parameter_bank: crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
    ) -> Self {
        self.parameter_bank = Some(parameter_bank);
        self
    }

    fn with_prediction<P>(
        self,
        prediction: SelectedPrediction<P>,
        capability: eredu_architectures::capability::CapabilityEstimate,
    ) -> Result<CompletedReplicatedText<A, S, D, SelectedPrediction<P>>, Error>
    where
        P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        >,
    {
        if prediction.extension.depth() == 0 || capability.speculative_draft_source().is_none() {
            return Err(Error::ArchitectureModel(
                "prediction extension contract is missing executable draft depth".into(),
            ));
        }
        Ok(CompletedReplicatedText {
            session: self.session,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            materialization: self.materialization,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            parameter_bank: self.parameter_bank,
            #[cfg(test)]
            selected_residency: self.selected_residency,
            partition_sampling_group: self.partition_sampling_group,
            partition_communication_authority: self.partition_communication_authority,
            partition_sampling_rank: self.partition_sampling_rank,
            partition_public_output: self.partition_public_output,
            stream: self.stream,
        })
    }
}

impl<A, S, D, P> CompletedReplicatedText<A, S, D, P>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    fn published<T>(&self, value: T) -> T {
        #[cfg(test)]
        super::path_instrumentation::state_publication();
        value
    }
}

impl<A, S, D, P> ErasedReplicatedTextExecutable for CompletedReplicatedText<A, S, D, P>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: ReplicatedPredictionCapability<A, S, D> + 'static,
{
    fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    fn has_partition_control(&self) -> bool {
        self.partition_communication_authority.is_some()
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        self.partition_sampling_group.as_ref().map(|group| {
            (
                group,
                self.partition_communication_authority
                    .as_ref()
                    .expect("partition sampling group has communication authority"),
                &self.stream,
                self.partition_sampling_rank
                    .expect("partition sampling group has selected owner rank"),
            )
        })
    }

    fn partition_public_output(&self) -> bool {
        self.partition_public_output
    }

    fn with_embedded_prediction(
        &mut self,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        P::lend(self, continuation)
    }

    fn has_embedded_prediction(&self) -> bool {
        P::present()
    }

    fn install_embedded_prediction_observers(
        &mut self,
        observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        if P::present() {
            self.embedded_prediction_observers = observers;
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.selected_residency
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot {
        self.session
            .report()
            .expect("MLX state report")
            .state_report()
            .presence
            .clone()
    }

    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().fixed_numeric.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error> {
        let before_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let before = before_report.state_report().presence.clone();
        let before_numeric = before_report.state_report().fixed_numeric.clone();
        let before_retained = before_report.state_report().retained_numeric.clone();
        let checkpoint = self
            .session
            .checkpoint(stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let continuation = self
            .session
            .forward(&MlxTensor::from_array(tokens.clone()), None, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .into_array()
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let advanced_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let advanced = advanced_report.state_report().presence.clone();
        let advanced_numeric = advanced_report.state_report().fixed_numeric.clone();
        self.session
            .rollback(checkpoint, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored = restored_report.state_report().presence.clone();
        let restored_numeric = restored_report.state_report().fixed_numeric.clone();
        let restored_retained = restored_report.state_report().retained_numeric.clone();
        assert_eq!(restored_retained, before_retained);
        Ok((
            before,
            advanced,
            restored,
            before_numeric,
            advanced_numeric,
            restored_numeric,
            continuation,
        ))
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.session
            .report()
            .map(|report| Some(report.execution_report().residency.clone()))
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.session
            .report()
            .map(|report| report.execution_report().dense.clone())
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.materialization.as_ref()
    }

    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        match &self.parameter_bank {
            Some(bank) => bank.report().map(Some),
            None => self.session.execution_strategy().parameter_bank_report(),
        }
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.session
            .reset(&self.stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        self.session
            .reset_distributed(&self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .load_prompt_cache(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .save_prompt_cache(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_distributed(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_for_input_distributed(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .save_prompt_cache_distributed(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .save_prompt_cache_for_input_distributed(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().residency.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let tokens = input::text_token_ids(input, stream)?;
        let output = self
            .session
            .prefill(&MlxTensor::from_array(tokens), None, stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let output = self
            .session
            .decode(&MlxTensor::from_array(tokens.clone()), stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .forward_with_observer(&tokens, mask.as_ref(), stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let tokens = input::text_token_ids(input, stream)?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .prefill_with_observer(&tokens, mask.as_ref(), stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode_with_observer(
        &mut self,
        tokens: &Array,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let tokens = MlxTensor::from_array(tokens.clone());
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .decode_with_observer(&tokens, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }
}

trait CompositePredictionCapability<A, D>: Sized
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn lend(
        model: &mut CompletedComposite<A, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>>;
    fn present() -> bool;
}

impl<A, D> CompositePredictionCapability<A, D> for NoSelectedPrediction
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn lend(
        _: &mut CompletedComposite<A, D, Self>,
        _: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }
    fn present() -> bool {
        false
    }
}

struct CompletedComposite<
    A,
    D = eredu_runtime::DirectReplicatedTextExecution,
    P = NoSelectedPrediction,
> where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    admission: A::AdmissionConfig,
    processor: eredu_runtime::SelectedProcessorExecution,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    prediction: P,
    embedded_prediction_observers: MlxEmbeddedPredictionObservers,
    #[cfg(test)]
    selected_residency: eredu_runtime::LayerWeightResidency,
    partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
    partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
    partition_sampling_rank: Option<usize>,
    partition_public_output: bool,
    stream: Stream,
}

struct MlxCompositePredictionInput<'a, A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    admission: &'a A::AdmissionConfig,
    processor: &'a eredu_runtime::SelectedProcessorExecution,
}

fn prepare_composite_prediction_input<A>(
    admission: &A::AdmissionConfig,
    processor: &eredu_runtime::SelectedProcessorExecution,
    input: input::ModelInput<'_>,
) -> Result<
    (
        eredu_runtime::PreparedModelInput<MlxTensor>,
        eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
        Option<eredu_runtime::PreparedInputCacheIdentity>,
    ),
    Error,
>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    if let Some(modality) = input
        .parts
        .iter()
        .map(|part| part.modality())
        .find(|modality| !processor.modalities().contains(modality))
    {
        return Err(Error::ArchitectureModel(format!(
            "prepared input modality {} is outside the selected composite modalities {:?}",
            modality.as_str(),
            processor.modalities()
        )));
    }
    if !processor.prepared_tensors()
        && input
            .parts
            .iter()
            .any(|part| part.modality() != eredu_core::InputModality::Text)
    {
        return Err(Error::ArchitectureModel(
            "prepared media tensors were not admitted by processor selection".into(),
        ));
    }
    if let Some(modality) = input.parts.iter().find_map(|part| {
        matches!(part.payload(), input::InputPayload::Embeddings(_))
            .then_some(part.modality())
            .filter(|modality| !processor.projected_modalities().contains(modality))
    }) {
        return Err(Error::ArchitectureModel(format!(
            "projected {} embeddings were not admitted by processor selection",
            modality.as_str()
        )));
    }
    let supplied_cache_identity = input.cache_identity().cloned();
    let text_fingerprint = if supplied_cache_identity.is_none()
        && input.parts.iter().all(|part| {
            part.modality() == eredu_core::InputModality::Text
                && matches!(part.payload(), input::InputPayload::TokenIds(_))
        }) {
        let mut tokens = Vec::new();
        for part in input.parts {
            let input::InputPayload::TokenIds(value) = part.payload() else {
                unreachable!("text-only token payload checked above")
            };
            let value = value.evaluated()?;
            tokens.extend_from_slice(
                value
                    .try_as_slice::<u32>()
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            );
        }
        Some(eredu_core::cache::prompt_cache_token_fingerprint(&tokens))
    } else {
        None
    };
    let prepared = prepared_composite_input(input)?;
    if supplied_cache_identity
        .as_ref()
        .is_some_and(|identity| identity.prepared() != prepared.identity())
    {
        return Err(Error::ArchitectureModel(
            "prepared-input cache identity differs from the submitted tensors".into(),
        ));
    }
    let admitted = A::admit_prepared_input(admission, &prepared, &input::MlxTensorInputInspector)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let cache_identity = match (supplied_cache_identity, text_fingerprint) {
        (Some(identity), _) => Some(identity),
        (None, Some(fingerprint)) => Some(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        ),
        (None, None) => None,
    };
    Ok((prepared, admitted, cache_identity))
}

impl<A>
    eredu_architectures::speculative_execution::ReplicatedPredictionInput<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        Exception,
    > for MlxCompositePredictionInput<'_, A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Input = super::MlxModelInput;

    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        context: &Stream,
        operation: impl for<'a> FnOnce(
            <PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Input<'a>,
            MlxTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        input.with_borrowed(|input| {
            let tokens = input::text_token_ids(input, context)
                .map(MlxTensor::from_array)
                .map_err(|error| Exception::custom(error.to_string()))?;
            let (prepared, admitted, identity) =
                prepare_composite_prediction_input::<A>(self.admission, self.processor, input)
                    .map_err(|error| Exception::custom(error.to_string()))?;
            let paired =
                PreparedCompositeInput::new(&prepared, &admitted).map_err(Exception::custom)?;
            operation(paired, tokens, identity.as_ref())
        })
    }

    fn with_decode<R>(
        &mut self,
        tokens: &MlxTensor,
        _context: &Stream,
        operation: impl for<'a> FnOnce(
            <PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Input<'a>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let part = input::token_ids_part(tokens.as_array())
            .map_err(|error| Exception::custom(error.to_string()))?;
        let input = input::ModelInput::new(std::slice::from_ref(&part));
        let (prepared, admitted, _) =
            prepare_composite_prediction_input::<A>(self.admission, self.processor, input)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Exception::custom)?;
        operation(paired)
    }
}

impl<A> CompletedComposite<A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
{
    fn new(
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        let selected_residency = prepared.selected().residency();
        let prompt_cache_identity = prepared.prompt_cache_identity().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let (architecture, source_architecture, contract, processor, admission) =
            prepared.into_parts();
        let materialization = Arc::new(std::sync::Mutex::new(None));
        let mechanisms = MlxReplicatedTextMechanisms::new(
            store,
            Arc::clone(&materialization),
            stream,
            weights_stream,
        );
        #[cfg(test)]
        super::path_instrumentation::constructor();
        let session = eredu_runtime::construct_replicated_text_session(
            architecture,
            source_architecture,
            contract,
            mechanisms,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let materialization = materialization
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("materialization report lock was poisoned".into())
            })?
            .clone();
        Ok(Self {
            session,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            #[cfg(test)]
            selected_residency,
            partition_sampling_group: None,
            partition_communication_authority: None,
            partition_sampling_rank: None,
            partition_public_output: true,
            stream: stream.clone(),
        })
    }
}

impl<A, D, P> CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn with_native_prediction_target_state<T>(
        &mut self,
        cache: &mut MlxPredictionTargetState,
        operation: impl FnOnce(
            &mut ReplicatedTextSession<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
            >,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if !cache.is::<MlxHybridState>() {
            return Err(Error::ArchitectureModel(
                "external-assistant target cache differs from the neutral composite state".into(),
            ));
        }
        let mut lane = cache.take_state::<MlxHybridState>()?;
        if let Err(error) = self
            .session
            .exchange_prediction_target_state(&mut lane, &self.stream)
        {
            cache.restore_state(lane);
            return Err(Error::ArchitectureModel(error.to_string()));
        }
        let result = operation(&mut self.session);
        let restored = match self
            .session
            .exchange_prediction_target_state(&mut lane, &self.stream)
        {
            Ok(()) => Ok(()),
            Err(error) => self
                .session
                .recover_prediction_target_state_after_failure(&mut lane)
                .map_err(|recovery| {
                    Error::ArchitectureModel(format!(
                        "prediction target state exchange failed: {error}; local ownership recovery failed: {recovery}"
                    ))
                })
                .and(Err(Error::ArchitectureModel(error.to_string()))),
        };
        cache.restore_state(lane);
        match (result, restored) {
            (Err(error), _) => Err(error),
            (Ok(output), Ok(())) => Ok(output),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    fn with_prediction<Q>(
        self,
        prediction: SelectedPrediction<Q>,
        capability: eredu_architectures::capability::CapabilityEstimate,
    ) -> Result<CompletedComposite<A, D, SelectedPrediction<Q>>, Error>
    where
        Q: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        >,
    {
        if prediction.extension.depth() == 0 || capability.speculative_draft_source().is_none() {
            return Err(Error::ArchitectureModel(
                "prediction extension contract is missing executable draft depth".into(),
            ));
        }
        Ok(CompletedComposite {
            session: self.session,
            admission: self.admission,
            processor: self.processor,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            materialization: self.materialization,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            #[cfg(test)]
            selected_residency: self.selected_residency,
            partition_sampling_group: self.partition_sampling_group,
            partition_communication_authority: self.partition_communication_authority,
            partition_sampling_rank: self.partition_sampling_rank,
            partition_public_output: self.partition_public_output,
            stream: self.stream,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn from_session(
        session: ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
            D,
        >,
        admission: A::AdmissionConfig,
        processor: eredu_runtime::SelectedProcessorExecution,
        prompt_cache_identity: PromptCacheModelIdentity,
        capability_estimate: eredu_architectures::capability::CapabilityEstimate,
        effective_model_type: String,
        materialization: Option<eredu_runtime::WeightMaterializationReport>,
        selected_residency: eredu_runtime::LayerWeightResidency,
        partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
        partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
        partition_sampling_rank: Option<usize>,
        partition_public_output: bool,
        stream: &Stream,
    ) -> CompletedComposite<A, D, NoSelectedPrediction> {
        #[cfg(not(test))]
        let _ = selected_residency;
        CompletedComposite {
            session,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            #[cfg(test)]
            selected_residency,
            partition_sampling_group,
            partition_communication_authority,
            partition_sampling_rank,
            partition_public_output,
            stream: stream.clone(),
        }
    }

    fn prepare(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<
        (
            eredu_runtime::PreparedModelInput<MlxTensor>,
            eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
            Option<eredu_runtime::PreparedInputCacheIdentity>,
        ),
        Error,
    > {
        if let Some(modality) = input
            .parts
            .iter()
            .map(|part| part.modality())
            .find(|modality| !self.processor.modalities().contains(modality))
        {
            return Err(Error::ArchitectureModel(format!(
                "prepared input modality {} is outside the selected composite modalities {:?}",
                modality.as_str(),
                self.processor.modalities()
            )));
        }
        if !self.processor.prepared_tensors()
            && input
                .parts
                .iter()
                .any(|part| part.modality() != eredu_core::InputModality::Text)
        {
            return Err(Error::ArchitectureModel(
                "prepared media tensors were not admitted by processor selection".into(),
            ));
        }
        if let Some(modality) = input.parts.iter().find_map(|part| {
            matches!(part.payload(), input::InputPayload::Embeddings(_))
                .then_some(part.modality())
                .filter(|modality| !self.processor.projected_modalities().contains(modality))
        }) {
            return Err(Error::ArchitectureModel(format!(
                "projected {} embeddings were not admitted by processor selection",
                modality.as_str()
            )));
        }
        let supplied_cache_identity = input.cache_identity().cloned();
        let text_fingerprint = if supplied_cache_identity.is_none()
            && input.parts.iter().all(|part| {
                part.modality() == eredu_core::InputModality::Text
                    && matches!(part.payload(), input::InputPayload::TokenIds(_))
            }) {
            let mut tokens = Vec::new();
            for part in input.parts {
                let input::InputPayload::TokenIds(value) = part.payload() else {
                    unreachable!("text-only token payload checked above")
                };
                let value = value.evaluated()?;
                tokens.extend_from_slice(
                    value
                        .try_as_slice::<u32>()
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                );
            }
            Some(eredu_core::cache::prompt_cache_token_fingerprint(&tokens))
        } else {
            None
        };
        let prepared = prepared_composite_input(input)?;
        if supplied_cache_identity
            .as_ref()
            .is_some_and(|identity| identity.prepared() != prepared.identity())
        {
            return Err(Error::ArchitectureModel(
                "prepared-input cache identity differs from the submitted tensors".into(),
            ));
        }
        let admitted =
            A::admit_prepared_input(&self.admission, &prepared, &input::MlxTensorInputInspector)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let cache_identity = match (supplied_cache_identity, text_fingerprint) {
            (Some(identity), _) => Some(identity),
            (None, Some(fingerprint)) => Some(
                prepared
                    .cache_identity(fingerprint)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            ),
            (None, None) => None,
        };
        Ok((prepared, admitted, cache_identity))
    }

    fn text_input(
        &self,
        tokens: &Array,
    ) -> Result<
        (
            eredu_runtime::PreparedModelInput<MlxTensor>,
            eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
            Option<eredu_runtime::PreparedInputCacheIdentity>,
        ),
        Error,
    > {
        let part = input::token_ids_part(tokens)?;
        self.prepare(input::ModelInput::new(std::slice::from_ref(&part)))
    }

    fn published<T>(&self, value: T) -> T {
        #[cfg(test)]
        super::path_instrumentation::state_publication();
        value
    }
}

impl<A, D, P> ErasedReplicatedTextExecutable for CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: CompositePredictionCapability<A, D> + 'static,
{
    fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    fn with_embedded_prediction(
        &mut self,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        P::lend(self, continuation)
    }

    fn has_embedded_prediction(&self) -> bool {
        P::present()
    }

    fn install_embedded_prediction_observers(
        &mut self,
        observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        if P::present() {
            self.embedded_prediction_observers = observers;
            true
        } else {
            false
        }
    }

    fn has_partition_control(&self) -> bool {
        self.partition_communication_authority.is_some()
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        self.partition_sampling_group.as_ref().map(|group| {
            (
                group,
                self.partition_communication_authority
                    .as_ref()
                    .expect("partition sampling group has communication authority"),
                &self.stream,
                self.partition_sampling_rank
                    .expect("partition sampling group has selected owner rank"),
            )
        })
    }

    fn partition_public_output(&self) -> bool {
        self.partition_public_output
    }

    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut (dyn ErasedExternalPredictionExecutable + 'static)> {
        A::external_assistant_target_profile(&self.admission).map(|_| self as _)
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.selected_residency
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot {
        self.session
            .report()
            .expect("MLX composite state report")
            .state_report()
            .presence
            .clone()
    }

    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().fixed_numeric.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error> {
        let before_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let before = before_report.state_report().presence.clone();
        let before_numeric = before_report.state_report().fixed_numeric.clone();
        let before_retained = before_report.state_report().retained_numeric.clone();
        let checkpoint = self
            .session
            .checkpoint(stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let (prepared, admitted, _) = self.text_input(tokens)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let continuation = self
            .session
            .decode_input(paired, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .into_array()
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let advanced_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let advanced = advanced_report.state_report().presence.clone();
        let advanced_numeric = advanced_report.state_report().fixed_numeric.clone();
        self.session
            .rollback(checkpoint, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored = restored_report.state_report().presence.clone();
        let restored_numeric = restored_report.state_report().fixed_numeric.clone();
        let restored_retained = restored_report.state_report().retained_numeric.clone();
        assert_eq!(restored_retained, before_retained);
        Ok((
            before,
            advanced,
            restored,
            before_numeric,
            advanced_numeric,
            restored_numeric,
            continuation,
        ))
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.session
            .report()
            .map(|report| Some(report.execution_report().residency.clone()))
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.session
            .report()
            .map(|report| report.execution_report().dense.clone())
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.materialization.as_ref()
    }

    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        self.session.execution_strategy().parameter_bank_report()
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.session
            .reset(&self.stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        self.session
            .reset_distributed(&self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        let _ = (directory, expected, prefix_token_ids);
        Err(Error::ArchitectureModel(
            "composite prompt-cache loading requires the prepared-input identity".into(),
        ))
    }

    fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .load_prompt_cache_for_input(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self
            .session
            .committed_prompt_input_identity()
            .cloned()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "composite prompt cache requires a committed prepared-input identity".into(),
                )
            })?;
        self.session
            .save_prompt_cache_for_input(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache_distributed(
        &mut self,
        _directory: &Path,
        _expected: &PromptCacheDescriptor,
        _prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(Error::ArchitectureModel(
            "composite prompt-cache loading requires the prepared-input identity".into(),
        ))
    }

    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_for_input_distributed(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        let identity = self
            .session
            .committed_prompt_input_identity()
            .cloned()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "composite prompt cache requires a committed prepared-input identity".into(),
                )
            })?;
        self.save_prompt_cache_for_input_distributed(
            destination,
            descriptor,
            prefix_token_ids,
            options,
            &identity,
        )
    }

    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .save_prompt_cache_for_input_distributed(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().residency.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        let (prepared, admitted, cache_identity) = self.prepare(input)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        #[cfg(test)]
        super::path_instrumentation::forward();
        let output = match cache_identity {
            Some(identity) => self
                .session
                .prefill_input_with_cache_identity(paired, identity, stream),
            None => self.session.prefill_input(paired, stream),
        }
        .map(MlxTensor::into_array)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let (prepared, admitted, _) = self.text_input(tokens)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let output = self
            .session
            .decode_input(paired, stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        if mask.is_some() {
            return Err(Error::ArchitectureModel(
                "explicit composite decoder masks require prepared-input metadata".into(),
            ));
        }
        let parts = [input::token_ids_part(tokens)?];
        self.prefill_with_observer(input::ModelInput::new(&parts), None, stream, observer)
    }

    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        if mask.is_some() {
            return Err(Error::ArchitectureModel(
                "explicit composite decoder masks require prepared-input metadata".into(),
            ));
        }
        #[cfg(test)]
        super::path_instrumentation::forward();
        let (prepared, admitted, cache_identity) = self.prepare(input)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = match cache_identity {
            Some(identity) => self.session.prefill_input_with_observer_and_cache_identity(
                paired,
                identity,
                stream,
                &mut observer,
            ),
            None => self
                .session
                .prefill_input_with_observer(paired, stream, &mut observer),
        }
        .map(MlxTensor::into_array)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode_with_observer(
        &mut self,
        tokens: &Array,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        super::path_instrumentation::forward();
        let (prepared, admitted, _) = self.text_input(tokens)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .decode_input_with_observer(paired, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }
}

impl<A, D, P> ErasedExternalPredictionExecutable for CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: 'static,
{
    fn prepare_external_prediction_target_cache(
        &mut self,
    ) -> Result<MlxPredictionTargetState, Error> {
        self.session
            .prepare_prediction_target_state(&self.stream)
            .map(MlxPredictionTargetState::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn prefill_external_prediction_target(
        &mut self,
        input: input::ModelInput<'_>,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error> {
        let paths = A::external_prediction_capture_paths(request)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "assistant capture request differs from the neutral target architecture".into(),
                )
            })?;
        let mut observer = ExactPredictionCaptureObserver::new(paths)?;
        let (prepared, admitted, _) = self.prepare(input)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let request = request.clone();
        let captured = observer.values.clone();
        let stream = self.stream.clone();
        self.with_native_prediction_target_state(cache, |session| {
            session
                .prefill_input_with_capture(paired, &stream, &mut observer, |forward| {
                    let values = captured
                        .borrow()
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| {
                            value.ok_or_else(|| {
                                eredu_nn::Error::backend(format!(
                                    "external-assistant target did not reach capture path {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    A::external_prediction_capture(&request, forward, values)?.ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "architecture did not produce its selected assistant capture",
                        )
                    })
                })
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
    }

    fn verify_external_prediction_target(
        &mut self,
        tokens: &MlxTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error> {
        let (prepared, admitted, _) = self.text_input(tokens.as_array())?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let paths = A::external_prediction_capture_paths(request)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "assistant capture request differs from the neutral target architecture".into(),
                )
            })?;
        let mut observer = ExactPredictionCaptureObserver::new(paths)?;
        let request = request.clone();
        let captured = observer.values.clone();
        let stream = self.stream.clone();
        self.with_native_prediction_target_state(cache, |session| {
            session
                .decode_input_with_capture(paired, &stream, &mut observer, |forward| {
                    let values = captured
                        .borrow()
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| {
                            value.ok_or_else(|| {
                                eredu_nn::Error::backend(format!(
                                    "external-assistant target did not reach capture path {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    A::external_prediction_capture(&request, forward, values)?.ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "architecture did not produce its selected assistant capture",
                        )
                    })
                })
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
    }

    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, MlxTensor>,
    ) -> Result<MlxTensor, Error> {
        self.session
            .apply_prediction_target_operation(
                CompositePredictionTargetOperation { operation },
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

impl<A, S, D, P> ReplicatedPredictionCapability<A, S, D> for SelectedPrediction<P>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let mut strategy =
            eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy::<
                A,
                MlxNeuralBackend,
                S,
                MlxReplicatedTextMechanisms<A, S>,
                D,
                P,
                super::prepared_speculative::MlxTextPredictionInput,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
            >::new(
                &mut model.session,
                &mut model.prediction.extension,
                selected,
                super::prepared_speculative::MlxTextPredictionInput,
                &model.stream,
            );
        let observers = std::mem::take(&mut model.embedded_prediction_observers);
        let mut executor = eredu_architectures::speculative_execution::EmbeddedPredictionExecutor::<
            _,
            super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
        >::with_observers(&mut strategy, observers);
        let result = {
            let mut erased = eredu_architectures::speculative_execution::DynEmbeddedExecutor::<
                super::prepared_speculative::MlxEmbeddedExecutorTypes,
            >::new(&mut executor);
            continuation.execute(selected, &mut erased)
        };
        model.embedded_prediction_observers = executor.into_observers();
        Some(result)
    }

    fn present() -> bool {
        true
    }
}

/// Family-agnostic MLX binder for architecture-owned composite ingress.
#[derive(Clone, Copy)]
pub(crate) struct CompositeBindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
}

impl<A, D, P> CompositePredictionCapability<A, D> for SelectedPrediction<P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn lend(
        model: &mut CompletedComposite<A, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let input = MlxCompositePredictionInput::<A> {
            admission: &model.admission,
            processor: &model.processor,
        };
        let mut strategy =
            eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy::<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
                P,
                MlxCompositePredictionInput<'_, A>,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
            >::new(
                &mut model.session,
                &mut model.prediction.extension,
                selected,
                input,
                &model.stream,
            );
        let observers = std::mem::take(&mut model.embedded_prediction_observers);
        let mut executor = eredu_architectures::speculative_execution::EmbeddedPredictionExecutor::<
            _,
            super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
        >::with_observers(&mut strategy, observers);
        let result = {
            let mut erased = eredu_architectures::speculative_execution::DynEmbeddedExecutor::<
                super::prepared_speculative::MlxEmbeddedExecutorTypes,
            >::new(&mut executor);
            continuation.execute(selected, &mut erased)
        };
        model.embedded_prediction_observers = executor.into_observers();
        Some(result)
    }

    fn present() -> bool {
        true
    }
}

impl
    eredu_architectures::replicated_text::CompositePredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<MlxNeuralBackend>>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        CompletedComposite::new(prepared, store, self.stream, self.weights_stream)?
            .with_prediction(
                SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                self.capability,
            )
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<MlxNeuralBackend>>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        let prompt_cache_identity = prepared.routed().text().prompt_cache_identity().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let selected_residency = prepared.routed().text().selected().residency();
        let bank_residency = prepared.routed().bank_residency();
        let (routed, processor, admission) = prepared.into_parts();
        let slot = Arc::new(std::sync::Mutex::new(None));
        let mechanisms =
            MlxReplicatedTextMechanisms::<PreparedCompositeArchitecture<A>, MlxHybridState>::new(
                Arc::clone(&store),
                Arc::clone(&slot),
                self.stream,
                self.weights_stream,
            );
        let materialization =
            |slot: &Arc<std::sync::Mutex<Option<eredu_runtime::WeightMaterializationReport>>>| {
                slot.lock()
                    .map_err(|_| {
                        Error::ArchitectureModel("materialization report lock was poisoned".into())
                    })
                    .map(|report| report.clone())
            };
        let prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        #[cfg(test)]
        super::path_instrumentation::constructor();
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = routed
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                    session,
                    admission,
                    processor,
                    prompt_cache_identity,
                    self.capability.clone(),
                    effective_model_type,
                    materialization(&slot)?,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )
                .with_prediction(prediction, self.capability)
                .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    routed.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = routed
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                    session,
                    admission,
                    processor,
                    prompt_cache_identity,
                    self.capability.clone(),
                    effective_model_type,
                    materialization(&slot)?,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )
                .with_prediction(prediction, self.capability)
                .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected composite bank residency".into(),
            )),
        }
    }
}

impl CompositeTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for CompositeBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
    {
        CompletedComposite::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let prompt_cache_identity = prepared.routed().text().prompt_cache_identity().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let selected_residency = prepared.routed().text().selected().residency();
        let bank_residency = prepared.routed().bank_residency();
        let (routed, processor, admission) = prepared.into_parts();
        let materialization_slot: Arc<
            std::sync::Mutex<Option<eredu_runtime::WeightMaterializationReport>>,
        > = Arc::new(std::sync::Mutex::new(None));
        let mechanisms: MlxReplicatedTextMechanisms<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        > = MlxReplicatedTextMechanisms::new(
            Arc::clone(&store),
            Arc::clone(&materialization_slot),
            self.stream,
            self.weights_stream,
        );
        #[cfg(test)]
        super::path_instrumentation::constructor();
        let materialization =
            |slot: &Arc<std::sync::Mutex<Option<eredu_runtime::WeightMaterializationReport>>>| {
                slot.lock()
                    .map_err(|_| {
                        Error::ArchitectureModel("materialization report lock was poisoned".into())
                    })
                    .map(|report| report.clone())
            };
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = routed
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                let materialization = materialization(&materialization_slot)?;
                Ok(Box::new(
                    CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                        session,
                        admission,
                        processor,
                        prompt_cache_identity,
                        capability_estimate,
                        effective_model_type,
                        materialization,
                        selected_residency,
                        None,
                        None,
                        None,
                        true,
                        self.stream,
                    ),
                ))
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    routed.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = routed
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                let materialization = materialization(&materialization_slot)?;
                Ok(Box::new(
                    CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                        session,
                        admission,
                        processor,
                        prompt_cache_identity,
                        capability_estimate,
                        effective_model_type,
                        materialization,
                        selected_residency,
                        None,
                        None,
                        None,
                        true,
                        self.stream,
                    ),
                ))
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected composite bank residency".into(),
            )),
        }
    }
}

pub(super) fn bind_routed_text(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::SelectedRoutedTextRealization,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    if selected.plan().relu2().is_some() {
        return eredu_architectures::visit_relu2_routed_text_architecture::<
            MlxNeuralBackend,
            MlxHybridState,
            _,
        >(
            inspection,
            selected,
            store,
            stream,
            Relu2RoutedBindingVisitor {
                stream,
                weights_stream,
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()));
    }
    let uses_pooling_attention = selected
        .text()
        .state()
        .components()
        .iter()
        .any(|component| {
            matches!(
                component.component().role(),
                eredu_core::cache::StateComponentRole::Fixed(
                    eredu_core::cache::StateTensorRole::Pooling { .. }
                )
            )
        });
    if uses_pooling_attention {
        return eredu_architectures::visit_pooling_routed_text_architecture::<
            MlxNeuralBackend,
            MlxPoolingAttentionState,
            _,
        >(
            inspection,
            selected,
            store,
            stream,
            PoolingRoutedBindingVisitor {
                stream,
                weights_stream,
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()));
    }
    eredu_architectures::visit_gated_routed_text_architecture::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        inspection,
        selected,
        store,
        stream,
        RoutedBindingVisitor {
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn selected_addressable_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<crate::backend::runtime::residency::parameter_bank::AddressableParameterBank, Error> {
    let selected =
        crate::backend::runtime::residency::parameter_bank::entries_from_selected_members(
            members,
            store.as_ref(),
        )?;
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank::new_selected_shared(
        store,
        selected,
        options,
        weights_stream.clone(),
        stream.clone(),
    )
    .map_err(Into::into)
}

fn shard_addressable_members(
    members: &[eredu_runtime::AddressableBankMember],
    store: &dyn CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<eredu_runtime::AddressableBankMember>, Error> {
    members
        .iter()
        .map(|member| {
            let source_bindings = member
                .parameters()
                .iter()
                .map(|parameter| {
                    eredu_runtime::WeightBinding::from_recipe(
                        parameter.binding_name(),
                        parameter.recipe().clone(),
                        parameter.source_bytes(),
                    )?
                    .with_logical_target(parameter.task().name())
                    .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let bindings = shard_addressable_member_bindings(source_bindings, store, layout)?;
            let parameters = member
                .parameters()
                .iter()
                .zip(bindings)
                .map(|(parameter, binding)| {
                    let recipe = binding.source_recipe();
                    let metadata = recipe.infer(store)?;
                    let selected_bytes = eredu_runtime::selected_addressable_parameter_bytes(
                        parameter.task(),
                        &metadata,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                    let companions = parameter.quantization_companions().cloned();
                    eredu_runtime::AddressableBankParameter::new(
                        parameter.binding_name(),
                        parameter.task().clone(),
                        recipe,
                        metadata,
                        selected_bytes,
                        companions,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            eredu_runtime::AddressableBankMember::new(
                member.key(),
                member.placement().clone(),
                parameters,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn selected_addressable_partition_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    layout: &eredu_runtime::LocalModelLayout,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<
    (
        std::collections::BTreeMap<eredu_runtime::ParameterBankKey, u64>,
        MlxSharedAddressableBank,
    ),
    Error,
> {
    let members = shard_addressable_members(members, store.as_ref(), layout)?;
    let bank = selected_addressable_bank(&members, store, options, weights_stream, stream)?;
    let selected_member_bytes = members
        .iter()
        .map(|member| {
            let bytes = <crate::backend::runtime::residency::parameter_bank::AddressableParameterBank as eredu_runtime::AddressableGroupedBank<MlxNeuralBackend>>::member_bytes(
                &bank,
                member.key(),
            )
            .expect("constructed addressable bank retains every selected member");
            (member.key(), bytes)
        })
        .collect();
    Ok((selected_member_bytes, MlxSharedAddressableBank::new(bank)))
}

#[derive(Clone, Copy)]
struct Relu2RoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::Relu2RoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for Relu2RoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRelu2RoutedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
        let capability_estimate = prepared.text().capability_estimate().clone();
        let effective_model_type = prepared.text().effective_model_type().to_owned();
        let selected_residency = prepared.text().selected().residency();
        let bank_residency = prepared.bank_residency();
        let materialization_slot = Arc::new(std::sync::Mutex::new(None));
        let mechanisms: MlxReplicatedTextMechanisms<A, MlxHybridState> =
            MlxReplicatedTextMechanisms::new(
                Arc::clone(&store),
                Arc::clone(&materialization_slot),
                self.stream,
                self.weights_stream,
            );
        #[cfg(test)]
        super::path_instrumentation::constructor();
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = prepared
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                let materialization = materialization_slot
                    .lock()
                    .map_err(|_| {
                        Error::ArchitectureModel("materialization report lock was poisoned".into())
                    })?
                    .clone();
                Ok(Box::new(CompletedReplicatedText::from_session(
                    session,
                    prompt_cache_identity,
                    capability_estimate,
                    effective_model_type,
                    materialization,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )))
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    prepared.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = prepared
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                let materialization = materialization_slot
                    .lock()
                    .map_err(|_| {
                        Error::ArchitectureModel("materialization report lock was poisoned".into())
                    })?
                    .clone();
                Ok(Box::new(CompletedReplicatedText::from_session(
                    session,
                    prompt_cache_identity,
                    capability_estimate,
                    effective_model_type,
                    materialization,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )))
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected addressable bank residency".into(),
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct RoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

#[derive(Clone, Copy)]
struct PoolingRoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

fn bind_prepared_routed<A, S>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
    let capability_estimate = prepared.text().capability_estimate().clone();
    let effective_model_type = prepared.text().effective_model_type().to_owned();
    let selected_residency = prepared.text().selected().residency();
    let bank_residency = prepared.bank_residency();
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mechanisms: MlxReplicatedTextMechanisms<A, S> = MlxReplicatedTextMechanisms::new(
        Arc::clone(&store),
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    #[cfg(test)]
    super::path_instrumentation::constructor();
    match bank_residency {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let session = prepared
                .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, stream)
                .map_err(Error::ArchitectureModel)?;
            let materialization = materialization_slot
                .lock()
                .map_err(|_| {
                    Error::ArchitectureModel("materialization report lock was poisoned".into())
                })?
                .clone();
            Ok(Box::new(CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability_estimate,
                effective_model_type,
                materialization,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )))
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            let bank = selected_addressable_bank(
                prepared.addressable_members(),
                store,
                options,
                weights_stream,
                stream,
            )?;
            let session = prepared
                .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                    mechanisms,
                    bank,
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    stream,
                )
                .map_err(Error::ArchitectureModel)?;
            let materialization = materialization_slot
                .lock()
                .map_err(|_| {
                    Error::ArchitectureModel("materialization report lock was poisoned".into())
                })?
                .clone();
            Ok(Box::new(CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability_estimate,
                effective_model_type,
                materialization,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )))
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported selected addressable bank residency".into(),
        )),
    }
}

fn bind_prepared_routed_prediction<A, S, P>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    extension: P,
    selected: eredu_runtime::SelectedSpeculativeRealization,
    capability: eredu_architectures::capability::CapabilityEstimate,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
    let effective_model_type = prepared.text().effective_model_type().to_owned();
    let selected_residency = prepared.text().selected().residency();
    let bank_residency = prepared.bank_residency();
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mechanisms = MlxReplicatedTextMechanisms::<A, S>::new(
        Arc::clone(&store),
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    let prediction = SelectedPrediction {
        extension,
        selected,
    };
    #[cfg(test)]
    super::path_instrumentation::constructor();
    match bank_residency {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let session = prepared
                .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, stream)
                .map_err(Error::ArchitectureModel)?;
            let materialization = materialization_slot
                .lock()
                .map_err(|_| {
                    Error::ArchitectureModel("materialization report lock was poisoned".into())
                })?
                .clone();
            CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability.clone(),
                effective_model_type,
                materialization,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )
            .with_prediction(prediction, capability)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            let bank = selected_addressable_bank(
                prepared.addressable_members(),
                store,
                options,
                weights_stream,
                stream,
            )?;
            let session = prepared
                .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                    mechanisms,
                    bank,
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    stream,
                )
                .map_err(Error::ArchitectureModel)?;
            let materialization = materialization_slot
                .lock()
                .map_err(|_| {
                    Error::ArchitectureModel("materialization report lock was poisoned".into())
                })?
                .clone();
            CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability.clone(),
                effective_model_type,
                materialization,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )
            .with_prediction(prediction, capability)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported selected addressable bank residency".into(),
        )),
    }
}

impl<S>
    eredu_architectures::routed_text::RoutedPredictionTargetVisitor<
        MlxNeuralBackend,
        S,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
where
    S: MlxStateMechanisms + 'static,
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed_prediction(
            prepared,
            extension,
            self.selected,
            self.capability,
            store,
            self.stream,
            self.weights_stream,
        )
    }
}

impl
    eredu_architectures::GatedRoutedTextArchitectureVisitor<
        MlxNeuralBackend,
        MlxPoolingAttentionState,
    > for PoolingRoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                Error = eredu_nn::Error,
            > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxPoolingAttentionState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream)
    }
}

impl eredu_architectures::GatedRoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for RoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream)
    }
}
/// Family-agnostic MLX visitor that binds neutral parameter topology.
#[derive(Clone, Copy)]
pub(crate) struct BindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
}

pub(crate) struct PredictionBindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

#[derive(Clone, Copy)]
struct OrdinaryReplicatedFinalizer;

struct PredictionReplicatedFinalizer<P> {
    prediction: SelectedPrediction<P>,
    capability: eredu_architectures::capability::CapabilityEstimate,
}

trait ReplicatedExecutableFinalizer<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static;
}

impl<A, S> ReplicatedExecutableFinalizer<A, S> for OrdinaryReplicatedFinalizer
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        Ok(Box::new(completed))
    }
}

impl<A, S, P> ReplicatedExecutableFinalizer<A, S> for PredictionReplicatedFinalizer<P>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        completed
            .with_prediction(self.prediction, self.capability)
            .map(|completed| Box::new(completed) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl<S>
    eredu_architectures::replicated_text::ReplicatedPredictionTargetVisitor<
        MlxNeuralBackend,
        S,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
where
    S: MlxStateMechanisms + 'static,
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)?
            .with_prediction(prediction, self.capability)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl
    eredu_architectures::replicated_text::ReplicatedPredictionProfileDispatcher<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;
    type State = MlxHybridState;
    type Visitor = Self;

    fn into_visitor(self) -> Self::Visitor {
        self
    }
}

impl
    eredu_architectures::routed_text::RoutedPredictionProfileDispatcher<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;
    type GatedState = MlxHybridState;
    type PoolingState = MlxPoolingAttentionState;
    type GatedVisitor = Self;
    type PoolingVisitor = Self;

    fn into_gated_visitor(self) -> Self::GatedVisitor {
        self
    }

    fn into_pooling_visitor(self) -> Self::PoolingVisitor {
        self
    }
}

struct PartitionedDenseDecoderBindingVisitor<'a> {
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

pub(crate) struct PartitionedPredictionBindingVisitor<'a> {
    pub distributed: crate::backend::distributed::MlxDistributedSession,
    pub additional_claimed_sources: std::collections::BTreeSet<String>,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

impl
    eredu_architectures::partitioned_execution::PartitionedPredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PartitionedPredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            PredictionReplicatedFinalizer {
                prediction: SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                capability: self.capability,
            },
        )
    }
}

struct PartitionedRoutedDecoderBindingVisitor<'a> {
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

struct PartitionedPoolingRoutedDecoderBindingVisitor<'a> {
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl
    eredu_architectures::partitioned_execution::RoutedPartitionedProductionVisitor<
        MlxNeuralBackend,
        MlxPoolingAttentionState,
    > for PartitionedPoolingRoutedDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            > + ReplicatedTextArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                Error = eredu_nn::Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_routed_resident(
            prepared,
            store,
            self.distributed,
            std::collections::BTreeSet::new(),
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

macro_rules! impl_partitioned_prediction_binding {
    ($state:ty) => {
        impl
            eredu_architectures::partitioned_execution::RoutedPartitionedPredictionTargetProductionVisitor<
                MlxNeuralBackend,
                $state,
                MlxEmbeddedPredictionMaterializer,
            > for PartitionedPredictionBindingVisitor<'_>
        {
            type Output = Box<dyn ErasedReplicatedTextExecutable>;
            type Error = Error;

            fn visit<A, G>(
                self,
                prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
                    MlxNeuralBackend,
                    A,
                    G,
                    <A as eredu_runtime::PartitionedLayeredArchitecture<
                        MlxNeuralBackend,
                        $state,
                    >>::Boundary,
                >,
                extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                    MlxNeuralBackend,
                >>::Extension<MlxEmbeddedPredictionMaterializer>,
                store: Arc<dyn CheckpointSource>,
            ) -> Result<Self::Output, Self::Error>
            where
                A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                        MlxNeuralBackend,
                        $state,
                    > + ReplicatedTextArchitecture<MlxNeuralBackend, $state, Error = eredu_nn::Error>
                    + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, $state>
                    + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                        MlxNeuralBackend,
                    > + 'static,
                A::StaticModules: Clone,
                G: 'static,
            {
                bind_partitioned_routed_resident(
                    prepared,
                    store,
                    self.distributed,
                    self.additional_claimed_sources,
                    self.stream,
                    self.weights_stream,
                    PredictionReplicatedFinalizer {
                        prediction: SelectedPrediction {
                            extension,
                            selected: self.selected,
                        },
                        capability: self.capability,
                    },
                )
            }
        }
    };
}

impl_partitioned_prediction_binding!(MlxHybridState);
impl_partitioned_prediction_binding!(MlxPoolingAttentionState);

impl
    eredu_architectures::partitioned_execution::RoutedPartitionedPredictionTargetProductionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
        eredu_nn::GroupedRelu2Spec,
    > for PartitionedPredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
            eredu_nn::GroupedRelu2Spec,
        >,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_relu2_resident(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            PredictionReplicatedFinalizer {
                prediction: SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                capability: self.capability,
            },
        )
    }
}

impl
    eredu_architectures::partitioned_execution::RoutedPartitionedProductionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedRoutedDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_routed_resident(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}
impl
    eredu_architectures::partitioned_execution::RoutedPartitionedProductionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        eredu_nn::GroupedRelu2Spec,
    > for PartitionedRoutedDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
            eredu_nn::GroupedRelu2Spec,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_relu2_resident(
            prepared,
            store,
            self.distributed,
            std::collections::BTreeSet::new(),
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}
impl
    eredu_architectures::partitioned_execution::PartitionedArchitectureVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedDenseDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

fn bind_partitioned<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    prepared.dispatch_execution(
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |prepared, (store, distributed, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_local(
                prepared,
                store,
                distributed,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
        |prepared, (store, distributed, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_pipeline(
                prepared,
                store,
                distributed,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
    )
}

fn bind_partitioned_local<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().residency();
    let tensor_group = prepared
        .prepared()
        .selected()
        .tensor_group()
        .ok_or_else(|| {
            Error::ArchitectureModel("direct partition has no selected tensor group".into())
        })?;
    let execution_plan = prepared
        .prepared()
        .selected()
        .direct_execution_plan()
        .map_err(Error::ArchitectureModel)?;
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "direct resident execution has no selected output publication authority".into(),
            )
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("direct partition has no state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mut mechanisms = MlxReplicatedTextMechanisms::new(
        store,
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    mechanisms.set_ignored_checkpoint_sources(additional_claimed_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    #[cfg(test)]
    {
        super::path_instrumentation::constructor();
        super::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, source_architecture, physical_layout, selected, context| {
                let (mut architecture, partition, manifest, tasks) = input.into_parts();
                let (mut source_architecture, source_layout) = source_architecture
                    .map_or((None, None), |(architecture, layout)| {
                        (Some(architecture), Some(layout))
                    });
                mechanisms.set_parallel_layout(physical_layout);
                mechanisms.set_source_parallel_layout(source_layout);
                let layout = partition.unit_layout().clone();
                partition_materialization_tasks(&tasks, &layout)?;
                let mut units = Vec::with_capacity(layout.len());
                for ordinal in 0..layout.len() {
                    let address = layout.address(ordinal).ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "partition unit ordinal {ordinal} has no canonical address"
                        ))
                    })?;
                    units.push(
                        <A as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::build_unit(
                            &architecture,
                            address.group(),
                            address.index(),
                            context,
                        )
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                    );
                }
                let mut source_units = source_architecture
                    .as_ref()
                    .map(|source| {
                        (0..layout.len())
                            .map(|ordinal| {
                                let address = layout.address(ordinal).ok_or_else(|| {
                                    Error::ArchitectureModel(format!(
                                        "source partition unit ordinal {ordinal} has no canonical address"
                                    ))
                                })?;
                                <A as LayeredArchitecture<
                                    MlxNeuralBackend,
                                    MlxHybridState,
                                >>::build_unit(
                                    source,
                                    address.group(),
                                    address.index(),
                                    context,
                                )
                                .map_err(|error| Error::ArchitectureModel(error.to_string()))
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?;
                mechanisms.prepare_materialization(
                    &mut architecture,
                    &layout,
                    &mut units,
                    source_architecture.as_mut(),
                    source_units.as_deref_mut(),
                    &tasks,
                    &[],
                    context,
                )?;
                let (execution_policy, bounded_policy) = match selected_residency {
                    eredu_runtime::LayerWeightResidency::FullyResident => (
                        mechanisms.resident_policy(&mut architecture, units, selected, context)?,
                        None,
                    ),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                        drop(units);
                        let policy =
                            mechanisms.bounded_policy(&mut architecture, selected, context)?;
                        (policy.clone(), Some(policy))
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "partitioned execution selected an unknown weight residency".into(),
                        ));
                    }
                };
                let local_state = partition.state().ok_or_else(|| {
                    Error::ArchitectureModel("direct partition has no state".into())
                })?;
                let selected_state = selected
                    .state()
                    .for_partitioned_geometry(local_state)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let rank = eredu_core::cache::CacheRankIdentity::new(
                    prompt_cache_topology.stage().map(|(_, rank)| rank),
                    prompt_cache_topology.shard().map(|(_, rank)| rank),
                    prompt_cache_topology.addressable().map(|(_, rank)| rank),
                );
                mechanisms.set_state_partition(rank, local_state.global_layer_offset());
                let state = <MlxHybridState as MlxStateMechanisms>::realize(
                    &selected_state,
                    Some(rank),
                    local_state.global_layer_offset(),
                )?;
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("direct partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest.clone(),
                        Some(tensor_group),
                        tensor_group,
                    )?;
                let parallel = parallel.ok_or_else(|| {
                    Error::Parallel("direct partition has no realized tensor group".into())
                })?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let layerwise =
                    eredu_runtime::LayerwiseRuntime::new(architecture, execution_policy);
                let executor =
                    eredu_architectures::partitioned_execution::DirectPartitionExecutor::new(
                        layerwise, parallel,
                    );
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::NoBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        MlxDirectPartitionStrategy::<A>::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization = materialization_slot
        .lock()
        .map_err(|_| Error::ArchitectureModel("materialization report lock was poisoned".into()))?
        .clone();
    finalizer.finish(CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        materialization,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    ))
}

fn bind_partitioned_routed_resident<A, S, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    match prepared.bank_residency() {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_gated_product_provider()
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                None,
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                return bind_partitioned_routed_with_provider(
                    prepared,
                    store,
                    distributed,
                    eredu_architectures::EmptyPartitionRoutedExpertProvider,
                    None,
                    additional_claimed_sources,
                    stream,
                    weights_stream,
                    finalizer,
                );
            }
            let (selected_member_bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                Arc::clone(&store),
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            let provider = prepared
                .addressable_gated_product_provider(
                    selected_member_bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    options,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                Some(bank),
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        _ => Err(Error::ArchitectureModel(
            "neutral routed partition selected an unsupported expert-bank residency".into(),
        )),
    }
}

fn bind_partitioned_relu2_resident<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::Boundary,
        eredu_nn::GroupedRelu2Spec,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    match prepared.bank_residency() {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_relu2_provider()
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                None,
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                let provider = eredu_architectures::EmptyPartitionRoutedExpertProvider;
                return bind_partitioned_routed_with_provider(
                    prepared,
                    store,
                    distributed,
                    provider,
                    None,
                    additional_claimed_sources,
                    stream,
                    weights_stream,
                    finalizer,
                );
            }
            let (selected_member_bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                Arc::clone(&store),
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            let provider = prepared
                .addressable_relu2_provider(
                    selected_member_bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    options,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                Some(bank),
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        _ => Err(Error::ArchitectureModel(
            "neutral ReLU-squared partition selected an unsupported expert-bank residency".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_partitioned_routed_with_provider<A, S, G, E, Provider, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
        E,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    provider: Provider,
    parameter_bank: Option<MlxSharedAddressableBank>,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: eredu_architectures::partitioned_execution::RoutedCollectiveSpec
        + eredu_architectures::routed_text::RoutedGroupedSpec
        + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    prepared.dispatch_execution(
        (
            store,
            distributed,
            provider,
            parameter_bank,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |prepared, (store, distributed, provider, bank, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_routed_local_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
        |prepared, (store, distributed, provider, bank, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_routed_pipeline_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_partitioned_routed_local_with_provider<A, S, G, E, Provider, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
        E,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    provider: Provider,
    parameter_bank: Option<MlxSharedAddressableBank>,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: eredu_architectures::routed_text::RoutedGroupedSpec + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().text().residency();
    let execution_plan = prepared.execution_handoff().execution_plan().clone();
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel("routed partition has no publication authority".into())
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("routed partition has no state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut ignored_expert_sources = prepared.unowned_expert_checkpoint_sources();
    ignored_expert_sources.extend(additional_claimed_sources);
    let addressable_parameters = if matches!(
        prepared.bank_residency(),
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        prepared.addressable_logical_targets()
    } else {
        std::collections::BTreeSet::new()
    };
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mut mechanisms = MlxReplicatedTextMechanisms::new(
        store,
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    mechanisms.set_ignored_checkpoint_sources(ignored_expert_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    let mut provider = Some(provider);
    #[cfg(test)]
    {
        super::path_instrumentation::constructor();
        super::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, physical_layout, selected, execution, context| {
                let (mut architecture, partition, manifest, tasks) = input.into_parts();
                mechanisms.set_parallel_layout(physical_layout);
                let global_layout = partition.unit_layout().clone();
                let addresses = partition.units().collect::<Vec<_>>();
                partition_local_materialization_tasks(&tasks, &global_layout, &addresses)?;
                let units = addresses
                    .iter()
                    .map(|address| {
                        <A as LayeredArchitecture<MlxNeuralBackend, S>>::build_unit(
                            &architecture,
                            address.group(),
                            address.index(),
                            context,
                        )
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                mechanisms.prepare_local_partition_materialization_with_addressable_parameters(
                    &architecture,
                    None,
                    &global_layout,
                    &addresses,
                    &units,
                    None,
                    None,
                    &tasks,
                    &addressable_parameters,
                )?;
                let (execution_policy, bounded_policy) = match selected_residency {
                    eredu_runtime::LayerWeightResidency::FullyResident => (
                        mechanisms.resident_policy(&mut architecture, units, selected, context)?,
                        None,
                    ),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                        drop(units);
                        let policy =
                            mechanisms.bounded_policy(&mut architecture, selected, context)?;
                        (policy.clone(), Some(policy))
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "routed partition selected an unknown ordinary weight residency".into(),
                        ));
                    }
                };
                let local_state = partition.state().ok_or_else(|| {
                    Error::ArchitectureModel("routed partition has no local state".into())
                })?;
                let selected_state = selected
                    .state()
                    .for_partitioned_geometry(local_state)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let rank = eredu_core::cache::CacheRankIdentity::new(
                    prompt_cache_topology.stage().map(|(_, rank)| rank),
                    prompt_cache_topology.shard().map(|(_, rank)| rank),
                    prompt_cache_topology.addressable().map(|(_, rank)| rank),
                );
                mechanisms.set_state_partition(rank, local_state.global_layer_offset());
                let state = <S as MlxStateMechanisms>::realize(
                    &selected_state,
                    Some(rank),
                    local_state.global_layer_offset(),
                )?;
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("routed partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest.clone(),
                        execution.communication_tensor_group(),
                        execution.sampling_group(),
                    )?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor = execution
                    .local_executor(
                        eredu_runtime::LayerwiseRuntime::new(architecture, execution_policy),
                        parallel,
                        provider.take().ok_or_else(|| {
                            Error::ArchitectureModel("routed provider was already consumed".into())
                        })?,
                        super::distributed::expert::MlxExpertRouteTensorMovement::new(context),
                    )
                    .map_err(Error::ArchitectureModel)?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::NoBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        eredu_runtime::PartitionedTextExecution::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization = materialization_slot
        .lock()
        .map_err(|_| Error::ArchitectureModel("materialization report lock was poisoned".into()))?
        .clone();
    let mut completed = CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        materialization,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    );
    if let Some(bank) = parameter_bank {
        completed = completed.with_parameter_bank(bank);
    }
    finalizer.finish(completed)
}

#[allow(clippy::too_many_arguments)]
fn bind_partitioned_routed_pipeline_with_provider<A, S, G, E, Provider, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
        E,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    provider: Provider,
    parameter_bank: Option<MlxSharedAddressableBank>,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: eredu_architectures::partitioned_execution::RoutedCollectiveSpec
        + eredu_architectures::routed_text::RoutedGroupedSpec
        + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().text().residency();
    let activation_dtype = prepared.execution_handoff().activation_dtype();
    let execution_plan = prepared.execution_handoff().execution_plan().clone();
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel("routed pipeline has no publication authority".into())
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("routed pipeline has no local state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut ignored_expert_sources = prepared.unowned_expert_checkpoint_sources();
    ignored_expert_sources.extend(additional_claimed_sources);
    let addressable_parameters = if matches!(
        prepared.bank_residency(),
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        prepared.addressable_logical_targets()
    } else {
        std::collections::BTreeSet::new()
    };
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mut mechanisms = MlxReplicatedTextMechanisms::new(
        store,
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    mechanisms.set_ignored_checkpoint_sources(ignored_expert_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    let mut provider = Some(provider);
    #[cfg(test)]
    {
        super::path_instrumentation::constructor();
        super::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, physical_layout, selected, execution, context| {
                let (mut architecture, partition, manifest, tasks) = input.into_parts();
                mechanisms.set_parallel_layout(physical_layout);
                let global_layout = partition.unit_layout().clone();
                let addresses = partition.units().collect::<Vec<_>>();
                partition_local_materialization_tasks(&tasks, &global_layout, &addresses)?;
                let mut units = addresses
                    .iter()
                    .map(|address| {
                        <A as LayeredArchitecture<MlxNeuralBackend, S>>::build_unit(
                            &architecture,
                            address.group(),
                            address.index(),
                            context,
                        )
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                mechanisms.prepare_local_partition_materialization_with_addressable_parameters(
                    &architecture,
                    None,
                    &global_layout,
                    &addresses,
                    &units,
                    None,
                    None,
                    &tasks,
                    &addressable_parameters,
                )?;
                let (execution_policy, bounded_policy) = match selected_residency {
                    eredu_runtime::LayerWeightResidency::FullyResident => (
                        mechanisms.resident_policy(
                            &mut architecture,
                            std::mem::take(&mut units),
                            selected,
                            context,
                        )?,
                        None,
                    ),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                        drop(units);
                        let policy =
                            mechanisms.bounded_policy(&mut architecture, selected, context)?;
                        (policy.clone(), Some(policy))
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "routed pipeline selected an unknown ordinary weight residency"
                                .into(),
                        ));
                    }
                };
                let local_state = partition.state().ok_or_else(|| {
                    Error::ArchitectureModel("routed pipeline has no local state".into())
                })?;
                let selected_state = selected
                    .state()
                    .for_partitioned_geometry(local_state)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let rank = eredu_core::cache::CacheRankIdentity::new(
                    prompt_cache_topology.stage().map(|(_, rank)| rank),
                    prompt_cache_topology.shard().map(|(_, rank)| rank),
                    prompt_cache_topology.addressable().map(|(_, rank)| rank),
                );
                mechanisms.set_state_partition(rank, local_state.global_layer_offset());
                let state = <S as MlxStateMechanisms>::realize(
                    &selected_state,
                    Some(rank),
                    local_state.global_layer_offset(),
                )?;
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("routed pipeline communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest,
                        execution.communication_tensor_group(),
                        execution.sampling_group(),
                    )?;
                let parallel = execution
                    .select_parallel(parallel)
                    .map_err(Error::ArchitectureModel)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let provider = provider.take().ok_or_else(|| {
                    Error::ArchitectureModel("routed provider was already consumed".into())
                })?;
                let movement =
                    super::distributed::expert::MlxExpertRouteTensorMovement::new(context);
                let unit_strategy = execution
                    .pipeline_unit_strategy(provider, movement)
                    .map_err(Error::ArchitectureModel)?;
                let executor = eredu_architectures::partitioned_execution::PipelinePartitionExecutor::new_with_unit_strategy(
                    architecture,
                    execution_policy,
                    addresses,
                    parallel,
                    MlxPartitionTensorAllocator,
                    activation_dtype,
                    unit_strategy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        eredu_runtime::PartitionedTextExecution::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization = materialization_slot
        .lock()
        .map_err(|_| Error::ArchitectureModel("materialization report lock was poisoned".into()))?
        .clone();
    let mut completed = CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        materialization,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    );
    if let Some(bank) = parameter_bank {
        completed = completed.with_parameter_bank(bank);
    }
    finalizer.finish(completed)
}

fn bind_partitioned_pipeline<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().residency();
    let activation_dtype = prepared.prepared().selected().activation_dtype();
    let tensor_group = prepared.prepared().selected().tensor_group();
    let session_group = prepared
        .prepared()
        .selected()
        .session_group()
        .ok_or_else(|| {
            Error::ArchitectureModel("pipeline partition has no selected session group".into())
        })?;
    let execution_plan = prepared
        .prepared()
        .selected()
        .pipeline_execution_plan()
        .map_err(Error::ArchitectureModel)?;
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "pipeline resident execution has no selected output publication authority".into(),
            )
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("pipeline partition has no state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mut mechanisms = MlxReplicatedTextMechanisms::new(
        store,
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    mechanisms.set_ignored_checkpoint_sources(additional_claimed_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    #[cfg(test)]
    {
        super::path_instrumentation::constructor();
        super::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, source_architecture, physical_layout, selected, context| {
                let (mut architecture, partition, manifest, tasks) = input.into_parts();
                let (source_architecture, source_layout) = source_architecture
                    .map_or((None, None), |(architecture, layout)| {
                        (Some(architecture), Some(layout))
                    });
                mechanisms.set_parallel_layout(physical_layout);
                let global_layout = partition.unit_layout().clone();
                let addresses = partition.units().collect::<Vec<_>>();
                partition_local_materialization_tasks(&tasks, &global_layout, &addresses)?;
                let mut units = addresses
                    .iter()
                    .map(|address| {
                        <A as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::build_unit(
                            &architecture,
                            address.group(),
                            address.index(),
                            context,
                        )
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let source_units = source_architecture
                    .as_ref()
                    .map(|source| {
                        addresses
                            .iter()
                            .map(|address| {
                                <A as LayeredArchitecture<
                                    MlxNeuralBackend,
                                    MlxHybridState,
                                >>::build_unit(
                                    source,
                                    address.group(),
                                    address.index(),
                                    context,
                                )
                                .map_err(|error| Error::ArchitectureModel(error.to_string()))
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?;
                mechanisms.prepare_local_partition_materialization(
                    &architecture,
                    source_architecture.as_ref(),
                    &global_layout,
                    &addresses,
                    &units,
                    source_units.as_deref(),
                    source_layout.as_ref(),
                    &tasks,
                )?;
                let (execution_policy, bounded_policy) = match selected_residency {
                    eredu_runtime::LayerWeightResidency::FullyResident => (
                        mechanisms.resident_policy(
                            &mut architecture,
                            std::mem::take(&mut units),
                            selected,
                            context,
                        )?,
                        None,
                    ),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                        drop(units);
                        let policy =
                            mechanisms.bounded_policy(&mut architecture, selected, context)?;
                        (policy.clone(), Some(policy))
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "partitioned execution selected an unknown weight residency".into(),
                        ));
                    }
                };
                let local_state = partition.state().ok_or_else(|| {
                    Error::ArchitectureModel("pipeline partition has no state".into())
                })?;
                let selected_state = selected
                    .state()
                    .for_partitioned_geometry(local_state)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let rank = eredu_core::cache::CacheRankIdentity::new(
                    prompt_cache_topology.stage().map(|(_, rank)| rank),
                    prompt_cache_topology.shard().map(|(_, rank)| rank),
                    prompt_cache_topology.addressable().map(|(_, rank)| rank),
                );
                mechanisms.set_state_partition(rank, local_state.global_layer_offset());
                let state = <MlxHybridState as MlxStateMechanisms>::realize(
                    &selected_state,
                    Some(rank),
                    local_state.global_layer_offset(),
                )?;
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("pipeline partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(manifest, tensor_group, session_group)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor =
                    eredu_architectures::partitioned_execution::PipelinePartitionExecutor::new(
                        architecture,
                        execution_policy,
                        addresses,
                        parallel,
                        MlxPartitionTensorAllocator,
                        activation_dtype,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        MlxPipelinePartitionStrategy::<A, MlxHybridState>::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization = materialization_slot
        .lock()
        .map_err(|_| Error::ArchitectureModel("materialization report lock was poisoned".into()))?
        .clone();
    finalizer.finish(CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        materialization,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    ))
}

struct PartitionedCompositeBindingVisitor<'a> {
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

trait CompositeExecutableFinalizer<A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static;
}

impl<A> CompositeExecutableFinalizer<A> for OrdinaryReplicatedFinalizer
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        Ok(Box::new(completed))
    }
}

impl<A, P> CompositeExecutableFinalizer<A> for PredictionReplicatedFinalizer<P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        completed
            .with_prediction(self.prediction, self.capability)
            .map(|completed| Box::new(completed) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedCompositeBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

pub(crate) struct PartitionedCompositePredictionBindingVisitor<'a> {
    pub store: Arc<dyn CheckpointSource>,
    pub distributed: crate::backend::distributed::MlxDistributedSession,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionPredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PartitionedCompositePredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            PredictionReplicatedFinalizer {
                prediction: SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                capability: self.capability,
            },
        )
    }
}

fn bind_prepared_partitioned_composite<A, G, W, F>(
    prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    F: CompositeExecutableFinalizer<A>,
{
    let execution_plan = prepared
        .execution_plan()
        .map_err(Error::ArchitectureModel)?;
    let publication_authority = execution_plan
        .publication_authority(prepared.communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "composite partition has no selected output publication authority".into(),
            )
        })?;
    let selected_residency = prepared
        .prepared()
        .selected()
        .base()
        .execution()
        .residency();
    let session_group = prepared
        .prepared()
        .selected()
        .session_group()
        .ok_or_else(|| {
            Error::ArchitectureModel("composite partition has no selected session group".into())
        })?;
    let prompt_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let local_state = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| {
            Error::ArchitectureModel("composite partition has no selected local state".into())
        })?;
    let prompt_cache_identity = local_state
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let admission = prepared.prepared().architecture().admission_config();
    let processor = prepared.prepared().selected().base().processor().clone();
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let materialization_slot = Arc::new(std::sync::Mutex::new(None));
    let mut mechanisms: MlxReplicatedTextMechanisms<
        PreparedCompositeArchitecture<A>,
        MlxHybridState,
    > = MlxReplicatedTextMechanisms::new(
        store,
        Arc::clone(&materialization_slot),
        stream,
        weights_stream,
    );
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    #[cfg(test)]
    {
        super::path_instrumentation::constructor();
        super::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime::<MlxNeuralBackend, _, _, _, _>(
            prompt_topology.clone(),
            stream,
            |input, physical_layout, executor_plan, selected, context| {
                let tensor_group = executor_plan.communication_tensor_group();
                let (mut architecture, partition, manifest, tasks) = input.into_parts();
                mechanisms.set_parallel_layout(physical_layout);
                let global_layout = partition.unit_layout().clone();
                let addresses = partition.units().collect::<Vec<_>>();
                let mut units = addresses
                    .iter()
                    .map(|address| {
                        architecture
                            .build_unit(address.group(), address.index(), context)
                            .map_err(|error| Error::ArchitectureModel(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                mechanisms.prepare_local_partition_materialization(
                    &architecture,
                    None,
                    &global_layout,
                    &addresses,
                    &units,
                    None,
                    None,
                    &tasks,
                )?;
                let (execution_policy, bounded_policy) = match selected_residency {
                    eredu_runtime::LayerWeightResidency::FullyResident => (
                        mechanisms.resident_policy(
                            &mut architecture,
                            std::mem::take(&mut units),
                            selected,
                            context,
                        )?,
                        None,
                    ),
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                        drop(units);
                        let policy =
                            mechanisms.bounded_policy(&mut architecture, selected, context)?;
                        (policy.clone(), Some(policy))
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "composite partition selected an unknown weight residency".into(),
                        ));
                    }
                };
                let local_state = partition.state().ok_or_else(|| {
                    Error::ArchitectureModel(
                        "composite partition has no selected local state".into(),
                    )
                })?;
                let selected_state = selected
                    .state()
                    .for_partitioned_geometry(local_state)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let rank = eredu_core::cache::CacheRankIdentity::new(
                    prompt_topology.stage().map(|(_, rank)| rank),
                    prompt_topology.shard().map(|(_, rank)| rank),
                    prompt_topology.addressable().map(|(_, rank)| rank),
                );
                mechanisms.set_state_partition(rank, local_state.global_layer_offset());
                let state = <MlxHybridState as MlxStateMechanisms>::realize(
                    &selected_state,
                    Some(rank),
                    local_state.global_layer_offset(),
                )?;
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("composite communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(manifest, tensor_group, session_group)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor = executor_plan.bind(
                    architecture.into_inner(),
                    execution_policy,
                    parallel,
                    MlxPartitionTensorAllocator,
                    super::distributed::expert::MlxExpertRouteTensorMovement::new(stream),
                )?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        eredu_runtime::PartitionedTextExecution::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let materialization = materialization_slot
        .lock()
        .map_err(|_| Error::ArchitectureModel("materialization report lock was poisoned".into()))?
        .clone();
    finalizer.finish(
        CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
            session,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            selected_residency,
            partition_sampling_group,
            partition_communication_authority,
            Some(publication_authority.owner_group_rank()),
            publication_authority.local_public_output(),
            stream,
        ),
    )
}

pub(crate) fn bind_partitioned_composite(
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        eredu_architectures::replicated_text::CompositeTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::composite_partitioned::visit_authoritative_composite_partition::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        selected,
        stream,
        PartitionedCompositeBindingVisitor {
            store,
            distributed,
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(crate) fn bind_partitioned_dense_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        SelectedReplicatedTextRealization,
        ReplicatedTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        inspection,
        selected,
        store,
        stream,
        PartitionedDenseDecoderBindingVisitor {
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(crate) fn bind_partitioned_routed_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::dispatch_routed_partitioned_production(
        inspection,
        selected,
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        ),
        |(store, distributed, additional, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxHybridState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedRoutedDecoderBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(store, distributed, additional, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_relu2_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxHybridState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedRoutedDecoderBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(store, distributed, _, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_pooling_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedPoolingRoutedDecoderBindingVisitor {
                    distributed,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_partitioned_routed_prediction_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    extension: MaterializedEmbeddedPrediction,
    realization: eredu_runtime::SelectedSpeculativeRealization,
    capability: eredu_architectures::capability::CapabilityEstimate,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::dispatch_routed_partitioned_production(
        inspection,
        selected,
        (
            extension,
            realization,
            capability,
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        ),
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxHybridState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_relu2_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxHybridState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_pooling_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
    )
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        super::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl ReplicatedTextProfileDispatcher<MlxNeuralBackend> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;
    type StatelessState = MlxHybridState;
    type AttentionState = MlxKeyValueState;
    type ComponentState = MlxHybridState;
    type AttentionComponentState = MlxHybridState;
    type CompressedState = MlxHybridState;
    type CompressedComponentState = MlxHybridState;
    type StatelessVisitor = Self;
    type AttentionVisitor = Self;
    type ComponentVisitor = Self;
    type AttentionComponentVisitor = Self;
    type CompressedVisitor = Self;
    type CompressedComponentVisitor = Self;

    fn into_stateless_visitor(self) -> Self::StatelessVisitor {
        self
    }

    fn into_attention_visitor(self) -> Self::AttentionVisitor {
        self
    }

    fn into_component_visitor(self) -> Self::ComponentVisitor {
        self
    }

    fn into_attention_component_visitor(self) -> Self::AttentionComponentVisitor {
        self
    }

    fn into_compressed_visitor(self) -> Self::CompressedVisitor {
        self
    }

    fn into_compressed_component_visitor(self) -> Self::CompressedComponentVisitor {
        self
    }
}

fn valid_packed_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    if descriptor.executable() != LinearFormat::Dense
        && descriptor.packed_axis() != descriptor.logical_shape().len().checked_sub(1)
    {
        return false;
    }
    let Some(extent) = descriptor.packed_extent() else {
        return false;
    };
    match descriptor.executable() {
        LinearFormat::Affine(format) => usize::try_from(format.group_size)
            .ok()
            .is_some_and(|group| group != 0 && group <= extent && extent.is_multiple_of(group)),
        LinearFormat::MxFp4 => extent.is_multiple_of(32),
        LinearFormat::GgufIQuant { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| extent.is_multiple_of(block)),
        LinearFormat::E4M3BlockFp8(_) => true,
        LinearFormat::Dense => true,
    }
}

fn valid_direct_source_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    let same_unpacked_dimensions = |packed_axis: usize| {
        descriptor
            .physical_shape()
            .iter()
            .zip(descriptor.logical_shape())
            .enumerate()
            .all(|(axis, (physical, logical))| axis == packed_axis || physical == logical)
    };
    match descriptor.source() {
        SourceTensorEncoding::Gguf { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| match descriptor.packed_axis() {
                Some(axis) if same_unpacked_dimensions(axis) => {
                    let physical = descriptor.physical_shape()[axis];
                    let logical = descriptor.logical_shape()[axis];
                    physical >= logical
                        && physical.is_multiple_of(block)
                        && physical - logical < block
                }
                Some(_) => false,
                None => descriptor.physical_shape() == descriptor.logical_shape(),
            }),
        SourceTensorEncoding::Safetensors(StoredDtype::U8)
            if descriptor.executable() == LinearFormat::MxFp4 =>
        {
            descriptor.physical_shape() == descriptor.logical_shape()
                || descriptor.packed_axis().is_some_and(|axis| {
                    let physical = descriptor.physical_shape();
                    let logical = descriptor.logical_shape();
                    physical.len() == logical.len() + 1
                        && physical.last() == Some(&16)
                        && physical[..axis] == logical[..axis]
                        && physical[axis].checked_mul(32) == Some(logical[axis])
                        && physical[axis + 1..physical.len() - 1] == logical[axis + 1..]
                })
        }
        SourceTensorEncoding::Safetensors(StoredDtype::U32) => {
            let Some(axis) = descriptor.packed_axis() else {
                return false;
            };
            if !same_unpacked_dimensions(axis) {
                return false;
            }
            let bits = match descriptor.executable() {
                LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                LinearFormat::MxFp4 => Some(4),
                _ => None,
            };
            bits.is_some_and(|bits| {
                descriptor.physical_shape()[axis].checked_mul(32)
                    == descriptor.logical_shape()[axis].checked_mul(bits)
            }) && valid_packed_geometry(descriptor)
        }
        SourceTensorEncoding::RecipeOutput(dtype) => {
            let equivalent = WeightLoweringDescriptor::new(
                SourceTensorEncoding::Safetensors(dtype.clone()),
                descriptor.executable(),
                descriptor.physical_shape().to_vec(),
                descriptor.logical_shape().to_vec(),
                descriptor.packed_axis(),
            )
            .expect("validated recipe-output descriptor remains valid");
            valid_direct_source_geometry(&equivalent)
        }
        _ => {
            descriptor.physical_shape() == descriptor.logical_shape()
                && (descriptor.executable() == LinearFormat::Dense
                    || valid_packed_geometry(descriptor))
        }
    }
}

pub(crate) fn supports_direct(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let supported = match (source, executable) {
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32,
            ),
            LinearFormat::Dense,
        ) => true,
        (
            SourceTensorEncoding::RecipeOutput(
                StoredDtype::F16
                | StoredDtype::BF16
                | StoredDtype::F32
                | StoredDtype::U8
                | StoredDtype::I32
                | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::U8 | StoredDtype::I32 | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U8), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U8), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3),
            LinearFormat::E4M3BlockFp8(format),
        ) => format.validate().is_ok(),
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Dense) => {
            matches!(
                ggml_type,
                eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
            ) || gguf_affine(*ggml_type).is_some()
                || *ggml_type == eredu_gguf::GgmlType::MxFp4
                || NativeQuantizationFormat::from_ggml_type(*ggml_type).is_some()
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Affine(format)) => {
            gguf_affine(*ggml_type).is_some_and(|native| native == format)
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::MxFp4) => {
            *ggml_type == eredu_gguf::GgmlType::MxFp4
        }
        (
            SourceTensorEncoding::Gguf { ggml_type, endian },
            LinearFormat::GgufIQuant {
                ggml_type: executable,
                endian: executable_endian,
            },
        ) => {
            *ggml_type == executable
                && *endian == executable_endian
                && NativeQuantizationFormat::from_ggml_type(executable).is_some()
        }
        _ => false,
    };
    supported && valid_direct_source_geometry(descriptor)
}

pub(crate) fn supports_transform(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let decodable = match source {
        SourceTensorEncoding::Safetensors(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::RecipeOutput(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::Gguf { ggml_type, .. } => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        _ => false,
    };
    decodable
        && descriptor.physical_shape() == descriptor.logical_shape()
        && descriptor.packed_axis() == descriptor.logical_shape().len().checked_sub(1)
        && valid_packed_geometry(descriptor)
        && match executable {
            LinearFormat::Affine(format) => format.validate().is_ok(),
            LinearFormat::MxFp4 => true,
            LinearFormat::Dense
            | LinearFormat::GgufIQuant { .. }
            | LinearFormat::E4M3BlockFp8(_) => false,
        }
}

fn gguf_affine(ggml_type: eredu_gguf::GgmlType) -> Option<eredu_checkpoint::AffineQuantization> {
    let (bits, group_size) = match ggml_type {
        eredu_gguf::GgmlType::Q2K => (2, 16),
        eredu_gguf::GgmlType::Q3K => (3, 16),
        eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::Q4_1 | eredu_gguf::GgmlType::Q4K => {
            (4, 32)
        }
        eredu_gguf::GgmlType::Q5_0 | eredu_gguf::GgmlType::Q5_1 | eredu_gguf::GgmlType::Q5K => {
            (5, 32)
        }
        eredu_gguf::GgmlType::Q6K => (6, 16),
        eredu_gguf::GgmlType::Q8_0 => (8, 32),
        _ => return None,
    };
    eredu_checkpoint::AffineQuantization::new(group_size, bits).ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::backend::ExecutionContext;
    use eredu_checkpoint::SourceTensorEncoding;
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ModelConfigurationResolver,
    };
    use eredu_runtime::{
        ParameterTransformConstraint, ReplicatedTextParameterRequirement,
        ReplicatedTextStateAccess, StateLayout, WeightLoweringDescriptor,
    };
    use safemlx::{Device, DeviceType};

    fn prediction_cache_manager() -> CacheResidencyManager {
        CacheResidencyManager::new(
            PagedCacheOptions::new(1, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap()
    }

    #[test]
    fn prediction_target_forks_preserve_kv_and_compressed_paging_sessions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let key_value_layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let canonical =
            MlxKeyValueState::paged(key_value_layout, prediction_cache_manager(), None).unwrap();
        let canonical_checkpoint = canonical.deep_checkpoint().unwrap();
        let mut fork = fork_mlx_prediction_target_state(&canonical, stream).unwrap();
        let fork_checkpoint = fork.deep_checkpoint().unwrap();
        fork.restore_checkpoint(&fork_checkpoint, stream).unwrap();
        assert_eq!(fork.offset(), canonical.offset());
        assert!(fork
            .restore_checkpoint(&canonical_checkpoint, stream)
            .unwrap_err()
            .to_string()
            .contains("does not belong to the same paged layer"));

        let compressed_layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![
                    LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 8, 4)
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let canonical =
            MlxHybridState::paged(compressed_layout, prediction_cache_manager(), None).unwrap();
        let canonical_checkpoint = canonical.deep_checkpoint().unwrap();
        let mut fork = fork_mlx_prediction_target_state(&canonical, stream).unwrap();
        let fork_checkpoint = fork.deep_checkpoint().unwrap();
        fork.restore_checkpoint(&fork_checkpoint, stream).unwrap();
        assert_eq!(fork.offset(), canonical.offset());
        assert!(fork
            .restore_checkpoint(&canonical_checkpoint, stream)
            .unwrap_err()
            .to_string()
            .contains("does not belong to the same paged layer"));
    }

    #[test]
    fn exact_local_transform_output_is_not_sharded_twice() {
        use eredu_checkpoint::store::{MemoryWeightStore, TensorSelection};
        use eredu_runtime::{LocalModelLayout, LocalTensorLayout, ParameterRole, TensorPlacement};

        let store = MemoryWeightStore::from_safetensors([
            (
                "transformed.weight".to_owned(),
                safetensors::Dtype::F32,
                vec![2, 2],
                vec![0; 2 * 2 * size_of::<f32>()],
            ),
            (
                "untouched.weight".to_owned(),
                safetensors::Dtype::F32,
                vec![4, 2],
                vec![0; 4 * 2 * size_of::<f32>()],
            ),
        ])
        .unwrap();
        let mut layout = LocalModelLayout::default();
        for name in ["transformed.weight", "untouched.weight"] {
            layout.insert(
                name.to_owned(),
                LocalTensorLayout::new(
                    "projection",
                    ParameterRole::ColumnProjection,
                    vec![4, 2],
                    vec![2, 2],
                    TensorPlacement::Shard {
                        axis: 0,
                        index: 0,
                        parts: 2,
                    },
                    None,
                    None,
                    false,
                ),
            );
        }
        let binding = |name: &str, bytes| {
            WeightBinding::new(name, name, TensorSelection::Full, bytes)
                .unwrap()
                .with_logical_target(name)
                .unwrap()
        };
        let bindings = vec![
            binding("transformed.weight", 16),
            binding("untouched.weight", 32),
        ];
        let output = shard_unmaterialized_bindings(
            bindings,
            &store,
            &layout,
            &["transformed.weight".to_owned()].into_iter().collect(),
        )
        .unwrap();

        assert_eq!(
            output[0].source_recipe().infer(&store).unwrap().shape(),
            [2, 2]
        );
        assert_eq!(
            output[1].source_recipe().infer(&store).unwrap().shape(),
            [2, 2]
        );
        assert_eq!(output[0].expected_bytes(), 16);
        assert_eq!(output[1].expected_bytes(), 16);
    }

    struct ForgedShapeSource {
        inner: eredu_checkpoint::store::SharedCheckpointSource,
        key: String,
    }

    impl eredu_checkpoint::store::CheckpointSource for ForgedShapeSource {
        fn source_keys(&self) -> Vec<String> {
            self.inner.source_keys()
        }

        fn source_metadata(
            &self,
            key: &str,
        ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError>
        {
            let mut metadata = self.inner.source_metadata(key)?;
            if key == self.key {
                metadata.physical_shape.push(2);
            }
            Ok(metadata)
        }

        fn acquire_lease(
            &self,
            request: eredu_checkpoint::store::TensorReadRequest,
        ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError>
        {
            self.inner.acquire_lease(request)
        }

        fn source_diagnostics(
            &self,
        ) -> Result<
            eredu_checkpoint::store::WeightStoreDiagnostics,
            eredu_checkpoint::store::StoreError,
        > {
            self.inner.source_diagnostics()
        }

        fn source_provenance(
            &self,
            key: &str,
        ) -> Result<
            eredu_checkpoint::store::TensorSourceProvenance,
            eredu_checkpoint::store::StoreError,
        > {
            self.inner.source_provenance(key)
        }
    }

    fn materialize_model_plan(
        plan: eredu_core::ModelPreparationPlan<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: crate::MlxLoadRequest,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<crate::backend::MlxModel, crate::backend::error::Error> {
        let selected =
            super::super::loading::select_preparation(plan.inspection(), options, plan.policy())?;
        super::super::loading::materialize_model_plan(plan, selected, None, stream, weights_stream)
    }

    #[test]
    fn exact_lowering_rejects_unsupported_encodings_and_incoherent_physical_geometry() {
        let affine =
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let gguf = |physical_shape| {
            WeightLoweringDescriptor::new(
                SourceTensorEncoding::Gguf {
                    ggml_type: eredu_gguf::GgmlType::Q4_0,
                    endian: eredu_gguf::Endian::Little,
                },
                affine,
                physical_shape,
                vec![64, 8],
                Some(1),
            )
            .unwrap()
        };
        assert!(supports_direct(&gguf(vec![64, 32])));
        assert!(!supports_direct(&gguf(vec![63, 32])));
        assert!(!supports_direct(&gguf(vec![64, 31])));
        assert!(!supports_direct(&gguf(vec![64, 64])));

        let safetensors = |source, physical_shape| {
            WeightLoweringDescriptor::new(source, affine, physical_shape, vec![64, 64], Some(1))
                .unwrap()
        };
        assert!(supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            vec![64, 8],
        )));
        assert!(!supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            vec![64, 7],
        )));
        assert!(!supports_direct(&safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::U8),
            vec![64, 64],
        )));

        let transform = safetensors(
            SourceTensorEncoding::Safetensors(StoredDtype::F16),
            vec![64, 32],
        );
        assert!(!supports_transform(&transform));
        let non_final_axis = WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::F16),
            affine,
            vec![64, 64],
            vec![64, 64],
            Some(0),
        )
        .unwrap();
        assert!(!supports_direct(&non_final_axis));
        assert!(!supports_transform(&non_final_axis));
        let final_axis = WeightLoweringDescriptor::new(
            SourceTensorEncoding::RecipeOutput(StoredDtype::F16),
            affine,
            vec![64, 64],
            vec![64, 64],
            Some(1),
        )
        .unwrap();
        assert!(supports_transform(&final_axis));
    }

    #[test]
    fn selected_store_geometry_mismatch_rejects_before_construction_or_payload_open() {
        super::super::path_instrumentation::reset();
        let artifact = tiny_heterogeneous_artifact(lfm2_config());
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .unwrap();
        let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
            eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
        );
        let key = requirements
            .parameters()
            .iter()
            .find_map(|parameter| parameter.sources().first())
            .unwrap()
            .clone();
        let forged: eredu_checkpoint::store::SharedCheckpointSource =
            Arc::new(ForgedShapeSource { inner: source, key });
        let (stream, weights_stream) = execution_streams();
        let error = super::super::loading::bind_replicated_text(
            inspection.architecture_plan(),
            selected,
            forged,
            &stream,
            &weights_stream,
        )
        .err()
        .expect("forged source geometry must fail");
        assert!(error.to_string().contains("physical geometry"));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn selected_graph_mismatch_rejects_before_module_construction() {
        super::super::path_instrumentation::reset();
        let artifact = tiny_artifact("llama", false);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let forged_graph = eredu_runtime::ExecutionGraph::chain(["forged-decoder"]).unwrap();
        let forged_units = eredu_runtime::ExecutionUnitLayout::new(
            &forged_graph,
            [requirements.execution_units().len()],
        )
        .unwrap();
        let forged_requirements = ReplicatedTextRequirements::new(
            requirements.architecture_identity().to_owned(),
            requirements.operators(),
            forged_graph,
            forged_units,
            requirements.group_transports().to_vec(),
            requirements.state_layout().clone(),
            requirements.state_access(),
            requirements.parameters().to_vec(),
        )
        .unwrap()
        .with_derived_recipes(
            requirements.derived_recipes().clone(),
            requirements.derived_recipe_outputs().clone(),
        )
        .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &forged_requirements,
            &request,
            &capabilities(&forged_requirements, &request),
        )
        .unwrap();
        let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
            eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
        );
        let (stream, weights_stream) = execution_streams();
        let error = super::super::loading::bind_replicated_text(
            inspection.architecture_plan(),
            selected,
            source,
            &stream,
            &weights_stream,
        )
        .err()
        .expect("forged graph must fail");
        assert!(error.to_string().contains("structure differs"), "{error}");
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn report_distinguishes_native_and_transforming_lowerings() {
        let parameter = ReplicatedTextParameterRequirement::new(
            "projection.weight",
            vec!["projection.weight".into()],
            vec![eredu_runtime::ReplicatedTextPhysicalSource::new(
                "projection.weight",
                "projection.weight",
                "/checkpoint/model.safetensors",
                "projection.weight",
                SourceTensorEncoding::Safetensors(StoredDtype::F16),
                64 * 64 * 2,
            )
            .unwrap()],
            Vec::new(),
            Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
            Some(vec![64, 64]),
            vec![64, 64],
            LinearFormat::Dense,
            eredu_runtime::ReplicatedTextParameterRole::LinearWeight,
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
                group: "decoder".into(),
                unit: 0,
            },
            eredu_runtime::ReplicatedTextParameterPresence::Required,
            ParameterTransformConstraint::Linear { packed_axis: 1 },
        )
        .unwrap();
        let graph = eredu_runtime::ExecutionGraph::chain(["decoder"]).unwrap();
        let requirements = ReplicatedTextRequirements::new(
            "test.generic-binding",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            eredu_runtime::ReplicatedTextStateAccess::KeyValue,
            vec![parameter],
        )
        .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(eredu_core::QuantizationRequest::Affine {
            group_size: 64,
            bits: 4,
        });
        let report = capabilities(&requirements, &request);
        assert!(report.weight_lowerings().iter().any(|lowering| {
            lowering.executable() == LinearFormat::Dense
                && lowering.kind() == WeightLoweringKind::Direct
        }));
        assert!(report.weight_lowerings().iter().any(|lowering| {
            matches!(lowering.executable(), LinearFormat::Affine(_))
                && lowering.kind() == WeightLoweringKind::Transform
        }));
    }

    pub(crate) fn tiny_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, false, false)
    }

    fn tiny_sharded_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, true, false)
    }

    fn tiny_packed_safetensors_artifact(model_type: &str) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, false, false, true)
    }

    #[test]
    fn dense_decoder_partition_classifier_is_architecture_owned_and_exhaustive() {
        for model_type in ["llama", "qwen2", "qwen3"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            assert!(
                eredu_architectures::partitioned_execution::is_supported_dense_decoder_partition(
                    inspection.architecture_plan(),
                ),
                "{model_type} must enter typed dense-decoder dispatch"
            );
        }
        for model_type in ["qwen3_moe", "gpt_oss"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            assert!(
                !eredu_architectures::partitioned_execution::is_supported_dense_decoder_partition(
                    inspection.architecture_plan(),
                ),
                "{model_type} must remain outside dense-decoder dispatch"
            );
        }
    }

    fn tiny_safetensors_artifact(
        model_type: &str,
        tied: bool,
        sharded: bool,
        packed: bool,
    ) -> tempfile::TempDir {
        use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

        let root = tempfile::tempdir().unwrap();
        let architecture = match model_type {
            "llama" => "LlamaForCausalLM",
            "mistral" => "MistralForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            "gpt_oss" => "GptOssForCausalLM",
            _ => unreachable!(),
        };
        let mut config = serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tied
        });
        if model_type == "qwen3_moe" {
            config["num_experts"] = 2.into();
            config["num_experts_per_tok"] = 1.into();
            config["moe_intermediate_size"] = 32.into();
        }
        if model_type == "gpt_oss" {
            config["num_local_experts"] = 2.into();
            config["num_experts_per_tok"] = 1.into();
            config["sliding_window"] = 16.into();
            config["layer_types"] = serde_json::json!(["sliding_attention"]);
            config["quantization_config"] = serde_json::json!({"quant_method": "mxfp4"});
            config["swiglu_limit"] = 7.0.into();
        }
        if packed {
            config["quantization_config"] = serde_json::json!({ "group_size": 32, "bits": 4 });
        }
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                let (dtype, bytes): (Dtype, Vec<u8>) = if matches!(
                    constraint.dtype,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U32
                    )
                ) {
                    (
                        Dtype::U32,
                        (0..elements).flat_map(|_| 0_u32.to_le_bytes()).collect(),
                    )
                } else if matches!(
                    constraint.dtype,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U8
                    )
                ) {
                    let fill = if constraint.key.ends_with("_scales") {
                        127
                    } else {
                        0
                    };
                    (Dtype::U8, vec![fill; elements])
                } else if constraint.key.ends_with(".A_log") {
                    (
                        Dtype::F32,
                        (-1.0_f32)
                            .to_le_bytes()
                            .into_iter()
                            .cycle()
                            .take(elements * 4)
                            .collect(),
                    )
                } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                    (
                        Dtype::F32,
                        (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                    )
                } else if constraint.role == eredu_checkpoint::schema::TensorRole::Companion {
                    (
                        Dtype::F32,
                        (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                    )
                } else {
                    let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                        value.wrapping_mul(31) ^ u32::from(byte)
                    });
                    (
                        Dtype::F32,
                        (0..elements)
                            .flat_map(|index| {
                                let signed =
                                    i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                                (signed as f32 * 0.001).to_le_bytes()
                            })
                            .collect(),
                    )
                };
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        let write = |path: &Path, tensors: &[(String, Vec<usize>, Dtype, Vec<u8>)]| {
            let views = tensors
                .iter()
                .map(|(name, shape, dtype, bytes)| {
                    (
                        name.as_str(),
                        TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, path).unwrap();
        };
        if sharded {
            let split = tensors.len() / 2;
            let first = "model-00001-of-00002.safetensors";
            let second = "model-00002-of-00002.safetensors";
            write(&root.path().join(first), &tensors[..split]);
            write(&root.path().join(second), &tensors[split..]);
            let weight_map = tensors
                .iter()
                .enumerate()
                .map(|(index, (name, _, _, _))| {
                    (name.clone(), if index < split { first } else { second })
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            std::fs::write(
                root.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
            )
            .unwrap();
        } else {
            write(&root.path().join("model.safetensors"), &tensors);
        }
        root
    }

    fn tiny_heterogeneous_artifact(config: serde_json::Value) -> tempfile::TempDir {
        tiny_heterogeneous_artifact_with_layout(config, false)
    }

    fn tiny_heterogeneous_artifact_with_layout(
        config: serde_json::Value,
        fused_qwen_next: bool,
    ) -> tempfile::TempDir {
        use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| {
                    if fused_qwen_next && group.variants.iter().any(|variant| variant.id == "fused")
                    {
                        group.variants.iter().find(|variant| variant.id == "fused")
                    } else {
                        group.variants.first()
                    }
                })
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                let (dtype, bytes): (Dtype, Vec<u8>) = if matches!(
                    constraint.dtype,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U32
                    )
                ) {
                    (
                        Dtype::U32,
                        (0..elements).flat_map(|_| 0_u32.to_le_bytes()).collect(),
                    )
                } else if matches!(
                    constraint.dtype,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::I32
                    )
                ) {
                    (
                        Dtype::I32,
                        (0..elements).flat_map(|_| 0_i32.to_le_bytes()).collect(),
                    )
                } else if constraint.key.ends_with(".A_log") {
                    (
                        Dtype::F32,
                        (-1.0_f32)
                            .to_le_bytes()
                            .into_iter()
                            .cycle()
                            .take(elements * 4)
                            .collect(),
                    )
                } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                    (
                        Dtype::F32,
                        (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                    )
                } else if constraint.role == eredu_checkpoint::schema::TensorRole::Companion {
                    (
                        Dtype::F32,
                        (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                    )
                } else {
                    let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                        value.wrapping_mul(31) ^ u32::from(byte)
                    });
                    (
                        Dtype::F32,
                        (0..elements)
                            .flat_map(|index| {
                                let signed =
                                    i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                                (signed as f32 * 0.001).to_le_bytes()
                            })
                            .collect(),
                    )
                };
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
        root
    }

    fn tiny_heterogeneous_gguf(family: &str, stream: &Stream) -> crate::test_utils::SyntheticGguf {
        tiny_heterogeneous_gguf_with_packed_qwen_next(family, None, stream)
    }

    fn tiny_heterogeneous_gguf_with_packed_qwen_next(
        family: &str,
        packed_qkvz: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::{MetadataArray, MetadataValue};

        let (plan, metadata) = match family {
            "lfm2" => {
                let args = eredu_architectures::lfm2::model_args_from_config_value(&lfm2_config())
                    .unwrap();
                let key = |suffix: &str| format!("lfm2.{suffix}");
                (
                    eredu_architectures::lfm2::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("lfm2".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(2)),
                        (key("embedding_length"), MetadataValue::Uint32(16)),
                        (
                            key("feed_forward_length"),
                            MetadataValue::Uint32(args.dense_intermediate_size as u32),
                        ),
                        (key("attention.head_count"), MetadataValue::Uint32(4)),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 2])),
                        ),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.norm_eps),
                        ),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (key("shortconv.l_cache"), MetadataValue::Uint32(3)),
                        (
                            key("rope.freq_base"),
                            MetadataValue::Float32(args.rope.theta),
                        ),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                    ]),
                )
            }
            "kimi_linear" => {
                let args = eredu_architectures::kimi_linear::model_args_from_config_value(
                    &kimi_linear_config(),
                )
                .unwrap();
                let key = |suffix: &str| format!("kimi-linear.{suffix}");
                (
                    eredu_architectures::kimi_linear::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("kimi-linear".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(2)),
                        (key("embedding_length"), MetadataValue::Uint32(12)),
                        (key("attention.head_count"), MetadataValue::Uint32(3)),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
                        ),
                        (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                        (key("attention.key_length_mla"), MetadataValue::Uint32(6)),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                        (key("feed_forward_length"), MetadataValue::Uint32(16)),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.rms_norm_eps),
                        ),
                        (key("kda.head_dim"), MetadataValue::Uint32(4)),
                        (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                        (key("expert_count"), MetadataValue::Uint32(2)),
                        (key("expert_feed_forward_length"), MetadataValue::Uint32(8)),
                        (key("attention.kv_lora_rank"), MetadataValue::Uint32(6)),
                        (key("attention.value_length_mla"), MetadataValue::Uint32(4)),
                        (key("leading_dense_block_count"), MetadataValue::Uint32(2)),
                        (key("expert_used_count"), MetadataValue::Uint32(1)),
                        (key("expert_shared_count"), MetadataValue::Uint32(1)),
                    ]),
                )
            }
            "nemotron_h" => {
                let args = eredu_architectures::nemotron_h::model_args_from_config_value(
                    &nemotron_h_config(),
                )
                .unwrap();
                let key = |suffix: &str| format!("nemotron_h.{suffix}");
                (
                    eredu_architectures::nemotron_h::gguf_plan(&args).unwrap(),
                    HashMap::from([
                        (
                            "general.architecture".into(),
                            MetadataValue::String("nemotron_h".into()),
                        ),
                        ("general.file_type".into(), MetadataValue::Uint32(0)),
                        (key("block_count"), MetadataValue::Uint32(4)),
                        (key("embedding_length"), MetadataValue::Uint32(16)),
                        (
                            key("feed_forward_length"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 24, 0])),
                        ),
                        (
                            key("attention.head_count_kv"),
                            MetadataValue::Array(MetadataArray::Uint32(vec![0, 2, 0, 0])),
                        ),
                        (key("attention.head_count"), MetadataValue::Uint32(4)),
                        (key("attention.key_length"), MetadataValue::Uint32(4)),
                        (
                            key("attention.layer_norm_rms_epsilon"),
                            MetadataValue::Float32(args.norm_eps),
                        ),
                        (key("context_length"), MetadataValue::Uint32(64)),
                        (key("ssm.inner_size"), MetadataValue::Uint32(16)),
                        (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
                        (key("ssm.state_size"), MetadataValue::Uint32(3)),
                        (key("ssm.group_count"), MetadataValue::Uint32(2)),
                        (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                        (key("vocab_size"), MetadataValue::Uint32(64)),
                    ]),
                )
            }
            "qwen35" | "qwen3next" => {
                let (config, architecture) = if family == "qwen35" {
                    (qwen_hybrid_config(), "qwen35")
                } else {
                    (qwen_next_config(), "qwen3next")
                };
                let args = eredu_architectures::qwen::hybrid::model_args_from_config_value(&config)
                    .unwrap()
                    .text;
                let key = |suffix: &str| format!("{architecture}.{suffix}");
                let plan = eredu_architectures::qwen::hybrid::gguf_plan(&args).unwrap();
                let metadata = HashMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String(architecture.into()),
                    ),
                    ("general.file_type".into(), MetadataValue::Uint32(0)),
                    (key("block_count"), MetadataValue::Uint32(2)),
                    (key("embedding_length"), MetadataValue::Uint32(32)),
                    (key("attention.head_count"), MetadataValue::Uint32(4)),
                    (key("attention.head_count_kv"), MetadataValue::Uint32(2)),
                    (key("attention.key_length"), MetadataValue::Uint32(8)),
                    (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                    (key("full_attention_interval"), MetadataValue::Uint32(2)),
                    (key("vocab_size"), MetadataValue::Uint32(64)),
                    (key("context_length"), MetadataValue::Uint32(128)),
                    (
                        key("attention.layer_norm_rms_epsilon"),
                        MetadataValue::Float32(args.rms_norm_eps),
                    ),
                    (key("feed_forward_length"), MetadataValue::Uint32(48)),
                    (key("ssm.conv_kernel"), MetadataValue::Uint32(4)),
                    (key("ssm.state_size"), MetadataValue::Uint32(8)),
                    (key("ssm.group_count"), MetadataValue::Uint32(2)),
                    (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
                ]);
                (plan, metadata)
            }
            _ => unreachable!("heterogeneous GGUF fixture family"),
        };
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| {
                    if family == "qwen3next"
                        && group.variants.iter().any(|variant| variant.id == "fused")
                    {
                        group.variants.iter().find(|variant| variant.id == "fused")
                    } else {
                        group.variants.first()
                    }
                })
                .flat_map(|variant| variant.tensors.iter()),
        );
        let arrays = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let shape = constraint
                    .shape
                    .iter()
                    .map(|dimension| i32::try_from(*dimension).unwrap())
                    .collect::<Vec<_>>();
                let array = if constraint.key.ends_with("ssm_a") {
                    Array::full::<f32>(&shape, Array::from_f32(-1.0), stream).unwrap()
                } else {
                    let seed = constraint
                        .key
                        .bytes()
                        .fold(0_usize, |sum, byte| sum.wrapping_add(usize::from(byte)));
                    let values = (0..shape.iter().map(|dimension| *dimension as usize).product())
                        .map(|index| ((index + seed) % 29 + 1) as f32 / 100.0)
                        .collect::<Vec<_>>();
                    Array::from_slice(&values, &shape)
                };
                (constraint.key.clone(), array)
            })
            .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
            packed_qkvz.filter(|_| name.contains("attn_qkvz.weight"))
        })
    }

    fn lfm2_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
            "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
        })
    }

    fn routed_lfm2_config() -> serde_json::Value {
        let mut config = lfm2_config();
        config["model_type"] = "lfm2_moe".into();
        config["num_dense_layers"] = 1.into();
        config["moe_intermediate_size"] = 8.into();
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        config
    }

    fn kimi_linear_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":64,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":16,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":8,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":2,"num_expert_group":1,"topk_group":1
        })
    }

    fn routed_kimi_linear_config() -> serde_json::Value {
        let mut config = kimi_linear_config();
        config["first_k_dense_replace"] = 1.into();
        config
    }

    fn nemotron_h_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":64, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-M", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":0
        })
    }

    fn packed_alias_nemotron_h_config() -> serde_json::Value {
        let mut config = nemotron_h_config();
        config["hidden_size"] = 32.into();
        config["intermediate_size"] = 32.into();
        config["head_dim"] = 8.into();
        config["mamba_head_dim"] = 8.into();
        config["moe_intermediate_size"] = 32.into();
        config["moe_shared_expert_intermediate_size"] = 32.into();
        config["quantization"] = serde_json::json!({ "group_size": 32, "bits": 4 });
        config
    }

    fn qwen_hybrid_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5_text", "vocab_size": 64, "hidden_size": 32,
            "num_hidden_layers": 2, "mtp_num_hidden_layers": 0,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "intermediate_size": 48, "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24, "num_experts_per_tok": 0,
            "num_experts": 0, "layer_types": ["linear_attention", "full_attention"]
        })
    }

    fn routed_qwen_hybrid_config() -> serde_json::Value {
        let mut config = qwen_hybrid_config();
        config["model_type"] = "qwen3_5_moe_text".into();
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        config
    }

    fn routed_qwen_next_config() -> serde_json::Value {
        let mut config = routed_qwen_hybrid_config();
        config["model_type"] = "qwen3_next".into();
        config
    }

    fn routed_deepseek_v3_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3", "hidden_size": 16,
            "intermediate_size": 24, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_nextn_predict_layers": 0,
            "num_attention_heads": 2, "vocab_size": 64,
            "max_position_embeddings": 64, "q_lora_rank": 4,
            "kv_lora_rank": 4, "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2, "v_head_dim": 8,
            "first_k_dense_replace": 1, "moe_layer_freq": 1,
            "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 1, "n_group": 1, "topk_group": 1,
            "topk_method": "noaux_tc", "scoring_func": "sigmoid",
            "norm_topk_prob": true, "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false
        })
    }

    fn routed_deepseek_v4_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4", "hidden_size": 8,
            "moe_intermediate_size": 4, "num_hidden_layers": 3,
            "num_nextn_predict_layers": 0, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "qk_rope_head_dim": 2,
            "q_lora_rank": 2, "o_lora_rank": 2, "o_groups": 2,
            "vocab_size": 64, "max_position_embeddings": 128,
            "sliding_window": 4, "compress_ratios": [0, 4, 0],
            "index_n_heads": 2, "index_head_dim": 4, "index_topk": 1,
            "hc_mult": 2, "hc_sinkhorn_iters": 2,
            "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 1, "num_hash_layers": 1,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "norm_topk_prob": true, "routed_scaling_factor": 1.0,
            "swiglu_limit": 4.0, "tie_word_embeddings": false
        })
    }

    fn qwen_next_config() -> serde_json::Value {
        let mut config = qwen_hybrid_config();
        config["model_type"] = "qwen3_next".into();
        config
    }

    fn execution_streams() -> (Stream, Stream) {
        let execution_device = if cfg!(feature = "metal") {
            safemlx::Device::new(safemlx::DeviceType::Gpu, 0)
        } else {
            safemlx::Device::new(safemlx::DeviceType::Cpu, 0)
        };
        let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        (
            Stream::new_with_device(&execution_device),
            Stream::new_with_device(&weights_device),
        )
    }

    #[test]
    fn mlx_pipeline_allocator_converts_f16_and_bf16_sources_to_exact_f32_wire_values() {
        use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

        let (stream, _) = execution_streams();
        let expected = [1.5_f32, -2.0, 0.25];
        for source_dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let source = Array::from_slice(&expected, &[1, 1, 3])
                .as_dtype(source_dtype, &stream)
                .unwrap();
            let converted = MlxPartitionTensorAllocator
                .tensor_to_wire(
                    MlxTensor::from_array(source),
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    eredu_runtime::PipelineActivationDtype::Float32,
                    &stream,
                )
                .unwrap();
            let evaluated = converted.as_array().evaluated().unwrap();
            assert_eq!(evaluated.as_array().dtype(), Dtype::Float32);
            assert_eq!(evaluated.as_slice::<f32>(), expected.as_slice());
        }
    }

    #[test]
    fn mlx_pipeline_allocator_rejects_nonfloating_source_before_wire_submission() {
        use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

        let (stream, _) = execution_streams();
        let source = MlxTensor::from_array(Array::from_slice(&[1_i32], &[1, 1, 1]));
        let error = MlxPartitionTensorAllocator
            .tensor_to_wire(
                source,
                eredu_runtime::BoundaryTensorDtype::Activation,
                eredu_runtime::PipelineActivationDtype::Float32,
                &stream,
            )
            .expect_err("integer source activation must not enter the pipeline wire");
        assert!(error.to_string().contains("logical boundary dtype"));
    }

    #[test]
    fn mlx_pipeline_allocator_preserves_exact_u32_and_i32_boundary_roles() {
        use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

        let (stream, _) = execution_streams();
        for (source, logical, expected) in [
            (
                MlxTensor::from_array(Array::from_slice(&[7_u32], &[1, 1])),
                eredu_runtime::BoundaryTensorDtype::Uint32,
                Dtype::Uint32,
            ),
            (
                MlxTensor::from_array(Array::from_slice(&[-7_i32], &[1, 1])),
                eredu_runtime::BoundaryTensorDtype::Int32,
                Dtype::Int32,
            ),
        ] {
            let converted = MlxPartitionTensorAllocator
                .tensor_to_wire(
                    source,
                    logical,
                    eredu_runtime::PipelineActivationDtype::Float32,
                    &stream,
                )
                .unwrap();
            assert_eq!(converted.as_array().dtype(), expected);
            let placeholder = MlxPartitionTensorAllocator
                .tensor_placeholder(
                    &[1, 1],
                    logical,
                    eredu_runtime::PipelineActivationDtype::Float32,
                    &stream,
                )
                .unwrap();
            assert_eq!(placeholder.as_array().dtype(), expected);
        }
    }

    fn complete_state_capabilities(
        components: impl IntoIterator<Item = StateComponentMechanism>,
    ) -> StateMechanismCapabilities {
        StateMechanismCapabilities::new(components)
            .with_transactions(true, true)
            .with_reset(true)
            .with_prompt_cache(true)
            .with_observation_retention(true)
    }

    fn capabilities_with(
        full: &BackendMechanismCapabilities,
        operators: eredu_nn::NeuralOperatorCapabilities,
        state: StateMechanismCapabilities,
    ) -> BackendMechanismCapabilities {
        BackendMechanismCapabilities::new(
            operators,
            full.weight_lowerings().to_vec(),
            full.weight_residencies().to_vec(),
            state,
        )
        .with_session(full.session())
        .with_prompt_cache(full.prompt_cache())
        .with_exact_completion(full.exact_completion())
        .with_grouped_operations(full.grouped_operations().iter().copied())
    }

    fn tiny_llama_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
        ]);
        let tensors = [
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            (
                name.to_string(),
                Array::zeros::<f32>(&shape, stream).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    fn tiny_qwen_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(1_000_000.0)),
        ]);
        let mut tensors = vec![
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ];
        if architecture == "qwen2" {
            tensors.extend([
                ("blk.0.attn_q.bias", vec![32]),
                ("blk.0.attn_k.bias", vec![8]),
                ("blk.0.attn_v.bias", vec![8]),
            ]);
        } else {
            tensors.extend([
                ("blk.0.attn_q_norm.weight", vec![8]),
                ("blk.0.attn_k_norm.weight", vec![8]),
            ]);
        }
        let tensors = tensors
            .into_iter()
            .map(|(name, shape)| {
                (
                    name.to_string(),
                    Array::zeros::<f32>(&shape, stream).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    fn tiny_qwen_moe_gguf(stream: &Stream) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("qwen3moe.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("qwen3moe".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("attention.key_length"), MetadataValue::Uint32(8)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (key("expert_feed_forward_length"), MetadataValue::Uint32(16)),
            (key("expert_count"), MetadataValue::Uint32(2)),
            (key("expert_used_count"), MetadataValue::Uint32(1)),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
        ]);
        let tensors = [
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.attn_q_norm.weight", vec![8]),
            ("blk.0.attn_k_norm.weight", vec![8]),
            ("blk.0.ffn_gate_inp.weight", vec![2, 32]),
            ("blk.0.ffn_gate_exps.weight", vec![2, 16, 32]),
            ("blk.0.ffn_up_exps.weight", vec![2, 16, 32]),
            ("blk.0.ffn_down_exps.weight", vec![2, 32, 16]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            (
                name.to_owned(),
                Array::zeros::<f32>(&shape, stream).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |_, _| None)
    }

    #[test]
    fn gguf_requirements_retain_shard_and_multi_output_provenance() {
        let (stream, _) = execution_streams();
        let gguf = tiny_llama_gguf("llama", Some(eredu_gguf::GgmlType::MxFp4), &stream);
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let shard = inspection.gguf_checkpoint().unwrap().shards()[0]
            .path()
            .to_path_buf();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let output = requirements
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == "lm_head.weight")
            .expect("tied GGUF output remains explicit in the logical topology");
        assert!(matches!(
            output.presence(),
            eredu_runtime::ReplicatedTextParameterPresence::Tied { target }
                if target == "model.embed_tokens.weight"
        ));
        let derived = requirements
            .parameters()
            .iter()
            .find(|parameter| {
                matches!(
                    parameter.presence(),
                    eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                ) && parameter
                    .physical_sources()
                    .iter()
                    .any(|source| source.output().ends_with(".scales"))
            })
            .expect("MXFP4 requirements include a derived scales output");
        let source = &derived.physical_sources()[0];
        assert_eq!(source.shard(), shard);
        assert!(source.tensor().ends_with(".weight"));
        assert!(source.output().ends_with(".scales"));
        let direct = requirements
            .parameters()
            .iter()
            .find(|parameter| {
                parameter
                    .physical_sources()
                    .iter()
                    .any(|candidate| candidate.tensor() == source.tensor())
                    && parameter.presence().has_physical_source()
            })
            .expect("the same MXFP4 tensor includes its direct weight output");
        assert_eq!(direct.physical_sources()[0].shard(), source.shard());
        assert_ne!(direct.physical_sources()[0].output(), source.output());
    }

    #[test]
    fn public_handoff_executes_ordinary_and_routed_qwen_with_repeated_decode() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (model_type, tied) in [
            ("llama", true),
            ("mistral", false),
            ("qwen2", false),
            ("qwen3", true),
            ("qwen3_moe", false),
            ("gpt_oss", false),
        ] {
            let root = tiny_artifact(model_type, tied);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            for token in [1_u32, 2] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap();
                assert_eq!(logits.shape(), &[1, 64]);
                logits.evaluated().unwrap();
            }
            assert!(executable.parameter_bank_report().unwrap().is_none());
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 6,
                state_allocations: 6,
                payload_opens: 6,
                constructors: 6,
                unit_constructions: 6,
                materializations: 0,
                local_static_bindings: 0,
                excluded_local_static_parameters: 0,
                forwards: 12,
                state_publications: 12,
                completions: 12,
            }
        );
    }

    #[test]
    fn invalid_token_completion_rolls_back_session_state_before_publication() {
        let (stream, weights_stream) = execution_streams();
        let root = tiny_artifact("llama", true);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let backend = crate::backend::MlxBackend::new(&stream, &weights_stream);
        let mut session = crate::composition::mlx::MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let first = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[1_u32], &[1, 1]),
        )
        .unwrap();
        let overlap = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[2_u32], &[1, 1]),
        )
        .err()
        .expect("an unresolved completion must gate the next state mutation");
        assert!(overlap
            .to_string()
            .contains("unresolved submission completion"));
        eredu_core::Completion::wait(&first.completion).unwrap();
        let before = session
            .neutral_prediction_target_mut()
            .unwrap()
            .state_snapshot();
        let before_numeric = session
            .neutral_prediction_target_mut()
            .unwrap()
            .fixed_numeric_state_snapshot()
            .unwrap();

        let error = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[-1_i32], &[1, 1]),
        )
        .err()
        .expect("out-of-domain token must fail exact mechanism completion");
        assert!(error.to_string().contains("outside 0..64"));
        assert_eq!(
            session
                .neutral_prediction_target_mut()
                .unwrap()
                .state_snapshot(),
            before
        );
        assert_eq!(
            session
                .neutral_prediction_target_mut()
                .unwrap()
                .fixed_numeric_state_snapshot()
                .unwrap(),
            before_numeric
        );
        let recovered = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[2_u32], &[1, 1]),
        )
        .expect("failed token completion must release the submission gate");
        eredu_core::Completion::wait(&recovered.completion).unwrap();
    }

    #[test]
    fn routed_qwen_gguf_executes_resident_and_addressable_through_generic_composition() {
        let (stream, weights_stream) = execution_streams();
        let artifact = tiny_qwen_moe_gguf(&stream);
        for addressable in [false, true] {
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let mut options = crate::MlxLoadRequest::default();
            if addressable {
                options = options.with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                );
            }
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let logits = generic
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap();
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
    }

    #[test]
    fn routed_addressable_storage_executes_qwen_and_gpt_oss_repeated_decode() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["qwen3_moe", "gpt_oss"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let residency = eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            );
            let options = crate::MlxLoadRequest::default().with_weight_residency(residency);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut complete = model.into_executable();
            {
                let executable = complete.erased_mut();
                let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
                let parts = [input::token_ids_part(&prompt).unwrap()];
                executable
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap_or_else(|error| panic!("{model_type}: {error}"))
                    .evaluated()
                    .unwrap();
                for token in [3_u32, 4, 5] {
                    let logits = executable
                        .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                        .unwrap_or_else(|error| panic!("{model_type}: {error}"));
                    assert_eq!(logits.shape(), &[1, 64]);
                    assert!(logits
                        .evaluated()
                        .unwrap()
                        .as_slice::<f32>()
                        .iter()
                        .all(|value| value.is_finite()));
                }
            }
            let report = complete
                .parameter_bank_report()
                .unwrap()
                .unwrap_or_else(|| panic!("{model_type}: no addressable-bank telemetry"));
            assert!(report.owned_entries() > 0, "{model_type}");
            assert!(report.bulk().requested_selections() > 0, "{model_type}");
            assert!(report.bulk().compact_banks() > 0, "{model_type}");
            assert!(
                report.incremental().requested_selections() > 0,
                "{model_type}"
            );
            assert!(report.incremental().compact_banks() > 0, "{model_type}");
        }
    }

    #[test]
    fn routed_addressable_load_time_transform_uses_selected_bank_geometry() {
        let (stream, weights_stream) = execution_streams();
        let mut nemotron = nemotron_h_config();
        nemotron["hybrid_override_pattern"] = "M*EM".into();
        nemotron["hidden_size"] = 32.into();
        nemotron["intermediate_size"] = 32.into();
        nemotron["head_dim"] = 8.into();
        nemotron["mamba_head_dim"] = 8.into();
        nemotron["moe_intermediate_size"] = 32.into();
        nemotron["moe_shared_expert_intermediate_size"] = 32.into();
        let artifacts = [
            ("qwen", tiny_artifact("qwen3_moe", false)),
            ("nemotron", tiny_heterogeneous_artifact(nemotron)),
        ];
        for (name, root) in artifacts {
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let residency = eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            );
            let options =
                crate::MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                })
                .with_weight_residency(residency);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut complete = model.into_executable();
            {
                let executable = complete.erased_mut();
                executable
                    .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
                    .evaluated()
                    .unwrap();
            }
            let report = complete
                .parameter_bank_report()
                .unwrap()
                .unwrap_or_else(|| panic!("{name}: no addressable-bank telemetry"));
            assert_eq!(
                report.weight_quantizations(),
                [eredu_checkpoint::WeightQuantization::Affine(
                    eredu_checkpoint::AffineQuantization::new(32, 4).unwrap()
                )],
                "{name}"
            );
            let materialization = report
                .materialization()
                .unwrap_or_else(|| panic!("{name}: no bank materialization telemetry"));
            assert!(materialization.transformed_weights > 0, "{name}");
            assert_eq!(materialization.output_bytes, report.owned_bytes(), "{name}");
            assert!(
                materialization.source_bytes_read > materialization.output_bytes,
                "{name}"
            );
        }
    }

    #[test]
    fn gpt_oss_load_time_transform_preserves_native_experts_in_both_residencies() {
        let (stream, weights_stream) = execution_streams();
        for addressable in [false, true] {
            let root = tiny_artifact("gpt_oss", false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
            let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            )
            .with_quantization(eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            });
            let weights = if addressable {
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::FullyResident,
                    eredu_runtime::ParameterBankLoadOptions::default(),
                )
            } else {
                eredu_runtime::WeightResidency::fully_resident()
            };
            let request =
                eredu_architectures::RoutedTextSelectionRequest::new(text, weights).unwrap();
            let selected = eredu_architectures::select_routed_text_realization(
                &requirements,
                &request,
                &capabilities(requirements.text(), request.text()),
            )
            .unwrap();
            let expert_targets = requirements.catalog().logical_targets();
            let selected_experts = requirements
                .text()
                .parameters()
                .iter()
                .filter(|parameter| {
                    expert_targets.contains(parameter.name())
                        && parameter.role()
                            == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                })
                .map(|requirement| {
                    selected
                        .text()
                        .parameters()
                        .iter()
                        .find(|parameter| parameter.name() == requirement.name())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert!(!selected_experts.is_empty());
            assert!(selected_experts.iter().all(|parameter| {
                parameter.executable() == eredu_checkpoint::LinearFormat::MxFp4
            }));
            assert!(selected.text().parameters().iter().any(|parameter| {
                !expert_targets.contains(parameter.name())
                    && parameter.lowering() == eredu_runtime::WeightLoweringKind::Transform
            }));

            let options =
                crate::MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                })
                .with_weight_residency(weights);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            assert!(
                model
                    .materialization_report()
                    .is_some_and(|report| report.transformed_weights > 0),
                "addressable={addressable}"
            );
            let mut complete = model.into_executable();
            let generic = complete.erased_mut();
            generic
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"))
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn routed_nemotron_relu2_executes_resident_and_addressable_with_mixed_state() {
        let (stream, weights_stream) = execution_streams();
        let mut config = nemotron_h_config();
        config["hybrid_override_pattern"] = "M*EM".into();
        for addressable in [false, true] {
            let root = tiny_heterogeneous_artifact(config.clone());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let mut options = crate::MlxLoadRequest::default();
            if addressable {
                options = options.with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                );
            }
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            for token in [1_u32, 2, 3] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
                assert_eq!(logits.shape(), &[1, 64]);
                logits.evaluated().unwrap();
            }
            let state = executable.state_snapshot();
            assert_eq!(state.len(), 4);
            assert!(state
                .iter()
                .all(|(_, components)| components.iter().all(|(_, present)| *present)));
        }
    }

    #[test]
    fn routed_gated_families_execute_resident_and_addressable_with_heterogeneous_state() {
        let (stream, weights_stream) = execution_streams();
        for (name, config) in [
            ("lfm2_moe", routed_lfm2_config()),
            ("kimi_linear", routed_kimi_linear_config()),
            ("qwen3_5_moe_text", routed_qwen_hybrid_config()),
            ("qwen3_next", routed_qwen_next_config()),
            ("deepseek_v3", routed_deepseek_v3_config()),
        ] {
            for addressable in [false, true] {
                let root = tiny_heterogeneous_artifact(config.clone());
                let inspection =
                    eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
                let mut options = crate::MlxLoadRequest::default();
                if addressable {
                    options = options.with_weight_residency(
                        eredu_runtime::WeightResidency::with_independent_parameter_banks(
                            eredu_runtime::OrdinaryWeightResidency::FullyResident,
                            eredu_runtime::ParameterBankLoadOptions::default(),
                        ),
                    );
                }
                let plan = eredu_core::plan_model_preparation(
                    inspection,
                    options.preparation_policy().unwrap(),
                    eredu_core::SessionCapabilities::default(),
                )
                .unwrap();
                let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                    .unwrap_or_else(|error| panic!("{name} addressable={addressable}: {error}"));
                let mut executable = model.into_executable();
                let executable = executable.erased_mut();
                for token in [1_u32, 2, 3] {
                    let logits = executable
                        .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                        .unwrap_or_else(|error| {
                            panic!("{name} addressable={addressable}: {error}")
                        });
                    assert_eq!(logits.shape(), &[1, 64]);
                    logits.evaluated().unwrap();
                }
                assert_eq!(
                    executable.state_snapshot().len(),
                    2,
                    "{name} state layout must retain both target layers"
                );
            }
        }
    }

    #[test]
    fn routed_session_observation_reports_shared_combination_and_intervenes_causally() {
        struct Observer {
            routing_path: Option<String>,
            semantic_outputs: bool,
            intervened: bool,
            stream: Stream,
        }

        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, _: &str, _: &Array) -> Result<(), Exception> {
                Ok(())
            }

            fn observe_routing(
                &mut self,
                observation: eredu_runtime::RoutingObservation<'_, Array>,
            ) -> Result<(), Exception> {
                self.semantic_outputs =
                    observation.shared_output.is_some() && observation.combined_output.is_some();
                self.routing_path = Some(observation.path.to_owned());
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                let routed_output = self
                    .routing_path
                    .as_deref()
                    .is_some_and(|routing| path == format!("{routing}.output"));
                if routed_output {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        for addressable in [false, true] {
            let root = tiny_heterogeneous_artifact(routed_kimi_linear_config());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let mut options = crate::MlxLoadRequest::default();
            if addressable {
                options = options.with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                );
            }
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let tokens = Array::from_slice(&[3_u32], &[1, 1]);
            let baseline = generic
                .decode(&tokens, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            generic.reset_cache().unwrap();
            let mut observer = Observer {
                routing_path: None,
                semantic_outputs: false,
                intervened: false,
                stream: stream.clone(),
            };
            let changed = generic
                .forward_with_observer(&tokens, None, &stream, &mut observer)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert!(observer.semantic_outputs, "addressable={addressable}");
            assert!(observer.intervened, "addressable={addressable}");
            assert_ne!(baseline, changed, "addressable={addressable}");
        }
    }

    #[test]
    fn routed_only_default_observation_intervenes_on_provider_output() {
        struct Observer {
            routing_path: Option<String>,
            routed_only: bool,
            intervened: bool,
            stream: Stream,
        }

        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, _: &str, _: &Array) -> Result<(), Exception> {
                Ok(())
            }

            fn observe_routing(
                &mut self,
                observation: eredu_runtime::RoutingObservation<'_, Array>,
            ) -> Result<(), Exception> {
                self.routed_only =
                    observation.shared_output.is_none() && observation.combined_output.is_none();
                self.routing_path = Some(observation.path.to_owned());
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                if self
                    .routing_path
                    .as_deref()
                    .is_some_and(|routing| path == format!("{routing}.output"))
                {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        let root = tiny_artifact("qwen3_moe", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let options = crate::MlxLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        );
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let tokens = Array::from_slice(&[3_u32], &[1, 1]);
        let baseline = generic
            .decode(&tokens, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        generic.reset_cache().unwrap();
        let mut observer = Observer {
            routing_path: None,
            routed_only: false,
            intervened: false,
            stream: stream.clone(),
        };
        let changed = generic
            .forward_with_observer(&tokens, None, &stream, &mut observer)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(observer.routed_only);
        assert!(observer.intervened);
        assert_ne!(baseline, changed);
    }

    #[test]
    fn routed_deepseek_v4_executes_resident_and_addressable_with_pooling_state() {
        let (stream, weights_stream) = execution_streams();
        for addressable in [false, true] {
            let root = tiny_heterogeneous_artifact(routed_deepseek_v4_config());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let mut options = crate::MlxLoadRequest::default();
            if addressable {
                options = options.with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                );
            }
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            for token in [1_u32, 2, 3, 4, 5] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
                assert_eq!(logits.shape(), &[1, 64]);
                assert!(logits
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .iter()
                    .all(|value| value.is_finite()));
            }
            assert_eq!(executable.state_snapshot().len(), 3);
        }
    }

    #[test]
    fn routed_deepseek_v4_pooling_state_uses_shared_checkpoint_and_prompt_cache_controls() {
        let (stream, weights_stream) = execution_streams();
        let root = tiny_heterogeneous_artifact(routed_deepseek_v4_config());
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let options = crate::MlxLoadRequest::default()
            .with_weight_residency(
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::FullyResident,
                    eredu_runtime::ParameterBankLoadOptions::default(),
                ),
            )
            .with_state_residency(CacheResidencyPolicy::Paged(paged));
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prefix = [1_u32, 2, 3, 4, 5];
        let prompt = Array::from_slice(&prefix, &[1, 5]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let before = generic.state_snapshot();
        let before_numeric = generic.fixed_numeric_state_snapshot().unwrap();
        assert!(before
            .iter()
            .any(|(_, components)| { components.iter().any(|(_, present)| *present) }));
        assert!(!before_numeric.is_empty());

        let continuation = Array::from_slice(&[6_u32], &[1, 1]);
        let probe = generic
            .checkpoint_restore_probe(&continuation, &stream)
            .unwrap();
        assert_eq!(probe.0, probe.2);
        assert_eq!(probe.3, probe.5);

        let descriptor = PromptCacheDescriptor::from_model_identity(
            generic.prompt_cache_model_identity().clone(),
            "deepseek-v4-checkpoint",
            "tokens:1,2,3,4,5",
            1,
        )
        .unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let destination = cache_root.path().join("cache");
        generic
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        generic.reset_cache().unwrap();
        assert!(generic.state_snapshot().iter().all(|(offset, components)| {
            *offset == 0 && components.iter().all(|(_, present)| !present)
        }));
        generic
            .load_prompt_cache(&destination, &descriptor, &prefix)
            .unwrap();
        assert_eq!(generic.state_snapshot(), before);
        assert_eq!(
            generic.fixed_numeric_state_snapshot().unwrap(),
            before_numeric
        );
        let restored = generic
            .decode(&continuation, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(restored, probe.6);
    }

    #[test]
    fn generic_handoff_executes_every_replicated_state_profile() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
            assert!(requirements
                .state_layout()
                .layers()
                .iter()
                .any(|layer| !layer.fixed_state().is_empty()));
            let stateful_layers = requirements
                .state_layout()
                .layers()
                .iter()
                .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
                .collect::<Vec<_>>();
            let policy = eredu_core::PreparationPolicy::default();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            let logits = executable
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap_or_else(|error| panic!("{name} prefill: {error}"));
            assert_eq!(logits.shape(), &[1, 64], "{name}");
            logits.evaluated().unwrap();
            let snapshot = executable.state_snapshot();
            assert!(
                snapshot
                    .iter()
                    .zip(&stateful_layers)
                    .all(|((position, _), stateful)| *position == if *stateful { 2 } else { 0 }),
                "{name} prefill: {snapshot:?}"
            );
            for (step, token) in [3_u32, 4].into_iter().enumerate() {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("{name} decode {step}: {error}"));
                assert_eq!(logits.shape(), &[1, 64], "{name}");
                logits.evaluated().unwrap();
                let snapshot = executable.state_snapshot();
                assert!(
                    snapshot
                        .iter()
                        .zip(&stateful_layers)
                        .all(|((position, _), stateful)| *position
                            == if *stateful { step as i32 + 3 } else { 0 }),
                    "{name} step {step}: {snapshot:?}"
                );
                assert!(snapshot
                    .iter()
                    .flat_map(|(_, fixed)| fixed)
                    .all(|(_, present)| *present));
            }
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 5,
                state_allocations: 5,
                payload_opens: 5,
                constructors: 5,
                unit_constructions: 12,
                materializations: 0,
                local_static_bindings: 0,
                excluded_local_static_parameters: 0,
                forwards: 15,
                state_publications: 15,
                completions: 15,
            }
        );
    }

    #[test]
    fn homogeneous_state_schedules_execute_with_only_their_exact_mechanisms() {
        let (stream, weights_stream) = execution_streams();
        let mut cases = Vec::new();
        let mut lfm = lfm2_config();
        lfm["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((
            "lfm_attention",
            lfm,
            ReplicatedTextStateAccess::KeyValue,
            None,
        ));
        let mut lfm = lfm2_config();
        lfm["layer_types"] = serde_json::json!(["conv", "conv"]);
        cases.push(("lfm_fixed", lfm, ReplicatedTextStateAccess::Fixed, None));

        let mut kimi = kimi_linear_config();
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([1, 2]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([]);
        cases.push((
            "kimi_kda",
            kimi,
            ReplicatedTextStateAccess::Fixed,
            Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
        ));
        let mut kimi = kimi_linear_config();
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
        cases.push((
            "kimi_mla",
            kimi,
            ReplicatedTextStateAccess::CompressedAttention,
            None,
        ));

        for (name, pattern, access, operator) in [
            (
                "nemotron_attention",
                "****",
                ReplicatedTextStateAccess::KeyValue,
                None,
            ),
            (
                "nemotron_mamba",
                "MMMM",
                ReplicatedTextStateAccess::Fixed,
                Some(eredu_nn::NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN),
            ),
            (
                "nemotron_stateless",
                "----",
                ReplicatedTextStateAccess::Stateless,
                None,
            ),
        ] {
            let mut nemo = nemotron_h_config();
            nemo["hybrid_override_pattern"] = pattern.into();
            cases.push((name, nemo, access, operator));
        }
        let mut qwen = qwen_hybrid_config();
        qwen["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((
            "qwen_attention",
            qwen,
            ReplicatedTextStateAccess::KeyValue,
            None,
        ));
        let mut qwen = qwen_hybrid_config();
        qwen["layer_types"] = serde_json::json!(["linear_attention", "linear_attention"]);
        cases.push((
            "qwen_fixed",
            qwen,
            ReplicatedTextStateAccess::Fixed,
            Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
        ));

        for (name, config, access, operator) in cases {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
            assert_eq!(requirements.state_access(), access, "{name}");
            if let Some(operator) = operator {
                assert!(requirements.operators().contains(operator), "{name}");
            } else {
                assert_eq!(
                    requirements.operators(),
                    eredu_nn::NeuralOperatorCapabilities::NONE,
                    "{name}"
                );
            }
            let stateful = requirements
                .state_layout()
                .layers()
                .iter()
                .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
                .collect::<Vec<_>>();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let logits = generic
                .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert!(
                logits.iter().all(|value| value.is_finite())
                    && logits.iter().any(|value| value.abs() > 1e-12),
                "{name}: {logits:?}"
            );
            assert!(generic
                .state_snapshot()
                .iter()
                .zip(&stateful)
                .all(|((position, _), stateful)| *position == if *stateful { 3 } else { 0 }));
        }
    }

    #[test]
    fn heterogeneous_gguf_artifacts_use_the_same_generic_state_contract() {
        super::super::path_instrumentation::reset();
        let (stream, weights_stream) = execution_streams();
        for (name, gguf_name, config) in [
            ("lfm2", "lfm2", lfm2_config()),
            ("kimi_linear", "kimi_linear", kimi_linear_config()),
            ("nemotron_h", "nemotron_h", nemotron_h_config()),
            ("qwen3_next", "qwen3next", qwen_next_config()),
            ("qwen3_5_text", "qwen35", qwen_hybrid_config()),
        ] {
            let gguf = tiny_heterogeneous_gguf(gguf_name, &stream);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|error| panic!("{name} GGUF requirements: {error}"));
            let stateful = (0..requirements.state_layout().len())
                .map(|layer| {
                    !requirements
                        .state_layout()
                        .components(layer)
                        .unwrap()
                        .is_empty()
                })
                .collect::<Vec<_>>();
            let safe = tiny_heterogeneous_artifact(config);
            let safe_inspection =
                eredu_architectures::configuration::inspect_artifact(safe.path()).unwrap();
            let safe_requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(
                    &safe_inspection,
                )
                .unwrap();
            assert_eq!(
                requirements.state_access(),
                safe_requirements.state_access()
            );
            assert_eq!(
                requirements.state_layout(),
                safe_requirements.state_layout()
            );
            assert_eq!(requirements.operators(), safe_requirements.operators());
            assert_eq!(
                requirements.execution_graph(),
                safe_requirements.execution_graph()
            );

            let execute = |token| {
                let fresh =
                    eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
                let plan = eredu_core::plan_model_preparation(
                    fresh,
                    eredu_core::PreparationPolicy::default(),
                    eredu_core::SessionCapabilities::default(),
                )
                .unwrap();
                let model = materialize_model_plan(
                    plan,
                    crate::MlxLoadRequest::default(),
                    &stream,
                    &weights_stream,
                )
                .unwrap_or_else(|error| panic!("{name} GGUF: {error}"));
                let mut executable = model.into_executable();
                let generic = executable.erased_mut();
                let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
                let parts = [input::token_ids_part(&prompt).unwrap()];
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
                let fixed_before = generic.fixed_numeric_state_snapshot().unwrap();
                let logits = generic
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                let fixed_after = generic.fixed_numeric_state_snapshot().unwrap();
                (logits, generic.state_snapshot(), fixed_before, fixed_after)
            };
            let token_three = execute(3_u32);
            let token_four = execute(4_u32);
            for (logits, snapshot, fixed_before, fixed_after) in [&token_three, &token_four] {
                assert!(
                    logits.iter().all(|value| value.is_finite())
                        && logits.iter().any(|value| value.abs() > 1e-12),
                    "{name} GGUF produced invalid logits: {logits:?}"
                );
                assert!(snapshot
                    .iter()
                    .zip(&stateful)
                    .all(|((position, fixed), stateful)| {
                        *position == if *stateful { 3 } else { 0 }
                            && fixed.iter().all(|(_, present)| *present)
                    }));
                assert_ne!(
                    fixed_before, fixed_after,
                    "{name} fixed state did not consume decode input"
                );
            }
            assert_ne!(
                token_three.0, token_four.0,
                "{name} ignored token identity at an identical state frontier"
            );
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts {
                architecture_constructions: 10,
                state_allocations: 10,
                payload_opens: 10,
                constructors: 10,
                unit_constructions: 24,
                materializations: 0,
                local_static_bindings: 0,
                excluded_local_static_parameters: 0,
                forwards: 20,
                state_publications: 20,
                completions: 20,
            }
        );
    }

    #[test]
    fn gpt_oss_gguf_uses_generic_routed_execution_for_both_residencies() {
        let (stream, weights_stream) = execution_streams();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("gpt-oss.gguf");
        crate::tests::distributed_pipeline_ring::write_gpt_oss_gguf_fixture(&path);
        for addressable in [false, true] {
            let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
            let mut options = crate::MlxLoadRequest::default();
            if addressable {
                options = options.with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                );
            }
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"))
                .evaluated()
                .unwrap();
            let logits = generic
                .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            assert_eq!(logits.shape(), &[1, 64]);
            assert!(logits
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn packed_fused_qwen_next_gguf_format_reaches_split_projection_execution() {
        let (stream, weights_stream) = execution_streams();
        let gguf = tiny_heterogeneous_gguf_with_packed_qwen_next(
            "qwen3next",
            Some(eredu_gguf::GgmlType::MxFp4),
            &stream,
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let fused_targets = requirements
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_qkv.weight")
                    || parameter.name().contains("linear_attn.in_proj_z.weight")
            })
            .collect::<Vec<_>>();
        assert_eq!(fused_targets.len(), 2);
        assert!(fused_targets.iter().all(|parameter| {
            matches!(
                parameter.presence(),
                eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
            ) && parameter.has_lowering_source()
                && parameter.native_executable() == eredu_checkpoint::LinearFormat::MxFp4
        }));
        let selection_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &selection_request,
            &capabilities(&requirements, &selection_request),
        )
        .unwrap();
        assert!(selected
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_qkv.weight")
                    || parameter.name().contains("linear_attn.in_proj_z.weight")
            })
            .all(|parameter| parameter.lowering() == eredu_runtime::WeightLoweringKind::Derived));

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            logits.iter().all(|value| value.is_finite())
                && logits.iter().any(|value| value.abs() > 1e-12)
        );
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 3));
    }

    #[test]
    fn fused_qwen_next_safetensors_preserves_both_source_families_into_execution() {
        let (stream, weights_stream) = execution_streams();
        let root = tiny_heterogeneous_artifact_with_layout(qwen_next_config(), true);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let fused = requirements
            .parameters()
            .iter()
            .filter(|parameter| {
                parameter.name().contains("linear_attn.in_proj_")
                    && matches!(
                        parameter.presence(),
                        eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                    )
            })
            .collect::<Vec<_>>();
        assert_eq!(fused.len(), 4);
        assert!(fused.iter().all(|parameter| {
            parameter.has_lowering_source()
                && matches!(
                    parameter.source_encoding(),
                    Some(eredu_checkpoint::SourceTensorEncoding::RecipeOutput(
                        eredu_checkpoint::StoredDtype::F32
                    ))
                )
                && parameter.physical_shape().is_some()
                && parameter.physical_sources().len() == 1
        }));
        assert_eq!(
            fused
                .iter()
                .filter(|parameter| parameter.physical_sources()[0].tensor().contains("qkvz"))
                .count(),
            2
        );
        assert_eq!(
            fused
                .iter()
                .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
                .count(),
            2
        );
        assert!(fused
            .iter()
            .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
            .all(
                |parameter| parameter.native_executable() == eredu_checkpoint::LinearFormat::Dense
            ));

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            logits.iter().all(|value| value.is_finite())
                && logits.iter().any(|value| value.abs() > 1e-12)
        );
        assert!(generic
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 3));
    }

    #[test]
    fn checkpoint_native_packed_safetensors_companions_are_consumed_once() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["llama", "qwen3"] {
            let root = tiny_packed_safetensors_artifact(model_type);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            assert!(requirements.parameters().iter().any(|parameter| {
                parameter.role() == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
                    && parameter.source_encoding()
                        == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                            eredu_checkpoint::StoredDtype::F32,
                        ))
            }));
            assert!(requirements.parameters().iter().any(|parameter| {
                parameter.role() == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    && matches!(
                        parameter.native_executable(),
                        eredu_checkpoint::LinearFormat::Affine(_)
                    )
                    && parameter.source_encoding()
                        == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                            eredu_checkpoint::StoredDtype::U32,
                        ))
            }));

            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let logits = generic
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert_eq!(logits.len(), 64);
            assert!(logits.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn alias_backed_packed_safetensors_companions_keep_their_exact_source() {
        let (stream, weights_stream) = execution_streams();
        let config = packed_alias_nemotron_h_config();
        assert_eq!(
            eredu_architectures::nemotron_h::model_args_from_config_value(&config)
                .unwrap()
                .hidden_size,
            32
        );
        let root = tiny_heterogeneous_artifact(config);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let eredu_architectures::configuration::SafetensorsModelConfig::NemotronH(inspected) =
            inspection
                .architecture_plan()
                .safetensors_architecture()
                .unwrap()
                .model()
        else {
            panic!("expected Nemotron-H inspection")
        };
        assert_eq!(inspected.hidden_size, 32);
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        for parameter in requirements.parameters().iter().filter(|parameter| {
            parameter.source_encoding()
                == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                    eredu_checkpoint::StoredDtype::U32,
                ))
        }) {
            let descriptor = parameter
                .lowering_descriptor(parameter.native_executable())
                .unwrap();
            assert!(
                supports_direct(&descriptor),
                "unsupported native packed requirement {} {:?} {:?} logical={:?}: {descriptor:?}",
                parameter.name(),
                parameter.role(),
                parameter.presence(),
                parameter.logical_shape()
            );
        }
        let aliased_companion = requirements
            .parameters()
            .iter()
            .find(|parameter| {
                parameter.role() == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
                    && parameter.name().starts_with("model.")
                    && parameter
                        .sources()
                        .first()
                        .is_some_and(|source| source.starts_with("backbone."))
            })
            .unwrap_or_else(|| {
                panic!(
                    "Nemotron-H fixture must select an official alias-backed companion: {:?}",
                    requirements
                        .parameters()
                        .iter()
                        .filter(|parameter| parameter.role()
                            == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion)
                        .map(|parameter| (parameter.name(), parameter.sources()))
                        .collect::<Vec<_>>()
                )
            });
        assert_ne!(aliased_companion.name(), aliased_companion.sources()[0]);

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let logits = generic
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn both_text_architectures_retain_exact_sharded_admission_with_one_cached_payload() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["mistral", "qwen3"] {
            let root = tiny_sharded_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let streaming =
                eredu_runtime::DenseDiskStreamLoadOptions::default().with_max_cached_shards(1);
            let options = crate::MlxLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::dense_disk_stream(streaming),
            );
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            std::fs::remove_file(root.path().join("model.safetensors.index.json")).unwrap();

            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            executable
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let report = executable.dense_stream_report().unwrap().unwrap();
            assert!(report.residency().weight_store().currently_cached_shards <= 1);
            assert!(!report
                .residency()
                .weight_store()
                .payload_shard_paths
                .is_empty());
        }
    }

    #[test]
    fn unsupported_topology_fails_before_checkpoint_payload_or_module_construction() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = request.with_topology(topology.topology());
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();

        let error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .expect_err("unsupported topology was admitted");
        let message = error.to_string();
        assert!(
            message.contains("replicated execution topology"),
            "{message}"
        );
        assert!(!message.contains("No such file"), "{message}");
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn heterogeneous_state_and_operator_gaps_reject_before_any_production_path() {
        use eredu_core::cache::{
            LayerCachePolicy, StateComponentRole, StateTensorDtype, StateTensorPolicy,
        };

        super::super::path_instrumentation::reset();
        let paged = CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(4, 4096, 4096, 1)
                .unwrap()
                .with_full_attention(true),
        );
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            );
            let full = capabilities(&requirements, &request);

            let fixed_components = full
                .state()
                .components()
                .iter()
                .filter(|mechanism| {
                    matches!(mechanism.component().role(), StateComponentRole::Fixed(_))
                })
                .cloned()
                .collect::<Vec<_>>();
            assert!(
                !fixed_components.is_empty(),
                "{name} fixture has no fixed state"
            );
            for fixed in &fixed_components {
                let missing_fixed = complete_state_capabilities(
                    full.state()
                        .components()
                        .iter()
                        .filter(|mechanism| *mechanism != fixed)
                        .cloned(),
                );
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(&full, full.operators(), missing_fixed),
                )
                .expect_err("missing fixed state was admitted");
                assert!(
                    error.issues().iter().any(|issue| {
                        issue.contains(&fixed.component().role().stable_name())
                            && issue.contains("state component")
                    }),
                    "{name}: {error}"
                );
            }
            if name == "kimi_linear" {
                let without_compressed = complete_state_capabilities(
                    full.state()
                        .components()
                        .iter()
                        .filter(|mechanism| {
                            mechanism.component().role() != StateComponentRole::CompressedLatent
                        })
                        .cloned(),
                );
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(&full, full.operators(), without_compressed),
                )
                .expect_err("missing compressed attention was admitted");
                assert!(error
                    .issues()
                    .iter()
                    .any(|issue| issue.contains("attention.compressed_latent")));
            }

            for fixed in &fixed_components {
                let StateComponentRole::Fixed(role) = fixed.component().role() else {
                    unreachable!("fixed component filter changed")
                };
                let wrong_shape =
                    LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                        role,
                        vec![eredu_core::cache::StateTensorDimension::fixed(999).unwrap()],
                        fixed.component().dtype(),
                        fixed.component().residency(),
                    )
                    .unwrap()])
                    .unwrap()
                    .components()
                    .pop()
                    .unwrap();
                let components = full.state().components().iter().map(|mechanism| {
                    if mechanism == fixed {
                        StateComponentMechanism::new(
                            mechanism.layer(),
                            wrong_shape.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(StateComponentPlacement::Device),
                        )
                    } else {
                        mechanism.clone()
                    }
                });
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(
                        &full,
                        full.operators(),
                        complete_state_capabilities(components),
                    ),
                )
                .expect_err("wrong fixed-state shape was admitted");
                assert!(error
                    .issues()
                    .iter()
                    .any(|issue| issue.contains("shape") && issue.contains("dtype")));

                let alternate_dtype = match fixed.component().dtype() {
                    StateTensorDtype::Float32 => StateTensorDtype::Floating,
                    _ => StateTensorDtype::Float32,
                };
                let wrong_dtype =
                    LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                        role,
                        fixed.component().shape().to_vec(),
                        alternate_dtype,
                        fixed.component().residency(),
                    )
                    .unwrap()])
                    .unwrap()
                    .components()
                    .pop()
                    .unwrap();
                let components = full.state().components().iter().map(|mechanism| {
                    if mechanism == fixed {
                        StateComponentMechanism::new(
                            mechanism.layer(),
                            wrong_dtype.clone(),
                            Some(StateComponentPlacement::Device),
                            Some(StateComponentPlacement::Device),
                        )
                    } else {
                        mechanism.clone()
                    }
                });
                assert!(
                    eredu_runtime::select_replicated_text_realization(
                        &requirements,
                        &request,
                        &capabilities_with(
                            &full,
                            full.operators(),
                            complete_state_capabilities(components),
                        ),
                    )
                    .is_err(),
                    "{name} admitted an incompatible fixed-state dtype"
                );
            }

            let paged_components = full.state().components().iter().map(|mechanism| {
                StateComponentMechanism::new(
                    mechanism.layer(),
                    mechanism.component().clone(),
                    Some(StateComponentPlacement::Device),
                    Some(StateComponentPlacement::Paged),
                )
            });
            let paged_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                paged.clone(),
            )
            .with_prompt_cache(true);
            assert!(
                eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &paged_request,
                    &capabilities_with(
                        &full,
                        full.operators(),
                        complete_state_capabilities(paged_components),
                    ),
                )
                .is_err(),
                "{name} admitted incompatible paged fixed-component placement"
            );

            if requirements.operators() != eredu_nn::NeuralOperatorCapabilities::NONE {
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(
                        &full,
                        eredu_nn::NeuralOperatorCapabilities::NONE,
                        full.state().clone(),
                    ),
                )
                .expect_err("missing semantic neural operations were admitted");
                let operation = match name {
                    "nemotron_h" => "selective_state_space_scan",
                    "kimi_linear" | "qwen3_next" | "qwen3_5_text" => "gated_delta_scan",
                    _ => unreachable!(),
                };
                assert!(
                    error.issues().iter().any(|issue| issue.contains(operation)),
                    "{name}: {error}"
                );
            }

            let paged_full = capabilities(&requirements, &paged_request);
            for (facility, state) in [
                (
                    "checkpoint",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(false, true)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "rollback",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, false)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "reset",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, true)
                    .with_reset(false)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
                ),
                (
                    "prompt-cache",
                    StateMechanismCapabilities::new(
                        paged_full.state().components().iter().cloned(),
                    )
                    .with_transactions(true, true)
                    .with_reset(true)
                    .with_prompt_cache(false)
                    .with_observation_retention(true),
                ),
            ] {
                let error = eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &paged_request,
                    &capabilities_with(&paged_full, paged_full.operators(), state),
                )
                .unwrap_err();
                assert!(
                    error.issues().iter().any(|issue| issue.contains(facility)),
                    "{name}: missing {facility} diagnostic: {error}"
                );
            }

            std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn routed_prediction_and_media_graphs_are_ineligible_without_production_work() {
        use eredu_architectures::replicated_text::ReplicatedTextIneligibility;

        super::super::path_instrumentation::reset();
        let mut routed = qwen_hybrid_config();
        routed["model_type"] = "qwen3_next".into();
        routed["num_experts"] = 2.into();
        routed["num_experts_per_tok"] = 1.into();

        let mut nemotron_prediction = nemotron_h_config();
        nemotron_prediction["num_nextn_predict_layers"] = 1.into();
        nemotron_prediction["mtp_hybrid_override_pattern"] = "*E".into();

        let mut qwen_prediction = qwen_hybrid_config();
        qwen_prediction["mtp_num_hidden_layers"] = 1.into();

        let text = qwen_hybrid_config();
        let media = serde_json::json!({
            "model_type": "qwen3_5",
            "image_token_id": 60,
            "video_token_id": 61,
            "text_config": text,
            "vision_config": {
                "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "num_heads": 2, "num_position_embeddings": 16,
                "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "out_hidden_size": 32
            }
        });

        for (name, config, expected) in [
            ("routed", routed, ReplicatedTextIneligibility::Routed),
            (
                "nemotron prediction",
                nemotron_prediction,
                ReplicatedTextIneligibility::EmbeddedPrediction,
            ),
            (
                "qwen prediction",
                qwen_prediction,
                ReplicatedTextIneligibility::EmbeddedPrediction,
            ),
            ("media", media, ReplicatedTextIneligibility::CompositeInput),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
            let error =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .expect_err("excluded graph entered replicated text admission");
            assert!(
                matches!(
                    error,
                    eredu_architectures::replicated_text::ReplicatedTextRequirementsError::Ineligible(actual)
                        if actual == expected
                ),
                "{name}: {error}"
            );
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn deepseek_prediction_artifact_selects_one_neutral_target_before_payload_work() {
        let mut config = routed_deepseek_v3_config();
        config["num_nextn_predict_layers"] = 1.into();
        let artifact = tiny_heterogeneous_artifact(config);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            2,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let policy = options.preparation_policy().unwrap();
        super::super::path_instrumentation::reset();

        let selected = super::super::loading::select_preparation(&inspection, options, policy)
            .expect("prediction target projection must enter neutral routed admission");

        assert!(selected.communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn invalid_source_and_missing_grouped_mechanism_never_reach_production_paths() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );

        let first = &requirements.parameters()[0];
        let invalid = ReplicatedTextParameterRequirement::new(
            first.name(),
            first.sources().to_vec(),
            first.physical_sources().to_vec(),
            first.aliases().to_vec(),
            Some(SourceTensorEncoding::Safetensors(StoredDtype::U8)),
            first.physical_shape().map(<[usize]>::to_vec),
            first.logical_shape().to_vec(),
            first.native_executable(),
            first.role(),
            first.owner().clone(),
            first.presence().clone(),
            first.transform_constraint(),
        )
        .unwrap();
        let mut parameters = requirements.parameters().to_vec();
        parameters[0] = invalid;
        let invalid_requirements = ReplicatedTextRequirements::new(
            requirements.architecture_identity().to_owned(),
            requirements.operators(),
            requirements.execution_graph().clone(),
            requirements.execution_units().clone(),
            requirements.group_transports().to_vec(),
            requirements.state_layout().clone(),
            requirements.state_access(),
            parameters,
        )
        .unwrap();
        let error = eredu_runtime::select_replicated_text_realization(
            &invalid_requirements,
            &request,
            &capabilities(&invalid_requirements, &request),
        )
        .unwrap_err();
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight lowering")));

        let routed_root = tiny_artifact("qwen3_moe", false);
        let routed =
            eredu_architectures::configuration::inspect_artifact(routed_root.path()).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let routed_options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            128,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        );
        let routed_policy = routed_options.preparation_policy().unwrap();
        let error = super::super::loading::select_preparation_with_grouped_capabilities(
            &routed,
            routed_options,
            routed_policy,
            &[GroupedOperationRequirement::GatedProduct],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("GatedProductTensorParallelPartial"));

        let affine_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request
                .clone()
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 128,
                    bits: 4,
                }),
            &capabilities(&requirements, &request),
        )
        .unwrap_err();
        assert!(affine_error
            .issues()
            .iter()
            .any(|issue| issue.contains("affine group size")));

        let linear_index = requirements
            .parameters()
            .iter()
            .position(|parameter| {
                matches!(
                    parameter.transform_constraint(),
                    ParameterTransformConstraint::Linear { .. }
                )
            })
            .unwrap();
        let linear = &requirements.parameters()[linear_index];
        let ParameterTransformConstraint::Linear { packed_axis } = linear.transform_constraint()
        else {
            unreachable!()
        };
        let mut logical_shape = linear.logical_shape().to_vec();
        logical_shape[packed_axis] = 48;
        let invalid_mxfp4 = ReplicatedTextParameterRequirement::new(
            linear.name(),
            linear.sources().to_vec(),
            linear.physical_sources().to_vec(),
            linear.aliases().to_vec(),
            linear.source_encoding().cloned(),
            Some(logical_shape.clone()),
            logical_shape,
            linear.native_executable(),
            linear.role(),
            linear.owner().clone(),
            linear.presence().clone(),
            linear.transform_constraint(),
        )
        .unwrap();
        let mut parameters = requirements.parameters().to_vec();
        parameters[linear_index] = invalid_mxfp4;
        let invalid_mxfp4_requirements = ReplicatedTextRequirements::new(
            requirements.architecture_identity().to_owned(),
            requirements.operators(),
            requirements.execution_graph().clone(),
            requirements.execution_units().clone(),
            requirements.group_transports().to_vec(),
            requirements.state_layout().clone(),
            requirements.state_access(),
            parameters,
        )
        .unwrap();
        let mxfp4_request = request
            .clone()
            .with_quantization(eredu_core::QuantizationRequest::MxFp4);
        let mxfp4_error = eredu_runtime::select_replicated_text_realization(
            &invalid_mxfp4_requirements,
            &mxfp4_request,
            &capabilities(&invalid_mxfp4_requirements, &mxfp4_request),
        )
        .unwrap_err();
        assert!(mxfp4_error
            .issues()
            .iter()
            .any(|issue| issue.contains("MXFP4 packed extent 48")));

        let full = capabilities(&requirements, &request);
        let only_basic = BackendMechanismCapabilities::new(
            full.operators(),
            full.weight_lowerings().to_vec(),
            vec![WeightResidencyMechanism::Resident],
            StateMechanismCapabilities::new(full.state().components().iter().map(|mechanism| {
                StateComponentMechanism::new(
                    mechanism.layer(),
                    mechanism.component().clone(),
                    Some(StateComponentPlacement::Device),
                    None,
                )
            })),
        );
        let paged = CacheResidencyPolicy::Paged(PagedCacheOptions::new(4, 4096, 4096, 1).unwrap());
        let state_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                paged,
            ),
            &only_basic,
        )
        .unwrap_err();
        assert!(state_error
            .issues()
            .iter()
            .any(|issue| issue.contains("state component")));
        let session_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request
                .clone()
                .with_session(eredu_core::SessionCapabilities::new(true, false, false)),
            &only_basic,
        )
        .unwrap_err();
        assert!(session_error
            .issues()
            .iter()
            .any(|issue| issue.contains("session capability")));
        let residency_error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::DenseDiskStream(
                    eredu_runtime::DenseDiskStreamLoadOptions::default(),
                ),
                CacheResidencyPolicy::Device,
            ),
            &only_basic,
        )
        .unwrap_err();
        assert!(residency_error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight residency")));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn dense_generic_addressable_request_rejects_before_production_paths() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let options = crate::MlxLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        );
        let policy = options.preparation_policy().unwrap();
        let error = super::super::loading::select_preparation_with_grouped_capabilities(
            &inspection,
            options,
            policy,
            &GROUPED_OPERATION_CAPABILITIES,
        )
        .expect_err("dense replicated text silently discarded addressable residency");
        assert!(matches!(
            error,
            Error::Artifact(eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(
                ref detail
            )) if detail.contains("independent expert caching")
                && detail.contains("llama")
        ));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn routed_selection_aggregates_text_and_addressable_mechanism_denials() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("qwen3_moe", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
        let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = eredu_architectures::RoutedTextSelectionRequest::new(
            text,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        )
        .unwrap();
        let full = capabilities(requirements.text(), request.text());
        let incomplete = BackendMechanismCapabilities::new(
            full.operators(),
            full.weight_lowerings().to_vec(),
            full.weight_residencies().to_vec(),
            full.state().clone(),
        )
        .with_session(full.session())
        .with_prompt_cache(full.prompt_cache())
        .with_exact_completion(full.exact_completion());
        let error = eredu_architectures::select_routed_text_realization(
            &requirements,
            &request,
            &incomplete,
        )
        .expect_err("incomplete addressable mechanisms were admitted");
        let grouped = error
            .issues()
            .iter()
            .position(|issue| issue.contains("grouped operation"))
            .expect("grouped-operation denial");
        let indexed = error
            .issues()
            .iter()
            .position(|issue| issue.contains("indexed selection"))
            .expect("indexed-movement denial");
        let storage = error
            .issues()
            .iter()
            .position(|issue| issue.contains("addressable storage"))
            .expect("addressable-storage denial");
        assert!(grouped < indexed && indexed < storage, "{error}");
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn routed_selection_rejects_top_two_when_scratch_holds_only_one_member() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("qwen3_moe", false);
        let config_path = root.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["num_experts_per_tok"] = 2.into();
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
        assert_eq!(requirements.routes_per_token(), 2);
        let one_member = requirements
            .catalog()
            .units()
            .iter()
            .filter_map(eredu_architectures::ExpertResidencyUnit::byte_len)
            .max()
            .unwrap();
        let bank = eredu_runtime::ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::default(),
            one_member,
            one_member,
        )
        .unwrap();
        let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = eredu_architectures::RoutedTextSelectionRequest::new(
            text,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                bank,
            ),
        )
        .unwrap();
        let error = eredu_architectures::select_routed_text_realization(
            &requirements,
            &request,
            &capabilities(requirements.text(), request.text()),
        )
        .expect_err("top-two route was admitted into one-member scratch");
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("one routed token row") && issue.contains("2 routes")));
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn routed_selection_rejects_each_missing_addressable_storage_tier_before_construction() {
        super::super::path_instrumentation::reset();
        let root = tiny_artifact("qwen3_moe", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
        let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = eredu_architectures::RoutedTextSelectionRequest::new(
            text,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        )
        .unwrap();
        let full = capabilities(requirements.text(), request.text());
        for (expected, tiers) in [
            (
                "addressable disk storage",
                eredu_runtime::AddressableStorageTiers::new(true, true, false),
            ),
            (
                "addressable host storage",
                eredu_runtime::AddressableStorageTiers::new(true, false, true),
            ),
            (
                "addressable device storage",
                eredu_runtime::AddressableStorageTiers::new(false, true, true),
            ),
        ] {
            let capabilities = BackendMechanismCapabilities::new(
                full.operators(),
                full.weight_lowerings().to_vec(),
                full.weight_residencies().to_vec(),
                full.state().clone(),
            )
            .with_session(full.session())
            .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
            .with_indexed_movement(true)
            .with_addressable_storage(
                eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                    .with_tiers(tiers),
            )
            .with_prompt_cache(full.prompt_cache())
            .with_exact_completion(full.exact_completion());
            let error = eredu_architectures::select_routed_text_realization(
                &requirements,
                &request,
                &capabilities,
            )
            .expect_err("missing storage tier was admitted");
            assert!(
                error.issues().iter().any(|issue| issue == expected),
                "{error}"
            );
        }
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            super::super::path_instrumentation::Counts::default()
        );
    }

    #[test]
    fn selected_paged_state_controls_generic_construction() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["llama", "qwen3"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let state = CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            );
            let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                state.clone(),
            );
            let selected = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities(&requirements, &request),
            )
            .unwrap();
            assert_eq!(selected.state().policy(), &state);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let architecture_plan = plan.inspection().architecture_plan().clone();
            let artifact = plan.into_artifact();
            let eredu_core::ModelArtifact::SafeTensors {
                configuration,
                tensors,
                shards,
                ..
            } = artifact
            else {
                panic!("expected SafeTensors fixture")
            };
            let prepared = super::super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                super::super::loading::prepared_safetensors_architecture(&architecture_plan)
                    .unwrap()
                    .clone(),
                tensors,
                shards,
                1,
            )
            .unwrap();
            let executable =
                eredu_architectures::replicated_text::visit_replicated_text_architecture::<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                    _,
                >(
                    &architecture_plan,
                    selected,
                    prepared.store(),
                    &stream,
                    BindingVisitor {
                        stream: &stream,
                        weights_stream: &weights_stream,
                    },
                )
                .unwrap();
            assert!(executable.cache_residency_report().unwrap().is_some());
        }
    }

    #[test]
    fn heterogeneous_requirements_are_invariant_across_caller_policies() {
        for config in [
            lfm2_config(),
            kimi_linear_config(),
            nemotron_h_config(),
            qwen_next_config(),
            qwen_hybrid_config(),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let expected =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let requests = [
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::FullyResident,
                    CacheResidencyPolicy::Device,
                ),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(
                        eredu_runtime::LayerwiseLoadOptions::default(),
                    ),
                    CacheResidencyPolicy::Paged(
                        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                            .unwrap()
                            .with_full_attention(true),
                    ),
                )
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                })
                .with_session(eredu_core::SessionCapabilities::new(true, true, true))
                .with_prompt_cache(true)
                .with_exact_completion(true),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(
                        eredu_runtime::DenseDiskStreamLoadOptions::default(),
                    ),
                    CacheResidencyPolicy::Device,
                )
                .with_quantization(eredu_core::QuantizationRequest::MxFp4),
            ];
            for request in requests {
                assert!(matches!(
                    request.residency(),
                    eredu_runtime::LayerWeightResidency::FullyResident
                        | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                ));
                assert_eq!(
                    expected,
                    eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn public_handoff_executes_selected_load_time_transform() {
        let (stream, weights_stream) = execution_streams();
        for (model_type, request) in [
            (
                "llama",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("llama", eredu_core::QuantizationRequest::MxFp4),
            (
                "qwen3",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("qwen3", eredu_core::QuantizationRequest::MxFp4),
        ] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let options = crate::MlxLoadRequest::with_quantization(request);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            assert!(model.materialization_report().is_some());
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn heterogeneous_generic_handoff_executes_selected_load_time_transforms() {
        let (stream, weights_stream) = execution_streams();
        let mut lfm_affine = lfm2_config();
        lfm_affine["hidden_size"] = 32.into();
        lfm_affine["intermediate_size"] = 32.into();
        lfm_affine["num_key_value_heads"] = 1.into();
        lfm_affine["block_auto_adjust_ff_dim"] = false.into();
        let mut kimi_affine = kimi_linear_config();
        kimi_affine["hidden_size"] = 32.into();
        kimi_affine["intermediate_size"] = 32.into();
        kimi_affine["kv_lora_rank"] = 32.into();
        kimi_affine["moe_intermediate_size"] = 32.into();
        kimi_affine["linear_attn_config"]["num_heads"] = 4.into();
        kimi_affine["linear_attn_config"]["head_dim"] = 32.into();
        kimi_affine["num_attention_heads"] = 4.into();
        kimi_affine["qk_nope_head_dim"] = 24.into();
        kimi_affine["qk_rope_head_dim"] = 8.into();
        kimi_affine["v_head_dim"] = 8.into();
        let mut nemotron_affine = nemotron_h_config();
        nemotron_affine["hidden_size"] = 32.into();
        nemotron_affine["intermediate_size"] = 32.into();
        nemotron_affine["num_attention_heads"] = 8.into();
        nemotron_affine["num_key_value_heads"] = 4.into();
        nemotron_affine["mamba_num_heads"] = 8.into();
        nemotron_affine["moe_intermediate_size"] = 32.into();
        nemotron_affine["moe_shared_expert_intermediate_size"] = 32.into();
        let mut lfm_mxfp4 = lfm2_config();
        lfm_mxfp4["hidden_size"] = 32.into();
        lfm_mxfp4["intermediate_size"] = 64.into();
        lfm_mxfp4["num_key_value_heads"] = 1.into();
        lfm_mxfp4["block_auto_adjust_ff_dim"] = false.into();
        let mut qwen_mxfp4 = qwen_hybrid_config();
        qwen_mxfp4["intermediate_size"] = 64.into();
        for (name, config, request) in [
            (
                "lfm2-affine",
                lfm_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "kimi-affine",
                kimi_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "nemotron-affine",
                nemotron_affine,
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            (
                "lfm2-mxfp4",
                lfm_mxfp4,
                eredu_core::QuantizationRequest::MxFp4,
            ),
            (
                "qwen-mxfp4",
                qwen_mxfp4,
                eredu_core::QuantizationRequest::MxFp4,
            ),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let options = crate::MlxLoadRequest::with_quantization(request);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let report = model
                .materialization_report()
                .unwrap_or_else(|| panic!("{name}: no materialization report"));
            assert!(report.transformed_weights > 0, "{name}");
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            generic
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn public_handoff_executes_admitted_gguf_mapping() {
        let (stream, weights_stream) = execution_streams();
        let artifacts = [
            tiny_llama_gguf("llama", None, &stream),
            tiny_llama_gguf("mistral", None, &stream),
            tiny_qwen_gguf("qwen2", None, &stream),
            tiny_qwen_gguf("qwen3", None, &stream),
        ];
        for artifact in artifacts {
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap();
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            let logits = executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap();
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
    }

    #[test]
    fn both_text_architectures_execute_checkpoint_native_packed_gguf_formats() {
        let (stream, weights_stream) = execution_streams();
        for (architecture, format) in [
            ("llama", eredu_gguf::GgmlType::Q4_0),
            ("llama", eredu_gguf::GgmlType::MxFp4),
            ("llama", eredu_gguf::GgmlType::IQ4NL),
            ("qwen2", eredu_gguf::GgmlType::Q4_0),
            ("qwen2", eredu_gguf::GgmlType::MxFp4),
            ("qwen2", eredu_gguf::GgmlType::IQ4NL),
        ] {
            let artifact = if architecture == "llama" {
                tiny_llama_gguf(architecture, Some(format), &stream)
            } else {
                tiny_qwen_gguf(architecture, Some(format), &stream)
            };
            let checkpoint = eredu_gguf::Checkpoint::open(artifact.path()).unwrap();
            let translated = if architecture == "llama" {
                checkpoint
                    .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
                    .unwrap()
            } else {
                checkpoint
                    .translated_outputs(|name| {
                        eredu_architectures::qwen::translate_gguf_weight_name(name, false)
                    })
                    .unwrap()
            };
            if matches!(
                format,
                eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::MxFp4
            ) {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".scales")
                        && mapping.layout.name.ends_with(".scales")
                        && mapping.layout.name.starts_with("model.")
                }));
            }
            if format == eredu_gguf::GgmlType::Q4_0 {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".biases")
                        && mapping.layout.name.ends_with(".biases")
                }));
            }
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            assert!(requirements.parameters().iter().any(|parameter| {
                matches!(
                    (format, parameter.native_executable()),
                    (eredu_gguf::GgmlType::Q4_0, LinearFormat::Affine(_))
                        | (eredu_gguf::GgmlType::MxFp4, LinearFormat::MxFp4)
                        | (eredu_gguf::GgmlType::IQ4NL, LinearFormat::GgufIQuant { .. })
                )
            }));
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{architecture} {format:?}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn generic_controls_cover_residency_cache_persistence_and_observation() {
        struct Observer {
            activation: bool,
            logits: bool,
            intervened: bool,
            stream: Stream,
        }
        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
                self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        let mut host = eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 7).unwrap(),
        );
        host = host.with_max_cached_shards(3);
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 4).unwrap();
        for (model_type, residency) in ["llama", "qwen2"].into_iter().flat_map(|family| {
            [
                eredu_runtime::WeightResidency::fully_resident(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
            ]
            .into_iter()
            .map(move |residency| (family, residency))
        }) {
            let root = tiny_artifact(model_type, false);
            let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true);
            let options = crate::MlxLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(CacheResidencyPolicy::Paged(paged.clone()));
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
            assert!(model.residency_report().unwrap().is_some());
            assert_eq!(
                model.dense_stream_report().unwrap().is_some(),
                matches!(
                    residency,
                    eredu_runtime::WeightResidency::Layers(
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    )
                )
            );
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            assert_eq!(generic.selected_residency(), residency.layers());
            generic
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();

            let mut observer = Observer {
                activation: false,
                logits: false,
                intervened: false,
                stream: stream.clone(),
            };
            let replacement = generic
                .forward_with_observer(
                    &Array::from_slice(&[3_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap();
            let replacement = replacement.evaluated().unwrap();
            assert!(observer.logits);
            assert!(observer.activation);
            assert!(observer.intervened);
            assert!(replacement
                .as_slice::<f32>()
                .iter()
                .all(|value| *value == 0.0));

            let identity = generic.prompt_cache_model_identity().clone();
            let descriptor = PromptCacheDescriptor::from_model_identity(
                identity,
                "tiny-checkpoint",
                "tokens:1,2,3",
                1,
            )
            .unwrap();
            let cache_root = tempfile::tempdir().unwrap();
            let destination = cache_root.path().join("cache");
            let prefix = [1_u32, 2, 3];
            generic.reset_cache().unwrap();
            generic
                .decode(&Array::from_slice(&prefix, &[1, 3]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let manifest = generic
                .save_prompt_cache(
                    &destination,
                    descriptor.clone(),
                    &prefix,
                    &PromptCacheOptions::default(),
                )
                .unwrap();
            assert_eq!(manifest.block_size_tokens, paged.block_size_tokens());
            let incompatible = descriptor
                .clone()
                .with_architecture_fingerprint(format!(
                    "{}-different",
                    descriptor.architecture_fingerprint()
                ))
                .unwrap();
            assert!(generic
                .load_prompt_cache(&destination, &incompatible, &prefix)
                .is_err());
            generic
                .load_prompt_cache(&destination, &descriptor, &prefix)
                .unwrap();
            assert!(generic.cache_residency_report().unwrap().is_some());
            generic
                .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn heterogeneous_generic_sessions_preserve_every_state_component_across_controls() {
        struct Observer {
            activation: bool,
            logits: bool,
            intervened: bool,
            stream: Stream,
        }
        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
                self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    self.intervened = true;
                    Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
                } else {
                    Ok(None)
                }
            }
        }

        let (stream, weights_stream) = execution_streams();
        let host = eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 3).unwrap(),
        )
        .with_max_cached_shards(2);
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let cases = vec![
            (
                "lfm2",
                lfm2_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Device,
            ),
            (
                "lfm2-paged",
                lfm2_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "kimi_linear",
                kimi_linear_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Device,
            ),
            (
                "kimi_linear-paged",
                kimi_linear_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "nemotron_h",
                nemotron_h_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Device,
            ),
            (
                "nemotron_h-paged",
                nemotron_h_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "qwen3_next",
                qwen_next_config(),
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Device,
            ),
            (
                "qwen3_next-paged",
                qwen_next_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
            (
                "qwen3_5_text",
                qwen_hybrid_config(),
                eredu_runtime::WeightResidency::layerwise_host(host),
                CacheResidencyPolicy::Device,
            ),
            (
                "qwen3_5_text-paged",
                qwen_hybrid_config(),
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Paged(paged),
            ),
        ];

        for (name, config, residency, state_policy) in cases {
            let root = tiny_heterogeneous_artifact(config);
            let options = crate::MlxLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(state_policy.clone());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(model.residency_report().unwrap().is_some());
            assert_eq!(
                model.dense_stream_report().unwrap().is_some(),
                matches!(
                    residency,
                    eredu_runtime::WeightResidency::Layers(
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    )
                ),
                "{name}"
            );
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            assert_eq!(generic.selected_residency(), residency.layers(), "{name}");
            assert_eq!(
                generic.cache_residency_report().unwrap().is_some(),
                matches!(state_policy, CacheResidencyPolicy::Paged(_)),
                "{name}"
            );

            let prefix = [1_u32, 2, 3];
            let prompt = Array::from_slice(&prefix, &[1, 3]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let saved_snapshot = generic.state_snapshot();
            let saved_numeric = generic
                .fixed_numeric_state_snapshot()
                .unwrap_or_else(|error| panic!("{name} numeric snapshot: {error}"));
            assert!(!saved_numeric.is_empty(), "{name}");
            assert!(saved_snapshot
                .iter()
                .flat_map(|(_, fixed)| fixed)
                .all(|(_, present)| *present));

            let persisted = matches!(state_policy, CacheResidencyPolicy::Paged(_)).then(|| {
                let identity = generic.prompt_cache_model_identity().clone();
                let descriptor = PromptCacheDescriptor::from_model_identity(
                    identity,
                    format!("{name}-checkpoint"),
                    "tokens:1,2,3",
                    1,
                )
                .unwrap();
                let cache_root = tempfile::tempdir().unwrap();
                let destination = cache_root.path().join("cache");
                generic
                    .save_prompt_cache(
                        &destination,
                        descriptor.clone(),
                        &prefix,
                        &PromptCacheOptions::default(),
                    )
                    .unwrap();
                (cache_root, destination, descriptor)
            });
            let continuation_token = Array::from_slice(&[4_u32], &[1, 1]);
            let persistence_baseline = persisted
                .as_ref()
                .map(|_| generic.checkpoint_restore_probe(&continuation_token, &stream))
                .transpose()
                .unwrap_or_else(|error| panic!("{name} persistence baseline: {error}"));
            generic.reset_cache().unwrap();
            assert!(generic.state_snapshot().iter().all(|(position, fixed)| {
                *position == 0 && fixed.iter().all(|(_, present)| !present)
            }));
            if let Some((_cache_root, destination, descriptor)) = persisted {
                let incompatible = descriptor
                    .clone()
                    .with_architecture_fingerprint(format!(
                        "{}-different",
                        descriptor.architecture_fingerprint()
                    ))
                    .unwrap();
                assert!(generic
                    .load_prompt_cache(&destination, &incompatible, &prefix)
                    .is_err());
                generic
                    .load_prompt_cache(&destination, &descriptor, &prefix)
                    .unwrap();
            } else {
                assert!(generic
                    .save_prompt_cache(
                        tempfile::tempdir().unwrap().path(),
                        PromptCacheDescriptor::from_model_identity(
                            generic.prompt_cache_model_identity().clone(),
                            format!("{name}-checkpoint"),
                            "tokens:1,2,3",
                            1,
                        )
                        .unwrap(),
                        &prefix,
                        &PromptCacheOptions::default(),
                    )
                    .is_err());
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
            }
            assert_eq!(generic.state_snapshot(), saved_snapshot, "{name}");
            assert_eq!(
                generic.fixed_numeric_state_snapshot().unwrap(),
                saved_numeric,
                "{name} fixed tensors changed across prompt-cache restoration"
            );
            let probe = generic
                .checkpoint_restore_probe(&continuation_token, &stream)
                .unwrap_or_else(|error| panic!("{name} checkpoint/restore: {error}"));
            if let Some(baseline) = persistence_baseline {
                assert_eq!(
                    probe, baseline,
                    "{name} continuation changed after prompt-cache restoration"
                );
            }
            let (
                before,
                advanced,
                restored,
                before_numeric,
                advanced_numeric,
                restored_numeric,
                continuation,
            ) = probe;
            assert_eq!(before, saved_snapshot, "{name}");
            assert_ne!(advanced, before, "{name}");
            assert_eq!(restored, before, "{name}");
            assert_eq!(before_numeric, saved_numeric, "{name}");
            assert_ne!(advanced_numeric, before_numeric, "{name}");
            assert_eq!(restored_numeric, before_numeric, "{name}");
            let replayed = generic.decode(&continuation_token, &stream).unwrap();
            let replayed = replayed.evaluated().unwrap();
            assert_eq!(
                replayed.as_slice::<f32>(),
                continuation.as_slice(),
                "{name}"
            );
            assert_eq!(generic.state_snapshot(), advanced, "{name}");
            assert_eq!(
                generic.fixed_numeric_state_snapshot().unwrap(),
                advanced_numeric,
                "{name}"
            );

            let mut observer = Observer {
                activation: false,
                logits: false,
                intervened: false,
                stream: stream.clone(),
            };
            let replacement = generic
                .forward_with_observer(
                    &Array::from_slice(&[5_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap();
            let replacement = replacement.evaluated().unwrap();
            assert!(observer.activation && observer.logits, "{name}");
            assert!(observer.intervened, "{name}");
            assert!(
                replacement
                    .as_slice::<f32>()
                    .iter()
                    .all(|value| *value == 0.0),
                "{name}"
            );
        }
    }

    #[test]
    fn heterogeneous_logits_and_fixed_state_match_across_weight_and_state_residency() {
        let (stream, weights_stream) = execution_streams();
        let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        for (name, config) in [
            ("lfm2", lfm2_config()),
            ("kimi_linear", kimi_linear_config()),
            ("nemotron_h", nemotron_h_config()),
            ("qwen3_next", qwen_next_config()),
            ("qwen3_5_text", qwen_hybrid_config()),
        ] {
            let root = tiny_heterogeneous_artifact(config);
            let mut results = Vec::new();
            for (residency, state) in [
                (
                    eredu_runtime::WeightResidency::fully_resident(),
                    CacheResidencyPolicy::Device,
                ),
                (
                    eredu_runtime::WeightResidency::dense_disk_stream(disk),
                    CacheResidencyPolicy::Paged(paged.clone()),
                ),
            ] {
                let options = crate::MlxLoadRequest::default()
                    .with_weight_residency(residency)
                    .with_state_residency(state);
                let inspection =
                    eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
                let plan = eredu_core::plan_model_preparation(
                    inspection,
                    options.preparation_policy().unwrap(),
                    eredu_core::SessionCapabilities::default(),
                )
                .unwrap();
                let model = materialize_model_plan(plan, options, &stream, &weights_stream)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let mut executable = model.into_executable();
                let generic = executable.erased_mut();
                let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
                let parts = [input::token_ids_part(&prompt).unwrap()];
                generic
                    .prefill(input::ModelInput::new(&parts), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap();
                let logits = generic
                    .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                assert!(
                    logits.iter().all(|value| value.is_finite())
                        && logits.iter().any(|value| value.abs() > 1e-12),
                    "{name}: {logits:?}"
                );
                results.push((
                    logits,
                    generic.state_snapshot(),
                    generic.fixed_numeric_state_snapshot().unwrap(),
                ));
            }
            let (resident_logits, resident_semantics, resident_fixed) = &results[0];
            let (bounded_logits, bounded_semantics, bounded_fixed) = &results[1];
            assert_eq!(resident_semantics, bounded_semantics, "{name}");
            assert_eq!(resident_fixed.len(), bounded_fixed.len(), "{name}");
            for (left, right) in resident_logits.iter().zip(bounded_logits) {
                assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
            }
            for (left, right) in resident_fixed.iter().zip(bounded_fixed) {
                assert_eq!((&left.0, &left.1, &left.2), (&right.0, &right.1, &right.2));
                for (left, right) in left.3.iter().zip(&right.3) {
                    assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
                }
            }
        }
    }
}
