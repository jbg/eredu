use super::*;

pub(super) type MlxDirectPartitionExecutor<A> =
    eredu_architectures::partitioned_execution::DirectPartitionExecutor<
        A,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<A, MlxHybridState>,
    >;

pub(super) type MlxDirectPartitionStrategy<A> = eredu_runtime::PartitionedTextExecution<
    MlxDirectPartitionExecutor<A>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::NoBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

pub(super) type MlxSharedAddressableBank =
    crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank;
pub(super) type MlxEmbeddedPredictionObservers =
    eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
        MlxTensor,
        Array,
        Exception,
    >;
#[derive(Clone, Copy, Default)]
pub(super) struct MlxPartitionTensorAllocator;

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

pub(super) fn mlx_boundary_dtype(
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

pub(super) fn mlx_pipeline_activation_dtype(
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

pub(super) type MlxPipelinePartitionExecutor<A, S> =
    eredu_architectures::partitioned_execution::PipelinePartitionExecutor<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxPartitionTensorAllocator,
    >;

pub(super) type MlxPipelinePartitionStrategy<A, S> = eredu_runtime::PartitionedTextExecution<
    MlxPipelinePartitionExecutor<A, S>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::OpaqueBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

pub(super) enum MlxSelectedLayerwisePolicyInner<U, P> {
    Resident(MlxResidentPolicy<U>),
    Bounded {
        policy: MlxLayerwisePolicy<U, P>,
        local_addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    },
}

pub(super) struct MlxSelectedLayerwisePolicy<U, P> {
    inner: Arc<Mutex<MlxSelectedLayerwisePolicyInner<U, P>>>,
}

pub(super) type MlxArchitectureLayerwisePolicy<A, S> = MlxSelectedLayerwisePolicy<
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
    pub(super) fn resident(policy: MlxResidentPolicy<U>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Resident(
                policy,
            ))),
        }
    }

    pub(super) fn bounded(
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

    pub(super) fn residency_report(&self) -> Result<ResidencyReport, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.residency_report(),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.residency_report(),
        }
    }

    pub(super) fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        let policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(_) => Ok(None),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy.dense_stream_report(),
        }
    }
}

pub(super) enum MlxSelectedUnitLease<U> {
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
                    crate::tests::support::path_instrumentation::bounded_unit_acquisition();
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
