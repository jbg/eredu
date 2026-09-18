use super::*;

pub(super) mod parameter_representation;

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
        crate::composition::mlx::speculative::IndependentLogits,
        Error,
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
            .map_err(eredu_nn::Error::backend_retained_source)
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
            .map_err(eredu_nn::Error::backend_retained_source)
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

pub(super) enum MlxSelectedLayerwisePolicyInner<U: 'static, P> {
    Resident(MlxResidentPolicy<U>),
    Bounded {
        policy: MlxLayerwisePolicy<U, P>,
        local_addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    },
}

pub(super) struct MlxSelectedLayerwisePolicy<U: 'static, P> {
    inner: Arc<Mutex<MlxSelectedLayerwisePolicyInner<U, P>>>,
    /// Exact architecture addresses paired with policy-local residency addresses.
    /// Pipeline cuts may retain a nonzero global unit index at local ordinal zero.
    parameter_locations: Arc<
        [(
            eredu_runtime::ExecutionUnitAddress,
            eredu_runtime::ExecutionUnitAddress,
        )],
    >,
}

pub(super) type MlxArchitectureLayerwisePolicy<A, S> = MlxSelectedLayerwisePolicy<
    <A as LayeredArchitecture<MlxNeuralBackend, S>>::Unit,
    MlxSelectiveUnitPopulator,
>;

impl<U: 'static, P> Clone for MlxSelectedLayerwisePolicy<U, P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            parameter_locations: Arc::clone(&self.parameter_locations),
        }
    }
}

impl<U: 'static, P> MlxSelectedLayerwisePolicy<U, P> {
    fn operation_policy(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, MlxSelectedLayerwisePolicyInner<U, P>>, Error> {
        self.inner.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => Error::PrefillScopeReentrant,
            std::sync::TryLockError::Poisoned(_) => {
                Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Poisoned)
            }
        })
    }

    pub(super) fn original_operation_plan<'source>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        retained_sources: Option<
            &'source crate::backend::runtime::execution::generic::LayerwiseWorkspace,
        >,
        groups: eredu_runtime::GroupSubmissionMechanism,
    ) -> Result<
        crate::backend::runtime::execution::generic::SelectedOriginalOperationPlan<'source, U>,
        Error,
    > {
        let selected = self.operation_policy()?;
        match &*selected {
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => policy
                .original_operation_plan(geometry, retained_sources, groups)
                .map(crate::backend::runtime::execution::generic::SelectedOriginalOperationPlan::Bounded),
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                if retained_sources.is_some() {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
                policy
                    .original_neural_plan(geometry, groups)
                    .map(crate::backend::runtime::execution::generic::SelectedOriginalOperationPlan::Resident)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn inspect_operation_sources_for_test(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) {
        let selected = self.operation_policy().expect("available selected policy");
        match &*selected {
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.inspect_operation_sources_for_test(geometry);
            }
            MlxSelectedLayerwisePolicyInner::Resident(_) => {
                panic!("bounded inspection requires the retained bounded policy");
            }
        }
    }

    fn parameter_locations(
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<
        Arc<
            [(
                eredu_runtime::ExecutionUnitAddress,
                eredu_runtime::ExecutionUnitAddress,
            )],
        >,
        Error,
    > {
        if layout.len() != addresses.len() {
            return Err(Error::ArchitectureModel(
                "prepared parameter addresses differ from policy units".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut locations = Vec::with_capacity(addresses.len());
        for (ordinal, address) in addresses.iter().copied().enumerate() {
            let local = layout.address(ordinal).expect("validated policy layout");
            if local.group() != address.group() || !seen.insert((address.group(), address.index()))
            {
                return Err(Error::ArchitectureModel(
                    "prepared parameter addresses differ from policy groups".into(),
                ));
            }
            locations.push((address, local));
        }
        Ok(locations.into())
    }

    pub(super) fn resident(
        policy: MlxResidentPolicy<U>,
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<Self, Error> {
        Ok(Self {
            parameter_locations: Self::parameter_locations(layout, addresses)?,
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Resident(
                policy,
            ))),
        })
    }

    pub(super) fn bounded(
        policy: MlxLayerwisePolicy<U, P>,
        layout: &eredu_runtime::ExecutionUnitLayout,
        addresses: &[eredu_runtime::ExecutionUnitAddress],
    ) -> Result<Self, Error> {
        let parameter_locations = Self::parameter_locations(layout, addresses)?;
        let local_addresses = (0..layout.len())
            .map(|ordinal| {
                layout
                    .address(ordinal)
                    .expect("validated execution layout covers every ordinal")
            })
            .collect();
        Ok(Self {
            parameter_locations,
            inner: Arc::new(Mutex::new(MlxSelectedLayerwisePolicyInner::Bounded {
                policy,
                local_addresses,
            })),
        })
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

impl<U: 'static> MlxSelectedLayerwisePolicy<U, MlxSelectiveUnitPopulator> {
    pub(super) fn layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::MetalAllocationFacts,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        self.layerwise_workspace_impl(allocation, None)
    }
    pub(super) fn prepared_layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::MetalAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        let frames = [
            std::mem::size_of::<Option<&eredu_nn::workspace::WorkspaceContext>>(),
            std::mem::size_of::<
                std::sync::MutexGuard<
                    '_,
                    MlxSelectedLayerwisePolicyInner<U, MlxSelectiveUnitPopulator>,
                >,
            >(),
            std::mem::size_of::<
                Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error>,
            >(),
            std::mem::size_of::<crate::backend::nn::workspace::MetalAllocationFacts>(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                    ))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.layerwise_workspace_impl(allocation, Some(context))
    }
    fn layerwise_workspace_impl(
        &self,
        allocation: crate::backend::nn::workspace::MetalAllocationFacts,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        let policy = match context {
            Some(_) => self.operation_policy(),
            None => self.inner.lock().map_err(|_| {
                Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
            }),
        }?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                let workspace = match context {
                    Some(context) => policy.layerwise_workspace_with_metadata(allocation, context),
                    None => policy.layerwise_workspace(allocation),
                }?;
                workspace.with_parameter_locations(Arc::clone(&self.parameter_locations), context)
            },
            MlxSelectedLayerwisePolicyInner::Resident(_) => Err(match context {
                Some(_) => Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ),
                None => Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )),
            }),
        }
    }
}

pub(super) enum MlxSelectedUnitLease<U: 'static> {
    Resident(MlxResidentUnit<U>),
    Bounded(MlxUnitLease<U>),
}

impl<U: 'static> Deref for MlxSelectedUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U: 'static> DerefMut for MlxSelectedUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Resident(lease) => lease,
            Self::Bounded(lease) => lease,
        }
    }
}

impl<U, P> eredu_runtime::LayerwisePolicy<MlxNeuralBackend, U> for MlxSelectedLayerwisePolicy<U, P>
where
    U: eredu_nn::Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxSelectedUnitLease<U>;
    type Error = Error;

    fn uses_shared_group_executor(&self, stream: &Stream) -> Result<bool, Error> {
        let selected = self.operation_policy()?;
        let policy = match &*selected {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => Err(policy),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => Ok(policy),
        };
        MlxLayerwisePolicy::selected_shared_neural_executor(policy, stream)
    }

    fn submit_group(
        &mut self,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<
        impl eredu_runtime::OrderedLayerwiseCompletion<Stream> + 'static,
        impl std::error::Error + Send + Sync + 'static,
    > {
        let selected = self.operation_policy()?;
        let policy = match &*selected {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => Err(policy),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => Ok(policy),
        };
        MlxLayerwisePolicy::submit_selected_neural(policy, stream, value)
    }

    fn resident_parameters_available(&self) -> bool {
        let Ok(policy) = self.inner.lock() else {
            return false;
        };
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.resident_parameters_available()
            }
            MlxSelectedLayerwisePolicyInner::Bounded { .. } => false,
        }
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        let policy = self.inner.try_lock().ok()?;
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => policy.retained_value_slot_bound(),
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.retained_value_slot_bound()
            }
        }
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        // Inspection can be reentered by a retained-value callback. Contention
        // and poison are incomplete evidence, never a reason to wait or reap.
        let Ok(policy) = self.inner.try_lock() else {
            return false;
        };
        match &*policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.visit_retained_values(visitor)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.visit_retained_values(visitor)
            }
        }
    }

    fn visit_resident_parameter_sources<V>(
        &self,
        visitor: &mut V,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>
    where
        V: for<'source> eredu_nn::ParameterSourceVisitor<'source, MlxTensor>,
    {
        self.visit_parameter_sources_with_metadata(visitor, context)
    }

    fn visit_resident_units(&mut self, visitor: &mut impl FnMut(&mut U)) -> bool {
        let Ok(mut policy) = self.inner.lock() else {
            return false;
        };
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.visit_resident_units(visitor)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { .. } => false,
        }
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> Result<bool, Error> {
        let mut policy = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("selected layerwise policy lock was poisoned".into())
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.publish_parameter_replacements(values, active)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.publish_parameter_replacements(values, active)
            }
        }
    }

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

    fn inspect_unit<E, F, V>(
        &mut self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        build: F,
        operation: V,
        context: &Stream,
    ) -> Result<bool, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        let &(expected, local) = self.parameter_locations.get(ordinal).ok_or_else(|| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "parameter unit ordinal is outside the selected policy".into(),
            ))
        })?;
        if expected != address {
            return Err(eredu_runtime::LayerwiseAcquireError::Policy(
                Error::ArchitectureModel(
                    "parameter unit address differs from the prepared owner".into(),
                ),
            ));
        }
        let mut policy = self.inner.lock().map_err(|_| {
            eredu_runtime::LayerwiseAcquireError::Policy(Error::ArchitectureModel(
                "selected layerwise policy lock was poisoned".into(),
            ))
        })?;
        match &mut *policy {
            MlxSelectedLayerwisePolicyInner::Resident(policy) => {
                policy.inspect_unit(ordinal, local, build, operation, context)
            }
            MlxSelectedLayerwisePolicyInner::Bounded { policy, .. } => {
                policy.inspect_unit(ordinal, local, build, operation, context)
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

#[cfg(test)]
mod retained_visit_tests;
