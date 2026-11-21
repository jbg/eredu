use super::*;
use std::sync::Arc;

/// MLX communication capability attached to one complete model/session.
///
/// This is the only public owner of topology-derived MLX communicators. Model
/// implementations may borrow its axis groups internally, but callers cannot
/// construct or route around those groups independently of the selected
/// backend session.
#[derive(Debug, Clone)]
pub struct MlxDistributedSession {
    manifest: Arc<eredu_core::communication::CommunicationManifest>,
    communicators: ParallelCommunicators,
    stream: Stream,
    world: NativeGroup,
    pub(super) authority: eredu_core::communication::CommunicationAuthority,
}

impl MlxDistributedSession {
    /// Completes an already constructed ordinary source graph without a host
    /// tensor copy. The retained world also fences every logical group sharing
    /// that native communicator. The nested recovery scope keeps the enclosing
    /// model submission live until this source reaches native terminal evidence.
    pub(crate) fn prepare_capture_source(
        &self,
        tensor: &MlxTensor,
        stream: &Stream,
        wait: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Error> {
        use eredu_core::BoundedCompletion;
        self.ensure_active()?;
        let operation = eredu_core::communication::CommunicationOperation::FailureAgreement;
        let completion = MlxCommunicationCompletion::submit(
            [tensor.as_array()],
            vec![tensor.as_array().clone()],
            vec![],
            vec![self.communicators.control_world().clone()],
            vec![],
            vec![stream.clone()],
        )
        .map_err(|error| self.submission_error(error, operation))?;
        completion
            .with_authority(self.authority.clone(), operation)
            .wait_bounded(wait)
            .map_err(Into::into)
    }

    /// Realizes the groups and routes of one agreed world binding.
    pub fn from_prepared(
        prepared: eredu_core::communication::WorldCommunicationBinding<
            crate::native::NativeWorldManifestTransport<'_>,
        >,
    ) -> Result<Self, Error> {
        let (transport, realization, completion, descriptors, identity, authority) =
            prepared.into_parts();
        let (world, stream) = transport.native_context();
        let manifest = Arc::new(realization.manifest().clone());
        let communicators = ParallelCommunicators::new_with_routes(
            realization,
            world,
            completion,
            descriptors,
            identity,
        )?;
        Ok(Self {
            manifest,
            communicators,
            stream: stream.clone(),
            world: world.clone(),
            authority,
        })
    }

    /// Common setup identity retained alongside the actual communication owner.
    pub fn session_identity(&self) -> eredu_core::communication::CommunicationSessionIdentity {
        self.communicators.session_identity()
    }

    pub(crate) fn has_bounded_control(&self) -> bool {
        self.authority.completion_policy().is_some()
    }

    /// Exact native world from which the retained communication was realized.
    pub(crate) const fn native_world(&self) -> &NativeGroup {
        &self.world
    }

    /// Reports the exact manifest retained by the native realization.
    pub(crate) fn communication_manifest(
        &self,
    ) -> &eredu_core::communication::CommunicationManifest {
        &self.manifest
    }

    /// Validates the retained native realization without creating communication resources.
    pub(crate) fn validate_selected_manifest(
        &self,
        manifest: &eredu_core::communication::CommunicationManifest,
    ) -> Result<(), Error> {
        if self.manifest.as_ref() != manifest {
            return Err(Error::Parallel(
                "native communication differs from the selected manifest".into(),
            ));
        }
        Ok(())
    }

    /// Returns the shared authority for the selected rank communication.
    pub(crate) const fn authority(&self) -> &eredu_core::communication::CommunicationAuthority {
        &self.authority
    }

    /// Returns the communicator for an active opaque group identity.
    pub(crate) fn selected_group(&self, id: CollectiveGroupId) -> Option<&Group> {
        self.communicators.group(id)
    }

    /// Submits hidden activations to the succeeding pipeline coordinate.
    pub(super) fn group(&self, scope: CollectiveScope) -> Result<&Group, Error> {
        match scope {
            CollectiveScope::World => Err(Error::Backend(BackendError::Unsupported {
                backend: "mlx".into(),
                capability: "manifest-realized sessions have no uncontracted world data plane"
                    .into(),
            })),
            CollectiveScope::Group(id) => {
                self.communicators.communication_group(id).ok_or_else(|| {
                    Error::Backend(BackendError::Unsupported {
                        backend: "mlx".into(),
                        capability: format!("collective group {} is inactive", id.value()),
                    })
                })
            }
            _ => Err(Error::Backend(BackendError::Unsupported {
                backend: "mlx".into(),
                capability: "unknown collective scope".into(),
            })),
        }
    }

    fn ensure_active(&self) -> Result<(), Error> {
        self.authority
            .ensure_active()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Reports native bounds for a fixed control gather owned by the text driver.
    pub(crate) fn text_preparation_admission(
        &self,
        words: usize,
    ) -> Result<Option<eredu_core::run_preparation::TextPreparationAdmission>, Error> {
        let Some(policy) = self.manifest.completion_policy() else {
            return Ok(None);
        };
        let (retained_bytes, host_bytes) =
            control_gather_storage(self.manifest.world_size(), words)
                .ok_or_else(|| Error::Parallel("text preparation native bound overflow".into()))?;
        Ok(Some(
            eredu_core::run_preparation::TextPreparationAdmission {
                identity: self.session_identity(),
                native_usage: eredu_core::run_preparation::TextPreparationUsage {
                    attempts: 0,
                    retained_bytes,
                    host_bytes,
                },
                wait: policy.bounded_wait(),
            },
        ))
    }

    fn submission_error(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: eredu_core::communication::CommunicationOperation,
    ) -> Error {
        Error::Other(Box::new(
            self.authority.submission_failure(error, operation, None),
        ))
    }

    fn selected_submission(
        &self,
        output: Array,
        retained: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        group: Group,
        operation: eredu_core::communication::CommunicationOperation,
    ) -> Result<Submission<MlxTensor, DistributedCompletion<MlxTensor>>, Error> {
        let value = MlxTensor::from_array(output.clone());
        let completion = DistributedCompletion::submit_authorized(
            MlxTensor::from_array(output.clone()),
            [&output],
            retained,
            count_buffers,
            vec![group],
            Vec::new(),
            vec![self.stream.clone()],
            self.authority.clone(),
            operation,
        )?;
        Ok(Submission {
            output: value,
            completion,
        })
    }

    fn value_dtype(value: &ValueDescriptor) -> Result<Dtype, Error> {
        match value.dtype() {
            TensorDtype::Bool => Ok(Dtype::Bool),
            TensorDtype::F32 => Ok(Dtype::Float32),
            TensorDtype::F16 => Ok(Dtype::Float16),
            TensorDtype::Bf16 => Ok(Dtype::Bfloat16),
            TensorDtype::I8 => Ok(Dtype::Int8),
            TensorDtype::U8 => Ok(Dtype::Uint8),
            TensorDtype::U16 => Ok(Dtype::Uint16),
            TensorDtype::U32 => Ok(Dtype::Uint32),
            TensorDtype::U64 => Ok(Dtype::Uint64),
            TensorDtype::I16 => Ok(Dtype::Int16),
            TensorDtype::I32 => Ok(Dtype::Int32),
            TensorDtype::I64 => Ok(Dtype::Int64),
            TensorDtype::F64 => Ok(Dtype::Float64),
            TensorDtype::Complex64 => Ok(Dtype::Complex64),
            TensorDtype::Encoded(name) => Err(Error::Backend(BackendError::Unsupported {
                backend: "mlx".into(),
                capability: format!("receiving encoded dtype {name}"),
            })),
        }
    }

    fn value_shape(value: &ValueDescriptor) -> Result<Vec<i32>, Error> {
        value
            .shape()
            .iter()
            .map(|dimension| {
                i32::try_from(*dimension).map_err(|_| {
                    Error::Parallel(format!(
                        "distributed receive dimension {dimension} exceeds i32"
                    ))
                })
            })
            .collect()
    }
}

impl DistributedSession for MlxDistributedSession {
    type Value = MlxTensor;
    type Completion = DistributedCompletion<MlxTensor>;
    type Error = Error;

    fn descriptor(&self) -> DistributedSessionDescriptor {
        DistributedSessionDescriptor::new(
            self.communicators.world_size(),
            self.communicators.global_rank(),
            self.communicators.descriptors(),
        )
        .expect("MLX collective realization is validated")
    }

    fn capabilities(&self) -> DistributedCapabilities {
        DistributedCapabilities::new(false, self.communicators.group_ids(), true, true, true)
    }

    fn all_reduce_sum(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        self.ensure_active()?;
        let group = self.group(scope)?.clone();
        let output =
            distributed::all_sum(input.as_array(), &group, &self.stream).map_err(|error| {
                self.submission_error(
                    error,
                    eredu_core::communication::CommunicationOperation::AllReduceSum,
                )
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_core::communication::CommunicationOperation::AllReduceSum,
        )
    }

    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        self.ensure_active()?;
        let group = self.group(scope)?.clone();
        let output =
            distributed::all_gather(input.as_array(), &group, &self.stream).map_err(|error| {
                self.submission_error(
                    error,
                    eredu_core::communication::CommunicationOperation::AllGatherEven,
                )
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_core::communication::CommunicationOperation::AllGatherEven,
        )
    }

    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        self.ensure_active()?;
        let group = self.group(scope)?.clone();
        let output = distributed::all_to_all_v(
            input.as_array(),
            send_counts,
            receive_counts,
            &group,
            &self.stream,
        )
        .map_err(|error| {
            self.submission_error(
                error,
                eredu_core::communication::CommunicationOperation::VariableAllToAll,
            )
        })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            vec![send_counts.to_vec(), receive_counts.to_vec()],
            group,
            eredu_core::communication::CommunicationOperation::VariableAllToAll,
        )
    }

    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        self.ensure_active()?;
        let group = self.group(scope)?.clone();
        let output =
            distributed::send(input.as_array(), peer, &group, &self.stream).map_err(|error| {
                self.submission_error(
                    error,
                    eredu_core::communication::CommunicationOperation::SendReceive,
                )
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_core::communication::CommunicationOperation::SendReceive,
        )
    }

    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        self.ensure_active()?;
        let shape = Self::value_shape(value)?;
        let group = self.group(scope)?.clone();
        let output = distributed::recv(
            &shape,
            Self::value_dtype(value)?,
            peer,
            &group,
            &self.stream,
        )
        .map_err(|error| {
            self.submission_error(
                error,
                eredu_core::communication::CommunicationOperation::SendReceive,
            )
        })?;
        self.selected_submission(
            output.clone(),
            vec![output],
            Vec::new(),
            group,
            eredu_core::communication::CommunicationOperation::SendReceive,
        )
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Error> {
        let output =
            <Self as eredu_core::consensus::BoundedConsensusTransport>::submit_all_gather_words(
                self, local,
            )?
            .wait()?;
        <Self as eredu_core::consensus::BoundedConsensusTransport>::resolve_all_gather_words(
            self, output,
        )
    }
}

impl eredu_core::consensus::ConsensusTransport for MlxDistributedSession {
    type Error = Error;

    fn participant_count(&self) -> usize {
        self.communicators.world_size()
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        DistributedSession::all_gather_words(self, local)
    }
}

impl eredu_core::consensus::BoundedConsensusTransport for MlxDistributedSession {
    type Completion = MlxCommunicationCompletion;
    type GatherOutput = MlxTensor;

    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Self::Error> {
        self.ensure_active()?;
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("distributed metadata exceeds i32".into()))?;
        let local = Array::from_slice(local, &[length]);
        let group = self.communicators.control_world().clone();
        let operation = eredu_core::communication::CommunicationOperation::AllGatherEven;
        let gathered = distributed::all_gather(&local, &group, &self.stream)
            .map_err(|error| self.submission_error(error, operation))?;
        let completion = MlxCommunicationCompletion::submit(
            [&gathered],
            vec![local, gathered.clone()],
            Vec::new(),
            vec![group],
            Vec::new(),
            vec![self.stream.clone()],
        )
        .map_err(|error| self.submission_error(error, operation))?
        .with_authority(self.authority.clone(), operation);
        Ok(Submission {
            output: MlxTensor::from_array(gathered),
            completion,
        })
    }

    fn resolve_all_gather_words(
        &self,
        output: Self::GatherOutput,
    ) -> Result<Vec<u32>, Self::Error> {
        Ok(output.as_array().evaluated()?.as_slice::<u32>().to_vec())
    }
}

impl eredu_core::capture::PartitionCaptureTransport for MlxDistributedSession {
    fn capture_rank(&self) -> usize {
        self.manifest.rank()
    }

    fn capture_wait(&self) -> Option<eredu_core::BoundedCompletionWait> {
        self.authority
            .completion_policy()
            .map(|policy| policy.bounded_wait())
    }

    fn ensure_capture_active(&self) -> Result<(), eredu_core::BackendFailure> {
        self.ensure_active()
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn estimate_capture_gather(
        &self,
        local_words: usize,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        use eredu_core::capture::{CaptureError, CaptureUsage};
        let (retained_bytes, host_bytes) =
            control_gather_storage(self.manifest.world_size(), local_words)
                .ok_or(CaptureError::Overflow)?;
        Ok(CaptureUsage {
            retained_bytes,
            host_bytes,
            ..Default::default()
        })
    }

    fn fence_capture_exchange(&self) {
        self.authority.fence(
            eredu_core::communication::CommunicationOperation::AllGatherEven,
            None,
        );
    }
}

// Native input/output, conservative staging, and exact completion metadata.
impl eredu_core::parameters::ParameterOperationTransport for MlxDistributedSession {
    fn parameter_rank(&self) -> usize {
        self.manifest.rank()
    }
    fn parameter_setup(&self) -> eredu_core::communication::CommunicationSessionIdentity {
        self.session_identity()
    }
    fn parameter_wait(
        &self,
    ) -> Result<eredu_core::BoundedCompletionWait, eredu_core::parameters::ParameterError> {
        self.authority
            .completion_policy()
            .map(|policy| policy.bounded_wait())
            .ok_or_else(|| {
                eredu_core::parameters::ParameterError::Unsupported(
                    "parameter control needs bounded completion".into(),
                )
            })
    }
    fn estimate_parameter_gather(
        &self,
        words: usize,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::parameters::ParameterError> {
        let (retained_bytes, host_bytes) =
            control_gather_storage(self.manifest.world_size(), words)
                .ok_or(eredu_core::parameters::ParameterError::Overflow)?;
        Ok(eredu_core::capture::CaptureUsage {
            retained_bytes,
            host_bytes,
            ..Default::default()
        })
    }
    fn ensure_parameter_active(&self) -> Result<(), eredu_core::BackendFailure> {
        self.ensure_active()
            .map_err(eredu_core::BackendFailure::from_error)
    }
    fn fail_parameter_operation(
        &self,
        _error: &eredu_core::parameters::ParameterCoordinationError,
    ) {
        self.authority.fence(
            eredu_core::communication::CommunicationOperation::AllGatherEven,
            None,
        );
    }
}

// Native input/output, conservative staging, and exact completion metadata.
// This prices logical storage and does not promise a physical allocator ceiling.
fn control_gather_storage(participants: usize, local_words: usize) -> Option<(u64, u64)> {
    let gathered = local_words.checked_mul(participants)?;
    i32::try_from(local_words).ok()?;
    i32::try_from(gathered).ok()?;
    Some((
        4096_u64.checked_add(
            (local_words as u64)
                .checked_add(gathered as u64)?
                .checked_mul(8)?,
        )?,
        4096_u64.checked_add((gathered as u64).checked_mul(4)?)?,
    ))
}

impl eredu_core::run_preparation::TextPreparationTransport for MlxDistributedSession {
    fn preparation_rank(&self) -> usize {
        self.manifest.rank()
    }
    fn ensure_preparation_active(&self) -> Result<(), eredu_core::BackendFailure> {
        self.ensure_active()
            .map_err(eredu_core::BackendFailure::from_error)
    }
    fn fence_preparation(&self) {
        self.authority.fence(
            eredu_core::communication::CommunicationOperation::AllGatherEven,
            None,
        );
    }
}

impl MlxDistributedSession {
    fn capture_hook_descriptor(
        &self,
        members: &[usize],
    ) -> Result<
        &eredu_core::communication::CommunicationGroupDescriptor,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;

        let descriptor = self
            .communicators
            .global_group_descriptors()
            .iter()
            .find(|group| {
                group.members() == members
                    && group.requirements().operations().iter().any(|operation| {
                        operation.operation()
                            == eredu_core::communication::CommunicationOperation::FailureAgreement
                            && operation.exact_completion()
                    })
            })
            .ok_or_else(|| {
                eredu_core::capture::CaptureError::Unsupported(
                    "capture hook group has no selected exact failure agreement".into(),
                )
            })?;
        let independent = crate::backend::runtime::distributed::independent_status_members(
            self.manifest.world_size(),
            members,
        );
        if !independent {
            return Err(CaptureError::Unsupported(
                "selected native subgroup requires a world participation wave for status agreement"
                    .into(),
            ));
        }
        i32::try_from(members.len()).map_err(|_| CaptureError::Overflow)?;
        if descriptor.local_index().is_some()
            && self
                .communicators
                .communication_group(descriptor.id())
                .is_none()
        {
            return Err(CaptureError::Invalid(
                "selected capture hook group was not realized".into(),
            ));
        }
        Ok(descriptor)
    }
}

impl eredu_core::capture::PartitionCaptureHookTransport for MlxDistributedSession {
    type HookOutput = crate::backend::runtime::distributed::completion::MlxFailureAgreement;

    fn estimate_capture_hook(
        &self,
        members: &[usize],
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        use eredu_core::capture::{add, mul, CaptureUsage};
        self.capture_hook_descriptor(members)?;
        // Scalar input/output, conservative logical reduction temporaries and
        // exact event/host-status storage. Private allocator workspace is excluded.
        Ok(CaptureUsage {
            retained_bytes: add(8192, mul(members.len() as u64, 256)?)?,
            host_bytes: add(4096, mul(members.len() as u64, 256)?)?,
            ..Default::default()
        })
    }

    fn submit_capture_hook(
        &self,
        members: &[usize],
        success: bool,
    ) -> Result<Submission<Self::HookOutput, MlxCommunicationCompletion>, Error> {
        use eredu_core::communication::FailureAgreementBackend;
        self.ensure_active()?;
        let descriptor = self
            .capture_hook_descriptor(members)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let group = self
            .communicators
            .communication_group(descriptor.id())
            .ok_or_else(|| {
                Error::Parallel("capture hook invoked outside its selected group".into())
            })?;
        let operation = eredu_core::communication::CommunicationOperation::FailureAgreement;
        let submission = crate::backend::nn::shared::MlxNeuralBackend::agree_success(
            success,
            None,
            group,
            &self.stream,
        )
        .map_err(|error| self.submission_error(error, operation))?;
        Ok(Submission {
            output: submission.output,
            completion: submission
                .completion
                .with_authority(self.authority.clone(), operation),
        })
    }

    fn resolve_capture_hook(&self, output: Self::HookOutput) -> Result<bool, Error> {
        use eredu_core::communication::FailureAgreementBackend;
        crate::backend::nn::shared::MlxNeuralBackend::resolve_failure_agreement(output)
            .map_err(Into::into)
    }
}

impl eredu_core::communication::RankCommunication for MlxDistributedSession {
    type Backend = crate::backend::nn::shared::MlxNeuralBackend;
    type Group = Group;
    type Parallel = Group;
    type Route = crate::backend::runtime::distributed::topology::CommunicationRouteRealization;
    type Metadata = crate::backend::nn::shared::MlxCommunicationTensorMetadata;
    type Error = Error;

    fn bound_group(&self, id: CollectiveGroupId) -> Option<&Group> {
        self.communicators.communication_group(id)
    }

    fn tensor_context(&self, id: CollectiveGroupId) -> Option<Group> {
        self.communicators.communication_group(id).cloned()
    }

    fn coordinated_tensor_context(&self, id: CollectiveGroupId) -> Option<Group> {
        let selected = self
            .manifest
            .groups()
            .iter()
            .find(|group| group.id() == id)?;
        let independent = crate::backend::runtime::distributed::independent_status_members(
            self.manifest.world_size(),
            selected.members(),
        );
        if !independent
            || !selected
                .requirements()
                .operations()
                .iter()
                .any(|requirement| {
                    requirement.operation()
                        == eredu_core::communication::CommunicationOperation::FailureAgreement
                        && requirement.exact_completion()
                })
        {
            return None;
        }
        self.tensor_context(id)
            .map(|group| group.with_readiness_authority(self.authority.clone()))
    }

    fn into_rank_resources(
        self,
        manifest: &eredu_core::communication::CommunicationManifest,
    ) -> Result<
        eredu_core::communication::RankCommunicationResources<
            crate::backend::nn::shared::MlxNeuralBackend,
            Group,
            Self::Route,
            Self::Metadata,
        >,
        Error,
    > {
        self.validate_selected_manifest(manifest)?;
        let (groups, routes) = self.communicators.into_partition_resources(manifest)?;
        Ok(eredu_core::communication::RankCommunicationResources {
            groups,
            routes,
            metadata: crate::backend::nn::shared::MlxCommunicationTensorMetadata,
            authority: self.authority,
            executor: self.stream,
        })
    }
}
