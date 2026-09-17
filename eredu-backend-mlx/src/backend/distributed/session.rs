use super::*;
use std::{rc::Rc, sync::Arc};

/// MLX communication capability attached to one complete model/session.
///
/// This is the only public owner of topology-derived MLX communicators. Model
/// implementations may borrow its axis groups internally, but callers cannot
/// construct or route around those groups independently of the selected
/// backend session.
#[derive(Debug, Clone)]
pub struct MlxDistributedSession {
    manifest: Arc<eredu_runtime::CommunicationManifest>,
    communicators: Rc<ParallelCommunicators>,
    stream: Stream,
    world: NativeGroup,
    pub(super) authority: eredu_runtime::PartitionCommunicationAuthority,
    preparation: Option<Arc<eredu_runtime::run_preparation::TextPreparationCoordinator>>,
    parameter_operations:
        Option<Arc<eredu_runtime::parameter_operations::ParameterOperationCoordinator>>,
}

impl MlxDistributedSession {
    /// Logical quota equation shared with the source-retained capture adapter.
    /// This prices no native scope and grants no submission or storage authority.
    pub(crate) fn capture_gather_usage(participants:usize,local_words:usize)
        ->Result<eredu_core::capture::CaptureUsage,eredu_core::capture::CaptureError> {
        let (retained_bytes,host_bytes)=control_gather_storage(participants,local_words)
            .ok_or(eredu_core::capture::CaptureError::Overflow)?;
        Ok(eredu_core::capture::CaptureUsage{retained_bytes,host_bytes,..Default::default()})
    }
    pub(crate) fn matches_capture_source(&self, source: &eredu_runtime::RetainedCommunicationSource,
        authority: &eredu_runtime::PartitionCommunicationAuthority) -> bool {
        self.authority.same_authority(authority)
            && self.communicators.control_world().retained_source()
                .is_some_and(|actual| actual.same_source(source))
    }
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
        let operation = eredu_runtime::CommunicationOperation::FailureAgreement;
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
            .with_authority(
                self.authority.clone(),
                operation,
                eredu_runtime::DistributedExecutionPhase::Execution,
            )
            .wait_bounded(wait)
            .map_err(Into::into)
    }

    /// Creates a session from one already selected architecture-owned manifest.
    pub(crate) fn from_manifest(
        manifest: &eredu_runtime::CommunicationManifest,
        world: &NativeGroup,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let communicators = ParallelCommunicators::from_manifest(manifest, world, stream)?;
        let authority = eredu_runtime::PartitionCommunicationAuthority::from_manifest(manifest)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let preparation = manifest
            .completion_policy()
            .map(|policy| {
                let (retained_bytes, host_bytes) = control_gather_storage(
                    manifest.world_size(),
                    eredu_runtime::run_preparation::TEXT_PREPARATION_WORDS,
                )
                .ok_or_else(|| Error::Parallel("text preparation native bound overflow".into()))?;
                eredu_runtime::run_preparation::TextPreparationCoordinator::new(
                    communicators.session_identity(),
                    eredu_core::run_preparation::TextPreparationUsage {
                        attempts: 0,
                        retained_bytes,
                        host_bytes,
                    },
                    policy.bounded_wait(),
                )
                .map(Arc::new)
                .map_err(Error::from)
            })
            .transpose()?;
        let mut session = Self {
            manifest: Arc::new(manifest.clone()),
            communicators: Rc::new(communicators),
            stream: stream.clone(),
            world: world.clone(),
            authority,
            preparation,
            parameter_operations: None,
        };
        if manifest.completion_policy().is_some() {
            session.parameter_operations = Some(Arc::new(
                eredu_runtime::parameter_operations::ParameterOperationCoordinator::new(&session)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            ));
        }
        Ok(session)
    }

    /// Fund a borrow of the actual checked native communication source. This
    /// creates no communicator, transport, completion, or original work grant.
    pub(crate) fn original_communication_source<'a>(
        &'a self, selected: &eredu_runtime::CommunicationManifest,
        world: &NativeGroup, funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
    ) -> Result<crate::backend::runtime::distributed::topology::OriginalCommunicationSource<'a>, Error> {
        self.communicators.bind_original_source(selected, world, &self.authority, funding)
    }

    /// Retain the actual immutable communicator table after the same checked
    /// source binding. Later phases borrow this owner; no native group or
    /// descriptor is rebuilt from a manifest.
    pub(crate) fn original_communication_owner(
        &self, selected: &eredu_runtime::CommunicationManifest,
        world: &NativeGroup, funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
    ) -> Result<crate::backend::runtime::distributed::topology::OriginalCommunicationOwner, Error> {
        crate::backend::runtime::distributed::topology::OriginalCommunicationOwner::bind(
            &self.communicators, selected, world, &self.authority, funding)
    }

    /// Borrows this session's actual manifest/world and retains the initialized
    /// tensor source for one complete frame. It creates no communicator.
    pub(crate) fn original_initialized_tensor_source(&self,id:eredu_core::CollectiveGroupId,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
        funding:&eredu_nn::workspace::WorkspaceMetadataFunding)
        ->Result<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource,Error> {
        self.original_communication_owner(&self.manifest,&self.world,funding)?
            .prepare_initialized_parallel_source(id,pool)
    }

    pub(crate) fn retained_buffer(&self)->Option<&safemlx::distributed::RetainedGroupBuffer>{
        self.communicators.control_world().retained_buffer()
    }
    pub(crate) fn collect_retained_buffers(&self,storage:&mut crate::backend::runtime::residency::storage::RetainedStorage)->Result<(),Error>{
        self.communicators.collect_retained_buffers(storage)
    }
    /// Common setup identity retained alongside the actual communication owner.
    pub fn session_identity(&self) -> eredu_runtime::CommunicationSessionIdentity {
        self.communicators.session_identity()
    }

    /// Retained live parameter control owner; model resets do not duplicate it.
    pub(crate) fn parameter_operations(
        &self,
    ) -> Result<
        Arc<eredu_runtime::parameter_operations::ParameterOperationCoordinator>,
        eredu_core::parameters::ParameterError,
    > {
        self.parameter_operations.clone().ok_or_else(|| {
            eredu_core::parameters::ParameterError::Unsupported(
                "parameter operations require selected bounded completion".into(),
            )
        })
    }

    /// One identity for this loaded model, shared by corresponding rank sessions.
    /// Models without selected bounded parameter control retain no identity.
    pub(crate) fn register_parameter_model(
        &self,
    ) -> Result<Option<eredu_runtime::parameter_operations::ParameterModelIdentity>, Error> {
        if self.parameter_operations.is_none() {
            return Ok(None);
        }
        self.parameter_operations()
            .and_then(|owner| owner.register_model())
            .map(Some)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Exact native world from which the retained communication was realized.
    pub(crate) const fn native_world(&self) -> &NativeGroup {
        &self.world
    }

    /// Validates the retained native realization without creating communication resources.
    pub(crate) fn validate_selected_manifest(
        &self,
        manifest: &eredu_runtime::CommunicationManifest,
    ) -> Result<(), Error> {
        if self.manifest.as_ref() != manifest {
            return Err(Error::Parallel(
                "native communication differs from the selected manifest".into(),
            ));
        }
        Ok(())
    }

    /// Returns the communicator for an active opaque group identity.
    #[cfg(test)]
    pub(crate) fn selected_group(&self, id: CollectiveGroupId) -> Option<&Group> {
        self.communicators.group(id)
    }

    /// Consumes architecture-selected communication into the neutral partition runtime.
    /// Initial construction that retains the actual setup inventory for later
    /// original execution. Existing consuming construction remains unchanged.
    pub(crate) fn retained_partition_communication(&self,manifest:eredu_runtime::CommunicationManifest,
        tensor_group:eredu_core::CollectiveGroupId)
        ->Result<(eredu_runtime::PartitionCommunication<crate::backend::nn::shared::MlxNeuralBackend,
            Group,crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
            crate::backend::nn::shared::MlxCommunicationTensorMetadata>,Group,Stream),Error> {
        if self.manifest.as_ref()!=&manifest {return Err(Error::Parallel("retained partition manifest differs".into()));}
        let parallel=self.communicators.communication_group(tensor_group)
            .ok_or_else(||Error::Parallel("retained partition tensor group is absent".into()))?.clone();
        let (groups,routes)=self.communicators.partition_resource_loans(&manifest)?;
        let communication=eredu_runtime::PartitionCommunication::new_with_authority(manifest,groups,routes,
            crate::backend::nn::shared::MlxCommunicationTensorMetadata,self.authority.clone())
            .map_err(|cause|Error::Parallel(cause.to_string()))?;
        Ok((communication,parallel,self.stream.clone()))
    }

    pub(crate) fn into_partition_communication(
        self,
        manifest: eredu_runtime::CommunicationManifest,
        tensor_group: Option<CollectiveGroupId>,
        sampling_group: CollectiveGroupId,
    ) -> Result<
        (
            eredu_runtime::PartitionCommunication<
                crate::backend::nn::shared::MlxNeuralBackend,
                Group,
                crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
                crate::backend::nn::shared::MlxCommunicationTensorMetadata,
            >,
            Option<Group>,
            Group,
            Stream,
        ),
        Error,
    > {
        let parallel = tensor_group
            .map(|tensor_group| {
                self.communicators
                    .communication_group(tensor_group)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "selected tensor group {} was not realized",
                            tensor_group.value()
                        ))
                    })
            })
            .transpose()?;
        let sampling = self
            .communicators
            .communication_group(sampling_group)
            .cloned()
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "selected sampling group {} was not realized",
                    sampling_group.value()
                ))
            })?;
        let Self {
            manifest: _,
            communicators,
            stream,
            world: _,
            authority,
            preparation: _,
            parameter_operations: _,
        } = self;
        // Initial runtime construction keeps the existing consuming worker.
        // A unique table moves directly; a previously shared ordinary setup
        // takes the same container copy that session cloning used to take.
        let (groups, routes) = Rc::unwrap_or_clone(communicators).into_partition_resources(&manifest)?;
        let communication = eredu_runtime::PartitionCommunication::new_with_authority(
            manifest,
            groups,
            routes,
            crate::backend::nn::shared::MlxCommunicationTensorMetadata,
            authority,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok((communication, parallel, sampling, stream))
    }

    #[cfg(test)]
    pub(crate) fn send_selected(
        &self,
        group: CollectiveGroupId,
        peer: usize,
        value: &MlxTensor,
    ) -> Result<DistributedCompletion<MlxTensor>, Error> {
        Ok(DistributedSession::send(self, CollectiveScope::Group(group), peer, value)?.completion)
    }

    #[cfg(test)]
    pub(crate) fn receive_selected(
        &self,
        group: CollectiveGroupId,
        peer: usize,
        shape: &[usize],
        dtype: TensorDtype,
    ) -> Result<DistributedCompletion<MlxTensor>, Error> {
        Ok(DistributedSession::receive(
            self,
            CollectiveScope::Group(group),
            peer,
            &ValueDescriptor::new(shape.to_vec(), dtype).map_err(Error::Backend)?,
        )?
        .completion)
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

    pub(crate) fn coordinate_speculative_step<B: AsRef<[eredu_core::SpeculativeScheduleState]> + AsMut<[eredu_core::SpeculativeScheduleState]>>(
        &self,
        local: B,
    ) -> Result<B, eredu_core::BackendFailure> {
        let coordinator = self.preparation.as_ref().ok_or_else(|| {
            eredu_core::BackendFailure::new(
                eredu_core::BackendFailureKind::Unsupported,
                eredu_runtime::run_preparation::TextPreparationAgreementError::Admission(
                    "selected session has no bounded scheduler transport",
                ),
            )
        })?;
        coordinator
            .coordinate_speculative_step(self, local)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    pub(crate) fn agree_text_preparation(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        let coordinator = self.preparation.as_ref().ok_or_else(|| {
            eredu_core::BackendFailure::new(
                eredu_core::BackendFailureKind::Unsupported,
                eredu_runtime::run_preparation::TextPreparationAgreementError::Admission(
                    "selected session has no bounded preparation transport",
                ),
            )
        })?;
        coordinator
            .agree(self, stage, status)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    pub(crate) fn text_preparation_usage(
        &self,
    ) -> Result<eredu_core::run_preparation::TextPreparationUsage, eredu_core::BackendFailure> {
        self.preparation
            .as_ref()
            .ok_or_else(|| {
                eredu_core::BackendFailure::new(
                    eredu_core::BackendFailureKind::Unsupported,
                    eredu_runtime::run_preparation::TextPreparationAgreementError::Admission(
                        "selected session has no bounded preparation transport",
                    ),
                )
            })?
            .usage()
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn submission_error(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: eredu_runtime::CommunicationOperation,
    ) -> Error {
        Error::Other(Box::new(
            self.authority
                .submission_failure(
                    error,
                    operation,
                    eredu_runtime::DistributedExecutionPhase::Execution,
                    None,
                ),
        ))
    }

    fn selected_submission(
        &self,
        output: Array,
        retained: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        group: Group,
        operation: eredu_runtime::CommunicationOperation,
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

    /// Returns whether `scope` is implemented by an MLX logical subgroup.
    #[cfg(test)]
    pub(crate) fn scope_is_logical(&self, scope: CollectiveScope) -> Result<bool, Error> {
        Ok(self.group(scope)?.is_logical())
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

impl<'a> DistributedBackend for MlxBackend<'a> {
    type DistributedSession = MlxDistributedSession;

    fn distributed_session(
        session: &crate::composition::mlx::MlxModelSession,
    ) -> Option<&Self::DistributedSession> {
        session.distributed()
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
                self.submission_error(error, eredu_runtime::CommunicationOperation::AllReduceSum)
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_runtime::CommunicationOperation::AllReduceSum,
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
                self.submission_error(error, eredu_runtime::CommunicationOperation::AllGatherEven)
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_runtime::CommunicationOperation::AllGatherEven,
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
                eredu_runtime::CommunicationOperation::VariableAllToAll,
            )
        })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            vec![send_counts.to_vec(), receive_counts.to_vec()],
            group,
            eredu_runtime::CommunicationOperation::VariableAllToAll,
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
                self.submission_error(error, eredu_runtime::CommunicationOperation::SendReceive)
            })?;
        self.selected_submission(
            output.clone(),
            vec![input.as_array().clone(), output],
            Vec::new(),
            group,
            eredu_runtime::CommunicationOperation::SendReceive,
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
            self.submission_error(error, eredu_runtime::CommunicationOperation::SendReceive)
        })?;
        self.selected_submission(
            output.clone(),
            vec![output],
            Vec::new(),
            group,
            eredu_runtime::CommunicationOperation::SendReceive,
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
        let operation = eredu_runtime::CommunicationOperation::AllGatherEven;
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
        .with_authority(
            self.authority.clone(),
            operation,
            eredu_runtime::DistributedExecutionPhase::Execution,
        );
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

impl eredu_runtime::capture::partition::PartitionCaptureTransport for MlxDistributedSession {
    fn capture_rank(&self) -> usize {
        self.manifest.rank()
    }

    fn capture_wait(
        &self,
    ) -> Result<eredu_core::BoundedCompletionWait, eredu_core::capture::CaptureError> {
        self.authority
            .completion_policy()
            .map(|policy| policy.bounded_wait())
            .ok_or_else(|| {
                eredu_core::capture::CaptureError::Unsupported(
                    "capture transport needs selected bounded completion".into(),
                )
            })
    }

    fn ensure_capture_active(&self) -> Result<(), eredu_core::BackendFailure> {
        self.ensure_active()
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn estimate_capture_gather(
        &self,
        local_words: usize,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        Self::capture_gather_usage(self.manifest.world_size(),local_words)
    }

    fn fail_capture_exchange(
        &self,
        error: &eredu_runtime::capture::partition::PartitionCaptureExchangeError,
    ) {
        let _ = self.authority.completion_error(
            error,
            eredu_runtime::CommunicationOperation::AllGatherEven,
            eredu_runtime::DistributedExecutionPhase::Execution,
            None,
        );
    }
}

// Native input/output, conservative staging, and exact completion metadata.
impl eredu_runtime::parameter_operations::ParameterOperationTransport for MlxDistributedSession {
    fn parameter_rank(&self) -> usize {
        self.manifest.rank()
    }
    fn parameter_setup(&self) -> eredu_runtime::CommunicationSessionIdentity {
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
    fn fail_parameter_operation(&self, error: &eredu_core::parameters::ParameterCoordinationError) {
        let _ = self.authority.completion_error(
            error,
            eredu_runtime::CommunicationOperation::AllGatherEven,
            eredu_runtime::DistributedExecutionPhase::Execution,
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

impl eredu_runtime::run_preparation::TextPreparationTransport for MlxDistributedSession {
    fn preparation_rank(&self) -> usize {
        self.manifest.rank()
    }
    fn ensure_preparation_active(&self) -> Result<(), eredu_core::BackendFailure> {
        self.ensure_active()
            .map_err(eredu_core::BackendFailure::from_error)
    }
    fn fail_preparation(
        &self,
        error: &eredu_runtime::run_preparation::TextPreparationAgreementError,
    ) {
        let _ = self.authority.completion_error(
            error,
            eredu_runtime::CommunicationOperation::AllGatherEven,
            eredu_runtime::DistributedExecutionPhase::InputPreparation,
            None,
        );
    }
}

impl MlxDistributedSession {
    /// Borrow the exact original setup descriptor, including remote stage
    /// groups needed by all-rank preparation. This grants no native group loan.
    pub(crate) fn capture_hook_source<'a>(groups:&'a [eredu_runtime::CommunicationGroupDescriptor],
        members:&[usize])->Option<&'a eredu_runtime::CommunicationGroupDescriptor> {
        groups.iter().find(|group|group.members()==members
            && group.requirements().operations().iter().any(|operation|
                operation.operation()==eredu_runtime::CommunicationOperation::FailureAgreement
                    && operation.exact_completion()))
    }
    fn capture_hook_descriptor(
        &self,
        members: &[usize],
    ) -> Result<&eredu_runtime::CommunicationGroupDescriptor, eredu_core::capture::CaptureError>
    {
        Self::capture_hook_source(self.communicators.global_group_descriptors(),members)
            .ok_or_else(|| {
                eredu_core::capture::CaptureError::Unsupported(
                    "capture hook group has no selected exact failure agreement".into(),
                )
            })
    }
}

impl eredu_runtime::capture::partition::PartitionCaptureHookTransport for MlxDistributedSession {
    type HookOutput = crate::backend::runtime::distributed::completion::MlxFailureAgreement;

    fn estimate_capture_hook(&self,members:&[usize])
        ->Result<eredu_core::capture::CaptureUsage,eredu_core::capture::CaptureError> {
        use eredu_core::capture::CaptureError;
        let usage=Self::capture_hook_usage(self.manifest.world_size(),members)?;
        if members.len()==1 {return Ok(usage);}
        let descriptor=self.capture_hook_descriptor(members)?;
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
        Ok(usage)
    }

    fn submit_capture_hook(
        &self,
        members: &[usize],
        success: bool,
    ) -> Result<Submission<Self::HookOutput, MlxCommunicationCompletion>, Error> {
        use eredu_runtime::FailureAgreementBackend;
        self.ensure_active()?;
        self.estimate_capture_hook(members)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let descriptor = self
            .capture_hook_descriptor(members)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let group = self
            .communicators
            .communication_group(descriptor.id())
            .ok_or_else(|| {
                Error::Parallel("capture hook invoked outside its selected group".into())
            })?;
        let operation = eredu_runtime::CommunicationOperation::FailureAgreement;
        let submission = crate::backend::nn::shared::MlxNeuralBackend::agree_success(
            success,
            group,
            &self.stream,
        )
        .map_err(|error| self.submission_error(error, operation))?;
        Ok(Submission {
            output: submission.output,
            completion: submission.completion.into_native().with_authority(
                self.authority.clone(),
                operation,
                eredu_runtime::DistributedExecutionPhase::Execution,
            ),
        })
    }

    fn resolve_capture_hook(&self, output: Self::HookOutput) -> Result<bool, Error> {
        use eredu_runtime::FailureAgreementBackend;
        crate::backend::nn::shared::MlxNeuralBackend::resolve_failure_agreement(output)
            .map_err(Into::into)
    }
}

mod readiness;
pub use readiness::MlxTextPreparationControl;

impl MlxDistributedSession {
    /// Existing member-only hook logical equation, shared by ordinary and
    /// original transports. This is a source fact and grants no native work.
    pub(crate) fn capture_hook_usage(world:usize,members:&[usize])
        ->Result<eredu_core::capture::CaptureUsage,eredu_core::capture::CaptureError> {
        use eredu_core::capture::{add, mul, CaptureError, CaptureUsage};
        if members.is_empty()
            || members.windows(2).any(|pair| pair[0] >= pair[1])
            || members
                .iter()
                .any(|rank| *rank >= world)
        {
            return Err(CaptureError::Invalid(
                "capture hook membership exceeds selected world".into(),
            ));
        }
        // A singleton requires no collective or native status tensor.
        if members.len() == 1 {
            return Ok(CaptureUsage::default());
        }
        let independent = crate::backend::runtime::distributed::independent_status_members(
            world,
            members,
        );
        if !independent {
            return Err(CaptureError::Unsupported(
                "selected native subgroup requires a world participation wave for status agreement"
                    .into(),
            ));
        }
        i32::try_from(members.len()).map_err(|_| CaptureError::Overflow)?;
        // Scalar input/output, conservative logical reduction temporaries and
        // exact event/host-status storage. Private allocator workspace is excluded.
        Ok(CaptureUsage {
            retained_bytes: add(8192, mul(members.len() as u64, 256)?)?,
            host_bytes: add(4096, mul(members.len() as u64, 256)?)?,
            ..Default::default()
        })
    }

}

mod realtime_consensus;
pub(crate) use realtime_consensus::PreparedRealtimeConsensusTransport;
