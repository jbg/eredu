use super::*;

/// MLX communication capability attached to one complete model/session.
///
/// This is the only public owner of topology-derived MLX communicators. Model
/// implementations may borrow its axis groups internally, but callers cannot
/// construct or route around those groups independently of the selected
/// backend session.
#[derive(Debug, Clone)]
pub struct MlxDistributedSession {
    communicators: ParallelCommunicators,
    stream: Stream,
    pub(super) authority: eredu_runtime::PartitionCommunicationAuthority,
}

impl MlxDistributedSession {
    /// Creates a session from one already selected architecture-owned manifest.
    pub(crate) fn from_manifest(
        manifest: &eredu_runtime::CommunicationManifest,
        world: &NativeGroup,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let communicators = ParallelCommunicators::from_manifest(manifest, world, stream)?;
        let authority = eredu_runtime::PartitionCommunicationAuthority::from_manifest(manifest)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(Self {
            communicators,
            stream: stream.clone(),
            authority,
        })
    }

    /// Returns the communicator for an active opaque group identity.
    #[cfg(test)]
    pub(crate) fn selected_group(&self, id: CollectiveGroupId) -> Option<&Group> {
        self.communicators.group(id)
    }

    /// Consumes architecture-selected communication into the neutral partition runtime.
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
            communicators,
            stream,
            authority,
        } = self;
        let (groups, routes) = communicators.into_partition_resources(&manifest)?;
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

    fn submission_error(
        &self,
        error: impl std::fmt::Display,
        operation: eredu_runtime::CommunicationOperation,
    ) -> Error {
        Error::Parallel(
            self.authority
                .submission_error(
                    error,
                    operation,
                    eredu_runtime::DistributedExecutionPhase::Execution,
                    None,
                )
                .to_string(),
        )
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
