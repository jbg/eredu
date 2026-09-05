//! MLX implementation of the optional distributed-session contract.

use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    BackendError, CollectiveGroupId, CollectiveScope, DistributedBackend, DistributedCapabilities,
    DistributedSession, DistributedSessionDescriptor, Submission, ValueDescriptor,
};
use safemlx::{
    distributed::Group as NativeGroup,
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, zeros_dtype},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::runtime::distributed::{
        self,
        completion::{DistributedCompletion, MlxCommunicationCompletion},
        topology::ParallelCommunicators,
        Group,
    },
};

use super::MlxBackend;
use crate::MlxTensor;

/// Mechanism-only topology-wide consensus transport for realtime scheduling.
///
/// This transport owns only the selected native group and stream. Schedule,
/// completion, and publication policy remain in the neutral runtime scheduler.
#[derive(Clone)]
pub struct MlxRealtimeConsensusTransport {
    group: Group,
    stream: Stream,
}

impl MlxRealtimeConsensusTransport {
    /// Binds scheduler consensus to one already selected world group.
    pub fn new(group: &NativeGroup, stream: &Stream) -> Self {
        Self {
            group: Group::uncontracted(group),
            stream: stream.clone(),
        }
    }
}

impl eredu_core::consensus::ConsensusTransport for MlxRealtimeConsensusTransport {
    type Error = Error;

    fn participant_count(&self) -> usize {
        self.group.size()
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let submission =
            <Self as eredu_core::consensus::BoundedConsensusTransport>::submit_all_gather_words(
                self, local,
            )?;
        let output = submission.wait()?;
        <Self as eredu_core::consensus::BoundedConsensusTransport>::resolve_all_gather_words(
            self, output,
        )
    }
}

impl eredu_core::consensus::BoundedConsensusTransport for MlxRealtimeConsensusTransport {
    type Completion = MlxCommunicationCompletion;
    type GatherOutput = MlxTensor;

    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Self::Error> {
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("realtime consensus metadata exceeds i32".into()))?;
        let local = Array::from_slice(local, &[length]);
        let gathered = distributed::all_gather(&local, &self.group, &self.stream)?;
        let completion = MlxCommunicationCompletion::submit(
            [&gathered],
            vec![local, gathered.clone()],
            Vec::new(),
            vec![self.group.clone()],
            Vec::new(),
            vec![self.stream.clone()],
        )?;
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

/// Gathers equal-shaped shards along an arbitrary existing tensor axis.
pub fn all_gather_axis(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    all_gather_axis_for(
        eredu_runtime::CommunicationOperation::AllGatherEven,
        input,
        axis,
        group,
        stream,
    )
}

fn all_gather_axis_for(
    operation: eredu_runtime::CommunicationOperation,
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    if axis == 0 {
        return distributed::all_gather_for(operation, input, group, stream);
    }
    let gathered = distributed::all_gather_for(operation, input, group, stream)?;
    let rank_height = input.shape()[0];
    let mut shards = Vec::with_capacity(group.size());
    for rank in 0..group.size() {
        let start = i32::try_from(rank)
            .ok()
            .and_then(|rank| rank.checked_mul(rank_height))
            .ok_or_else(|| Exception::custom("gathered rank offset exceeds i32"))?;
        let end = start
            .checked_add(rank_height)
            .ok_or_else(|| Exception::custom("gathered rank end exceeds i32"))?;
        shards.push(gathered.try_index_device(start..end, stream)?);
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    let output = concatenate_axis(&shard_refs, axis, stream)?;
    let mut expected = input.shape().to_vec();
    expected[axis as usize] = expected[axis as usize]
        .checked_mul(
            i32::try_from(group.size())
                .map_err(|_| Exception::custom("group size does not fit in i32"))?,
        )
        .ok_or_else(|| Exception::custom("axis all-gather output shape exceeds i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "axis all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

fn all_gather_axis_unchecked(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Exception> {
    if axis == 0 {
        return distributed::all_gather_unchecked(input, group, stream);
    }
    let gathered = distributed::all_gather_unchecked(input, group, stream)?;
    let rank_height = input.shape()[0];
    let mut shards = Vec::with_capacity(group.size());
    for rank in 0..group.size() {
        let start = i32::try_from(rank)
            .ok()
            .and_then(|rank| rank.checked_mul(rank_height))
            .ok_or_else(|| Exception::custom("gathered rank offset exceeds i32"))?;
        let end = start
            .checked_add(rank_height)
            .ok_or_else(|| Exception::custom("gathered rank end exceeds i32"))?;
        shards.push(gathered.try_index_device(start..end, stream)?);
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    concatenate_axis(&shard_refs, axis, stream)
}

/// Gathers unequal contiguous shards along an arbitrary tensor axis.
pub fn all_gather_uneven_axis(
    input: &Array,
    axis: i32,
    widths: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    if widths.len() != group.size() {
        return Err(Exception::custom(format!(
            "uneven all-gather received {} widths for group size {}",
            widths.len(),
            group.size()
        )));
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "uneven axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "uneven all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    let rank = group.rank();
    let local_width = usize::try_from(input.shape()[axis as usize])
        .map_err(|_| Exception::custom("input shape contains a negative dimension"))?;
    if local_width != widths[rank] {
        return Err(Exception::custom(format!(
            "rank {rank} local width {local_width} does not match declared width {}",
            widths[rank]
        )));
    }
    let max_width = widths.iter().copied().max().unwrap_or(0);
    if max_width == 0 {
        return Err(Exception::custom(
            "uneven all-gather requires at least one non-empty shard",
        ));
    }
    let operation = eredu_runtime::CommunicationOperation::AllGatherUneven;
    group.validate_tensor(operation, input, false)?;
    let output_width = widths.iter().try_fold(0usize, |total, width| {
        total
            .checked_add(*width)
            .ok_or_else(|| Exception::custom("uneven all-gather output width overflowed usize"))
    })?;
    let non_axis_elements = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(dimension, _)| *dimension != axis as usize)
        .try_fold(1usize, |total, (_, dimension)| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| Exception::custom("input shape contains a negative dimension"))?;
            total.checked_mul(dimension).ok_or_else(|| {
                Exception::custom("uneven all-gather output elements overflowed usize")
            })
        })?;
    let output_elements = non_axis_elements
        .checked_mul(output_width)
        .ok_or_else(|| Exception::custom("uneven all-gather output elements overflowed usize"))?;
    group.validate_expected_output(operation, input.dtype(), input.ndim(), output_elements)?;
    let padded = if local_width == max_width {
        input.clone()
    } else {
        let mut padding_shape = input.shape().to_vec();
        padding_shape[axis as usize] = i32::try_from(max_width - local_width)
            .map_err(|_| Exception::custom("padding width does not fit in i32"))?;
        let padding = zeros_dtype(&padding_shape, input.dtype(), stream)?;
        concatenate_axis(&[input, &padding], axis, stream)?
    };
    let gathered = all_gather_axis_unchecked(&padded, axis, group, stream)?;
    let group_size = i32::try_from(group.size())
        .map_err(|_| Exception::custom("distributed group size does not fit in i32"))?;
    let padded_shards = gathered.split(group_size, Some(axis), stream)?;
    let mut shards = Vec::with_capacity(widths.len());
    for (padded, &width) in padded_shards.into_iter().zip(widths) {
        if width == max_width {
            shards.push(padded);
        } else {
            let width = i32::try_from(width)
                .map_err(|_| Exception::custom("shard width does not fit in i32"))?;
            shards.push(
                padded
                    .split_axis(&[width], Some(axis), stream)?
                    .into_iter()
                    .next()
                    .expect("one split index produces a leading shard"),
            );
        }
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    let output = concatenate_axis(&shard_refs, axis, stream)?;
    group.validate_tensor(operation, &output, true)?;
    let mut expected = input.shape().to_vec();
    expected[axis as usize] = i32::try_from(output_width)
        .map_err(|_| Exception::custom("uneven all-gather output width exceeds i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "uneven all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

/// Exchanges variable-sized blocks along an arbitrary existing tensor axis.
pub fn all_to_all_v_axis(
    input: &Array,
    axis: usize,
    send_counts: &[usize],
    receive_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    if input.ndim() == 0 {
        return Err(Exception::custom(
            "variable all-to-all requires a non-scalar input",
        ));
    }
    if axis >= input.ndim() {
        return Err(Exception::custom(format!(
            "variable all-to-all axis {axis} is outside input rank {}",
            input.ndim()
        )));
    }
    if axis == 0 {
        return distributed::all_to_all_v(input, send_counts, receive_counts, group, stream);
    }

    let ndim = input.ndim();
    let axis = i32::try_from(axis)
        .map_err(|_| Exception::custom("variable all-to-all axis exceeds i32"))?;
    let mut to_front = Vec::with_capacity(ndim);
    to_front.push(axis);
    for current in 0..ndim {
        let current = i32::try_from(current)
            .map_err(|_| Exception::custom("variable all-to-all rank exceeds i32"))?;
        if current != axis {
            to_front.push(current);
        }
    }
    let transposed = input.transpose_axes(&to_front, stream)?;
    let exchanged =
        distributed::all_to_all_v(&transposed, send_counts, receive_counts, group, stream)?;
    let mut restore = vec![0i32; ndim];
    for (position, original_axis) in to_front.into_iter().enumerate() {
        restore[original_axis as usize] = i32::try_from(position)
            .map_err(|_| Exception::custom("variable all-to-all rank exceeds i32"))?;
    }
    exchanged.transpose_axes(&restore, stream)
}

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
    authority: eredu_runtime::PartitionCommunicationAuthority,
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
    fn group(&self, scope: CollectiveScope) -> Result<&Group, Error> {
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

#[cfg(test)]
mod bounded_consensus_tests {
    use super::*;
    use eredu_core::{
        consensus::BoundedConsensusTransport as _, BoundedCompletionWait, BoundedSubmissionOutcome,
        CompletionCancellationMode,
    };
    use safemlx::{Device, DeviceType};

    fn singleton_data_manifest() -> (
        eredu_runtime::CommunicationManifest,
        eredu_core::CollectiveGroupId,
    ) {
        let id = eredu_core::CollectiveGroupId::new(41);
        let requirement = eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllReduceSum,
            [eredu_core::checkpoint::TensorDtype::F32],
            eredu_runtime::CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
            true,
        )
        .unwrap();
        let group = eredu_runtime::CommunicationGroupDescriptor::new(
            id,
            0,
            vec![0],
            Some(0),
            eredu_runtime::CommunicationGroupRequirements::new([requirement]).unwrap(),
        )
        .unwrap();
        let policy = eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        (
            eredu_runtime::CommunicationManifest::new(1, 0, vec![group], Vec::new())
                .unwrap()
                .with_completion_policy(policy),
            id,
        )
    }

    #[test]
    fn bounded_consensus_submission_retains_exact_native_work_until_resolution() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
            .expect("singleton Ring world should initialize");
        let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        let submission = session
            .submit_all_gather_words(&[0, 1, u32::MAX, 0x8000_0000])
            .unwrap();
        assert_eq!(submission.completion.retained_arrays(), 2);
        assert_eq!(submission.completion.retained_groups(), 1);
        assert_eq!(submission.completion.retained_streams(), 1);
        let wait = BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let output = match submission.wait_bounded(wait).unwrap() {
            BoundedSubmissionOutcome::Completed(output) => output,
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                panic!("singleton consensus timed out with {cancellation:?}")
            }
        };
        assert_eq!(
            session.resolve_all_gather_words(output).unwrap(),
            [0, 1, u32::MAX, 0x8000_0000]
        );

        let completion = eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(completion);
        let manifest_session = MlxDistributedSession::from_manifest(&manifest, &world, &stream)
            .expect("manifest session should retain a control-only world handle");
        assert!(!DistributedSession::capabilities(&manifest_session).world_collectives());
        assert!(manifest_session.group(CollectiveScope::World).is_err());
        let output = manifest_session
            .submit_all_gather_words(&[17, u32::MAX])
            .unwrap()
            .wait_bounded(wait)
            .unwrap();
        let output = match output {
            BoundedSubmissionOutcome::Completed(output) => output,
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                panic!("manifest consensus timed out with {cancellation:?}")
            }
        };
        assert_eq!(
            manifest_session.resolve_all_gather_words(output).unwrap(),
            [17, u32::MAX]
        );
    }

    #[test]
    fn retained_public_view_and_partition_runtime_share_one_poison_authority() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
            .expect("singleton Ring world should initialize");
        let (manifest, group) = singleton_data_manifest();
        let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        let retained = session.clone();
        let (communication, _, _, _) = session
            .into_partition_communication(manifest, None, group)
            .unwrap();

        let _ = retained.authority.completion_error(
            "injected public completion failure",
            eredu_runtime::CommunicationOperation::AllReduceSum,
            eredu_runtime::DistributedExecutionPhase::Execution,
            None,
        );
        assert!(communication.authority().ensure_active().is_err());
    }

    #[test]
    fn public_collective_completion_retains_inputs_group_and_stream_after_session_drop() {
        use eredu_core::Completion as _;

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
            .expect("singleton Ring world should initialize");
        let (manifest, group) = singleton_data_manifest();
        let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        let input = MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1]));
        let submission = session
            .all_reduce_sum(CollectiveScope::Group(group), &input)
            .unwrap();
        assert_eq!(submission.completion.retained_resources(), 2);
        assert_eq!(
            submission.completion.retained_native_resources(),
            (0, 1, 0, 1)
        );
        drop(session);
        drop(input);
        submission.completion.wait().unwrap();
    }

    #[test]
    fn public_collective_wait_uses_manifest_deadline_and_shared_poison() {
        use eredu_core::Completion as _;

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
            .expect("singleton Ring world should initialize");
        let (manifest, group) = singleton_data_manifest();
        let manifest = manifest.with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_millis(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
        let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        crate::backend::runtime::distributed::completion::force_next_communication_pending();
        let input = MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1]));
        let submission = session
            .all_reduce_sum(CollectiveScope::Group(group), &input)
            .unwrap();
        assert!(submission.completion.wait().is_err());
        assert!(submission.completion.wait_on(&stream).is_err());
        assert!(session.authority.ensure_active().is_err());
        assert_eq!(
            crate::backend::runtime::distributed::completion::distributed_completion_orphan_count(),
            1
        );
        crate::backend::runtime::distributed::completion::release_forced_pending_orphans();
        assert_eq!(
            crate::backend::runtime::distributed::completion::distributed_completion_orphan_count(),
            0
        );
    }
}
