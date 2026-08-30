//! MLX implementation of the optional distributed-session contract.

use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    BackendError, CollectiveScope, DistributedBackend, DistributedCapabilities, DistributedSession,
    DistributedSessionDescriptor, ParallelAxis, ParallelCoordinates, Submission, ValueDescriptor,
};
use eredu_runtime::Sampler;
use safemlx::{
    distributed::Group as NativeGroup,
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, zeros_dtype},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::runtime::{
        distributed::{
            self,
            completion::DistributedCompletion,
            parallel::{sample_and_synchronize, ParallelExecutionContext, SynchronizedToken},
            topology::ParallelCommunicators,
            Group,
        },
        generation::MlxSamplingBackend,
    },
};

use super::{MlxBackend, MlxParallelContext};
use crate::MlxTensor;

/// Gathers equal-shaped shards along an arbitrary existing tensor axis.
pub fn all_gather_axis(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
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
        return distributed::all_gather(input, group, stream);
    }
    let gathered = distributed::all_gather(input, group, stream)?;
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
    let padded = if local_width == max_width {
        input.clone()
    } else {
        let mut padding_shape = input.shape().to_vec();
        padding_shape[axis as usize] = i32::try_from(max_width - local_width)
            .map_err(|_| Exception::custom("padding width does not fit in i32"))?;
        let padding = zeros_dtype(&padding_shape, input.dtype(), stream)?;
        concatenate_axis(&[input, &padding], axis, stream)?
    };
    let gathered = all_gather_axis(&padded, axis, group, stream)?;
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
    concatenate_axis(&shard_refs, axis, stream)
}

/// Inputs needed to attach MLX communication to one selected backend session.
pub struct MlxDistributedConfig<'a> {
    /// Validated rank-local topology.
    pub topology: MlxParallelContext,
    /// MLX world communicator selected for this complete session.
    pub world: &'a NativeGroup,
}

/// MLX communication capability attached to one complete model/session.
///
/// This is the only public owner of topology-derived MLX communicators. Model
/// implementations may borrow its axis groups internally, but callers cannot
/// construct or route around those groups independently of the selected
/// backend session.
#[derive(Debug)]
pub struct MlxDistributedSession<'a> {
    topology: MlxParallelContext,
    communicators: ParallelCommunicators<'a>,
    stream: Stream,
}

impl<'a> MlxDistributedSession<'a> {
    /// Creates a distributed session and validates its execution stream.
    pub fn new(config: MlxDistributedConfig<'a>, stream: &Stream) -> Result<Self, Error> {
        config.topology.validate_execution_stream(stream)?;
        let communicators = ParallelCommunicators::new(config.topology, config.world)?;
        Ok(Self {
            topology: config.topology,
            communicators,
            stream: stream.clone(),
        })
    }

    /// Returns the validated MLX runtime topology.
    pub const fn topology(&self) -> MlxParallelContext {
        self.topology
    }

    /// Returns the execution stream selected for this session.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the world communicator for the session.
    pub const fn world(&self) -> &Group {
        self.communicators.world()
    }

    /// Creates the tensor-parallel execution context, or a replicated context.
    pub fn tensor_context(&self) -> Result<ParallelExecutionContext<'_>, Error> {
        match self.communicators.tensor_group() {
            Some(group) => {
                ParallelExecutionContext::tensor_parallel(self.topology(), group, &self.stream)
            }
            None => Ok(ParallelExecutionContext::replicated(&self.stream)),
        }
    }

    /// Returns the tensor-parallel communicator when that axis is partitioned.
    pub fn tensor_group(&self) -> Option<&Group> {
        self.communicators.tensor_group()
    }

    /// Returns the expert-parallel communicator when that axis is partitioned.
    pub fn expert_group(&self) -> Option<&Group> {
        self.communicators.expert_group()
    }

    /// Returns the pipeline-parallel communicator when that axis is partitioned.
    pub fn pipeline_group(&self) -> Option<&Group> {
        self.communicators.pipeline_group()
    }

    fn world_control_consensus(&self, local: [i32; 2]) -> Result<[i32; 2], Error> {
        let world = self.world();
        if world.size() == 1 {
            return Ok(local);
        }
        let rank = world.rank();
        let last = world.size() - 1;
        let mut result = local;
        if rank < last {
            let received = distributed::recv(&[2], Dtype::Int32, rank + 1, world, &self.stream)?;
            let evaluated = received.evaluated()?;
            let values = evaluated
                .try_as_slice::<i32>()
                .map_err(|error| Error::Parallel(error.to_string()))?;
            result[0] |= i32::from(values[0] != 0);
            result[1] |= i32::from(values[1] != 0);
        }
        if rank > 0 {
            let result_array = Array::from_slice(&result, &[2]);
            distributed::send(&result_array, rank - 1, world, &self.stream)?.evaluated()?;
        }

        // Broadcast around the same descending ring direction. Rank zero and
        // the final rank are direct neighbors, so this never requires the
        // unsupported non-neighbor Ring sends used by a star gather.
        if rank == 0 {
            let result_array = Array::from_slice(&result, &[2]);
            distributed::send(&result_array, last, world, &self.stream)?.evaluated()?;
        } else {
            let successor = if rank == last { 0 } else { rank + 1 };
            let received = distributed::recv(&[2], Dtype::Int32, successor, world, &self.stream)?;
            let evaluated = received.evaluated()?;
            let values = evaluated
                .try_as_slice::<i32>()
                .map_err(|error| Error::Parallel(error.to_string()))?;
            result = [values[0], values[1]];
            if rank > 1 {
                let result_array = Array::from_slice(&result, &[2]);
                distributed::send(&result_array, rank - 1, world, &self.stream)?.evaluated()?;
            }
        }
        Ok(result)
    }

    /// Submits hidden activations to the succeeding pipeline coordinate.
    pub fn send_pipeline(
        &self,
        hidden: &MlxTensor,
    ) -> Result<DistributedCompletion<MlxTensor>, Error> {
        let topology = self.topology();
        if topology.pipeline_parallel_rank + 1 == topology.pipeline_parallel_size {
            return Err(Error::Parallel(
                "the final pipeline stage has no successor".into(),
            ));
        }
        Ok(DistributedSession::send(
            self,
            CollectiveScope::Axis(eredu_core::topology::ParallelAxis::Pipeline),
            topology.pipeline_parallel_rank + 1,
            hidden,
        )?
        .completion)
    }

    /// Submits a receive from the preceding pipeline coordinate.
    pub fn receive_pipeline(
        &self,
        shape: &[usize],
        dtype: TensorDtype,
    ) -> Result<DistributedCompletion<MlxTensor>, Error> {
        let topology = self.topology();
        if topology.pipeline_parallel_rank == 0 {
            return Err(Error::Parallel(
                "the first pipeline stage has no predecessor".into(),
            ));
        }
        Ok(DistributedSession::receive(
            self,
            CollectiveScope::Axis(eredu_core::topology::ParallelAxis::Pipeline),
            topology.pipeline_parallel_rank - 1,
            &ValueDescriptor {
                shape: shape.to_vec(),
                dtype,
            },
        )?
        .completion)
    }

    fn group(&self, scope: CollectiveScope) -> Result<&Group, Error> {
        match scope {
            CollectiveScope::World => Ok(self.world()),
            CollectiveScope::Axis(ParallelAxis::Data) => {
                Err(Error::Backend(BackendError::Unsupported {
                    backend: "mlx".into(),
                    capability: "data-parallel subgroup".into(),
                }))
            }
            CollectiveScope::Axis(axis) => self.communicators.group(axis).ok_or_else(|| {
                Error::Backend(BackendError::Unsupported {
                    backend: "mlx".into(),
                    capability: format!("{axis:?} collective on a singleton axis"),
                })
            }),
        }
    }

    /// Returns whether `scope` is implemented by an MLX logical subgroup.
    pub fn scope_is_logical(&self, scope: CollectiveScope) -> Result<bool, Error> {
        Ok(self.group(scope)?.is_logical())
    }

    /// Samples on the canonical final-stage rank and synchronizes generation globally.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut crate::backend::random::RandomState>,
        finished: bool,
    ) -> Result<SynchronizedToken, Error> {
        let topology = self.topology();
        let sampling_rank = topology.global_rank_for(ParallelCoordinates {
            tensor: 0,
            pipeline: topology.pipeline_parallel_size - 1,
            expert: 0,
            data: topology.data_parallel_rank,
        })?;
        self.sample_and_synchronize_on_rank(
            logits,
            batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
        )
    }

    /// Samples on an explicitly selected world rank and synchronizes globally.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize_on_rank<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut crate::backend::random::RandomState>,
        finished: bool,
        sampling_rank: usize,
    ) -> Result<SynchronizedToken, Error> {
        sample_and_synchronize(
            logits,
            batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            self.world(),
            &self.stream,
        )
    }

    /// Reaches global failure or cancellation consensus in a fixed order.
    pub fn operation_consensus(
        &self,
        local_failed: bool,
        local_cancelled: bool,
    ) -> Result<(bool, bool), Error> {
        let consensus =
            self.world_control_consensus([i32::from(local_failed), i32::from(local_cancelled)])?;
        Ok((consensus[0] != 0, consensus[1] != 0))
    }

    fn value_dtype(value: &ValueDescriptor) -> Result<Dtype, Error> {
        match &value.dtype {
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
            .shape
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
    type DistributedSession = MlxDistributedSession<'a>;

    fn distributed_session<'session>(
        session: &'session crate::composition::mlx::MlxModelSession<'a>,
    ) -> Option<&'session Self::DistributedSession> {
        session.distributed()
    }
}

impl DistributedSession for MlxDistributedSession<'_> {
    type Value = MlxTensor;
    type Completion = DistributedCompletion<MlxTensor>;
    type Error = Error;

    fn descriptor(&self) -> DistributedSessionDescriptor {
        let topology = self.topology();
        DistributedSessionDescriptor::new(topology.topology(), topology.global_rank)
            .expect("MLX rank belongs to its topology")
    }

    fn capabilities(&self) -> DistributedCapabilities {
        let topology = self.topology();
        let mut collective_axes = Vec::with_capacity(3);
        if topology.tensor_parallel_size > 1 {
            collective_axes.push(eredu_core::topology::ParallelAxis::Tensor);
        }
        if topology.pipeline_parallel_size > 1 {
            collective_axes.push(eredu_core::topology::ParallelAxis::Pipeline);
        }
        if topology.expert_parallel_size > 1 {
            collective_axes.push(eredu_core::topology::ParallelAxis::Expert);
        }
        DistributedCapabilities {
            world_collectives: true,
            collective_axes,
            point_to_point: true,
            variable_all_to_all: true,
            exact_completion: true,
        }
    }

    fn all_reduce_sum(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        let output = distributed::all_sum(input.as_array(), self.group(scope)?, &self.stream)?;
        let completion =
            DistributedCompletion::submit(MlxTensor::from_array(output.clone()), [&output])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }

    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        let output = distributed::all_gather(input.as_array(), self.group(scope)?, &self.stream)?;
        let completion =
            DistributedCompletion::submit(MlxTensor::from_array(output.clone()), [&output])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }

    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &MlxTensor,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        let output = distributed::all_to_all_v(
            input.as_array(),
            send_counts,
            receive_counts,
            self.group(scope)?,
            &self.stream,
        )?;
        let completion =
            DistributedCompletion::submit(MlxTensor::from_array(output.clone()), [&output])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }

    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &MlxTensor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        let output = distributed::send(input.as_array(), peer, self.group(scope)?, &self.stream)?;
        let completion =
            DistributedCompletion::submit(MlxTensor::from_array(output.clone()), [&output])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }

    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<MlxTensor, Self::Completion>, Error> {
        let shape = Self::value_shape(value)?;
        let output = distributed::recv(
            &shape,
            Self::value_dtype(value)?,
            peer,
            self.group(scope)?,
            &self.stream,
        )?;
        let completion =
            DistributedCompletion::submit(MlxTensor::from_array(output.clone()), [&output])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Error> {
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("distributed metadata exceeds i32".into()))?;
        let local = MlxTensor::from_array(Array::from_slice(local, &[length]));
        let gathered = DistributedSession::all_gather(self, CollectiveScope::World, &local)?;
        Ok(gathered
            .wait()?
            .as_array()
            .evaluated()?
            .as_slice::<u32>()
            .to_vec())
    }
}

impl eredu_core::consensus::ConsensusTransport for MlxDistributedSession<'_> {
    type Error = Error;

    fn participant_count(&self) -> usize {
        self.topology().world_size
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        DistributedSession::all_gather_words(self, local)
    }
}
