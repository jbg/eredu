//! MLX implementation of the optional distributed-session contract.

use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    BackendError, CollectiveScope, DistributedBackend, DistributedCapabilities, DistributedSession,
    DistributedSessionDescriptor, ParallelAxis, ParallelCoordinates, Submission, ValueDescriptor,
};
use safemlx::{
    distributed::{self, Group},
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::{
        distributed::{
            completion::DistributedCompletion,
            parallel::{sample_and_synchronize, ParallelExecutionContext, SynchronizedToken},
            topology::ParallelCommunicators,
        },
        generation::sampler::Sampler,
    },
};

use super::{MlxBackend, MlxParallelContext};

/// Inputs needed to attach MLX communication to one selected backend session.
pub(crate) struct MlxDistributedConfig<'a> {
    /// Validated rank-local topology.
    pub(crate) topology: MlxParallelContext,
    /// MLX world communicator selected for this complete session.
    pub(crate) world: &'a Group,
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
    pub(crate) fn new(config: MlxDistributedConfig<'a>, stream: &Stream) -> Result<Self, Error> {
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

    pub(crate) const fn world(&self) -> &Group {
        self.communicators.world()
    }

    pub(crate) fn tensor_context(&self) -> Result<ParallelExecutionContext<'_>, Error> {
        match self.communicators.tensor_group() {
            Some(group) => {
                ParallelExecutionContext::tensor_parallel(self.topology(), group, &self.stream)
            }
            None => Ok(ParallelExecutionContext::replicated(&self.stream)),
        }
    }

    pub(crate) fn tensor_group(&self) -> Option<&Group> {
        self.communicators.tensor_group()
    }

    pub(crate) fn expert_group(&self) -> Option<&Group> {
        self.communicators.expert_group()
    }

    pub(crate) fn pipeline_group(&self) -> Option<&Group> {
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
    pub fn send_pipeline(&self, hidden: &Array) -> Result<DistributedCompletion<Array>, Error> {
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
    ) -> Result<DistributedCompletion<Array>, Error> {
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
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
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
    pub fn sample_and_synchronize_on_rank<S: Sampler>(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
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
    type Value = Array;
    type Completion = DistributedCompletion<Array>;
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
        input: &Array,
    ) -> Result<Submission<Array, Self::Completion>, Error> {
        let output = distributed::all_sum(input, self.group(scope)?, &self.stream)?;
        let completion = DistributedCompletion::submit(output.clone(), [&output])?;
        Ok(Submission { output, completion })
    }

    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &Array,
    ) -> Result<Submission<Array, Self::Completion>, Error> {
        let output = distributed::all_gather(input, self.group(scope)?, &self.stream)?;
        let completion = DistributedCompletion::submit(output.clone(), [&output])?;
        Ok(Submission { output, completion })
    }

    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &Array,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<Array, Self::Completion>, Error> {
        let output = distributed::all_to_all_v(
            input,
            send_counts,
            receive_counts,
            self.group(scope)?,
            &self.stream,
        )?;
        let completion = DistributedCompletion::submit(output.clone(), [&output])?;
        Ok(Submission { output, completion })
    }

    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &Array,
    ) -> Result<Submission<Array, Self::Completion>, Error> {
        let output = distributed::send(input, peer, self.group(scope)?, &self.stream)?;
        let completion = DistributedCompletion::submit(output.clone(), [&output])?;
        Ok(Submission { output, completion })
    }

    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<Array, Self::Completion>, Error> {
        let shape = Self::value_shape(value)?;
        let output = distributed::recv(
            &shape,
            Self::value_dtype(value)?,
            peer,
            self.group(scope)?,
            &self.stream,
        )?;
        let completion = DistributedCompletion::submit(output.clone(), [&output])?;
        Ok(Submission { output, completion })
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Error> {
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("distributed metadata exceeds i32".into()))?;
        let local = Array::from_slice(local, &[length]);
        let gathered = DistributedSession::all_gather(self, CollectiveScope::World, &local)?;
        Ok(gathered.wait()?.evaluated()?.as_slice::<u32>().to_vec())
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
