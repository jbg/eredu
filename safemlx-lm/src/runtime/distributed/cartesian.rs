//! Cartesian distributed execution contexts shared by combined parallel modes.
//!
//! This module owns no architecture dispatch. It binds one validated
//! [`ParallelTopology`](crate::ParallelTopology) to topology-derived TP, PP,
//! and EP communicators and
//! provides the transport and consensus primitives used by model adapters.

use safemlx::{
    distributed::{self, Group},
    ops::{ones, zeros},
    Array, Dtype, Stream,
};

use crate::{
    error::Error,
    runtime::{
        distributed::{
            completion::{synchronize_outputs, DistributedCompletion},
            parallel::{sample_and_synchronize, ParallelExecutionContext, SynchronizedToken},
            topology::{
                ParallelCommunicators, ParallelCoordinates, ParallelTopology,
                TopologyPreflightReport,
            },
        },
        generation::sampler::Sampler,
    },
};

/// Borrowed execution resources for one Cartesian distributed rank.
///
/// Construct this value before materializing checkpoint weights. Native
/// subgroup construction and backend compatibility are validated by
/// [`ParallelCommunicators::new`].
#[derive(Debug)]
pub struct CartesianExecution<'a> {
    preflight: TopologyPreflightReport,
    communicators: ParallelCommunicators<'a>,
}

impl<'a> CartesianExecution<'a> {
    /// Validates geometry and creates all required communicator contexts.
    pub fn new(
        topology: ParallelTopology,
        decoder_layers: Option<usize>,
        routed_experts: Option<usize>,
        world: &'a Group,
    ) -> Result<Self, Error> {
        let preflight = topology.preflight(decoder_layers, routed_experts)?;
        let communicators = ParallelCommunicators::new(topology, world)?;
        Ok(Self {
            preflight,
            communicators,
        })
    }

    /// Returns the complete weight-independent preflight report.
    pub const fn preflight(&self) -> &TopologyPreflightReport {
        &self.preflight
    }

    /// Returns the Cartesian topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.preflight.topology
    }

    /// Returns the global consensus group.
    pub const fn world(&self) -> &Group {
        self.communicators.world()
    }

    /// Creates the stage-local TP execution context.
    pub fn tensor_context<'s>(
        &'s self,
        stream: &'s Stream,
    ) -> Result<ParallelExecutionContext<'s>, Error> {
        match self.communicators.tensor_group() {
            Some(group) => {
                ParallelExecutionContext::tensor_parallel(self.topology(), group, stream)
            }
            None => Ok(ParallelExecutionContext::replicated(stream)),
        }
    }

    /// Returns the stage- and TP-local EP exchange group.
    pub fn expert_group(&self) -> Option<&Group> {
        self.communicators.expert_group()
    }

    /// Returns the matching-coordinate pipeline lane group.
    pub fn pipeline_group(&self) -> Option<&Group> {
        self.communicators.pipeline_group()
    }

    /// Submits a receive from the preceding matching-coordinate stage.
    ///
    /// The returned value owns both the received array and its exact completion.
    /// Reading it on the host requires `into_value` or `synchronize`; dependent
    /// work on another compatible stream must first call `wait_on`.
    pub fn receive_pipeline(
        &self,
        shape: &[i32],
        dtype: Dtype,
        stream: &Stream,
    ) -> Result<DistributedCompletion<Array>, Error> {
        let topology = self.topology();
        if topology.pipeline_parallel_rank == 0 {
            return Err(Error::Parallel(
                "the first pipeline stage has no predecessor".into(),
            ));
        }
        let group = self.pipeline_group().ok_or_else(|| {
            Error::Parallel("pipeline receive requires an active PP communicator".into())
        })?;
        let received = distributed::recv(
            shape,
            dtype,
            topology.pipeline_parallel_rank - 1,
            group,
            stream,
        )?;
        DistributedCompletion::submit(received.clone(), [&received])
    }

    /// Submits hidden activations to the succeeding matching-coordinate stage.
    ///
    /// The returned completion retains the send endpoint. Callers may wait on
    /// it from a compatible stream or synchronize it before host-visible reuse.
    pub fn send_pipeline(
        &self,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<DistributedCompletion<()>, Error> {
        let topology = self.topology();
        if topology.pipeline_parallel_rank + 1 == topology.pipeline_parallel_size {
            return Err(Error::Parallel(
                "the final pipeline stage has no successor".into(),
            ));
        }
        let group = self.pipeline_group().ok_or_else(|| {
            Error::Parallel("pipeline send requires an active PP communicator".into())
        })?;
        let sent = distributed::send(hidden, topology.pipeline_parallel_rank + 1, group, stream)?;
        DistributedCompletion::submit((), [&sent])
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
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        let topology = self.topology();
        let sampling_rank = topology.global_rank_for(ParallelCoordinates {
            tensor: 0,
            pipeline: topology.pipeline_parallel_size - 1,
            expert: 0,
        })?;
        sample_and_synchronize(
            logits,
            batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            self.world(),
            stream,
        )
    }

    /// Reaches global failure or cancellation consensus with two fixed collectives.
    ///
    /// Every rank must call this method in the same order, including ranks whose
    /// local operation succeeded. The returned pair is `(failed, cancelled)`.
    pub fn operation_consensus(
        &self,
        local_failed: bool,
        local_cancelled: bool,
        stream: &Stream,
    ) -> Result<(bool, bool), Error> {
        let failed = if local_failed {
            ones::<i32>(&[], stream)?
        } else {
            zeros::<i32>(&[], stream)?
        };
        let cancelled = if local_cancelled {
            ones::<i32>(&[], stream)?
        } else {
            zeros::<i32>(&[], stream)?
        };
        let failed = distributed::all_sum(&failed, self.world(), stream)?;
        let cancelled = distributed::all_sum(&cancelled, self.world(), stream)?;
        synchronize_outputs([&failed, &cancelled])?;
        Ok((
            failed.try_item::<i32>(stream)? != 0,
            cancelled.try_item::<i32>(stream)? != 0,
        ))
    }
}
