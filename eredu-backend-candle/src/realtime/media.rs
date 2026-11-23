//! Local resources for realtime media graphs.

use std::convert::Infallible;

use eredu_core::{
    communication::pipeline::NativePipelineWorld,
    execution_control::SnapshotRetention,
    input::MediaInputLowering,
    realtime_mechanisms::{MediaGraphNative, RealtimeGraphResources},
    sampling::host_tensor::HostTensorSampling,
    BackendFailure,
};

use crate::{CandleContext, CandleNeuralBackend, CandleTensor};

use super::unsupported;

impl RealtimeGraphResources for CandleContext {
    type Backend = CandleNeuralBackend;
    type Sampling = HostTensorSampling<CandleTensor>;

    fn graph_context(&self) -> &CandleContext {
        self
    }

    fn open_input_recovery(
        context: &CandleContext,
    ) -> Result<impl Sized + 'static + use<>, BackendFailure> {
        context.check_healthy()?;
        Ok(crate::device::CandleSettlement::new(context, ()))
    }

    fn hold_fork_retention(
        context: &CandleContext,
        retention: SnapshotRetention,
    ) -> Result<impl Sized + 'static + use<>, BackendFailure> {
        context.check_healthy()?;
        Ok(crate::device::CandleSettlement::new(context, retention))
    }
}

impl MediaGraphNative for CandleContext {
    type Neural = CandleNeuralBackend;
    type Graph = CandleContext;
    type World = CandleLocalWorld;

    fn graph(&self) -> Self::Graph {
        self.clone()
    }

    fn media_lowering(&self) -> impl MediaInputLowering<CandleTensor> {
        self.clone()
    }

    fn pipeline_world(&self) -> Self::World {
        CandleLocalWorld
    }
}

/// The local backend has no distributed media world or partitioned rank.
#[doc(hidden)]
pub struct CandleLocalWorld;

impl NativePipelineWorld for CandleLocalWorld {
    type Transport = Infallible;

    fn open_world(&self) -> Result<Self::Transport, BackendFailure> {
        Err(unsupported("Candle media graphs require local execution"))
    }

    fn identity(world: &Infallible) -> (usize, usize) {
        match *world {}
    }
}
