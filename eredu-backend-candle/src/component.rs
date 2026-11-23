//! Native binding and rank-local resources for prepared components.

use std::convert::Infallible;

use eredu_core::{
    automatic::ColdPlanPlacement,
    backend::{local_text::NoTextTransport, LocalComponentStorage},
    communication::{
        pipeline::{ComponentPipelineMechanisms, NativeComponentRank},
        BoundaryTensorDtype, BroadcastBackend, CommunicationAuthority, FailureAgreementBackend,
        PartitionTensorAllocator, PipelineActivationDtype, PointToPointBackend,
        RoleExactBoundaryValue, SumReductionBackend, UnevenGatherBackend,
        WorldCommunicationBinding,
    },
    decision::{DecisionBinding, DecisionBindingFactory},
    BackendFailure, BoundedCompletionOutcome, BoundedCompletionWait, DevicePlan, Submission,
};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{backend::TextMediaNative, input::MediaInputLowering};
use eredu_nn::Error;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::CandleInputInspector;
use crate::{
    realtime::NoCandleRank, CandleBackendFactory, CandleContext, CandleNeuralBackend, CandleTensor,
};

impl DecisionBindingFactory for CandleBackendFactory {
    type Binding = CandleContext;

    fn decision_binding(&self, device: &DevicePlan) -> Result<Self::Binding, BackendFailure> {
        self.plan_placement(device, None)
            .map_err(BackendFailure::from_error)?;
        self.open()
    }
}

impl DecisionBinding for CandleContext {
    type Backend = CandleNeuralBackend;
    type Parallel = Infallible;
    #[cfg(feature = "image")]
    type InputLowering = Self;
    #[cfg(not(feature = "image"))]
    type InputLowering = ();

    fn context(&self) -> &Self {
        self
    }

    fn input_lowering(&self) -> Self::InputLowering {
        #[cfg(feature = "image")]
        return self.clone();
        #[cfg(not(feature = "image"))]
        ()
    }

    fn synchronize(&self) -> Result<(), BackendFailure> {
        CandleContext::synchronize(self)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl MediaInputLowering<CandleTensor> for CandleContext {
    type ProcessorTensor = CandleTensor;
    type ProcessorError = BackendFailure;
    type Scope = crate::text_native::CandleProcessorScope;

    fn begin(&self) -> Result<Self::Scope, BackendFailure> {
        self.begin_processor()
    }

    fn inspector(&self) -> &impl eredu_core::input::PreparedInputInspector<CandleTensor> {
        &CandleInputInspector
    }

    fn convert_tensor(&self, tensor: &CandleTensor) -> CandleTensor {
        tensor.clone()
    }
}

impl NativeComponentRank<CandleNeuralBackend, Infallible> for CandleContext {
    type PipelineRank = NoCandleRank;
    type PipelineSetupError = BackendFailure;
    type PipelineSetup<'a> = NoTextTransport;
    type PipelineAllocator = UnavailablePipeline;
    type PipelineExecution = UnavailablePipeline;

    fn supports_independent_agreement(&self, _: usize, _: &[usize]) -> bool {
        false
    }

    fn pipeline_context(&self) -> &Self {
        self
    }

    fn open_pipeline_setup(&self, communication: &Infallible) -> Option<NoTextTransport> {
        match *communication {}
    }

    fn realize_pipeline_rank(
        &self,
        communication: &Infallible,
        _: Option<WorldCommunicationBinding<NoTextTransport>>,
    ) -> Result<NoCandleRank, BackendFailure> {
        match *communication {}
    }

    fn pipeline_mechanisms(
        &self,
        _: Vec<Infallible>,
    ) -> Result<(UnavailablePipeline, UnavailablePipeline), BackendFailure> {
        Err(crate::realtime::unsupported(
            "Candle component binding requires local execution",
        ))
    }
}

#[doc(hidden)]
pub struct UnavailablePipeline(Infallible);

impl PartitionTensorAllocator<CandleNeuralBackend> for UnavailablePipeline {
    fn tensor_placeholder(
        &mut self,
        _: &[i32],
        _: BoundaryTensorDtype,
        _: PipelineActivationDtype,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        match self.0 {}
    }
}

impl ComponentPipelineMechanisms<CandleNeuralBackend> for UnavailablePipeline {
    type Scope = crate::residency::CandleUnitScope;

    fn begin(&self, _: Vec<CandleTensor>) -> Result<Self::Scope, BackendFailure> {
        match self.0 {}
    }

    fn complete_outputs(
        &self,
        _: Vec<CandleTensor>,
        _: &CommunicationAuthority,
        _: BoundedCompletionWait,
        _: &CandleContext,
    ) -> Result<BoundedCompletionOutcome, BackendFailure> {
        match self.0 {}
    }

    fn settle_values<'a>(
        values: impl Iterator<Item = &'a CandleTensor>,
    ) -> Result<(), BackendFailure> {
        CandleNeuralBackend::settle_local_values(values)
    }
}

impl PointToPointBackend for CandleNeuralBackend {
    fn send_receive(
        _: Vec<RoleExactBoundaryValue<CandleTensor>>,
        route: &Infallible,
        _: &CandleContext,
    ) -> Result<Submission<Vec<CandleTensor>, Self::CommunicationCompletion>, Error> {
        match *route {}
    }
}

impl BroadcastBackend for CandleNeuralBackend {
    fn broadcast(
        _: CandleTensor,
        _: usize,
        group: &Infallible,
        _: &CandleContext,
    ) -> Result<Submission<CandleTensor, Self::CommunicationCompletion>, Error> {
        match *group {}
    }
}

impl FailureAgreementBackend for CandleNeuralBackend {
    type FailureAgreementOutput = Infallible;

    fn agree_success(
        _: bool,
        _: Option<[u64; 8]>,
        group: &Infallible,
        _: &CandleContext,
    ) -> Result<Submission<Infallible, Self::CommunicationCompletion>, Error> {
        match *group {}
    }

    fn resolve_failure_agreement(output: Infallible) -> Result<bool, Error> {
        match output {}
    }
}

impl SumReductionBackend for CandleNeuralBackend {
    fn all_reduce_sum(
        _: CandleTensor,
        group: &Infallible,
        _: &CandleContext,
    ) -> Result<Submission<CandleTensor, Self::CommunicationCompletion>, Error> {
        match *group {}
    }
}

impl UnevenGatherBackend for CandleNeuralBackend {
    fn all_gather_uneven(
        _: CandleTensor,
        _: &[usize],
        _: usize,
        group: &Infallible,
        _: &CandleContext,
    ) -> Result<Submission<CandleTensor, Self::CommunicationCompletion>, Error> {
        match *group {}
    }
}
