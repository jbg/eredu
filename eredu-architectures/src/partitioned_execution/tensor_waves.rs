//! Shared inactive-stage tensor participation for selected world collectives.
use super::*;
use std::sync::{Arc, OnceLock};

/// Architecture-derived tensor-only stage schedule. Native composition supplies
/// only whether its selected group mechanism requires complete-world participation.
#[derive(Debug)]
pub struct TensorPipelineCollectiveWaves {
    schedule: RoutedExpertCollectiveWaveSchedule,
    group: eredu_core::CollectiveGroupId,
    requires_world: OnceLock<bool>,
}

impl<A, R, Q, G, D> PreparedPartitionedAdmission<A, R, Q, G, D>
{
    /// Retains the existing architecture reduction and partition declarations.
    /// This cold method never inspects a native communicator or executes a model.
    pub(crate) fn tensor_pipeline_collective_waves<B, S>(
        &self,
    ) -> Result<Option<Arc<TensorPipelineCollectiveWaves>>, String>
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: TextPartitionArchitecture<B, S>,
        A::Error: std::fmt::Display,
    {
        let selected = self.selected();
        let topology = selected.topology();
        if topology.tensor_parallel_size() <= 1 || topology.pipeline_parallel_size() <= 1 {
            return Ok(None);
        }
        if let Some(waves) = self.pipeline_tensor_waves.get() { return Ok(Some(waves.clone())); }
        let group = selected
            .tensor_group()
            .ok_or_else(|| "tensor pipeline selection has no tensor group".to_owned())?;
        let hidden = selected
            .boundary_routes()
            .first()
            .and_then(|route| route.schema().primary().shape().last())
            .copied()
            .and_then(|width| usize::try_from(width).ok())
            .filter(|width| *width > 0)
            .ok_or_else(|| "tensor pipeline selection has no hidden wire width".to_owned())?;
        let architecture = self.architecture();
        let width = usize::try_from(architecture.partition_output_width())
            .map_err(|_| "tensor pipeline output width is negative".to_owned())?;
        let blocks = (0..selected.partition().unit_layout().len())
            .map(|unit| {
                architecture
                    .partition_routed_tensor_reductions(unit, false)
                    .map(|sums| vec![RoutedExpertUnitWave::ordinary(unit, hidden, sums)])
                    .map_err(|cause| cause.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let schedule = RoutedExpertCollectiveWaveSchedule::from_block_waves(
            blocks,
            topology.tensor_parallel_size(),
            topology.tensor_parallel_rank(),
            topology.pipeline_parallel_size(),
            width,
        )?;
        let waves = Arc::new(TensorPipelineCollectiveWaves {
            schedule, group, requires_world: OnceLock::new(),
        });
        let _ = self.pipeline_tensor_waves.set(waves);
        Ok(self.pipeline_tensor_waves.get().cloned())
    }
}

impl<B, A, G, D> PreparedPartitionedArchitecture<B, A, G, D>
where B: eredu_nn::NeuralBackend,
{
    /// Returns the exact retained semantic plan shared with later workspace quotes.
    pub fn tensor_pipeline_collective_waves<S>(&self)
        -> Result<Option<Arc<TensorPipelineCollectiveWaves>>, String>
    where S: eredu_runtime::RuntimeState<B>, A: TextPartitionArchitecture<B, S>,
        A::Error: std::fmt::Display,
    {
        self.prepared.tensor_pipeline_collective_waves::<B, S>()
    }
}

impl<A, B, S, P, F, U> PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Binds the retained semantic schedule to the selected native mechanism fact.
    pub fn with_tensor_collective_waves(
        mut self,
        waves: Option<Arc<TensorPipelineCollectiveWaves>>,
        requires_world: bool,
    ) -> Result<Self, eredu_nn::Error> {
        if let Some(waves) = &waves {
            match waves.requires_world.set(requires_world) {
                Ok(()) => (),
                Err(value) if waves.requires_world.get() == Some(&value) => (),
                Err(_) => return Err(eredu_nn::Error::backend(
                    "retained tensor participation mechanism changed after binding")),
            }
        }
        if requires_world {
            self.tensor_waves = Some(waves.ok_or_else(|| eredu_nn::Error::backend(
                "world tensor mechanism has no retained pipeline participation schedule"))?);
        }
        Ok(self)
    }
    /// Quotes borrow the actual bound mechanism fact without reconstructing it
    /// from topology sizes or receiving a native communicator.
    pub(crate) fn with_retained_tensor_collective_waves(mut self,
        waves: Option<&Arc<TensorPipelineCollectiveWaves>>, context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(Option<Arc<TensorPipelineCollectiveWaves>>,
            &Self, Option<&Arc<TensorPipelineCollectiveWaves>>, Option<&bool>, Result<(), eredu_nn::Error>)>())?;
        if let Some(waves) = waves {
            let required = waves.requires_world.get().copied().ok_or_else(|| context.metadata_error(
                format_args!("tensor pipeline quote has no retained mechanism binding")))?;
            if required { self.tensor_waves = Some(waves.clone()); }
        }
        Ok(self)
    }

}

impl TensorPipelineCollectiveWaves {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn participate<B, G, R, I, F>(
        &self,
        wave: usize,
        demand: eredu_core::OutputDemand,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        executor: &B::Executor,
        allocator: &mut F,
        dtype: PipelineActivationDtype,
        batch: i32,
        sequence: i32,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        parallel: Option<&B::ParallelContext>,
    ) -> Result<(), eredu_nn::Error>
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_runtime::CommunicationBackend
            + eredu_runtime::SumReductionBackend
            + eredu_runtime::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        let metadata = B::construction_metadata(context).filter(|context| context.uses_checked_metadata());
        if let Some(metadata) = metadata {
            let frames = [
                std::mem::size_of::<(&Self, usize, eredu_core::OutputDemand,
                    &eredu_runtime::PartitionCommunication<B,G,R,I>, &B::Executor, &mut F,
                    PipelineActivationDtype, i32, i32, &<B::Tensor as eredu_nn::Tensor>::Context)>(),
                std::mem::size_of::<Option<&eredu_nn::workspace::WorkspaceContext>>(),
                std::mem::size_of::<Result<(),eredu_nn::Error>>(),
                std::mem::size_of::<Option<&[RoutedExpertUnitWave]>>(),
                std::mem::size_of::<std::slice::Iter<'_,RoutedExpertUnitWave>>(),
                std::mem::size_of::<std::ops::Range<usize>>(),
                std::mem::size_of::<(usize, i32, i32, &[RoutedExpertUnitWave], &RoutedExpertUnitWave)>(),
                std::mem::size_of::<(Option<&RoutedTensorCollectiveWaveSchedule>, &[usize])>(),
                std::mem::size_of::<([i32;3], B::Tensor, Result<B::Tensor,eredu_nn::Error>,
                    Result<B::Tensor,eredu_runtime::PartitionExecutionError>)>(),
                // Reduction and error closures retain only these borrowed fields.
                std::mem::size_of::<(&Self, &mut F, &B::Executor,
                    &eredu_runtime::PartitionCommunication<B,G,R,I>,
                    &<B::Tensor as eredu_nn::Tensor>::Context, i32, i32,
                    PipelineActivationDtype, Option<&eredu_nn::workspace::WorkspaceContext>, &'static str)>(),
            ];
            let bytes = frames.iter().copied().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            metadata.charge_metadata(bytes)?;
        }
        let error = |message: &'static str| match metadata {
            Some(metadata) => metadata.metadata_error(format_args!("{message}")),
            None => eredu_nn::Error::backend(message),
        };
        let units = self.schedule.stage(wave).ok_or_else(|| {
            error("tensor pipeline wave is outside its retained schedule")
        })?;
        let mut reduce = |width: usize| -> Result<(), eredu_nn::Error> {
            let width = i32::try_from(width).map_err(|_| {
                error("tensor pipeline hidden width exceeds i32")
            })?;
            let value = allocator.tensor_placeholder(
                &[batch, sequence, width],
                eredu_runtime::BoundaryTensorDtype::Activation,
                dtype,
                context,
            )?;
            communication
                .all_reduce_sum_with_parallel(value, self.group, executor, parallel)
                .map(|_| ())
                .map_err(|cause| match metadata {
                    Some(metadata) => metadata.metadata_source(cause),
                    None => eredu_nn::Error::backend_source(cause),
                })
        };
        if wave == 0 {
            reduce(
                units
                    .first()
                    .ok_or_else(|| error("tensor pipeline stage has no units"))?
                    .hidden_width(),
            )?;
        }
        for unit in units {
            for _ in 0..unit.tensor_reductions_before() {
                reduce(unit.hidden_width())?;
            }
            for _ in 0..unit.tensor_reductions_after() {
                reduce(unit.hidden_width())?;
            }
        }
        if wave + 1 == self.schedule.stage_count() && demand != eredu_core::OutputDemand::StateOnly
        {
            let tensor = self.schedule.tensor().ok_or_else(|| {
                error("tensor pipeline schedule has no output partition")
            })?;
            let width = i32::try_from(tensor.vocabulary_widths[tensor.rank])
                .map_err(|_| error("tensor pipeline vocabulary exceeds i32"))?;
            let positions =
                i32::try_from(demand.positions(u64::try_from(sequence).map_err(|_| {
                    error("tensor pipeline sequence is negative")
                })?))
                .map_err(|_| error("tensor pipeline readout exceeds i32"))?;
            let value = allocator.tensor_placeholder(
                &[batch, positions, width],
                eredu_runtime::BoundaryTensorDtype::Activation,
                dtype,
                context,
            )?;
            communication
                .all_gather_uneven_with_parallel(value, &tensor.vocabulary_widths, 2, self.group, executor, parallel)
                .map_err(|cause| match metadata {
                    Some(metadata) => metadata.metadata_source(cause),
                    None => eredu_nn::Error::backend_source(cause),
                })?;
        }
        Ok(())
    }
}
