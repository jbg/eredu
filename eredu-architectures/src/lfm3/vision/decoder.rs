//! Portable language-backbone execution for retained Decision and VL inputs.

use std::num::NonZeroUsize;

use eredu_core::{
    residency::ParameterConversionRetentionEligibility, BackendFailure, DistributedCommitEpoch,
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Tensor, TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    BoundComponentPipeline, BoundLocalComponent, DecoderPipelineState, DenseDiskStreamReport,
    LayerWeightResidency, LocalComponentPolicy, NoopObserver, PreparedComponentPipeline,
    PreparedLocalComponent, ResidencyReport, RuntimeStateComponents,
};

use crate::lfm3::{
    LanguagePipelineCommunication, LayeredModel, Lfm3Input, PartitionedLayeredModel,
    PreparedLanguagePartition,
};

use super::{
    PolicyVision, PreparedVlDecoder, PreparedVlDecoderStage, VisionModule, VisionPromptPreparation,
    VlPrefill, VlPrefillProgress,
};

impl PreparedVlDecoder {
    /// Constructs a local backbone with the requested parameter residency.
    pub fn prepare_local<B>(
        self,
        residency: LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedLocalComponent<LayeredModel<B>>, BackendFailure>
    where
        B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    {
        let (architecture, component) = self
            .into_layerwise::<B>(context)
            .map_err(BackendFailure::from_error)?;
        let conversion_eligibility = match residency {
            LayerWeightResidency::LayerwiseHost(_) => {
                ParameterConversionRetentionEligibility::HostLayerwise
            }
            LayerWeightResidency::DenseDiskStream(_) => {
                ParameterConversionRetentionEligibility::DiskStreamed
            }
            _ => ParameterConversionRetentionEligibility::Eligible,
        };
        PreparedLocalComponent::new(
            architecture,
            component,
            LocalComponentPolicy {
                residency,
                state: None,
                conversion_retention: None,
                conversion_eligibility,
            },
        )
        .map_err(BackendFailure::from_error)
    }
}

/// A bound LFM language backbone, with request state supplied by its caller.
pub struct VlDecoder<B, S, L, D> {
    execution: Execution<L, D>,
    types: std::marker::PhantomData<fn(B, S)>,
}

enum Execution<L, D> {
    Local(L),
    Partitioned {
        stage: PreparedVlDecoderStage<D>,
        epoch: DistributedCommitEpoch,
    },
}

impl<B, S, L, D> VlDecoder<B, S, L, D>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    L: BoundLocalComponent<LayeredModel<B>, B, S>,
    D: BoundComponentPipeline<PartitionedLayeredModel<B>, B, S>,
{
    /// Uses the same local component mechanism as ordinary layerwise execution.
    pub fn local(component: L) -> Self {
        Self {
            execution: Execution::Local(component),
            types: std::marker::PhantomData,
        }
    }

    /// Constructs a rank's retained decoder and binds it through the generic pipeline mechanism.
    pub fn bind_rank(
        prepared: PreparedLanguagePartition,
        residency: LayerWeightResidency,
        communication: LanguagePipelineCommunication,
        context: &<B::Tensor as Tensor>::Context,
        bind: impl FnOnce(
            PreparedComponentPipeline<PartitionedLayeredModel<B>>,
        ) -> Result<D, BackendFailure>,
    ) -> Result<Self, BackendFailure> {
        let stage =
            super::prepare_vl_decoder_stage::<B, S>(prepared, residency, communication, context)?
                .bind(bind)?;
        Ok(Self {
            execution: Execution::Partitioned {
                stage,
                epoch: DistributedCommitEpoch::FIRST,
            },
            types: std::marker::PhantomData,
        })
    }

    /// Allocates fresh caller-owned decoder state.
    pub fn new_state(&self) -> Result<S, BackendFailure> {
        match &self.execution {
            Execution::Local(local) => local.fresh_state(),
            Execution::Partitioned { stage, .. } => {
                S::allocate_pipeline_state(stage.state_layout.clone(), stage.global_layer_start)
            }
        }
    }

    /// Executes an input against the caller's current state.
    pub fn forward(
        &mut self,
        input: Lfm3Input<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, BackendFailure> {
        match &mut self.execution {
            Execution::Local(local) => local
                .execute(input, state, &mut NoopObserver)
                .map_err(BackendFailure::from_error),
            Execution::Partitioned { stage, epoch } => super::rank_session::execute::<B, S, D>(
                &mut stage.pipeline,
                Ok(input),
                state,
                epoch,
                stage.prefix,
                context,
            ),
        }
    }

    /// Agrees a caller request before entering model execution.
    pub fn agree_application(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, BackendFailure> {
        match &mut self.execution {
            Execution::Local(_) => Ok(ready),
            Execution::Partitioned { stage, .. } => {
                stage.pipeline.agree_application(descriptor, ready, context)
            }
        }
    }

    /// Reports the storage mechanism's residency and disk activity.
    pub fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        match &self.execution {
            Execution::Local(local) => local.residency(),
            Execution::Partitioned { stage, .. } => Ok((
                stage.pipeline.residency()?,
                stage.pipeline.dense_stream_report()?,
            )),
        }
    }

    /// Admits one prefill request and allocates its independent state.
    pub fn start_prefill(
        &mut self,
        prompt: VisionPromptPreparation<B::Tensor>,
        chunk: NonZeroUsize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(VlPrefill<B::Tensor>, S), BackendFailure> {
        let state = self.new_state().map(|state| ((prompt, chunk), state));
        let maximum = match &self.execution {
            Execution::Local(_) => None,
            Execution::Partitioned { stage, .. } => Some(stage.prefix),
        };
        super::admit_partitioned_vl_prefill(state, maximum, |descriptor, ready| {
            self.agree(descriptor, ready, context)
        })
    }

    /// Advances a request through its supplied vision projection and decoder state.
    pub fn advance_prefill<P>(
        &mut self,
        vision: &mut PolicyVision<B, P>,
        cursor: &mut VlPrefill<B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<VlPrefillProgress<B::Tensor>>, BackendFailure>
    where
        P: eredu_runtime::LayerwisePolicy<B, VisionModule<B>>,
        P::Error: std::error::Error + Send + Sync + 'static,
    {
        let distributed = matches!(self.execution, Execution::Partitioned { .. });
        super::prefill::advance_partitioned_vl_prefill(
            cursor,
            context,
            vision,
            self,
            distributed,
            |decoder, descriptor, ready| decoder.agree(descriptor, ready, context),
            |decoder, prompt| decoder.forward(Lfm3Input::Vision(prompt), state, context),
            |cursor| {
                if distributed {
                    D::settle_values(cursor.retained_values())
                } else {
                    L::settle(cursor.retained_values())
                }
            },
        )
    }

    fn agree(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        match &mut self.execution {
            Execution::Local(_) => Ok(()),
            Execution::Partitioned { stage, .. } => {
                stage.pipeline.agree(descriptor, ready, context)
            }
        }
    }
}
