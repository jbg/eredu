//! Shared Nemotron residency realization from exact retained VoiceChat bindings.

use eredu_architectures::{nemotron_h, voicechat::backbone::PreparedBackboneParameters};
use eredu_checkpoint::store::CheckpointSource;
use eredu_core::BackendFailure;
use eredu_nn::NeuralBackend;
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayeredInvocation, LayerwiseRuntime,
    LayerwiseRuntimeError, ParallelLayeredInvocation, ResidencyReport,
};
use safemlx::{Array, Stream};

use crate::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        cache::state::MlxHybridState,
        execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
    },
    submission_recovery::detached_preparation,
};
use crate::MlxTensor;

type Architecture = nemotron_h::LayeredModel<MlxNeuralBackend>;
type Unit = nemotron_h::Unit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<Architecture, MlxNeuralBackend, MlxHybridState, P>;

enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Ordinary Nemotron traversal with resident, host-layerwise or disk-streamed blocks.
/// Static embedding, final normalization and agent readout weights remain pinned
/// and are charged inside the selected provider's device allowance.
pub struct MlxVoiceChatBackbone {
    execution: Execution,
}

impl MlxVoiceChatBackbone {
    /// Exact restricted source retained by the selected shared parameter policy.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store(),
        }
    }

    /// Number of retained static bundles outside independently acquired blocks.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().static_lease_count(),
            Execution::Bounded(runtime) => runtime.policy().static_lease_count(),
        }
    }

    /// Parameter residency and retained-source accounting for the selected policy.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry when the corresponding shared policy is selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }
}

impl LayeredInvocation<MlxNeuralBackend, MlxHybridState> for MlxVoiceChatBackbone {
    type Architecture = Architecture;
    type Error = LayerwiseRuntimeError<eredu_nn::Error, Error>;

    fn architecture_mut(&mut self) -> &mut Architecture {
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.architecture_mut(),
            Execution::Bounded(runtime) => runtime.architecture_mut(),
        }
    }

    fn invoke_with_context<'a>(
        &mut self,
        input: nemotron_h::EmbeddedInput<'a, MlxTensor>,
        state: &mut MlxHybridState,
        context: &Stream,
    ) -> Result<(MlxTensor, nemotron_h::ForwardContext<MlxTensor>), Self::Error> {
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.invoke_with_context(input, state, context),
            Execution::Bounded(runtime) => runtime.invoke_with_context(input, state, context),
        }
    }
}

impl ParallelLayeredInvocation<MlxNeuralBackend, MlxHybridState> for MlxVoiceChatBackbone {
    fn invoke_parallel<'a>(
        &mut self,
        input: nemotron_h::EmbeddedInput<'a, MlxTensor>,
        state: &mut MlxHybridState,
        parallel: &<MlxNeuralBackend as NeuralBackend>::ParallelContext,
        context: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.invoke_parallel(input, state, parallel, context)
            }
            Execution::Bounded(runtime) => runtime.invoke_parallel(input, state, parallel, context),
        }
    }
}

/// Binds the retained target with the ordinary shared parameter provider.
///
/// Source interpretation, unit geometry and equations remain architecture-owned.
/// This constructor creates no live cache or conversation state and returns no
/// partially loaded runtime on admission, materialization or completion failure.
pub fn materialize_voicechat_backbone(
    prepared: &PreparedBackboneParameters,
    residency: &LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<MlxVoiceChatBackbone, BackendFailure> {
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (mut architecture, bindings) = prepared
                .prepare_layerwise::<MlxNeuralBackend, MlxHybridState>(execution_stream)
                .map_err(BackendFailure::from_error)?;
            let (source, layout, static_bindings, units) = bindings.into_parts();
            let (policy, _) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<MlxHybridState>,
                *residency,
                execution_stream,
                weights_stream,
                |_| false,
                layout,
                static_bindings,
                units,
            )
            .map_err(BackendFailure::from_error)?;
            let execution = if residency.is_fully_resident() {
                let policy = policy
                    .into_resident(
                        &architecture,
                        execution_stream,
                        std::marker::PhantomData::<MlxHybridState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok(MlxVoiceChatBackbone { execution })
        },
        BackendFailure::from_error,
    )
}
