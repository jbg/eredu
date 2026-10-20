//! MLX realization of codec-prepared, independently leased EAR neural decoder.

use eredu_codec::{
    ear_decoder::{DecoderConfig, DecoderState, DecoderUnit},
    ear_decoder_layered::{DecoderExecution, DecoderKernel, DecoderRuntimeState, LayerwiseDecoder},
    ear_decoder_preparation::PreparedDecoderBindings,
    Error,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, LocalModelLayout,
    ResidencyReport,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::Group,
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = DecoderUnit<MlxTensor>;
enum Execution {
    Resident(LayerwiseDecoder<MlxNeuralBackend, MlxResidentPolicy<Unit>>),
    Bounded(LayerwiseDecoder<MlxNeuralBackend, MlxLayerwisePolicy<Unit>>),
}

/// Native EAR neural decoder over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain codec-owned.
pub struct MlxEarDecoder {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxEarDecoder {
    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().residency_report(),
            Execution::Bounded(model) => model.runtime().policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry, present only when that policy was selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .runtime()
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Number of pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().static_lease_count(),
            Execution::Bounded(model) => model.runtime().policy().static_lease_count(),
        }
    }
}

impl DecoderExecution<MlxNeuralBackend> for MlxEarDecoder {
    fn config(&self) -> &DecoderConfig {
        match &self.execution {
            Execution::Resident(model) => model.config(),
            Execution::Bounded(model) => model.config(),
        }
    }
    fn step(
        &mut self,
        latent: &MlxTensor,
        parent: &DecoderState<MlxTensor>,
        context: &Stream,
    ) -> Result<(MlxTensor, DecoderState<MlxTensor>), Error> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.step_parallel(latent, parent, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.step_parallel(latent, parent, parallel, context)
            }
            (Execution::Resident(model), None) => model.step(latent, parent, context),
            (Execution::Bounded(model), None) => model.step(latent, parent, context),
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The codec supplies geometry, unit order, and all storage conversions.
/// This component constructor does not assemble the quantizer or spectral decoder
/// and does not select a conversational execution policy.
pub fn materialize_ear_decoder(
    prepared: PreparedDecoderBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarDecoder, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds codec-selected feed-forward shards to native tensor parallel execution.
/// Streaming convolution state and spectral outputs remain replicated.
pub fn materialize_ear_decoder_parallel(
    prepared: PreparedDecoderBindings,
    parallel: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarDecoder, LayerwiseModelMetadata), BackendFailure> {
    let parallel = Group::uncontracted(parallel);
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, &parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedDecoderBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarDecoder, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(Error::InvalidShape(
            "decoder parameter placement differs from the selected execution".into(),
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (config, prefix, local_layout, layout, source, units) = prepared.into_parts();
            let mut architecture = match local_layout.as_ref() {
                Some(local) => {
                    DecoderKernel::<MlxNeuralBackend>::new_parallel(config, &prefix, local)
                }
                None => DecoderKernel::<MlxNeuralBackend>::new(config, &prefix),
            }
            .and_then(DecoderKernel::into_component)
            .map_err(BackendFailure::from_error)?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<DecoderRuntimeState<MlxTensor>>,
                residency,
                execution_stream,
                weights_stream,
                |_| false,
                layout,
                Vec::new(),
                units,
            )
            .map_err(BackendFailure::from_error)?;
            let execution = if residency.is_fully_resident() {
                let policy = policy
                    .into_resident(
                        &architecture,
                        execution_stream,
                        std::marker::PhantomData::<DecoderRuntimeState<MlxTensor>>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseDecoder::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseDecoder::new(architecture, policy))
            };
            Ok((
                MlxEarDecoder {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
