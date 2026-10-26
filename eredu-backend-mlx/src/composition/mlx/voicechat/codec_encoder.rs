//! MLX realization of codec-prepared, independently leased EAR neural encoder.

use eredu_codec::{
    ear_encoder::{EncoderConfig, EncoderUnit},
    ear_encoder_layered::{EncoderExecution, EncoderKernel, ParallelEncoderExecution},
    ear_encoder_preparation::PreparedEncoderBindings,
    Error,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, DenseDiskStreamReport, LayerWeightResidency,
    LayerwiseModelMetadata, LayerwiseRuntime, LocalModelLayout, ResidencyReport,
    StatelessComponentState,
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

type Unit = EncoderUnit<MlxTensor>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, EncoderKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    StatelessComponentState,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native EAR neural encoder over the selected shared resident or bounded parameter policy.
/// Finite spectral-to-latent equations remain codec-owned.
pub struct MlxEarEncoder {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxEarEncoder {
    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.policy().residency_report(),
            Execution::Bounded(model) => model.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry, present only when that policy was selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Number of pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.policy().static_lease_count(),
            Execution::Bounded(model) => model.policy().static_lease_count(),
        }
    }
}

impl EncoderExecution<MlxNeuralBackend> for MlxEarEncoder {
    fn config(&self) -> &EncoderConfig {
        match &self.execution {
            Execution::Resident(model) => model.config(),
            Execution::Bounded(model) => model.config(),
        }
    }
    fn forward(&mut self, spectrum: &MlxTensor, context: &Stream) -> Result<MlxTensor, Error> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                ParallelEncoderExecution::forward_parallel(model, spectrum, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                ParallelEncoderExecution::forward_parallel(model, spectrum, parallel, context)
            }
            (Execution::Resident(model), None) => {
                EncoderExecution::forward(model, spectrum, context)
            }
            (Execution::Bounded(model), None) => {
                EncoderExecution::forward(model, spectrum, context)
            }
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The codec supplies geometry, unit order, and all storage conversions.
/// This component constructor does not assemble the quantizer or spectral analysis
/// and does not select a conversational execution policy.
pub fn materialize_ear_encoder(
    prepared: PreparedEncoderBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarEncoder, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds codec-selected convolution and feed-forward shards to native execution.
/// Finite encoder latents remain replicated for the independently leased quantizer.
pub fn materialize_ear_encoder_parallel(
    prepared: PreparedEncoderBindings,
    parallel: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarEncoder, LayerwiseModelMetadata), BackendFailure> {
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
    prepared: PreparedEncoderBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarEncoder, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(Error::InvalidShape(
            "encoder parameter placement differs from the selected execution".into(),
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (config, prefix, definition, local_layout, layout, source, units) =
                prepared.into_parts();
            let kernel = match local_layout.as_ref() {
                Some(local) => {
                    EncoderKernel::<MlxNeuralBackend>::new_parallel(config, &prefix, local)
                }
                None => EncoderKernel::<MlxNeuralBackend>::new(config, &prefix),
            }
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(Error::from(error)))?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<StatelessComponentState>,
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
                        std::marker::PhantomData::<StatelessComponentState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok((
                MlxEarEncoder {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
