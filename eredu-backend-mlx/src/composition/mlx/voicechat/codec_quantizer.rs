//! MLX realization of codec-prepared, independently leased PRVQ codebooks.

use eredu_codec::{
    prvq::{Codebook, DistanceMetric, Geometry, Quantized},
    prvq_layered::{LayerwiseQuantizer, QuantizerExecution, QuantizerInput, QuantizerKernel},
    prvq_preparation::{validate_parallel_communication, PreparedQuantizerBindings},
    Error,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_nn::Tensor;
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, LocalModelLayout,
    ResidencyReport,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::{
            coordinator::{MlxLayeredAgreement, MlxReplicatedCoordinator},
            Group,
        },
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = Codebook<MlxTensor>;
enum Execution {
    Resident(LayerwiseQuantizer<MlxNeuralBackend, MlxResidentPolicy<Unit>>),
    Bounded(LayerwiseQuantizer<MlxNeuralBackend, MlxLayerwisePolicy<Unit>>),
}

/// Native PRVQ codebooks over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain codec-owned.
pub struct MlxPrvq {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxPrvq {
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

impl QuantizerExecution<MlxNeuralBackend> for MlxPrvq {
    fn validate_coordinated_execution(&self) -> Result<(), Error> {
        // Every parallel provider is constructed from a retained coordinator and
        // installs both acquisition and prepared-result failure gates.
        Ok(())
    }

    fn decode(&mut self, codes: &MlxTensor, context: &Stream) -> Result<MlxTensor, Error> {
        // Invalid local shapes must reach the guarded invocation header rather
        // than returning before another cohort participant enters agreement.
        let depth = codes.shape().get(2).copied().unwrap_or(0).max(0) as usize;
        self.decode_range(codes, 0..depth, context)
    }
    fn geometry(&self) -> Geometry {
        match &self.execution {
            Execution::Resident(model) => model.geometry(),
            Execution::Bounded(model) => model.geometry(),
        }
    }
    fn decode_range(
        &mut self,
        codes: &MlxTensor,
        range: std::ops::Range<usize>,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    QuantizerInput::Decode { codes, range },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(Error::from)
                    },
                )
                .map(|(output, _)| output),
            (Execution::Bounded(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    QuantizerInput::Decode { codes, range },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(Error::from)
                    },
                )
                .map(|(output, _)| output),
            (Execution::Resident(model), None) => model.decode_range(codes, range, context),
            (Execution::Bounded(model), None) => model.decode_range(codes, range, context),
        }
    }
    fn quantize_range(
        &mut self,
        latent: &MlxTensor,
        range: std::ops::Range<usize>,
        metric: DistanceMetric,
        context: &Stream,
    ) -> Result<Quantized<MlxTensor>, Error> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    QuantizerInput::Quantize {
                        latent,
                        range,
                        metric,
                    },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(Error::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_quantized()),
            (Execution::Bounded(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    QuantizerInput::Quantize {
                        latent,
                        range,
                        metric,
                    },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(Error::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_quantized()),
            (Execution::Resident(model), None) => {
                model.quantize_range(latent, range, metric, context)
            }
            (Execution::Bounded(model), None) => {
                model.quantize_range(latent, range, metric, context)
            }
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The codec supplies geometry, unit order, and all storage conversions.
/// This component constructor does not assemble a neural or spectral codec
/// and does not select a conversational execution policy.
pub fn materialize_prvq(
    prepared: PreparedQuantizerBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPrvq, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds codec-selected channel shards to native tensor parallel execution.
/// Codes, decoded audio latents, and quantized residuals remain replicated.
pub fn materialize_prvq_parallel(
    prepared: PreparedQuantizerBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPrvq, LayerwiseModelMetadata), BackendFailure> {
    validate_parallel_communication(
        prepared.geometry(),
        coordinator.manifest(),
        coordinator.group(),
    )
    .map_err(BackendFailure::from_error)?;
    let parallel = coordinator.parallel_group();
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedQuantizerBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPrvq, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(Error::InvalidShape(
            "quantizer parameter placement differs from the selected execution".into(),
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (geometry, prefix, local_layout, layout, source, units) = prepared.into_parts();
            let mut architecture = match local_layout.as_ref() {
                Some(local) => {
                    QuantizerKernel::<MlxNeuralBackend>::new_parallel(geometry, &prefix, local)
                }
                None => QuantizerKernel::<MlxNeuralBackend>::new(geometry, &prefix),
            }
            .and_then(QuantizerKernel::into_component)
            .map_err(BackendFailure::from_error)?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
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
                        std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseQuantizer::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseQuantizer::new(architecture, policy))
            };
            Ok((
                MlxPrvq {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
