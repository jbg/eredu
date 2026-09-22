//! Ordinary physical controls derived from the shared equation populations.
use super::*;
mod diagnostic;
pub(super) use diagnostic::{OrdinaryCallFailure, OrdinaryTraceCallControls};
use safemlx::OrdinaryControlPopulation;

mod indexed;
mod materialization;
mod metal;
mod request;
pub(crate) use indexed::OrdinaryIndexedPrograms;

#[derive(Clone, Copy, Debug)]
pub(super) struct OrdinaryNeuralCalls {
    calls: OrdinaryCallControls,
    waits: usize,
}

/// Checked transports used while pricing the retained ordinary sources. The
/// caller charges the attempted query even when source qualification refuses.
struct OrdinaryQueryMetadata {
    bytes: Option<usize>,
}
impl OrdinaryQueryMetadata {
    fn new() -> Self {
        Self { bytes: Some(0) }
    }
    fn include(&mut self, bytes: Option<usize>) {
        self.bytes = self.bytes.and_then(|sum| sum.checked_add(bytes?));
    }
    fn charge(self, context: &WorkspaceContext) -> Result<(), Error> {
        context.charge_metadata(self.bytes.ok_or(WorkspaceMetadataError::Overflow)?)?;
        Ok(())
    }
}

/// Allocation-source allowances, distinct from tensor backing, caller metadata,
/// and the opaque overhead of platform events. This is prospective accounting,
/// not a physical-residency observation or an execution grant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OrdinaryNativeControls {
    pub(crate) observed_host_bytes: u64,
    pub(crate) control_allocations: usize,
    pub(crate) platform_events: usize,
}
impl OrdinaryNativeControls {
    pub(crate) fn include(&mut self, source: OrdinaryControlPopulation) -> Option<()> {
        self.observed_host_bytes = self
            .observed_host_bytes
            .checked_add(u64::try_from(source.observed_host_control_bytes()).ok()?)?;
        self.control_allocations = self
            .control_allocations
            .checked_add(source.control_allocations())?;
        self.platform_events = self.platform_events.checked_add(source.platform_events())?;
        Some(())
    }
    pub(crate) fn append(mut self, other: Self) -> Option<Self> {
        self.observed_host_bytes = self
            .observed_host_bytes
            .checked_add(other.observed_host_bytes)?;
        self.control_allocations = self
            .control_allocations
            .checked_add(other.control_allocations)?;
        self.platform_events = self.platform_events.checked_add(other.platform_events)?;
        Some(self)
    }
    pub(crate) fn repeat(self, count: usize) -> Option<Self> {
        Some(Self {
            observed_host_bytes: self
                .observed_host_bytes
                .checked_mul(u64::try_from(count).ok()?)?,
            control_allocations: self.control_allocations.checked_mul(count)?,
            platform_events: self.platform_events.checked_mul(count)?,
        })
    }
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            observed_host_bytes: self.observed_host_bytes.max(other.observed_host_bytes),
            control_allocations: self.control_allocations.max(other.control_allocations),
            platform_events: self.platform_events.max(other.platform_events),
        }
    }

    /// Every observed control allocation owns its canonical pending rows and
    /// accounting custody just like a tensor backing. Reserve that host
    /// bookkeeping beside the native allocation; publication is a conversion.
    pub(crate) fn host_allowance_with_ledger_metadata(
        self,
    ) -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let count =
            u64::try_from(self.control_allocations).map_err(|_| WorkingMemoryError::Overflow)?;
        crate::backend::managed_memory::ordinary_root_metadata_bytes()?
            .checked_mul(count)
            .and_then(|metadata| self.observed_host_bytes.checked_add(metadata))
            .ok_or(WorkingMemoryError::Overflow)
    }
}

impl ResidentCompletionRecipe {
    /// The selected workers' ordinary control source. Dispatch and retained
    /// stream identities stay with this recipe instead of being inferred.
    pub(crate) fn ordinary_controls(self) -> Option<OrdinaryNativeControls> {
        if self.dispatch?.cpu_model.is_some() {
            self.ordinary_cpu_controls()
        } else {
            self.ordinary_metal_controls(0)
        }
    }
    /// Shared CPU source controls for this equation and its existing completed
    /// value frontiers. Payloads and C wrapper metadata remain separate.
    pub(crate) fn ordinary_cpu_controls(self) -> Option<OrdinaryNativeControls> {
        self.ordinary_cpu_controls_with(None, std::iter::empty())
    }
    /// Additional indexed discovery/remap nodes join the same reachable DAG.
    /// Existing final/nested roots come from this retained recipe; the caller
    /// supplies only the additional actual (roots, occurrences) frontiers.
    pub(crate) fn ordinary_cpu_controls_with(
        self,
        extra: Option<OrdinaryCpuPopulation>,
        frontiers: impl IntoIterator<Item = (usize, usize)>,
    ) -> Option<OrdinaryNativeControls> {
        let mut population = OrdinaryCpuPopulation::from_completion(self)?;
        if let Some(extra) = extra {
            population = population.append(extra)?;
        }
        let nested = if self.nested_completions == 0 {
            None
        } else {
            Some((self.nested_traversal()?.roots(), self.nested_completions))
        };
        population.controls(
            std::iter::once((self.traversal.roots(), 1))
                .chain(nested)
                .chain(frontiers),
        )
    }
}

impl SpeculativeNumericalRecipe {
    pub(crate) fn ordinary_cpu_controls(self) -> Option<OrdinaryNativeControls> {
        self.completion.ordinary_cpu_controls()
    }
    pub(crate) fn ordinary_cpu_controls_with(
        self,
        extra: Option<OrdinaryCpuPopulation>,
        frontiers: impl IntoIterator<Item = (usize, usize)>,
    ) -> Option<OrdinaryNativeControls> {
        self.completion.ordinary_cpu_controls_with(extra, frontiers)
    }
}

impl ResidentNativeRecipe {
    /// Actual safe-call metadata is retained independently from native graph
    /// extents. Every selected producer must identify its caller population.
    pub(crate) fn ordinary_call_controls(&self) -> Option<OrdinaryCallControls> {
        if !self.ordinary_model_completion_bound {
            return None;
        }
        self.records
            .iter()
            .map(|row| self.ordinary_row_call_controls(row))
            .chain(self.sampling.rows().iter().map(|row| row.ordinary_calls))
            .try_fold(OrdinaryCallControls::default(), |sum, row| sum.append(row?))
    }
    pub(super) fn ordinary_row_call_controls(
        &self,
        row: &ResidentSpanRecipe,
    ) -> Option<OrdinaryCallControls> {
        row.ordinary_calls?.append(self.ordinary_neural?.calls)
    }

    /// The retained selected policy supplies actual group/final submission and
    /// maximum per-submission consumer populations, including an explicit zero.
    /// These are ordinary worker facts, not original arena installation rights.
    pub(crate) fn bind_ordinary_neural_calls(
        &mut self,
        geometry: InferenceGeometry,
        per_forward: usize,
        consumers: usize,
    ) -> Result<(), crate::backend::error::Error> {
        use crate::backend::{error::Error, nn::shared::MlxNeuralBackend};
        use eredu_runtime::working_memory::WorkingMemoryError;
        if self.plan.geometry() != geometry || self.ordinary_neural.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let waits = per_forward
            .checked_mul(consumers)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let calls = MlxNeuralBackend::ordinary_submission_call_controls(1)
            .and_then(|calls| calls.repeat(per_forward))
            .and_then(|calls| {
                calls.append(MlxNeuralBackend::ordinary_consumer_call_controls()?.repeat(waits)?)
            })
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        self.ordinary_neural = Some(OrdinaryNeuralCalls { calls, waits });
        Ok(())
    }

    /// Composition supplies its actual final-array worker source. Roots are
    /// handle slots from the same equation, never deduplicated storage owners.
    /// Prefill validation batches remain live across chunks of the same Work.
    pub(crate) fn bind_ordinary_model_completion_calls(
        &mut self,
        quote: impl Fn(usize, usize) -> Option<OrdinaryCallControls>,
    ) -> Result<(), crate::backend::error::Error> {
        use crate::backend::error::Error;
        use eredu_runtime::working_memory::WorkingMemoryError;
        if self.ordinary_model_completion_bound {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut prefill_validations = 0usize;
        for row in &mut self.records {
            let validations = if matches!(row.span, InferenceWorkspaceSpan::Prefill(_)) {
                prefill_validations = prefill_validations
                    .checked_add(row.validation_roots)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
                prefill_validations
            } else {
                row.validation_roots
            };
            let roots = row
                .traversal
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?
                .roots();
            let calls = quote(roots, validations)
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            row.ordinary_calls = row.ordinary_calls.and_then(|prior| prior.append(calls));
        }
        self.ordinary_model_completion_bound = true;
        Ok(())
    }

    /// Ordinary native allocations across the recorded request. Indexed child
    /// construction joins each enclosing lazy graph before evaluation is priced.
    /// Caller C shells, tensor backing and separately completed transfers remain
    /// independently attributed producers.
    pub(crate) fn ordinary_native_controls(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Option<OrdinaryNativeControls>, Error> {
        let mut query = OrdinaryQueryMetadata::new();
        query.include(Some(size_of::<(
            &Self,
            &WorkspaceContext,
            OrdinaryQueryMetadata,
            (&Self, &mut OrdinaryQueryMetadata),
            std::slice::Iter<'_, ResidentSpanRecipe>,
            Option<&ResidentSpanRecipe>,
            ResidentCompletionRecipe,
            Option<&OrdinaryAddressableProgram>,
            super::super::parallel::OrdinaryParallelControls,
            [Option<OrdinaryCpuPopulation>; 3],
            [OrdinaryNativeControls; 3],
            usize,
            bool,
            Option<OrdinaryNativeControls>,
            Result<Option<OrdinaryNativeControls>, Error>,
        )>()));
        let value = (|| {
            let mut controls = OrdinaryNativeControls::default();
            let waits = self.ordinary_neural?.waits;
            let cpu = self
                .records
                .first()
                .and_then(|row| row.dispatch)
                .and_then(|d| d.cpu_model)
                .is_some();
            if waits != 0 && cpu {
                let mut wait = OrdinaryNativeControls::default();
                wait.include(safemlx::OperationEvent::ordinary_cpu_wait_control_layout()?)?;
                controls = controls.append(wait.repeat(waits.checked_mul(self.records.len())?)?)?;
            }
            for row in &self.records {
                if row.addressable.is_some()
                    || row.first_missing_operation.is_some()
                    || row.unqualified_kernel_owner.is_some()
                {
                    return None;
                }
                let completion = self.completion_for_row(row)?;
                let program = row.ordinary_addressable.as_deref();
                let parallel = row.ordinary_parallel?;
                let indexed = match program {
                    Some(program) => Some(program.raw_cpu_population()?),
                    None => None,
                };
                let extra = match (indexed, parallel.cpu) {
                    (Some(indexed), Some(parallel)) => Some(indexed.append(parallel)?),
                    (indexed, parallel) => indexed.or(parallel),
                };
                let native = if completion.dispatch?.cpu_model.is_some() {
                    completion.ordinary_cpu_controls_with(
                        extra,
                        program
                            .into_iter()
                            .flat_map(|program| program.completion_frontiers())
                            .chain(std::iter::once((
                                parallel.completion_roots,
                                parallel.completions,
                            ))),
                    )?
                } else {
                    if program.is_some() {
                        return None;
                    }
                    completion.ordinary_metal_controls_with(
                        waits,
                        parallel.cpu,
                        std::iter::once((parallel.completion_roots, parallel.completions)),
                        &mut query,
                    )?
                };
                controls = controls.append(native)?.append(parallel.native)?;
            }
            for row in self.sampling.rows() {
                if row.first_missing_operation.is_some() {
                    return None;
                }
                match (row.preparation, row.completion) {
                    (_, Some(completion)) => {
                        let native = if completion.dispatch?.cpu_model.is_some() {
                            completion.ordinary_cpu_controls()?
                        } else {
                            completion.ordinary_metal_controls_with(
                                0,
                                None,
                                std::iter::empty(),
                                &mut query,
                            )?
                        };
                        controls = controls.append(native)?;
                    }
                    (Some(ResidentSamplingPreparation::Empty), None) => {}
                    (Some(ResidentSamplingPreparation::EagerKey(graph)), None) => {
                        controls.include(
                            safemlx::OperationEvent::ordinary_frontend_control_layout(
                                graph.primitives(),
                                graph.seeds(),
                                graph.maximum_rank(),
                                graph.maximum_operands().max(4),
                            )?,
                        )?;
                    }
                    (None, None) => return None,
                }
            }
            Some(controls)
        })();
        query.charge(context)?;
        Ok(value)
    }
}

impl ResidentRecipeRecorder {
    pub(super) fn ordinary_trace_call_controls(
        &self,
        report: &WorkspaceTraceReport,
        input_operation: Option<usize>,
        layerwise_completions: Option<usize>,
        addressable: Option<&OrdinaryAddressableProgram>,
        parallel: Option<&parallel::OriginalParallelInvocation>,
    ) -> Result<OrdinaryTraceCallControls, Error> {
        if self.mechanism.allocation().original_storage {
            return Ok(OrdinaryTraceCallControls::default());
        }
        let Some(handle_clones) = report.tensor_handle_clones else {
            return self.missing_ordinary_caller("tensor handle clone population", report, None);
        };
        if let Some(context) = &self.context {
            context.charge_metadata(size_of::<(
                &Self,
                &WorkspaceTraceReport,
                Option<usize>,
                Option<usize>,
                Option<&OrdinaryAddressableProgram>,
                Option<&parallel::OriginalParallelInvocation>,
                super::super::parallel::OrdinaryParallelControls,
                OrdinaryCallControls,
                Option<OrdinaryCallControls>,
                OrdinaryTraceCallControls,
                Result<OrdinaryTraceCallControls, Error>,
            )>())?;
        }
        let completions = report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::ValueCompletion))
            .count();
        if completions != 0 && layerwise_completions != Some(completions) {
            return self.missing_ordinary_caller_detail(
                format_args!("authenticated layerwise completion population; recorded completions {completions}, acquired source units {layerwise_completions:?}"),
                report,
                None,
            );
        }
        let mut controls = OrdinaryCallControls::default();
        let mut parallel_controls = super::super::parallel::OrdinaryParallelControls::default();
        let mut indexed = false;
        let mut paged_source = false;
        for (index, operation) in report.operations.iter().enumerate() {
            if matches!(
                operation.kind,
                WorkspaceOperationKind::CommunicationControl(_)
            ) {
                if !operation.inputs.is_empty() || !operation.outputs.is_empty() {
                    return self.missing_ordinary_caller(
                        "model communication control marker geometry",
                        report,
                        Some(index),
                    );
                }
                // Existing request controls own the actual agreement callback;
                // descriptive markers do not introduce another native caller.
                continue;
            }
            // The shared driver replaces this placeholder with its separately
            // admitted input. Its safe constructors belong to that input owner.
            if Some(index) == input_operation {
                continue;
            }
            if matches!(
                operation.kind,
                WorkspaceOperationKind::Collective(_)
                    | WorkspaceOperationKind::ExpertRegion(_)
                    | WorkspaceOperationKind::ExpertInactiveWave(_)
                    | WorkspaceOperationKind::ExpertProviderWave(_)
            ) {
                let selected = match self.cpu {
                    Some(cpu) => ResidentExecutionMechanisms::Cpu {
                        ordinary: self.mechanism,
                        cpu,
                    },
                    None => ResidentExecutionMechanisms::Metal(self.mechanism),
                };
                let calls = match parallel {
                    Some(source) => super::super::parallel::ordinary::controls(
                        source,
                        index,
                        operation.as_view(),
                        selected,
                    )?,
                    None => None,
                };
                let Some(calls) = calls else {
                    return self.missing_ordinary_caller(
                        "selected communication caller",
                        report,
                        Some(index),
                    );
                };
                indexed |= parallel
                    .and_then(|source| source.expert_occurrence(index))
                    .is_some_and(|source| source.ordinary_addressable.is_some());
                controls = controls
                    .append(calls.calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                parallel_controls = parallel_controls
                    .append(calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                continue;
            }
            if matches!(operation.kind, WorkspaceOperationKind::AddressableRegion(_)) {
                if addressable.is_none() {
                    return self.missing_ordinary_caller(
                        "addressable parameter caller",
                        report,
                        Some(index),
                    );
                }
                indexed = true;
                continue;
            }
            if matches!(operation.kind, WorkspaceOperationKind::ParameterPlaceholder)
                && self.layerwise_constructors.is_some()
            {
                // validate_layerwise_constructor_trace has authenticated the
                // exact scalar slots/dtypes against this selected unit source.
                let quoted = safemlx::ops::OrdinaryRecipeCall::Fill { rank: 0 }
                    .control_bytes()
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                let parts = [
                    quoted.metadata_bytes(),
                    size_of::<crate::backend::nn::module::PhysicalParam<safemlx::Array>>(),
                    size_of::<crate::backend::nn::module::PhysicalParam<Option<safemlx::Array>>>(),
                    size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
                    size_of::<(&[i32], safemlx::Dtype, &safemlx::Stream)>(),
                ];
                let bytes = parts
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                let mut observed = OrdinaryNativeControls::default();
                if let Some(source) = quoted.observed_controls() {
                    observed
                        .include(source)
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                }
                controls = controls
                    .append(OrdinaryCallControls {
                        metadata_bytes: u64::try_from(bytes)
                            .map_err(|_| WorkspaceMetadataError::Overflow)?,
                        observed,
                    })
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                continue;
            }
            if matches!(
                operation.kind,
                WorkspaceOperationKind::CachePublicationCompletion
                    | WorkspaceOperationKind::CacheScanCompletion
            ) {
                let roots = match operation.kind {
                    WorkspaceOperationKind::CachePublicationCompletion => 2,
                    WorkspaceOperationKind::CacheScanCompletion => 1,
                    _ => unreachable!(),
                };
                let calls = (operation.inputs.len() == roots && operation.outputs.is_empty())
                    .then(|| crate::backend::runtime::cache::residency::ordinary_cache_evaluation_call_controls(roots))
                    .flatten();
                let Some(calls) = calls else {
                    return self.missing_ordinary_caller(
                        "selected cache completion caller",
                        report,
                        Some(index),
                    );
                };
                controls = controls
                    .append(calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                // The numerical trace supplies no manager identity. Native
                // publication requires the independently retained source bank.
                paged_source = true;
                continue;
            }
            if matches!(
                operation.kind,
                WorkspaceOperationKind::CommunicationDependencies
            ) {
                // This recorder and its mechanism retain the same selected
                // parallel source. The typed event describes the driver's
                // actual local worker; it does not supply that source itself.
                let Some(calls) = parallel.and_then(|_| {
                    crate::backend::nn::shared::MlxNeuralBackend::ordinary_local_dependencies_call_controls(
                        operation.inputs.len(),
                    )
                }) else {
                    return self.missing_ordinary_caller(
                        "selected local communication dependency caller",
                        report,
                        Some(index),
                    );
                };
                controls = controls
                    .append(calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                continue;
            }
            if matches!(
                operation.kind,
                WorkspaceOperationKind::HostStoreFloating(..)
                    | WorkspaceOperationKind::HostLoadStoredFloating(..)
            ) {
                // The trace describes actual copies, but its numeric storage
                // slot is not canonical cache authority. The source bank must
                // retain and authenticate this complete transfer itinerary.
                paged_source = true;
            }
            if matches!(operation.kind, WorkspaceOperationKind::ValueRetention) {
                // The paged traversal moves these handles into its native
                // append, scan or transfer owner. This marker submits no work;
                // the independently retained source bank supplies custody.
                let calls = operation.outputs.is_empty().then(|| {
                    crate::backend::runtime::cache::kv::PagedKeyValueCache::ordinary_retention_control_bytes(operation.inputs.len())
                }).flatten();
                let Some(bytes) = calls else {
                    return self.missing_ordinary_caller(
                        "paged value retention",
                        report,
                        Some(index),
                    );
                };
                controls = controls
                    .metadata(bytes)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                paged_source = true;
                continue;
            }
            if matches!(operation.kind, WorkspaceOperationKind::ValueCompletion) {
                let Some(calls) =
                    crate::backend::nn::shared::MlxNeuralBackend::ordinary_completion_call_controls(
                        operation.inputs.len(),
                    )
                else {
                    return self.missing_ordinary_caller(
                        "layerwise completion wrapper",
                        report,
                        Some(index),
                    );
                };
                controls = controls
                    .append(calls)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                continue;
            }
            let Some(calls) = (match self.cpu {
                Some(cpu) => cpu.ordinary_call_controls(operation.as_view()),
                None => self.mechanism.ordinary_call_controls(operation.as_view()),
            })
            .map_err(MlxWorkspaceFactError::ordinary)?
            else {
                return self.missing_ordinary_caller(
                    "selected operation caller",
                    report,
                    Some(index),
                );
            };
            controls = controls
                .append(calls)
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        if indexed {
            let Some(calls) = addressable.and_then(OrdinaryAddressableProgram::ordinary_calls)
            else {
                return self.missing_ordinary_caller(
                    "indexed discovery and remap caller",
                    report,
                    None,
                );
            };
            controls = controls
                .append(calls)
                .ok_or(WorkspaceMetadataError::Overflow)?;
        } else if addressable.is_some() {
            return self.missing_ordinary_caller(
                "unmatched addressable caller source",
                report,
                None,
            );
        }
        let Some(clone_bytes) = safemlx::Array::ordinary_clone_control_bytes() else {
            return self.missing_ordinary_caller("native Array clone wrapper", report, None);
        };
        let clone_bytes = u64::try_from(clone_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_mul(u64::try_from(handle_clones).ok()?))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        controls.metadata_bytes = controls
            .metadata_bytes
            .checked_add(clone_bytes)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(OrdinaryTraceCallControls::complete(
            controls,
            parallel_controls,
            paged_source,
        ))
    }
}
