//! Actual physical capture spans traced alongside the independent model equation.
use super::*;
use crate::backend::array_copy::CaptureNativePopulation;
use crate::composition::mlx::model::CaptureRecorder;
use crate::composition::mlx::session::{
    capture_workspace::CaptureWorkspaceObserver, intervention::PreparedModelInterventions,
};
use eredu_architectures::prepared_execution::InferenceEquationTraceObserver;
use eredu_core::{
    InferenceGeometry,
    capture::{CaptureInvocationShape, CaptureUsage},
    speculative::SpeculativePrefillSpan,
};
use eredu_nn::{
    Index, Tensor,
    workspace::{WorkspaceStoragePopulation, WorkspaceTensor, WorkspaceTraceReport},
};
use eredu_runtime::{
    ActivationObserver,
    capture::{OriginalSpeculativeCaptureInvocation, OriginalSpeculativeCaptureProspect},
    working_memory::{
        AutoregressiveCaptureFrameHostPlan, InferenceWorkspaceObserver, InferenceWorkspaceSpan,
        WorkingMemoryError,
    },
};
use std::cell::{Cell, RefCell};
mod partition;
use eredu_nn::workspace::WorkspaceFloatingType;
pub(super) use partition::PartitionQuote;

pub(super) struct Quote<'a> {
    pub source: &'a OriginalSpeculativeCaptureProspect,
    pub hosts: Vec<AutoregressiveCaptureFrameHostPlan<'a>>,
    pub populations: Vec<CaptureNativePopulation>,
    pub edits: Vec<RefCell<Option<PreparedModelInterventions>>>,
    pub partitions: Vec<RefCell<Option<PartitionQuote>>>,
    pub readout_construction: Vec<Cell<Option<safemlx::ResidentGraphLayout>>>,
}

pub(super) fn span_origin(
    geometry: InferenceGeometry,
    chunk: &eredu_runtime::prefill::PrefillChunk,
) -> Result<SpeculativePrefillSpan, Error> {
    let sequence = chunk
        .input
        .end
        .checked_sub(chunk.input.start)
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    Ok(SpeculativePrefillSpan {
        prompt_tokens: geometry.input_positions,
        input_start: chunk.input.start,
        input_end: chunk.input.end,
        position: chunk.position,
        hidden_start: chunk.input.start,
        token_start: chunk.input.start,
        sequence,
        seed_start: chunk.position,
    })
}
pub(super) fn descriptor<'a>(
    source: &'a OriginalSpeculativeCaptureProspect,
    invocation: AutoregressiveInvocation,
    geometry: InferenceGeometry,
    span: &InferenceWorkspaceSpan,
    ordinal: usize,
) -> Result<OriginalSpeculativeCaptureInvocation<'a>, Error> {
    if invocation.execution_pass() == eredu_runtime::ExpertPass::Prefill {
        let InferenceWorkspaceSpan::Prefill(chunk) = span else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        source
            .prefill_invocation(
                span_origin(geometry, chunk)?,
                u64::try_from(ordinal)
                    .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(eredu_nn::Error::backend_retained_source(cause)))
    } else if ordinal == 0 {
        Ok(source.invocation())
    } else {
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    }
}

pub(super) struct Observer<'source, 'trace> {
    source: &'source OriginalSpeculativeCaptureProspect,
    invocation: AutoregressiveInvocation,
    geometry: InferenceGeometry,
    sources: &'trace AutoregressiveSourcePair,
    context: WorkspaceContext,
    transfers: &'trace Cell<CaptureNativePopulation>,
    edits: &'trace [RefCell<Option<PreparedModelInterventions>>],
    hosts: &'trace RefCell<Vec<AutoregressiveCaptureFrameHostPlan<'source>>>,
    partition_scalars: &'trace [Vec<Cell<Option<WorkspaceFloatingType>>>],
    partitions: &'trace [RefCell<Option<PartitionQuote>>],
    readout_operations: &'trace [Cell<Option<usize>>],
    inner: Option<CaptureWorkspaceObserver<'trace>>,
    current: Option<eredu_runtime::prefill::PrefillChunk>,
    physical: Option<InferenceGeometry>,
    last_row: Option<WorkspaceTensor>,
    ordinal: usize,
}
impl<'source: 'trace, 'trace> Observer<'source, 'trace> {
    pub(super) fn new(
        source: &'source OriginalSpeculativeCaptureProspect,
        invocation: AutoregressiveInvocation,
        geometry: InferenceGeometry,
        sources: &'trace AutoregressiveSourcePair,
        context: &WorkspaceContext,
        transfers: &'trace Cell<CaptureNativePopulation>,
        edits: &'trace [RefCell<Option<PreparedModelInterventions>>],
        hosts: &'trace RefCell<Vec<AutoregressiveCaptureFrameHostPlan<'source>>>,
        partition_scalars: &'trace [Vec<Cell<Option<WorkspaceFloatingType>>>],
        partitions: &'trace [RefCell<Option<PartitionQuote>>],
        readout_operations: &'trace [Cell<Option<usize>>],
    ) -> Result<Self, Error> {
        context
            .charge_metadata(size_of::<(
                Self,
                Option<Self>,
                Result<Self, Error>,
                InferenceGeometry,
                Option<eredu_runtime::prefill::PrefillChunk>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        Ok(Self {
            source,
            invocation,
            geometry,
            sources,
            context: context.clone(),
            transfers,
            edits,
            hosts,
            partition_scalars,
            partitions,
            readout_operations,
            inner: None,
            current: None,
            physical: None,
            last_row: None,
            ordinal: 0,
        })
    }
    fn inner(&mut self) -> Result<&mut CaptureWorkspaceObserver<'trace>, eredu_nn::Error> {
        self.inner.as_mut().ok_or_else(|| {
            self.context
                .metadata_source(WorkingMemoryError::IdentityMismatch)
        })
    }
    fn prepare_span(&mut self, span: &InferenceWorkspaceSpan) -> Result<(), Error> {
        let InferenceWorkspaceSpan::Prefill(chunk) = span else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        self.context.charge_metadata(eredu_runtime::working_memory::OriginalSpeculativeRequest::embedded_capture_inspection_control_bytes().ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)
            .map_err(|cause|Error::Neural(cause.into()))?;
        let descriptor = descriptor(
            self.source,
            self.invocation,
            self.geometry,
            span,
            self.ordinal,
        )?;
        let width = chunk
            .input
            .end
            .checked_sub(chunk.input.start)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let physical = InferenceGeometry {
            cached_positions: chunk.position,
            input_positions: width,
            prefill_chunk_positions: width,
            max_output_tokens: 0,
            output: chunk.output,
            ..self.geometry
        };
        let shape = CaptureInvocationShape {
            batch: physical.batch_size,
            sequence: width,
            context: descriptor
                .source()
                .plan()
                .admission()
                .invocation_bounds()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
                .max_context
                .map(|_| {
                    chunk
                        .position
                        .checked_add(width)
                        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
                })
                .transpose()?,
        };
        descriptor
            .source()
            .validate_pool(self.sources.numerical_sources().pool())
            .map_err(|cause| self.sources.retain_startup_error(cause))?;
        let lineage = descriptor.lineage().ok_or_else(|| {
            self.sources
                .retain_startup_error(WorkingMemoryError::IdentityMismatch)
        })?;
        let inherited = self
            .sources
            .request()
            .inspect_model_capture_lineage_usage(descriptor.source(), lineage)
            .map_err(|cause| self.sources.retain_startup_error(cause))?;
        let (inner, host) = CaptureWorkspaceObserver::with_invocation_window(
            descriptor.source().plan(),
            physical,
            &self.context,
            self.transfers,
            descriptor.capture_phase(),
            descriptor.origin().prediction as u64,
            shape,
            descriptor.selected(),
            descriptor
                .window()
                .map_err(|cause| self.sources.retain_startup_error(cause))?,
        )?;
        let inner = inner.with_inherited_usage(inherited)?.with_envelope_usage(
            descriptor
                .envelope_usage()
                .map_err(|cause| self.sources.retain_startup_error(cause))?,
        )?;
        let host = host
            .with_skip_reasons(descriptor.skip_reasons())
            .map_err(|cause| self.sources.retain_startup_error(cause))?;
        let host = AutoregressiveCaptureFrameHostPlan::prepare(
            descriptor.source(),
            host,
            self.invocation,
            self.geometry,
            span,
            descriptor.origin(),
        )
        .map_err(|cause| self.sources.retain_startup_error(cause))?;
        let (inner, host) = match descriptor.interventions() {
            None => (inner, host),
            Some((source, selected)) => {
                source
                    .validate_pool(self.sources.numerical_sources().pool())
                    .map_err(|cause| self.sources.retain_startup_error(cause))?;
                let slot = self
                    .edits
                    .get(self.ordinal)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
                *slot
                    .try_borrow_mut()
                    .map_err(|cause| self.sources.retain_startup_error(cause))? = Some(
                    PreparedModelInterventions::prepare_with_evidence(
                        source,
                        selected,
                        descriptor.capture_phase(),
                        descriptor.origin().prediction as u64,
                        shape,
                        descriptor
                            .window()
                            .map_err(|cause| self.sources.retain_startup_error(cause))?,
                        descriptor.intervention_evidence_skips(),
                        &self.context,
                    )
                    .map_err(|cause| self.sources.retain_startup_error(cause))?,
                );
                (
                    inner
                        .with_model_interventions(slot)
                        .map_err(|cause| self.sources.retain_startup_error(cause))?,
                    host.with_intervention_evidence(
                        source,
                        selected,
                        descriptor.intervention_evidence_skips(),
                    )
                    .map_err(|cause| self.sources.retain_startup_error(cause))?,
                )
            }
        };
        let (inner, host) = match self.sources.partition_source(self.invocation.source()) {
            None => (inner, host),
            Some(native) => {
                if descriptor.interventions().is_some() {
                    let mut edits = self
                        .edits
                        .get(self.ordinal)
                        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
                        .try_borrow_mut()
                        .map_err(|cause| self.sources.retain_startup_error(cause))?;
                    edits
                        .as_mut()
                        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
                        .bind_partition(
                            native.layouts(),
                            native.rank(),
                            native.control().capture_retained_source()?,
                            &self.context,
                        )
                        .map_err(Error::Neural)?;
                }
                let cells = self
                    .partition_scalars
                    .get(self.ordinal)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
                let mut inner = inner;
                inner.bind_partition_sources(cells, (native.layouts(), native.rank()))?;
                let coordinate = crate::composition::mlx::session::bounded_capture::partition::PartitionCaptureInvocation {
                    phase: descriptor.capture_phase(), prediction: descriptor.origin().prediction as u64,
                    physical: shape, window: descriptor.window().map_err(|cause|self.sources.retain_startup_error(cause))?,
                };
                let (source, fragments) = native.prepare_invocation(
                    descriptor.source().plan(),
                    coordinate,
                    self.sources.metadata_funding(),
                )?;
                let quote =
                    PartitionQuote::prepare(source, native, descriptor, &fragments, &self.context)?;
                let slot = self
                    .partitions
                    .get(self.ordinal)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
                let mut slot = slot
                    .try_borrow_mut()
                    .map_err(|cause| self.sources.retain_startup_error(cause))?;
                if slot.is_some() {
                    return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                }
                *slot = Some(quote);
                (
                    inner,
                    host.with_partition_fragments(fragments)
                        .map_err(|cause| self.sources.retain_startup_error(cause))?,
                )
            }
        };
        let host = match descriptor.lineage() {
            Some(lineage) => host
                .with_lineage(lineage)
                .map_err(|cause| self.sources.retain_startup_error(cause))?,
            None => host,
        };
        let mut hosts = self
            .hosts
            .try_borrow_mut()
            .map_err(|cause| self.sources.retain_startup_error(cause))?;
        if hosts.len() != self.ordinal || hosts.len() == hosts.capacity() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        hosts.push(host.with_quoted_usage(inherited));
        drop(hosts);
        self.inner = Some(inner);
        self.current = Some(chunk.clone());
        self.physical = Some(physical);
        self.last_row = None;
        Ok(())
    }
    fn trace_prefill_readout(&mut self, value: &WorkspaceTensor) -> Result<(), eredu_nn::Error> {
        self.context.charge_metadata(size_of::<(
            &mut Self,
            &WorkspaceTensor,
            Result<(), eredu_nn::Error>,
        )>())?;
        if self.geometry.output == eredu_core::OutputDemand::Sequence
            && self.invocation.pass() == AutoregressivePass::TargetPrefill
            && self
                .current
                .as_ref()
                .is_some_and(|span| span.input.end == self.geometry.input_positions)
        {
            let shape = value.shape();
            if shape.len() != 3 || shape[1] <= 0 {
                return Err(self
                    .context
                    .metadata_source(WorkingMemoryError::IdentityMismatch));
            }
            let slot = self.readout_operations.get(self.ordinal).ok_or_else(|| {
                self.context
                    .metadata_source(WorkingMemoryError::IdentityMismatch)
            })?;
            if slot.get().is_some() {
                return Err(self
                    .context
                    .metadata_source(WorkingMemoryError::IdentityMismatch));
            }
            let operation = self.context.operation_count();
            self.last_row = Some(value.index(
                &[Index::Full, Index::At(shape[1] - 1), Index::Full],
                &self.context,
            )?);
            if self.context.operation_count().checked_sub(operation) != Some(1) {
                return Err(self
                    .context
                    .metadata_source(WorkingMemoryError::IdentityMismatch));
            }
            slot.set(Some(operation));
        }
        Ok(())
    }
}
impl ActivationObserver<WorkspaceTensor, eredu_nn::Error> for Observer<'_, '_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        self.source
            .invocation()
            .requires_sequence_readout()
            .unwrap_or(true)
    }
    fn observe(&mut self, path: &str, value: &WorkspaceTensor) -> Result<(), eredu_nn::Error> {
        self.inner()?.observe(path, value)?;
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            self.trace_prefill_readout(value)?;
        }
        Ok(())
    }
    fn observe_replica(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?.observe_replica(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
    ) -> Result<Option<WorkspaceTensor>, eredu_nn::Error> {
        self.inner()?.intervene(path, value)
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<WorkspaceTensor>>, eredu_nn::Error>
    {
        self.inner()?.routed_unit_observer(path)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, eredu_nn::Error>
    {
        self.inner()?.routing_control(path, rows)
    }
    fn routing_unmodified_interest(&self, path: &str) -> eredu_runtime::RoutingUnmodifiedInterest {
        self.inner
            .as_ref()
            .map_or(eredu_runtime::RoutingUnmodifiedInterest::None, |inner| {
                inner.routing_unmodified_interest(path)
            })
    }
    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: eredu_runtime::RoutingDecision<'_, WorkspaceTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?.routing_unmodified(path, effective)
    }
    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, WorkspaceTensor>>,
        effective: eredu_runtime::RoutingDecision<'_, WorkspaceTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?.routing_applied(path, original, effective)
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?
            .observe_generated_retained(path, prototype, source, factory)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn FnMut() -> Result<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?
            .observe_generated(path, prototype, source, factory)
    }
}
impl InferenceWorkspaceObserver for Observer<'_, '_> {
    fn prefill_invocation_span(&self) -> Option<&eredu_runtime::prefill::PrefillChunk> {
        self.current.as_ref()
    }
    fn begin_span(
        &mut self,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error> {
        if self.inner.is_some()
            || geometry != self.geometry
            || !self.context.shares_trace(context)
            || prediction != self.source.invocation().origin().prediction as u64
        {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        }
        self.prepare_span(span)
            .map_err(|cause| context.metadata_source(cause))?;
        let physical = self
            .physical
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        let InferenceWorkspaceSpan::Prefill(chunk) = span else {
            unreachable!("validated span");
        };
        let local = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
            input: 0..physical.input_positions,
            position: chunk.position,
            output: chunk.output,
        });
        self.inner()?
            .begin_span(physical, &local, prediction, context)
    }
    fn observe_remote_output(
        &mut self,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        self.inner()?.observe_remote_output(value, context)?;
        // Publication receivers construct the same final-row Index from their
        // actual received output; their local replica hook is not that source.
        self.trace_prefill_readout(value)
    }
    fn visit_retained(&self, visit: &mut dyn FnMut(&WorkspaceTensor)) {
        if let Some(inner) = &self.inner {
            inner.visit_retained(visit);
        }
        if let Some(row) = &self.last_row {
            visit(row);
        }
    }
    fn end_span(
        &mut self,
        span: &InferenceWorkspaceSpan,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        let InferenceWorkspaceSpan::Prefill(chunk) = span else {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        };
        if self.current.as_ref() != Some(chunk) {
            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
        }
        let physical = self
            .physical
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        let local = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
            input: 0..physical.input_positions,
            position: chunk.position,
            output: chunk.output,
        });
        self.inner()?.end_span(&local, context)?;
        self.inner = None;
        let slot = self
            .partitions
            .get(self.ordinal)
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
        if let Some(partition) = slot
            .try_borrow_mut()
            .map_err(|cause| context.metadata_source(cause))?
            .as_mut()
        {
            let cells = self
                .partition_scalars
                .get(self.ordinal)
                .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
            let edits = self
                .edits
                .get(self.ordinal)
                .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?
                .try_borrow()
                .map_err(|cause| context.metadata_source(cause))?;
            let native = self
                .sources
                .partition_source(self.invocation.source())
                .ok_or_else(|| context.metadata_source(WorkingMemoryError::IdentityMismatch))?;
            let (bytes, evidence) = partition
                .finish(cells, context, native, edits.as_ref())
                .map_err(|cause| context.metadata_source(cause))?;
            let mut hosts = self
                .hosts
                .try_borrow_mut()
                .map_err(|cause| context.metadata_source(cause))?;
            if hosts.len() != self.ordinal + 1 {
                return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
            }
            let host = hosts.pop().expect("current quoted frame");
            let descriptor = descriptor(
                self.source,
                self.invocation,
                self.geometry,
                span,
                self.ordinal,
            )
            .map_err(|cause| context.metadata_source(cause))?;
            let host = match descriptor.interventions() {
                Some((source, _)) => host
                    .with_partition_evidence_fragments(source, evidence)
                    .map_err(|cause| context.metadata_source(cause))?,
                None if evidence.is_empty() => host,
                _ => return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch)),
            };
            hosts.push(
                host.with_transport_metadata(bytes)
                    .map_err(|cause| context.metadata_source(cause))?,
            );
        }
        self.current = None;
        self.physical = None;
        self.last_row = None;
        self.ordinal = self
            .ordinal
            .checked_add(1)
            .ok_or_else(|| context.metadata_source(WorkingMemoryError::Overflow))?;
        Ok(())
    }
}

pub(super) struct Trace<'a> {
    pub recorder: &'a mut dyn CaptureRecorder,
    pub scalars: &'a [Vec<Cell<Option<WorkspaceFloatingType>>>],
    pub transfers: &'a Cell<CaptureNativePopulation>,
    pub populations: &'a mut Vec<CaptureNativePopulation>,
    pub context: &'a WorkspaceContext,
    pub mechanism: crate::backend::nn::workspace::ResidentExecutionMechanisms,
    pub readout_operations: &'a [Cell<Option<usize>>],
    pub readout_construction: &'a [Cell<Option<safemlx::ResidentGraphLayout>>],
}
impl Trace<'_> {
    fn readout(&self, report: &WorkspaceTraceReport) -> Result<(), eredu_nn::Error> {
        let invalid = || {
            self.context
                .metadata_source(WorkingMemoryError::IdentityMismatch)
        };
        let row = self.populations.len();
        let operation = self.readout_operations.get(row).ok_or_else(invalid)?.get();
        let output = self.readout_construction.get(row).ok_or_else(invalid)?;
        if output.get().is_some() {
            return Err(invalid());
        }
        if let Some(operation) = operation {
            output.set(Some(
                crate::backend::nn::workspace::AutoregressiveReadoutRecipe::index_construction(
                    report.operations.get(operation).ok_or_else(invalid)?,
                    self.mechanism,
                    self.context,
                )?,
            ));
        }
        Ok(())
    }
    fn population(&mut self) -> Result<(), eredu_nn::Error> {
        let population = self.transfers.get();
        self.recorder
            .capture_scalars(self.scalars.get(self.populations.len()).map(Vec::as_slice))?;
        self.recorder.capture_population(population)?;
        if self.populations.len() == self.populations.capacity() {
            return Err(self
                .context
                .metadata_source(WorkingMemoryError::IdentityMismatch));
        }
        self.populations.push(population);
        Ok(())
    }
}
impl InferenceEquationTraceObserver for Trace<'_> {
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained: usize,
        outputs: usize,
        input: usize,
    ) -> Result<(), eredu_nn::Error> {
        self.readout(report)?;
        self.recorder
            .observe(span, report, retained, outputs, input)?;
        self.population()
    }
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained: usize,
        outputs: usize,
        input: usize,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), eredu_nn::Error> {
        self.readout(report)?;
        self.recorder
            .observe_with_storage(span, report, retained, outputs, input, output)?;
        self.population()
    }
    fn observe_prepared_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained: usize,
        outputs: usize,
        input: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), eredu_nn::Error> {
        self.readout(report)?;
        self.recorder
            .observe_prepared_with_storage(span, report, retained, outputs, input, output)?;
        self.population()
    }
}
