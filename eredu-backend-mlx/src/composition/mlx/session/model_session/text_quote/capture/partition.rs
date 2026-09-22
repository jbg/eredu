//! One issued text operation's native source for the shared receipt program.
use super::*;
mod evidence;
mod geometry;
use geometry::FrameGeometry;
pub(in crate::composition::mlx) use geometry::PartitionCaptureInvocation;
pub(in crate::composition::mlx) use host::{
    PartitionCaptureInvocationSource, PartitionCaptureRowMode,
};
pub(in crate::composition::mlx) mod host;
use crate::backend::runtime::distributed::topology::original_source::control::OriginalCaptureTransport;
use eredu_core::capture::{
    CaptureCandidateGeometry, CaptureHistogramGeometry, CapturePhase, CapturePrefillTransformPlan,
    CaptureSummaryGeometry, CaptureTensorGeometry, CaptureTokenScoreGeometry, CaptureTransform,
};
use eredu_core::checkpoint::TensorDtype;
use eredu_core::consensus::ConsensusTransport;
use eredu_nn::workspace::WorkspaceFloatingType;
use eredu_runtime::capture::{
    FundedCaptureSession,
    partition::{
        PartitionCaptureNativeEstimate, PartitionCaptureProducerSource,
        PartitionCaptureReceiptLimits, PartitionCaptureTransport, PreparedPartitionCaptureProgram,
        PreparedPartitionCaptureRow, PreparedPartitionCaptureRunIdentity,
    },
};
pub(in crate::composition::mlx) use host::evidence::{ModelEvidenceHostSource, ModelEvidenceHosts};
use std::mem::{size_of, size_of_val};

/// Payload-free loaded identities and exact output authority. Request/native
/// source custody stays last and survives both construction and callback errors.
pub(in crate::composition::mlx::session::model_session) type OriginalPartitionCaptureFrame =
    PartitionCaptureFrame<OriginalCaptureTransport>;

/// The native source supplies the real admitted transport and its metadata payer.
/// This interface grants no allocation, role, or communication authority.
pub(in crate::composition::mlx) trait NativePartitionCaptureTransport:
    PartitionCaptureTransport<Error = Error>
    + eredu_runtime::capture::partition::PartitionCaptureHookTransport
{
    fn prepare_capture_metadata(&self) -> Result<&eredu_nn::workspace::HostMetadataFunding, Error>;
    fn retain_capture_preparation_error(&self, cause: Error) -> Error;
}
impl<
    O: crate::backend::runtime::distributed::topology::original_source::control::CaptureSourceOwner,
> NativePartitionCaptureTransport for OriginalCaptureTransport<O>
{
    fn prepare_capture_metadata(&self) -> Result<&eredu_nn::workspace::HostMetadataFunding, Error> {
        OriginalCaptureTransport::prepare_capture_metadata(self)
    }
    fn retain_capture_preparation_error(&self, cause: Error) -> Error {
        OriginalCaptureTransport::retain_capture_preparation_error(self, cause)
    }
}

/// Lexical loan of an already funded capture run. Both model and text owners
/// lend their original source and the same once-bound run identity worker.
pub(in crate::composition::mlx) trait PartitionCaptureRunSource {
    fn capture_source(&self) -> &eredu_core::capture::SharedCapturePlan;
    fn prepare_run_identity(
        &mut self,
        artifact: &str,
        execution: &str,
        setup: eredu_runtime::CommunicationSessionIdentity,
        overlay: Option<&str>,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        PreparedPartitionCaptureRunIdentity,
        eredu_runtime::capture::partition::PartitionCaptureProgramError,
    >;
}
impl PartitionCaptureRunSource for FundedCaptureSession {
    fn capture_source(&self) -> &eredu_core::capture::SharedCapturePlan {
        self.source()
    }
    fn prepare_run_identity(
        &mut self,
        artifact: &str,
        execution: &str,
        setup: eredu_runtime::CommunicationSessionIdentity,
        overlay: Option<&str>,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        PreparedPartitionCaptureRunIdentity,
        eredu_runtime::capture::partition::PartitionCaptureProgramError,
    > {
        self.prepare_partition_run_identity(artifact, execution, setup, overlay, metadata)
    }
}
impl<R> PartitionCaptureRunSource for eredu_runtime::capture::FundedModelCaptureInvocation<R> {
    fn capture_source(&self) -> &eredu_core::capture::SharedCapturePlan {
        self.source().plan()
    }
    fn prepare_run_identity(
        &mut self,
        artifact: &str,
        execution: &str,
        setup: eredu_runtime::CommunicationSessionIdentity,
        overlay: Option<&str>,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        PreparedPartitionCaptureRunIdentity,
        eredu_runtime::capture::partition::PartitionCaptureProgramError,
    > {
        self.prepare_partition_run_identity(artifact, execution, setup, overlay, metadata)
    }
}

/// Consumed source rows and their original Host destinations for one real frame.
/// Transport custody remains last through preparation and callback failure.
pub(in crate::composition::mlx) struct PartitionCaptureFrame<T> {
    artifact: String,
    execution: String,
    overlay: Option<String>,
    source: eredu_core::SharedStorageIdentity,
    setup: eredu_runtime::CommunicationSessionIdentity,
    producers: Vec<Option<usize>>,
    hosts: Vec<Option<host::HostRow>>,
    evidence: Vec<Option<host::evidence::Operation>>,
    scalars: Vec<Option<WorkspaceFloatingType>>,
    prediction: u64,
    geometry: FrameGeometry,
    consumed: bool,
    source_metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
    interventions:
        Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>,
    model_evidence: Option<host::evidence::ModelEvidenceHosts>,
    transport: T,
}
impl TextExecutionQuote {
    pub(in crate::composition::mlx::session::model_session) fn prepare_partition_capture_frame(
        &self,
        session: &MlxModelSession,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        prediction: u64,
    ) -> Result<Option<OriginalPartitionCaptureFrame>, Error> {
        let Some(capture) = &self.capture else {
            return Ok(None);
        };
        self.request
            .validate_same_request(step.request())
            .map_err(memory)?;
        let (control, base) = match (&self.parallel_control, &session.payload.distributed) {
            (None, None) => return Ok(None),
            (Some(control), Some(base)) => (control, base),
            _ => {
                return Err(Error::InvalidOperation(
                    "capture request and loaded parallel owner differ",
                ));
            }
        };
        let transport = control.capture_transport(base, step)?;
        let metadata = transport.prepare_capture_metadata()?;
        let result = (|| {
            let parts = [size_of::<OriginalPartitionCaptureFrame>() * 2,
                size_of::<Result<OriginalPartitionCaptureFrame, Error>>(),
                size_of::<(&Self, &MlxModelSession, &eredu_runtime::working_memory::InferenceTextStep, u64)>(),
                size_of::<Vec<Option<WorkspaceFloatingType>>>() * 2, size_of::<Error>(), size_of::<Vec<Option<usize>>>() * 2,
                size_of::<Vec<Option<host::HostRow>>>()*2, size_of::<host::HostRow>(),
                evidence::control_bytes::<OriginalCaptureTransport>().ok_or_else(||memory(WorkingMemoryError::Overflow))?,
                size_of::<Option<eredu_architectures::component_partition::CompletePartitionCaptureSource>>(),
                size_of::<std::slice::Iter<'_, eredu_core::capture::CaptureSelection>>(),
                eredu_architectures::component_partition::ContiguousPartitionCaptureSource::control_bytes()
                    .ok_or_else(||memory(WorkingMemoryError::Overflow))?,
                eredu_architectures::component_partition::RoutedPartitionCaptureSource::control_bytes()
                    .ok_or_else(||memory(WorkingMemoryError::Overflow))?,
                size_of::<std::cell::RefMut<'_,Option<host::HostRows>>>(),
                size_of::<Result<Vec<Option<host::HostRow>>,Error>>(),
                eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes()
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?];
            metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?)?;
            // Loaded public capture discovery established these exact common
            // labels. This entry only borrows it: no lazy source/declaration work
            // occurs under the active original native submission.
            let loaded = session.partition_capture_source()
                .ok_or(Error::InvalidOperation("original parallel capture lacks loaded discovery source"))?;
            let (artifact, execution, setup) = loaded.source_labels();
            if setup != base.session_identity() || setup.participant_count() != transport.participant_count() {
                return Err(Error::InvalidOperation("capture discovery setup differs from retained transport"));
            }
            let admission = capture.capture_source.admission();
            let mut producers = metadata.metadata_vec(admission.plan().selections.len())?;
            for selection in &admission.plan().selections {
                if matches!(selection.transform,CaptureTransform::RoutedUnits) {
                    loaded.layouts().routed_capture_source(&selection.path)
                        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                    producers.push(None);
                    continue;
                }
                let producer=match loaded.layouts().complete_capture_source(&selection.path){
                    Some(source)=>Some(source.producer()),
                    None=>{
                        loaded.layouts().contiguous_capture_source(&selection.path)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                        None
                    }
                };
                if producer.is_some_and(|rank|rank>=setup.participant_count()){
                    return Err(Error::InvalidOperation("capture observation lacks this native hook source"));
                }
                producers.push(producer);
            }
            let hosts={
                let mut pending=capture.partition_hosts.borrow_mut();
                pending.as_mut().map(|hosts|hosts.take(&capture.capture_source,step.request().geometry(),prediction))
                    .transpose()?.unwrap_or_default()
            };
            let evidence= {
                let mut pending=capture.partition_evidence.borrow_mut();
                match (pending.as_mut(),capture.text_interventions.as_ref()) {
                    (Some(hosts),Some(rows))=>hosts.take(rows.source(),step.request().geometry(),prediction,metadata)?,
                    (None,_)=>Vec::new(),
                    _=>return Err(memory(WorkingMemoryError::IdentityMismatch)),
                }
            };
            let scalars = self.native_recipe.as_ref().ok_or(Error::PrefillScopeUnavailable)?
                .capture_scalars_for_step(step, admission.plan().selections.len(), metadata)?;
            let artifact = metadata.metadata_string(format_args!("{artifact}"))?;
            let execution = metadata.metadata_string(format_args!("{execution}"))?;
            let overlay = session.payload.parameter_state.active.as_deref()
                .map(|value| metadata.metadata_string(format_args!("{value}"))).transpose()?;
            Ok((artifact, execution, overlay, setup, producers, scalars, hosts, evidence))
        })().map_err(|cause| transport.retain_capture_preparation_error(cause))?;
        let (artifact, execution, overlay, setup, producers, scalars, hosts, evidence) = result;
        Ok(Some(OriginalPartitionCaptureFrame {
            artifact,
            execution,
            overlay,
            setup,
            producers,
            scalars,
            hosts,
            evidence,
            source: capture.source.clone(),
            prediction,
            geometry: FrameGeometry::Text(step.request().geometry()),
            consumed: false,
            source_metadata: None,
            interventions: capture.text_interventions.clone(),
            model_evidence: None,
            transport,
        }))
    }
}
impl<T: NativePartitionCaptureTransport> PartitionCaptureFrame<T>
where
    <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
{
    pub(super) fn program_control_bytes<C: PartitionCaptureRunSource>() -> Option<usize> {
        let parts = [
            size_of::<(&mut Self, &mut C, u64)>(),
            size_of::<
                Option<&crate::composition::mlx::session::intervention::PreparedModelInterventions>,
            >(),
            size_of::<PreparedPartitionCaptureProgram<'_, T>>() * 2,
            size_of::<Vec<Option<PartitionCaptureProducerSource>>>(),
            size_of::<PartitionCaptureProducerSource>(),
            size_of::<Vec<PreparedPartitionCaptureRow<'_>>>(),
            size_of::<PreparedPartitionCaptureRow<'_>>(),
            size_of::<Option<host::HostRow>>(),
            size_of::<std::slice::IterMut<'_, Option<host::HostRow>>>(),
            size_of::<
                Result<
                    (
                        host::BoundSource,
                        eredu_runtime::working_memory::PreparedPartitionFragmentHostFunding,
                    ),
                    Error,
                >,
            >(),
            size_of::<host::BoundSource>(),
            size_of::<usize>() * 2,
            size_of::<PreparedPartitionCaptureRunIdentity>(),
            size_of::<Result<Option<PreparedPartitionCaptureProgram<'_, T>>, Error>>(),
            size_of::<CaptureTensorGeometry<'_>>() * 2,
            size_of::<Option<TensorDtype>>(),
            size_of::<CaptureHistogramGeometry<'_>>() * 2,
            size_of::<CaptureCandidateGeometry<'_>>() * 2,
            size_of::<CaptureTokenScoreGeometry<'_>>() * 2,
            size_of::<Option<usize>>(),
            CaptureHistogramGeometry::preparation_control_bytes()?,
            size_of::<CaptureSummaryGeometry<'_>>() * 2,
            size_of::<CapturePrefillTransformPlan<'_>>() * 2,
            CaptureSummaryGeometry::preparation_control_bytes()?,
            size_of::<Error>(),
            CaptureTensorGeometry::preparation_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::composition::mlx) fn program<'a, C: PartitionCaptureRunSource>(
        &'a mut self,
        capture: &mut C,
        prediction: u64,
    ) -> Result<Option<PreparedPartitionCaptureProgram<'a, T>>, Error> {
        self.program_with_model(capture, prediction, None)
    }
    pub(in crate::composition::mlx) fn model_evidence_attachment_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(
                &mut Self,
                ModelEvidenceHostSource,
                Vec<(
                    usize,
                    [eredu_runtime::working_memory::PreparedPartitionFragmentHostFunding; 2],
                )>,
                &eredu_nn::workspace::HostMetadataFunding,
            )>(),
            size_of::<Result<(), Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::composition::mlx) fn install_model_evidence(
        &mut self,
        source: ModelEvidenceHostSource,
        hosts: Vec<(
            usize,
            [eredu_runtime::working_memory::PreparedPartitionFragmentHostFunding; 2],
        )>,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<(), Error> {
        metadata
            .reserve_metadata(
                Self::model_evidence_attachment_control_bytes()
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| {
                self.transport
                    .retain_capture_preparation_error(Error::WorkspacePlanning(cause))
            })?;
        if self.consumed || self.interventions.is_some() || self.model_evidence.is_some() {
            return Err(self
                .transport
                .retain_capture_preparation_error(memory(WorkingMemoryError::IdentityMismatch)));
        }
        let evidence = source
            .bind(hosts, metadata)
            .map_err(|cause| self.transport.retain_capture_preparation_error(cause))?;
        self.model_evidence = Some(evidence);
        Ok(())
    }
    pub(in crate::composition::mlx) fn program_with_model<'a, C: PartitionCaptureRunSource>(
        &'a mut self,
        capture: &mut C,
        prediction: u64,
        model: Option<&crate::composition::mlx::session::intervention::PreparedModelInterventions>,
    ) -> Result<Option<PreparedPartitionCaptureProgram<'a, T>>, Error> {
        let metadata = self.transport.prepare_capture_metadata()?;
        let result = (|| {
            metadata.reserve_metadata(Self::program_control_bytes::<C>()
                .ok_or_else(||memory(WorkingMemoryError::Overflow))?)?;
            if std::mem::replace(&mut self.consumed, true)
                || self.prediction != prediction || capture.capture_source().storage_identity() != &self.source {
                return Err(Error::InvalidOperation("capture operation source, coordinate or attempt differs"));
            }
            if self.interventions.is_some() && model.is_some() { return Err(memory(WorkingMemoryError::IdentityMismatch)); }
            if capture.capture_source().admission().is_empty() && self.interventions.is_none() && model.is_none() { return Ok(None); }
            let run = capture.prepare_run_identity(&self.artifact, &self.execution,
                self.setup, self.overlay.as_deref(), metadata)
                .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
            if !self.geometry.matches_prediction(prediction) { return Err(memory(WorkingMemoryError::IdentityMismatch)); }
            let phase = self.geometry.phase(prediction);
            let invocation = self.geometry.invocation().map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
            let admission = capture.capture_source().admission();
            let mut rows = metadata.metadata_vec(admission.plan().selections.len())?;
            for (index, selection) in admission.plan().selections.iter().enumerate() {
                if !matches!(selection.transform, CaptureTransform::FullTensor | CaptureTransform::Slice | CaptureTransform::Preview { .. } | CaptureTransform::Summary | CaptureTransform::Histogram { .. } | CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. } | CaptureTransform::RoutedUnits) {
                    return Err(Error::InvalidOperation("capture receipt has no complete native tensor producer"));
                }
                let active = selection.schedule.includes(phase, prediction);
                if !active || self.producers.get(index).copied().flatten().is_none(){
                    rows.push(None);continue;
                }
                let estimate = if active && matches!(selection.transform, CaptureTransform::Summary) {
                    if let Some(inference) = self.geometry.prefill(prediction) {
                        let plan = CapturePrefillTransformPlan::prepare(admission, index, inference)
                            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_prefill_summary(&plan)
                            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?
                    } else {
                        let geometry = match self.geometry.receipt_window() {
                            Some(interval) => CaptureSummaryGeometry::prepare_window(admission, index, phase, prediction, interval.physical, interval.window),
                            None => CaptureSummaryGeometry::prepare(admission, index, phase, prediction, invocation),
                        }
                            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_summary(&geometry)
                            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?
                    }
                } else if active && matches!(selection.transform,CaptureTransform::Histogram { .. }) {
                    if let Some(inference) = self.geometry.prefill(prediction) {
                        let plan=CapturePrefillTransformPlan::prepare(admission,index,inference)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_prefill_histogram(&plan)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?
                    } else {
                        let geometry=match self.geometry.receipt_window() {
                            Some(interval) => CaptureHistogramGeometry::prepare_window(admission, index, phase, prediction, interval.physical, interval.window),
                            None => CaptureHistogramGeometry::prepare(admission, index, phase, prediction, invocation),
                        }
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_histogram(&geometry)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?
                    }
                } else if active && matches!(selection.transform,CaptureTransform::TopCandidates{..}|CaptureTransform::TokenScores{..}) {
                    let terminal_rows=if let Some(inference) = self.geometry.prefill(prediction) {
                        Some(usize::try_from(if inference.output==eredu_core::OutputDemand::Sequence {
                            inference.input_positions.checked_sub(1).and_then(|n|n.checked_rem(inference.prefill_chunk_positions))
                                .and_then(|n|n.checked_add(1)).ok_or_else(||memory(WorkingMemoryError::Overflow))?
                        }else{1}).map_err(|_|memory(WorkingMemoryError::Overflow))?)
                    }else{None};
                    if matches!(selection.transform,CaptureTransform::TopCandidates{..}) {
                        let geometry=CaptureCandidateGeometry::prepare(admission,index,phase,prediction,invocation)
                            .and_then(|value|match terminal_rows {Some(rows)=>value.terminal_readout(rows),None=>Ok(value)})
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_candidates(&geometry)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?
                    }else{
                        let geometry=CaptureTokenScoreGeometry::prepare(admission,index,phase,prediction,invocation)
                            .and_then(|value|match terminal_rows {Some(rows)=>value.terminal_readout(rows),None=>Ok(value)})
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                        crate::composition::mlx::session::bounded_capture::estimate_token_scores(&geometry)
                            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?
                    }
                } else if active {
                    let geometry = match self.geometry.receipt_window() {
                            Some(interval) => CaptureTensorGeometry::prepare_window(admission, index, phase, prediction, interval.physical, interval.window),
                            None => CaptureTensorGeometry::prepare(admission, index, phase, prediction, invocation),
                        }
                        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
                    crate::composition::mlx::session::bounded_capture::estimate_tensor_geometry(&geometry)
                        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?
                } else { Default::default() };
                rows.push(Some(PartitionCaptureProducerSource { producer: self.producers.get(index).copied().flatten()
                    .ok_or(Error::InvalidOperation("capture producer row differs from its retained source"))?,
                    dtype: if active { *self.scalars.get(index)
                        .ok_or(Error::InvalidOperation("capture scalar row differs from its retained source"))? } else { None }
                        .map(|scalar| match scalar {
                        WorkspaceFloatingType::Float32 => TensorDtype::F32,
                        WorkspaceFloatingType::Float16 => TensorDtype::F16,
                        WorkspaceFloatingType::Bfloat16 => TensorDtype::Bf16,
                    }), estimate: PartitionCaptureNativeEstimate { capture: estimate, generated_creation_bytes: 0 } }));
            }
            let mut max_fragments=1;
            for host in self.hosts.iter().flatten() { max_fragments=max_fragments.max(host.fragment_limit()); }
            let limits = PartitionCaptureReceiptLimits { max_producers: self.setup.participant_count(),
                max_fragments, max_record_bytes: admission.plan().limits.per_step.encoded_bytes };
            Ok(Some((run, phase, invocation, rows, limits)))
        })().map_err(|cause| self.transport.retain_capture_preparation_error(cause))?;
        let Some((run, phase, invocation, rows, limits)) = result else {
            return Ok(None);
        };
        let mut selected = metadata.metadata_vec(rows.len())?;
        for (index, row) in rows.iter().enumerate() {
            selected.push(match row {
                Some(row) => PreparedPartitionCaptureRow::Complete(row),
                None if capture.capture_source().admission().plan().selections[index]
                    .schedule
                    .includes(phase, prediction) =>
                {
                    if matches!(
                        capture.capture_source().admission().plan().selections[index].transform,
                        CaptureTransform::RoutedUnits
                    ) {
                        PreparedPartitionCaptureRow::Routed
                    } else {
                        PreparedPartitionCaptureRow::Contiguous
                    }
                }
                None => PreparedPartitionCaptureRow::Inactive,
            });
        }
        let mut program = PreparedPartitionCaptureProgram::new_selected_for_run_at(
            &self.transport,
            capture.capture_source(),
            &run,
            phase,
            prediction,
            invocation,
            self.geometry.receipt_window(),
            &selected,
            limits,
            metadata,
        )
        .map_err(|cause| {
            self.transport
                .retain_capture_preparation_error(Error::Neural(metadata.metadata_source(cause)))
        })?;
        for (index, row) in selected.iter().enumerate() {
            if matches!(
                row,
                PreparedPartitionCaptureRow::Contiguous | PreparedPartitionCaptureRow::Routed
            ) {
                let host = self.hosts.get_mut(index).and_then(Option::take).ok_or(
                    Error::InvalidOperation("projected capture has no original Host source"),
                )?;
                let scalar = self.scalars.get(index).copied().flatten();
                let (source, host) = host
                    .bind(
                        capture.capture_source(),
                        self.transport.capture_rank(),
                        scalar,
                        metadata,
                    )
                    .map_err(|cause| self.transport.retain_capture_preparation_error(cause))?;
                program = match (row, source) {
                    (
                        PreparedPartitionCaptureRow::Contiguous,
                        host::BoundSource::Contiguous(source),
                    ) => program.with_contiguous_row(index, source, host),
                    (PreparedPartitionCaptureRow::Routed, host::BoundSource::Routed(source)) => {
                        program.with_routed_row(index, source, host)
                    }
                    _ => {
                        return Err(self.transport.retain_capture_preparation_error(
                            Error::InvalidOperation(
                                "capture source kind differs from its prepared row",
                            ),
                        ));
                    }
                }
                .map_err(|cause| {
                    self.transport
                        .retain_capture_preparation_error(Error::Neural(
                            metadata.metadata_source(cause),
                        ))
                })?;
            }
        }
        if let Some(interventions) = &self.interventions {
            let rows = interventions
                .partition_sources(phase, prediction, metadata)
                .map_err(|cause| {
                    self.transport
                        .retain_capture_preparation_error(Error::Neural(cause))
                })?;
            let companions = evidence::prepare(
                &self.transport,
                &self.artifact,
                &self.execution,
                self.setup,
                self.overlay.as_deref(),
                capture.capture_source(),
                &**interventions,
                phase,
                prediction,
                &mut self.evidence,
                metadata,
            )
            .map_err(|cause| self.transport.retain_capture_preparation_error(cause))?;
            program
                .set_intervention_sources_with_evidence(
                    interventions.source(),
                    rows.into_iter(),
                    companions.into_iter(),
                )
                .map_err(|cause| {
                    self.transport
                        .retain_capture_preparation_error(Error::Neural(
                            metadata.metadata_source(cause),
                        ))
                })?;
        }
        if let Some(model) = model {
            let rows = model.partition_sources(metadata).map_err(|cause| {
                self.transport
                    .retain_capture_preparation_error(Error::Neural(cause))
            })?;
            let evidence = self
                .model_evidence
                .as_mut()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
                .prepare(&self.transport, capture.capture_source(), model, metadata)
                .map_err(|cause| self.transport.retain_capture_preparation_error(cause))?;
            program
                .set_intervention_sources_with_evidence(
                    model.source(),
                    rows.into_iter(),
                    evidence.into_iter(),
                )
                .map_err(|cause| {
                    self.transport
                        .retain_capture_preparation_error(Error::Neural(
                            metadata.metadata_source(cause),
                        ))
                })?;
        } else if self.model_evidence.is_some() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(Some(program))
    }
}
