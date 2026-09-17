//! The same logical progression and physical Selection runner as funded chunks.
use super::*;
use eredu_nn::{
    GeneratedTensorProgram, GeneratedTensorSourceRole, RetainedGeneratedTensorFactory,
    workspace::WorkspaceDtype,
};
use eredu_runtime::{
    capture::{CapturePrefillHookDecision as Decision, CapturePrefillObservationPolicy},
    prefill::PrefillChunk,
};

#[derive(Debug, thiserror::Error)]
#[error("cold capture selection {selection}, rank {rank:?}, chunk {chunk}: {cause}")]
struct ColdCaptureProgressFailure {
    selection: usize,
    rank: Option<usize>,
    chunk: u64,
    #[source]
    cause: eredu_runtime::capture::CapturePrefillProgressError,
}

impl CaptureWorkspaceObserver<'_> {
    pub(super) fn begin_fragment_span(
        &mut self,
        chunk: &PrefillChunk,
        prediction: u64,
    ) -> Result<bool> {
        let metadata = Metadata::new(&self.context)?;
        if prediction != 0
            || self.prefill_complete
            || self.chunk.is_some()
            || chunk.input.start != self.prefill_end
            || chunk.input.end
                != self
                    .prefill_end
                    .saturating_add(self.geometry.prefill_chunk_positions)
                    .min(self.geometry.input_positions)
            || self.geometry.cached_positions.checked_add(self.prefill_end) != Some(chunk.position)
            || chunk.output
                != self
                    .geometry
                    .output
                    .for_chunk(chunk.input.end == self.geometry.input_positions)
            || !self.roots.is_empty()
        {
            return Err(metadata.coordinate());
        }
        if chunk.input.start == 0 {
            self.begin_prediction(CapturePhase::Prefill, 0)?;
        }
        if self.active != Some(0) {
            return Err(metadata.coordinate());
        }
        self.chunk = Some(chunk.clone());
        if let Some(edits)=self.text_interventions {
            let mut edits=edits.try_borrow_mut().map_err(|cause|metadata.error(cause))?;
            let edits=edits.as_mut().ok_or_else(||metadata.coordinate())?;
            if let Some(communication)=self.partition_intervention_source {
                let (layouts,rank)=self.placement.ok_or_else(||metadata.coordinate())?;
                edits.begin_partition_prefill(self.geometry,chunk,layouts,rank,communication,&self.context)?;
            } else {edits.begin_prefill(self.geometry,chunk,&self.context)?;}
        }
        Ok(true)
    }
    pub(super) fn end_fragment_span(&mut self, chunk: &PrefillChunk) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        if self.chunk.as_ref() != Some(chunk) || self.active != Some(0) {
            return Err(metadata.coordinate());
        }
        let policy = CapturePrefillObservationPolicy::from_bound(
            self.bound.ok_or_else(|| metadata.coordinate())?,
        )
        .map_err(|cause| metadata.error(cause))?;
        let k = chunk.input.start / self.geometry.prefill_chunk_positions;
        // A rank absent from this invocation has no tensor to inspect. Follow
        // the same admitted logical prefill progression for its remote complete
        // producer while leaving the physical scalar witness absent. The runtime
        // source vote must resolve that witness before any real observation.
        if let Some((layouts, rank)) = self.placement {
            self.context.charge_metadata(std::mem::size_of::<(
                &eredu_architectures::component_partition::ComponentPartitionLayouts, usize,
                Option<&eredu_architectures::component_partition::PartitionedObservation>,
                Decision, CaptureUsage, eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
                eredu_core::capture::CapturePrefillFragment<'_, '_>,
                CapturePrefillObservationPolicy<'_>,
                eredu_runtime::capture::CapturePrefillObservationRow<'_>,
                Option<eredu_core::capture::CaptureSkipReason>,
            )>())?;
            let layout = layouts.rank(rank).ok_or_else(|| metadata.coordinate())?;
            for (index, progress) in self.progress.iter_mut().enumerate() {
                let selection = &self.source.admission().plan().selections[index];
                if matches!(self.source.admission().points()[index].value_type,
                    eredu_core::ObservationValueType::RoutedUnits { .. }) {
                    if super::routed::partition::local(self.source,(layouts,rank),index,&self.context)?.is_some() {
                        continue;
                    }
                    let row=policy.row(index).map_err(|e|metadata.error(e))?;
                    let Some(plan)=row.routed_plan() else{continue};
                    let decision=row.begin_routed_batch(progress,chunk,&selection.path).map_err(|e|metadata.error(e))?;
                    if decision==Decision::Ignore {continue;}
                    if decision==Decision::First {
                        let usage=super::super::bounded_capture::estimate_routed_geometry(plan.geometry()).map_err(|e|metadata.error(e))?;
                        if row.reserve_first(progress,&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){continue;}
                    }
                    row.finish_routed_hook(progress,&plan.fragment(k).map_err(|e|metadata.error(e))?)
                        .map_err(|e|metadata.error(e))?;
                    continue;
                }
                let point = layout.observation(&selection.path).ok_or_else(|| metadata.coordinate())?;
                if point.coordinates().is_some() { continue; }
                if point.exports() { return Err(metadata.coordinate()); }
                // Final publication already calls observe_remote_output after
                // the selected publisher on ranks without a local readout.
                // Preserve that actual hook's progression and missing-hook
                // validation instead of synthesizing a duplicate fragment.
                if point.site() == eredu_runtime::inspection::ObservationHookSite::Publication {
                    continue;
                }
                let row = policy.row(index).map_err(|cause| metadata.error(cause))?;
                let decision = row.begin_hook(progress, chunk, &selection.path)
                    .map_err(|cause| metadata.error(cause))?;
                if decision == Decision::Ignore { continue; }
                if row.terminal() {
                    let usage=if row.candidate().is_some() {
                        let geometry=row.candidate_for_chunk(chunk).map_err(|cause|metadata.error(cause))?;
                        super::super::bounded_capture::estimate_candidates(&geometry)
                    } else {
                        let geometry=row.token_scores_for_chunk(chunk).map_err(|cause|metadata.error(cause))?;
                        super::super::bounded_capture::estimate_token_scores(&geometry)
                    }.map_err(|cause|metadata.error(cause))?;
                    if row.reserve_first(progress,&mut self.ledger,usage).map_err(|cause|metadata.error(cause))?.is_some(){continue;}
                    if row.candidate().is_some(){row.finish_candidate_hook(progress,chunk)}
                    else{row.finish_token_scores_hook(progress,chunk)}.map_err(|cause|metadata.error(cause))?;
                    continue;
                }
                if let Some(plan) = row.transform_plan()
                    .filter(|plan| matches!(plan.selection().transform, CaptureTransform::Summary | CaptureTransform::Histogram { .. })) {
                    let fragment = plan.fragment(k).map_err(|cause| metadata.error(cause))?;
                    if decision == Decision::First {
                        let usage=match plan.selection().transform {
                            CaptureTransform::Histogram { .. }=>super::super::bounded_capture::estimate_prefill_histogram(plan),
                            _=>super::super::bounded_capture::estimate_prefill_summary(plan),
                        }.map_err(|cause|metadata.error(cause))?;
                        if row.reserve_first(progress, &mut self.ledger, usage)
                            .map_err(|cause| metadata.error(cause))?.is_some() { continue; }
                    }
                    row.finish_transform_hook(progress, &fragment).map_err(|cause| metadata.error(cause))?;
                } else {
                    let assembly = row.assembly().ok_or_else(|| metadata.coordinate())?;
                    let fragment = assembly.fragment(k).map_err(|cause| metadata.error(cause))?;
                    if decision == Decision::First {
                        let usage = super::super::bounded_capture::estimate_tensor_geometry(assembly.logical_geometry())
                            .map_err(|cause| metadata.error(cause))?;
                        if row.reserve_first(progress, &mut self.ledger, usage)
                            .map_err(|cause| metadata.error(cause))?.is_some() { continue; }
                    }
                    row.finish_hook(progress, &fragment).map_err(|cause| metadata.error(cause))?;
                }
            }
        }
        // No row advances if another is missing/failed, including zero fragments.
        self.context.charge_metadata(std::mem::size_of::<ColdCaptureProgressFailure>())?;
        for (i, progress) in self.progress.iter().enumerate() {
            policy
                .row(i)
                .map_err(|cause| metadata.error(cause))?
                .validate_chunk_end(progress, k)
                .map_err(|cause| metadata.error(ColdCaptureProgressFailure {
                    selection: i, rank: self.placement.map(|(_, rank)| rank), chunk: k, cause,
                }))?;
        }
        for (i, progress) in self.progress.iter_mut().enumerate() {
            policy
                .row(i)
                .map_err(|cause| metadata.error(cause))?
                .advance_chunk(progress, k)
                .map_err(|cause| metadata.error(cause))?;
        }
        if let Some(edits)=self.text_interventions {
            edits.try_borrow_mut().map_err(|cause|metadata.error(cause))?.as_mut()
                .ok_or_else(||metadata.coordinate())?.end_prefill(self.geometry,chunk,&self.context)?;
        }
        self.prefill_end = chunk.input.end;
        self.chunk = None;
        // The span report has retained all roots through the final model operator.
        // Canonical native retirement is a separate required execution contract.
        self.roots.clear();
        if self.prefill_end == self.geometry.input_positions {
            self.finish_text_interventions()?;
            self.prefill_complete = true;
            self.active = None;
            self.next_prediction = 1;
        }
        Ok(())
    }
    pub(super) fn observe_fragment_value(
        &mut self,
        path: &str,
        value: &WorkspaceTensor,
    ) -> Result<()> {
        self.observe_fragment_value_with_origin(path,value,true,true)
    }
    pub(super) fn observe_fragment_value_with_origin(
        &mut self,path:&str,value:&WorkspaceTensor,local:bool,source_present:bool,
    )->Result<()> {
        let metadata = Metadata::new(&self.context)?;
        self.context.charge_metadata(std::mem::size_of::<(&mut Self,&str,&WorkspaceTensor,bool,bool)>())?;
        let chunk = self.chunk.as_ref().ok_or_else(|| metadata.coordinate())?;
        let bound = self.bound.ok_or_else(|| metadata.coordinate())?;
        let policy = CapturePrefillObservationPolicy::from_bound(bound)
            .map_err(|cause| metadata.error(cause))?;
        let projected=if let Some((layouts,rank))=self.placement {
            self.context.charge_metadata(
                eredu_architectures::component_partition::ComponentPartitionLayouts::complete_capture_source_control_bytes()
                    .and_then(|n|n.checked_add(std::mem::size_of::<Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts,usize)>>()))
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
            layouts.complete_capture_source(path).is_none().then_some((layouts,rank))
        }else{None};
        for (index, progress) in self.progress.iter_mut().enumerate() {
            let row = policy.row(index).map_err(|cause| metadata.error(cause))?;
            let decision = row
                .begin_hook(progress, chunk, path)
                .map_err(|cause| metadata.error(cause))?;
            if decision == Decision::Ignore {
                continue;
            }
            // The reached source remains present when the first logical row
            // is skipped for quota; keep its actual scalar without tracing it.
            if source_present && self.placement.is_some() {
                record_source(self.scalar_source, index, &self.context, value)?;
            }
            if let Some(placement)=projected {
                super::partition::observer::observe(self.source,index,&row,progress,decision,placement,chunk,self.geometry,
                    source_present,value,&self.context,&mut self.roots,&mut self.ledger,self.transfers,self.scalar_source)?;
                continue;
            }
            if row.candidate().is_some() {
                let geometry = row
                    .candidate_for_chunk(chunk)
                    .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::CandidateExtraction::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program.validate_workspace(value, &self.context)?;
                let usage = super::super::bounded_capture::estimate_candidates(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if row
                    .reserve_first(progress, &mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    continue;
                }
                record_source(self.scalar_source,index,&self.context,value)?;
                if !local && source_present {record_replica(self.transfers,&self.context,&mut self.roots,value)?;}
                if local {
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                let outputs = program.trace(value, &self.context)?;
                metadata.reserve(&mut self.roots, outputs.len())?;
                self.roots.extend(outputs);
                if self.transfers.is_some() {
                    let population = CaptureNativePopulation::candidates()
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                }
                row.finish_candidate_hook(progress, chunk)
                    .map_err(|cause| metadata.error(cause))?;
                continue;
            }
            if row.token_scores().is_some() {
                let geometry = row
                    .token_scores_for_chunk(chunk)
                    .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::TokenScoreProgram::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                let usage = super::super::bounded_capture::estimate_token_scores(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if row
                    .reserve_first(progress, &mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    continue;
                }
                record_source(self.scalar_source,index,&self.context,value)?;
                if !local && source_present {record_replica(self.transfers,&self.context,&mut self.roots,value)?;}
                if local {
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                program
                    .trace(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                if self.transfers.is_some() {
                    let population = CaptureNativePopulation::token_scores(program)
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    record_population(self.transfers, &self.context, population)?;
                }
                }
                row.finish_token_scores_hook(progress, chunk)
                    .map_err(|cause| metadata.error(cause))?;
                continue;
            }
            if let Some(plan) = row
                .transform_plan()
                .filter(|plan| matches!(plan.selection().transform, CaptureTransform::Summary))
            {
                let fragment = plan
                    .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                    .map_err(|cause| metadata.error(cause))?;
                let geometry = eredu_core::capture::CaptureSummaryGeometry::prepare(
                    plan.admission(),
                    index,
                    CapturePhase::Prefill,
                    0,
                    None,
                )
                .and_then(|geometry| geometry.fragment(&fragment))
                .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::PreparedCaptureSummary::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                if decision == Decision::First {
                    let usage = super::super::bounded_capture::estimate_prefill_summary(plan)
                        .map_err(|cause| metadata.error(cause))?;
                    if row
                        .reserve_first(progress, &mut self.ledger, usage)
                        .map_err(|cause| metadata.error(cause))?
                        .is_some()
                    {
                        continue;
                    }
                }
                record_source(self.scalar_source, index, &self.context, value)?;
                if !local && source_present && fragment.selected_elements() != 0 {
                    record_replica(self.transfers, &self.context, &mut self.roots, value)?;
                }
                if local && fragment.selected_elements() != 0 {
                    metadata.reserve(&mut self.roots, 1)?;
                    self.roots.push(value.clone());
                    program
                        .trace(value, &self.context, &mut self.roots)
                        .map_err(|cause| metadata.error(cause))?;
                    if self.transfers.is_some() {
                        let population = program
                            .population()
                            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                        record_population(self.transfers, &self.context, population)?;
                    }
                }
                row.finish_transform_hook(progress, &fragment)
                    .map_err(|cause| metadata.error(cause))?;
                continue;
            }
            if let Some(plan) = row.transform_plan().filter(|plan| {
                matches!(
                    plan.selection().transform,
                    CaptureTransform::Histogram { .. }
                )
            }) {
                let fragment = plan
                    .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                    .map_err(|cause| metadata.error(cause))?;
                let geometry = eredu_core::capture::CaptureHistogramGeometry::prepare(
                    plan.admission(),
                    index,
                    CapturePhase::Prefill,
                    0,
                    None,
                )
                .and_then(|geometry| geometry.fragment(&fragment))
                .map_err(|cause| metadata.error(cause))?;
                let program =
                    crate::backend::array_copy::PreparedCaptureHistogram::from_geometry(&geometry)
                        .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                if decision == Decision::First {
                    let usage = super::super::bounded_capture::estimate_prefill_histogram(plan)
                        .map_err(|cause| metadata.error(cause))?;
                    if row
                        .reserve_first(progress, &mut self.ledger, usage)
                        .map_err(|cause| metadata.error(cause))?
                        .is_some()
                    {
                        continue;
                    }
                }
                record_source(self.scalar_source,index,&self.context,value)?;
                if !local && source_present && fragment.selected_elements()!=0 {
                    record_replica(self.transfers,&self.context,&mut self.roots,value)?;
                }
                if local && fragment.selected_elements() != 0 {
                    metadata.reserve(&mut self.roots, 1)?;
                    self.roots.push(value.clone());
                    program
                        .trace(value, &self.context, &mut self.roots)
                        .map_err(|cause| metadata.error(cause))?;
                    if self.transfers.is_some() {
                        let population = program
                            .population()
                            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                        record_population(self.transfers, &self.context, population)?;
                    }
                }
                row.finish_transform_hook(progress, &fragment)
                    .map_err(|cause| metadata.error(cause))?;
                continue;
            }
            let assembly = row.assembly().ok_or_else(|| metadata.coordinate())?;
            let fragment = assembly
                .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                .map_err(|cause| metadata.error(cause))?;
            let program = CaptureTensorSelection::from_fragment(&fragment)
                .map_err(|cause| metadata.error(cause))?;
            program
                .validate_workspace_source(value, &self.context)
                .map_err(|cause| metadata.error(cause))?;
            record_source(self.scalar_source, index, &self.context, value)?;
            if decision == Decision::First {
                let usage = super::super::bounded_capture::estimate_tensor_geometry(
                    assembly.logical_geometry(),
                )
                .map_err(|cause| metadata.error(cause))?;
                if row
                    .reserve_first(progress, &mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    continue;
                }
            }
            // Ordinary zero contributions are source-checked semantic hooks only.
            if !local && source_present && fragment.output_elements() != 0 {
                record_replica(self.transfers, &self.context, &mut self.roots, value)?;
            }
            if local && fragment.output_elements() != 0 {
                metadata.reserve(&mut self.roots, 1)?;
                self.roots.push(value.clone());
                program
                    .trace_retained_within(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
                record_transfer(self.transfers, &self.context)?;
            }
            row.finish_hook(progress, &fragment)
                .map_err(|cause| metadata.error(cause))?;
        }
        Ok(())
    }
    pub(super) fn observe_fragment_generated(
        &mut self,
        path: &str,
        prototype: &WorkspaceTensor,
        source: &GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<()> {
        let metadata = Metadata::new(&self.context)?;
        let chunk = self.chunk.as_ref().ok_or_else(|| metadata.coordinate())?;
        let policy = CapturePrefillObservationPolicy::from_bound(
            self.bound.ok_or_else(|| metadata.coordinate())?,
        )
        .map_err(|cause| metadata.error(cause))?;
        let mut generated = None;
        for (index, progress) in self.progress.iter_mut().enumerate() {
            let row = policy.row(index).map_err(|cause| metadata.error(cause))?;
            let decision = row
                .begin_hook(progress, chunk, path)
                .map_err(|cause| metadata.error(cause))?;
            if decision == Decision::Ignore {
                continue;
            }
            let assembly = row.assembly().ok_or_else(|| metadata.coordinate())?;
            let fragment = assembly
                .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                .map_err(|cause| metadata.error(cause))?;
            let selection = CaptureTensorSelection::from_fragment(&fragment)
                .map_err(|cause| metadata.error(cause))?;
            selection
                .validate_workspace_source(prototype, &self.context)
                .map_err(|cause| metadata.error(cause))?;
            row.validate_generated_fragment(&fragment, source, factory.program())
                .map_err(|cause| metadata.error(cause))?;
            if decision == Decision::First {
                let usage = super::super::bounded_capture::estimate_tensor_geometry(
                    assembly.logical_geometry(),
                )
                .map_err(|cause| metadata.error(cause))?;
                let usage = row
                    .full_generated_usage(usage)
                    .map_err(|cause| metadata.error(cause))?;
                if row
                    .reserve_first(progress, &mut self.ledger, usage)
                    .map_err(|cause| metadata.error(cause))?
                    .is_some()
                {
                    continue;
                }
            }
            if row
                .factory_required(&fragment)
                .map_err(|cause| metadata.error(cause))?
            {
                if generated.is_none() {
                    let GeneratedTensorProgram::BlockFp8Input(plan) = factory.program();
                    let (values_shape, scales_shape) = (plan.values_shape(), plan.scales_shape());
                    if self.context.uses_checked_metadata() {
                        self.context.reserve_metadata_vec(&mut self.roots, 9)?;
                    } else {
                        self.roots
                            .try_reserve_exact(9)
                            .map_err(|cause| metadata.error(cause))?;
                    }
                    let mut sources = 0;
                    factory.visit_sources(&mut |role, value| {
                        let (expected, shape, dtype) = match sources {
                            0 => (
                                GeneratedTensorSourceRole::CompactValues,
                                values_shape,
                                WorkspaceDtype::Uint8,
                            ),
                            1 => (
                                GeneratedTensorSourceRole::BlockScales,
                                scales_shape,
                                WorkspaceDtype::Float32,
                            ),
                            _ => return Err(metadata.error(CaptureProtocolError::GeneratedSource)),
                        };
                        self.context.validate_values([value])?;
                        if role != expected
                            || value.shape() != shape
                            || value.layout().dtype() != dtype
                        {
                            return Err(metadata.coordinate());
                        }
                        metadata.reserve(&mut self.roots, 1)?;
                        self.roots.push(value.clone());
                        sources += 1;
                        Ok(())
                    })?;
                    if sources != 2 {
                        return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                    }
                    let mut outputs = 0;
                    let value = factory.generate(&mut |value| {
                        if outputs == 7 {
                            return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                        }
                        self.context.validate_values([value])?;
                        metadata.reserve(&mut self.roots, 1)?;
                        self.roots.push(value.clone());
                        outputs += 1;
                        Ok(())
                    })?;
                    if outputs != 7 {
                        return Err(metadata.error(CaptureProtocolError::GeneratedSource));
                    }
                    generated = Some(value);
                }
                let value = generated.as_ref().expect("one physical factory per hook");
                selection
                    .validate_workspace_source(value, &self.context)
                    .map_err(|cause| metadata.error(cause))?;
                if value.layout().dtype() != WorkspaceDtype::Float32 {
                    return Err(metadata.coordinate());
                }
                // Preview(0) still traces this exact nonempty selected factory
                // and Selection program, matching the native fragment worker.
                selection
                    .trace_retained_within(value, &self.context, &mut self.roots)
                    .map_err(|cause| metadata.error(cause))?;
            }
            row.finish_hook(progress, &fragment)
                .map_err(|cause| metadata.error(cause))?;
        }
        Ok(())
    }
}
