//! Actual routed callbacks retain and price the same five native source loans.
use super::*;
use crate::backend::array_copy::CompletedRoutedCaptureSource;
use eredu_nn::workspace::WorkspaceDtype;
use eredu_runtime::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
use eredu_runtime::{RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver};
pub(super) mod partition;
mod source;
mod addressable;

fn routed_metadata(context: &WorkspaceContext) -> Result<Metadata> {
    let metadata = Metadata::new(context)?;
    context.charge_metadata(CaptureRoutedUnitsGeometry::preparation_control_bytes()
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    context.charge_metadata(std::mem::size_of::<(
        CaptureObservationStep<'_>, CaptureRoutedUnitsGeometry<'_>,
        std::result::Result<CaptureRoutedUnitsGeometry<'_>, CaptureTensorGeometryError>,
        CaptureUsage, CaptureUsage,
        ([&[i32]; 5], Option<[safemlx::Dtype; 5]>, u64,
            eredu_core::capture::RoutedUnitGeometry, u64),
        std::result::Result<(usize,u64),crate::backend::array_copy::CaptureTensorNativeError>,
    )>())?;
    Ok(metadata)
}

fn native_dtype(value: &WorkspaceTensor) -> Option<safemlx::Dtype> {
    Some(match value.layout().dtype() {
        WorkspaceDtype::Int32 => safemlx::Dtype::Int32,
        WorkspaceDtype::Uint32 => safemlx::Dtype::Uint32,
        _ => match value.layout().representation()?.dtype() {
            WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
            WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
            WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
        },
    })
}
#[derive(Debug, thiserror::Error)]
#[error("routed capture source {index} lacks its native scalar representation")]
struct RoutedSourceRepresentation { index: usize }

fn native_dtypes(values: [&WorkspaceTensor; 5], context: &WorkspaceContext)
    -> Result<[safemlx::Dtype; 5]>
{
    context.charge_metadata(std::mem::size_of::<(
        [&WorkspaceTensor; 5], [safemlx::Dtype; 5], usize,
        RoutedSourceRepresentation, Option<safemlx::Dtype>,
    )>())?;
    let mut result = [safemlx::Dtype::Float32; 5];
    for (index, value) in values.into_iter().enumerate() {
        result[index] = native_dtype(value)
            .ok_or_else(|| context.metadata_source(RoutedSourceRepresentation { index }))?;
    }
    Ok(result)
}
#[derive(Debug, thiserror::Error)]
#[error("routed capture source {index} has invalid logical dtype {dtype:?}")]
struct RoutedLogicalDtype { index: usize, dtype: WorkspaceDtype }

/// Preliminary shape accounting has only conservative floating metadata.
/// Native recorder callbacks must additionally preserve actual scalar facts.
fn routed_dtypes(values: [&WorkspaceTensor; 5], context: &WorkspaceContext, native: bool)
    -> Result<Option<[safemlx::Dtype; 5]>>
{
    context.charge_metadata(std::mem::size_of::<(
        [&WorkspaceTensor; 5], bool, usize, WorkspaceDtype, RoutedLogicalDtype,
        Option<[safemlx::Dtype; 5]>, Result<Option<[safemlx::Dtype; 5]>>,
    )>())?;
    for (index, value) in values.iter().enumerate() {
        let dtype = value.layout().dtype();
        let valid = if matches!(index, 0 | 3) {
            dtype == WorkspaceDtype::Float32
        } else {
            matches!(dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        };
        if !valid { return Err(context.metadata_source(RoutedLogicalDtype { index, dtype })); }
    }
    if native { native_dtypes(values, context).map(Some) } else { Ok(None) }
}

fn validate_routed_layouts(
    shapes: [&[i32]; 5], dtypes: Option<[safemlx::Dtype; 5]>, token_offset: u64,
    bank: eredu_core::capture::RoutedUnitGeometry, source_tokens: u64,
) -> std::result::Result<(usize,u64),crate::backend::array_copy::CaptureTensorNativeError> {
    match dtypes {
        Some(dtypes) => CompletedRoutedCaptureSource::validate_layouts(
            shapes,dtypes,token_offset,bank,source_tokens),
        None => CompletedRoutedCaptureSource::validate_geometry(
            shapes,token_offset,bank,source_tokens),
    }
}
impl CaptureWorkspaceObserver<'_> {
    fn same_routed_selection(&self, first: usize, index: usize) -> bool {
        match (
            &self.source.admission().points()[first].value_type,
            &self.source.admission().points()[index].value_type,
        ) {
            (
                eredu_core::ObservationValueType::RoutedUnits { routing: a, .. },
                eredu_core::ObservationValueType::RoutedUnits { routing: b, .. },
            ) => a == b,
            _ => false,
        }
    }
    fn routed_batch(
        &mut self,
        batch: &RoutedUnitBatch<'_, WorkspaceTensor>,
        effective: bool,
    ) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        if !self.routed_active || self.routed_addressable_rows.is_some() || (self.placement.is_none() && (batch.origins.is_some() || batch.unit_coordinates.is_some())) {
            return Err(metadata.coordinate());
        }
        let Some(first) = self.routed_selection else { return Ok(()); };
        if self.chunk.is_none() {
            return self.routed_invocation_batch(batch, effective, first);
        }
        self.context.charge_metadata(std::mem::size_of::<eredu_runtime::prefill::PrefillChunk>())?;
        let chunk = self.chunk.clone().ok_or_else(|| metadata.coordinate())?;
        let bound = self.bound.ok_or_else(|| metadata.coordinate())?;
        let policy =
            CapturePrefillObservationPolicy::from_bound(bound).map_err(|e| metadata.error(e))?;
        let source = batch.capture_source().map_err(|e| metadata.error(e))?;
        self.context.charge_metadata(std::mem::size_of::<(
            RoutedUnitCaptureSource<'_, WorkspaceTensor>,
            [&WorkspaceTensor; 5],
            [&[i32]; 5],
            [safemlx::Dtype; 5],
            Result<()>,
            (usize, u64, u64),
        )>())?;
        let values = [
            source.values,
            source.token_indices,
            source.selection_indices,
            source.coefficients,
            source.source_groups,
        ];
        let shapes = values.map(|value| value.layout().shape());
        let dtypes = routed_dtypes(values, &self.context,
            self.transfers.is_some() || self.scalar_source.is_some())?;
        let mut visited = None;
        for index in 0..self.progress.len() {
            if !self.same_routed_selection(first, index)
                || (self.source.admission().points()[index].position
                    == eredu_core::ObservationPosition::AfterIntervention)
                    != effective
            {
                continue;
            }
            let row = policy.row(index).map_err(|e| metadata.error(e))?;
            let Some(plan) = row.routed_plan() else {
                continue;
            };
            let fragment = plan
                .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                .map_err(|e| metadata.error(e))?;
            let progress = &mut self.progress[index];
            let decision = row
                .begin_routed_batch(
                    progress,
                    &chunk,
                    &self.source.admission().plan().selections[index].path,
                )
                .map_err(|e| metadata.error(e))?;
            if decision == CapturePrefillHookDecision::Ignore {
                continue;
            }
            let (_, tokens) = if let Some(placement)=self.placement {
                partition::validate(self.source,placement,index,shapes,dtypes,source.token_offset,batch,
                    plan.geometry().bank(),fragment.source_tokens(),self.routed_native_rows,&self.context)?
            } else {
                validate_routed_layouts(shapes,dtypes,source.token_offset,plan.geometry().bank(),
                    fragment.source_tokens()).map_err(|e|metadata.error(e))?
            };
            record_source(self.scalar_source,index,&self.context,source.values)?;
            visited = Some(tokens);
            if decision == CapturePrefillHookDecision::First {
                let usage =
                    super::super::bounded_capture::estimate_routed_geometry(plan.geometry())
                        .map_err(|e| metadata.error(e))?;
                if row
                    .reserve_first(progress, &mut self.ledger, usage)
                    .map_err(|e| metadata.error(e))?
                    .is_some()
                {
                    continue;
                }
            }
            self.retain_selected_routed_source(index, &source, plan.geometry())?;
        }
        if let Some(tokens) = visited {
            self.routed_tokens[usize::from(effective)] = self.routed_tokens[usize::from(effective)]
                .checked_add(tokens)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        }
        Ok(())
    }
}
impl RoutedUnitObserver<WorkspaceTensor> for CaptureWorkspaceObserver<'_> {
    fn observe_addressable_source(&mut self,source:eredu_nn::workspace::WorkspaceAddressableObservationView<'_>)
        ->Result<eredu_nn::workspace::WorkspaceAddressableObservationSource>{
        self.routed_addressable_source(source)
    }

    fn observe_region_source(&mut self,source:eredu_nn::workspace::WorkspaceExpertObservationView<'_>)
        ->Result<eredu_nn::workspace::WorkspaceExpertObservationSource> {
        self.routed_region_source(source)
    }

    fn invocation_active(&self) -> bool {
        self.routed_active
    }
    fn begin_invocation(
        &mut self,
        input: &RoutedUnitInvocation<'_, WorkspaceTensor>,
    ) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        if self.routed_active
            || (self.routed_selection.is_none() && self.routed_intervention_selection.is_none())
            || (self.placement.is_none() && input.origins.is_some())
        {
            return Err(metadata.coordinate());
        }
        if self.chunk.is_none() {
            if let Some(first) = self.routed_selection {
                for index in 0..self.statuses.len() {
                    if self.same_routed_selection(first, index)
                        && !matches!(self.statuses[index], CaptureRecordStatus::Missing | CaptureRecordStatus::Skipped) {
                        return Err(metadata.coordinate());
                    }
                }
            }
        }
        self.routed_native_rows = if let Some(placement)=self.placement {
            let first=self.routed_selection.ok_or_else(||metadata.coordinate())?;
            let source_tokens=if let Some(chunk)=self.chunk.as_ref() {
                self.geometry.batch_size.checked_mul(chunk.input.end-chunk.input.start)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?
            } else {
                self.policy()?.routed_geometry(first).map_err(|e|metadata.error(e))?.source_shape()[0] as u64
            };
            Some(partition::begin(self.source,placement,first,input,source_tokens,&self.context)?)
        } else { None };
        if let Some(first) = self.routed_intervention_selection {
            if self.placement.is_some() || self.chunk.is_some() { return Err(metadata.coordinate()); }
            self.interventions.ok_or_else(|| metadata.coordinate())?.try_borrow_mut()
                .map_err(|cause| metadata.error(cause))?.as_mut().ok_or_else(|| metadata.coordinate())?
                .begin_routed(first, input, &self.context, &mut self.ledger)?;
        }
        self.begin_partition_routed_input(input.input)?;
        self.routed_addressable_rows=None;
        self.routed_active = true;
        self.routed_tokens = [0; 2];
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        if !self.routed_active {
            return Err(metadata.coordinate());
        }
        self.routed_active = false;
        if let Some(first) = self.routed_intervention_selection {
            self.interventions.ok_or_else(|| metadata.coordinate())?.try_borrow_mut()
                .map_err(|cause| metadata.error(cause))?.as_mut().ok_or_else(|| metadata.coordinate())?
                .finish_routed(first, success, &self.context)?;
        }
        let Some(first) = self.routed_selection else { return Ok(()); };
        if self.chunk.is_none() {
            return self.finish_routed_ordinary_invocation(success, first);
        }
        self.context.charge_metadata(std::mem::size_of::<eredu_runtime::prefill::PrefillChunk>())?;
        let chunk = self.chunk.clone().ok_or_else(|| metadata.coordinate())?;
        let policy = CapturePrefillObservationPolicy::from_bound(
            self.bound.ok_or_else(|| metadata.coordinate())?,
        )
        .map_err(|e| metadata.error(e))?;
        for index in 0..self.progress.len() {
            if !self.same_routed_selection(first, index) {
                continue;
            }
            let row = policy.row(index).map_err(|e| metadata.error(e))?;
            let Some(plan) = row.routed_plan() else {
                continue;
            };
            if !success {
                self.progress[index].fail_hook();
                continue;
            }
            if self.routed_native_rows==Some(0) {
                // An actual idle provider has no batch/result dtype. Advance the
                // ordinary logical row without inventing a tensor or completion.
                let decision=row.begin_routed_batch(&mut self.progress[index],&chunk,
                    &self.source.admission().plan().selections[index].path).map_err(|e|metadata.error(e))?;
                if decision==CapturePrefillHookDecision::First {
                    let usage=super::super::bounded_capture::estimate_routed_geometry(plan.geometry()).map_err(|e|metadata.error(e))?;
                    if row.reserve_first(&mut self.progress[index],&mut self.ledger,usage)
                        .map_err(|e|metadata.error(e))?.is_some(){continue;}
                }
            }
            if row
                .is_skipped(&self.progress[index])
                .map_err(|e| metadata.error(e))?
            {
                continue;
            }
            let fragment = plan
                .fragment(chunk.input.start / self.geometry.prefill_chunk_positions)
                .map_err(|e| metadata.error(e))?;
            let effective = self.source.admission().points()[index].position
                == eredu_core::ObservationPosition::AfterIntervention;
            if self.routed_addressable_rows.unwrap_or(self.routed_tokens[usize::from(effective)])
                != self.routed_native_rows.unwrap_or(fragment.source_tokens()) {
                return Err(metadata.coordinate());
            }
            row.finish_routed_hook(&mut self.progress[index], &fragment)
                .map_err(|e| metadata.error(e))?;
        }
        Ok(())
    }
    fn intervene(&mut self, batch: &RoutedUnitBatch<'_, WorkspaceTensor>) -> Result<Option<WorkspaceTensor>> {
        let Some(first) = self.routed_intervention_selection else { return Ok(None); };
        let metadata = routed_metadata(&self.context)?;
        if !self.routed_active { return Err(metadata.coordinate()); }
        let (output, population) = self.interventions.ok_or_else(|| metadata.coordinate())?
            .try_borrow_mut().map_err(|cause| metadata.error(cause))?
            .as_mut().ok_or_else(|| metadata.coordinate())?
            .trace_routed(first, batch, &self.context, &mut self.roots)?;
        record_population(self.transfers, &self.context, population)?;
        Ok(output)
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, WorkspaceTensor>) -> Result<()> {
        self.routed_batch(batch, false)
    }
    fn observe_effective(&mut self, batch: &RoutedUnitBatch<'_, WorkspaceTensor>) -> Result<()> {
        self.routed_batch(batch, true)
    }
}

impl CaptureWorkspaceObserver<'_> {
    /// The original partition hook completes the provider input once for each
    /// admitted selection before receiving any batch, including idle providers.
    /// Use the same row progression here so skipped rows gain no native work.
    fn begin_partition_routed_input(&mut self,input:&WorkspaceTensor)->Result<()> {
        let count=self.prepare_partition_routed_input()?;
        for _ in 0..count {record_replica(self.transfers,&self.context,&mut self.roots,input)?;}
        Ok(())
    }
    /// Shared admission/progression before actual input settlement or a cold
    /// region descriptor. This changes no native roots or source precision.
    fn prepare_partition_routed_input(&mut self)->Result<usize> {
        if self.placement.is_none(){return Ok(0);}
        self.prepare_routed_input()
    }
    fn prepare_routed_input(&mut self)->Result<usize>{
        let mut count=0usize;
        let metadata=routed_metadata(&self.context)?;
        let first=self.routed_selection.ok_or_else(||metadata.coordinate())?;
        self.context.charge_metadata(std::mem::size_of::<(
            &mut Self,&WorkspaceTensor,Option<eredu_runtime::prefill::PrefillChunk>,
            CapturePrefillObservationPolicy<'_>,CapturePrefillHookDecision,
            CaptureObservationStep<'_>,CapturePhase,usize,u64,Result<()>,
        )>())?;
        if let Some(chunk)=self.chunk.clone() {
            let policy=CapturePrefillObservationPolicy::from_bound(
                self.bound.ok_or_else(||metadata.coordinate())?).map_err(|e|metadata.error(e))?;
            for index in 0..self.progress.len() {
                if !self.same_routed_selection(first,index){continue;}
                let row=policy.row(index).map_err(|e|metadata.error(e))?;
                let Some(plan)=row.routed_plan() else{continue};
                let decision=row.begin_routed_batch(&mut self.progress[index],&chunk,
                    &self.source.admission().plan().selections[index].path).map_err(|e|metadata.error(e))?;
                if decision==CapturePrefillHookDecision::Ignore{continue;}
                if decision==CapturePrefillHookDecision::First {
                    let usage=super::super::bounded_capture::estimate_routed_geometry(plan.geometry())
                        .map_err(|e|metadata.error(e))?;
                    if row.reserve_first(&mut self.progress[index],&mut self.ledger,usage)
                        .map_err(|e|metadata.error(e))?.is_some(){continue;}
                }
                count=count.checked_add(1).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            }
        } else {
            let prediction=self.active.ok_or_else(||metadata.coordinate())?;
            let phase=self.invocation.map_or(if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode},|i|i.phase);
            let policy=CaptureObservationStep::with_invocation(self.source.admission(),phase,prediction,
                self.invocation.map(|i|i.shape)).and_then(|p|p.with_window(self.invocation.and_then(|i|i.window)))
                .map_err(|e|metadata.error(e))?;
            for index in 0..self.statuses.len() {
                if !self.same_routed_selection(first,index)||self.statuses[index]==CaptureRecordStatus::Skipped{continue;}
                let geometry=policy.routed_geometry(index).map_err(|e|metadata.error(e))?;
                if self.statuses[index]!=CaptureRecordStatus::Missing{return Err(metadata.coordinate());}
                if let Some(usage)=policy.window_metadata_usage(index).map_err(|e|metadata.error(e))? {
                    if policy.reserve_value(&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){
                        self.statuses[index]=CaptureRecordStatus::Skipped;continue;
                    }
                }
                let usage=super::super::bounded_capture::estimate_routed_geometry(&geometry).map_err(|e|metadata.error(e))?;
                if policy.reserve_value(&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){
                    self.statuses[index]=CaptureRecordStatus::Skipped;continue;
                }
                self.statuses[index]=CaptureRecordStatus::Consumed;
                count=count.checked_add(1).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            }
        }
        Ok(count)
    }
    fn retain_selected_routed_source(&mut self,index:usize,source:&RoutedUnitCaptureSource<'_,WorkspaceTensor>,
        geometry:&CaptureRoutedUnitsGeometry<'_>)->Result<()> {
        let Some(placement)=self.placement else{return self.retain_routed_source(index,source)};
        let copies=partition::fragments(self.source,placement,index,geometry,&self.context)?;
        record_source(self.scalar_source,index,&self.context,source.values)?;
        if copies==0 {
            for value in [source.values,source.token_indices,source.selection_indices,source.coefficients,source.source_groups] {
                record_replica(self.transfers,&self.context,&mut self.roots,value)?;
            }
        } else {
            let metadata=routed_metadata(&self.context)?;
            for _ in 0..copies {
                metadata.reserve(&mut self.roots,5)?;
                for value in [source.values,source.token_indices,source.selection_indices,source.coefficients,source.source_groups] {
                    self.roots.push(value.clone());
                }
                record_population(self.transfers,&self.context,CaptureNativePopulation::routed_partition()
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
            }
        }
        Ok(())
    }

    fn retain_routed_source(
        &mut self, index: usize, source: &RoutedUnitCaptureSource<'_, WorkspaceTensor>,
    ) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        record_source(self.scalar_source, index, &self.context, source.values)?;
        metadata.reserve(&mut self.roots, 5)?;
        for value in [source.values, source.token_indices, source.selection_indices,
            source.coefficients, source.source_groups] {
            self.roots.push(value.clone());
        }
        let population = if self.invocation.is_some() {
            CaptureNativePopulation::routed_model()
        } else {
            CaptureNativePopulation::routed_prefill()
        }.ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        record_population(self.transfers, &self.context, population)?;
        Ok(())
    }

    fn routed_invocation_batch(
        &mut self, batch: &RoutedUnitBatch<'_, WorkspaceTensor>, effective: bool, first: usize,
    ) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        let prediction = self.active.ok_or_else(|| metadata.coordinate())?;
        let phase = self.invocation.map_or(
            if prediction == 0 { CapturePhase::Prefill } else { CapturePhase::Decode },
            |invocation| invocation.phase);
        let policy = CaptureObservationStep::with_invocation(
            self.source.admission(), phase, prediction,
            self.invocation.map(|invocation| invocation.shape),
        ).and_then(|policy| policy.with_window(self.invocation.and_then(|invocation| invocation.window)))
            .map_err(|cause| metadata.error(cause))?;
        let source = batch.capture_source().map_err(|cause| metadata.error(cause))?;
        self.context.charge_metadata(std::mem::size_of::<(
            RoutedUnitCaptureSource<'_, WorkspaceTensor>, [&WorkspaceTensor; 5],
            [&[i32]; 5], [safemlx::Dtype; 5], Result<()>, (usize, u64, u64),
        )>())?;
        let values = [source.values, source.token_indices, source.selection_indices,
            source.coefficients, source.source_groups];
        let shapes = values.map(|value| value.layout().shape());
        let dtypes = routed_dtypes(values, &self.context,
            self.transfers.is_some() || self.scalar_source.is_some())?;
        let mut visited = None;
        for index in 0..self.statuses.len() {
            if !self.same_routed_selection(first, index)
                || (self.source.admission().points()[index].position
                    == eredu_core::ObservationPosition::AfterIntervention) != effective
                || self.statuses[index] == CaptureRecordStatus::Skipped {
                continue;
            }
            let geometry = policy.routed_geometry(index).map_err(|cause| metadata.error(cause))?;
            let (_,tokens)=if let Some(placement)=self.placement {
                partition::validate(self.source,placement,index,shapes,dtypes,source.token_offset,batch,
                    geometry.bank(),geometry.source_shape()[0] as u64,self.routed_native_rows,&self.context)?
            }else{validate_routed_layouts(shapes,dtypes,source.token_offset,geometry.bank(),
                geometry.source_shape()[0] as u64).map_err(|e|metadata.error(e))?};
            record_source(self.scalar_source,index,&self.context,source.values)?;
            if self.statuses[index] == CaptureRecordStatus::Missing {
                let usage = super::super::bounded_capture::estimate_routed_geometry(&geometry)
                    .map_err(|cause| metadata.error(cause))?;
                if let Some(usage) = policy.window_metadata_usage(index).map_err(|cause| metadata.error(cause))? {
                    if policy.reserve_value(&mut self.ledger, usage).map_err(|cause| metadata.error(cause))?.is_some() {
                        self.statuses[index] = CaptureRecordStatus::Skipped;
                        continue;
                    }
                }
                if policy.reserve_value(&mut self.ledger, usage).map_err(|cause| metadata.error(cause))?.is_some() {
                    self.statuses[index] = CaptureRecordStatus::Skipped;
                    continue;
                }
                self.statuses[index] = CaptureRecordStatus::Consumed;
            }
            visited = Some(tokens);
            self.retain_selected_routed_source(index, &source, &geometry)?;
        }
        if let Some(tokens) = visited {
            self.routed_tokens[usize::from(effective)] =
                self.routed_tokens[usize::from(effective)].checked_add(tokens)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        }
        Ok(())
    }

    fn finish_routed_ordinary_invocation(&mut self, success: bool, first: usize) -> Result<()> {
        let metadata = routed_metadata(&self.context)?;
        if !success { return Ok(()); }
        let prediction = self.active.ok_or_else(|| metadata.coordinate())?;
        let phase = self.invocation.map_or(
            if prediction == 0 { CapturePhase::Prefill } else { CapturePhase::Decode },
            |invocation| invocation.phase);
        let policy = CaptureObservationStep::with_invocation(
            self.source.admission(), phase, prediction,
            self.invocation.map(|invocation| invocation.shape),
        ).and_then(|policy| policy.with_window(self.invocation.and_then(|invocation| invocation.window)))
            .map_err(|cause| metadata.error(cause))?;
        for index in 0..self.statuses.len() {
            if !self.same_routed_selection(first, index)
                || self.statuses[index] == CaptureRecordStatus::Skipped { continue; }
            let geometry = policy.routed_geometry(index).map_err(|cause| metadata.error(cause))?;
            if self.routed_native_rows==Some(0) && self.statuses[index]==CaptureRecordStatus::Missing {
                self.statuses[index]=CaptureRecordStatus::Consumed;
                if let Some(usage)=policy.window_metadata_usage(index).map_err(|e|metadata.error(e))? {
                    if policy.reserve_value(&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){
                        self.statuses[index]=CaptureRecordStatus::Skipped;continue;
                    }
                }
                let usage=super::super::bounded_capture::estimate_routed_geometry(&geometry).map_err(|e|metadata.error(e))?;
                if policy.reserve_value(&mut self.ledger,usage).map_err(|e|metadata.error(e))?.is_some(){
                    self.statuses[index]=CaptureRecordStatus::Skipped;continue;
                }
            }
            let effective = self.source.admission().points()[index].position
                == eredu_core::ObservationPosition::AfterIntervention;
            if self.statuses[index] != CaptureRecordStatus::Consumed
                || self.routed_addressable_rows.unwrap_or(self.routed_tokens[usize::from(effective)])
                    != self.routed_native_rows.unwrap_or(geometry.source_shape()[0] as u64) {
                return Err(metadata.coordinate());
            }
        }
        Ok(())
    }
}
