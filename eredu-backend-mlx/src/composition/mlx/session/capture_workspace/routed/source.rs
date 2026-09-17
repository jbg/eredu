//! Prospective EP retention uses ordinary capture policy and original placement.
use super::*;
use eredu_nn::workspace::{WorkspaceExpertObservationView,WorkspaceExpertObservationSource,
    WorkspaceGroupedSourceRetention,WorkspaceMetadataError};

impl CaptureWorkspaceObserver<'_> {
    pub(super) fn routed_region_source(&mut self,source:WorkspaceExpertObservationView<'_>)
        ->Result<WorkspaceExpertObservationSource> {
        let metadata=routed_metadata(&self.context)?;
        self.context.charge_metadata(std::mem::size_of::<(WorkspaceExpertObservationView<'_>,
            WorkspaceExpertObservationSource,WorkspaceGroupedSourceRetention,
            Result<WorkspaceExpertObservationSource>,usize,u64,Option<WorkspaceFloatingType>)>())?;
        if self.routed_active {return Err(metadata.coordinate());}
        let first=self.routed_selection.ok_or_else(||metadata.coordinate())?;
        let placement=self.placement.ok_or_else(||metadata.coordinate())?;
        source.region.validate()?;
        let source_tokens=if let Some(chunk)=self.chunk.as_ref(){
            self.geometry.batch_size.checked_mul(chunk.input.end-chunk.input.start)
                .ok_or(WorkspaceMetadataError::Overflow)?
        }else{
            u64::try_from(self.policy()?.routed_geometry(first).map_err(|e|metadata.error(e))?
                .source_shape()[0]).map_err(|_|WorkspaceMetadataError::Overflow)?
        };
        let local=partition::local(self.source,placement,first,&self.context)?
            .ok_or_else(||metadata.coordinate())?;
        let geometry=match self.source.admission().points()[first].value_type {
            eredu_core::ObservationValueType::RoutedUnits{geometry,..}=>geometry,
            _=>return Err(metadata.coordinate()),
        };
        let layout=PartitionRoutedUnitCaptureLayout{geometry,source_tokens,ownership:local.ownership};
        layout.validate().map_err(|e|metadata.error(e))?;
        let expected=layout.native_shape().map_err(|e|metadata.error(e))?;
        let view=source.region;
        let maximum=view.maximum_received_rows().ok_or(WorkspaceMetadataError::Overflow)?;
        if u64::try_from(view.source_rows).ok()!=Some(source_tokens)
            || u64::try_from(view.routes_per_row).ok()!=Some(geometry.routes_per_token)
            || u64::try_from(view.peers).ok()!=Some(local.ownership.source_peers)
            || local.ownership.source_peer.is_none()
            || source.envelope.maximum_provider_rows()!=maximum
            || source.envelope.routes_per_row()!=1
            || u64::try_from(source.envelope.unit_columns()).ok()!=Some(expected[2])
            || (view.local_experts()!=0 && u64::try_from(maximum).ok()!=Some(expected[0]))
            || local.ownership.coordinates.experts().local_count()!=view.local_experts()
            || view.owners.iter().enumerate().any(|(expert,&owner)|owner==view.rank
                && local.ownership.coordinates.experts().global_to_local(expert)!=Some(view.owner_local[expert])) {
            return Err(metadata.coordinate());
        }
        // A prospective nonempty source supplies its actual selected Units
        // dtype. It is not an assertion that this invocation receives any rows.
        let dtype=source.units.representation().map(|value|value.dtype());
        if maximum!=0 {
            for index in 0..self.progress.len(){
                if self.same_routed_selection(first,index){
                    record_scalar_source(self.scalar_source,index,&self.context,dtype)?;
                }
            }
        }
        let inputs=self.prepare_partition_routed_input()?;
        let input_controls=super::super::super::model_session::text_funding::capture_replica_control_bytes()
            .and_then(|n|n.checked_mul(inputs)).ok_or(WorkspaceMetadataError::Overflow)?;
        let mut result=WorkspaceExpertObservationSource{input_settlements:inputs,
            input_control_bytes:input_controls,unit_dtype:dtype,..Default::default()};
        if let Some(chunk)=self.chunk.clone(){
            let policy=CapturePrefillObservationPolicy::from_bound(
                self.bound.ok_or_else(||metadata.coordinate())?).map_err(|e|metadata.error(e))?;
            for index in 0..self.progress.len(){
                if !self.same_routed_selection(first,index){continue;}
                let row=policy.row(index).map_err(|e|metadata.error(e))?;
                let Some(plan)=row.routed_plan() else{continue};
                if row.is_skipped(&self.progress[index]).map_err(|e|metadata.error(e))?{continue;}
                let fragment=plan.fragment(chunk.input.start/self.geometry.prefill_chunk_positions)
                    .map_err(|e|metadata.error(e))?;
                let retention=self.region_retention(index,plan.geometry())?;
                add_retention(&mut result,self.source.admission().points()[index].position
                    ==eredu_core::ObservationPosition::AfterIntervention,retention)?;
                // Only the cold logical source advances here. Actual callbacks
                // still validate received extents, origins and final hook vote.
                row.finish_routed_hook(&mut self.progress[index],&fragment).map_err(|e|metadata.error(e))?;
            }
        }else{
            let policy=self.policy()?;
            for index in 0..self.statuses.len(){
                if !self.same_routed_selection(first,index)||self.statuses[index]==CaptureRecordStatus::Skipped{continue;}
                if self.statuses[index]!=CaptureRecordStatus::Consumed{return Err(metadata.coordinate());}
                let geometry=policy.routed_geometry(index).map_err(|e|metadata.error(e))?;
                let retention=self.region_retention(index,&geometry)?;
                add_retention(&mut result,self.source.admission().points()[index].position
                    ==eredu_core::ObservationPosition::AfterIntervention,retention)?;
            }
        }
        result.validate()?;
        Ok(result)
    }
    pub(super) fn region_retention(&self,index:usize,geometry:&CaptureRoutedUnitsGeometry<'_>)
        ->Result<WorkspaceGroupedSourceRetention>{
        let metadata=routed_metadata(&self.context)?;
        let copies=partition::fragments(self.source,self.placement.ok_or_else(||metadata.coordinate())?,
            index,geometry,&self.context)?;
        let (replicas,controls)=if copies==0 {
            (1,super::super::super::model_session::text_funding::capture_replica_control_bytes()
                .and_then(|n|n.checked_mul(5)).ok_or(WorkspaceMetadataError::Overflow)?)
        }else{(0,CaptureNativePopulation::routed_partition().and_then(|v|v.controls.checked_mul(copies))
            .ok_or(WorkspaceMetadataError::Overflow)?)};
        Ok(WorkspaceGroupedSourceRetention{partition_copies:copies,replica_settlements:replicas,
            callback_control_bytes:controls})
    }
}
fn add_retention(source:&mut WorkspaceExpertObservationSource,effective:bool,
    value:WorkspaceGroupedSourceRetention)->Result<()> {
    let target=if effective{&mut source.after}else{&mut source.before};
    target.partition_copies=target.partition_copies.checked_add(value.partition_copies).ok_or(WorkspaceMetadataError::Overflow)?;
    target.replica_settlements=target.replica_settlements.checked_add(value.replica_settlements).ok_or(WorkspaceMetadataError::Overflow)?;
    target.callback_control_bytes=target.callback_control_bytes.checked_add(value.callback_control_bytes).ok_or(WorkspaceMetadataError::Overflow)?;
    Ok(())
}
