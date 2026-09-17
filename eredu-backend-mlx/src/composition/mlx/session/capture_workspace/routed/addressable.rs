//! Prospective member callbacks share the original invocation reservation.
use super::*;
use eredu_nn::workspace::{WorkspaceAddressableObservationView,WorkspaceAddressableObservationSource,
    WorkspaceGroupedSourceRetention,WorkspaceMetadataError};

impl CaptureWorkspaceObserver<'_>{
    pub(super) fn routed_addressable_source(&mut self,source:WorkspaceAddressableObservationView<'_>)
        ->Result<WorkspaceAddressableObservationSource>{
        let metadata=routed_metadata(&self.context)?;
        self.context.charge_metadata(std::mem::size_of::<(WorkspaceAddressableObservationView<'_>,
            WorkspaceAddressableObservationSource,WorkspaceGroupedSourceRetention,
            Result<WorkspaceAddressableObservationSource>,[usize;4],[u64;3])>())?;
        if !self.routed_active||self.routed_addressable_rows.is_some(){return Err(metadata.coordinate());}
        let first=self.routed_selection.ok_or_else(||metadata.coordinate())?;
        source.region.validate()?;
        let bank=match self.source.admission().points()[first].value_type{
            eredu_core::ObservationValueType::RoutedUnits{geometry,..}=>geometry,
            _=>return Err(metadata.coordinate()),
        };
        let source_tokens=if let Some(chunk)=self.chunk.as_ref(){
            self.geometry.batch_size.checked_mul(chunk.input.end-chunk.input.start).ok_or(WorkspaceMetadataError::Overflow)?
        }else{u64::try_from(self.policy()?.routed_geometry(first).map_err(|e|metadata.error(e))?.source_shape()[0])
            .map_err(|_|WorkspaceMetadataError::Overflow)?};
        if u64::try_from(source.region.chunks.rows).ok()!=Some(source_tokens)
            || u64::try_from(source.region.chunks.routes).ok()!=Some(bank.routes_per_token)
            || source.envelope.maximum_provider_rows()!=source.region.chunks.rows.min(source.region.chunks.chunk_rows)
            || source.envelope.routes_per_row()!=source.region.chunks.routes {
            return Err(metadata.coordinate());
        }
        let width=source.envelope.unit_columns();
        if let Some(placement)=self.placement {
            let local=partition::local(self.source,placement,first,&self.context)?.ok_or_else(||metadata.coordinate())?;
            // This source is an addressable local provider, never an exchanged
            // receive envelope. Actual member and scalar maps remain borrowed.
            if local.ownership.source_peer.is_some()||local.ownership.source_peers!=1
                || local.ownership.coordinates.experts().local_count()!=source.region.chunks.members
                || local.ownership.coordinates.units().local_count()!=width
                || source.unit_coordinates.is_some_and(|map|map!=local.ownership.coordinates.units()) {
                return Err(metadata.coordinate());
            }
            for member in 0..source.region.chunks.members {
                let global=source.region.local_members.map_or(member,|ids|ids[member]);
                if local.ownership.coordinates.experts().global_to_local(global)!=Some(member){return Err(metadata.coordinate());}
            }
        }else if source.region.local_members.is_some()
            || u64::try_from(source.region.chunks.members).ok()!=Some(bank.experts)
            || u64::try_from(width).ok()!=Some(bank.units_per_expert)
            || source.unit_coordinates.is_some_and(|map|map.local_count()!=width||map.global_count()!=width){
            return Err(metadata.coordinate());
        }
        // Serial begin reserves at its first actual batch; the prospective path
        // reaches the same reservation worker here without creating a batch.
        if self.placement.is_none(){self.prepare_routed_input()?;}
        let dtype=source.units.representation().map(|v|v.dtype());
        let mut result=WorkspaceAddressableObservationSource{unit_dtype:dtype,..Default::default()};
        for index in 0..self.statuses.len(){
            if !self.same_routed_selection(first,index){continue;}
            record_scalar_source(self.scalar_source,index,&self.context,dtype)?;
            let retention=if let Some(chunk)=self.chunk.as_ref(){
                let policy=CapturePrefillObservationPolicy::from_bound(self.bound.ok_or_else(||metadata.coordinate())?)
                    .map_err(|e|metadata.error(e))?;
                let row=policy.row(index).map_err(|e|metadata.error(e))?;
                let Some(plan)=row.routed_plan()else{continue};
                if row.is_skipped(&self.progress[index]).map_err(|e|metadata.error(e))?{continue;}
                // Original chunk alignment is supplied by the same bound plan.
                let _=plan.fragment(chunk.input.start/self.geometry.prefill_chunk_positions).map_err(|e|metadata.error(e))?;
                self.addressable_retention(index,plan.geometry())?
            }else{
                if self.statuses[index]==CaptureRecordStatus::Skipped{continue;}
                if self.statuses[index]!=CaptureRecordStatus::Consumed{return Err(metadata.coordinate());}
                let geometry=self.policy()?.routed_geometry(index).map_err(|e|metadata.error(e))?;
                self.addressable_retention(index,&geometry)?
            };
            let target=if self.source.admission().points()[index].position==eredu_core::ObservationPosition::AfterIntervention{
                &mut result.after
            }else{&mut result.before};
            target.partition_copies=target.partition_copies.checked_add(retention.partition_copies).ok_or(WorkspaceMetadataError::Overflow)?;
            target.replica_settlements=target.replica_settlements.checked_add(retention.replica_settlements).ok_or(WorkspaceMetadataError::Overflow)?;
            target.callback_control_bytes=target.callback_control_bytes.checked_add(retention.callback_control_bytes).ok_or(WorkspaceMetadataError::Overflow)?;
        }
        result.validate()?;
        self.routed_addressable_rows=Some(source_tokens);
        Ok(result)
    }
    fn addressable_retention(&self,index:usize,geometry:&CaptureRoutedUnitsGeometry<'_>)
        ->Result<WorkspaceGroupedSourceRetention>{
        if self.placement.is_some(){return self.region_retention(index,geometry);}
        Ok(WorkspaceGroupedSourceRetention{partition_copies:1,replica_settlements:0,
            callback_control_bytes:CaptureNativePopulation::routed_prefill().ok_or(WorkspaceMetadataError::Overflow)?.controls})
    }
}
