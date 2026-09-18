//! Prospective sparse observer source; no concrete route batch is synthesized.
use super::*;

/// Number of retained five-source copies or replica settlements for one actual
/// callback. Each field describes the selected observer program, never a grant.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct WorkspaceGroupedSourceRetention {
    pub partition_copies: usize,
    pub replica_settlements: usize,
    /// Exact host worker census supplied by the selected observer mechanism.
    pub callback_control_bytes: usize,
}
impl WorkspaceGroupedSourceRetention {
    pub fn source_loans(self)->Option<usize>{
        self.partition_copies.checked_add(self.replica_settlements)?.checked_mul(5)
    }
}
/// Immutable observer selection attached to a region's existing numerical source.
/// Counts apply per actual invocation/batch. The selected native worker must
/// derive its cumulative population across every applicable schedule branch.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct WorkspaceExpertObservationSource {
    pub input_settlements: usize,
    pub input_control_bytes: usize,
    pub before: WorkspaceGroupedSourceRetention,
    pub after: WorkspaceGroupedSourceRetention,
    /// Derived from the selected Units worker, never from its input dtype.
    /// An idle source with no Units callback has no result witness.
    pub unit_dtype: Option<WorkspaceFloatingType>,
}
impl WorkspaceExpertObservationSource {
    pub fn validate(self)->Result<(),WorkspaceMetadataError>{
        self.before.source_loans().and_then(|n|n.checked_add(self.after.source_loans()?))
            .and_then(|n|n.checked_add(self.input_settlements)).ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
}
/// Borrowed exact cold source. `units` is a prospective layout from the native
/// Units producer, not a completed tensor. The envelope names the maximum-row
/// schedule branch only; native qualification must cover other branches too.
#[derive(Clone,Copy,Debug)]
pub struct WorkspaceExpertObservationView<'a> {
    pub region: WorkspaceExpertRegionView<'a>,
    pub envelope: WorkspaceGroupedObservationEnvelope,
    pub units: WorkspaceLayoutView<'a>,
}

pub(super) fn inspect(
    region:&WorkspaceExpertRegion,inputs:&[&WorkspaceTensor],context:&WorkspaceContext,
    observe:&mut dyn FnMut(WorkspaceExpertObservationView<'_>)->Result<WorkspaceExpertObservationSource,Error>,
)->Result<WorkspaceExpertObservationSource,Error>{
    context.charge_metadata(size_of::<(WorkspaceExpertObservationView<'_>,WorkspaceExpertObservationSource,
        Result<WorkspaceExpertObservationSource,Error>,WorkspaceOperationView<'_>,
        [WorkspaceLayout;4],Vec<WorkspaceLayout>,usize,i32,
        &mut dyn FnMut(WorkspaceExpertObservationView<'_>)->Result<WorkspaceExpertObservationSource,Error>)>())?;
    let view=region.as_view();
    let maximum=view.maximum_received_rows().ok_or(WorkspaceMetadataError::Overflow)?;
    let rows=i32::try_from(maximum).map_err(|_|WorkspaceMetadataError::Overflow)?;
    let unit_columns=match view.kernel {
        WorkspaceExpertKernel::Gated(v)=>v.intermediate_dimensions(),
        WorkspaceExpertKernel::Linear(v)=>v.output_dimensions(),
        WorkspaceExpertKernel::Relu2(v)=>v.intermediate_dimensions(),
    };
    if let Some(addressable)=view.addressable {
        context.charge_metadata(size_of::<(super::super::WorkspaceAddressableRegionView<'_>,
            super::super::WorkspaceAddressableObservationLayout,Result<super::super::WorkspaceAddressableObservationLayout,Error>,
            Vec<WorkspaceLayout>,[i32;2],usize)>())?;
        if inputs.len()<4{return Err(WorkspaceMetadataError::Unqualified.into());}
        let mut local=context.metadata_vec(4)?;
        for (index,input) in inputs[..4].iter().enumerate() {
            let shape=if index==0 {[rows,view.kernel.dimensions().0]}else{[rows,1]};
            let dtype=if index==1 {WorkspaceDtype::Int32}else{input.layout().dtype()};
            local.push(context.layout(&shape,dtype)?.with_representation(input.layout().representation()));
        }
        let physical=super::super::addressable_region::observation::layout(addressable,&local,context)?;
        let source=observe(WorkspaceExpertObservationView{region:view,envelope:physical.envelope,units:physical.units.as_view()})?;
        source.validate()?;
        if source.unit_dtype!=physical.units.representation().map(|value|value.dtype()) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        return Ok(source);
    }
    let schedule=context.mechanisms.grouped_observation_schedule(region.kernel(),
        u32::try_from(maximum).map_err(|_|WorkspaceMetadataError::Overflow)?)?
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    let envelope=WorkspaceGroupedObservationEnvelope::new(maximum,1,
        usize::try_from(unit_columns).map_err(|_|WorkspaceMetadataError::Unqualified)?,schedule,false)?;
    let mut units=context.layout(&[rows,unit_columns],WorkspaceDtype::Float32)?;
    if maximum!=0 {
        if inputs.len()<4{return Err(WorkspaceMetadataError::Unqualified.into());}
        let mut local=context.metadata_vec(inputs.len())?;
        for (index,input) in inputs.iter().enumerate(){
            let shape=if index==0 { Some([rows,view.kernel.dimensions().0]) }
                else if index<4 {Some([rows,1])} else {None};
            let dtype=if index==1 {WorkspaceDtype::Int32}else{input.layout().dtype()};
            local.push(context.layout(shape.as_ref().map_or(input.shape(),|s|s.as_slice()),dtype)?
                .with_representation(input.layout().representation()));
        }
        let mut outputs=context.metadata_vec(4)?;
        outputs.push(context.layout(units.shape(),units.dtype())?.with_representation(units.representation()));
        for _ in 0..3{outputs.push(context.layout(&[rows],WorkspaceDtype::Uint32)?);}
        let operation=WorkspaceOperationView {kind:WorkspaceOperationKindView::Grouped {
            bank:region.kernel(),phase:WorkspaceGroupedPhase::Units,partitions:view.tensor_partitions},
            inputs:WorkspaceLayoutList::Owned(&local),outputs:WorkspaceLayoutList::Owned(&outputs)};
        units=units.with_representation(context.mechanisms.output_representation(operation,0));
    }
    let source=observe(WorkspaceExpertObservationView {region:view,envelope,units:units.as_view()})?;
    source.validate()?;
    if source.unit_dtype!=units.representation().map(|v|v.dtype()) {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Facts;
    impl WorkspaceMechanisms for Facts {
        fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{Ok(None)}
        fn grouped_observation_schedule(&self,_:&WorkspaceGroupedBank,_:u32)
            ->Result<Option<WorkspaceGroupedObservationSchedule>,Error>{
            Ok(Some(WorkspaceGroupedObservationSchedule::WholeBatch))
        }
        fn output_representation(&self,operation:WorkspaceOperationView<'_>,output:usize)->Option<WorkspaceRepresentation>{
            assert!(matches!(operation.kind,WorkspaceOperationKindView::Grouped{phase:WorkspaceGroupedPhase::Units,..}));
            assert_eq!(output,0);
            assert_eq!(operation.inputs.get(0)?.shape(),[12,3]);
            assert_eq!(operation.inputs.get(1)?.shape(),[12,1]);
            assert_eq!(operation.inputs.get(4)?.shape(),[1,4,3]);
            Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true))
        }
    }
    #[test]
    fn prospective_expert_observer_uses_units_source_without_concrete_received_rows(){
        let context=WorkspaceContext::new(Facts);
        let projection=crate::GroupedProjectionSpec::new(crate::ParameterSpec::trainable("linear").unwrap(),None,
            crate::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
        let spec=crate::GroupedLinearSpec::new(1,3,4,crate::GroupedLinearActivation::Identity,projection).unwrap();
        let view=WorkspaceExpertRegionView{bank:7,unit:2,prefill:true,group:eredu_core::CollectiveGroupId::new(1),
            rank:0,peers:2,source_rows:3,routes_per_row:2,owners:&[0,1],owner_local:&[0,0],
            kernel:WorkspaceExpertKernel::Linear(&spec),tensor_partitions:None,
            provider_tensor_group:None,provider_wave_group:None,addressable:None,
            movement:WorkspaceExpertMovementPopulation{row_gathers:1,scalar_gathers:2,zeros:1,indexed_adds:1},
            transfers:WorkspaceExpertTransfers([None;9])};
        let region=view.retain(&context).unwrap();
        let floating=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let values=[floating(&[3,3]),WorkspaceTensor::existing(context.layout(&[3,2],WorkspaceDtype::Int32).unwrap(),&context).unwrap(),
            floating(&[3,2]),floating(&[3,2]),floating(&[1,4,3])];
        let inputs=values.iter().collect::<Vec<_>>();
        context.begin_span();
        let mut calls=0;
        let selected=inspect(&region,&inputs,&context,&mut |source|{
            calls+=1;
            assert_eq!(source.region.source_rows,3);
            assert_eq!(source.envelope.maximum_provider_rows(),12);
            assert_eq!(source.envelope.refine(5).unwrap().selected_rows(),5);
            assert_eq!(source.units.shape(),[12,4]);
            Ok(WorkspaceExpertObservationSource{unit_dtype:source.units.representation().map(|v|v.dtype()),
                before:WorkspaceGroupedSourceRetention{partition_copies:2,..Default::default()},..Default::default()})
        }).unwrap();
        assert_eq!(calls,1);
        assert_eq!(selected.unit_dtype,Some(WorkspaceFloatingType::Float16));
        assert_eq!(selected.before.source_loans(),Some(10));
        assert!(context.finish_report(&[]).unwrap().operations.is_empty());
        assert!(inspect(&region,&inputs,&context,&mut |_|Ok(WorkspaceExpertObservationSource{
            unit_dtype:Some(WorkspaceFloatingType::Float32),..Default::default()})).is_err());
    }
    #[derive(Debug)]
    struct AddressableFacts;
    impl WorkspaceMechanisms for AddressableFacts {
        fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{Ok(None)}
        fn output_representation(&self,operation:WorkspaceOperationView<'_>,_:usize)->Option<WorkspaceRepresentation>{
            assert!(!matches!(operation.kind,WorkspaceOperationKindView::Grouped{..}),
                "cached local observation must not inspect resident parameter placeholders");
            None
        }
        fn addressable_observation_layout(&self,source:super::super::super::WorkspaceAddressableRegionView<'_>,
            inputs:&[WorkspaceLayout],context:&WorkspaceContext)
            ->Result<Option<super::super::super::WorkspaceAddressableObservationLayout>,Error>{
            assert_eq!(inputs.len(),4);assert_eq!(inputs[0].shape(),[12,3]);
            assert_eq!(inputs[1].shape(),[12,1]);assert_eq!(source.local_members,Some([0].as_slice()));
            Ok(Some(super::super::super::WorkspaceAddressableObservationLayout {
                envelope:WorkspaceGroupedObservationEnvelope::new(3,1,4,WorkspaceGroupedObservationSchedule::WholeBatch,false)?,
                units:context.layout(&[3,4],WorkspaceDtype::Float32)?.with_representation(Some(
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true))),
            }))
        }
    }
    struct NoResidentWeights;
    impl crate::Parameterized<WorkspaceTensor> for NoResidentWeights {
        fn visit_parameter_sources<'a,V>(&'a self,_:&mut V)->Result<(),crate::ParameterSourceError>
        where V:crate::ParameterSourceVisitor<'a,WorkspaceTensor> {panic!("independent source cannot read resident weights")}
        fn visit_parameters_mut<'a,V>(&'a mut self,_:&mut V) where V:crate::ParameterVisitorMut<'a,WorkspaceTensor> {unreachable!()}
        fn set_trainable(&mut self,_:bool){unreachable!()}
        fn retained_value_slot_bound(&self)->Option<usize>{panic!("independent source has no resident slot census")}
    }
    #[test]
    fn exchanged_addressable_observer_uses_actual_compact_units_source() {
        let context=WorkspaceContext::new(AddressableFacts);
        let projection=crate::GroupedProjectionSpec::new(crate::ParameterSpec::trainable("linear").unwrap(),None,
            crate::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
        let spec=crate::GroupedLinearSpec::new(1,3,4,crate::GroupedLinearActivation::Identity,projection).unwrap();
        let addressable=super::super::super::WorkspaceAddressableRegionView {owner_group:"layers",bank:7,unit:2,prefill:true,
            chunks:super::super::super::WorkspaceAddressableChunkPlan{rows:12,chunk_rows:3,routes:1,members:1},
            local_members:Some(&[0]),kernel:WorkspaceExpertKernel::Linear(&spec),tensor_partitions:None,
            compact_scratch_bytes:256,bulk_target_bytes:256,callback_control_bytes:64};
        let view=WorkspaceExpertRegionView{bank:7,unit:2,prefill:true,group:eredu_core::CollectiveGroupId::new(1),
            rank:0,peers:2,source_rows:3,routes_per_row:2,owners:&[0,1],owner_local:&[0,0],
            kernel:WorkspaceExpertKernel::Linear(&spec),tensor_partitions:None,addressable:Some(addressable),
            provider_tensor_group:None,provider_wave_group:None,
            movement:WorkspaceExpertMovementPopulation{row_gathers:1,scalar_gathers:2,zeros:1,indexed_adds:1},
            transfers:WorkspaceExpertTransfers([None;9])};
        let region=view.retain(&context).unwrap();
        let floating=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let values=[floating(&[3,3]),WorkspaceTensor::existing(context.layout(&[3,2],WorkspaceDtype::Int32).unwrap(),&context).unwrap(),
            floating(&[3,2]),floating(&[3,2])];
        let inputs=values.iter().collect::<Vec<_>>();context.begin_span();let mut calls=0;
        let selected=inspect(&region,&inputs,&context,&mut |source| {
            calls+=1;assert_eq!(source.region.addressable,Some(addressable));
            assert_eq!(source.envelope.maximum_provider_rows(),3);assert_eq!(source.units.shape(),[3,4]);
            Ok(WorkspaceExpertObservationSource {input_settlements:2,input_control_bytes:32,
                before:WorkspaceGroupedSourceRetention{partition_copies:1,..Default::default()},
                unit_dtype:source.units.representation().map(|value|value.dtype()),..Default::default()})
        }).unwrap();
        assert_eq!(calls,1);assert_eq!(selected.input_settlements,2);
        assert_eq!(selected.unit_dtype,Some(WorkspaceFloatingType::Float16));
        assert!(context.finish_report(&[]).unwrap().operations.is_empty());
        assert!(inspect(&region,&inputs,&context,&mut |_|Ok(WorkspaceExpertObservationSource {
            unit_dtype:Some(WorkspaceFloatingType::Float32),..Default::default()})).is_err());
        context.begin_span();
        let routes=crate::GroupSelection::new(values[1].clone(),values[2].clone(),values[3].clone());
        let (output,bias)=record_expert_region(view,&NoResidentWeights,&values[0],&routes,&context).unwrap().into_parts();
        assert!(bias.is_none());
        let report=context.finish_report(&[output]).unwrap();
        let mut regions=report.operations.iter().filter(|operation|
            matches!(operation.as_view().kind,WorkspaceOperationKindView::ExpertRegion(_)));
        let operation=regions.next().unwrap();
        assert!(regions.next().is_none());
        assert_eq!(operation.inputs.len(),4);
        assert!(matches!(operation.as_view().kind,WorkspaceOperationKindView::ExpertRegion(region)
            if region.as_view().addressable==Some(addressable)));
    }

}
