//! Borrowed addressable source: original members and chunk policy, no exchanges.
use super::*;

/// Exact per-delivered-batch retention selected by an observer. Input invocation
/// settlement remains in the enclosing ordinary begin/finish callback.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct WorkspaceAddressableObservationSource {
    pub before:WorkspaceGroupedSourceRetention,
    pub after:WorkspaceGroupedSourceRetention,
    pub unit_dtype:Option<WorkspaceFloatingType>,
}
impl WorkspaceAddressableObservationSource {
    pub fn validate(self)->Result<(),WorkspaceMetadataError>{
        self.before.source_loans().and_then(|n|n.checked_add(self.after.source_loans()?))
            .ok_or(WorkspaceMetadataError::Overflow)?;Ok(())
    }
}
/// Native source projection for the maximum full-chunk branch. Its physical
/// layout must come from actual retained parameter rows and the Units worker.
/// Other row/cardinality branches remain the native quote's responsibility.
#[derive(Debug)]
pub struct WorkspaceAddressableObservationLayout {
    pub envelope:WorkspaceGroupedObservationEnvelope,
    pub units:WorkspaceLayout,
}
#[derive(Clone,Copy,Debug)]
pub struct WorkspaceAddressableObservationView<'a> {
    pub region:WorkspaceAddressableRegionView<'a>,
    pub envelope:WorkspaceGroupedObservationEnvelope,
    pub units:WorkspaceLayoutView<'a>,
    /// Exact original scalar map supplied by the existing partition adapter.
    pub unit_coordinates:Option<&'a eredu_core::component::ComponentCoordinateMap>,
}
pub(super) fn inspect(source:WorkspaceAddressableRegionView<'_>,inputs:&[&WorkspaceTensor],
    context:&WorkspaceContext,
    observe:&mut dyn FnMut(WorkspaceAddressableObservationView<'_>)->Result<WorkspaceAddressableObservationSource,Error>)
    ->Result<WorkspaceAddressableObservationSource,Error> {
    context.charge_metadata(size_of::<(WorkspaceAddressableObservationLayout,WorkspaceAddressableObservationView<'_>,
        WorkspaceAddressableObservationSource,Result<WorkspaceAddressableObservationSource,Error>,Vec<WorkspaceLayout>)>())?;
    let mut layouts=context.metadata_vec(inputs.len())?;
    for input in inputs {layouts.push(context.layout(input.shape(),input.layout().dtype())?
        .with_representation(input.layout().representation()));}
    let value=layout(source,&layouts,context)?;
    let result=observe(WorkspaceAddressableObservationView{region:source,envelope:value.envelope,
        units:value.units.as_view(),unit_coordinates:None})?;
    result.validate()?;
    if result.unit_dtype!=value.units.representation().map(|v|v.dtype()){
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    Ok(result)
}

/// Shared physical Units source for direct and expert-exchanged local providers.
pub(in crate::workspace) fn layout(source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],
    context:&WorkspaceContext)->Result<WorkspaceAddressableObservationLayout,Error> {
    context.charge_metadata(size_of::<(WorkspaceAddressableRegionView<'_>,&[WorkspaceLayout],&WorkspaceContext,
        Option<WorkspaceAddressableObservationLayout>,WorkspaceAddressableObservationLayout,
        Result<WorkspaceAddressableObservationLayout,Error>,WorkspaceGroupedBank,usize,i32,
        WorkspaceGroupedObservationSchedule,WorkspaceGroupedObservationEnvelope)>())?;
    let native=context.mechanisms.addressable_observation_layout(source,inputs,context)?;
    let value=match native {
        Some(value)=>value,
        None=>{
            // Preliminary portable geometry carries no physical witness. The
            // native source must later replace this absence from actual rows.
            let bank=source.kernel.retain(context)?;
            let maximum=source.chunks.rows.min(source.chunks.chunk_rows);
            let schedule=context.mechanisms.grouped_observation_schedule(&bank,
                u32::try_from(maximum).map_err(|_|WorkspaceMetadataError::Overflow)?)?
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let width=match source.kernel {WorkspaceExpertKernel::Gated(v)=>v.intermediate_dimensions(),
                WorkspaceExpertKernel::Linear(v)=>v.output_dimensions(),WorkspaceExpertKernel::Relu2(v)=>v.intermediate_dimensions()};
            let envelope=WorkspaceGroupedObservationEnvelope::new(maximum,source.chunks.routes,
                usize::try_from(width).map_err(|_|WorkspaceMetadataError::Unqualified)?,schedule,false)?;
            let rows=maximum.checked_mul(source.chunks.routes).ok_or(WorkspaceMetadataError::Overflow)?;
            WorkspaceAddressableObservationLayout{envelope,units:context.layout(&[
                i32::try_from(rows).map_err(|_|WorkspaceMetadataError::Overflow)?,width],WorkspaceDtype::Float32)?}
        }
    };
    let unit_columns=match source.kernel {WorkspaceExpertKernel::Gated(v)=>v.intermediate_dimensions(),
        WorkspaceExpertKernel::Linear(v)=>v.output_dimensions(),WorkspaceExpertKernel::Relu2(v)=>v.intermediate_dimensions()};
    let selected=value.envelope.maximum_provider_rows().checked_mul(source.chunks.routes)
        .and_then(|v|i32::try_from(v).ok()).ok_or(WorkspaceMetadataError::Overflow)?;
    if value.envelope.maximum_provider_rows()!=source.chunks.rows.min(source.chunks.chunk_rows)
        || value.envelope.routes_per_row()!=source.chunks.routes
        || usize::try_from(unit_columns).ok()!=Some(value.envelope.unit_columns())
        || value.units.shape()!=[selected,unit_columns]||value.units.dtype()!=WorkspaceDtype::Float32 {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[derive(Debug)]struct Facts;
    impl WorkspaceMechanisms for Facts{
        fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{Ok(None)}
        fn addressable_observation_layout(&self,source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],context:&WorkspaceContext)
            ->Result<Option<WorkspaceAddressableObservationLayout>,Error>{
            assert_eq!(source.local_members,Some([2,4,6,8].as_slice()));
            assert_eq!(inputs[1].shape(),[5,2]);
            Ok(Some(WorkspaceAddressableObservationLayout{
                envelope:WorkspaceGroupedObservationEnvelope::new(3,2,4,WorkspaceGroupedObservationSchedule::WholeBatch,false)?,
                units:context.layout(&[6,4],WorkspaceDtype::Float32)?.with_representation(Some(
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true))),
            }))
        }
    }
    #[test]
    fn addressable_observer_borrows_original_members_without_selected_id_values(){
        let context=WorkspaceContext::new(Facts);
        let projection=crate::GroupedProjectionSpec::new(crate::ParameterSpec::trainable("linear").unwrap(),None,
            crate::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
        let spec=crate::GroupedLinearSpec::new(4,3,4,crate::GroupedLinearActivation::Identity,projection).unwrap();
        let source=WorkspaceAddressableRegionView{owner_group:"layers",bank:7,unit:2,prefill:true,
            chunks:WorkspaceAddressableChunkPlan{rows:5,chunk_rows:3,routes:2,members:4},local_members:Some(&[2,4,6,8]),
            kernel:WorkspaceExpertKernel::Linear(&spec),tensor_partitions:None,compact_scratch_bytes:0,
            bulk_target_bytes:0,callback_control_bytes:0};
        let floating=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let values=[floating(&[5,3]),WorkspaceTensor::existing(context.layout(&[5,2],WorkspaceDtype::Int32).unwrap(),&context).unwrap(),
            floating(&[5,2]),floating(&[5,2])];
        let inputs=values.iter().collect::<Vec<_>>();context.begin_span();let mut calls=0;
        let observed=inspect(source,&inputs,&context,&mut |view|{
            calls+=1;assert_eq!(view.region.chunks.range(1),Some(3..5));
            assert_eq!(view.region.local_members,source.local_members);
            assert_eq!(view.units.shape(),[6,4]);assert_eq!(view.envelope.refine(2).unwrap().selected_rows(),4);
            assert!(view.unit_coordinates.is_none());
            Ok(WorkspaceAddressableObservationSource{unit_dtype:view.units.representation().map(|v|v.dtype()),
                before:WorkspaceGroupedSourceRetention{partition_copies:2,..Default::default()},..Default::default()})
        }).unwrap();
        assert_eq!(calls,1);assert_eq!(observed.before.source_loans(),Some(10));
        assert_eq!(observed.unit_dtype,Some(WorkspaceFloatingType::Float16));
        assert!(context.finish_report(&[]).unwrap().operations.is_empty());
        assert!(inspect(source,&inputs,&context,&mut |_|Ok(WorkspaceAddressableObservationSource{
            unit_dtype:Some(WorkspaceFloatingType::Float32),..Default::default()})).is_err());
    }
}
