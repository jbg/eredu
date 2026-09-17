//! Actual compact-row source for prospective sparse member callbacks.
use super::*;
use super::parameters::ParameterRows;
use crate::backend::runtime::residency::parameter_bank::IndexedBankSource;

pub(super) fn source_groups(source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],
    mechanism:ResidentExecutionMechanisms,funding:&WorkspaceMetadataFunding,owner:&WorkspaceContext)
    ->Result<WorkspaceLayout,Error>{
    let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone())?;
    let value=inputs.get(1).ok_or(WorkspaceMetadataError::Unqualified)?;
    let value=WorkspaceTensor::existing(context.layout(value.shape(),value.dtype())?
        .with_representation(value.representation()),&context)?;
    let value=value.reshape(&[i32::try_from(source.chunks.rows).map_err(|_|WorkspaceMetadataError::Overflow)?,
        i32::try_from(source.chunks.routes).map_err(|_|WorkspaceMetadataError::Overflow)?],&context)?;
    Ok(owner.layout(value.shape(),value.layout().dtype())?.with_representation(value.layout().representation()))
}

pub(super) fn inspect(bank:&IndexedBankSource,source:WorkspaceAddressableRegionView<'_>,
    inputs:&[WorkspaceLayout],mechanism:ResidentExecutionMechanisms,funding:&WorkspaceMetadataFunding)
    ->Result<WorkspaceAddressableObservationLayout,Error>{
    let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone())?;
    context.charge_metadata(size_of::<(ParameterRows,AddressableChildSource,WorkspaceAddressableObservationLayout,
        Vec<WorkspaceLayout>,WorkspaceLayout,Result<WorkspaceAddressableObservationLayout,Error>)>())?;
    source.validate()?;
    let parameters=ParameterRows::prepare(bank,source,mechanism,&context)?;
    // Every possible selected row must retain the same field geometry/dtype.
    // The quote retains actual replacement identity and enumerates cardinalities.
    for member in 1..source.chunks.members {for field in 0..parameters.fields {
        let a=&parameters.rows[member*parameters.fields+field].layout;let b=&parameters.rows[field].layout;
        if a!=b||a.representation()!=b.representation(){return Err(WorkspaceMetadataError::Unqualified.into());}
    }}
    let rows=source.chunks.rows.min(source.chunks.chunk_rows);
    if rows==0{return Err(WorkspaceMetadataError::Unqualified.into());}
    let members=rows.checked_mul(source.chunks.routes).ok_or(WorkspaceMetadataError::Overflow)?.min(source.chunks.members);
    let local=super::quote::normalized_inputs(source,inputs,rows,mechanism,funding,&context)?;
    let groups=source_groups(source,inputs,mechanism,funding,&context)?;
    let mut fields=context.metadata_vec(members.checked_mul(parameters.fields).ok_or(WorkspaceMetadataError::Overflow)?)?;
    for row in &parameters.rows[..members*parameters.fields]{fields.push(context.layout(row.layout.shape(),row.layout.dtype())?
        .with_representation(row.layout.representation()));}
    let child=AddressableChildSource::prepare_with_observation(source,&local,members,&fields,mechanism,funding,
        Some(WorkspaceAddressableObservationSource::default()),Some(&groups))?;
    let representation=child.unit_representation.ok_or(WorkspaceMetadataError::Unqualified)?;
    let compact=source.kernel.compact(i32::try_from(members).map_err(|_|WorkspaceMetadataError::Overflow)?,&context)?;
    let schedule=mechanism.grouped_observation_schedule(&compact,u32::try_from(rows).map_err(|_|WorkspaceMetadataError::Overflow)?)?
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    let width=match source.kernel {WorkspaceExpertKernel::Gated(v)=>v.intermediate_dimensions(),
        WorkspaceExpertKernel::Linear(v)=>v.output_dimensions(),WorkspaceExpertKernel::Relu2(v)=>v.intermediate_dimensions()};
    let envelope=WorkspaceGroupedObservationEnvelope::new(rows,source.chunks.routes,
        usize::try_from(width).map_err(|_|WorkspaceMetadataError::Unqualified)?,schedule,false)?;
    let selected=rows.checked_mul(source.chunks.routes).ok_or(WorkspaceMetadataError::Overflow)?;
    Ok(WorkspaceAddressableObservationLayout{envelope,units:context.layout(&[
        i32::try_from(selected).map_err(|_|WorkspaceMetadataError::Overflow)?,width],WorkspaceDtype::Float32)?
        .with_representation(Some(representation))})
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests{
    use super::*;
    use std::sync::atomic::{AtomicUsize,Ordering};
    #[derive(Debug,Default)]struct Account(AtomicUsize);
    impl WorkspaceMetadataAccount for Account{
        fn reserve_metadata(&self,bytes:usize)->Result<(),WorkspaceMetadataFundingError>{
            self.0.fetch_update(Ordering::SeqCst,Ordering::SeqCst,|total|total.checked_add(bytes))
                .map(|_|()).map_err(|_|WorkspaceMetadataFundingError::Overflow)
        }
    }
    fn projection(name:&str)->eredu_nn::GroupedProjectionSpec{
        eredu_nn::GroupedProjectionSpec::new(eredu_nn::ParameterSpec::trainable(name).unwrap(),None,
            eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap()
    }
    #[test]
    fn addressable_child_keeps_full_route_source_across_compact_cardinalities(){
        let _sources=crate::tests::support::test_utils::initialize_original_sources();
        let mechanism=ResidentExecutionMechanisms::Metal(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let funding=WorkspaceMetadataFunding::new(Account::default()).unwrap();
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone()).unwrap();
        let spec=eredu_nn::GroupedGatedProductSpec::new(2,32,32,32,eredu_nn::GatedProductPolicy::ordinary_silu(),
            eredu_nn::GatedProductGroupLayout::Packed{gate_up:projection("read"),down:projection("write")}).unwrap();
        let source=WorkspaceAddressableRegionView{owner_group:"layers",bank:0,unit:0,prefill:true,
            chunks:WorkspaceAddressableChunkPlan{rows:129,chunk_rows:65,routes:2,members:2},local_members:None,
            kernel:WorkspaceExpertKernel::Gated(&spec),tensor_partitions:None,compact_scratch_bytes:0,bulk_target_bytes:0,callback_control_bytes:0};
        let floating=|shape:&[i32]|context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let full=context.layout(&[129,2],WorkspaceDtype::Int32).unwrap();
        let per_callback=crate::backend::array_copy::CaptureNativePopulation::routed_partition().unwrap();
        let retention=WorkspaceGroupedSourceRetention{partition_copies:1,replica_settlements:0,callback_control_bytes:per_callback.controls};
        for rows in [1,65]{for members in [1,2]{
            let inputs=[floating(&[rows,32]),context.layout(&[rows,2],WorkspaceDtype::Int32).unwrap(),floating(&[rows,2]),floating(&[rows,2])];
            let mut fields=Vec::new();for _ in 0..members{fields.extend([floating(&[1,64,32]),floating(&[1,32,32])]);}
            let baseline=AddressableChildSource::prepare_with_observation(source,&inputs,members,&fields,mechanism,&funding,
                Some(WorkspaceAddressableObservationSource::default()),Some(&full)).unwrap();
            let observed=AddressableChildSource::prepare_with_observation(source,&inputs,members,&fields,mechanism,&funding,
                Some(WorkspaceAddressableObservationSource{before:retention,after:retention,
                    unit_dtype:baseline.unit_representation.map(|v|v.dtype())}),Some(&full)).unwrap();
            let bank=source.kernel.compact(members as i32,&context).unwrap();
            let schedule=mechanism.grouped_observation_schedule(&bank,rows as u32).unwrap().unwrap();
            let envelope=WorkspaceGroupedObservationEnvelope::new(rows as usize,2,32,schedule,false).unwrap();
            assert_eq!(observed.capture.retained_roots,10*envelope.maximum_callbacks());
            assert_eq!(observed.capture.publications,observed.capture.retained_roots);
            let recipe=SpeculativeNumericalRecipe::inspect_owned_child_with_capture(&observed.report,1,mechanism,&context,observed.capture).unwrap();
            let base=SpeculativeNumericalRecipe::inspect_owned_child_with_capture(&baseline.report,1,mechanism,&context,baseline.capture).unwrap();
            assert_eq!(recipe.completion.nested_completions,base.completion.nested_completions+observed.capture.completions);
            let truncated=context.layout(&[rows,2],WorkspaceDtype::Int32).unwrap();
            assert!(AddressableChildSource::prepare_with_observation(source,&inputs,members,&fields,mechanism,&funding,
                Some(WorkspaceAddressableObservationSource::default()),Some(&truncated)).is_err());
        }}
    }
    #[test]
    fn addressable_sequential_child_qualifies_the_actual_compact_reduction() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = ResidentExecutionMechanisms::Metal(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let funding = WorkspaceMetadataFunding::new(Account::default()).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone()).unwrap();
        let spec = eredu_nn::GroupedGatedProductSpec::new(2, 16, 6, 16,
            eredu_nn::GatedProductPolicy::ordinary_silu(),
            eredu_nn::GatedProductGroupLayout::Packed {gate_up:projection("read"),down:projection("write")})
            .unwrap().with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder);
        // Match the actual retained source precision without asserting dense
        // strides that the loaded parameter/source loan does not supply.
        let floating = |shape:&[i32]| context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)));
        for (rows,routes,members) in [(1,1,1usize),(2,2,1),(65,2,2)] {
            let source = WorkspaceAddressableRegionView {owner_group:"layers",bank:0,unit:0,prefill:true,
                chunks:WorkspaceAddressableChunkPlan {rows:rows as usize,chunk_rows:rows as usize,routes:routes as usize,members:2},
                local_members:None,kernel:WorkspaceExpertKernel::Gated(&spec),tensor_partitions:None,
                compact_scratch_bytes:0,bulk_target_bytes:0,callback_control_bytes:0};
            let groups=context.layout(&[rows,routes],WorkspaceDtype::Int32).unwrap();
            let inputs=[floating(&[rows,16]),groups.clone(),floating(&[rows,routes]),floating(&[rows,routes])];
            let mut parameters=Vec::new();
            for _ in 0..members {parameters.extend([floating(&[1,12,16]),floating(&[1,16,6])]);}
            for observation in [None,Some(WorkspaceAddressableObservationSource::default())] {
                let child=AddressableChildSource::prepare_with_observation(source,&inputs,members,&parameters,
                    mechanism,&funding,observation,observation.map(|_|&groups)).unwrap();
                assert!(child.report.operations.iter().any(|op| matches!(&op.kind,
                    WorkspaceOperationKind::Grouped {bank,..}
                        if matches!(bank.as_ref(),WorkspaceGroupedBank::GatedProduct(value)
                            if value.group_count()==members as i32 && value.reduction()==eredu_nn::GroupReduction::SequentialGroupOrder))));
                let recipe=SpeculativeNumericalRecipe::inspect_owned_child_with_capture(&child.report,1,
                    mechanism,&context,child.capture).unwrap();
                assert!(recipe.graph_capacity>0 && recipe.record_capacity>0 && recipe.kernels>0);
                assert!(recipe.completion.nested_completions>0);
            }
        }
    }

}
