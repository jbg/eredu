#[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
#[test]
fn projected_decode_source_shares_host_geometry_and_preserves_raw_additive_reductions(){
    use super::*;
    use eredu_core::*;
    use eredu_nn::workspace::*;
    use eredu_runtime::capture::partition::{PartitionCaptureProducer,PartitionCaptureReceiptLimits};
    use eredu_runtime::working_memory::PartitionFragmentHostPlan;
    use crate::backend::nn::workspace::{MlxCpuMatmulMechanism,MlxCpuWorkspaceMechanisms,MlxMetalWorkspaceMechanisms};
    let native=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(native.allocation(),selected);
    for transform in [CaptureTransform::Slice,CaptureTransform::Preview{max_elements:5},CaptureTransform::Summary,
        CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]}] {for sum in [false,true] {
        let source=source(transform.clone());
        let slice=ResolvedCaptureSlice{starts:vec![0,0,1],ends:vec![2,1,7],strides:vec![1,1,2],shape:vec![2,1,3]};
        let combination=if sum{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
        let producer=|rank,range|PartitionCaptureProducer{rank,projection:CaptureContiguousProjectionPlan::prepare(&[2,1,7],&slice,2,range,1).unwrap().construct()};
        let producers=if sum{vec![producer(0,0..7),producer(1,0..7)]}else{vec![producer(0,3..7),producer(1,0..3)]};
        let context=PartitionCaptureContext{artifact_identity:"fixture".into(),execution_identity:"retained".into(),run_identity:"source".into(),overlay_identity:None,
            capture_plan_identity:source.admission().identity().into(),selection_index:0,phase:CapturePhase::Decode,prediction:2,forward_epoch:8,invocation:None};
        let mut ledger=CaptureLedger::new(source.admission());ledger.begin_step();
        let limits=PartitionCaptureReceiptLimits{max_producers:2,max_fragments:2,max_record_bytes:64<<10};
        let receipt=if sum{PartitionCaptureReceiptPlan::new_sum_shared(source.clone(),context,producers,2,limits,&mut ledger)}
            else{PartitionCaptureReceiptPlan::new_shared(source.clone(),context,producers,2,limits,&mut ledger)}.unwrap();
        let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();
        assert_eq!(host.fragment_count(),2);assert!(host.fragment_peak_bytes()>0);
        assert_eq!(host.assembly_peak_bytes()>0,sum&&matches!(transform,CaptureTransform::Summary|CaptureTransform::Histogram{..}));
        let context=WorkspaceContext::new(cpu);
        let equation=PartitionInvocationEquation::prepare(&receipt,0,0,&context).unwrap();
        let native=equation.native_source(eredu_core::checkpoint::TensorDtype::F32);
        assert_eq!(native.local_shape,if sum{&[2,1,7][..]}else{&[2,1,4][..]});
        assert_eq!(native.transform,if sum&&matches!(transform,CaptureTransform::Summary|CaptureTransform::Histogram{..}){&CaptureTransform::Slice}else{&transform});
        assert!(native.estimate.capture.retained_bytes>0);assert_eq!(native.estimate.generated_creation_bytes,0);
        assert!(std::ptr::eq(equation.source.admission(),source.admission()));
        assert!(std::ptr::eq(equation.source.projection(),receipt.producer(0).unwrap()));
        let scalar=WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true);
        let value=WorkspaceTensor::existing(context.layout(&[2,1,if sum{7}else{4}],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(scalar)),&context).unwrap();
        let mut roots=vec![];let (dtype,population)=equation.trace(&value,&context,&mut roots).unwrap();
        assert_eq!(dtype,WorkspaceFloatingType::Float32);assert!(population.publications>0);assert!(population.completions>0);
        let report=context.report(&roots).unwrap();assert!(report.unpriced_operations.is_empty());assert!(!report.operations.is_empty());
        // The actual observer dispatch derives the same local equation and
        // records the scalar before handing it to the late Source vote.
        let context2=WorkspaceContext::new(cpu);
        let value2=WorkspaceTensor::existing(context2.layout(&[2,1,if sum{7}else{4}],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(scalar)),&context2).unwrap();
        let transfers=Cell::new(CaptureNativePopulation::default());let scalars=[Cell::new(None)];let mut roots2=vec![];
        super::super::observer::trace_invocation(&source,0,CapturePhase::Decode,2,receipt.producer(0).unwrap(),
            combination,0,true,&value2,&context2,&mut roots2,Some(&transfers),Some(&scalars)).unwrap();
        assert_eq!(scalars[0].get(),Some(WorkspaceFloatingType::Float32));let observed=transfers.get();
        assert_eq!((observed.publications,observed.completions,observed.controls,observed.retained_roots),
            (population.publications,population.completions,population.controls,population.retained_roots));
        assert!(context2.report(&roots2).unwrap().unpriced_operations.is_empty());

        // No scalar witness or wrong local physical rectangle is refused before
        // appending any source/intermediate to the parent's retained roots.
        let before=roots.len();
        let opaque=WorkspaceTensor::existing(context.layout(&[2,1,if sum{7}else{4}],WorkspaceDtype::Float32).unwrap(),&context).unwrap();
        assert!(equation.trace(&opaque,&context,&mut roots).is_err());assert_eq!(roots.len(),before);
        let wrong=WorkspaceTensor::existing(context.layout(&[2,2,if sum{7}else{4}],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(scalar)),&context).unwrap();
        assert!(equation.trace(&wrong,&context,&mut roots).is_err());assert_eq!(roots.len(),before);
        assert!(PartitionInvocationCaptureGeometry::prepare(source.admission(),0,CapturePhase::Prefill,0,
            receipt.producer(0).unwrap(),0,combination).is_err());
        assert!(PartitionInvocationCaptureGeometry::from_receipt(&receipt,0,1).is_err());
        assert!(PartitionInvocationCaptureGeometry::from_receipt(&receipt,9,0).is_err());
    }}
}

#[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
fn source(transform:eredu_core::capture::CaptureTransform)->eredu_core::capture::SharedCapturePlan {
    use eredu_core::*;
    use eredu_core::capture::*;
    let point=ObservationPoint{path:"block.output".into(),node_id:"block".into(),meaning:"projected decode activation".into(),
        value_type:ObservationValueType::Tensor,dtype:ObservationDtype::Floating,
        axes:Some([SymbolicDimension::Known(2),SymbolicDimension::Sequence,SymbolicDimension::Known(7)].into_iter().enumerate()
            .map(|(i,dimension)|TensorAxis{name:format!("axis{i}"),dimension}).collect()),prefill:true,decode:true,
        requirements:vec![ObservationRequirement::ActivationHooks],position:ObservationPosition::BeforeIntervention,retained_bytes:None,host_bytes:None};
    let support=ObservationSupportReport{schema_version:1,capture:Default::default(),points:vec![ObservationSupport{path:point.path.clone(),
        prefill:ObservationSupportStatus::Supported,decode:ObservationSupportStatus::Supported,floating_to_f32:true}]};
    let catalog=ObservationCatalog{schema_version:1,points:vec![point],completeness:DescriptionCompleteness::Complete};
    let maximum=CaptureUsage{captures:u64::MAX,retained_bytes:u64::MAX,host_bytes:u64::MAX,encoded_bytes:u64::MAX};
    let declaration=CapturePlan{schema_version:1,selections:vec![CaptureSelection{id:"projected decode".into(),path:"block.output".into(),
        schedule:CaptureSchedule{prefill:false,decode:true,..Default::default()},transform,
        slices:vec![CaptureSlice{axis:"axis2".into(),start:1,end:7,stride:2}]}],
        limits:CaptureLimits{per_step:maximum,cumulative:maximum,physical_native_bytes:None,on_limit:CaptureLimitPolicy::Fail}};
    let caps=CaptureCapabilities{transformations:vec![CaptureTransformKind::Slice,CaptureTransformKind::Preview,CaptureTransformKind::Summary,CaptureTransformKind::Histogram],
        max_histogram_bins:2,physical_native_limit:false,conditions:vec![]};
    SharedCapturePlan::new(declaration.admit_with_text_origin(&catalog,&support,&caps,
        CaptureRequestShape{batch:1,prompt_tokens:3,max_predictions:4},CaptureTextOrigin{cached_positions:2}).unwrap())
}
