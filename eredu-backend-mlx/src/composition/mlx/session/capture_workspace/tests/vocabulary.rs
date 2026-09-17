#[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
#[test]
fn complete_vocabulary_cold_quotes_preserve_source_and_only_the_actual_reducer() {
    use super::*;
    use crate::backend::nn::workspace::{MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms,MlxCpuMatmulMechanism};
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    for scores in [false,true] {
        let point=ObservationPoint {path:MODEL_LOGITS_OBSERVATION_PATH.into(),node_id:"output".into(),meaning:"actual raw logits".into(),
            value_type:ObservationValueType::Tensor,dtype:ObservationDtype::Floating,
            axes:Some([SymbolicDimension::Batch,SymbolicDimension::Sequence,SymbolicDimension::Known(4)]
                .into_iter().enumerate().map(|(i,dimension)|TensorAxis {name:format!("axis{i}"),dimension}).collect()),
            prefill:true,decode:true,requirements:vec![ObservationRequirement::ActivationHooks],
            position:ObservationPosition::BeforeIntervention,retained_bytes:None,host_bytes:None};
        let catalog=ObservationCatalog {schema_version:1,points:vec![point],completeness:DescriptionCompleteness::Complete};
        let support=ObservationSupportReport {schema_version:1,capture:Default::default(),points:vec![ObservationSupport {
            path:MODEL_LOGITS_OBSERVATION_PATH.into(),prefill:ObservationSupportStatus::Supported,
            decode:ObservationSupportStatus::Supported,floating_to_f32:true}]};
        let mut raw=CapturePlan::none();raw.selections.push(CaptureSelection {id:"readout".into(),path:MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule:CaptureSchedule {prefill:false,..Default::default()},slices:vec![],
            transform:if scores {CaptureTransform::TokenScores {token_ids:vec![3,1]}}else{CaptureTransform::TopCandidates {count:2}}});
        let usage=CaptureUsage {captures:u64::MAX,retained_bytes:u64::MAX,host_bytes:u64::MAX,encoded_bytes:u64::MAX};
        raw.limits.per_step=usage;raw.limits.cumulative=usage;
        let source=SharedCapturePlan::new(raw.admit(&catalog,&support,&CaptureCapabilities {
            transformations:vec![CaptureTransformKind::TopCandidates,CaptureTransformKind::TokenScores],max_histogram_bins:0,
            physical_native_limit:false,conditions:vec![]},CaptureRequestShape {batch:1,prompt_tokens:3,max_predictions:4}).unwrap());
        let mut local_usage=None;
        for local in [true,false] {
            let context=WorkspaceContext::new(cpu);
            let value=WorkspaceTensor::existing(context.layout(&[1,1,4],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let transfers=Cell::new(CaptureNativePopulation::default());let scalar=[Cell::new(None)];
            let (mut observer,_host)=CaptureWorkspaceObserver::new(&source,geometry(),&context).unwrap();
            observer.transfers=Some(&transfers);observer.scalar_source=Some(&scalar);
            begin(&mut observer,&context);context.begin_span();
            observer.observe_value_with_origin(MODEL_LOGITS_OBSERVATION_PATH,&value,local).unwrap();
            assert_eq!(scalar[0].get(),Some(WorkspaceFloatingType::Float32));assert_eq!(observer.roots.is_empty(),!local);
            if local {
                let report=context.report(&observer.roots).unwrap();assert!(report.unpriced_operations.is_empty());assert!(!report.operations.is_empty());
                assert!(transfers.get().completions>0);local_usage=Some(observer.ledger.total());
            }else{
                assert_eq!(observer.ledger.total(),local_usage.unwrap());let native=transfers.get();
                assert_eq!((native.publications,native.completions,native.controls,native.retained_roots),(0,0,0,0));
                assert!(context.report(&[]).unwrap().operations.is_empty());
            }
        }
    }
}
