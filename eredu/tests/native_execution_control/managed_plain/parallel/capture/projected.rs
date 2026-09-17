//! Actual sharded units and completed writes through the existing public driver.
use super::*;
use eredu_core::{SymbolicDimension,checkpoint::TensorDtype};
const CASE:&str="managed_plain::parallel::capture::projected::native_projected_prefill_receipts_match_ordinary_and_controlled";
const MODE:&str="EREDU_PUBLIC_PROJECTED_PREFILL_MODE";
const RESULT:&str="PUBLIC_PROJECTED_PREFILL_RESULT:";
const UNITS:&str="model.layers.0.feed_forward.units";
const WRITE:&str="model.layers.0.feed_forward.write";
const NAMES:[&str;4]=["strided unit shard λ","raw completed write","summary of completed write","histogram of completed write"];

#[derive(Debug)]
struct ExpectedSource { producers:Vec<usize>, combination:PartitionCaptureCombination }
fn topology<const TENSOR:usize,const PIPELINE:usize>()->eredu_core::ParallelTopology {
    eredu_core::ParallelTopology::new(TENSOR,PIPELINE,1,1).unwrap()
}
fn expected_source(descriptor:&eredu_core::ArchitectureDescriptor,topology:eredu_core::ParallelTopology,
    path:&str)->ExpectedSource {
    use eredu_core::component::ComponentWritePartition;
    let mut components=descriptor.components.iter().filter(|component|
        component.activation==path||component.write_output.as_deref()==Some(path));
    let component=components.next().expect("actual selected component source");
    assert!(components.next().is_none(),"ambiguous component source");
    // These fixtures have one ordinary two-layer pass. PP2 therefore assigns
    // exactly one actual layer per stage; reject other architecture layouts
    // instead of inventing a general pipeline balancing policy in the test.
    assert_eq!(descriptor.layer_groups.len(),1);
    let group=&descriptor.layer_groups[0];assert_eq!(group.physical_layer_count,2);
    assert_eq!(group.passes.len(),1);assert_eq!(group.passes[0].executions.len(),2);
    for (index,execution) in group.passes[0].executions.iter().enumerate(){assert_eq!(execution.physical_layer_index,index);}
    assert!(component.layer_index<2);assert!(matches!(topology.pipeline(),1|2));
    assert!(matches!(topology.tensor(),1|2));assert_eq!((topology.expert(),topology.data()),(1,1));
    let stage=if topology.pipeline()==1{0}else{component.layer_index};
    let members:Vec<_>=(0..topology.world_size()).filter(|&rank|
        topology.coordinates(rank).unwrap().pipeline()==stage).collect();
    assert_eq!(members.len(),topology.tensor());
    let activation=component.activation==path;
    let additive=!activation&&component.write_partition==ComponentWritePartition::TensorParallelSum&&topology.tensor()>1;
    if activation {assert_eq!(component.count%topology.tensor(),0);assert!(component.count>=topology.tensor());}
    ExpectedSource{producers:if activation||additive{members}else{vec![members[0]]},
        combination:if additive{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint}}
}
fn plan_world(world:usize)->CapturePlan {
    match world{2=>super::plan_world::<2,true,2>(),4=>super::plan_world::<2,true,4>(),_=>panic!("fixture world")}
}
fn selected<const DECODE:bool>(discovery:&eredu_core::capture::CaptureDiscovery,world:usize)->CapturePlan {
    // Consume actual loaded discovery, including its realized support report.
    // These are the existing Qwen fixture's exposed component/readout points.
    for (path,width,axis) in [(UNITS,32,"component"),(WRITE,16,"hidden")] {
        let point=discovery.catalog.points.iter().find(|point|point.path==path)
            .expect("fixture exposes its actual component hook");
        assert!(point.prefill);if DECODE{assert!(point.decode);}assert_eq!(point.value_type,eredu_core::ObservationValueType::Tensor);
        let axes=point.axes.as_ref().unwrap();assert_eq!(axes.len(),3);
        assert_eq!(axes[1].dimension,SymbolicDimension::Sequence);
        assert_eq!(axes[2].dimension,SymbolicDimension::Known(width));assert_eq!(axes[2].name,axis);
    }
    let mut plan=plan_world(world);plan.selections.clear();
    for (index,transform) in [CaptureTransform::Slice,CaptureTransform::FullTensor,
        CaptureTransform::Summary,CaptureTransform::Histogram{edges:HISTOGRAM_EDGES.to_vec()}].into_iter().enumerate() {
        plan.selections.push(CaptureSelection{id:NAMES[index].into(),path:if index==0{UNITS}else{WRITE}.into(),
            schedule:CaptureSchedule{prefill:true,decode:DECODE,..Default::default()},
            slices:if index==0 {
                let mut slices=vec![CaptureSlice{axis:"component".into(),start:1,end:31,stride:3}];
                if !DECODE{slices.push(CaptureSlice{axis:"sequence".into(),start:1,end:5,stride:2});}
                slices
            }else{vec![]},transform});
    }
    // Four rows and at most two component producers per row. Keep the existing
    // source-bound framing allowance, scaled by plan_world for every receiver.
    // These are logical limits; the shared native admission still derives its
    // separate original Host/Graph/Record capacities from actual sources.
    plan.limits.per_step.captures=4*(2+world as u64);
    plan.limits.cumulative.captures=4*plan.limits.per_step.captures;
    plan.limits.per_step.host_bytes*=2;plan.limits.per_step.retained_bytes*=2;
    plan.limits.cumulative.host_bytes*=2;plan.limits.cumulative.retained_bytes*=2;
    plan
}
fn loaded<const DECODE:bool,const TENSOR:usize,const PIPELINE:usize>(mode:&str,
    model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture,partitioned:bool)->serde_json::Value {
    let topology=topology::<TENSOR,PIPELINE>();let descriptor=inspect_architecture(&root.0).unwrap();
    let expected=[expected_source(&descriptor,topology,UNITS),expected_source(&descriptor,topology,WRITE)];
    // The complete TP decode quote is 8,392,686,959 bytes at chunk2, in
    // addition to existing source/model reservations. The original 8 GiB run
    // correctly returned BudgetExceeded (recorded separately); use 16 GiB for
    // positive delivery parity, preserving every production source/population.
    let capacity = DECODE.then_some(16 * 1024 * 1024 * 1024u64);
    capture_loaded_plan_with_capacity(mode,model,root,partitioned,capacity,|discovery|selected::<DECODE>(discovery,topology.world_size()),
        |ids,text,frames,partitioned|evaluate::<DECODE>(ids,text,frames,partitioned,&expected))
}
fn ordinary<const DECODE:bool,const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    loaded::<DECODE,TENSOR,PIPELINE>("ordinary",model,root,true)
}
fn managed<const DECODE:bool,const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    loaded::<DECODE,TENSOR,PIPELINE>("managed",model,root,true)
}
fn controlled<const DECODE:bool,const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    loaded::<DECODE,TENSOR,PIPELINE>("controlled",model,root,true)
}
fn run<const DECODE:bool,const TENSOR:usize,const PIPELINE:usize>(mode:&str)->serde_json::Value {
    if mode=="serial" {
        let root=managed_fixture(fixture(false));
        let execution=ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx","metal:0").unwrap());
        let (model,_)=LoadedModel::load_execution_plan(&MlxBackendFactory::default(),&root.0,&execution)
            .unwrap().into_parts();
        return loaded::<DECODE,TENSOR,PIPELINE>("ordinary",model,root,false);
    }
    let worker=match mode {"ordinary"=>ordinary::<DECODE,TENSOR,PIPELINE>,"managed"=>managed::<DECODE,TENSOR,PIPELINE>,
        "controlled"=>controlled::<DECODE,TENSOR,PIPELINE>,_=>panic!("capture mode")};
    run_partitioned_with_lifecycle(mode,topology::<TENSOR,PIPELINE>(),Some(worker))
}
fn evaluate<const DECODE:bool>(ids:&[u32],text:&str,frames:&[&CapturedStep],partitioned:bool,expected:&[ExpectedSource;2])->serde_json::Value {
    assert_eq!(ids.len(),4);assert_eq!(frames.len(),4);
    assert_eq!(frames[0].prediction_index,0);
    evaluate_frames::<DECODE>(ids,text,frames,partitioned,expected)
}
// Saved branches deliver a contiguous suffix at their original logical indices.
// Keep one source/shape/nonlinear evaluator for both full runs and those suffixes.
fn evaluate_frames<const DECODE:bool>(ids:&[u32],text:&str,frames:&[&CapturedStep],partitioned:bool,expected:&[ExpectedSource;2])->serde_json::Value {
    let mut output=Vec::new();
    let mut prior=None;
    for frame in frames {
        let prediction=usize::try_from(frame.prediction_index).unwrap();
        if let Some(previous)=prior{assert_eq!(prediction,previous+1);}
        prior=Some(prediction);
        assert_eq!(frame.outcome,CaptureStepOutcome::Committed);
        assert_eq!(frame.records.len(),4);
        assert_eq!(frame.partitions.len(),if partitioned&&(prediction==0||DECODE){4}else{0});
        let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
        assert_eq!(frame.phase,phase);
        if prediction==0||DECODE {
            for (index,evidence) in frame.partitions.iter().enumerate() {
                evidence.context.validate().unwrap();assert_eq!(evidence.context.selection_index,index);
                assert_eq!(evidence.context.prediction,prediction as u64);assert_eq!(evidence.context.phase,phase);
                // Qwen emits unit activations before the down projection, but
                // feed_forward.write after the existing row-parallel reduction.
                let source=&expected[usize::from(index!=0)];let producers=&source.producers;
                assert_eq!(&evidence.producers,producers,"source {} at prediction {prediction}",NAMES[index]);
                assert_eq!(evidence.contributions.len(),producers.len(),"source {}",NAMES[index]);
                assert_eq!(evidence.combination,source.combination,"source {}",NAMES[index]);
                assert!(evidence.contributions.iter().all(|term|evidence.producers.contains(&term.producer_rank)&&term.routed.is_none()));
                assert!(!evidence.receipt_plan_identity.is_empty());
                assert_eq!(evidence.context.run_identity,frame.partitions[0].context.run_identity);
                assert_eq!(evidence.context.forward_epoch,frame.partitions[0].context.forward_epoch);
            }
        }
        let positions=if prediction==0{5}else{1};
        let selected_positions=if prediction==0&&!DECODE{2}else{positions};
        let mut rows=Vec::new();let mut raw:Option<&[f32]>=None;
        for (index,record) in frame.records.iter().enumerate() {
            assert_eq!(record.selection_id,NAMES[index]);assert_eq!(record.path,if index==0{UNITS}else{WRITE});
            if prediction!=0&&!DECODE {
                assert!(matches!(record.outcome,CaptureOutcome::Skipped{reason:CaptureSkipReason::Schedule}));
                assert!(record.payload.is_none());
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":null,"outcome":record.outcome,"values":[]}));continue;
            }
            assert_eq!(record.source_dtype,Some(TensorDtype::F32));
            let source_shape=[1,positions,if index==0{32}else{16}];
            let selected_shape=[1,if index==0{selected_positions}else{positions},if index==0{10}else{16}];
            assert_eq!(record.source_shape.as_deref(),Some(&source_shape[..]));
            assert_eq!(record.selected_shape.as_deref(),Some(&selected_shape[..]));
            if index<2 {
                let tensor=record.payload.as_ref().unwrap().as_tensor().unwrap();
                let TensorObservationData::F32(values)=tensor.data() else{panic!("F32 projected payload")};
                assert_eq!(values.len(),usize::try_from(if index==0{selected_positions*10}else{positions*16}).unwrap());
                assert!(values.iter().all(|v|v.is_finite()));assert!(values.iter().any(|v|v.abs()>1e-6));
                if index==1{raw=Some(values);}
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":tensor.shape(),"outcome":record.outcome,"values":values}));continue;
            }
            let raw=raw.expect("actual completed write precedes nonlinear checks");
            if index==2 {
                let CapturePayload::Summary(summary)=record.payload.as_ref().unwrap() else{panic!("summary of completed write")};
                let expected=[raw.iter().copied().fold(f32::INFINITY,f32::min) as f64,
                    raw.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64,
                    raw.iter().map(|&v|f64::from(v)).sum::<f64>()/raw.len() as f64,
                    (raw.iter().map(|&v|f64::from(v).powi(2)).sum::<f64>()/raw.len() as f64).sqrt()];
                let actual=[summary.min.unwrap(),summary.max.unwrap(),summary.mean.unwrap(),summary.rms.unwrap()];
                for (a,e) in actual.into_iter().zip(expected){assert!((a-e).abs()<3e-5,"summary {a} vs global raw {e}");}
                assert_eq!((summary.elements,summary.finite,summary.non_finite),(positions*16,positions*16,0));
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":null,"outcome":record.outcome,"values":actual,
                    "counts":[summary.elements,summary.finite,summary.non_finite]}));
            }else{
                let CapturePayload::Histogram(histogram)=record.payload.as_ref().unwrap() else{panic!("histogram of completed write")};
                let mut counts=[0u64;HISTOGRAM_EDGES.len()-1];let (mut below,mut above)=(0,0);
                for &value in raw {
                    if value<HISTOGRAM_EDGES[0]{below+=1;}else if value>*HISTOGRAM_EDGES.last().unwrap(){above+=1;}
                    else{let bin=HISTOGRAM_EDGES.partition_point(|&edge|edge<=value).saturating_sub(1).min(counts.len()-1);counts[bin]+=1;}
                }
                assert_eq!(histogram.edges,HISTOGRAM_EDGES);assert_eq!(histogram.counts,counts);
                assert_eq!((histogram.below,histogram.above,histogram.non_finite),(below,above,0));
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":null,"outcome":record.outcome,"values":histogram.edges,
                    "counts":[histogram.counts,[below,above,histogram.non_finite]]}));
            }
        }
        output.push(serde_json::json!({"prediction":prediction,"phase":phase,"rows":rows}));
    }
    serde_json::json!({"ids":ids,"text":text,"frames":output})
}
#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_projected_prefill_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(CASE,MODE,RESULT,"projected TP prefill",2,
        &["ordinary","managed","controlled"],run::<false,2,1>,super::compare)
}


#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_projected_decode_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_projected_decode_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_PROJECTED_DECODE_MODE","PUBLIC_PROJECTED_DECODE_RESULT:","projected TP cached decode",2,
        &["ordinary","managed","controlled"],run::<true,2,1>,super::compare)
}

const ADDITIVE:&str="model.layers.1.feed_forward.shared.write";
const ADDITIVE_IDS:[&str;3]=["shared raw term sum","shared summary after sum","shared bins after sum"];
fn additive_source()->Fixture {
    // Reuse the existing nonzero admitted V3 fixture. This exact shared-expert
    // hook precedes the fused TP sum, unlike Qwen's completed write above.
    let root=crate::v3_components::source_with_prediction(false,true,0);
    let descriptor=inspect_architecture(&root.0).unwrap();
    let component=descriptor.components.iter().find(|component|component.write_output.as_deref()==Some(ADDITIVE))
        .expect("existing shared-expert write declaration");
    assert_eq!(component.write_partition,eredu_core::component::ComponentWritePartition::TensorParallelSum);
    root
}
fn additive_selected(discovery:&eredu_core::capture::CaptureDiscovery,world:usize)->CapturePlan {
    let point=discovery.catalog.points.iter().find(|point|point.path==ADDITIVE).expect("actual shared-expert hook");
    assert!(point.prefill&&point.decode);assert_eq!(point.value_type,eredu_core::ObservationValueType::Tensor);
    let axes=point.axes.as_ref().unwrap();assert_eq!(axes.len(),3);
    assert_eq!(axes[1].dimension,SymbolicDimension::Sequence);assert_eq!(axes[2].dimension,SymbolicDimension::Known(8));
    assert_eq!(axes[2].name,"hidden");
    let mut plan=plan_world(world);plan.selections.clear();
    for (index,transform) in [CaptureTransform::Slice,CaptureTransform::Summary,
        CaptureTransform::Histogram{edges:HISTOGRAM_EDGES.to_vec()}].into_iter().enumerate() {
        plan.selections.push(CaptureSelection{id:ADDITIVE_IDS[index].into(),path:ADDITIVE.into(),
            schedule:CaptureSchedule::default(),transform,
            slices:vec![CaptureSlice{axis:"hidden".into(),start:1,end:7,stride:2}]});
    }
    plan.limits.per_step.captures=3*(2+world as u64);plan.limits.cumulative.captures=4*plan.limits.per_step.captures;
    plan.limits.per_step.host_bytes*=2;plan.limits.per_step.retained_bytes*=2;
    plan.limits.cumulative.host_bytes*=2;plan.limits.cumulative.retained_bytes*=2;
    plan
}
fn additive_loaded<const TENSOR:usize,const PIPELINE:usize>(mode:&str,
    model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture,partitioned:bool)->serde_json::Value {
    let topology=topology::<TENSOR,PIPELINE>();let descriptor=inspect_architecture(&root.0).unwrap();
    let expected=expected_source(&descriptor,topology,ADDITIVE);
    // PP's original finite quote needs 13,668,918,891 bytes plus existing
    // retained source/model owners. The 8 GiB fixture correctly refused it;
    // preserve that recorded refusal and fund positive PP/combined parity.
    let capacity = (PIPELINE > 1).then_some(16 * 1024 * 1024 * 1024u64);
    capture_loaded_plan_with_capacity(mode,model,root,partitioned,capacity,|discovery|additive_selected(discovery,topology.world_size()),
        |ids,text,frames,partitioned|additive_evaluate(ids,text,frames,partitioned,&expected))
}
fn additive_ordinary<const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    additive_loaded::<TENSOR,PIPELINE>("ordinary",model,root,true)
}
fn additive_managed<const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    additive_loaded::<TENSOR,PIPELINE>("managed",model,root,true)
}
fn additive_controlled<const TENSOR:usize,const PIPELINE:usize>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {
    additive_loaded::<TENSOR,PIPELINE>("controlled",model,root,true)
}
fn additive_run<const TENSOR:usize,const PIPELINE:usize>(mode:&str)->serde_json::Value {
    if mode=="serial" {
        let root=managed_fixture(additive_source());
        let execution=ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx","metal:0").unwrap());
        let (model,_)=LoadedModel::load_execution_plan(&MlxBackendFactory::default(),&root.0,&execution)
            .unwrap().into_parts();
        return additive_loaded::<TENSOR,PIPELINE>("ordinary",model,root,false);
    }
    let worker=match mode {"ordinary"=>additive_ordinary::<TENSOR,PIPELINE>,"managed"=>additive_managed::<TENSOR,PIPELINE>,
        "controlled"=>additive_controlled::<TENSOR,PIPELINE>,_=>panic!("capture mode")};
    run_partitioned_with_fixture(mode,topology::<TENSOR,PIPELINE>(),Some(worker),additive_source)
}
fn additive_evaluate(ids:&[u32],text:&str,frames:&[&CapturedStep],partitioned:bool,expected:&ExpectedSource)->serde_json::Value {
    assert_eq!(ids.len(),4);assert_eq!(frames.len(),4);let mut output=Vec::new();
    for (prediction,frame) in frames.iter().enumerate() {
        let positions=if prediction==0{5u64}else{1};
        let phase=if prediction==0{CapturePhase::Prefill}else{CapturePhase::Decode};
        assert_eq!(frame.outcome,CaptureStepOutcome::Committed);assert_eq!(frame.phase,phase);
        assert_eq!(frame.prediction_index as usize,prediction);assert_eq!(frame.records.len(),3);
        assert_eq!(frame.partitions.len(),if partitioned{3}else{0});
        for (index,evidence) in frame.partitions.iter().enumerate() {
            evidence.context.validate().unwrap();assert_eq!(evidence.context.selection_index,index);
            assert_eq!(evidence.context.prediction,prediction as u64);assert_eq!(evidence.context.phase,phase);
            assert_eq!(evidence.producers,expected.producers);assert_eq!(evidence.contributions.len(),expected.producers.len());
            assert_eq!(evidence.combination,expected.combination);
            assert!(!evidence.receipt_plan_identity.is_empty());
            assert_eq!(evidence.context.run_identity,frame.partitions[0].context.run_identity);
            assert_eq!(evidence.context.forward_epoch,frame.partitions[0].context.forward_epoch);
            for contribution in &evidence.contributions {
                assert!(evidence.producers.contains(&contribution.producer_rank));assert!(contribution.routed.is_none());
                assert_eq!(contribution.local.starts,[0,0,1]);
                assert_eq!(&contribution.local.ends[..2],&[1,positions]);
                assert_eq!((contribution.local.ends[2]-1).div_ceil(2),3);
                assert_eq!(contribution.local.strides,[1,1,2]);assert_eq!(contribution.local.shape,[1,positions,3]);
                assert_eq!(contribution.destination.starts,[0,0,0]);assert_eq!(contribution.destination.shape,[1,positions,3]);
            }
        }
        for (index,record) in frame.records.iter().enumerate() {
            assert_eq!(record.selection_id,ADDITIVE_IDS[index]);assert_eq!(record.path,ADDITIVE);
            assert_eq!(record.source_dtype,Some(TensorDtype::F32));
            assert_eq!(record.source_shape.as_deref(),Some(&[1,positions,8][..]));
            assert_eq!(record.selected_shape.as_deref(),Some(&[1,positions,3][..]));
        }
        let tensor=frame.records[0].payload.as_ref().unwrap().as_tensor().unwrap();
        let TensorObservationData::F32(raw)=tensor.data() else{panic!("actual raw additive F32 output")};
        assert_eq!(raw.len(),usize::try_from(positions*3).unwrap());
        assert!(raw.iter().all(|v|v.is_finite()));assert!(raw.iter().any(|v|v.abs()>1e-8));
        let CapturePayload::Summary(summary)=frame.records[1].payload.as_ref().unwrap() else{panic!("post-sum summary")};
        let actual=[summary.min.unwrap(),summary.max.unwrap(),summary.mean.unwrap(),summary.rms.unwrap()];
        let expected=[raw.iter().copied().fold(f32::INFINITY,f32::min) as f64,
            raw.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64,
            raw.iter().map(|&v|f64::from(v)).sum::<f64>()/raw.len() as f64,
            (raw.iter().map(|&v|f64::from(v).powi(2)).sum::<f64>()/raw.len() as f64).sqrt()];
        for (a,e) in actual.into_iter().zip(expected){assert!((a-e).abs()<3e-5,"summary {a} vs post-sum F32 {e}");}
        assert_eq!((summary.elements,summary.finite,summary.non_finite),(positions*3,positions*3,0));
        let CapturePayload::Histogram(histogram)=frame.records[2].payload.as_ref().unwrap() else{panic!("post-sum histogram")};
        let mut counts=[0u64;HISTOGRAM_EDGES.len()-1];let (mut below,mut above)=(0,0);
        for &value in raw {
            if value<HISTOGRAM_EDGES[0]{below+=1;}else if value>*HISTOGRAM_EDGES.last().unwrap(){above+=1;}
            else{let bin=HISTOGRAM_EDGES.partition_point(|&edge|edge<=value).saturating_sub(1).min(counts.len()-1);counts[bin]+=1;}
        }
        assert_eq!(histogram.edges,HISTOGRAM_EDGES);assert_eq!(histogram.counts,counts);
        assert_eq!((histogram.below,histogram.above,histogram.non_finite),(below,above,0));
        let rows=vec![
            serde_json::json!({"id":ADDITIVE_IDS[0],"shape":frame.records[0].selected_shape,"payload_shape":tensor.shape(),"outcome":frame.records[0].outcome,"values":raw}),
            serde_json::json!({"id":ADDITIVE_IDS[1],"shape":frame.records[1].selected_shape,"payload_shape":null,"outcome":frame.records[1].outcome,"values":actual,"counts":[summary.elements,summary.finite,summary.non_finite]}),
            serde_json::json!({"id":ADDITIVE_IDS[2],"shape":frame.records[2].selected_shape,"payload_shape":null,"outcome":frame.records[2].outcome,"values":histogram.edges,"counts":[histogram.counts,[below,above,histogram.non_finite]]}),
        ];
        output.push(serde_json::json!({"prediction":prediction,"rows":rows}));
    }
    serde_json::json!({"ids":ids,"text":text,"frames":output})
}
#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_additive_shared_write_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_additive_shared_write_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ADDITIVE_CAPTURE_MODE","PUBLIC_ADDITIVE_CAPTURE_RESULT:","shared expert additive capture",2,
        &["ordinary","managed","controlled"],additive_run::<2,1>,super::compare)
}

#[test]
#[ignore="requires Metal and 2 local Ring processes; run after matching TP case passes"]
fn native_projected_pipeline_prefill_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_projected_pipeline_prefill_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_PROJECTED_PP_PREFILL_MODE","PUBLIC_PROJECTED_PP_PREFILL_RESULT:","projected PP prefill",2,
        &["ordinary","managed","controlled"],run::<false,1,2>,super::compare)
}

#[test]
#[ignore="requires Metal and 4 local Ring processes; run after matching TP case passes"]
fn native_projected_combined_prefill_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_projected_combined_prefill_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_PROJECTED_COMBINED_PREFILL_MODE","PUBLIC_PROJECTED_COMBINED_PREFILL_RESULT:","projected TP/PP prefill",4,
        &["ordinary","managed","controlled"],run::<false,2,2>,super::compare)
}

#[test]
#[ignore="requires Metal and 2 local Ring processes; run after matching TP case passes"]
fn native_projected_pipeline_decode_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_projected_pipeline_decode_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_PROJECTED_PP_DECODE_MODE","PUBLIC_PROJECTED_PP_DECODE_RESULT:","projected PP cached decode",2,
        &["ordinary","managed","controlled"],run::<true,1,2>,super::compare)
}

#[test]
#[ignore="requires Metal and 4 local Ring processes; run after matching TP case passes"]
fn native_projected_combined_decode_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_projected_combined_decode_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_PROJECTED_COMBINED_DECODE_MODE","PUBLIC_PROJECTED_COMBINED_DECODE_RESULT:","projected TP/PP cached decode",4,
        &["ordinary","managed","controlled"],run::<true,2,2>,super::compare)
}

#[test]
#[ignore="requires Metal and 2 local Ring processes; run after matching TP case passes"]
fn native_shared_write_pipeline_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_shared_write_pipeline_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_SHARED_WRITE_PP_MODE","PUBLIC_SHARED_WRITE_PP_RESULT:","shared expert PP capture",2,
        &["ordinary","managed","controlled"],additive_run::<1,2>,super::compare)
}

#[test]
#[ignore="requires Metal and 4 local Ring processes; run after matching TP case passes"]
fn native_additive_shared_write_combined_receipts_match_ordinary_and_controlled(){
    compare_selected_modes_by(
        "managed_plain::parallel::capture::projected::native_additive_shared_write_combined_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_ADDITIVE_COMBINED_MODE","PUBLIC_ADDITIVE_COMBINED_RESULT:","shared expert additive TP/PP capture",4,
        &["ordinary","managed","controlled"],additive_run::<2,2>,super::compare)
}

#[path = "projected/saved.rs"]
mod saved;
