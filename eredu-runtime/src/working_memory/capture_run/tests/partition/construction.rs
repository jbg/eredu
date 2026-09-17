use super::*;
use crate::capture::partition::{PartitionCaptureProducer, PartitionCaptureReceiptLimits,
    PartitionCaptureReceiptPlan};
use eredu_core::capture::CaptureContiguousProjectionPlan;

#[test]
fn paid_contiguous_receipts_preserve_ordinary_shard_and_sum_order_with_empty_peers() {
    use crate::capture::partition::PartitionCaptureContiguousProducer as Producer;
    for selection in [0,2] {for sum in [false,true] {
        let source=source();let context=context(&source,selection);
        let geometry=CaptureTensorGeometry::prepare(source.admission(),selection,CapturePhase::Prefill,0,None).unwrap();
        let global:Vec<u64>=geometry.source_shape().iter().map(|n|*n as u64).collect();
        let axis=global.len()-1;let width=global[axis];assert!(width>1);
        let slice=resolve_slice(&source.admission().points()[selection],&source.admission().plan().selections[selection],&global).unwrap();
        let rows=if sum {vec![Producer{rank:3,coordinates:0..width},Producer{rank:0,coordinates:0..width},Producer{rank:1,coordinates:0..width}]}
            else {vec![Producer{rank:3,coordinates:width/2..width},Producer{rank:0,coordinates:0..0},Producer{rank:1,coordinates:0..width/2}]};
        let ordinary_rows=rows.iter().map(|row|PartitionCaptureProducer {rank:row.rank,
            projection:CaptureContiguousProjectionPlan::prepare(&global,&slice,axis,row.coordinates.clone(),3).unwrap().construct()}).collect();
        let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10};
        let mut ordinary_quota=CaptureLedger::new(source.admission());ordinary_quota.begin_step();
        let ordinary=if sum {PartitionCaptureReceiptPlan::new_sum_shared(source.clone(),context.clone(),ordinary_rows,4,limits,&mut ordinary_quota)}
            else {PartitionCaptureReceiptPlan::new_shared(source.clone(),context.clone(),ordinary_rows,4,limits,&mut ordinary_quota)}.unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,used,_,retired)=funding();
        let combination=if sum {PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
        let actual=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context,axis,&rows,combination,4,limits,&funding,&mut quota).unwrap();
        assert!(used.load(Ordering::SeqCst)>0);assert_eq!(actual.identity(),ordinary.identity());
        assert_eq!(actual.producers().map(|(rank,_)|rank).collect::<Vec<_>>(),[0,1,3]);
        assert!(actual.producer(2).is_none());
        for (rank,projection) in actual.producers() {
            let expected=ordinary.producer(rank).unwrap();
            assert_eq!(projection.local_shape(),expected.local_shape());
            assert_eq!(projection.global_slice(),expected.global_slice());
            assert_eq!(projection.fragments(),expected.fragments());
            assert_eq!(actual.encoding_usage(rank).unwrap(),ordinary.encoding_usage(rank).unwrap());
        }
        if !sum {assert!(actual.producer(0).unwrap().fragments().is_empty());}
        assert_eq!(actual.delivery_usage().unwrap(),ordinary.delivery_usage().unwrap());
        drop(ordinary);drop(geometry);drop(quota);drop(ordinary_quota);drop(source);drop(funding);
        assert!(!retired.load(Ordering::SeqCst));drop(actual);assert!(retired.load(Ordering::SeqCst));
    }}
}

#[test]
fn paid_contiguous_receipt_refuses_overlap_missing_terms_and_foreign_probability_sources() {
    use crate::capture::partition::PartitionCaptureContiguousProducer as Producer;
    for case in ["duplicate","overlap","missing","sum-shard","funding","vocabulary"] {
        let source=if case=="vocabulary" {super::vocabulary::vocabulary_source(false,3)}else{source()};
        let context=context(&source,0);
        let global:Vec<u64>=if case=="vocabulary" {vec![1,3,4]}else{
            CaptureTensorGeometry::prepare(source.admission(),0,CapturePhase::Prefill,0,None).unwrap()
                .source_shape().iter().map(|n|*n as u64).collect()};
        let axis=global.len()-1;let width=global[axis];assert!(width>1);
        let mut rows=vec![Producer{rank:3,coordinates:0..width/2},Producer{rank:1,coordinates:width/2..width}];
        match case {"duplicate"=>rows[1].rank=3,"overlap"=>rows[1].coordinates=0..width,
            "missing"=>rows.pop().map(|_|()).unwrap(),_=>()}
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,_,refuse,retired)=funding();refuse.store(case=="funding",Ordering::SeqCst);
        let error=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context,axis,&rows,
            if case=="sum-shard" {PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint},
            4,PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10},&funding,&mut quota).unwrap_err();
        drop(quota);drop(source);drop(funding);assert!(!retired.load(Ordering::SeqCst));
        drop(error);assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn paid_complete_receipt_keeps_ordinary_identity_geometry_and_source_custody() {
    for selection in [0, 2] {
        let source = source();
        let mut context = context(&source, selection);
        context.overlay_identity = Some("immutable-edit-λ".into());
        let geometry = CaptureTensorGeometry::prepare(source.admission(), selection,
            CapturePhase::Prefill, 0, None).unwrap();
        let global: Vec<u64> = geometry.source_shape().iter().map(|n| *n as u64).collect();
        let slice = resolve_slice(&source.admission().points()[selection],
            &source.admission().plan().selections[selection], &global).unwrap();
        let projection = CaptureContiguousProjectionPlan::prepare(&global, &slice,
            0, 0..global[0], 1).unwrap().construct();
        let limits = PartitionCaptureReceiptLimits {
            max_producers: 1, max_fragments: 1, max_record_bytes: 64 << 10,
        };
        let mut ordinary_quota = CaptureLedger::new(source.admission());
        ordinary_quota.begin_step();
        let ordinary = PartitionCaptureReceiptPlan::new_shared(source.clone(), context.clone(),
            vec![PartitionCaptureProducer { rank: 3, projection }], 4, limits,
            &mut ordinary_quota).unwrap();
        let mut original_quota = CaptureLedger::new(source.admission());
        original_quota.begin_step();
        let (funding, used, _, retired) = funding();
        let before = used.load(Ordering::SeqCst);
        let original = PartitionCaptureReceiptPlan::new_complete_shared_funded(&source,
            &context, 3, 4, limits, &funding, &mut original_quota).unwrap();
        assert!(used.load(Ordering::SeqCst) > before);
        assert_eq!(original.identity(), ordinary.identity());
        assert_eq!(original.context(), ordinary.context());
        assert!(std::ptr::eq(original.shared_plan_source().unwrap().admission(), source.admission()));
        assert_eq!(original.producers().map(|(rank, _)| rank).collect::<Vec<_>>(), [3]);
        assert!(original.producer(0).is_none());
        let actual = original.producer(3).unwrap();
        let expected = ordinary.producer(3).unwrap();
        assert_eq!(actual.global_shape(), expected.global_shape());
        assert_eq!(actual.local_shape(), expected.local_shape());
        assert_eq!(actual.global_slice(), expected.global_slice());
        assert_eq!(actual.fragments()[0].local(), expected.fragments()[0].local());
        assert_eq!(actual.fragments()[0].destination(), expected.fragments()[0].destination());
        assert_eq!(original.delivery_usage().unwrap(), ordinary.delivery_usage().unwrap());
        assert_eq!(original.encoding_usage(3).unwrap(), ordinary.encoding_usage(3).unwrap());
        drop(ordinary); drop(geometry); drop(original_quota); drop(ordinary_quota);
        drop(source); drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(original);
        assert!(retired.load(Ordering::SeqCst));
    }
    for refuse in [false, true] {
        let source = source();
        let mut context = context(&source, 0);
        context.capture_plan_identity = "foreign-source".into();
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let (funding, _, rejection, retired) = funding();
        rejection.store(refuse, Ordering::SeqCst);
        let error = PartitionCaptureReceiptPlan::new_complete_shared_funded(&source,
            &context, 3, 4, PartitionCaptureReceiptLimits {
                max_producers: 1, max_fragments: 1, max_record_bytes: 64 << 10,
            }, &funding, &mut quota).unwrap_err();
        drop(quota); drop(source); drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn paid_explicit_invocation_receipt_preserves_original_axes_bounds_and_host_source() {
    use crate::capture::partition::{PartitionCaptureContiguousProducer as Producer,
        PartitionInvocationCaptureGeometry,PartitionInvocationCaptureKind};
    for transform in [CaptureTransform::Slice,CaptureTransform::Preview{max_elements:3},CaptureTransform::Summary] {
        for sum in [false,true] {
            let mut declaration=raw();declaration.selections.truncate(1);
            declaration.selections[0].transform=transform.clone();
            declaration.selections[0].slices=vec![CaptureSlice{axis:"width".into(),start:1,end:8,stride:3}];
            let mut point=point();point.axes=Some(vec![
                TensorAxis{name:"batch".into(),dimension:SymbolicDimension::Batch},
                TensorAxis{name:"sequence".into(),dimension:SymbolicDimension::Sequence},
                TensorAxis{name:"width".into(),dimension:SymbolicDimension::Known(8)},
            ]);
            let source=admit(declaration,point,4,true);
            let shape=CaptureInvocationShape{batch:1,sequence:2,context:Some(9)};
            let mut context=context(&source,0);context.phase=CapturePhase::Decode;context.prediction=1;context.invocation=Some(shape);
            let global=[1,2,8];
            let slice=resolve_slice(&source.admission().points()[0],&source.admission().plan().selections[0],&global).unwrap();
            let rows=if sum{vec![Producer{rank:3,coordinates:0..8},Producer{rank:1,coordinates:0..8}]}
                else{vec![Producer{rank:3,coordinates:4..8},Producer{rank:1,coordinates:0..4}]};
            let ordinary_rows=rows.iter().map(|row|PartitionCaptureProducer{rank:row.rank,
                projection:CaptureContiguousProjectionPlan::prepare(&global,&slice,2,row.coordinates.clone(),2).unwrap().construct()}).collect();
            let limits=PartitionCaptureReceiptLimits{max_producers:2,max_fragments:2,max_record_bytes:64<<10};
            let combination=if sum{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
            let mut ordinary_quota=CaptureLedger::new(source.admission());ordinary_quota.begin_step();
            let ordinary=if sum{PartitionCaptureReceiptPlan::new_sum_shared(source.clone(),context.clone(),ordinary_rows,4,limits,&mut ordinary_quota)}
                else{PartitionCaptureReceiptPlan::new_shared(source.clone(),context.clone(),ordinary_rows,4,limits,&mut ordinary_quota)}.unwrap();
            let (funding,used,_,retired)=funding();
            let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
            let actual=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context,2,&rows,
                combination,4,limits,&funding,&mut quota).unwrap();
            assert_eq!(actual.identity(),ordinary.identity());assert_eq!(actual.context().invocation,Some(shape));
            assert!(used.load(Ordering::SeqCst)>0);
            {
                let host=PartitionFragmentHostPlan::prepare(&actual).unwrap();assert_eq!(host.fragment_count(),2);
                assert!(host.fragment_peak_bytes()>0);
                assert_eq!(host.assembly_peak_bytes()>0,sum&&matches!(transform,CaptureTransform::Summary));
                for (rank,projection) in actual.producers() {
                    assert_eq!(projection.global_shape(),global);assert_eq!(projection.fragments(),ordinary.producer(rank).unwrap().fragments());
                    let local=PartitionInvocationCaptureGeometry::from_receipt(&actual,rank,0).unwrap();
                    assert!(std::ptr::eq(local.admission(),source.admission()));assert!(std::ptr::eq(local.projection(),projection));
                    let local_shape=match local.kind(){PartitionInvocationCaptureKind::Tensor(g)=>g.source_shape(),
                        PartitionInvocationCaptureKind::Summary(g)=>g.source_shape(),PartitionInvocationCaptureKind::Histogram(g)=>g.source_shape()};
                    assert_eq!(local_shape,&[1,2,if sum{8}else{4}]);
                    // Omitting the separately admitted physical axes is not the
                    // same source, even when a fallback request has equal bytes.
                    assert!(PartitionInvocationCaptureGeometry::prepare(source.admission(),0,CapturePhase::Decode,1,
                        projection,0,combination).is_err());
                }
            }
            let mut changed=context.clone();changed.invocation=Some(CaptureInvocationShape{sequence:3,..shape});
            let mut changed_quota=CaptureLedger::new(source.admission());changed_quota.begin_step();
            let changed_receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&changed,2,&rows,
                combination,4,limits,&funding,&mut changed_quota).unwrap();
            assert_ne!(changed_receipt.identity(),actual.identity());
            assert_eq!(changed_receipt.producer(1).unwrap().global_shape(),&[1,3,8]);
            drop(changed_receipt);drop(changed_quota);
            let mut invalid=context.clone();invalid.invocation=Some(CaptureInvocationShape{sequence:4,..shape});
            let failure=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&invalid,2,&rows,
                combination,4,limits,&funding,&mut quota).unwrap_err();
            drop(ordinary);drop(ordinary_quota);drop(quota);drop(source);drop(funding);drop(actual);
            assert!(!retired.load(Ordering::SeqCst));drop(failure);assert!(retired.load(Ordering::SeqCst));
        }
    }
}


#[test]
fn projected_invocation_windows_keep_logical_receipt_and_physical_spatial_coordinates() {
    use crate::capture::partition::{PartitionCaptureContiguousProducer as Producer,
        PartitionInvocationCaptureGeometry,PartitionInvocationCaptureKind as K};
    for transform in [CaptureTransform::Slice,CaptureTransform::Preview{max_elements:2},
        CaptureTransform::Summary,CaptureTransform::Histogram{edges:vec![0.0,100.0,300.0]}] {
        for sum in [false,true] {
            let mut declaration=raw();declaration.selections.truncate(1);
            declaration.selections[0].transform=transform.clone();
            declaration.selections[0].slices=vec![
                CaptureSlice{axis:"sequence".into(),start:0,end:3,stride:2},
                CaptureSlice{axis:"width".into(),start:1,end:8,stride:3}];
            let mut point=point();point.axes=Some(vec![
                TensorAxis{name:"batch".into(),dimension:SymbolicDimension::Batch},
                TensorAxis{name:"sequence".into(),dimension:SymbolicDimension::Sequence},
                TensorAxis{name:"width".into(),dimension:SymbolicDimension::Known(8)},
            ]);
            let support=ObservationSupportReport{schema_version:1,capture:Default::default(),points:vec![
                ObservationSupport{path:point.path.clone(),prefill:ObservationSupportStatus::Supported,
                    decode:ObservationSupportStatus::Supported,floating_to_f32:true}]};
            let catalog=ObservationCatalog{schema_version:1,points:vec![point],completeness:DescriptionCompleteness::Complete};
            let caps=CaptureCapabilities{transformations:vec![CaptureTransformKind::Slice,CaptureTransformKind::Preview,
                CaptureTransformKind::Summary,CaptureTransformKind::Histogram],max_histogram_bins:2,
                physical_native_limit:false,conditions:vec![]};
            let source=SharedCapturePlan::new(declaration.admit_invocations(&catalog,&support,&caps,
                CaptureInvocationBounds{batch:1,max_sequence:3,max_context:Some(12),max_predictions:4}).unwrap());
            let logical=CaptureInvocationShape{batch:1,sequence:3,context:Some(9)};
            let mut context=context(&source,0);context.phase=CapturePhase::Decode;context.prediction=1;context.invocation=Some(logical);
            let rows=if sum{vec![Producer{rank:1,coordinates:0..8},Producer{rank:3,coordinates:0..8}]}
                else{vec![Producer{rank:1,coordinates:0..4},Producer{rank:3,coordinates:4..8}]};
            let combination=if sum{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
            let (funding,used,_,retired)=funding();let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
            let receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context,2,&rows,combination,4,
                PartitionCaptureReceiptLimits{max_producers:2,max_fragments:2,max_record_bytes:64<<10},&funding,&mut quota).unwrap();
            let identity=receipt.identity().to_owned();
            for (start,count) in [(0,1),(1,2),(1,1)] {
                let physical=CaptureInvocationShape{sequence:count,..logical};
                let window=CaptureInvocationWindow{logical_sequence:3,start};
                for row in &rows {
                    let projected=PartitionInvocationCaptureGeometry::from_window_receipt(&receipt,row.rank,0,physical,window).unwrap();
                    assert_eq!(projected.invocation_window(),Some((physical,window)));
                    assert!(std::ptr::eq(projected.admission(),source.admission()));
                    let (source_shape,starts,ends,strides,output_elements)=match projected.kind(){
                        K::Tensor(g)=>(g.source_shape(),g.starts(),g.ends(),g.strides(),g.shape().iter().product::<usize>()),
                        K::Summary(g)=>(g.source_shape(),g.starts(),g.ends(),g.strides(),g.shape().iter().product::<usize>()),
                        K::Histogram(g)=>(g.source_shape(),g.starts(),g.ends(),g.strides(),g.shape().iter().product::<usize>()),
                    };
                    assert_eq!(source_shape,&[1,count as usize,(row.coordinates.end-row.coordinates.start) as usize]);
                    // Read an independently numbered physical tensor by the
                    // actual local geometry, then compare global coordinates.
                    let mut actual=Vec::new();
                    for r in (starts[1]..ends[1]).step_by(strides[1] as usize){
                        for x in (starts[2]..ends[2]).step_by(strides[2] as usize){
                            actual.push(((start+r)*100+row.coordinates.start+x) as f32+0.125);
                        }
                    }
                    let mut expected=Vec::new();
                    for r in (0..3).step_by(2){for x in (1..8).step_by(3){
                        if (start..start+count).contains(&r)&&row.coordinates.contains(&x){expected.push((r*100+x) as f32+0.125);}
                    }}
                    assert_eq!(actual,expected);
                    let wanted=if matches!(transform,CaptureTransform::Preview{..}){expected.len().min(2)}else{expected.len()};
                    assert_eq!(output_elements,wanted);
                    assert_eq!(matches!(projected.kind(),K::Tensor(_)),sum||matches!(transform,CaptureTransform::Slice|CaptureTransform::Preview{..}));
                    // The same typed Host planners accept these exact physical
                    // source geometries; no logical-sized destination is used.
                    match projected.into_kind(){
                        K::Tensor(g)=>{assert!(CaptureTensorHostPlan::prepare(g).unwrap().initialization_peak_bytes()>0);},
                        K::Summary(g)=>{assert!(CaptureSummaryHostPlan::prepare(g).unwrap().initialization_peak_bytes()>0);},
                        K::Histogram(g)=>{assert!(CaptureHistogramHostPlan::prepare(g).unwrap().initialization_peak_bytes()>0);},
                    }
                }
                assert_eq!(receipt.identity(),identity);
            }
            let physical=CaptureInvocationShape{sequence:1,..logical};
            for (shape,window) in [
                (physical,CaptureInvocationWindow{logical_sequence:2,start:0}),
                (physical,CaptureInvocationWindow{logical_sequence:3,start:3}),
                (CaptureInvocationShape{context:Some(10),..physical},CaptureInvocationWindow{logical_sequence:3,start:0}),
            ]{assert!(PartitionInvocationCaptureGeometry::from_window_receipt(&receipt,1,0,shape,window).is_err());}
            assert!(PartitionInvocationCaptureGeometry::window_control_bytes().unwrap()>PartitionInvocationCaptureGeometry::control_bytes().unwrap());
            assert!(used.load(Ordering::SeqCst)>0);drop(quota);drop(funding);drop(source);
            assert!(!retired.load(Ordering::SeqCst));drop(receipt);assert!(retired.load(Ordering::SeqCst));
        }
    }
}

mod coordinates;
