use super::*;

pub(super) fn projected_source(transform:CaptureTransform)->SharedCapturePlan {
    projected_source_limits(transform,None)
}
pub(super) fn projected_source_limits(transform:CaptureTransform,limit:Option<(CaptureLimitPolicy,u64)>)->SharedCapturePlan {
    projected_source_at(transform,limit,false)
}
pub(in super::super) fn projected_decode_source(transform:CaptureTransform)->SharedCapturePlan {
    projected_source_at(transform,None,true)
}
fn projected_source_at(transform:CaptureTransform,limit:Option<(CaptureLimitPolicy,u64)>,decode:bool)->SharedCapturePlan {
    let mut declaration=raw();
    if let Some((policy,bytes))=limit {
        declaration.limits.on_limit=policy;
        declaration.limits.per_step.retained_bytes=bytes;
    }
declaration.selections.truncate(1);
    declaration.selections[0].transform=transform;
    declaration.selections[0].schedule.prefill=!decode;
    declaration.selections[0].schedule.decode=decode;
    declaration.selections[0].slices=vec![CaptureSlice{axis:"row".into(),start:1,end:3,stride:1},
        CaptureSlice{axis:"width".into(),start:1,end:7,stride:2}];
    if decode{declaration.selections[0].slices.remove(0);}
    let mut point=point();point.axes=Some(vec![
        TensorAxis{name:"head".into(),dimension:SymbolicDimension::Known(2)},
        TensorAxis{name:"row".into(),dimension:SymbolicDimension::Sequence},
        TensorAxis{name:"width".into(),dimension:SymbolicDimension::Known(7)},
    ]);
    let support=ObservationSupportReport{schema_version:1,capture:Default::default(),points:vec![ObservationSupport{
        path:point.path.clone(),prefill:ObservationSupportStatus::Supported,decode:ObservationSupportStatus::Supported,floating_to_f32:true}]};
    let catalog=ObservationCatalog{schema_version:1,points:vec![point],completeness:DescriptionCompleteness::Complete};
    let capabilities=CaptureCapabilities{transformations:vec![CaptureTransformKind::Slice,CaptureTransformKind::Preview,
        CaptureTransformKind::Summary,CaptureTransformKind::Histogram],max_histogram_bins:2,physical_native_limit:false,conditions:vec![]};
    SharedCapturePlan::new(declaration.admit_with_text_origin(&catalog,&support,&capabilities,
        CaptureRequestShape{batch:1,prompt_tokens:3,max_predictions:4},CaptureTextOrigin{cached_positions:2}).unwrap())
}
pub(super) fn inference()->InferenceGeometry {InferenceGeometry{batch_size:1,cached_positions:2,input_positions:3,
    max_output_tokens:4,prefill_chunk_positions:1,output:OutputDemand::LastPosition}}
pub(super) fn projected_receipt(source:&SharedCapturePlan,combination:PartitionCaptureCombination,funding:&HostMetadataFunding,
    quota:&mut CaptureLedger)->PartitionCaptureReceiptPlan {
    projected_receipt_at(source,combination,funding,quota,context(source,0))
}
pub(in super::super) fn projected_decode_receipt(source:&SharedCapturePlan,combination:PartitionCaptureCombination,
    funding:&HostMetadataFunding,quota:&mut CaptureLedger)->PartitionCaptureReceiptPlan {
    let mut context=context(source,0);context.phase=CapturePhase::Decode;context.prediction=1;
    projected_receipt_at(source,combination,funding,quota,context)
}
fn projected_receipt_at(source:&SharedCapturePlan,combination:PartitionCaptureCombination,funding:&HostMetadataFunding,
    quota:&mut CaptureLedger,context:PartitionCaptureContext)->PartitionCaptureReceiptPlan {
    let rows=if combination==PartitionCaptureCombination::SumF64ToF32 {
        vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..7},PartitionCaptureContiguousProducer{rank:3,coordinates:0..7}]
    }else{vec![PartitionCaptureContiguousProducer{rank:3,coordinates:0..3},PartitionCaptureContiguousProducer{rank:0,coordinates:3..7},
        PartitionCaptureContiguousProducer{rank:1,coordinates:0..0}]};
    PartitionCaptureReceiptPlan::new_contiguous_shared_funded(source,&context,2,&rows,combination,4,
        PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10},funding,quota).unwrap()
}

#[test]
fn partition_prefill_host_bank_reuses_paid_scatter_and_keeps_additive_terms_raw() {
    for (transform,combination) in [
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint),
        (CaptureTransform::Preview{max_elements:5},PartitionCaptureCombination::Disjoint),
        (CaptureTransform::Summary,PartitionCaptureCombination::SumF64ToF32),
        (CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]},PartitionCaptureCombination::SumF64ToF32),
    ] {
        let source=projected_source(transform.clone());let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,_,_,_)=funding();let transport=Receipts{source:vec![],funding:funding.clone(),reject_delivery:false,calls:RefCell::new(vec![])};
        let mut receipt=projected_receipt(&source,combination,&funding,&mut quota);
        let geometries:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;let native=if combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{&transform};
        let sources:Vec<_>=geometries.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:native,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&sources,&funding,&mut quota).unwrap();
        let host=PartitionFragmentHostPlan::prepare_prefill(&receipt,inference()).unwrap();
        let h=host.initialization_peak_bytes();let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
        let spent=quota.total();let mut loan=bank.begin_local_prefill(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap();
        loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
        assert!(bank.begin_local_prefill(&receipt,0,&TensorDtype::F16,estimate(0)).is_err());
        assert!(bank.take_local(&receipt,0,&TensorDtype::F16,estimate(0)).is_err());
        let projection=receipt.producer(0).unwrap();
        let rows=if combination==PartitionCaptureCombination::SumF64ToF32 {
            CapturePrefillRowAssembly::prepare_additive_transform_partition(source.admission(),0,inference(),projection,0).unwrap()
        }else{CapturePrefillRowAssembly::prepare_partition(source.admission(),0,inference(),projection,0,combination).unwrap()};
        let width=projection.local_shape()[2] as usize;let offset=if combination==PartitionCaptureCombination::Disjoint{3}else{0};
        for index in 0..rows.chunk_count(){
            let chunk=rows.fragment(index).unwrap();let mut writer=bank.take_local_prefill_chunk(&receipt,0,&chunk).unwrap().prepare().unwrap();
            assert_eq!(chunk.position(),2+index);
            for mapping in chunk.mappings(){
                let source_index=mapping.source_index();let column=source_index%width+offset;
                let head=source_index/width;let value=head as f32*100.0+index as f32*10.0+column as f32+0.125;
                writer.push_f32(value).unwrap();
            }
            writer.finish().unwrap();bank.complete_local_prefill_chunk(&receipt,0,index).unwrap();
            assert!(bank.take_local_prefill_chunk(&receipt,0,&chunk).is_err(),"a physical chunk cannot be replayed");
            assert!(bank.value(0,0).is_none(),"no initialized prefix is a completed receipt");
        }
        bank.finish_local_prefill(&receipt,0).unwrap();
        let Some(PartitionFragmentValue::Tensor(value))=bank.value(0,0) else{panic!("raw local values")};
        let escaped=value.observation().clone();
        let TensorObservationData::F32(actual)=escaped.data() else{panic!("original F32 host values")};
        let mut expected=vec![];for head in 0..2 {for row in 1..3 {for column in [1,3,5] {
            if column>=offset {expected.push(head as f32*100.0+row as f32*10.0+column as f32+0.125);}
        }}}
        if let CaptureTransform::Preview{max_elements}=transform {expected.truncate(max_elements as usize);}
        assert_eq!(actual,&expected);assert_eq!(quota.total(),spent);
        assert!(!bank.complete(),"remote fragments still need real delivery");
        assert!(bank.finish_local_prefill(&receipt,0).is_err());assert!(transport.calls.borrow().is_empty());
        drop(bank);drop(loan);drop(run);drop(reservation);assert!(ledger(&pool).0>0);
        drop(escaped);assert_eq!(ledger(&pool).0,0);
    }
}

#[test]
fn partition_prefill_host_bank_rejects_foreign_source_and_retains_failed_attempt_custody() {
    for fail in ["drop","missing"] {
        let source=projected_source(CaptureTransform::Slice);let foreign=projected_source(CaptureTransform::Slice);
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();let (funding,_,_,_)=funding();
        let transport=Receipts{source:vec![],funding:funding.clone(),reject_delivery:false,calls:RefCell::new(vec![])};
        let mut receipt=projected_receipt(&source,PartitionCaptureCombination::Disjoint,&funding,&mut quota);
        let geometries:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let sources:Vec<_>=geometries.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&CaptureTransform::Slice,
            dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&sources,&funding,&mut quota).unwrap();
        let host=PartitionFragmentHostPlan::prepare_prefill(&receipt,inference()).unwrap();let h=host.initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();let spent=quota.total();
        let loan=bank.begin_local_prefill(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap();
        let projection=receipt.producer(0).unwrap();
        let wrong=CapturePrefillRowAssembly::prepare_partition(foreign.admission(),0,inference(),projection,0,PartitionCaptureCombination::Disjoint).unwrap();
        let wrong_chunk=wrong.fragment(0).unwrap();assert!(bank.take_local_prefill_chunk(&receipt,0,&wrong_chunk).is_err());
        let rows=CapturePrefillRowAssembly::prepare_partition(source.admission(),0,inference(),projection,0,PartitionCaptureCombination::Disjoint).unwrap();
        let chunk=rows.fragment(0).unwrap();
        let error=if fail=="drop" {
            drop(bank.take_local_prefill_chunk(&receipt,0,&chunk).unwrap().prepare().unwrap());
            bank.complete_local_prefill_chunk(&receipt,0,0).unwrap_err()
        }else{bank.finish_local_prefill(&receipt,0).unwrap_err()};
        assert!(bank.take_local_prefill_chunk(&receipt,0,&chunk).is_err());assert!(bank.value(0,0).is_none());
        assert_eq!(quota.total(),spent);drop(bank);drop(loan);drop(run);drop(reservation);
        assert!(ledger(&pool).0>0);drop(error);assert_eq!(ledger(&pool).0,0);
    }
}

#[test]
fn partition_prefill_reductions_preserve_typed_chunks_and_reject_dropped_receipts() {
    for histogram in [false,true] {for drop_chunk in [false,true] {
        let transform=if histogram{CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]}}else{CaptureTransform::Summary};
        let source=projected_source(transform.clone());let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,_,_,_)=funding();let transport=Receipts{source:vec![],funding:funding.clone(),reject_delivery:false,calls:RefCell::new(vec![])};
        let mut receipt=projected_receipt(&source,PartitionCaptureCombination::Disjoint,&funding,&mut quota);
        let geometries:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let sources:Vec<_>=geometries.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&transform,
            dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&sources,&funding,&mut quota).unwrap();
        let host=PartitionFragmentHostPlan::prepare_prefill(&receipt,inference()).unwrap();let h=host.initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();let spent=quota.total();
        let mut loan=bank.begin_local_prefill(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap();
        loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
        let plan=CapturePrefillTransformPlan::prepare_partition(source.admission(),0,inference(),receipt.producer(0).unwrap(),0,
            PartitionCaptureCombination::Disjoint).unwrap();
        let mut failure=None;
        for index in 0..plan.chunk_count(){
            let chunk=plan.fragment(index).unwrap();
            let claim=bank.take_local_prefill_reduction(&receipt,0,&chunk).unwrap();
            if drop_chunk&&index==1 {
                drop(claim);failure=Some(bank.complete_local_prefill_chunk(&receipt,0,index).unwrap_err());
                assert!(bank.take_local_prefill_reduction(&receipt,0,&chunk).is_err());break;
            }
            let values:Vec<f32>=if index==0 {vec![]} else {(0..2).flat_map(|_|if index==1{[0.5,2.5]}else{[f32::NAN,-1.0]}).collect()};
            assert_eq!(values.len() as u64,chunk.selected_elements());
            let value=match claim {
                PartitionFragmentDestination::Summary(claim)=>{
                    let summary=crate::capture::partition::summarize_f32(&values);
                    PartitionFragmentValue::Summary(claim.finish_partition(summary).unwrap())
                },
                PartitionFragmentDestination::Histogram(claim)=>{
                    let mut builder=claim.prepare().unwrap();builder.fill_partition_sum(&values).unwrap();
                    PartitionFragmentValue::Histogram(builder.finish_partition().unwrap())
                },_=>panic!("typed reduction destination"),
            };
            bank.record_local_prefill_reduction(&receipt,0,&chunk,value).unwrap();
            bank.complete_local_prefill_chunk(&receipt,0,index).unwrap();
            assert!(bank.take_local_prefill_reduction(&receipt,0,&chunk).is_err());
        }
        if !drop_chunk {
            bank.finish_local_prefill(&receipt,0).unwrap();
            match bank.value(0,0).unwrap(){
                PartitionFragmentValue::Summary(value)=>{
                    let value=value.observation();assert_eq!(value.elements,8);assert_eq!(value.finite,6);
                    assert_eq!(value.nan,2);assert_eq!(value.min,Some(-1.0));assert_eq!(value.max,Some(2.5));
                    assert!((value.mean.unwrap()-2.0/3.0).abs()<1e-6);
                },
                PartitionFragmentValue::Histogram(value)=>{
                    let value=value.observation();assert_eq!(value.edges,[-1.0,0.25,2.0]);assert_eq!(value.counts,[2,2]);
                    assert_eq!((value.below,value.above,value.non_finite),(0,2,2));
                },_=>panic!("typed whole fragment"),
            }
            assert!(!bank.complete(),"remote source values remain undelivered");
        } else {assert!(bank.finish_local_prefill(&receipt,0).is_err());}
        assert_eq!(quota.total(),spent);assert!(transport.calls.borrow().is_empty());
        drop(bank);drop(loan);drop(run);drop(reservation);
        assert_eq!(ledger(&pool).0>0,drop_chunk);drop(failure);assert_eq!(ledger(&pool).0,0);
    }}
}
