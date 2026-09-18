use super::*;
#[test]
fn scheduled_decode_projection_joins_original_host_vote_callback_and_final_delivery(){
    use super::super::super::prefill::{projected_decode_source,projected_decode_receipt};
    for sum in [false,true] {for reject in [false,true] {
        let transform=if sum{CaptureTransform::Summary}else{CaptureTransform::Slice};
        let combination=if sum{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
        let source=projected_decode_source(transform.clone());let (metadata,_,_,_)=funding();
        let transport=ProgramRanks(Ranks{local:0,bytes:RefCell::new(std::array::from_fn(|_|None)),funding:metadata.clone(),reject,calls:RefCell::new(vec![])});
        let mut quote=CaptureLedger::new(source.admission());quote.begin_step();
        let mut prototype=projected_decode_receipt(&source,combination,&metadata,&mut quote);
        if !sum {
            use crate::capture::partition::PartitionInvocationReceiverSource;
            for (rank,shape) in [(1,[2,1,0]),(2,[2,1,4])] {
                let source=PartitionInvocationReceiverSource::prepare_local(&prototype,rank,TensorDtype::F16,&shape).unwrap();
                assert_eq!(source.source_shape(),shape.map(|n|n as usize));assert_eq!(source.dtype(),&TensorDtype::F16);
            }
            assert!(PartitionInvocationReceiverSource::prepare_local(&prototype,0,TensorDtype::F16,&[2,1,4]).is_err());
            assert!(PartitionInvocationReceiverSource::prepare_local(&prototype,2,TensorDtype::I32,&[2,1,4]).is_err());
        }
        let geometry:Vec<_>=prototype.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let sources:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&raw,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut prototype,&sources,&metadata,&mut quote).unwrap();
        let final_charge=allowance.assembly_record_charge(&prototype).unwrap();
        let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10};
        let producers=if sum{vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..7},PartitionCaptureContiguousProducer{rank:3,coordinates:0..7}]}
            else{vec![PartitionCaptureContiguousProducer{rank:3,coordinates:0..3},PartitionCaptureContiguousProducer{rank:0,coordinates:3..7},PartitionCaptureContiguousProducer{rank:1,coordinates:0..0}]};
        let geometries:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|crate::capture::partition::PartitionCaptureFragmentGeometry{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&raw,estimate:estimate(*rank)}).collect();
        let local_shape=if sum{[2,1,7]}else{[2,1,4]};
        let local=||Some(crate::capture::partition::PartitionCaptureLocalSource{producer:0,shape:&local_shape,dtype:TensorDtype::F16});
        assert!(PreparedPartitionContiguousSource::new_local_decode(&source,0,2,&producers,&geometries,local(),combination,0,&metadata).is_err());
        let retained=PreparedPartitionContiguousSource::new_local_decode(&source,0,2,&producers,&geometries,local(),combination,1,&metadata).unwrap();
        let mut oracle_quota=CaptureLedger::new(source.admission());oracle_quota.begin_step();
        let ordinary_rows=prototype.producers().map(|(rank,p)|PartitionCaptureProducer{rank,projection:p.clone()}).collect();
        let oracle_limits=PartitionCaptureReceiptLimits{max_record_bytes:prototype.max_record_bytes(),..limits};
        let ordinary=if sum{PartitionCaptureReceiptPlan::new_sum(source.clone(),prototype.context().clone(),ordinary_rows,4,oracle_limits,&mut oracle_quota)}
            else{PartitionCaptureReceiptPlan::new(source.clone(),prototype.context().clone(),ordinary_rows,4,oracle_limits,&mut oracle_quota)}.unwrap();
        assert_eq!(ordinary.identity(),prototype.identity());let mut ordinary=ordinary.into_delivery();
        for (rank,projection) in prototype.producers(){
            let mut fragments=vec![];
            if let Some(fragment)=projection.fragments().first(){
                let g=fragment.local();let values:Vec<_>=(0..2).flat_map(|head|(0..1).flat_map(move |row|[1,3,5].into_iter()
                    .filter(move |column|sum||if rank==3{*column<3}else{*column>=3})
                    .map(move |column|head as f32*100.0+row as f32*10.0+column as f32+rank as f32*0.25))).collect();
                let selection=&source.admission().plan().selections[0];let point=&source.admission().points()[0];
                fragments.push(PartitionCaptureFragmentRecord{fragment_index:0,record:CaptureRecord{
                    schema_version:CAPTURE_SCHEMA_VERSION,selection_id:selection.id.clone(),path:selection.path.clone(),node_id:point.node_id.clone(),position:point.position,
                    source_shape:Some(projection.local_shape().to_vec()),source_dtype:Some(TensorDtype::F16),selected_shape:Some(g.shape.clone()),outcome:CaptureOutcome::Captured,
                    payload:Some(CapturePayload::Tensor(TensorObservation::new(g.shape.iter().map(|n|*n as usize).collect(),TensorObservationData::F32(values)).unwrap())),
                    charged:allowance.fragment_record_charge(&prototype,rank,0).unwrap().1}});
            }
            let envelope=PartitionCaptureProducerRecord{schema_version:PARTITION_CAPTURE_SCHEMA_VERSION,combination,receipt_plan_identity:prototype.identity().into(),
                context:prototype.context().clone(),producer_rank:rank,source_dtype:Some(TensorDtype::F16),fragments};
            let bytes=serde_json::to_vec(&envelope).unwrap();ordinary.receive(rank,&bytes,&mut oracle_quota).unwrap();transport.0.bytes.borrow_mut()[rank]=Some(bytes);
        }
        let expected=ordinary.finish(&mut oracle_quota).unwrap();drop(allowance);
        let h=PartitionFragmentHostPlan::prepare(&prototype).unwrap().initialization_peak_bytes()+plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let context=prototype.context().clone();
        let host=run.prepare_partition_fragment_host(&reservation,prototype,None).unwrap();
        let mut final_bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        drop(final_bank.begin_step(CapturePhase::Prefill,0).unwrap());
        let mut frame=final_bank.begin_step(CapturePhase::Decode,1).unwrap().prepare().unwrap();
        let mut program=PreparedPartitionCaptureProgram::new_selected(&transport,&source,&context,&[PreparedPartitionCaptureRow::Contiguous],limits,&metadata).unwrap()
            .with_contiguous_row(0,retained,host).unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let epoch=DistributedCommitEpoch::new(17).unwrap();
        program.prepare(&source,CapturePhase::Decode,1,epoch,&mut frame,&mut quota).unwrap();
        assert!(program.take_invocation_projection(0).is_err());
        program.coordinate(epoch,&quota).unwrap();let spent=quota.total();
        let mut native=Native::default();
        assert!(program.produces(0).unwrap());
        assert!(program.take_invocation_receiver_source(0).unwrap().is_none());
        let hook=program.take_invocation_projection(0).unwrap().unwrap();
        let width=if sum{7}else{4};let offset=if sum{0}else{3};
        let value=Tensor{shape:[2,1,width],values:(0..2).flat_map(|head|(0..width)
            .map(move |column|head as f32*100.0+(column+offset) as f32)).collect()};
        let hook=hook.observe_invocation(&mut native,&value).unwrap();program.return_invocation_projection(0,hook).unwrap();
        assert!(program.take_invocation_projection(0).is_err(),"the same decode invocation cannot issue another native loan");
        let result=program.deliver(&mut frame);assert_eq!(result.is_err(),reject);
        if reject{assert!(frame.records()[0].payload.is_none());}else{
            assert_eq!(frame.records()[0].charged,final_charge);assert_eq!(frame.partition_evidence().len(),1);
            assert_eq!(serde_json::to_value(frame.records()[0].payload.as_ref().unwrap()).unwrap(),serde_json::to_value(expected.capture().record().payload.as_ref().unwrap()).unwrap());
        }
        assert_eq!(*transport.0.calls.borrow(),[PartitionCaptureFrameKind::Source,PartitionCaptureFrameKind::Coordination,
            PartitionCaptureFrameKind::Preparation,PartitionCaptureFrameKind::Payload,PartitionCaptureFrameKind::Delivery]);
        assert!(program.deliver(&mut frame).is_err());assert_eq!(quota.total(),spent);
        let escaped=if reject{drop(frame);None}else{Some(frame.finish(CaptureStepOutcome::Aborted,unlimited(),unlimited(),0.0).unwrap())};
        drop(final_bank);drop(program);drop(run);drop(reservation);assert!(ledger(&pool).0>0);
        drop(escaped);drop(result);assert_eq!(ledger(&pool).0,0);
    }}
}
