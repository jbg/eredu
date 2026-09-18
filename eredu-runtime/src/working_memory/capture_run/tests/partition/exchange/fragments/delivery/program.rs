//! Shared scheduled program, real projected hook loans and final receipt vote.
use super::*;
use super::hook_callback::{Native,Tensor};
use crate::capture::partition::{PreparedPartitionCaptureProgram,PreparedPartitionCaptureRow,
    PreparedPartitionContiguousSource,ScheduledPartitionCapture};
struct ProgramRanks(Ranks);
impl ConsensusTransport for ProgramRanks {
    type Error=Infallible;
    fn participant_count(&self)->usize{4}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Infallible>{panic!("unbounded program transport")}
}
impl BoundedConsensusTransport for ProgramRanks {
    type Completion=Done;type GatherOutput=();
    fn submit_all_gather_words(&self,_:&[u32])->Result<Submission<(),Done>,Infallible>{panic!("untyped program transport")}
    fn resolve_all_gather_words(&self,_:())->Result<Vec<u32>,Infallible>{panic!("untyped program output")}
}
impl PartitionCaptureTransport for ProgramRanks {
    fn capture_rank(&self)->usize{self.0.local}
    fn capture_wait(&self)->Result<BoundedCompletionWait,CaptureError>{self.0.capture_wait()}
    fn ensure_capture_active(&self)->Result<(),BackendFailure>{Ok(())}
    fn estimate_capture_gather(&self,n:usize)->Result<CaptureUsage,CaptureError>{self.0.estimate_capture_gather(n)}
    fn fail_capture_exchange(&self,e:&PartitionCaptureExchangeError){self.0.fail_capture_exchange(e)}
    fn capture_word_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{self.0.capture_word_destination(n)}
    fn capture_byte_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u8>,PartitionCaptureExchangeError>{self.0.capture_byte_destination(n)}
    fn gather_capture_frame(&self,frame:&PartitionCaptureFrame<'_>,wait:BoundedCompletionWait)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{
        if !matches!(frame.kind(),PartitionCaptureFrameKind::Source|PartitionCaptureFrameKind::Coordination){
            return self.0.gather_capture_frame(frame,wait);
        }
        self.0.calls.borrow_mut().push(frame.kind());
        let mut out=PartitionCaptureBuffer::funded(frame.gathered_words(),&self.0.funding)?;
        for rank in 0..4 {
            let mut words:[u32;16]=frame.words().try_into().unwrap();
            match frame.kind(){
                PartitionCaptureFrameKind::Source=>{words[3]=rank as u32;words[13]=u32::from(rank==0||rank==3);},
                PartitionCaptureFrameKind::Coordination=>words[2]=rank as u32,
                _=>unreachable!(),
            }
            out.extend_from_slice(&words)?;
        }
        Ok(out)
    }
}
#[test]
fn scheduled_contiguous_program_preserves_projected_hooks_votes_and_additive_delivery(){
    use super::super::prefill::{projected_source,projected_receipt,inference};
    for sum in [false,true] {for reject in [false,true] {
        let transform=if sum{CaptureTransform::Summary}else{CaptureTransform::Slice};
        let combination=if sum{PartitionCaptureCombination::SumF64ToF32}else{PartitionCaptureCombination::Disjoint};
        let source=projected_source(transform.clone());let (metadata,_,_,_)=funding();
        let transport=ProgramRanks(Ranks{local:0,bytes:RefCell::new(std::array::from_fn(|_|None)),funding:metadata.clone(),reject,calls:RefCell::new(vec![])});
        let mut quote=CaptureLedger::new(source.admission());quote.begin_step();
        let mut prototype=projected_receipt(&source,combination,&metadata,&mut quote);
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
        let dtypes=vec![Some(TensorDtype::F16);producers.len()];
        let retained=PreparedPartitionContiguousSource::new(&source,0,2,&producers,&dtypes,&sources,combination,inference(),&metadata).unwrap();
        let mut oracle_quota=CaptureLedger::new(source.admission());oracle_quota.begin_step();
        let ordinary_rows=prototype.producers().map(|(rank,p)|PartitionCaptureProducer{rank,projection:p.clone()}).collect();
        let oracle_limits=PartitionCaptureReceiptLimits{max_record_bytes:prototype.max_record_bytes(),..limits};
        let ordinary=if sum{PartitionCaptureReceiptPlan::new_sum(source.clone(),prototype.context().clone(),ordinary_rows,4,oracle_limits,&mut oracle_quota)}
            else{PartitionCaptureReceiptPlan::new(source.clone(),prototype.context().clone(),ordinary_rows,4,oracle_limits,&mut oracle_quota)}.unwrap();
        assert_eq!(ordinary.identity(),prototype.identity());let mut ordinary=ordinary.into_delivery();
        for (rank,projection) in prototype.producers(){
            let mut fragments=vec![];
            if let Some(fragment)=projection.fragments().first(){
                let g=fragment.local();let values:Vec<_>=(0..2).flat_map(|head|(1..3).flat_map(move |row|[1,3,5].into_iter()
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
        let h=PartitionFragmentHostPlan::prepare_prefill(&prototype,inference()).unwrap().initialization_peak_bytes()+plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let host=run.prepare_partition_fragment_host(&reservation,prototype,Some(inference())).unwrap();
        let mut final_bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        let mut frame=final_bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(inference()).unwrap();
        // Other selections may need a wider program envelope. This selection
        // must retain its original receipt identity and Host geometry exactly.
        let program_limits=PartitionCaptureReceiptLimits{max_producers:limits.max_producers+1,
            max_fragments:limits.max_fragments+2,max_record_bytes:limits.max_record_bytes*2};
        let mut program=PreparedPartitionCaptureProgram::new_selected(&transport,&source,&context(&source,0),&[PreparedPartitionCaptureRow::Contiguous],program_limits,&metadata).unwrap()
            .with_contiguous_row(0,retained,host).unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let epoch=DistributedCommitEpoch::new(17).unwrap();
        program.prepare(&source,CapturePhase::Prefill,0,epoch,&mut frame,&mut quota).unwrap();
        assert!(program.take_prefill_projection(0,&mut frame,true).is_err());
        program.coordinate(epoch,&quota).unwrap();let spent=quota.total();
        let raw=if sum{None}else{Some(CapturePrefillRowAssembly::prepare(source.admission(),0,inference()).unwrap())};
        let reduced=if sum{Some(CapturePrefillTransformPlan::prepare(source.admission(),0,inference()).unwrap())}else{None};
        let mut native=Native::default();
        for k in 0..3 {
            let epoch=DistributedCommitEpoch::new(17+k).unwrap();if k!=0{program.coordinate(epoch,&quota).unwrap();}
            assert!(program.produces(0).unwrap());assert!(program.take_receiver_source(0).unwrap().is_none());
            let chunk=crate::prefill::PrefillChunk{input:k..k+1,position:2+k,output:inference().output.for_chunk(k==2)};
            let decision=frame.begin_prefill_hook(0,&chunk,&source.admission().plan().selections[0].path).unwrap();
            let hook=program.take_prefill_projection(0,&mut frame,decision==crate::capture::CapturePrefillHookDecision::First).unwrap().unwrap();
            let width=if sum{7}else{4};let offset=if sum{0}else{3};
            let value=Tensor{shape:[2,1,width],values:(0..2).flat_map(|head|(0..width)
                .map(move |column|head as f32*100.0+k as f32*10.0+(column+offset) as f32)).collect()};
            let hook=hook.observe(&mut native,&value,inference(),k).unwrap();program.return_prefill_projection(0,hook).unwrap();
            assert!(program.take_prefill_projection(0,&mut frame,false).is_err(),"same epoch cannot mint another local loan");
            if let Some(reduced)=&reduced{frame.finish_summary_prefill_hook(0,&reduced.fragment(k).unwrap()).unwrap();}
            else{frame.finish_prefill_hook(0,&raw.as_ref().unwrap().fragment(k).unwrap()).unwrap();}
            frame.complete_prefill_chunk(k).unwrap();
        }
        assert_eq!(native.validations.get(),3);assert_eq!(native.raw,2);assert_eq!(quota.total(),spent);
        frame.finish_local_prefill_targets().unwrap();let result=program.deliver(&mut frame);assert_eq!(result.is_err(),reject);
        if reject{assert!(frame.records()[0].payload.is_none());}else{
            frame.finish_prefill_targets().unwrap();assert_eq!(frame.records()[0].charged,final_charge);assert_eq!(frame.partition_evidence().len(),1);
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

struct VotedRanks{base:ProgramRanks,reject_source:bool,failures:std::cell::Cell<usize>}
impl ConsensusTransport for VotedRanks{
    type Error=Infallible;
    fn participant_count(&self)->usize{4}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Infallible>{panic!("unbounded voted transport")}
}
impl BoundedConsensusTransport for VotedRanks{
    type Completion=Done;type GatherOutput=();
    fn submit_all_gather_words(&self,_:&[u32])->Result<Submission<(),Done>,Infallible>{panic!("untyped voted transport")}
    fn resolve_all_gather_words(&self,_:())->Result<Vec<u32>,Infallible>{panic!("untyped voted output")}
}
impl PartitionCaptureTransport for VotedRanks{
    fn capture_rank(&self)->usize{self.base.capture_rank()}
    fn capture_wait(&self)->Result<BoundedCompletionWait,CaptureError>{self.base.capture_wait()}
    fn ensure_capture_active(&self)->Result<(),BackendFailure>{Ok(())}
    fn estimate_capture_gather(&self,n:usize)->Result<CaptureUsage,CaptureError>{self.base.estimate_capture_gather(n)}
    fn fail_capture_exchange(&self,e:&PartitionCaptureExchangeError){
        assert!(self.reject_source,"an agreeing source frame must not invalidate transport");
        assert!(matches!(e,PartitionCaptureExchangeError::Protocol("selected producer scalar is missing or inconsistent")));
        self.failures.set(self.failures.get()+1);
    }
    fn capture_word_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{self.base.capture_word_destination(n)}
    fn capture_byte_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u8>,PartitionCaptureExchangeError>{self.base.capture_byte_destination(n)}
    fn gather_capture_frame(&self,frame:&PartitionCaptureFrame<'_>,wait:BoundedCompletionWait)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{
        if frame.kind()!=PartitionCaptureFrameKind::Source{return self.base.gather_capture_frame(frame,wait);}
        let mut out=PartitionCaptureBuffer::funded(frame.gathered_words(),&self.base.0.funding)?;
        for rank in 0..4{
            let mut words:[u32;16]=frame.words().try_into().unwrap();words[3]=rank as u32;
            words[13]=if rank==0||rank==3||rank==self.capture_rank(){1}else{0};
            if rank==2{words[13]=0;}
            if self.reject_source&&rank==3{words[13]=3;}
            out.extend_from_slice(&words)?;
        }
        Ok(out)
    }
}
#[test]
fn late_source_vote_binds_original_host_and_preserves_absent_and_replica_sources(){
    use super::super::prefill::{projected_source,projected_receipt,inference};
    use crate::capture::partition::{PartitionCaptureFragmentGeometry,PartitionCaptureLocalSource};
    for local in [0,1,2]{for reject_source in [false,true]{
        let source=projected_source(CaptureTransform::Summary);let (metadata,_,_,_)=funding();
        let transport=VotedRanks{base:ProgramRanks(Ranks{local,bytes:RefCell::new(std::array::from_fn(|_|None)),
            funding:metadata.clone(),reject:false,calls:RefCell::new(vec![])}),reject_source,failures:std::cell::Cell::new(0)};
        let mut quotation=CaptureLedger::new(source.admission());quotation.begin_step();
        let mut prototype=projected_receipt(&source,PartitionCaptureCombination::SumF64ToF32,&metadata,&mut quotation);
        let geometry:Vec<_>=prototype.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentGeometry{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&raw,estimate:estimate(*rank)}).collect();
        let quoted=crate::capture::partition::PreparedPartitionFragmentSourceAllowance::prepare(&transport,&mut prototype,&rows,&metadata,&mut quotation).unwrap();
        drop(quoted);
        let producers=[PartitionCaptureContiguousProducer{rank:0,coordinates:0..7},PartitionCaptureContiguousProducer{rank:3,coordinates:0..7}];
        let physical=if local==2{None}else{Some(PartitionCaptureLocalSource{producer:local,shape:&[2,3,7],dtype:TensorDtype::F16})};
        let retained=PreparedPartitionContiguousSource::new_local(&source,0,2,&producers,&rows,physical,
            PartitionCaptureCombination::SumF64ToF32,inference(),&metadata).unwrap();
        let h=PartitionFragmentHostPlan::prepare_prefill(&prototype,inference()).unwrap().initialization_peak_bytes()+plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let host=run.prepare_partition_fragment_host(&reservation,prototype,Some(inference())).unwrap();
        let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(inference()).unwrap();
        let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10};
        let mut program=PreparedPartitionCaptureProgram::new_selected(&transport,&source,&context(&source,0),
            &[PreparedPartitionCaptureRow::Contiguous],limits,&metadata).unwrap().with_contiguous_row(0,retained,host).unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let epoch=DistributedCommitEpoch::new(17).unwrap();
        program.prepare(&source,CapturePhase::Prefill,0,epoch,&mut frame,&mut quota).unwrap();
        let spent=quota.total();let result=program.coordinate(epoch,&quota);
        assert_eq!(transport.failures.get(),usize::from(reject_source),"source mismatch poisons transport exactly once");
        assert_eq!(quota.total(),spent,"precision binding does not reserve or refund global work");
        if !reject_source{
            result.unwrap();
            if local!=0{
                let receiver=program.take_projected_receiver_source(0,inference(),0).unwrap();
                assert_eq!(receiver.is_some(),local==1);
                if let Some(receiver)=receiver{assert_eq!(receiver.source_shape(),[2,1,7]);assert_eq!(receiver.dtype(),&TensorDtype::F16);}
            }
            drop(program);drop(frame);drop(bank);drop(run);drop(reservation);
            assert_eq!(pool.used_bytes().unwrap(),0);
        }else{
            let error=result.unwrap_err();drop(program);drop(frame);drop(bank);drop(run);drop(reservation);
            assert!(pool.used_bytes().unwrap()>0,"failed Source vote retains original fragment Host");
            drop(error);assert_eq!(pool.used_bytes().unwrap(),0);
        }
    }}
}

#[test]
fn inactive_projected_selection_needs_no_fabricated_source_or_host_owner() {
    use super::super::prefill::projected_source;
    let source=projected_source(CaptureTransform::Summary);let (metadata,_,_,_)=funding();
    let transport=ProgramRanks(Ranks{local:2,bytes:RefCell::new(std::array::from_fn(|_|None)),
        funding:metadata.clone(),reject:false,calls:RefCell::new(vec![])});
    let limits=PartitionCaptureReceiptLimits{max_producers:4,max_fragments:4,max_record_bytes:64<<10};
    let mut coordinate=context(&source,0);
    assert!(PreparedPartitionCaptureProgram::new_selected(&transport,&source,&coordinate,
        &[PreparedPartitionCaptureRow::Inactive],limits,&metadata).is_err(),"active prefill still requires its actual source");
    coordinate.phase=CapturePhase::Decode;coordinate.prediction=1;
    let mut program=PreparedPartitionCaptureProgram::new_selected(&transport,&source,&coordinate,
        &[PreparedPartitionCaptureRow::Inactive],limits,&metadata).unwrap();
    let h=plan(&source).initialization_peak_bytes();let pool=WorkingMemoryPool::new(h,0).unwrap();
    let (reservation,run)=fresh(&pool,h);let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
    // The finite run starts at prediction zero. Retiring its initial claim
    // spends that coordinate; it cannot be skipped by requesting decode one.
    drop(bank.begin_step(CapturePhase::Prefill,0).unwrap());
    assert_eq!(bank.spent_steps(),1);
    let mut frame=bank.begin_step(CapturePhase::Decode,1).unwrap().prepare().unwrap();
    let mut ledger=CaptureLedger::new(source.admission());ledger.begin_step();let epoch=DistributedCommitEpoch::new(29).unwrap();
    program.prepare(&source,CapturePhase::Decode,1,epoch,&mut frame,&mut ledger).unwrap();
    program.coordinate(epoch,&ledger).unwrap();
    assert_eq!(*transport.0.calls.borrow(),[PartitionCaptureFrameKind::Coordination]);
    assert!(frame.records().iter().all(|record|record.payload.is_none()));
    drop(program);drop(frame);drop(bank);drop(run);drop(reservation);assert_eq!(pool.used_bytes().unwrap(),0);
}

struct SkippedRanks { base:ProgramRanks, reject:bool, failures:std::cell::Cell<usize> }
impl ConsensusTransport for SkippedRanks {
    type Error=Infallible;
    fn participant_count(&self)->usize{4}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Infallible>{panic!("unbounded skipped transport")}
}
impl BoundedConsensusTransport for SkippedRanks {
    type Completion=Done;type GatherOutput=();
    fn submit_all_gather_words(&self,_:&[u32])->Result<Submission<(),Done>,Infallible>{panic!("untyped skipped transport")}
    fn resolve_all_gather_words(&self,_:())->Result<Vec<u32>,Infallible>{panic!("untyped skipped output")}
}
impl PartitionCaptureTransport for SkippedRanks {
    fn capture_rank(&self)->usize{self.base.capture_rank()}
    fn capture_wait(&self)->Result<BoundedCompletionWait,CaptureError>{self.base.capture_wait()}
    fn ensure_capture_active(&self)->Result<(),BackendFailure>{Ok(())}
    fn estimate_capture_gather(&self,n:usize)->Result<CaptureUsage,CaptureError>{self.base.estimate_capture_gather(n)}
    fn fail_capture_exchange(&self,cause:&PartitionCaptureExchangeError){
        assert!(self.reject);
        assert!(matches!(cause,PartitionCaptureExchangeError::Protocol("skipped source has an active scalar")));
        self.failures.set(self.failures.get()+1);
    }
    fn capture_word_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{self.base.capture_word_destination(n)}
    fn capture_byte_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u8>,PartitionCaptureExchangeError>{self.base.capture_byte_destination(n)}
    fn gather_capture_frame(&self,frame:&PartitionCaptureFrame<'_>,wait:BoundedCompletionWait)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{
        if frame.kind()!=PartitionCaptureFrameKind::Source{return self.base.gather_capture_frame(frame,wait)}
        self.base.0.calls.borrow_mut().push(frame.kind());
        assert_eq!(frame.words()[13],0,"skipped rows never advertise a scalar");
        let mut out=PartitionCaptureBuffer::funded(frame.gathered_words(),&self.base.0.funding)?;
        for rank in 0..4 {
            let mut words:[u32;16]=frame.words().try_into().unwrap();words[3]=rank as u32;
            if self.reject&&rank==3{words[13]=1;}
            out.extend_from_slice(&words)?;
        }
        Ok(out)
    }
}

#[test]
fn projected_limit_policy_coordinates_skip_and_retains_original_host_on_failure() {
    use super::super::prefill::{projected_source_limits,projected_receipt,inference};
    use crate::capture::partition::{PartitionCaptureFragmentGeometry,PartitionCaptureLocalSource};
    for (policy,reject,foreign_geometry) in [
        (CaptureLimitPolicy::Fail,false,false),
        (CaptureLimitPolicy::Skip,false,false),
        (CaptureLimitPolicy::Skip,true,false),
        (CaptureLimitPolicy::Skip,false,true),
    ] {
        let source=projected_source_limits(CaptureTransform::Summary,Some((policy,1<<20)));
        let (metadata,_,_,_)=funding();
        let transport=SkippedRanks{base:ProgramRanks(Ranks{local:0,bytes:RefCell::new(std::array::from_fn(|_|None)),
            funding:metadata.clone(),reject:false,calls:RefCell::new(vec![])}),reject,failures:std::cell::Cell::new(0)};
        let mut quotation=CaptureLedger::new(source.admission());quotation.begin_step();
        let prototype=projected_receipt(&source,PartitionCaptureCombination::SumF64ToF32,&metadata,&mut quotation);
        let mut geometry:Vec<_>=prototype.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        if foreign_geometry{geometry[0].3.ends[2]-=1;}
        let raw=CaptureTransform::Slice;
        let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|{
            let mut estimate=estimate(*rank);estimate.capture.retained_bytes=2<<20;
            PartitionCaptureFragmentGeometry{producer:*rank,fragment:*fragment,
                local_shape:shape,local_slice:slice,transform:&raw,estimate}
        }).collect();
        let producers=[PartitionCaptureContiguousProducer{rank:0,coordinates:0..7},PartitionCaptureContiguousProducer{rank:3,coordinates:0..7}];
        let retained=PreparedPartitionContiguousSource::new_local(&source,0,2,&producers,&rows,
            Some(PartitionCaptureLocalSource{producer:0,shape:&[2,3,7],dtype:TensorDtype::F16}),
            PartitionCaptureCombination::SumF64ToF32,inference(),&metadata).unwrap();
        let h=PartitionFragmentHostPlan::prepare_prefill(&prototype,inference()).unwrap().initialization_peak_bytes()
            +plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let host=run.prepare_partition_fragment_host(&reservation,prototype,Some(inference())).unwrap();
        let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(inference()).unwrap();
        let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10};
        let mut program=PreparedPartitionCaptureProgram::new_selected(&transport,&source,&context(&source,0),
            &[PreparedPartitionCaptureRow::Contiguous],limits,&metadata).unwrap().with_contiguous_row(0,retained,host).unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let epoch=DistributedCommitEpoch::new(41).unwrap();
        let prepared=program.prepare(&source,CapturePhase::Prefill,0,epoch,&mut frame,&mut quota);
        let spent=quota.total();assert!(spent.host_bytes>0,"the initial coordination remains spent");
        let failure=if policy==CaptureLimitPolicy::Fail||foreign_geometry {
            assert!(transport.base.0.calls.borrow().is_empty());
            Some(prepared.unwrap_err())
        }else{
            prepared.unwrap();
            assert!(matches!(&frame.records()[0].outcome,CaptureOutcome::Skipped{
                reason:CaptureSkipReason::Limit{budget:CaptureBudget::Retention,cumulative:false}}));
            let coordinated=program.coordinate(epoch,&quota);
            if reject {Some(coordinated.unwrap_err())} else {
                coordinated.unwrap();assert!(!program.produces(0).unwrap());
                assert!(program.take_prefill_projection(0,&mut frame,true).unwrap().is_none());
                assert!(program.take_projected_receiver_source(0,inference(),0).unwrap().is_none());
                program.deliver(&mut frame).unwrap();
                assert_eq!(*transport.base.0.calls.borrow(),[PartitionCaptureFrameKind::Source,PartitionCaptureFrameKind::Coordination]);
                assert!(frame.records()[0].payload.is_none());None
            }
        };
        assert_eq!(transport.failures.get(),usize::from(reject));
        assert_eq!(quota.total(),spent,"skips, refusals and votes never refund earlier credits");
        drop(program);drop(frame);drop(bank);drop(run);drop(reservation);
        if let Some(failure)=failure {
            assert!(pool.used_bytes().unwrap()>0,"the preparation or vote failure owns original Host");
            drop(failure);
        }
        assert_eq!(pool.used_bytes().unwrap(),0);
        assert_eq!(quota.total(),spent);
    }
}

mod decode;
