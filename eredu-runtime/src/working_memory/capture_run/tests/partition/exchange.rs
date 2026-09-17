//! Closed original destination plus the real shared preparation/delivery votes.
mod vocabulary;
mod histogram;
use super::*;
use crate::capture::partition::*;
use eredu_core::component::ComponentCoordinateMap;
use eredu_core::consensus::{BoundedConsensusTransport, ConsensusTransport};
use eredu_core::{BackendFailure, BoundedCompletion, BoundedCompletionOutcome,
    BoundedCompletionWait, Completion, CompletionCancellationMode, Submission};
use std::{cell::RefCell, convert::Infallible};

struct Done;
impl Completion for Done {
    type Error=Infallible;
    fn is_complete(&self)->Result<bool,Infallible>{Ok(true)}
    fn wait(&self)->Result<(),Infallible>{Ok(())}
}
impl BoundedCompletion for Done {
    fn wait_bounded(self,_:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Infallible>{Ok(BoundedCompletionOutcome::Completed)}
}
struct Receipts {
    source:Vec<u8>,funding:WorkspaceMetadataFunding,reject_delivery:bool,
    calls:RefCell<Vec<PartitionCaptureFrameKind>>,
}
impl ConsensusTransport for Receipts {
    type Error=Infallible;
    fn participant_count(&self)->usize{4}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Infallible>{panic!("unbounded transport")}
}
impl BoundedConsensusTransport for Receipts {
    type Completion=Done;
    type GatherOutput=();
    fn submit_all_gather_words(&self,_:&[u32])->Result<Submission<(),Done>,Infallible>{panic!("untyped frame")}
    fn resolve_all_gather_words(&self,_:())->Result<Vec<u32>,Infallible>{panic!("untyped output")}
}
impl PartitionCaptureTransport for Receipts {
    fn capture_rank(&self)->usize{0}
    fn capture_wait(&self)->Result<BoundedCompletionWait,CaptureError>{
        Ok(BoundedCompletionWait::new(std::time::Duration::from_secs(1),CompletionCancellationMode::QuarantineUntilComplete).unwrap())
    }
    fn ensure_capture_active(&self)->Result<(),BackendFailure>{Ok(())}
    fn estimate_capture_gather(&self,_:usize)->Result<CaptureUsage,CaptureError>{Ok(CaptureUsage::default())}
    fn fail_capture_exchange(&self,_:&PartitionCaptureExchangeError){panic!("completed refusal is not unagreed transport failure")}
    fn capture_word_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{Ok(PartitionCaptureBuffer::funded(n,&self.funding)?)}
    fn capture_byte_destination(&self,n:usize)->Result<PartitionCaptureBuffer<u8>,PartitionCaptureExchangeError>{Ok(PartitionCaptureBuffer::funded(n,&self.funding)?)}
    fn gather_capture_frame(&self,frame:&PartitionCaptureFrame<'_>,_:BoundedCompletionWait)
        ->Result<PartitionCaptureBuffer<u32>,PartitionCaptureExchangeError>{
        self.calls.borrow_mut().push(frame.kind());
        let mut output=PartitionCaptureBuffer::funded(frame.gathered_words(),&self.funding)?;
        match frame.kind() {
            PartitionCaptureFrameKind::Preparation|PartitionCaptureFrameKind::Delivery=>{
                for rank in 0..4 {
                    let mut words:[u32;16]=frame.words().try_into().unwrap();words[3]=rank;
                    if frame.kind()==PartitionCaptureFrameKind::Preparation {
                        words[13]=if rank==3{2}else{1};
                        let length=if rank==3{self.source.len() as u64}else{0};
                        words[14]=length as u32;words[15]=(length>>32) as u32;
                    } else if self.reject_delivery && rank==1 {words[13]=0;}
                    output.extend_from_slice(&words)?;
                }
            }
            PartitionCaptureFrameKind::Payload=>{
                for rank in 0..4 {for index in 0..frame.words().len() {
                    let mut bytes=[0;4];
                    if rank==3 {for (offset,byte) in bytes.iter_mut().enumerate(){
                        *byte=self.source.get(index*4+offset).copied().unwrap_or(0);
                    }}
                    output.extend_from_slice(&[u32::from_le_bytes(bytes)])?;
                }}
            }
            _=>panic!("unexpected protocol phase"),
        }
        Ok(output)
    }
}

#[test]
fn scheduled_partition_exchange_keeps_decoded_h_until_final_delivery_vote() {
    for reject in [false,true] {
        let source=source();let h=plan(&source).initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
        let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
        let context=context(&source,0);
        let geometry=CaptureTensorGeometry::prepare(source.admission(),0,CapturePhase::Prefill,0,None).unwrap();
        let shape:Vec<u64>=geometry.source_shape().iter().map(|n|*n as u64).collect();
        let slice=resolve_slice(&source.admission().points()[0],&source.admission().plan().selections[0],&shape).unwrap();
        let coordinates=ComponentCoordinateMap::range(shape[0] as usize,0..shape[0] as usize).unwrap();
        let projection=CaptureSlicePartition::new(&shape,&slice,0,&coordinates,1).unwrap();
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let receipt=PartitionCaptureReceiptPlan::new_shared(source.clone(),context.clone(),
            vec![PartitionCaptureProducer{rank:3,projection}],4,
            PartitionCaptureReceiptLimits{max_producers:1,max_fragments:1,max_record_bytes:64<<10},&mut quota).unwrap();
        let (bytes,charged,expected)=wire(frame.records()[0].clone(),&geometry,&context);
        let bytes=String::from_utf8(bytes).unwrap().replace("retained-receipt",receipt.identity()).into_bytes();
        let (funding,used,_,retired)=funding();
        let transport=Receipts {source:bytes,funding:funding.clone(),reject_delivery:reject,calls:RefCell::new(Vec::new())};
        frame.prepare_partition_evidence(&funding).unwrap();
        let identity=receipt.identity().to_owned();
        let evidence=PreparedPartitionCaptureEvidence::prepare(&receipt,charged,&funding,&mut quota).unwrap();
        let exchange=PartitionCaptureExchange::admit(&transport,receipt,&mut quota).unwrap();
        let prepared=PreparedPartitionTensorDelivery::prepare(exchange,TensorDtype::F16,charged,&funding).unwrap()
            .with_evidence(evidence);
        let result=prepared.deliver(&mut frame);
        assert_eq!(frame.records()[0].charged, charged);
        assert_eq!(*transport.calls.borrow(),vec![PartitionCaptureFrameKind::Preparation,PartitionCaptureFrameKind::Payload,PartitionCaptureFrameKind::Delivery]);
        let Some(CapturePayload::SharedTensor(value))=&frame.records()[0].payload else {panic!("original host destination")};
        let TensorObservationData::F32(actual)=value.data() else {panic!("f32")};
        for (a,b) in actual.iter().zip(expected){assert!(a.to_bits()==b.to_bits() || a.is_nan() && b.is_nan());}
        assert!(frame.take_tensor(0).is_err());assert!(used.load(Ordering::SeqCst)>0);
        assert_eq!(result.is_err(),reject);
        assert_eq!(frame.partition_evidence().len(),1);
        let evidence=&frame.partition_evidence()[0];
        assert_eq!(evidence.context,context);assert_eq!(evidence.receipt_plan_identity,identity);
        assert_eq!(evidence.producers,vec![3]);assert_eq!(evidence.contributions[0].charged,charged);
        assert_eq!(evidence.contributions[0].local.starts,geometry.starts());
        // Completed transport may publish only an aborted frame after a rejected
        // final vote. Both outcomes exercise the same escaped metadata custody.
        let delivered=frame.finish(CaptureStepOutcome::Aborted,unlimited(),unlimited(),0.0).unwrap();
        let alias=delivered.clone();
        drop(bank);drop(run);drop(reservation);drop(transport);drop(funding);drop(result);
        assert_eq!(pool.used_bytes().unwrap(),h);assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(delivered.partitions()[0].context,context);
        drop(delivered);assert!(!retired.load(Ordering::SeqCst));
        drop(alias);assert_eq!(pool.used_bytes().unwrap(),0);assert!(retired.load(Ordering::SeqCst));
    }
}


// One peer is sufficient to exercise the real source/final-vote transaction
// and its failure custody. No native work or independent receipt worker runs.
struct ProgramVotes {
    funding: WorkspaceMetadataFunding,
    fail_at: Option<usize>,
    participants: usize,
    calls: RefCell<Vec<PartitionCaptureFrameKind>>,
    failed: Cell<bool>,
}
impl ConsensusTransport for ProgramVotes {
    type Error = Infallible;
    fn participant_count(&self) -> usize { self.participants }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Infallible> { panic!("unbounded transport") }
}
impl BoundedConsensusTransport for ProgramVotes {
    type Completion = Done;
    type GatherOutput = ();
    fn submit_all_gather_words(&self, _: &[u32]) -> Result<Submission<(), Done>, Infallible> { panic!("untyped frame") }
    fn resolve_all_gather_words(&self, _: ()) -> Result<Vec<u32>, Infallible> { panic!("untyped output") }
}
impl PartitionCaptureTransport for ProgramVotes {
    fn capture_rank(&self) -> usize { 0 }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(std::time::Duration::from_secs(1), CompletionCancellationMode::QuarantineUntilComplete).unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> { Ok(()) }
    fn estimate_capture_gather(&self, _: usize) -> Result<CaptureUsage, CaptureError> { Ok(CaptureUsage::default()) }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) { self.failed.set(true); }
    fn capture_word_destination(&self, count: usize) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(count, &self.funding)?)
    }
    fn capture_byte_destination(&self, count: usize) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::funded(count, &self.funding)?)
    }
    fn gather_capture_frame(&self, frame: &PartitionCaptureFrame<'_>, _: BoundedCompletionWait)
        -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        assert!(matches!(frame.kind(), PartitionCaptureFrameKind::Source | PartitionCaptureFrameKind::Coordination));
        let ordinal = self.calls.borrow().len();
        self.calls.borrow_mut().push(frame.kind());
        if self.fail_at == Some(ordinal) {
            return Err(PartitionCaptureExchangeError::Protocol("injected completed vote failure"));
        }
        let mut words = PartitionCaptureBuffer::funded(frame.gathered_words(), &self.funding)?;
        for rank in 0..self.participants {
            let mut header: [u32; 16] = frame.words().try_into().unwrap();
            // Source receipts use the shared payload protocol; the final
            // coordination transcript has its own versioned rank/world prefix.
            match frame.kind() {
                PartitionCaptureFrameKind::Source => header[3] = rank as u32,
                PartitionCaptureFrameKind::Coordination => header[2] = rank as u32,
                _ => unreachable!("validated vote frame"),
            }
            words.extend_from_slice(&header)?;
        }
        Ok(words)
    }
}

#[test]
fn failed_partition_source_or_final_vote_spends_epoch_and_disables_capture_delivery() {
    // The first failure occurs after one row has already resolved its source;
    // the second fails the final vote after both sources resolved. The final
    // two cases invalidate a successful prior vote with duplicate/older epochs.
    for case in 0..4 {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
        let mut frame = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
        let (funding, used, _, retired) = funding();
        let transport = ProgramVotes {
            funding: funding.clone(), fail_at: (case < 2).then_some(case + 1), participants: 1,
            calls: RefCell::new(Vec::new()), failed: Cell::new(false),
        };
        let rows: Vec<_> = source.admission().plan().selections.iter().map(|_| PartitionCaptureProducerSource {
            producer: 0, dtype: Some(TensorDtype::F32),
            estimate: PartitionCaptureNativeEstimate { capture: native_usage(), generated_creation_bytes: 0 },
        }).collect();
        let mut program = PreparedPartitionCaptureProgram::new(&transport, &source, &context(&source, 0),
            &rows, PartitionCaptureReceiptLimits { max_producers: 1, max_fragments: 1, max_record_bytes: 64 << 10 },
            &funding).unwrap();
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let epoch = DistributedCommitEpoch::FIRST.next().unwrap();
        program.prepare(&source, CapturePhase::Prefill, 0, epoch, &mut frame, &mut ledger).unwrap();
        assert!(transport.calls.borrow().is_empty());
        let charged = ledger.total();
        let first_error = if case < 2 {
            program.coordinate(epoch, &ledger).unwrap_err()
        } else {
            program.coordinate(epoch, &ledger).unwrap();
            assert!(program.produces(0).unwrap());
            assert!(program.reservation(0, &TensorDtype::F32, native_usage()).is_ok());
            program.coordinate(if case == 2 { epoch } else { DistributedCommitEpoch::FIRST }, &ledger).unwrap_err()
        };
        let calls = transport.calls.borrow().clone();
        assert_eq!(calls, if case == 0 {
            vec![PartitionCaptureFrameKind::Source; 2]
        } else {
            vec![PartitionCaptureFrameKind::Source, PartitionCaptureFrameKind::Source, PartitionCaptureFrameKind::Coordination]
        });
        assert_eq!(transport.failed.get(), case < 2);
        assert!(program.produces(0).is_err());
        assert!(program.take_receiver_source(0).is_err());
        assert!(program.reservation(0, &TensorDtype::F32, native_usage()).is_err());
        assert!(program.deliver(&mut frame).is_err());
        assert!(program.coordinate(epoch, &ledger).is_err());
        assert!(program.coordinate(epoch.next().unwrap(), &ledger).is_err());
        assert_eq!(*transport.calls.borrow(), calls, "refused epochs cannot issue another frame");
        assert_eq!(ledger.total(), charged, "failed attempts retain every reserved credit");
        assert!(frame.records().iter().all(|record| record.payload.is_none()));
        assert!(used.load(Ordering::SeqCst) > 0);
        drop(program); drop(frame); drop(bank); drop(run); drop(reservation);
        drop(transport); drop(funding);
        assert!(!retired.load(Ordering::SeqCst), "escaped error retains paid preparation custody");
        drop(first_error);
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}


#[test]
fn receiver_source_readiness_spends_each_coordinated_epoch_without_refunding() {
    for fail_vote in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
        let mut frame = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
        let (funding, used, _, retired) = funding();
        let transport = ProgramVotes { funding: funding.clone(), fail_at: fail_vote.then_some(2),
            participants: 2, calls: RefCell::new(Vec::new()), failed: Cell::new(false) };
        let rows: Vec<_> = source.admission().plan().selections.iter().enumerate()
            .map(|(index, _)| PartitionCaptureProducerSource {
                producer: if index == 2 { 0 } else { 1 }, dtype: Some(TensorDtype::F16),
                estimate: PartitionCaptureNativeEstimate { capture: native_usage(), generated_creation_bytes: 0 },
            }).collect();
        let mut program = PreparedPartitionCaptureProgram::new(&transport, &source, &context(&source, 0),
            &rows, PartitionCaptureReceiptLimits { max_producers: 1, max_fragments: 1, max_record_bytes: 64 << 10 },
            &funding).unwrap();
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let first = DistributedCommitEpoch::FIRST.next().unwrap();
        program.prepare(&source, CapturePhase::Prefill, 0, first, &mut frame, &mut ledger).unwrap();
        let charged = ledger.total();
        let before = used.load(Ordering::SeqCst);
        let error = if fail_vote {
            let error = program.coordinate(first, &ledger).unwrap_err();
            assert!(program.take_receiver_source(0).is_err());
            assert!(program.coordinate(first.next().unwrap(), &ledger).is_err());
            error
        } else {
            program.coordinate(first, &ledger).unwrap();
            assert_eq!(program.take_receiver_source(2).unwrap(), None, "producer owns its transform's source completion");
            assert_eq!(program.take_receiver_source(1).unwrap(), None, "inactive selection has no dependency work");
            assert_eq!(program.take_receiver_source(0).unwrap(), Some(TensorDtype::F16));
            let duplicate = program.take_receiver_source(0).unwrap_err();
            let spent = used.load(Ordering::SeqCst);
            drop(duplicate);
            assert_eq!(used.load(Ordering::SeqCst), spent, "dropping refusal does not refund controls");
            let next = first.next().unwrap();
            program.coordinate(next, &ledger).unwrap();
            assert_eq!(program.take_receiver_source(0).unwrap(), Some(TensorDtype::F16));
            assert!(program.take_receiver_source(0).is_err());
            let error = program.coordinate(next, &ledger).unwrap_err();
            assert!(program.take_receiver_source(0).is_err(), "failed epoch cannot reuse prior readiness");
            assert!(program.coordinate(next.next().unwrap(), &ledger).is_err());
            error
        };
        assert_eq!(ledger.total(), charged, "source readiness never refunds the original capture budget");
        assert!(used.load(Ordering::SeqCst) > before);
        assert_eq!(*transport.calls.borrow(), vec![PartitionCaptureFrameKind::Source,
            PartitionCaptureFrameKind::Source, PartitionCaptureFrameKind::Coordination]);
        assert!(frame.records().iter().all(|record| record.payload.is_none()));
        drop(program); drop(frame); drop(bank); drop(run); drop(reservation);
        drop(transport); drop(funding);
        assert!(!retired.load(Ordering::SeqCst), "escaped refusal retains original source and H");
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

mod fragments;
