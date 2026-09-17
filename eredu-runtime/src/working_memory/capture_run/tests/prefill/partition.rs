//! Remote targets use the same chunk cursor and one original whole-run H.
use super::*;
use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
use crate::capture::partition::PartitionCaptureRecordEncoding;
use eredu_nn::workspace::{WorkspaceMetadataAccount, WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

#[derive(Debug)]
struct Metadata(Arc<AtomicUsize>);
impl WorkspaceMetadataAccount for Metadata {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        self.0.fetch_add(bytes, Ordering::SeqCst); Ok(())
    }
}
fn prompt() -> InferenceGeometry { InferenceGeometry { prefill_chunk_positions:2, ..geometry() } }
fn context(source: &SharedCapturePlan, index:usize) -> PartitionCaptureContext {
    PartitionCaptureContext {
        artifact_identity:"remote-artifact".into(), execution_identity:"parallel".into(),
        run_identity:"run-é".into(), overlay_identity:None,
        capture_plan_identity:source.admission().identity().into(), selection_index:index,
        phase:CapturePhase::Prefill, prediction:0, forward_epoch:19, invocation:None,
    }
}
fn advance(step:&mut ScheduledCaptureStep<'_>, source:&SharedCapturePlan,
    quota:&mut CaptureLedger, remote:bool, k:u64) {
    let policy=CapturePrefillObservationPolicy::new(source,prompt()).unwrap();
    for index in 0..2 {
        let row=policy.row(index).unwrap();
        let assembly=row.assembly().unwrap();
        let fragment=assembly.fragment(k).unwrap();
        let chunk=crate::prefill::PrefillChunk { input:fragment.input().clone(),
            position:fragment.position(), output:fragment.output_demand() };
        let decision=step.begin_prefill_hook(index,&chunk,"block.output").unwrap();
        if decision==CapturePrefillHookDecision::First {
            assert!(step.reserve_prefill_hook(index,quota,TensorDtype::F32,charge()).unwrap().is_none());
            if remote { step.mark_remote_prefill_target(index).unwrap(); }
        }
        if remote {
            assert!(step.prefill_storage(index).is_none());
            assert!(step.take_prefill_fragment(index,&fragment).is_err());
        } else { write_fragment(step,index,assembly,k); }
        step.finish_prefill_hook(index,&fragment).unwrap();
    }
    step.complete_prefill_chunk(k).unwrap();
}

#[test]
fn remote_prefill_receipt_reuses_original_h_after_real_chunk_progress() {
    let source=rows(5); let h=plan(&source).initialization_peak_bytes();
    let producer_pool=WorkingMemoryPool::new(h,0).unwrap();
    let receiver_pool=WorkingMemoryPool::new(h,0).unwrap();
    let (pr,run_p)=fresh(&producer_pool,h); let (rr,run_r)=fresh(&receiver_pool,h);
    let mut pb=run_p.prepare_capture_run(&pr,plan(&source)).unwrap();
    let mut rb=run_r.prepare_capture_run(&rr,plan(&source)).unwrap();
    let mut producer=pb.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(prompt()).unwrap();
    let mut receiver=rb.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(prompt()).unwrap();
    let mut pquota=CaptureLedger::new(source.admission());pquota.begin_step();
    let mut rquota=CaptureLedger::new(source.admission());rquota.begin_step();
    for k in 0..2 {
        advance(&mut producer,&source,&mut pquota,false,k);
        advance(&mut receiver,&source,&mut rquota,true,k);
        if k==0 { assert!(receiver.take_remote_prefill_tensor(0).is_err()); }
    }
    assert_eq!(pquota.total(),rquota.total());
    producer.finish_prefill_targets().unwrap();
    receiver.finish_local_prefill_targets().unwrap();
    assert!(receiver.finish_prefill_targets().is_err());
    let used=Arc::new(AtomicUsize::new(0));
    let funding=WorkspaceMetadataFunding::new(Metadata(used.clone())).unwrap();
    for index in 0..2 {
        let context=context(&source,index);let record=&producer.records()[index];
        let bytes=PartitionCaptureRecordEncoding::new(&context,"exact-receipt",3,
            PartitionCaptureCombination::Disjoint,record,64<<10).encode(&funding).unwrap();
        let claim=receiver.take_remote_prefill_tensor(index).unwrap();
        let decoded=claim.decode_partition_receipt(&bytes,PartitionCaptureTensorReceipt {
            context:&context, identity:"exact-receipt", producer:3, dtype:TensorDtype::F32,
            charged:record.charged,
        },&funding).unwrap();
        receiver.record_remote_prefill_tensor(decoded,TensorDtype::F32).unwrap();
        assert!(receiver.take_remote_prefill_tensor(index).is_err());
        assert_eq!(values(tensor(&receiver.records()[index])),values(tensor(record)));
        if index==0 { assert!(receiver.finish_prefill_targets().is_err()); }
    }
    receiver.finish_prefill_targets().unwrap();
    assert!(used.load(Ordering::SeqCst)>0);
    assert_eq!(charged(&receiver),charged(&producer));
    let escaped=tensor(&receiver.records()[0]).clone();
    drop(receiver);drop(rb);drop(run_r);drop(rr);
    assert_eq!(receiver_pool.used_bytes().unwrap(),h);
    drop(escaped);assert_eq!(receiver_pool.used_bytes().unwrap(),0);
    drop(producer);drop(pb);drop(run_p);drop(pr);
    assert_eq!(producer_pool.used_bytes().unwrap(),0);
}

#[test]
fn remote_prefill_dropped_final_claim_cannot_refund_or_finish_target() {
    let source=rows(5);let h=plan(&source).initialization_peak_bytes();
    let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
    let mut bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
    let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill_with_progression(prompt()).unwrap();
    let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
    for k in 0..2 { advance(&mut frame,&source,&mut quota,true,k); }
    let spent=quota.total();
    drop(frame.take_remote_prefill_tensor(0).unwrap());
    assert!(frame.take_remote_prefill_tensor(0).is_err());
    assert!(frame.finish_local_prefill_targets().is_err());
    assert!(frame.finish_prefill_targets().is_err());
    assert_eq!(quota.total(),spent);assert_eq!(pool.used_bytes().unwrap(),h);
    drop(frame);drop(bank);drop(run);drop(reservation);
    assert_eq!(pool.used_bytes().unwrap(),0);
}
