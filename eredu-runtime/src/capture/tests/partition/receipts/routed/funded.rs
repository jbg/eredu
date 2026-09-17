use super::*;
use eredu_core::capture::SharedCapturePlan;
use crate::capture::partition::PartitionCaptureRoutedProducerSource;
use eredu_nn::workspace::{WorkspaceMetadataAccount,WorkspaceMetadataFunding,WorkspaceMetadataFundingError};
use std::sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}};
#[derive(Debug)]
struct Account {used:Arc<AtomicUsize>,refuse:Arc<AtomicBool>,retired:Arc<AtomicBool>}
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),WorkspaceMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {return Err(WorkspaceMetadataFundingError::Overflow);}
        self.used.fetch_add(bytes,Ordering::SeqCst);Ok(())
    }
}
impl Drop for Account {fn drop(&mut self){self.retired.store(true,Ordering::SeqCst);}}
fn funding()->(WorkspaceMetadataFunding,Arc<AtomicUsize>,Arc<AtomicBool>,Arc<AtomicBool>) {
    let used=Arc::new(AtomicUsize::new(0));let refuse=Arc::new(AtomicBool::new(false));let retired=Arc::new(AtomicBool::new(false));
    (WorkspaceMetadataFunding::new(Account{used:used.clone(),refuse:refuse.clone(),retired:retired.clone()}).unwrap(),used,refuse,retired)
}
#[test]
fn funded_routed_receipts_preserve_ordinary_permuted_ownership_identity_records_and_custody() {
    for empty in [false,true] {for exchange in [false,true] {
        let source=SharedCapturePlan::new(plan(empty));
        let mut ordinary_ledger=CaptureLedger::new(source.admission());ordinary_ledger.begin_step();
        let ordinary=PartitionCaptureReceiptPlan::new_routed_shared(source.clone(),context(source.admission()),
            producers(source.admission(),exchange),8,limits(),&mut ordinary_ledger).unwrap();
        let mut declarations=producers(source.admission(),exchange);declarations.reverse();
        let rows:Vec<_>=declarations.iter().map(|row|PartitionCaptureRoutedProducerSource{rank:row.rank,ownership:&row.ownership}).collect();
        let mut ledger=CaptureLedger::new(source.admission());ledger.begin_step();
        let (funding,used,_,retired)=funding();
        let actual=PartitionCaptureReceiptPlan::new_routed_shared_funded(&source,&context(source.admission()),
            &rows,8,limits(),&funding,&mut ledger).unwrap();
        assert!(used.load(Ordering::SeqCst)>0);
        assert_eq!(actual.identity(),ordinary.identity());
        assert_eq!(ledger.total(),ordinary_ledger.total());
        assert_eq!(actual.producers().map(|(rank,_)|rank).collect::<Vec<_>>(),[0,1,2,3,4,5,6]);
        for (rank,projection) in actual.producers() {
            assert_eq!(projection,ordinary.producer(rank).unwrap());
            assert_eq!(actual.routed_producer(rank),ordinary.routed_producer(rank));
            assert_eq!(actual.encoding_usage(rank).unwrap(),ordinary.encoding_usage(rank).unwrap());
        }
        assert!(actual.producer(6).unwrap().fragments().is_empty());
        assert_eq!(actual.delivery_usage().unwrap(),ordinary.delivery_usage().unwrap());
        // The existing nonzero sparse record fixture preserves route slots,
        // authoritative peers, expert IDs and selected global unit coordinates.
        assert_eq!(serde_json::to_vec(&records(source.admission(),&actual)).unwrap(),
            serde_json::to_vec(&records(source.admission(),&ordinary)).unwrap());
        drop(rows);drop(declarations);drop(ordinary);drop(source);drop(funding);
        assert!(!retired.load(Ordering::SeqCst));drop(actual);assert!(retired.load(Ordering::SeqCst));
    }}
}
#[test]
fn funded_routed_receipt_refusals_keep_exact_source_guards_and_spent_owner() {
    for case in ["duplicate","missing","peer","fragments","funding"] {
        let source=SharedCapturePlan::new(plan(false));
        let mut declarations=producers(source.admission(),false);
        let mut bound=limits();
        match case {
            "duplicate"=>declarations[1].rank=declarations[0].rank,
            "missing"=>{declarations.remove(2);},
            "peer"=>declarations[1].ownership.source_peer=Some(0),
            "fragments"=>bound.max_fragments=1,
            _=>(),
        }
        let rows:Vec<_>=declarations.iter().map(|row|PartitionCaptureRoutedProducerSource{rank:row.rank,ownership:&row.ownership}).collect();
        let mut ledger=CaptureLedger::new(source.admission());ledger.begin_step();
        let (funding,used,refuse,retired)=funding();refuse.store(case=="funding",Ordering::SeqCst);
        let error=PartitionCaptureReceiptPlan::new_routed_shared_funded(&source,&context(source.admission()),
            &rows,8,bound,&funding,&mut ledger).unwrap_err();
        if case!="funding" {assert!(used.load(Ordering::SeqCst)>0);}
        let spent=ledger.total();drop(rows);drop(declarations);drop(source);drop(funding);
        assert!(!retired.load(Ordering::SeqCst));drop(error);assert!(retired.load(Ordering::SeqCst));
        assert_eq!(ledger.total(),spent);
    }
}
