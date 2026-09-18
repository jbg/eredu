use super::*;
use crate::capture::partition::{PartitionCaptureRoutedFragmentGeometry,PartitionCaptureRoutedFragmentSource,
    PreparedPartitionFragmentAllowance,PreparedPartitionFragmentSourceAllowance};
use eredu_core::capture::{SharedCapturePlan,CaptureRoutedUnitsGeometry};
use eredu_nn::workspace::{HostMetadataAccount,HostMetadataFunding,HostMetadataFundingError};
#[derive(Debug)]
struct Account(Arc<AtomicUsize>,Arc<AtomicBool>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError> {self.0.fetch_add(bytes,Ordering::SeqCst);Ok(())}
}
impl Drop for Account {fn drop(&mut self){self.1.store(true,Ordering::SeqCst);}}
fn funding()->(HostMetadataFunding,Arc<AtomicUsize>,Arc<AtomicBool>) {
    let used=Arc::new(AtomicUsize::new(0));let retired=Arc::new(AtomicBool::new(false));
    (HostMetadataFunding::new(Account(used.clone(),retired.clone())).unwrap(),used,retired)
}
#[test]
fn routed_fragment_allowance_preserves_shared_costs_source_vote_and_move_only_native_credits() {
    for dtype in [TensorDtype::F32,TensorDtype::F64] {
        let source=SharedCapturePlan::new(fixture::plan(false));
        let mut ledger=CaptureLedger::new(source.admission());let mut second=CaptureLedger::new(source.admission());
        let make=|ledger:&mut CaptureLedger|PartitionCaptureReceiptPlan::new_routed(source.clone(),context(source.admission()),
            fixture::producers(source.admission(),true),8,fixture::limits(),ledger).unwrap();
        let mut receipt=make(&mut ledger);let mut voted_receipt=make(&mut second);
        let raw:Vec<_>=receipt.producers().flat_map(|(rank,projection)|projection.fragments().iter().enumerate().map(move |(fragment,_)| {
            let a=CaptureRoutedUnitsGeometry::fragment_axes(projection,fragment).unwrap();
            (rank,fragment,ResolvedCaptureSlice{starts:a[0].to_vec(),ends:a[1].to_vec(),strides:a[2].to_vec(),shape:a[3].to_vec()})
        })).collect();
        let ownership=fixture::producers(source.admission(),true);
        let estimate=PartitionCaptureNativeEstimate{capture:CaptureUsage{captures:1,retained_bytes:4096,host_bytes:8192,encoded_bytes:16384},generated_creation_bytes:0};
        let geometry:Vec<_>=raw.iter().map(|(rank,fragment,slice)|PartitionCaptureRoutedFragmentGeometry {
            producer:*rank,fragment:*fragment,request:PartitionRoutedUnitCaptureRequest {
                geometry:fixture::GEOMETRY,source_tokens:3,ownership:&ownership[*rank].ownership,slice},estimate,
        }).collect();
        let typed:Vec<_>=geometry.iter().map(|row|PartitionCaptureRoutedFragmentSource{geometry:*row,dtype:dtype.clone()}).collect();
        let transport=transport(world(8),2,Fault::None);
        let (funding,used,retired)=funding();
        let mut actual=PreparedPartitionFragmentAllowance::prepare_routed(&transport,&mut receipt,&typed,&funding,&mut ledger).unwrap();
        let pending=PreparedPartitionFragmentSourceAllowance::prepare_routed(&transport,&mut voted_receipt,&geometry,&funding,&mut second).unwrap();
        let voted=pending.bind(&voted_receipt,dtype.clone()).unwrap();
        assert_eq!(actual.global_reserved(),voted.global_reserved());assert_eq!(actual.local_reserved(),voted.local_reserved());
        assert_eq!(actual.descriptor(),voted.descriptor());assert_eq!(ledger.total(),second.total());
        assert_eq!(transport.calls.load(Ordering::SeqCst),0);
        assert!(used.load(Ordering::SeqCst)>0);
        let mut bad=geometry.clone();bad[0].request.source_tokens=4;
        let mut rejected_ledger=CaptureLedger::new(source.admission());
        let mut rejected_receipt=make(&mut rejected_ledger);let before=rejected_ledger.total();
        assert!(PreparedPartitionFragmentSourceAllowance::prepare_routed(&transport,&mut rejected_receipt,&bad,&funding,&mut rejected_ledger).is_err());
        assert_eq!(rejected_ledger.total(),before);
        let spent=ledger.total();
        let loan=actual.take_local_fragment(&receipt,0,&dtype,estimate).unwrap();
        assert!(actual.take_local_fragment(&receipt,0,&dtype,estimate).is_err());
        // A mismatch spends the attempt before refusal; it cannot be retried.
        let mut changed=estimate;changed.capture.host_bytes+=1;
        assert!(actual.take_local_fragment(&receipt,1,&dtype,changed).is_err());
        assert!(actual.take_local_fragment(&receipt,1,&dtype,estimate).is_err());
        assert_eq!(ledger.total(),spent);
        drop(actual);drop(voted);drop(funding);assert!(!retired.load(Ordering::SeqCst));
        drop(loan);assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn routed_source_vote_preserves_idle_presence_and_requires_real_common_result_precision() {
    for mode in 0..3 {
        let shared=world(8);
        let threads:Vec<_>=(0..8).map(|rank|{
            let shared=Arc::clone(&shared);
            std::thread::spawn(move||{
                let source=SharedCapturePlan::new(fixture::plan(false));
                let transport=transport(shared,rank,Fault::None);
                let mut ledger=CaptureLedger::new(source.admission());
                let context=context(source.admission());
                let receipt=PartitionCaptureReceiptPlan::new_routed(source.clone(),context.clone(),
                    fixture::producers(source.admission(),true),8,fixture::limits(),&mut ledger).unwrap();
                let (metadata,_,_)=funding();
                let coordination=crate::capture::partition::PreparedPartitionCaptureCoordination::prepare(
                    &transport,&source,&context,&metadata,&mut ledger).unwrap();
                let dtype=if rank==0||rank==7||mode==2 {None}
                    else if mode==1&&rank==2 {Some(TensorDtype::F16)}else{Some(TensorDtype::F32)};
                let result=coordination.coordinate_routed_source(0,&receipt,rank<7,dtype.as_ref(),CaptureUsage::default(),&ledger);
                if mode==0 {
                    let vote=result.unwrap();assert_eq!(vote.dtype,TensorDtype::F32);
                    assert_eq!(vote.ranks.len(),7);
                    assert_eq!(vote.ranks[0].producer,0);assert_eq!(vote.ranks[0].dtype,None);
                    assert!(vote.ranks.iter().skip(1).all(|row|row.dtype==Some(TensorDtype::F32)));
                }else{assert!(result.is_err());}
                assert_eq!(transport.calls.load(Ordering::SeqCst),1,"same single Source frame");
            })
        }).collect();
        for thread in threads {thread.join().unwrap();}
    }
}
