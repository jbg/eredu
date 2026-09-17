use super::*;
use super::hook_callback::{Native,Tensor};
use crate::capture::partition::PreparedPartitionContiguousSource;

#[test]
fn retained_contiguous_row_binds_real_epoch_and_lends_each_original_chunk_once() {
    use super::super::prefill::{projected_source,projected_receipt,inference};
    for (transform,combination,local,abandon) in [
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,0,false),
        (CaptureTransform::Summary,PartitionCaptureCombination::Disjoint,0,false),
        (CaptureTransform::Histogram{edges:vec![0.0,15.0,200.0]},PartitionCaptureCombination::Disjoint,0,false),
        (CaptureTransform::Summary,PartitionCaptureCombination::SumF64ToF32,0,false),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,1,false),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,2,false),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,0,true),
    ] {
        let sum=combination==PartitionCaptureCombination::SumF64ToF32;
        let source=projected_source(transform.clone());let (metadata,_,_,_)=funding();
        let transport=Ranks{local,bytes:RefCell::new(std::array::from_fn(|_|None)),funding:metadata.clone(),reject:false,calls:RefCell::new(vec![])};
        let mut quote=CaptureLedger::new(source.admission());quote.begin_step();
        let mut prototype=projected_receipt(&source,combination,&metadata,&mut quote);
        let geometry:Vec<_>=prototype.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let sources:Vec<_>=geometry.iter().map(|(producer,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*producer,fragment:*fragment,local_shape:shape,local_slice:slice,transform:if sum{&raw}else{&transform},
            dtype:TensorDtype::F16,estimate:estimate(*producer)}).collect();
        let producers=if sum {vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..7},PartitionCaptureContiguousProducer{rank:3,coordinates:0..7}]}
            else {vec![PartitionCaptureContiguousProducer{rank:3,coordinates:0..3},PartitionCaptureContiguousProducer{rank:0,coordinates:3..7},
                PartitionCaptureContiguousProducer{rank:1,coordinates:0..0}]};
        let dtypes=vec![Some(TensorDtype::F16);producers.len()];
        let retained=PreparedPartitionContiguousSource::new(&source,0,2,&producers,&dtypes,&sources,combination,inference(),&metadata).unwrap();
        // Finalize the prototype's real source-derived encoding bounds before
        // reading the exact host plan; its separate ledger grants no live work.
        let quoted=PreparedPartitionFragmentAllowance::prepare(&transport,&mut prototype,&sources,&metadata,&mut quote).unwrap();
        drop(quoted);
        let h=PartitionFragmentHostPlan::prepare_prefill(&prototype,inference()).unwrap().initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let epoch=DistributedCommitEpoch::new(19).unwrap();
        let mut actual=context(&source,0);actual.forward_epoch=epoch.value();
        let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10};
        let mut ledger_capture=CaptureLedger::new(source.admission());ledger_capture.begin_step();
        let prototype_identity=prototype.identity().to_owned();
        let use_prepared=local==0&&!abandon;
        let mut row=if use_prepared {
            let host=run.prepare_partition_fragment_host(&reservation,prototype,Some(inference())).unwrap();
            assert_eq!(host.protected_bytes(),h);
            retained.prepare_with_host(&transport,&actual,epoch,host,limits,&mut ledger_capture).unwrap()
        }else{
            retained.prepare(&transport,&actual,epoch,&run,&reservation,limits,&mut ledger_capture).unwrap()
        };
        assert_eq!(row.receipt().unwrap().context().forward_epoch,19);
        assert_ne!(row.receipt().unwrap().identity(),prototype_identity);
        assert_eq!(row.produces(),local==0);let spent=ledger_capture.total();let mut native=Native::default();
        if abandon {
            let failure=row.finish().unwrap_err();drop(run);drop(reservation);assert!(ledger(&pool).0>0);
            assert_eq!(ledger_capture.total(),spent);drop(failure);assert_eq!(ledger(&pool).0,0);continue;
        }
        for chunk in 0..3 {
            let epoch=DistributedCommitEpoch::new(19+chunk).unwrap();
            if local==0 {
                let hook=row.take_local(chunk,epoch).unwrap().unwrap();
                assert!(row.receipt().is_err());assert!(row.take_local(chunk,epoch).is_err());
                let width=if sum{7}else{4};let offset=if sum{0}else{3};
                let value=Tensor{shape:[2,1,width],values:(0..2).flat_map(|head|(0..width).map(move |column|
                    head as f32*100.0+chunk as f32*10.0+(column+offset)as f32)).collect()};
                let hook=hook.observe(&mut native,&value,inference(),chunk).unwrap();row.return_local(hook).unwrap();
                assert!(row.take_local(chunk,epoch).is_err());
            }else{
                assert!(row.take_local(chunk,epoch).unwrap().is_none());
                let receiver=row.take_receiver(chunk,epoch).unwrap();
                if local==1 {
                    assert_eq!(receiver.unwrap().source_shape(),[2,1,0]);
                    assert!(row.take_receiver(chunk,epoch).is_err());
                }else{assert!(receiver.is_none());}
            }
        }
        assert_eq!(ledger_capture.total(),spent);assert!(transport.calls.borrow().is_empty());
        if local==0{assert_eq!(native.validations.get(),3);assert_eq!(native.raw+native.summary+native.histogram,2);}
        let delivery=row.finish().unwrap();assert_eq!(delivery.receipt_plan().context().forward_epoch,19);
        drop(run);drop(reservation);assert!(ledger(&pool).0>0);
        drop(delivery);assert_eq!(ledger(&pool).0,0);
    }
}
