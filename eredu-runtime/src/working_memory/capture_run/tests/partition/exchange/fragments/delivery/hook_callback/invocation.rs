use super::*;

#[test]
fn local_partition_invocation_uses_typed_workers_and_keeps_spent_failure_custody(){
    use super::super::super::prefill::{projected_source,projected_receipt};
    for (transform,combination,mode) in [
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,0),
        (CaptureTransform::Preview{max_elements:5},PartitionCaptureCombination::Disjoint,0),
        (CaptureTransform::Summary,PartitionCaptureCombination::Disjoint,0),
        (CaptureTransform::Histogram{edges:vec![0.0,15.0,200.0]},PartitionCaptureCombination::Disjoint,0),
        (CaptureTransform::Summary,PartitionCaptureCombination::SumF64ToF32,0),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,1),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,2),
        (CaptureTransform::Slice,PartitionCaptureCombination::Disjoint,3),
    ] {
        let sum=combination==PartitionCaptureCombination::SumF64ToF32;
        let source=projected_source(transform.clone());let (funding,_,_,_)=funding();
        let transport=Ranks{local:0,bytes:RefCell::new(std::array::from_fn(|_|None)),funding:funding.clone(),reject:false,calls:RefCell::new(vec![])};
        let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let mut receipt=projected_receipt(&source,combination,&funding,&mut quota);
        let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(i,g)|(rank,i,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let sources:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:if sum{&raw}else{&transform},
            dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&sources,&funding,&mut quota).unwrap();
        let ranks:Vec<_>=receipt.producers().map(|(producer,_)|PartitionCaptureRankSource{producer,dtype:Some(TensorDtype::F16)}).collect();
        let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();let h=host.initialization_peak_bytes();
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
        let delivery=PreparedPartitionFragmentDelivery::prepare(&transport,receipt,bank,&ranks,&funding).unwrap();
        let spent=quota.total();let (continuation,hook)=delivery.into_local_hook();
        let hook=crate::capture::partition::PartitionCaptureLocalHook::new(hook);
        let width=if sum{7}else{4};let offset=if sum{0}else{3};
        let value=Tensor{shape:[2,3,width],values:(0..2).flat_map(|head|(0..3).flat_map(move |row|
            (0..width).map(move |column|head as f32*100.0+row as f32*10.0+(column+offset) as f32+0.125))).collect()};
        let mut native=Native{fail:mode==1,wrong_usage:mode==2,..Default::default()};
        let mut result=hook.observe_invocation(&mut native,&value);
        if mode==3 {result=result.unwrap().observe_invocation(&mut native,&value);}
        assert_eq!(quota.total(),spent);assert!(transport.calls.borrow().is_empty());
        if mode!=0 {
            let error=result.unwrap_err();assert_eq!(native.raw,usize::from(mode!=2));
            if mode==1{assert!(error.to_string().contains("native sentinel"));}
            drop(continuation);drop(run);drop(reservation);assert!(ledger(&pool).0>0);
            drop(error);assert_eq!(ledger(&pool).0,0);continue;
        }
        assert_eq!(native.validations.get(),1);
        let mut hook=result.unwrap().into_inner();let (receipt,bank)=hook.parts();
        let mut values:Vec<_>=(0..2).flat_map(|head|(1..3).flat_map(move |row|[1,3,5].into_iter().filter(move |c|sum||*c>=3)
            .map(move |column|head as f32*100.0+row as f32*10.0+column as f32+0.125))).collect();
        if let CaptureTransform::Preview{max_elements}=&transform{values.truncate(*max_elements as usize);}
        let mut escaped=None;
        match bank.value(0,0).unwrap() {
            PartitionFragmentValue::Routed(_)=>panic!("dense fixture returned sparse assembly"),
            PartitionFragmentValue::Tensor(value)=>{assert_eq!((native.raw,native.summary,native.histogram),(1,0,0));
                assert_eq!(value.observation().data(),&TensorObservationData::F32(values));escaped=Some(value.observation().clone());},
            PartitionFragmentValue::Summary(value)=>{assert_eq!((native.raw,native.summary,native.histogram),(0,1,0));
                let expected=crate::capture::partition::summarize_f32(&values);assert_eq!(value.observation(),&expected);},
            PartitionFragmentValue::Histogram(value)=>{assert_eq!((native.raw,native.summary,native.histogram),(0,0,1));
                let mut expected=value.observation().clone();expected.counts.fill(0);expected.below=0;expected.above=0;expected.non_finite=0;
                crate::capture::partition::fill_histogram_f32(&values,&mut expected).unwrap();assert_eq!(value.observation(),&expected);},
        }
        assert!(!bank.complete(),"remote sources still need delivery");
        assert!(bank.take_local(receipt,0,&TensorDtype::F16,estimate(0)).is_err());
        let delivery=continuation.resume(hook).unwrap();drop(delivery);drop(run);drop(reservation);
        assert_eq!(ledger(&pool).0>0,escaped.is_some());drop(escaped);assert_eq!(ledger(&pool).0,0);
    }
}
