use super::*;

fn receipt(source: &SharedCapturePlan, combination: PartitionCaptureCombination,
    funding: &WorkspaceMetadataFunding, ledger: &mut CaptureLedger) -> PartitionCaptureReceiptPlan
{
    let global = CaptureTensorGeometry::prepare(source.admission(), 0, CapturePhase::Prefill, 0, None).unwrap();
    let width = *global.source_shape().last().unwrap() as u64;
    let rows = if combination == PartitionCaptureCombination::SumF64ToF32 {
        vec![PartitionCaptureContiguousProducer { rank: 3, coordinates: 0..width },
            PartitionCaptureContiguousProducer { rank: 0, coordinates: 0..width }]
    } else {
        vec![PartitionCaptureContiguousProducer { rank: 3, coordinates: 1..width },
            PartitionCaptureContiguousProducer { rank: 0, coordinates: 0..1 },
            PartitionCaptureContiguousProducer { rank: 1, coordinates: 0..0 }]
    };
    PartitionCaptureReceiptPlan::new_contiguous_shared_funded(source, &context(source, 0),
        global.source_shape().len()-1, &rows, combination, 4,
        PartitionCaptureReceiptLimits { max_producers: 3, max_fragments: 3, max_record_bytes: 64 << 10 },
        funding, ledger).unwrap()
}
fn estimate(rank: usize) -> PartitionCaptureNativeEstimate {
    PartitionCaptureNativeEstimate { capture: CaptureUsage { retained_bytes: 100 + rank as u64,
        host_bytes: 200 + rank as u64, encoded_bytes: 8192, captures: 1 }, generated_creation_bytes: 0 }
}

#[test]
fn original_fragment_allowance_prices_real_sources_and_retains_one_use_local_custody() {
    for combination in [PartitionCaptureCombination::Disjoint, PartitionCaptureCombination::SumF64ToF32] {
        let source = source(); let mut ledger = CaptureLedger::new(source.admission()); ledger.begin_step();
        let (funding, used, _, retired) = funding();
        let transport = Receipts { source: Vec::new(), funding: funding.clone(), reject_delivery: false, calls: RefCell::new(Vec::new()) };
        let mut receipt = receipt(&source, combination, &funding, &mut ledger);
        // Actual native source facts have their own cold geometry, not a loan
        // into the mutable receipt whose encoding bound is finalized below.
        let geometry: Vec<_> = receipt.producers().flat_map(|(rank, projection)| {
            projection.fragments().iter().enumerate().map(move |(fragment, value)|
                (rank, fragment, projection.local_shape().to_vec(), value.local().clone()))
        }).collect();
        let raw = CaptureTransform::Slice;
        let transform = if combination == PartitionCaptureCombination::SumF64ToF32 { &raw }
            else { &source.admission().plan().selections[0].transform };
        let rows: Vec<_> = geometry.iter().map(|(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
            producer: *rank, fragment: *fragment, local_shape: shape, local_slice: slice, transform,
            dtype: TensorDtype::F16, estimate: estimate(*rank),
        }).collect();
        assert_eq!(rows.len(), 2, "empty shard acknowledges without inventing a native primitive");
        let prior = ledger.total();
        let mut allowance = PreparedPartitionFragmentAllowance::prepare(&transport, &mut receipt, &rows, &funding, &mut ledger).unwrap();
        assert!(allowance.matches(&receipt));
        assert_eq!(ledger.total(), prior.checked_add(allowance.global_reserved()).unwrap());
        assert!(allowance.global_reserved().retained_bytes >= estimate(0).capture.retained_bytes + estimate(3).capture.retained_bytes);
        let total = ledger.total();
        let mut loan = allowance.take_local_fragment(&receipt, 0, &TensorDtype::F16, estimate(0)).unwrap();
        assert!(loan.source().same_storage(&source));
        loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
        assert!(allowance.take_local_fragment(&receipt, 0, &TensorDtype::F16, estimate(0)).is_err());
        assert!(allowance.take_local_fragment(&receipt, 1, &TensorDtype::F16, estimate(0)).is_err());
        assert_eq!(ledger.total(), total, "child consumption and refusals never reset global credits");
        assert!(used.load(Ordering::SeqCst) > 0); assert!(transport.calls.borrow().is_empty());
        drop(rows); drop(geometry); drop(allowance); drop(receipt); drop(source); drop(transport); drop(funding);
        assert!(!retired.load(Ordering::SeqCst)); drop(loan); assert!(retired.load(Ordering::SeqCst));
        assert_eq!(ledger.total(), total);
    }
}

#[test]
fn original_fragment_source_rejects_wrong_coordinates_nonlinear_terms_and_replayed_native_attempts() {
    for mismatch in ["shape", "transform", "missing", "extra", "rank", "dtype", "native", "funding"] {
        let source = source(); let mut ledger = CaptureLedger::new(source.admission()); ledger.begin_step();
        let (funding, _, refuse, retired) = funding();
        let transport = Receipts { source: Vec::new(), funding: funding.clone(), reject_delivery: false, calls: RefCell::new(Vec::new()) };
        let mut receipt = receipt(&source, PartitionCaptureCombination::SumF64ToF32, &funding, &mut ledger);
        let mut geometry: Vec<_> = receipt.producers().flat_map(|(rank, projection)| {
            projection.fragments().iter().enumerate().map(move |(fragment, value)|
                (rank, fragment, projection.local_shape().to_vec(), value.local().clone()))
        }).collect();
        if mismatch == "shape" { geometry[0].2[0] += 1; }
        let raw = CaptureTransform::Slice; let summary = CaptureTransform::Summary;
        let mut rows: Vec<_> = geometry.iter().map(|(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
            producer: *rank, fragment: *fragment, local_shape: shape, local_slice: slice,
            transform: if mismatch == "transform" { &summary } else { &raw },
            dtype: TensorDtype::F32, estimate: estimate(*rank),
        }).collect();
        match mismatch {
            "missing" => { rows.pop(); },
            "extra" => rows.push(PartitionCaptureFragmentSource { producer: 1, fragment: 0,
                local_shape: &geometry[0].2, local_slice: &geometry[0].3, transform: &raw,
                dtype: TensorDtype::F32, estimate: estimate(1) }),
            "rank" => rows[0].producer = 2,
            "dtype" => rows[0].dtype = TensorDtype::I32,
            "funding" => refuse.store(true, Ordering::SeqCst),
            _ => (),
        }
        let before = ledger.total();
        let result = PreparedPartitionFragmentAllowance::prepare(&transport, &mut receipt, &rows, &funding, &mut ledger);
        let error = if mismatch == "native" {
            let mut allowance = result.unwrap();
            let error = allowance.take_local_fragment(&receipt, 0, &TensorDtype::Bf16, estimate(0)).unwrap_err();
            assert!(allowance.take_local_fragment(&receipt, 0, &TensorDtype::F32, estimate(0)).is_err());
            error
        } else { assert_eq!(ledger.total(), before); result.unwrap_err() };
        drop(rows); drop(geometry); drop(receipt); drop(source); drop(transport); drop(funding);
        assert!(!retired.load(Ordering::SeqCst)); drop(error); assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn fragment_geometries_keep_original_admission_and_raw_additive_nonlinear_inputs() {
    for transform in [CaptureTransform::FullTensor, CaptureTransform::Preview { max_elements: 2 },
        CaptureTransform::Summary, CaptureTransform::Histogram { edges: vec![-1.0, 0.25, 2.0] }]
    {
        let mut raw = raw(); raw.selections[0].transform = transform.clone();
        let source = if matches!(transform, CaptureTransform::Histogram { .. }) { super::super::histogram::histogram_source() }
            else { admit(raw, point(), 4, false) };
        let global = vec![5, 2];
        let slice = resolve_slice(&source.admission().points()[0], &source.admission().plan().selections[0], &global).unwrap();
        let shard = CaptureContiguousProjectionPlan::prepare(&global, &slice, 1, 1..2, 1).unwrap().construct();
        let term = CaptureContiguousProjectionPlan::prepare(&global, &slice, 1, 0..2, 1).unwrap().construct();
        let geometry = CaptureTensorGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0,
            None, &term, 0, PartitionCaptureCombination::SumF64ToF32).unwrap();
        assert!(std::ptr::eq(geometry.admission(), source.admission()));
        assert_eq!(geometry.admission().plan().selections[0].transform, transform);
        assert_eq!(geometry.source_shape(), [5, 2]);
        if matches!(transform, CaptureTransform::Preview { .. }) {
            assert_eq!(geometry.native_transform(), &transform); assert_eq!(geometry.shape(), [2]);
        } else { assert_eq!(geometry.native_transform(), &CaptureTransform::Slice); assert_eq!(geometry.shape(), [5, 2]); }
        assert!(CaptureTensorGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0,
            None, &shard, 0, PartitionCaptureCombination::SumF64ToF32).is_err());
        assert!(CaptureTensorGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0,
            None, &term, 1, PartitionCaptureCombination::SumF64ToF32).is_err());
        match transform {
            CaptureTransform::Summary => {
                let part = CaptureSummaryGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0, None, &shard, 0).unwrap();
                assert_eq!(part.source_shape(), [5, 1]); assert_eq!(part.shape(), [5, 1]);
            }
            CaptureTransform::Histogram { .. } => {
                let part = CaptureHistogramGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0, None, &shard, 0).unwrap();
                assert_eq!(part.source_shape(), [5, 1]); assert_eq!(part.shape(), [5, 1]); assert_eq!(part.edges(), [-1.0, 0.25, 2.0]);
            }
            _ => {
                let part = CaptureTensorGeometry::prepare_partition(source.admission(), 0, CapturePhase::Prefill, 0,
                    None, &shard, 0, PartitionCaptureCombination::Disjoint).unwrap();
                assert_eq!(part.source_shape(), [5, 1]); assert_eq!(part.starts(), [0, 0]); assert_eq!(part.ends(), [5, 1]);
            }
        }
    }
}

#[test]
fn original_fragment_host_bank_protects_each_native_and_received_destination() {
    for combination in [PartitionCaptureCombination::Disjoint,PartitionCaptureCombination::SumF64ToF32] {
        let source=source();let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,_,_,_)=funding();
        let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
        let mut receipt=receipt(&source,combination,&funding,&mut quota);
        let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let transform=if combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{&source.admission().plan().selections[0].transform};
        let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
        let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();assert_eq!(host.fragment_count(),2);
        assert!(host.initialization_peak_bytes()>host.fragment_peak_bytes());
        let h=host.initialization_peak_bytes();let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
        let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
        assert_eq!(ledger(&pool).2,h);let before=quota.total();
        let native=bank.take_local(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap();
        let (destination,mut loan)=native.into_parts();
        loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
        let PartitionFragmentDestination::Tensor(claim)=destination else{panic!("raw selected source")};
        assert_eq!(claim.geometry().source_shape(),if combination==PartitionCaptureCombination::Disjoint{&[5,1][..]}else{&[5,2][..]});
        let mut output=claim.prepare().unwrap();for i in 0..output.len(){output.push_f32(i as f32-2.25).unwrap();}
        let output=output.finish().unwrap();let escaped=output.observation().clone();
        bank.record(&receipt,0,0,&TensorDtype::F16,estimate(0).capture,PartitionFragmentValue::Tensor(output)).unwrap();
        assert!(bank.take_local(&receipt,0,&TensorDtype::F16,estimate(0)).is_err());
        assert!(bank.take_received(&receipt,1,0).is_err(),"empty producer has no payload constructor");
        let PartitionFragmentDestination::Tensor(claim)=bank.take_received(&receipt,3,0).unwrap() else{panic!("raw received source")};
        let mut output=claim.prepare().unwrap();for i in 0..output.len(){output.push_f32(11.5-i as f32).unwrap();}
        let output=output.finish().unwrap();
        bank.record(&receipt,3,0,&TensorDtype::F16,estimate(3).capture,PartitionFragmentValue::Tensor(output)).unwrap();
        assert!(bank.complete());assert!(bank.value(3,0).is_some());assert_eq!(quota.total(),before);
        assert!(transport.calls.borrow().is_empty(),"host claims imply no transport or native work");
        drop((bank,loan));drop(run);drop(reservation);assert!(ledger(&pool).0>0);
        assert_eq!(escaped.shape(),if combination==PartitionCaptureCombination::Disjoint{&[5,1][..]}else{&[5,2][..]});
        drop(escaped);assert_eq!(ledger(&pool).0,0);
    }
}

#[test]
fn original_fragment_host_bank_rejects_underfunding_and_same_shape_rank_swaps() {
    for short in [true,false] {
        let source=source();let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
        let (funding,_,_,_)=funding();let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
        let mut receipt=receipt(&source,PartitionCaptureCombination::Disjoint,&funding,&mut quota);
        let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&source.admission().plan().selections[0].transform,
            dtype:TensorDtype::F32,estimate:estimate(*rank)}).collect();
        let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
        let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();let h=host.initialization_peak_bytes()-u64::from(short);
        let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);let spent=quota.total();
        let result=run.prepare_partition_fragments(&reservation,host,allowance);
        let error=if short {result.unwrap_err()}else{
            let mut bank=result.unwrap();
            let (destination,loan)=bank.take_local(&receipt,0,&TensorDtype::F32,estimate(0)).unwrap().into_parts();
            let PartitionFragmentDestination::Tensor(claim)=destination else{panic!("tensor")};
            let mut output=claim.prepare().unwrap();for i in 0..output.len(){output.push_f32(i as f32+0.125).unwrap();}
            let value=output.finish().unwrap();
            // Both shards are [5,1], but each received claim has a distinct
            // original host identity. A local completion cannot satisfy rank 3.
            drop(bank.take_received(&receipt,3,0).unwrap());
            let error=bank.record(&receipt,3,0,&TensorDtype::F32,estimate(3).capture,PartitionFragmentValue::Tensor(value)).unwrap_err();
            assert!(!bank.complete());assert!(bank.take_received(&receipt,3,0).is_err());
            assert!(bank.take_local(&receipt,0,&TensorDtype::F32,estimate(0)).is_err());
            drop((bank,loan));error
        };
        assert_eq!(quota.total(),spent);drop(run);drop(reservation);assert!(ledger(&pool).0>0);
        drop(error);assert_eq!(ledger(&pool).0,0);assert!(transport.calls.borrow().is_empty());
    }
}

#[test]
fn contiguous_receivers_decode_exact_local_shapes_raw_terms_and_empty_envelopes() {
    for combination in [PartitionCaptureCombination::Disjoint,PartitionCaptureCombination::SumF64ToF32] {
        for transform in [CaptureTransform::FullTensor,CaptureTransform::Preview{max_elements:2},CaptureTransform::Summary,
            CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]}] {
            let mut raw=raw();raw.selections.truncate(1);raw.selections[0].transform=transform.clone();
            let source=if matches!(transform,CaptureTransform::Histogram{..}) {super::super::histogram::histogram_source()} else {admit(raw,point(),4,false)};
            let mut quota=CaptureLedger::new(source.admission());quota.begin_step();let (funding,_,_,_)=funding();
            let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
            let producers=if combination==PartitionCaptureCombination::Disjoint {
                vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..1},PartitionCaptureContiguousProducer{rank:1,coordinates:0..0},PartitionCaptureContiguousProducer{rank:3,coordinates:1..2}]
            }else{vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..2},PartitionCaptureContiguousProducer{rank:3,coordinates:0..2}]};
            let mut receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context(&source,0),1,&producers,combination,4,
                PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10},&funding,&mut quota).unwrap();
            let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
                .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
            let native=if combination==PartitionCaptureCombination::SumF64ToF32 && !matches!(transform,CaptureTransform::Preview{..}) {&CaptureTransform::Slice}else{&transform};
            let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
                producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:native,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
            let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
            let (_,charged,_)=allowance.fragment_record_charge(&receipt,3,0).unwrap();
            let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();let h=host.initialization_peak_bytes();
            let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
            let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
            let projection=receipt.producer(3).unwrap();let selected=projection.fragments()[0].local().shape.clone();let elements=selected.iter().product::<u64>();
            let values:Vec<f32>=(0..elements).map(|i|[-0.75,0.0,1.25,-0.5,1.0][i as usize%5]).collect();
            let raw_payload=combination==PartitionCaptureCombination::SumF64ToF32 || matches!(transform,CaptureTransform::FullTensor|CaptureTransform::Preview{..});
            let payload=if raw_payload {
                let kept=if let CaptureTransform::Preview{max_elements}=transform{elements.min(max_elements)}else{elements};
                let shape=if matches!(transform,CaptureTransform::Preview{..}){vec![kept as usize]}else{selected.iter().map(|n|*n as usize).collect()};
                CapturePayload::Tensor(TensorObservation::new(shape,TensorObservationData::F32(values[..kept as usize].to_vec())).unwrap())
            }else if matches!(transform,CaptureTransform::Summary) {
                CapturePayload::Summary(CaptureSummary{elements,finite:elements,non_finite:0,nan:0,positive_infinity:0,negative_infinity:0,
                    min:Some(-0.75),max:Some(1.25),mean:Some(0.2),rms:Some(0.675f64.sqrt())})
            }else{CapturePayload::Histogram(CaptureHistogram{edges:vec![-1.0,0.25,2.0],counts:vec![3,2],below:0,above:0,non_finite:0})};
            let selected_source=&source.admission().plan().selections[0];let point=&source.admission().points()[0];
            let outcome=if let CaptureTransform::Preview{max_elements}=transform {CaptureOutcome::Truncated{available_elements:elements,emitted_elements:elements.min(max_elements)}}else{CaptureOutcome::Captured};
            let record=CaptureRecord{schema_version:CAPTURE_SCHEMA_VERSION,selection_id:selected_source.id.clone(),path:selected_source.path.clone(),node_id:point.node_id.clone(),position:point.position,
                source_shape:Some(projection.local_shape().to_vec()),source_dtype:Some(TensorDtype::F16),selected_shape:Some(selected),outcome,payload:Some(payload.clone()),charged};
            let envelope=PartitionCaptureProducerRecord{schema_version:PARTITION_CAPTURE_SCHEMA_VERSION,combination,receipt_plan_identity:receipt.identity().into(),context:receipt.context().clone(),producer_rank:3,
                source_dtype:Some(TensorDtype::F16),fragments:vec![PartitionCaptureFragmentRecord{fragment_index:0,record}]};
            let wire=serde_json::to_vec(&envelope).unwrap();bank.decode_contiguous_producer(&receipt,3,&wire,None,&funding).unwrap();
            match (bank.value(3,0).unwrap(),&payload) {
                (PartitionFragmentValue::Tensor(actual),CapturePayload::Tensor(expected))=>assert_eq!(actual.observation().as_ref(),expected),
                (PartitionFragmentValue::Summary(actual),CapturePayload::Summary(expected))=>assert_eq!(actual.observation(),expected),
                (PartitionFragmentValue::Histogram(actual),CapturePayload::Histogram(expected))=>assert_eq!(actual.observation(),expected),
                _=>panic!("actual source selected wrong destination"),
            }
            assert!(bank.decode_contiguous_producer(&receipt,3,&wire,None,&funding).is_err(),"no destination replay");
            if combination==PartitionCaptureCombination::Disjoint {
                let mut empty=envelope.clone();empty.producer_rank=1;empty.fragments.clear();
                let wire=serde_json::to_vec(&empty).unwrap();bank.decode_contiguous_producer(&receipt,1,&wire,Some(&TensorDtype::F16),&funding).unwrap();
                assert!(bank.decode_contiguous_producer(&receipt,1,&wire,None,&funding).is_err(),"empty dtype comes from actual retained source");
                empty.fragments=envelope.fragments.clone();let wire=serde_json::to_vec(&empty).unwrap();
                assert!(bank.decode_contiguous_producer(&receipt,1,&wire,Some(&TensorDtype::F16),&funding).is_err(),"empty producer cannot smuggle a payload");
            }
            drop(bank);drop(run);drop(reservation);assert_eq!(ledger(&pool).0,0);
        }
    }
}

#[test]
fn contiguous_producer_encoding_borrows_paid_payloads_and_preserves_canonical_raw_terms() {
    for combination in [PartitionCaptureCombination::Disjoint,PartitionCaptureCombination::SumF64ToF32] {
        for transform in [CaptureTransform::FullTensor,CaptureTransform::Preview{max_elements:2},CaptureTransform::Summary,
            CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]}] {
            let mut raw=raw();raw.selections.truncate(1);raw.selections[0].transform=transform.clone();
            let source=if matches!(transform,CaptureTransform::Histogram{..}) {super::super::histogram::histogram_source()} else {admit(raw,point(),4,false)};
            let mut quota=CaptureLedger::new(source.admission());quota.begin_step();let (funding,_,refuse,retired)=funding();
            let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
            let producers=if combination==PartitionCaptureCombination::Disjoint {
                vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..1},PartitionCaptureContiguousProducer{rank:1,coordinates:0..0},PartitionCaptureContiguousProducer{rank:3,coordinates:1..2}]
            }else{vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..2},PartitionCaptureContiguousProducer{rank:3,coordinates:0..2}]};
            let mut receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context(&source,0),1,&producers,combination,4,
                PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10},&funding,&mut quota).unwrap();
            let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
                .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
            let native=if combination==PartitionCaptureCombination::SumF64ToF32 && !matches!(transform,CaptureTransform::Preview{..}) {&CaptureTransform::Slice}else{&transform};
            let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
                producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:native,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
            let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
            let (_,charged,_)=allowance.fragment_record_charge(&receipt,0,0).unwrap();
            let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();let h=host.initialization_peak_bytes();
            let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
            let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
            assert!(bank.encode_contiguous_producer(&receipt,None,&funding).is_err(),"uncompleted local slot is not encodable");
            let projection=receipt.producer(0).unwrap();let selected=projection.fragments()[0].local().shape.clone();let elements=selected.iter().product::<u64>();
            let raw_payload=combination==PartitionCaptureCombination::SumF64ToF32 || matches!(transform,CaptureTransform::FullTensor|CaptureTransform::Preview{..});
            let payload=if raw_payload {
                let kept=if let CaptureTransform::Preview{max_elements}=transform{elements.min(max_elements)}else{elements};
                let shape=if matches!(transform,CaptureTransform::Preview{..}){vec![kept as usize]}else{selected.iter().map(|n|*n as usize).collect()};
                CapturePayload::Tensor(TensorObservation::new(shape,TensorObservationData::F32((0..kept).map(|i|
                    [f32::NAN,-0.0,f32::INFINITY,f32::NEG_INFINITY,1.25][i as usize%5]).collect())).unwrap())
            }else if matches!(transform,CaptureTransform::Summary) {
                CapturePayload::Summary(CaptureSummary{elements,finite:elements,non_finite:0,nan:0,positive_infinity:0,negative_infinity:0,
                    min:Some(-0.75),max:Some(1.25),mean:Some(0.2),rms:Some(0.675f64.sqrt())})
            }else{CapturePayload::Histogram(CaptureHistogram{edges:vec![-1.0,0.25,2.0],counts:vec![3,2],below:0,above:0,non_finite:0})};
            let selection=&source.admission().plan().selections[0];let point=&source.admission().points()[0];
            let record=CaptureRecord{schema_version:CAPTURE_SCHEMA_VERSION,selection_id:selection.id.clone(),path:selection.path.clone(),node_id:point.node_id.clone(),position:point.position,
                source_shape:Some(projection.local_shape().to_vec()),source_dtype:Some(TensorDtype::F16),selected_shape:Some(selected),
                outcome:crate::capture::completed_capture_outcome(&transform,elements),payload:Some(payload),charged};
            let envelope=PartitionCaptureProducerRecord{schema_version:PARTITION_CAPTURE_SCHEMA_VERSION,combination,receipt_plan_identity:receipt.identity().into(),context:receipt.context().clone(),producer_rank:0,
                source_dtype:Some(TensorDtype::F16),fragments:vec![PartitionCaptureFragmentRecord{fragment_index:0,record}]};
            let expected=serde_json::to_vec(&envelope).unwrap();
            let (destination,mut loan)=bank.take_local(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap().into_parts();
            loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
            // The existing bounded decoder acts as the completed mock producer;
            // the real local path supplies the identical typed native receipt.
            let identity=PartitionCaptureTensorReceipt{context:receipt.context(),identity:receipt.identity(),producer:0,dtype:TensorDtype::F16,charged};
            let value=match destination {
                PartitionFragmentDestination::Routed(_)=>panic!("dense fixture returned sparse assembly"),
                PartitionFragmentDestination::Tensor(v)=>PartitionFragmentValue::Tensor(v.decode_partition_receipt_combined(&expected,identity,&funding,combination).unwrap()),
                PartitionFragmentDestination::Summary(v)=>PartitionFragmentValue::Summary(v.decode_partition_receipt(&expected,identity,&funding).unwrap()),
                PartitionFragmentDestination::Histogram(v)=>PartitionFragmentValue::Histogram(v.decode_partition_receipt(&expected,identity,&funding).unwrap()),
            };
            bank.record(&receipt,0,0,&TensorDtype::F16,estimate(0).capture,value).unwrap();
            let escaped=match bank.value(0,0).unwrap(){PartitionFragmentValue::Tensor(v)=>Some(v.observation().clone()),_=>None};
            let pointer=match bank.value(0,0).unwrap(){
                PartitionFragmentValue::Routed(_)=>panic!("dense fixture returned sparse assembly"),
                PartitionFragmentValue::Tensor(v)=>match v.observation().data(){TensorObservationData::F32(v)=>v.as_ptr() as usize,_=>unreachable!()},
                PartitionFragmentValue::Summary(v)=>v.observation() as *const CaptureSummary as usize,
                PartitionFragmentValue::Histogram(v)=>v.observation().counts.as_ptr() as usize,
            };
            let spent=quota.total();let output=bank.encode_contiguous_producer(&receipt,None,&funding).unwrap();
            assert_eq!(output.as_ref(),expected);assert_eq!(quota.total(),spent);
            let after=match bank.value(0,0).unwrap(){
                PartitionFragmentValue::Routed(_)=>panic!("dense fixture returned sparse assembly"),
                PartitionFragmentValue::Tensor(v)=>match v.observation().data(){TensorObservationData::F32(v)=>v.as_ptr() as usize,_=>unreachable!()},
                PartitionFragmentValue::Summary(v)=>v.observation() as *const CaptureSummary as usize,
                PartitionFragmentValue::Histogram(v)=>v.observation().counts.as_ptr() as usize,
            };assert_eq!(pointer,after,"wire borrows the existing payload");
            refuse.store(true,Ordering::SeqCst);let error=bank.encode_contiguous_producer(&receipt,None,&funding).unwrap_err();
            assert_eq!(quota.total(),spent);assert!(bank.value(0,0).is_some());
            drop((bank,loan));drop(run);drop(reservation);drop(error);
            assert_eq!(ledger(&pool).0==0,escaped.is_none());drop(escaped);assert_eq!(ledger(&pool).0,0);
            drop(rows);drop(geometry);drop(receipt);drop(source);drop(transport);drop(funding);
            assert!(!retired.load(Ordering::SeqCst));assert_eq!(output.as_ref(),expected);
            drop(output);assert!(retired.load(Ordering::SeqCst));
        }
    }
}

#[test]
fn paid_fragment_assembly_matches_ordinary_rank_order_and_post_sum_nonlinear_results() {
    for combination in [PartitionCaptureCombination::Disjoint,PartitionCaptureCombination::SumF64ToF32] {
        for transform in [CaptureTransform::FullTensor,CaptureTransform::Preview{max_elements:2},CaptureTransform::Summary,
            CaptureTransform::Histogram{edges:vec![-1.0,0.25,2.0]}] {
            let mut raw=raw();raw.selections.truncate(1);raw.selections[0].transform=transform.clone();
            let source=if matches!(transform,CaptureTransform::Histogram{..}) {super::super::histogram::histogram_source()} else {admit(raw,point(),4,false)};
            let mut quota=CaptureLedger::new(source.admission());quota.begin_step();let (funding,_,_,_)=funding();
            let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
            let producers=if combination==PartitionCaptureCombination::Disjoint {
                vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..1},PartitionCaptureContiguousProducer{rank:1,coordinates:0..0},PartitionCaptureContiguousProducer{rank:3,coordinates:1..2}]
            }else{vec![PartitionCaptureContiguousProducer{rank:0,coordinates:0..2},PartitionCaptureContiguousProducer{rank:3,coordinates:0..2}]};
            let mut receipt=PartitionCaptureReceiptPlan::new_contiguous_shared_funded(&source,&context(&source,0),1,&producers,combination,4,
                PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:64<<10},&funding,&mut quota).unwrap();
            let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
                .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
            let native=if combination==PartitionCaptureCombination::SumF64ToF32 && !matches!(transform,CaptureTransform::Preview{..}) {&CaptureTransform::Slice}else{&transform};
            let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
                producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:native,dtype:TensorDtype::F16,estimate:estimate(*rank)}).collect();
            let mut allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
            let evidence=crate::capture::partition::PreparedPartitionCaptureEvidence::prepare_contiguous(&receipt,&mut allowance,&funding).unwrap();
            let charges:Vec<_>=receipt.producers().filter_map(|(rank,p)|(!p.fragments().is_empty()).then(||
                (rank,allowance.fragment_record_charge(&receipt,rank,0).unwrap().1))).collect();
            let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();
            assert_eq!(host.assembly_peak_bytes()!=0,combination==PartitionCaptureCombination::SumF64ToF32&&matches!(transform,CaptureTransform::Summary|CaptureTransform::Histogram{..}));
            let h=host.initialization_peak_bytes()+plan(&source).initialization_peak_bytes();
            let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
            let mut final_bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
            let mut frame=final_bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
            frame.prepare_partition_evidence(&funding).unwrap();
            let charge=allowance.take_assembly_charge(&receipt).unwrap();
            assert!(charge.validate_at(&source,0,CapturePhase::Prefill,0));
            assert!(!charge.validate_at(&source,0,CapturePhase::Decode,1));
            let (destination,target)=frame.take_assembled_invocation_destination(0,TensorDtype::F16,charge).unwrap();
            assert!(allowance.take_assembly_charge(&receipt).is_err());
            let mut bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
            let ordinary_rows=receipt.producers().map(|(rank,p)|PartitionCaptureProducer{rank,projection:p.clone()}).collect();
            let limits=PartitionCaptureReceiptLimits{max_producers:3,max_fragments:3,max_record_bytes:receipt.max_record_bytes()};
            let mut ordinary_quota=CaptureLedger::new(source.admission());ordinary_quota.begin_step();
            let ordinary=if combination==PartitionCaptureCombination::SumF64ToF32 {
                PartitionCaptureReceiptPlan::new_sum_shared(source.clone(),receipt.context().clone(),ordinary_rows,4,limits,&mut ordinary_quota)
            }else{PartitionCaptureReceiptPlan::new_shared(source.clone(),receipt.context().clone(),ordinary_rows,4,limits,&mut ordinary_quota)}.unwrap();
            assert_eq!(ordinary.identity(),receipt.identity());let mut delivery=ordinary.into_delivery();
            for (producer,projection) in receipt.producers() {
                let mut fragments=Vec::new();
                if !projection.fragments().is_empty() {
                    let selected=projection.fragments()[0].local().shape.clone();let elements=selected.iter().product::<u64>();
                    let values:Vec<f32>=(0..elements).map(|i|if producer==0 {[-0.75,0.0,1.25,-0.5,1.0][i as usize%5]}
                        else{[0.5,1.0,-0.75,0.25,1.5][i as usize%5]}).collect();
                    let raw_payload=combination==PartitionCaptureCombination::SumF64ToF32 || matches!(transform,CaptureTransform::FullTensor|CaptureTransform::Preview{..});
                    let payload=if raw_payload {
                        let kept=if let CaptureTransform::Preview{max_elements}=transform{elements.min(max_elements)}else{elements};
                        let shape=if matches!(transform,CaptureTransform::Preview{..}){vec![kept as usize]}else{selected.iter().map(|n|*n as usize).collect()};
                        CapturePayload::Tensor(TensorObservation::new(shape,TensorObservationData::F32(values[..kept as usize].to_vec())).unwrap())
                    }else if matches!(transform,CaptureTransform::Summary) {CapturePayload::Summary(crate::capture::partition::summarize_f32(&values))}
                    else{let mut histogram=CaptureHistogram{edges:vec![-1.0,0.25,2.0],counts:vec![0,0],below:0,above:0,non_finite:0};
                        crate::capture::partition::fill_histogram_f32(&values,&mut histogram).unwrap();CapturePayload::Histogram(histogram)};
                    let selection=&source.admission().plan().selections[0];let point=&source.admission().points()[0];
                    fragments.push(PartitionCaptureFragmentRecord{fragment_index:0,record:CaptureRecord{schema_version:CAPTURE_SCHEMA_VERSION,
                        selection_id:selection.id.clone(),path:selection.path.clone(),node_id:point.node_id.clone(),position:point.position,
                        source_shape:Some(projection.local_shape().to_vec()),source_dtype:Some(TensorDtype::F16),selected_shape:Some(selected),
                        outcome:crate::capture::completed_capture_outcome(&transform,elements),payload:Some(payload),
                        charged:charges.iter().find(|(rank,_)|*rank==producer).unwrap().1}});
                }
                let envelope=PartitionCaptureProducerRecord{schema_version:PARTITION_CAPTURE_SCHEMA_VERSION,combination,
                    receipt_plan_identity:receipt.identity().into(),context:receipt.context().clone(),producer_rank:producer,source_dtype:Some(TensorDtype::F16),fragments};
                let bytes=serde_json::to_vec(&envelope).unwrap();delivery.receive(producer,&bytes,&mut ordinary_quota).unwrap();
                if producer!=0 {bank.decode_contiguous_producer(&receipt,producer,&bytes,Some(&TensorDtype::F16),&funding).unwrap();}
                else {
                    let (destination,mut loan)=bank.take_local(&receipt,0,&TensorDtype::F16,estimate(0)).unwrap().into_parts();
                    loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
                    let identity=PartitionCaptureTensorReceipt{context:receipt.context(),identity:receipt.identity(),producer:0,dtype:TensorDtype::F16,charged:envelope.fragments[0].record.charged};
                    let value=match destination {
                        PartitionFragmentDestination::Routed(_)=>panic!("dense fixture returned sparse assembly"),
                        PartitionFragmentDestination::Tensor(v)=>PartitionFragmentValue::Tensor(v.decode_partition_receipt_combined(&bytes,identity,&funding,combination).unwrap()),
                        PartitionFragmentDestination::Summary(v)=>PartitionFragmentValue::Summary(v.decode_partition_receipt(&bytes,identity,&funding).unwrap()),
                        PartitionFragmentDestination::Histogram(v)=>PartitionFragmentValue::Histogram(v.decode_partition_receipt(&bytes,identity,&funding).unwrap()),
                    };bank.record(&receipt,producer,0,&TensorDtype::F16,estimate(0).capture,value).unwrap();
                }
            }
            assert!(bank.complete());let expected=delivery.finish(&mut ordinary_quota).unwrap();
            let spent=quota.total();let (value,charged)=bank.assemble_into(&receipt,destination,&TensorDtype::F16).unwrap();
            assert_eq!(quota.total(),spent);assert_eq!(charged,expected.capture().record().charged);
            let actual=match &value {
                PartitionFragmentValue::Routed(_)=>panic!("dense fixture returned sparse assembly"),
                PartitionFragmentValue::Tensor(v)=>eredu_core::capture::CapturePayloadWire::Tensor(v.observation().as_observation()),
                PartitionFragmentValue::Summary(v)=>eredu_core::capture::CapturePayloadWire::Summary(v.observation()),
                PartitionFragmentValue::Histogram(v)=>eredu_core::capture::CapturePayloadWire::Histogram(v.observation()),
            };
            assert_eq!(serde_json::to_value(actual).unwrap(),serde_json::to_value(expected.capture().record().payload.as_ref().unwrap()).unwrap());
            if combination==PartitionCaptureCombination::SumF64ToF32&&matches!(transform,CaptureTransform::Histogram{..}) {
                let PartitionFragmentValue::Histogram(histogram)=&value else{unreachable!()};
                assert_eq!(histogram.observation().counts,[4,4]);assert_eq!(histogram.observation().above,2);
            }
            target.record(&mut frame,crate::working_memory::capture_run::PartitionFragmentDelivered{value,evidence,charged,dtype:TensorDtype::F16}).unwrap();
            assert_eq!(frame.records()[0].charged,expected.capture().record().charged);
            assert_eq!(frame.partition_evidence().len(),1);assert_eq!(quota.total(),spent);
            let escaped=match frame.records()[0].payload.as_ref().unwrap(){
                CapturePayload::SharedTensor(tensor)=>Some(tensor.clone()),_=>None,
            };
            drop(frame);drop(final_bank);drop(run);drop(reservation);assert_eq!(ledger(&pool).0>0,escaped.is_some());
            drop(escaped);assert_eq!(ledger(&pool).0,0);assert!(transport.calls.borrow().is_empty());
        }
    }
}

#[test]
fn incomplete_fragment_assembly_consumes_final_claim_and_retains_both_paid_destinations() {
    let source=source();let mut quota=CaptureLedger::new(source.admission());quota.begin_step();
    let (funding,_,_,_)=funding();let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
    let mut receipt=receipt(&source,PartitionCaptureCombination::Disjoint,&funding,&mut quota);
    let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
        .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
    let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
        producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform:&source.admission().plan().selections[0].transform,
        dtype:TensorDtype::F32,estimate:estimate(*rank)}).collect();
    let allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut quota).unwrap();
    let host=PartitionFragmentHostPlan::prepare(&receipt).unwrap();let h=host.initialization_peak_bytes()+plan(&source).initialization_peak_bytes();
    let pool=WorkingMemoryPool::new(h,0).unwrap();let (reservation,run)=fresh(&pool,h);
    let bank=run.prepare_partition_fragments(&reservation,host,allowance).unwrap();
    let mut final_bank=run.prepare_capture_run(&reservation,plan(&source)).unwrap();
    let mut frame=final_bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
    let spent=quota.total();let destination=PartitionFragmentDestination::Tensor(frame.take_tensor(0).unwrap());
    let error=bank.assemble_into(&receipt,destination,&TensorDtype::F32).unwrap_err();
    assert!(frame.take_tensor(0).is_err());assert!(frame.records()[0].payload.is_none());assert_eq!(quota.total(),spent);
    drop(frame);drop(final_bank);drop(run);drop(reservation);assert!(ledger(&pool).0>0);
    drop(error);assert_eq!(ledger(&pool).0,0);assert!(transport.calls.borrow().is_empty());
}

#[test]
fn paid_contiguous_evidence_keeps_sorted_empty_ranks_and_exact_fragment_charges() {
    for combination in [PartitionCaptureCombination::Disjoint, PartitionCaptureCombination::SumF64ToF32] {
        let source=source();let mut ledger=CaptureLedger::new(source.admission());ledger.begin_step();
        let (funding,_,_,retired)=funding();
        let transport=Receipts{source:Vec::new(),funding:funding.clone(),reject_delivery:false,calls:RefCell::new(Vec::new())};
        let mut receipt=receipt(&source,combination,&funding,&mut ledger);
        let geometry:Vec<_>=receipt.producers().flat_map(|(rank,p)|p.fragments().iter().enumerate()
            .map(move |(fragment,g)|(rank,fragment,p.local_shape().to_vec(),g.local().clone()))).collect();
        let raw=CaptureTransform::Slice;
        let transform=if combination==PartitionCaptureCombination::SumF64ToF32{&raw}else{&source.admission().plan().selections[0].transform};
        let rows:Vec<_>=geometry.iter().map(|(rank,fragment,shape,slice)|PartitionCaptureFragmentSource{
            producer:*rank,fragment:*fragment,local_shape:shape,local_slice:slice,transform,
            dtype:TensorDtype::Bf16,estimate:estimate(*rank)}).collect();
        let mut allowance=PreparedPartitionFragmentAllowance::prepare(&transport,&mut receipt,&rows,&funding,&mut ledger).unwrap();
        let spent=ledger.total();let prior=allowance.local_used();
        let evidence=crate::capture::partition::PreparedPartitionCaptureEvidence::prepare_contiguous(&receipt,&mut allowance,&funding).unwrap();
        assert!(evidence.matches(&receipt));assert_eq!(ledger.total(),spent);
        assert!(allowance.local_used().host_bytes>prior.host_bytes);
        assert_eq!(evidence.value.producers,if combination==PartitionCaptureCombination::Disjoint{vec![0,1,3]}else{vec![0,3]});
        assert_eq!(evidence.value.contributions.len(),2);
        for row in &evidence.value.contributions {
            let projection=receipt.producer(row.producer_rank).unwrap();let fragment=&projection.fragments()[0];
            assert_eq!(row.local.starts,fragment.local().starts);
            assert_eq!(row.local.ends,fragment.local().ends);
            assert_eq!(row.local.strides,fragment.local().strides);
            assert_eq!(row.local.shape,fragment.local().shape);
            assert_eq!(row.destination.starts,fragment.destination().starts);
            assert_eq!(row.destination.ends,fragment.destination().ends);
            assert_eq!(row.destination.strides,fragment.destination().strides);
            assert_eq!(row.destination.shape,fragment.destination().shape);
            assert_eq!(row.charged,allowance.fragment_record_charge(&receipt,row.producer_rank,0).unwrap().1);
            assert!(row.routed.is_none());
        }
        assert!(transport.calls.borrow().is_empty());
        drop(rows);drop(geometry);drop(allowance);drop(receipt);drop(source);drop(transport);drop(funding);
        assert!(!retired.load(Ordering::SeqCst));drop(evidence);assert!(retired.load(Ordering::SeqCst));assert_eq!(ledger.total(),spent);
    }
}

mod delivery;

mod prefill;

mod host_owner;

mod voted;
