use super::super::routed as fixture;
use super::*;

#[test]
fn sparse_exchange_uses_existing_completion_and_all_rank_delivery_agreement() {
    for failure in [None, Some("preparation"), Some("missing_row")] {
        let world = world(8);
        let threads: Vec<_> = (0..8)
            .map(|rank| {
                let world = Arc::clone(&world);
                std::thread::spawn(move || {
                    let plan = fixture::plan(false);
                    let transport = transport(world, rank, Fault::None);
                    let mut ledger = CaptureLedger::new(&plan);
                    let receipt = fixture::receipt(&plan, true, &mut ledger);
                    let mut records = fixture::records(&plan, &receipt);
                    if failure == Some("missing_row") && rank == 2 {
                        let Some(CapturePayload::RoutedUnits(payload)) =
                            &mut records[rank].fragments[0].record.payload
                        else {
                            panic!("sparse")
                        };
                        payload.rows.pop();
                    }
                    let mut delivery_quota = ledger
                        .reserve_quota(receipt.delivery_usage().unwrap())
                        .unwrap();
                    let exchange =
                        PartitionCaptureExchange::admit(&transport, receipt, &mut ledger).unwrap();
                    assert_eq!(
                        transport.calls.load(Ordering::SeqCst),
                        0,
                        "cold admission performs no transport"
                    );
                    let local = if failure == Some("preparation") && rank == 2 {
                        Err(CaptureError::Invalid(
                            "sparse producer failed before collection".into(),
                        )
                        .into())
                    } else {
                        Ok(records
                            .get(rank)
                            .map(|record| serde_json::to_vec(record).unwrap()))
                    };
                    let charged = ledger.total();
                    let result = exchange.exchange(local, &mut delivery_quota);
                    assert_eq!(
                        ledger.total(),
                        charged,
                        "prepaid delivery is cumulative even after failure"
                    );
                    assert!(
                        !transport.poison.load(Ordering::SeqCst),
                        "agreed rejection is not uncertain native completion"
                    );
                    match failure {
                        None => {
                            let result = result.unwrap();
                            fixture::assert_values(&result, true);
                            assert_eq!(transport.calls.load(Ordering::SeqCst), 3);
                            assert_eq!(transport.resolves.load(Ordering::SeqCst), 3);
                        }
                        Some("preparation") => {
                            assert!(result.is_err());
                            assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
                        }
                        _ => {
                            assert!(matches!(
                                result,
                                Err(PartitionCaptureExchangeError::LocalRejected {
                                    stage: PartitionCaptureExchangeStage::Delivery,
                                    ..
                                })
                            ));
                            assert_eq!(transport.calls.load(Ordering::SeqCst), 3);
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}

mod funded;
