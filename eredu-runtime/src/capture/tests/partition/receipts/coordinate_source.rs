use super::*;
use eredu_core::capture::SharedCapturePlan;
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Account {
    used: Arc<AtomicUsize>,
    limit: usize,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map(|_| ())
            .map_err(|used| HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: (self.limit - used) as u64,
            })
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
#[test]
fn coordinate_source_quote_funds_permuted_maps_and_retains_refusal_payer() {
    let source = SharedCapturePlan::new(plan_for(CaptureTransform::Slice, false));
    let maps = [ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()];
    let mut ledger = CaptureLedger::new(source.admission());
    let receipt = authority(source.admission(), &maps, &mut ledger);
    let producers = [PartitionCaptureCoordinateProducer {
        rank: 0,
        coordinates: &maps[0],
    }];
    let projection = receipt.producer(0).unwrap();
    let native: Vec<_> = projection
        .fragments()
        .iter()
        .enumerate()
        .map(|(fragment, row)| PartitionCaptureFragmentGeometry {
            producer: 0,
            fragment,
            local_shape: projection.local_shape(),
            local_slice: row.local(),
            transform: &source.admission().plan().selections[0].transform,
            estimate: PartitionCaptureNativeEstimate {
                capture: CaptureUsage::default(),
                generated_creation_bytes: 0,
            },
        })
        .collect();
    let local = || {
        Some(PartitionCaptureLocalSource {
            producer: 0,
            shape: projection.local_shape(),
            dtype: TensorDtype::F32,
        })
    };
    let quote = PreparedPartitionContiguousSource::local_coordinates_metadata_bytes(
        producers.len(),
        producers
            .iter()
            .map(|row| PartitionCaptureCoordinateProducer {
                rank: row.rank,
                coordinates: row.coordinates,
            }),
        native.iter().map(|row| PartitionCaptureFragmentGeometry {
            producer: row.producer,
            fragment: row.fragment,
            local_shape: row.local_shape,
            local_slice: row.local_slice,
            transform: row.transform,
            estimate: row.estimate,
        }),
        local(),
    )
    .unwrap();
    assert!(
        PreparedPartitionContiguousSource::local_coordinates_metadata_bytes(
            2,
            producers
                .iter()
                .map(|row| PartitionCaptureCoordinateProducer {
                    rank: row.rank,
                    coordinates: row.coordinates
                }),
            std::iter::empty(),
            local()
        )
        .is_none()
    );
    let mut actual_spent = quote;
    for short in [false, true] {
        let baseline = HostMetadataFunding::constructor_bytes::<Account>().unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            used: used.clone(),
            limit: baseline + actual_spent - usize::from(short),
            retired: retired.clone(),
        })
        .unwrap();
        let result = PreparedPartitionContiguousSource::new_local_coordinates_invocation(
            &source,
            0,
            1,
            &producers,
            &native,
            local(),
            PartitionCaptureCombination::Disjoint,
            CapturePhase::Prefill,
            0,
            &funding,
        );
        if short {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok(), "{result:?}");
            actual_spent = used.load(Ordering::SeqCst) - baseline;
            assert!(actual_spent <= quote);
        }
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(result);
        assert!(retired.load(Ordering::SeqCst));
    }
}
