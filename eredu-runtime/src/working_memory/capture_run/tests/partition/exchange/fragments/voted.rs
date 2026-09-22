use super::*;
use crate::capture::partition::{
    PartitionCaptureFragmentGeometry, PreparedPartitionFragmentSourceAllowance,
};

#[test]
fn unvoted_fragment_allowance_uses_same_costs_and_binds_scalar_without_refund() {
    for reject in [false, true] {
        let source = source();
        let (metadata, _, _, retired) = funding();
        let transport = Receipts {
            source: vec![],
            funding: metadata.clone(),
            reject_delivery: false,
            calls: RefCell::new(vec![]),
        };
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let mut receipt = receipt(
            &source,
            PartitionCaptureCombination::SumF64ToF32,
            &metadata,
            &mut ledger,
        );
        let geometry: Vec<_> = receipt
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments().iter().enumerate().map(move |(fragment, g)| {
                    (rank, fragment, p.local_shape().to_vec(), g.local().clone())
                })
            })
            .collect();
        let raw = CaptureTransform::Slice;
        let rows: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentGeometry {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform: &raw,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let unbound = PreparedPartitionFragmentSourceAllowance::prepare(
            &transport,
            &mut receipt,
            &rows,
            &metadata,
            &mut ledger,
        )
        .unwrap();
        let spent = ledger.total();
        let scalar = if reject {
            TensorDtype::I32
        } else {
            TensorDtype::F16
        };
        let result = unbound.bind(&receipt, scalar);
        if reject {
            let error = result.unwrap_err();
            drop(rows);
            drop(geometry);
            drop(receipt);
            drop(source);
            drop(transport);
            drop(metadata);
            assert!(!retired.load(Ordering::SeqCst));
            drop(error);
            assert!(retired.load(Ordering::SeqCst));
        } else {
            let mut allowance = result.unwrap();
            assert!(allowance.matches(&receipt));
            let mut loan = allowance
                .take_local_fragment(&receipt, 0, &TensorDtype::F16, estimate(0))
                .unwrap();
            loan.quota_mut().reserve_quota(estimate(0).capture).unwrap();
            assert!(
                allowance
                    .take_local_fragment(&receipt, 0, &TensorDtype::F16, estimate(0))
                    .is_err()
            );
            drop(rows);
            drop(geometry);
            drop(receipt);
            drop(source);
            drop(transport);
            drop(metadata);
            drop(allowance);
            assert!(!retired.load(Ordering::SeqCst));
            drop(loan);
            assert!(retired.load(Ordering::SeqCst));
        }
        assert_eq!(ledger.total(), spent);
    }
}
