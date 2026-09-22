use super::*;
use crate::capture::partition::PartitionCaptureCoordinateProducer;
use eredu_core::{capture::CaptureCoordinateProjectionPlan, component::ComponentCoordinateMap};
#[test]
fn funded_component_receipt_preserves_permuted_runs_empty_ack_and_ordinary_identity() {
    for selection in [0, 2] {
        let source = source();
        let context = context(&source, selection);
        let geometry = CaptureTensorGeometry::prepare(
            source.admission(),
            selection,
            CapturePhase::Prefill,
            0,
            None,
        )
        .unwrap();
        let global: Vec<u64> = geometry.source_shape().iter().map(|&n| n as u64).collect();
        let axis = global.len() - 1;
        let width = global[axis] as usize;
        let maps = [
            ComponentCoordinateMap::indices(width, (0..width).rev().step_by(2).collect()).unwrap(),
            ComponentCoordinateMap::indices(width, vec![]).unwrap(),
            ComponentCoordinateMap::indices(width, (0..width).rev().skip(1).step_by(2).collect())
                .unwrap(),
        ];
        let rows = [
            PartitionCaptureCoordinateProducer {
                rank: 3,
                coordinates: &maps[0],
            },
            PartitionCaptureCoordinateProducer {
                rank: 0,
                coordinates: &maps[1],
            },
            PartitionCaptureCoordinateProducer {
                rank: 1,
                coordinates: &maps[2],
            },
        ];
        let slice = resolve_slice(
            &source.admission().points()[selection],
            &source.admission().plan().selections[selection],
            &global,
        )
        .unwrap();
        let limits = PartitionCaptureReceiptLimits {
            max_producers: 3,
            max_fragments: width,
            max_record_bytes: 64 << 10,
        };
        let ordinary_rows = rows
            .iter()
            .map(|row| PartitionCaptureProducer {
                rank: row.rank,
                projection: CaptureCoordinateProjectionPlan::prepare(
                    &global,
                    &slice,
                    axis,
                    row.coordinates,
                    width,
                )
                .unwrap()
                .construct(),
            })
            .collect();
        let mut ordinary_ledger = CaptureLedger::new(source.admission());
        ordinary_ledger.begin_step();
        let ordinary = PartitionCaptureReceiptPlan::new(
            source.clone(),
            context.clone(),
            ordinary_rows,
            4,
            limits,
            &mut ordinary_ledger,
        )
        .unwrap();
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let (funding, _, _, retired) = funding();
        let actual = PartitionCaptureReceiptPlan::new_coordinates_shared_funded(
            &source,
            &context,
            axis,
            &rows,
            PartitionCaptureCombination::Disjoint,
            4,
            limits,
            &funding,
            &mut ledger,
        )
        .unwrap();
        assert_eq!(actual.identity(), ordinary.identity());
        assert!(actual.producer(0).unwrap().fragments().is_empty());
        assert!(actual.producer(2).is_none());
        for (rank, projection) in actual.producers() {
            assert_eq!(projection, ordinary.producer(rank).unwrap());
        }
        assert_eq!(
            actual.delivery_usage().unwrap(),
            ordinary.delivery_usage().unwrap()
        );
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(actual);
        assert!(retired.load(Ordering::SeqCst));
    }
}
