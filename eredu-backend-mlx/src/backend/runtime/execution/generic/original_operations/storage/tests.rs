use super::*;

#[test]
fn impossible_payload_destination_refuses_before_factory_or_original_custody() {
    // Layout paths receive no guard; invoking either real factory would panic.
    // A nonzero valid layout can be inspected without creating slots or native work.
    let transfer = TransferPayloadShape {
        pending_sources: 3,
        retained_arrays: 7,
        retained_host: 2,
        retained_events: 1,
        unit_ids: 4,
        ..TransferPayloadShape::default()
    };
    let weight = MaterializationPayloadShape {
        inputs: 1,
        outputs: 7,
        pending_sources: 3,
    };
    assert!(transfer_layout(2, transfer, NamedStorageLayout::default()).unwrap() > 0);
    assert!(weight_layout(2, weight).unwrap() > 0);
    for shape in [
        TransferPayloadShape {
            pending_sources: usize::MAX,
            ..transfer
        },
        TransferPayloadShape {
            retained_arrays: usize::MAX,
            ..transfer
        },
        TransferPayloadShape {
            unit_ids: usize::MAX,
            ..transfer
        },
    ] {
        assert!(transfer_layout(1, shape, NamedStorageLayout::default()).is_none());
    }
    assert!(transfer_layout(
        2,
        transfer,
        NamedStorageLayout {
            destination_requested_bytes: usize::MAX,
            ..NamedStorageLayout::default()
        }
    )
    .is_none());
    for shape in [
        MaterializationPayloadShape {
            inputs: usize::MAX,
            ..weight
        },
        MaterializationPayloadShape {
            outputs: usize::MAX,
            ..weight
        },
        MaterializationPayloadShape {
            pending_sources: usize::MAX,
            ..weight
        },
    ] {
        assert!(weight_layout(1, shape).is_none());
    }
}
