use super::*;

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 19,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: eredu_core::OutputDemand::LastPosition,
    }
}

#[test]
fn ordinary_coordinates_preserve_prefix_frontier_and_final_capture() {
    let coordinates = PredictionCoordinates::new(0, geometry()).unwrap();
    for (absolute, frontier) in [(0, 19), (1, 24), (2, 25), (3, 26)] {
        assert_eq!(coordinates.local(absolute), Ok(absolute));
        assert_eq!(coordinates.frontier(absolute), Ok(frontier));
    }
    assert_eq!(
        coordinates.local(4),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 4,
            limit: 3,
        })
    );
}

#[test]
fn nonzero_origin_keeps_absolute_observations_distinct_from_local_receipts() {
    // Arithmetic conformance only: no resumed request or native grant exists.
    let mut geometry = geometry();
    geometry.input_positions = 1;
    geometry.prefill_chunk_positions = 1;
    let coordinates = PredictionCoordinates::new(7, geometry).unwrap();
    for (absolute, local, frontier) in [(7, 0, 19), (8, 1, 20), (9, 2, 21), (10, 3, 22)] {
        assert_eq!(coordinates.local(absolute), Ok(local));
        assert_eq!(coordinates.frontier(absolute), Ok(frontier));
    }
    assert_eq!(
        coordinates.local(6),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        coordinates.frontier(6),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        coordinates.frontier(11),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 4,
            limit: 3,
        })
    );
}

#[test]
fn stateless_coordinates_need_no_array_and_keep_nonzero_absolute_frontier() {
    // The same metadata is sufficient when every native layer is Stateless;
    // this does not fabricate a stateless model or a saved-source proof.
    let mut geometry = geometry();
    geometry.input_positions = 1;
    geometry.prefill_chunk_positions = 1;
    let coordinates = PredictionCoordinates::new(42, geometry).unwrap();
    assert_eq!(coordinates.frontier(42), Ok(19));
    assert_eq!(coordinates.frontier(43), Ok(20));
    assert_ne!(coordinates.frontier(43).unwrap(), 43);
    assert_ne!(
        coordinates.frontier(43).unwrap(),
        coordinates.local(43).unwrap()
    );
}

#[test]
fn coordinate_endpoints_reject_overflow_before_any_later_query() {
    assert!(matches!(
        PredictionCoordinates::new(u64::MAX - 2, geometry()),
        Err(WorkingMemoryError::Overflow)
    ));
    let mut geometry = geometry();
    geometry.cached_positions = u64::MAX - 4;
    assert!(matches!(
        PredictionCoordinates::new(0, geometry),
        Err(WorkingMemoryError::Overflow)
    ));
    geometry.cached_positions = u64::MAX - 8;
    let coordinates = PredictionCoordinates::new(u64::MAX - 3, geometry).unwrap();
    assert_eq!(coordinates.local(u64::MAX), Ok(3));
    assert_eq!(coordinates.frontier(u64::MAX), Ok(u64::MAX - 1));
}
