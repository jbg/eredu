use super::*;
use std::cell::Cell;

thread_local! { static CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) }; }
pub(super) fn before_construct() {
    CONSTRUCTIONS.set(CONSTRUCTIONS.get() + 1);
}

#[test]
fn fixed_text_descriptor_has_exact_capacity_and_canonical_words() {
    let mut geometries = vec![(1, 5), (2, 17)];
    if (u32::MAX as u64) * 4 <= isize::MAX as u64 {
        geometries.push((1, u32::MAX as u64));
    }
    for (batch, positions) in geometries {
        let plan = TextTokenInputDescriptorPlan::new(batch, positions).unwrap();
        assert_eq!(plan.token_count() as u64, batch * positions);
        let bytes = plan.retained_bytes();
        let descriptor = plan.construct();
        assert_eq!(descriptor.capacity_bytes(), Some(bytes));
        assert_eq!(descriptor.parts.capacity(), 1);
        assert_eq!(descriptor.parts[0].payload.shape.capacity(), 2);
        assert!(descriptor.parts[0].metadata.is_empty());
        assert!(descriptor.parts[0].extents.is_empty());
        assert_eq!(
            descriptor.encode_words().unwrap(),
            vec![1, 0, 0, 3, 2, batch as u32, positions as u32, 0, 0]
        );
    }
}

#[test]
fn invalid_geometry_rejects_before_descriptor_construction() {
    let before = CONSTRUCTIONS.get();
    for (batch, positions) in [
        (0, 5),
        (1, 0),
        (u64::MAX, 1),
        (1, u64::MAX),
        (u32::MAX as u64, u32::MAX as u64),
    ] {
        assert!(TextTokenInputDescriptorPlan::new(batch, positions).is_err());
    }
    assert_eq!(CONSTRUCTIONS.get(), before);
}
