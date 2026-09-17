use super::*;
use eredu_core::{
    checkpoint::TensorDtype, InputModality, InputPartDescriptor, InputPayloadKind,
    InputTensorIdentity, PreparedInputIdentity,
};
use std::cell::Cell;

thread_local! { static CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) }; }
pub(super) fn before_construct() {
    CONSTRUCTIONS.set(CONSTRUCTIONS.get() + 1);
}

fn legacy(batch: usize, positions: usize, tokens: &[u32]) -> PreparedInputCacheIdentity {
    PreparedInputCacheIdentity::new(
        PreparedInputIdentity::new(vec![InputPartDescriptor::new(
            InputModality::Text,
            InputPayloadKind::TokenIds,
            InputTensorIdentity::new(TensorDtype::U32, vec![batch, positions]).unwrap(),
            [],
        )
        .unwrap()])
        .unwrap(),
        eredu_core::cache::prompt_cache_token_fingerprint(tokens),
    )
    .unwrap()
}

#[test]
fn fixed_worker_matches_legacy_nonzero_and_large_input_fingerprints() {
    for (batch, positions) in [(1, 5), (2, 17), (1, 4097)] {
        let tokens = (0..batch * positions)
            .map(|index| (index as u32).wrapping_mul(0x01020305).wrapping_add(17))
            .collect::<Vec<_>>();
        let expected = legacy(batch, positions, &tokens);
        let plan = TextInputIdentityPlan::new(batch as u64, positions as u64).unwrap();
        let bytes = plan.retained_bytes();
        assert!(plan.peak_bytes() > bytes);
        let bound = plan.bind(&tokens).unwrap();
        assert_eq!(bound.tokens().as_ptr(), tokens.as_ptr());
        let actual = bound.construct().unwrap();
        assert_eq!(actual.as_ref(), &expected);
        assert_eq!(actual.capacity_bytes(), Some(bytes));
        assert_eq!(actual.as_ref().semantic_content_fingerprint.capacity(), 64);
        assert_eq!(actual.as_ref().prefix_content_fingerprint.capacity(), 64);
        assert_eq!(
            actual.prepared().encode_words().unwrap(),
            [1, 0, 0, 3, 2, batch as u32, positions as u32, 0, 0]
        );
    }
}

#[test]
fn bad_geometry_or_token_count_never_enters_the_allocating_worker() {
    let before = CONSTRUCTIONS.get();
    for (batch, positions) in [
        (0, 1),
        (1, 0),
        (u64::MAX, 1),
        (u32::MAX as u64, u32::MAX as u64),
    ] {
        assert!(TextInputIdentityPlan::new(batch, positions).is_err());
    }
    for tokens in [&[][..], &[1, 2, 3][..], &[1, 2, 3, 4, 5, 6][..]] {
        let error = TextInputIdentityPlan::new(1, 5)
            .unwrap()
            .bind(tokens)
            .unwrap_err();
        assert!(
            matches!(error, TextInputIdentityError::TokenCount { expected: 5, actual } if actual == tokens.len())
        );
    }
    assert_eq!(CONSTRUCTIONS.get(), before);
}

#[test]
fn equal_geometry_does_not_hide_changed_token_content_or_batch_identity() {
    let first = [17, 0, u32::MAX, 31, 2, 89];
    let mut second = first;
    second[3] += 1;
    let a = TextInputIdentityPlan::new(1, 6)
        .unwrap()
        .bind(&first)
        .unwrap()
        .construct()
        .unwrap();
    let b = TextInputIdentityPlan::new(1, 6)
        .unwrap()
        .bind(&second)
        .unwrap()
        .construct()
        .unwrap();
    let reshaped = TextInputIdentityPlan::new(2, 3)
        .unwrap()
        .bind(&first)
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(a.prepared(), b.prepared());
    assert_ne!(
        a.semantic_content_fingerprint(),
        b.semantic_content_fingerprint()
    );
    assert_ne!(
        a.prefix_content_fingerprint(),
        b.prefix_content_fingerprint()
    );
    assert_eq!(
        a.semantic_content_fingerprint(),
        reshaped.semantic_content_fingerprint()
    );
    assert_ne!(
        a.prefix_content_fingerprint(),
        reshaped.prefix_content_fingerprint()
    );
}

#[test]
fn final_owner_clones_share_fixed_payload_after_source_retirement() {
    let tokens = vec![17, 31, 59, 83, 107];
    let owner = TextInputIdentityPlan::new(1, 5)
        .unwrap()
        .bind(&tokens)
        .unwrap()
        .construct()
        .unwrap();
    let fingerprint = owner.prefix_content_fingerprint().to_owned();
    let bytes = owner.capacity_bytes();
    let cloned = owner.clone();
    assert!(owner.same_storage(&cloned));
    drop((owner, tokens));
    assert_eq!(cloned.prefix_content_fingerprint(), fingerprint);
    assert_eq!(cloned.capacity_bytes(), bytes);
    assert_eq!(cloned.prepared().parts()[0].payload().shape(), [1, 5]);
}
