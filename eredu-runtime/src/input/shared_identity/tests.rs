use super::*;
use eredu_core::{
    checkpoint::TensorDtype, InputExtent, InputMetadataKey, InputModality, InputPartDescriptor,
    InputPayloadKind, InputTensorIdentity,
};
use std::{
    convert::Infallible,
    mem::size_of,
    sync::atomic::{AtomicUsize, Ordering},
};

fn tensor(dtype: TensorDtype, dimensions: &[usize]) -> InputTensorIdentity {
    let mut shape = Vec::with_capacity(dimensions.len() + 5);
    shape.extend_from_slice(dimensions);
    InputTensorIdentity::new(dtype, shape).unwrap()
}

fn identity() -> PreparedInputCacheIdentity {
    let mut parts = Vec::with_capacity(11);
    parts.push(
        InputPartDescriptor::new(
            InputModality::Text,
            InputPayloadKind::TokenIds,
            tensor(TensorDtype::U32, &[1, 5]),
            [],
        )
        .unwrap(),
    );
    parts.push(
        InputPartDescriptor::new_with_extents(
            InputModality::Image,
            InputPayloadKind::Tensor,
            tensor(TensorDtype::F32, &[7, 16]),
            [
                (
                    InputMetadataKey::PatchPositions,
                    tensor(TensorDtype::I32, &[7, 2]),
                ),
                (
                    InputMetadataKey::PatchGrid,
                    tensor(TensorDtype::I32, &[1, 3]),
                ),
            ],
            [InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 4,
            }],
        )
        .unwrap(),
    );
    parts.push(
        InputPartDescriptor::new_with_extents(
            InputModality::Video,
            InputPayloadKind::Embeddings,
            tensor(TensorDtype::F32, &[1, 14, 16]),
            [],
            [InputExtent::PatchGrid {
                time: 2,
                height: 2,
                width: 4,
            }],
        )
        .unwrap(),
    );
    parts.push(
        InputPartDescriptor::new_with_extents(
            InputModality::Audio,
            InputPayloadKind::Tensor,
            tensor(TensorDtype::F32, &[1, 8, 13]),
            [(
                InputMetadataKey::AudioMask,
                tensor(TensorDtype::Bool, &[1, 8]),
            )],
            [InputExtent::AudioValidFrames(6)],
        )
        .unwrap(),
    );
    let mut fingerprint = String::with_capacity(129);
    fingerprint.push_str("semantic-π-image-video-audio");
    PreparedInputCacheIdentity::new(PreparedInputIdentity::new(parts).unwrap(), fingerprint)
        .unwrap()
}

#[test]
fn consuming_shared_identity_preserves_actual_capacity_and_original_fingerprints() {
    let original = identity();
    let expected = size_of::<PreparedInputCacheIdentity>() as u64
        + original.prepared.capacity_bytes().unwrap()
        - size_of::<PreparedInputIdentity>() as u64
        + original.semantic_content_fingerprint.capacity() as u64
        + original.prefix_content_fingerprint.capacity() as u64;
    assert!(
        original.semantic_content_fingerprint.capacity()
            > original.semantic_content_fingerprint.len()
    );
    assert_eq!(original.capacity_bytes(), Some(expected));
    let pointers = (
        original.prepared().parts().as_ptr(),
        original.semantic_content_fingerprint().as_ptr(),
        original.prefix_content_fingerprint().as_ptr(),
    );
    let fingerprint = original.prefix_content_fingerprint().to_owned();
    let words = original.prepared().encode_words().unwrap();
    let shared = SharedPreparedInputCacheIdentity::new(original);
    assert_eq!(shared.prepared().len(), 4);
    let alias = shared.clone();
    for owner in [&shared, &alias] {
        assert_eq!(owner.capacity_bytes(), Some(expected));
        assert_eq!(owner.prefix_content_fingerprint(), fingerprint);
        assert_eq!(owner.prepared().encode_words().unwrap(), words);
        assert_eq!(
            (
                owner.prepared().parts().as_ptr(),
                owner.semantic_content_fingerprint().as_ptr(),
                owner.prefix_content_fingerprint().as_ptr()
            ),
            pointers
        );
    }
    assert!(shared.same_storage(&alias));
    assert_eq!(shared.identity(), alias.identity());
    assert!(std::ptr::eq(shared.as_ref(), alias.as_ref()));
    let independent = SharedPreparedInputCacheIdentity::new(shared.as_ref().clone());
    assert_eq!(shared, independent);
    assert!(!shared.same_storage(&independent));
    assert_ne!(shared.identity(), independent.identity());
    assert_ne!(
        shared.semantic_content_fingerprint().as_ptr(),
        independent.semantic_content_fingerprint().as_ptr()
    );
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn existing_aliases_share_domain_custody_and_identity_keys_do_not_pin_payload() {
    let owner = SharedPreparedInputCacheIdentity::new(identity());
    let weak = Arc::downgrade(owner.0.as_ref().expect("live cache"));
    let alias = owner.clone();
    let key = owner.identity().clone();
    let a = SharedStorageDomain::default();
    let b = SharedStorageDomain::default();
    let retired = Arc::new(AtomicUsize::new(0));
    assert!(owner
        .try_attach::<Infallible>(&a, || Ok(Box::new(Retired(retired.clone()))))
        .unwrap());
    assert!(!alias
        .try_attach::<Infallible>(&a, || panic!("same domain must not reacquire"))
        .unwrap());
    assert!(alias
        .try_attach::<Infallible>(&b, || Ok(Box::new(Retired(retired.clone()))))
        .unwrap());
    drop(owner);
    assert!(weak.upgrade().is_some());
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(alias.prepared().len(), 4);
    drop(alias);
    assert!(weak.upgrade().is_none());
    assert_eq!(retired.load(Ordering::SeqCst), 2);
    assert_eq!(key, key.clone());
}

#[test]
fn failed_attachment_preserves_payload_and_does_not_claim_domain() {
    let owner = SharedPreparedInputCacheIdentity::new(identity());
    let alias = owner.clone();
    let domain = SharedStorageDomain::default();
    let key = owner.identity().clone();
    let before = (
        owner.capacity_bytes(),
        owner.prefix_content_fingerprint().as_ptr(),
    );
    let error = owner
        .try_attach::<&'static str>(&domain, || Err("rejected source allowance"))
        .unwrap_err();
    assert!(matches!(
        error,
        SharedStorageAttachmentError::Provider("rejected source allowance")
    ));
    assert_eq!(
        (
            owner.capacity_bytes(),
            owner.prefix_content_fingerprint().as_ptr()
        ),
        before
    );
    assert_eq!(alias.identity(), &key);
    let retired = Arc::new(AtomicUsize::new(0));
    assert!(alias
        .try_attach::<Infallible>(&domain, || Ok(Box::new(Retired(retired.clone()))))
        .unwrap());
    drop((owner, alias));
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
