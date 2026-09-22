use super::*;
use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    InputModality, InputPartDescriptor, InputPayloadKind, InputTensorIdentity, OutputDemand,
    PreparedInputIdentity, SharedStorageAccountingId,
};
use eredu_nn::workspace::WorkspaceTensor;
use std::{
    convert::Infallible,
    sync::atomic::{AtomicUsize, Ordering},
};

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}

fn identity() -> PreparedInputCacheIdentity {
    let mut shape = Vec::with_capacity(9);
    shape.extend([1, 5]);
    let prepared = PreparedInputIdentity::new(vec![InputPartDescriptor::new(
        InputModality::Text,
        InputPayloadKind::TokenIds,
        InputTensorIdentity::new(TensorDtype::U32, shape).unwrap(),
        [],
    )
    .unwrap()])
    .unwrap();
    let mut fingerprint = String::with_capacity(93);
    fingerprint.push_str("ordered-text-identity-λ");
    PreparedInputCacheIdentity::new(prepared, fingerprint).unwrap()
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn text_and_composite_prefill_retain_the_same_shared_identity_and_custody() {
    let shared = SharedPreparedInputCacheIdentity::new(identity());
    let key = shared.identity().clone();
    let capacity = shared.capacity_bytes();
    let pointer = shared.semantic_content_fingerprint().as_ptr();
    let tokens: Arc<[i32]> = Arc::from([3, 5, 7, 11, 13]);
    let text = PreparedTextPrefill::<WorkspaceTensor>::from_token_ids(tokens.clone(), geometry())
        .unwrap()
        .with_shared_cache_identity(shared.clone());
    let copy = text.clone();
    let composite = PreparedCompositeTextPrefill::<(), WorkspaceTensor, ()>::from_token_ids(
        tokens,
        geometry(),
        (),
        (),
    )
    .unwrap()
    .with_shared_cache_identity(shared.clone());
    let retired = Arc::new(AtomicUsize::new(0));
    shared
        .try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
            Ok(Box::new(Retired(retired.clone())))
        })
        .unwrap();
    for owner in [
        text.identity.as_ref().unwrap(),
        copy.identity.as_ref().unwrap(),
        composite.text.identity.as_ref().unwrap(),
    ] {
        assert_eq!(owner.identity(), &key);
        assert_eq!(owner.capacity_bytes(), capacity);
        assert_eq!(owner.semantic_content_fingerprint().as_ptr(), pointer);
        assert!(owner.same_storage(&shared));
    }
    drop((shared, text, composite));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        copy.identity
            .as_ref()
            .unwrap()
            .semantic_content_fingerprint(),
        "ordered-text-identity-λ"
    );
    drop(copy);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn legacy_raw_prefill_adapter_moves_identity_payload_without_rehashing_or_cloning() {
    let raw = identity();
    let capacity = raw.capacity_bytes();
    let pointers = (
        raw.prepared().parts().as_ptr(),
        raw.semantic_content_fingerprint().as_ptr(),
        raw.prefix_content_fingerprint().as_ptr(),
    );
    let text = PreparedTextPrefill::<WorkspaceTensor>::from_token_ids(
        Arc::from([3, 5, 7, 11, 13]),
        geometry(),
    )
    .unwrap()
    .with_cache_identity(raw);
    let retained = text.identity.as_ref().unwrap();
    assert_eq!(retained.capacity_bytes(), capacity);
    assert_eq!(
        (
            retained.prepared().parts().as_ptr(),
            retained.semantic_content_fingerprint().as_ptr(),
            retained.prefix_content_fingerprint().as_ptr()
        ),
        pointers
    );
    let raw = identity();
    let fingerprint = raw.prefix_content_fingerprint().as_ptr();
    let composite = PreparedCompositeTextPrefill::<(), WorkspaceTensor, ()>::from_token_ids(
        Arc::from([3, 5, 7, 11, 13]),
        geometry(),
        (),
        (),
    )
    .unwrap()
    .with_cache_identity(raw);
    assert_eq!(
        composite
            .text
            .identity
            .as_ref()
            .unwrap()
            .prefix_content_fingerprint()
            .as_ptr(),
        fingerprint
    );
}
