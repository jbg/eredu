//! Scoped reports cover original admission roots as well as prepared products.
use super::*;
use eredu_architectures::{
    qwen::ingress::PreparedInputTensorRole as IngressRole,
    qwen4_exp::{
        conditional::prefill::{MediaPrefillRequest, MediaPrefillTensorRole as Role},
        media::PreparedMediaTensorRole as PreparedRole,
    },
};
use eredu_runtime::input::PreparedInputTensorRole as OriginalRole;

#[test]
fn fresh_request_from_chunk_admission_releases_old_continuation_roots() {
    use eredu_architectures::{
        composite_execution::PreparedCompositeArchitecture,
        qwen4_exp::conditional::ConditionalModel,
    };
    use eredu_runtime::prefill::ChunkedPrefillRequest;
    type Request = MediaPrefillRequest<NumericTensor>;
    type Model = PreparedCompositeArchitecture<ConditionalModel<NumericBackend>>;
    let (_dir, _target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let request = Request::new(
        ingress
            .admission_config()
            .admit(&source, &inspector)
            .unwrap()
            .into_composite(),
        &ctx,
    )
    .unwrap();
    // This opaque integer tensor is an ownership probe only; no encoder or
    // decoder equation consumes it. Its exact integer storage already uses Arc.
    let probe = NumericTensor::from_i32_slice(&[7], &[1], &ctx).unwrap();
    let weak = std::sync::Arc::downgrade(probe.exact_i32.as_ref().unwrap());
    let continuation = Some(probe);
    let chunk_admission =
        <Request as ChunkedPrefillRequest<Model, NumericBackend, State>>::with_chunk(
            &request,
            0..1,
            Some(&continuation),
            |input| input.admitted().clone(),
        )
        .unwrap();
    drop(continuation);
    drop(request);
    assert!(
        weak.upgrade().is_some(),
        "chunk admission retains its continuation"
    );
    let fresh = Request::new(chunk_admission, &ctx).unwrap();
    assert!(
        weak.upgrade().is_none(),
        "fresh request must release the previous chunk products"
    );
    let snapshot = fresh.inspect_storage(None).unwrap();
    snapshot.validate().unwrap();
    assert_eq!(snapshot.values.len(), 21);
    assert!(!snapshot
        .values
        .iter()
        .any(|value| value.role == Role::EncoderOutput));
}

#[test]
fn request_storage_covers_original_metadata_joined_patches_and_encoder_continuation() {
    let (_dir, _target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let admitted = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap();
    let prepared = admitted.prepare(&ctx).unwrap();
    let projected = prepared
        .with_vision_input(|input| tower(&ingress, &ctx).forward(input.unwrap(), &ctx))
        .unwrap()
        .embeddings;
    let request = MediaPrefillRequest::new(admitted.into_composite(), &ctx).unwrap();
    let reads = inspector.0.get();
    let before = request.inspect_storage(None).unwrap();
    before.validate().unwrap();
    assert_eq!(before.values.len(), 21);
    let role_bytes = |role| {
        before
            .values
            .iter()
            .find(|value| value.role == role)
            .unwrap()
            .logical_bytes
    };
    for part in [1, 3, 5] {
        assert_eq!(
            role_bytes(Role::Original(OriginalRole::Metadata {
                part,
                key: K::PatchGrid
            })),
            Some(12)
        );
    }
    assert_eq!(
        role_bytes(Role::Original(OriginalRole::Payload { part: 1 })),
        Some(16 * 24 * 4)
    );
    assert_eq!(
        role_bytes(Role::Prepared(PreparedRole::Ingress(IngressRole::Pixels))),
        Some(32 * 24 * 4)
    );
    assert_eq!(
        role_bytes(Role::Prepared(PreparedRole::TokenIds)),
        Some(18 * 4)
    );
    assert_eq!(
        role_bytes(Role::Prepared(PreparedRole::RotaryCosine)),
        Some(18 * 6 * 4)
    );
    assert_eq!(
        role_bytes(Role::Prepared(PreparedRole::RotarySine)),
        Some(18 * 6 * 4)
    );
    assert!(before.storage.values.iter().all(Option::is_none));
    assert!(before.storage.backings.is_empty());
    let after = request.inspect_storage(Some(&projected)).unwrap();
    assert_eq!(&after.values[..before.values.len()], before.values);
    assert_eq!(after.values.last().unwrap().role, Role::EncoderOutput);
    assert_eq!(after.values.last().unwrap().logical_bytes, Some(8 * 32 * 4));

    assert_eq!(
        inspector.0.get(),
        reads,
        "storage observation performs no metadata transfers"
    );
    let cloned = request.clone();
    drop(request);
    drop(source);
    assert_eq!(cloned.inspect_storage(Some(&projected)).unwrap(), after);
    drop(cloned);
    drop(projected);
    after.validate().unwrap();
}

#[test]
fn projected_and_text_only_requests_report_every_root_without_inventing_encoder_output() {
    let (_dir, _target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    for projected in [false, true] {
        let source = if projected {
            input(
                vec![PreparedInputPart::new(
                    M::Image,
                    P::Embeddings(NumericTensor::new(
                        [1, 2, 32],
                        (0..64).map(|i| i as f32 / 64.).collect(),
                    )),
                    [
                        (
                            K::OriginalTokenIds,
                            NumericTensor::from_i32_slice(&[12, 12], &[1, 2], &ctx).unwrap(),
                        ),
                        (K::PatchGrid, grid(1, 2, 4)),
                    ],
                )
                .unwrap()],
                &inspector,
            )
        } else {
            input(vec![text(&[3, 4])], &inspector)
        };
        let admitted = ingress
            .admission_config()
            .admit(&source, &inspector)
            .unwrap();
        let request = MediaPrefillRequest::new(admitted.into_composite(), &ctx).unwrap();
        let snapshot = request.inspect_storage(None).unwrap();
        snapshot.validate().unwrap();
        assert_eq!(snapshot.values.len(), if projected { 8 } else { 5 });
        assert!(!snapshot.values.iter().any(|value| matches!(
            value.role,
            Role::EncoderOutput | Role::Prepared(PreparedRole::Ingress(IngressRole::Pixels))
        )));
        assert_eq!(
            snapshot
                .values
                .iter()
                .filter(|value| matches!(
                    value.role,
                    Role::Prepared(PreparedRole::Ingress(IngressRole::Projected(0)))
                ))
                .count(),
            usize::from(projected)
        );
        assert!(snapshot.storage.values.iter().all(Option::is_none));
    }
}
