//! Admission without a Tensor implementation, a native context or a retained source.
use super::*;

struct HostValue {
    identity: eredu_core::InputTensorIdentity,
    integers: Vec<i32>,
    copies: std::rc::Rc<std::cell::Cell<usize>>,
}
impl Clone for HostValue {
    fn clone(&self) -> Self {
        self.copies.set(self.copies.get() + 1);
        Self {
            identity: self.identity.clone(),
            integers: self.integers.clone(),
            copies: self.copies.clone(),
        }
    }
}
fn host(dtype: TensorDtype, shape: &[usize], integers: &[i32]) -> HostValue {
    HostValue {
        identity: eredu_core::InputTensorIdentity::new(dtype, shape.to_vec()).unwrap(),
        integers: integers.to_vec(),
        copies: Default::default(),
    }
}

#[test]
fn composite_media_proof_clones_request_handles_once_and_shares_them_across_parts() {
    let (_dir, _target, ingress) = setup();
    let inspect = HostInspector(0.into());
    let source = request(vec![
        host_text(&[3, 14]),
        PreparedInputPart::new(
            M::Image,
            P::Tensor(host(TensorDtype::F32, &[16, 24], &[])),
            [(K::PatchGrid, host(TensorDtype::I32, &[1, 3], &[1, 4, 4]))],
        )
        .unwrap(),
        host_text(&[15, 6]),
    ]);
    let copies: Vec<_> = source
        .parts()
        .iter()
        .flat_map(|part| {
            let value = match part.payload() {
                P::Tensor(t) | P::TokenIds(t) | P::Embeddings(t) => t,
                _ => unreachable!(),
            };
            std::iter::once(value.copies.clone())
                .chain(part.metadata().values().map(|v| v.copies.clone()))
        })
        .collect();
    let proof = ingress
        .admission_config()
        .admit(&source, &inspect)
        .unwrap()
        .into_composite();
    assert_eq!(copies.len(), 4);
    assert!(copies.iter().all(|n| n.get() == 1));
    let retained = proof.clone();
    drop(proof);
    drop(source);
    let plans: Vec<eredu_architectures::media_plan::PreparedInputPartPlan> =
        retained.parts().iter().cloned().map(Into::into).collect();
    assert_eq!(plans.len(), 3);
    assert_eq!(retained.decoder_positions(), 8);
    assert!(
        copies.iter().all(|n| n.get() == 1),
        "part-plan and admission clones must only share the request"
    );
}
struct HostInspector(std::cell::Cell<usize>);
impl PreparedInputInspector<HostValue> for HostInspector {
    fn identity(
        &self,
        value: &HostValue,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        Ok(value.identity.clone())
    }
    fn i32_values(&self, value: &HostValue) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        self.0.set(self.0.get() + 1);
        Ok(value.integers.clone())
    }
    fn bool_values(&self, _: &HostValue) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        panic!("this policy has no boolean metadata")
    }
}
fn request(parts: Vec<PreparedInputPart<HostValue>>) -> PreparedModelInput<HostValue> {
    PreparedModelInput::new(parts, |value| Ok(value.identity.clone())).unwrap()
}
fn host_text(ids: &[i32]) -> PreparedInputPart<HostValue> {
    PreparedInputPart::new(
        M::Text,
        P::TokenIds(host(TensorDtype::I32, &[1, ids.len()], ids)),
        [],
    )
    .unwrap()
}

#[test]
fn media_admission_is_source_free_and_preserves_exact_host_positions() {
    let (dir, target, ingress) = setup();
    let policy = ingress.admission_config().clone();
    drop(ingress);
    drop(target);
    dir.close().unwrap();
    let inspect = HostInspector(0.into());
    // Raw pixels are never read during admission; the inspector sees only the grid and IDs.
    let source = request(vec![
        host_text(&[3, 14]),
        PreparedInputPart::new(
            M::Image,
            P::Tensor(host(TensorDtype::F32, &[16, 24], &[])),
            [(K::PatchGrid, host(TensorDtype::I32, &[1, 3], &[1, 4, 4]))],
        )
        .unwrap(),
        host_text(&[15, 6]),
    ]);
    let proof = policy.admit(&source, &inspect).unwrap();
    assert_eq!(proof.admission().decoder_shape(), [1, 8]);
    assert_eq!(proof.token_ids(), [3, 14, 12, 12, 12, 12, 15, 6]);
    assert_eq!(
        proof.position_ids(),
        &[
            vec![0, 1, 2, 2, 2, 2, 4, 5],
            vec![0, 1, 2, 2, 3, 3, 4, 5],
            vec![0, 1, 2, 3, 2, 3, 4, 5],
        ]
    );
    assert_eq!(proof.rotary_delta(), -2);
    assert_eq!(inspect.0.get(), 3);

    // Directly supplied embeddings keep explicit original IDs and the same geometry.
    let projected = request(vec![
        host_text(&[3, 14]),
        PreparedInputPart::new(
            M::Image,
            P::Embeddings(host(TensorDtype::F32, &[1, 4, 32], &[])),
            [
                (
                    K::OriginalTokenIds,
                    host(TensorDtype::I32, &[1, 4], &[12; 4]),
                ),
                (K::PatchGrid, host(TensorDtype::I32, &[1, 3], &[1, 4, 4])),
            ],
        )
        .unwrap(),
        host_text(&[15, 6]),
    ]);
    let projected = policy.admit(&projected, &inspect).unwrap();

    assert_eq!(projected.token_ids(), proof.token_ids());
    assert_eq!(projected.position_ids(), proof.position_ids());
    assert_eq!(projected.rotary_delta(), proof.rotary_delta());
    assert_eq!(
        projected.rotary_invocation().unwrap(),
        proof.rotary_invocation().unwrap()
    );
    let rotary = proof.rotary_invocation().unwrap().logical_values().unwrap();
    for name in ["cosine", "sine"] {
        let value = rotary.iter().find(|v| v.name == name).unwrap();
        assert_eq!(value.shape, [8, 6]);
        assert_eq!(value.element, eredu_nn::TensorElementType::F32);
    }
}

#[test]
fn media_admission_rejects_overbounds_before_inspection_and_invalid_ids_before_allocation() {
    let (_dir, _target, ingress) = setup();
    let inspect = HostInspector(0.into());
    let policy = ingress.admission_config();
    let excessive = policy.maximum_request_tokens() + 1;
    for source in [
        request(vec![host_text(&vec![3; excessive])]),
        request((0..excessive).map(|_| host_text(&[3])).collect()),
        request(vec![PreparedInputPart::new(
            M::Image,
            P::Tensor(host(TensorDtype::F32, &[16, 24], &[])),
            [(K::PatchGrid, host(TensorDtype::I32, &[excessive, 3], &[]))],
        )
        .unwrap()]),
    ] {
        assert!(matches!(
            policy.admit(&source, &inspect),
            Err(MediaInputError::Geometry(_))
        ));
        assert_eq!(inspect.0.get(), 0);
    }
    for ids in [[-1, 3], [i32::MAX, 3], [16, 3]] {
        let source = request(vec![host_text(&ids)]);
        assert!(matches!(
            policy.admit(&source, &inspect),
            Err(MediaInputError::Tokens(
                eredu_runtime::TokenInputError::Vocabulary
            ))
        ));
    }
    let malformed = request(vec![PreparedInputPart::new(
        M::Text,
        P::TokenIds(host(TensorDtype::I32, &[1, 3], &[3, 4])),
        [],
    )
    .unwrap()]);
    assert!(matches!(
        policy.admit(&malformed, &inspect),
        Err(MediaInputError::Tokens(
            eredu_runtime::TokenInputError::Geometry
        ))
    ));
}
