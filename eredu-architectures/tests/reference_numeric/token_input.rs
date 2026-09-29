//! Neutral exact-token ingress exercised with the numerical backend.
use super::*;
use eredu_core::checkpoint::TensorDtype;
use eredu_runtime::{OriginalTokenIds, TokenInputError, TokenVisibility};
use std::borrow::Cow;

#[test]
fn original_token_input_preserves_large_integers_and_borrowed_host_ids() {
    let ctx = NumericContext::default();
    let ids = [0, 16_777_217, i32::MAX - 1];
    let host = ids.map(|id| id as u64);
    let borrowed = OriginalTokenIds::<NumericTensor>::Host(&host)
        .resolve(1, 3, i32::MAX, 3, &ctx)
        .unwrap();
    assert!(matches!(borrowed, Cow::Borrowed(_)));
    assert_eq!(borrowed.as_ref(), host);
    let tensor = NumericTensor::from_i32_slice(&ids, &[1, 3], &ctx).unwrap();
    assert_ne!(tensor.data[1] as i32, ids[1]); // The floating proxy loses this ID.
    for dtype in [TensorDtype::I32, TensorDtype::U32] {
        let tensor = tensor.clone().with_dtype(dtype);
        let extracted = OriginalTokenIds::Tensor(&tensor)
            .resolve(1, 3, i32::MAX, 3, &ctx)
            .unwrap();
        assert!(matches!(extracted, Cow::Owned(_)));
        assert_eq!(extracted.as_ref(), host);
    }
}

#[test]
fn original_token_input_rejects_ambiguous_ids_and_request_overruns() {
    let ctx = NumericContext::default();
    let tensor = NumericTensor::from_i32_slice(&[3, 4], &[1, 2], &ctx).unwrap();
    for dtype in [
        TensorDtype::F32,
        TensorDtype::F16,
        TensorDtype::Bf16,
        TensorDtype::Bool,
        TensorDtype::I64,
    ] {
        let ambiguous = tensor.clone().with_dtype(dtype);
        assert!(matches!(
            OriginalTokenIds::Tensor(&ambiguous).resolve(1, 2, 32, 2, &ctx),
            Err(TokenInputError::ScalarType)
        ));
    }
    for (batch, tokens, bound) in [
        (2, 1, 2),
        (1, 2, 1),
        (0, 2, 2),
        (-1, 2, 2),
        (i32::MAX, i32::MAX, 2),
    ] {
        assert!(matches!(
            OriginalTokenIds::Tensor(&tensor).resolve(batch, tokens, 32, bound, &ctx),
            Err(TokenInputError::Geometry)
        ));
    }
    for ids in [[-1, 3], [3, 32]] {
        let tensor = NumericTensor::from_i32_slice(&ids, &[1, 2], &ctx).unwrap();
        assert!(matches!(
            OriginalTokenIds::Tensor(&tensor).resolve(1, 2, 32, 2, &ctx),
            Err(TokenInputError::Vocabulary)
        ));
    }
    for ids in [[0, u64::MAX], [0, 32]] {
        assert!(matches!(
            OriginalTokenIds::<NumericTensor>::Host(&ids).resolve(1, 2, 32, 2, &ctx),
            Err(TokenInputError::Vocabulary)
        ));
    }
    assert!(matches!(
        OriginalTokenIds::<NumericTensor>::Host(&[3]).resolve(1, 2, 32, 2, &ctx),
        Err(TokenInputError::Geometry)
    ));
    assert!(matches!(
        OriginalTokenIds::Tensor(&tensor).resolve(1, 2, 0, 2, &ctx),
        Err(TokenInputError::Vocabulary)
    ));
}

#[test]
fn token_visibility_accepts_only_bounded_exact_zero_one_values() {
    let ctx = NumericContext::default();
    let host = [true, false, true];
    let borrowed = TokenVisibility::<NumericTensor>::Host(&host)
        .resolve(1, 3, 3, &ctx)
        .unwrap();
    assert!(matches!(borrowed, Cow::Borrowed(_)));
    for dtype in [
        TensorDtype::F32,
        TensorDtype::F16,
        TensorDtype::Bf16,
        TensorDtype::I32,
        TensorDtype::U32,
        TensorDtype::Bool,
    ] {
        let tensor = NumericTensor::new([1, 3], vec![1., 0., 1.]).with_dtype(dtype);
        assert_eq!(
            TokenVisibility::Tensor(&tensor)
                .resolve(1, 3, 3, &ctx)
                .unwrap()
                .as_ref(),
            host
        );
    }
    for invalid in [-1., 0.5, 2., f32::NAN, f32::INFINITY] {
        let tensor = NumericTensor::new([1, 3], vec![1., invalid, 0.]);
        assert!(matches!(
            TokenVisibility::Tensor(&tensor).resolve(1, 3, 3, &ctx),
            Err(TokenInputError::Visibility)
        ));
    }
    let tensor = NumericTensor::new([1, 3], vec![1., 0., 1.]);
    for (batch, tokens, bound) in [(3, 1, 3), (1, 3, 2), (0, 3, 3)] {
        assert!(matches!(
            TokenVisibility::Tensor(&tensor).resolve(batch, tokens, bound, &ctx),
            Err(TokenInputError::Geometry)
        ));
    }
    let wide = tensor.with_dtype(TensorDtype::F64);
    assert!(matches!(
        TokenVisibility::Tensor(&wide).resolve(1, 3, 3, &ctx),
        Err(TokenInputError::Visibility)
    ));
    assert!(matches!(
        TokenVisibility::<NumericTensor>::Host(&[true]).resolve(1, 3, 3, &ctx),
        Err(TokenInputError::Geometry)
    ));
}
