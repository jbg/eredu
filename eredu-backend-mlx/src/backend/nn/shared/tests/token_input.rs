use super::*;
use eredu_runtime::{OriginalTokenIds, TokenInputError, TokenVisibility};

#[test]
fn mlx_original_token_input_preserves_exact_ids_and_rejects_unsigned_overflow() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let context = execution.stream();
    let ids = [0_i32, 16_777_217, i32::MAX - 1];
    let signed = MlxTensor::from_array(Array::from_slice(&ids, &[1, 3]));
    let unsigned = MlxTensor::from_array(Array::from_slice(&ids.map(|v| v as u32), &[1, 3]));
    for value in [&signed, &unsigned] {
        assert_eq!(
            OriginalTokenIds::Tensor(value)
                .resolve(1, 3, i32::MAX, 3, context)
                .unwrap()
                .as_ref(),
            ids.map(|v| v as u64)
        );
    }
    let overflow = MlxTensor::from_array(Array::from_slice(&[u32::MAX], &[1, 1]));
    assert!(matches!(
        OriginalTokenIds::Tensor(&overflow).resolve(1, 1, i32::MAX, 1, context),
        Err(TokenInputError::Vocabulary)
    ));
    let floating = MlxTensor::from_array(Array::from_slice(&[3_f32], &[1, 1]));
    assert!(matches!(
        OriginalTokenIds::Tensor(&floating).resolve(1, 1, 32, 1, context),
        Err(TokenInputError::ScalarType)
    ));
    let padding = MlxTensor::from_array(Array::from_slice(&[true, false, true], &[1, 3]));
    assert_eq!(
        TokenVisibility::Tensor(&padding)
            .resolve(1, 3, 3, context)
            .unwrap()
            .as_ref(),
        [true, false, true]
    );
    let bad = MlxTensor::from_array(Array::from_slice(&[0_f32, 0.5], &[1, 2]));
    assert!(matches!(
        TokenVisibility::Tensor(&bad).resolve(1, 2, 2, context),
        Err(TokenInputError::Visibility)
    ));
}
