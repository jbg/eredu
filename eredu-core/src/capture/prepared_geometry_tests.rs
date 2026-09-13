use super::*;

#[test]
fn prefill_capture_compares_assembled_decoder_geometry() {
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 4,
        max_predictions: 3,
    };
    request.validate_prefill(1, 4).unwrap();
    // Neither the text-only prefix nor the unmerged image patch count is the
    // actual four-position decoder input admitted for this request.
    for shape in [[1, 2], [1, 8], [2, 4], [0, 4], [1, 0]] {
        assert!(matches!(
            request.validate_prefill(shape[0], shape[1]),
            Err(CaptureError::Invalid(_))
        ));
    }
    CaptureRequestShape {
        batch: 2,
        ..request
    }
    .validate_prefill(2, 4)
    .unwrap();
    assert!(CaptureRequestShape {
        prompt_tokens: 0,
        ..request
    }
    .validate_prefill(1, 0)
    .is_err());
}
