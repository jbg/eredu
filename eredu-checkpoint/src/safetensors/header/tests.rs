use super::*;
use safetensors::tensor::{Dtype, TensorView};

#[test]
fn borrowed_header_matches_safe_tensors_mixed_dtype_unicode_offsets_and_refusals() {
    let wide = [1u8, 7, 3, 9, 4, 11];
    let narrow = [13u8, 17, 19];
    let expected = safetensors::tensor::serialize(
        [
            (
                "z\"雪",
                TensorView::new(Dtype::U8, vec![3], &narrow).unwrap(),
            ),
            ("a", TensorView::new(Dtype::I16, vec![1, 3], &wide).unwrap()),
        ],
        None,
    )
    .unwrap();
    let entries = [
        (
            "a",
            TensorInfo {
                dtype: Dtype::I16,
                shape: vec![1, 3],
                data_offsets: (0, 6),
            },
        ),
        (
            "z\"雪",
            TensorInfo {
                dtype: Dtype::U8,
                shape: vec![3],
                data_offsets: (6, 9),
            },
        ),
    ];
    let plan = SafetensorsHeaderPlan::prepare(&entries).unwrap();
    assert_eq!(plan.file_bytes(), expected.len());
    let mut output = vec![0; plan.header_bytes()];
    plan.write_header(&mut output).unwrap();
    assert_eq!(output, expected[..output.len()]);
    assert!(matches!(
        plan.write_header(&mut []),
        Err(SafetensorsHeaderError::Destination)
    ));
    let mut invalid = entries.clone();
    invalid[1].1.data_offsets = (7, 10);
    assert!(matches!(
        SafetensorsHeaderPlan::prepare(&invalid),
        Err(SafetensorsHeaderError::Offset)
    ));
    invalid = entries.clone();
    invalid[1].0 = "a";
    assert!(matches!(
        SafetensorsHeaderPlan::prepare(&invalid),
        Err(SafetensorsHeaderError::Name)
    ));
    invalid = entries.clone();
    invalid.swap(0, 1);
    assert!(SafetensorsHeaderPlan::prepare(&invalid).is_err());
    let same_dtype = [
        (
            "z",
            TensorInfo {
                dtype: Dtype::U8,
                shape: vec![1],
                data_offsets: (0, 1),
            },
        ),
        (
            "a",
            TensorInfo {
                dtype: Dtype::U8,
                shape: vec![1],
                data_offsets: (1, 2),
            },
        ),
    ];
    assert!(matches!(
        SafetensorsHeaderPlan::prepare(&same_dtype),
        Err(SafetensorsHeaderError::Order)
    ));
    SafetensorsHeaderPlan::control_bytes().unwrap();
}
