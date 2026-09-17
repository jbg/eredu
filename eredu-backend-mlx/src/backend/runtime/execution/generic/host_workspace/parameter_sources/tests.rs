use super::*;

#[test]
fn fixed_parameter_dtype_keeps_representation_separate_from_physical_storage() {
    for native in [
        safemlx::Dtype::Float16,
        safemlx::Dtype::Bfloat16,
        safemlx::Dtype::Float32,
    ] {
        assert_eq!(dtype(native, 7, 11), Ok(WorkspaceDtype::Float32));
        let unknown = representation(native, None).unwrap();
        assert!(!unknown.row_contiguous());
        let copied = representation(
            native,
            Some(safemlx::PreparedHostTransferPlan::COPY_OUTPUT_LAYOUT),
        ).unwrap();
        assert!(copied.row_contiguous());
        assert_eq!(copied.dtype(), unknown.dtype());
    }
    assert_eq!(
        dtype(safemlx::Dtype::Int32, 7, 11),
        Ok(WorkspaceDtype::Int32)
    );
    assert_eq!(
        dtype(safemlx::Dtype::Uint32, 7, 11),
        Ok(WorkspaceDtype::Uint32)
    );
    assert_eq!(
        dtype(safemlx::Dtype::Uint8, 7, 11),
        Ok(WorkspaceDtype::Uint8)
    );
    assert_eq!(dtype(safemlx::Dtype::Bool, 7, 11), Ok(WorkspaceDtype::Bool));
    assert_eq!(
        dtype(safemlx::Dtype::Float64, 7, 11),
        Err(Failure::UnsupportedDtype { unit: 7, row: 11 })
    );
}
