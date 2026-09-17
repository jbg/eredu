use eredu_gguf::{Error, GgmlType, UnsupportedGgmlType};

#[test]
fn fixed_block_refusals_preserve_removed_unknown_and_explicit_unknown_codes() {
    fn copy_cause<T: Copy>(cause: T) -> (T, T) {
        (cause, cause)
    }
    for encoding in [
        GgmlType::RemovedIQ4NL4_4,
        GgmlType::RemovedIQ4NL4_8,
        GgmlType::RemovedIQ4NL8_8,
        GgmlType::from_code(4),
        GgmlType::Unknown(0),
        GgmlType::Unknown(u32::MAX),
    ] {
        let (cause, retained): (UnsupportedGgmlType, _) =
            copy_cause(encoding.block_and_bytes_fixed().unwrap_err());
        assert_eq!(cause, retained);
        assert_eq!(cause.code(), encoding.code());
        assert!(matches!(
            encoding.block_and_bytes(),
            Err(Error::UnsupportedTensorType(code)) if code == cause.code()
        ));
        assert_eq!(
            cause.to_string(),
            format!("unsupported GGML tensor type {}", encoding.code())
        );
    }
    // Explicit Unknown(0) stays distinct from the recognized code-zero format.
    assert_eq!(GgmlType::from_code(0).block_and_bytes_fixed(), Ok((1, 4)));
    assert_eq!(GgmlType::Q6K.block_and_bytes_fixed(), Ok((256, 210)));
    assert_eq!(GgmlType::MxFp4.block_and_bytes_fixed(), Ok((32, 17)));
}
