use super::*;
use crate::backend::runtime::execution::generic::gguf_host_typed::TypedDestination;

fn compare<T: Copy + std::fmt::Debug, const N: usize>(
    bytes: Vec<u8>,
    decode: impl Fn([u8; N]) -> T + Copy,
    bits: impl Fn(T) -> u64,
) {
    let ordinary = decode_native(bytes.clone(), decode).unwrap();
    let chunks = native_chunks::<N>(&bytes).unwrap();
    let mut destination = TypedDestination::try_new(chunks.len()).unwrap();
    let pointer = destination.values().as_ptr();
    let capacity = destination.values().capacity();
    destination
        .fill(chunks.len(), chunks.iter().map(|v| decode(*v)))
        .unwrap();
    assert_eq!(
        (
            destination.values().as_ptr(),
            destination.values().capacity()
        ),
        (pointer, capacity)
    );
    let actual = destination.into_values();
    assert_eq!(actual.len(), ordinary.len());
    assert_eq!(
        actual
            .as_slice()
            .iter()
            .copied()
            .map(&bits)
            .collect::<Vec<_>>(),
        ordinary.iter().copied().map(bits).collect::<Vec<_>>()
    );
}
#[test]
fn prepared_dense_decoders_preserve_all_scalar_bits_without_collect_capacity_assumptions() {
    let words32 = [0u32, 0x8000_0000, 0x7fc0_0021, 0x3fc0_0000, 0xffff_ffff];
    let bytes32: Vec<_> = words32.into_iter().flat_map(u32::to_ne_bytes).collect();
    compare(bytes32.clone(), f32::from_ne_bytes, |v| {
        u64::from(v.to_bits())
    });
    compare(bytes32, i32::from_ne_bytes, |v| v as u64);
    let words16 = [0u16, 0x8000, 0x7e21, 0x3e00, 0xffff];
    let bytes16: Vec<_> = words16.into_iter().flat_map(u16::to_ne_bytes).collect();
    compare(
        bytes16.clone(),
        |v| half::f16::from_bits(u16::from_ne_bytes(v)),
        |v| u64::from(v.to_bits()),
    );
    compare(
        bytes16.clone(),
        |v| half::bf16::from_bits(u16::from_ne_bytes(v)),
        |v| u64::from(v.to_bits()),
    );
    compare(bytes16, i16::from_ne_bytes, |v| v as u64);
    let words64 = [0u64, 1 << 63, 0x7ff8_0000_0000_0021, u64::MAX];
    let bytes64: Vec<_> = words64.into_iter().flat_map(u64::to_ne_bytes).collect();
    compare(bytes64.clone(), f64::from_ne_bytes, f64::to_bits);
    compare(bytes64, i64::from_ne_bytes, |v| v as u64);
    let bytes = vec![0u8, 1, 127, 128, 255];
    let ordinary: Vec<i8> = bytes.clone().into_iter().map(|v| v as i8).collect();
    let mut destination = TypedDestination::try_new(bytes.len()).unwrap();
    destination
        .fill(
            bytes.len(),
            native_chunks::<1>(&bytes)
                .unwrap()
                .iter()
                .map(|v| i8::from_ne_bytes(*v)),
        )
        .unwrap();
    assert_eq!(destination.into_values().as_slice(), ordinary.as_slice());
}
#[test]
fn malformed_dense_width_keeps_original_error_and_affine_bits_keep_separate_destination() {
    let bytes = vec![0x31; 17];
    let ordinary = decode_native(bytes.clone(), f32::from_ne_bytes).unwrap_err();
    let prepared = native_chunks::<4>(&bytes).unwrap_err();
    assert_eq!(ordinary.to_string(), prepared.to_string());
    assert!(matches!(prepared, IoError::InvalidFormat(_)));
    let bits = vec![0, 0x8000, 0x7e01, 0x3c00, 0xffff];
    let original_pointer = bits.as_ptr();
    let mut destination = TypedDestination::try_new(bits.len()).unwrap();
    let capacity = destination.values().capacity();
    let pointer = destination.values().as_ptr();
    destination
        .fill(bits.len(), bits.iter().copied().map(half::f16::from_bits))
        .unwrap();
    assert_ne!(original_pointer.cast::<u8>(), pointer.cast::<u8>());
    assert_eq!(destination.values().capacity(), capacity);
    assert_eq!(
        destination
            .values()
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
    drop(bits);
    assert_eq!(destination.into_values().as_ptr(), pointer);
}
