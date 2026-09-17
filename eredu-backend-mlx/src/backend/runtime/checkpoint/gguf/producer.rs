//! One converter, with explicit ordinary or original array realization.
use super::*;

// These are the actual typed outputs of the portable GGUF converter. No dynamic
// downcast or unconstrained caller type enters an original failure owner.
pub(in crate::backend::runtime::checkpoint) trait ArrayProducer {
    type Error: From<IoError>;
    // Original metadata binding may be recorded here, but its refusal is delayed
    // until a reached producer/transform, after existing names/shape/width checks.
    fn bind_group(&mut self, _descriptor: &eredu_gguf::TensorDescriptor) {}

    fn f32_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.f32(decode_native(bytes, f32::from_ne_bytes)?, shape)
    }
    fn f16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.f16(
            decode_native(bytes, |bytes| {
                half::f16::from_bits(u16::from_ne_bytes(bytes))
            })?,
            shape,
        )
    }
    fn bf16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.bf16(
            decode_native(bytes, |bytes| {
                half::bf16::from_bits(u16::from_ne_bytes(bytes))
            })?,
            shape,
        )
    }
    fn i16_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.i16(decode_native(bytes, i16::from_ne_bytes)?, shape)
    }
    fn i32_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.i32(decode_native(bytes, i32::from_ne_bytes)?, shape)
    }
    fn i64_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.i64(decode_native(bytes, i64::from_ne_bytes)?, shape)
    }
    fn f64_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.f64(decode_native(bytes, f64::from_ne_bytes)?, shape)
    }
    fn i8_bytes(&mut self, bytes: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.i8(bytes.into_iter().map(|value| value as i8).collect(), shape)
    }
    fn f16_bits(&mut self, values: Vec<u16>, shape: &[i32]) -> Result<Array, Self::Error> {
        self.f16(
            values.into_iter().map(half::f16::from_bits).collect(),
            shape,
        )
    }
    fn f32(&mut self, values: Vec<f32>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn f16(&mut self, values: Vec<half::f16>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn bf16(&mut self, values: Vec<half::bf16>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i8(&mut self, values: Vec<i8>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i16(&mut self, values: Vec<i16>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i32(&mut self, values: Vec<i32>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i64(&mut self, values: Vec<i64>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn f64(&mut self, values: Vec<f64>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn u8(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error>;
    fn u32(&mut self, values: Vec<u32>, shape: &[i32]) -> Result<Array, Self::Error>;
}

pub(super) struct Ordinary {
    pub(super) host_owned: bool,
}
impl ArrayProducer for Ordinary {
    type Error = IoError;
    fn f32(&mut self, values: Vec<f32>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn f16(&mut self, values: Vec<half::f16>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn bf16(&mut self, values: Vec<half::bf16>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn i8(&mut self, values: Vec<i8>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn i16(&mut self, values: Vec<i16>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn i32(&mut self, values: Vec<i32>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn i64(&mut self, values: Vec<i64>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn f64(&mut self, values: Vec<f64>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn u8(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
    fn u32(&mut self, values: Vec<u32>, shape: &[i32]) -> Result<Array, Self::Error> {
        array_from_owned_data(values, shape, self.host_owned)
    }
}
