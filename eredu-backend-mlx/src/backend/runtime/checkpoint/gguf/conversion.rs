//! One native conversion dispatch over ordinary or supplied owners.
use super::*;
use eredu_gguf::{ConvertedParts, TensorDescriptorView};

pub(in crate::backend::runtime::checkpoint) trait Group {
    type Input;
    type Descriptor;
    type Names;
    type CheckedNames<const N: usize>;
    type Shape: AsRef<[u64]>;
    type Bytes;
    type Bits;
    type Words;
    type Output;
    fn split(
        input: Self::Input,
    ) -> (
        Self::Descriptor,
        Self::Names,
        ConvertedParts<Self::Shape, Self::Bytes, Self::Bits, Self::Words>,
    );
    fn descriptor(value: &Self::Descriptor) -> TensorDescriptorView<'_>;
    fn names<const N: usize>(
        physical: &str,
        names: Self::Names,
    ) -> Result<Self::CheckedNames<N>, IoError>;
    fn dense(
        descriptor: Self::Descriptor,
        names: Self::CheckedNames<1>,
        array: Array,
    ) -> Self::Output;
    fn iquant(
        descriptor: Self::Descriptor,
        names: Self::CheckedNames<1>,
        ggml_type: GgmlType,
        endian: Endian,
        logical_shape: Vec<i32>,
        array: Array,
    ) -> Self::Output;
    fn affine(
        descriptor: Self::Descriptor,
        names: Self::CheckedNames<3>,
        bits: u8,
        group_size: u32,
        arrays: [Array; 3],
    ) -> Self::Output;
    fn mxfp4(
        descriptor: Self::Descriptor,
        names: Self::CheckedNames<2>,
        arrays: [Array; 2],
    ) -> Self::Output;
}
pub(in crate::backend::runtime::checkpoint) trait Realization<G: Group> {
    type Error: From<IoError>;
    fn bind_group(&mut self, descriptor: &G::Descriptor);
    fn f32_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn f16_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn bf16_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i8_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i16_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i32_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn i64_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn f64_bytes(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn u8(&mut self, values: G::Bytes, shape: &[i32]) -> Result<Array, Self::Error>;
    fn u32(&mut self, values: G::Words, shape: &[i32]) -> Result<Array, Self::Error>;
    fn f16_bits(&mut self, values: G::Bits, shape: &[i32]) -> Result<Array, Self::Error>;
}
pub(in crate::backend::runtime::checkpoint) struct OrdinaryGroup;
impl Group for OrdinaryGroup {
    type Input = eredu_gguf::ConvertedCheckpointTensor;
    type Descriptor = eredu_gguf::TensorDescriptor;
    type Names = Vec<String>;
    type CheckedNames<const N: usize> = [String; N];
    type Shape = Vec<u64>;
    type Bytes = Vec<u8>;
    type Bits = Vec<u16>;
    type Words = Vec<u32>;
    type Output = GgufTensor;
    fn split(
        input: Self::Input,
    ) -> (
        Self::Descriptor,
        Self::Names,
        ConvertedParts<Self::Shape, Self::Bytes, Self::Bits, Self::Words>,
    ) {
        let (d, n, c) = input.into_parts();
        (d, n, c.into_storage_parts())
    }
    fn descriptor(d: &Self::Descriptor) -> TensorDescriptorView<'_> {
        d.view()
    }
    fn names<const N: usize>(
        physical: &str,
        names: Self::Names,
    ) -> Result<Self::CheckedNames<N>, IoError> {
        converted_output_names(physical, names)
    }
    fn dense(_: Self::Descriptor, [name]: [String; 1], array: Array) -> GgufTensor {
        GgufTensor::Dense(GgufArray { name, array })
    }
    fn iquant(
        descriptor: Self::Descriptor,
        [name]: [String; 1],
        ggml_type: GgmlType,
        endian: Endian,
        logical_shape: Vec<i32>,
        array: Array,
    ) -> GgufTensor {
        GgufTensor::IQuant(GgufIQuantTensor {
            physical_name: descriptor.name.clone(),
            ggml_type,
            endian,
            logical_shape,
            packed: GgufArray { name, array },
        })
    }
    fn affine(
        descriptor: Self::Descriptor,
        [weight_name, scales_name, biases_name]: [String; 3],
        bits: u8,
        group_size: u32,
        [weight, scales, biases]: [Array; 3],
    ) -> GgufTensor {
        GgufTensor::Affine(GgufAffineTensor {
            physical_name: descriptor.name,
            bits,
            group_size,
            weight: GgufArray {
                name: weight_name,
                array: weight,
            },
            scales: GgufArray {
                name: scales_name,
                array: scales,
            },
            biases: GgufArray {
                name: biases_name,
                array: biases,
            },
        })
    }
    fn mxfp4(
        descriptor: Self::Descriptor,
        [weight_name, scales_name]: [String; 2],
        [weight, scales]: [Array; 2],
    ) -> GgufTensor {
        GgufTensor::MxFp4(GgufMxFp4Tensor {
            physical_name: descriptor.name,
            weight: GgufArray {
                name: weight_name,
                array: weight,
            },
            scales: GgufArray {
                name: scales_name,
                array: scales,
            },
        })
    }
}
impl<P: producer::ArrayProducer> Realization<OrdinaryGroup> for P {
    type Error = P::Error;
    fn bind_group(&mut self, descriptor: &eredu_gguf::TensorDescriptor) {
        producer::ArrayProducer::bind_group(self, descriptor)
    }
    fn f32_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::f32_bytes(self, values, shape)
    }
    fn f16_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::f16_bytes(self, values, shape)
    }
    fn bf16_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::bf16_bytes(self, values, shape)
    }
    fn i8_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::i8_bytes(self, values, shape)
    }
    fn i16_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::i16_bytes(self, values, shape)
    }
    fn i32_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::i32_bytes(self, values, shape)
    }
    fn i64_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::i64_bytes(self, values, shape)
    }
    fn f64_bytes(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::f64_bytes(self, values, shape)
    }
    fn u8(&mut self, values: Vec<u8>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::u8(self, values, shape)
    }
    fn u32(&mut self, values: Vec<u32>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::u32(self, values, shape)
    }
    fn f16_bits(&mut self, values: Vec<u16>, shape: &[i32]) -> Result<Array, Self::Error> {
        producer::ArrayProducer::f16_bits(self, values, shape)
    }
}
pub(in crate::backend::runtime::checkpoint) fn convert<G: Group, P: Realization<G>>(
    input: G::Input,
    producer: &mut P,
) -> Result<G::Output, P::Error> {
    let (descriptor, names, converted) = G::split(input);
    producer.bind_group(&descriptor);
    let physical = G::descriptor(&descriptor).name;
    match converted {
        ConvertedParts::Dense { shape, dtype, data } => {
            let names = G::names::<1>(physical, names)?;
            let shape = mlx_shape_i32(physical, shape.as_ref())?;
            let array = match dtype {
                eredu_gguf::DenseDtype::F32 => producer.f32_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::F16 => producer.f16_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::Bf16 => producer.bf16_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::I8 => producer.i8_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::I16 => producer.i16_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::I32 => producer.i32_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::I64 => producer.i64_bytes(data, &shape)?,
                eredu_gguf::DenseDtype::F64 => producer.f64_bytes(data, &shape)?,
            };
            Ok(G::dense(descriptor, names, array))
        }
        ConvertedParts::IQuant {
            shape,
            ggml_type,
            endian,
            data,
        } => {
            let names = G::names::<1>(physical, names)?;
            let packed_shape = mlx_shape_i32(
                physical,
                &eredu_gguf::packed_iquant_shape(shape.as_ref(), ggml_type).map_err(gguf_error)?,
            )?;
            let logical_shape = mlx_shape_i32(physical, shape.as_ref())?;
            let array = producer.u8(data, &packed_shape)?;
            Ok(G::iquant(
                descriptor,
                names,
                ggml_type,
                endian,
                logical_shape,
                array,
            ))
        }
        ConvertedParts::Affine {
            weight_shape,
            scale_shape,
            bits,
            group_size,
            weights,
            scales,
            biases,
        } => {
            let names = G::names::<3>(physical, names)?;
            let weight_shape = mlx_shape_i32(physical, weight_shape.as_ref())?;
            let scale_shape = mlx_shape_i32(physical, scale_shape.as_ref())?;
            let weight = producer.u32(weights, &weight_shape)?;
            let scales = producer.f16_bits(scales, &scale_shape)?;
            let biases = producer.f16_bits(biases, &scale_shape)?;
            Ok(G::affine(
                descriptor,
                names,
                bits,
                group_size,
                [weight, scales, biases],
            ))
        }
        ConvertedParts::MxFp4 {
            weight_shape,
            scale_shape,
            weights,
            scales,
        } => {
            let names = G::names::<2>(physical, names)?;
            let weight_shape = mlx_shape_i32(physical, weight_shape.as_ref())?;
            let scale_shape = mlx_shape_i32(physical, scale_shape.as_ref())?;
            let weight = producer.u32(weights, &weight_shape)?;
            let scales = producer.u8(scales, &scale_shape)?;
            Ok(G::mxfp4(descriptor, names, [weight, scales]))
        }
    }
}
