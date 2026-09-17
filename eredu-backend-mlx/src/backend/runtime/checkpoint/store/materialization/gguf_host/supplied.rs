//! Supplied owning group flavor of the same native conversion driver.
use super::*;
use crate::backend::runtime::execution::generic::gguf_host_typed::supplied::AdmittedFamily;
use eredu_gguf::{
    ConvertedParts, StoredBuffer, StoredCheckpointTensor, StoredDescriptor, StoredOutputNames,
    TensorDescriptorView,
};
use gguf::conversion::{Group, Realization};

pub(super) struct StoredGroup;
impl Group for StoredGroup {
    type Input = StoredCheckpointTensor<AdmittedFamily>;
    type Descriptor = StoredDescriptor<AdmittedFamily>;
    type Names = StoredOutputNames<AdmittedFamily>;
    type CheckedNames<const N: usize> = StoredOutputNames<AdmittedFamily>;
    type Shape = StoredBuffer<AdmittedFamily, u64>;
    type Bytes = StoredBuffer<AdmittedFamily, u8>;
    type Bits = StoredBuffer<AdmittedFamily, u16>;
    type Words = StoredBuffer<AdmittedFamily, u32>;
    type Output = CachedGgufArrays;
    fn split(
        input: Self::Input,
    ) -> (
        Self::Descriptor,
        Self::Names,
        ConvertedParts<Self::Shape, Self::Bytes, Self::Bits, Self::Words>,
    ) {
        (
            input.descriptor,
            input.output_names,
            input.converted.into_storage_parts(),
        )
    }
    fn descriptor(d: &Self::Descriptor) -> TensorDescriptorView<'_> {
        d.view()
    }
    fn names<const N: usize>(
        physical: &str,
        names: Self::Names,
    ) -> Result<Self::CheckedNames<N>, IoError> {
        if names.len() != N {
            return Err(IoError::InvalidFormat(format!(
                "GGUF tensor {physical:?} cataloged {} logical outputs, expected {N}",
                names.len()
            )));
        }
        Ok(names)
    }
    fn dense(_: Self::Descriptor, names: Self::CheckedNames<1>, array: Array) -> Self::Output {
        arrays(names, [array])
    }
    fn iquant(
        _: Self::Descriptor,
        names: Self::CheckedNames<1>,
        _: eredu_gguf::GgmlType,
        _: eredu_gguf::Endian,
        _: Vec<i32>,
        array: Array,
    ) -> Self::Output {
        arrays(names, [array])
    }
    fn affine(
        _: Self::Descriptor,
        names: Self::CheckedNames<3>,
        _: u8,
        _: u32,
        values: [Array; 3],
    ) -> Self::Output {
        arrays(names, values)
    }
    fn mxfp4(
        _: Self::Descriptor,
        names: Self::CheckedNames<2>,
        values: [Array; 2],
    ) -> Self::Output {
        arrays(names, values)
    }
}
fn arrays<const N: usize>(
    names: StoredOutputNames<AdmittedFamily>,
    values: [Array; N],
) -> CachedGgufArrays {
    let mut values = values.into_iter();
    CachedGgufArrays::Supplied {
        values: std::array::from_fn(|_| values.next()),
        names,
    }
}
impl Realization<StoredGroup> for OriginalProducer<'_> {
    type Error = GgufHostCopyCause;
    fn bind_group(&mut self, descriptor: &StoredDescriptor<AdmittedFamily>) {
        self.host.bind_plan_view(descriptor.view());
    }
    fn f32_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F32,
            f32::from_ne_bytes,
            FailedCopy::F32,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F32,
            self.observer,
            FailedCopy::F32,
        )
    }
    fn f16_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F16,
            |bytes| half::f16::from_bits(u16::from_ne_bytes(bytes)),
            FailedCopy::F16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F16,
            self.observer,
            FailedCopy::F16,
        )
    }
    fn bf16_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::Bf16,
            |bytes| half::bf16::from_bits(u16::from_ne_bytes(bytes)),
            FailedCopy::Bf16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::Bf16,
            self.observer,
            FailedCopy::Bf16,
        )
    }
    fn i8_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I8,
            i8::from_ne_bytes,
            FailedCopy::I8,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I8,
            self.observer,
            FailedCopy::I8,
        )
    }
    fn i16_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I16,
            i16::from_ne_bytes,
            FailedCopy::I16,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I16,
            self.observer,
            FailedCopy::I16,
        )
    }
    fn i32_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I32,
            i32::from_ne_bytes,
            FailedCopy::I32,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I32,
            self.observer,
            FailedCopy::I32,
        )
    }
    fn i64_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::I64,
            i64::from_ne_bytes,
            FailedCopy::I64,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::I64,
            self.observer,
            FailedCopy::I64,
        )
    }
    fn f64_bytes(
        &mut self,
        bytes: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_bytes(
            bytes,
            LogicalDtype::F64,
            f64::from_ne_bytes,
            FailedCopy::F64,
        )?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F64,
            self.observer,
            FailedCopy::F64,
        )
    }
    fn f16_bits(
        &mut self,
        bits: StoredBuffer<AdmittedFamily, u16>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        let values = self.host.transform_half_bits(bits)?;
        self.host.create(
            values,
            shape,
            LogicalDtype::F16,
            self.observer,
            FailedCopy::F16,
        )
    }
    fn u8(
        &mut self,
        values: StoredBuffer<AdmittedFamily, u8>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::U8,
            self.observer,
            FailedCopy::U8,
        )
    }
    fn u32(
        &mut self,
        values: StoredBuffer<AdmittedFamily, u32>,
        shape: &[i32],
    ) -> Result<Array, Self::Error> {
        self.host.create(
            values.into(),
            shape,
            LogicalDtype::U32,
            self.observer,
            FailedCopy::U32,
        )
    }
}
impl PendingWeightMaterialization {
    pub(in crate::backend::runtime::checkpoint::store) fn convert_original_stored_gguf(
        &mut self,
        portable: StoredCheckpointTensor<AdmittedFamily>,
    ) -> Result<CachedGgufArrays, PreparedGgufHostCopyFailure> {
        let observer = self
            .retained
            .original_observer()
            .expect("original pending recovery")
            .clone();
        let host = self.bind_original_gguf_host();
        let result = gguf::conversion::convert::<StoredGroup, _>(
            portable,
            &mut OriginalProducer {
                host: &mut *host,
                observer: &observer,
            },
        );
        result.map_err(|cause| host.failure(cause))
    }
}
