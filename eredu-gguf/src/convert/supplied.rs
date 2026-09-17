//! Supplied G2 owners; all conversion equations remain in the ordinary worker.
use super::*;
use crate::supplied_storage::{
    StorageFamily, StorageProvider, StoredBuffer, StoredDescriptor, SuppliedStorageError,
};
use destination::StoredKind;

/// Final G2 buffers and exact descriptor binding, retained on every refusal.
#[derive(Debug)]
pub struct StoredConversion<F: StorageFamily> {
    shape1: StoredBuffer<F, u64>,
    shape2: StoredBuffer<F, u64>,
    bytes: StoredBuffer<F, u8>,
    words: StoredBuffer<F, u32>,
    scales: StoredBuffer<F, u16>,
    biases: StoredBuffer<F, u16>,
    e8m0: StoredBuffer<F, u8>,
    descriptor: StoredDescriptor<F>,
    endian: Endian,
    limits: [usize; 7],
    scratch: [u8; 32],
    consumed: bool,
    pair_attempted: bool,
    completed: Option<StoredKind>,
}

/// Actual preparation cause plus every reached supplied allocation.
#[derive(Debug)]
pub struct StoredConversionFailure<F: StorageFamily> {
    cause: SuppliedStorageError<F::Error>,
    destination: StoredConversion<F>,
}
impl<F: StorageFamily> StoredConversionFailure<F> {
    /// Preserve cause and destination without dropping or cloning either.
    pub fn into_parts(self) -> (SuppliedStorageError<F::Error>, StoredConversion<F>) {
        (self.cause, self.destination)
    }
    /// Actual failed preparation cause.
    pub fn cause(&self) -> &SuppliedStorageError<F::Error> {
        &self.cause
    }
    /// Complete retained storage prefix.
    pub fn destination(&self) -> &StoredConversion<F> {
        &self.destination
    }
}
impl<F: StorageFamily> std::fmt::Display for StoredConversionFailure<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<F: StorageFamily> std::error::Error for StoredConversionFailure<F> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl<F: StorageFamily> StoredConversion<F> {
    /// Consume the genuinely selected descriptor owner and follow the original
    /// G2 reserve/shape order. No alternate format dispatch or plan is built.
    pub fn prepare<P: StorageProvider<Family = F>>(
        descriptor: StoredDescriptor<F>,
        endian: Endian,
        provider: &mut P,
    ) -> std::result::Result<Self, StoredConversionFailure<F>> {
        let mut out = Self {
            shape1: StoredBuffer::default(),
            shape2: StoredBuffer::default(),
            bytes: StoredBuffer::default(),
            words: StoredBuffer::default(),
            scales: StoredBuffer::default(),
            biases: StoredBuffer::default(),
            e8m0: StoredBuffer::default(),
            descriptor,
            endian,
            limits: [0; 7],
            scratch: [0; 32],
            consumed: false,
            pair_attempted: false,
            completed: None,
        };
        let result = preparation::prepare(
            out.descriptor.view(),
            endian,
            false,
            &mut SuppliedPreparation {
                provider,
                shape1: &mut out.shape1,
                shape2: &mut out.shape2,
                bytes: &mut out.bytes,
                words: &mut out.words,
                scales: &mut out.scales,
                biases: &mut out.biases,
                e8m0: &mut out.e8m0,
                limits: &mut out.limits,
            },
        );
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(StoredConversionFailure {
                cause,
                destination: out,
            }),
        }
    }
    /// Exact selected descriptor fields, borrowed from their supplied owners.
    pub fn descriptor(&self) -> TensorDescriptorView<'_> {
        self.descriptor.view()
    }
    /// Existing requested element limits; these are not emitted lengths.
    pub fn requested_elements(&self) -> [usize; 7] {
        self.limits
    }
    /// Actual capacities, in the same order as the requested elements.
    pub fn capacities(&self) -> [usize; 7] {
        [
            self.shape1.capacity(),
            self.shape2.capacity(),
            self.bytes.capacity(),
            self.words.capacity(),
            self.scales.capacity(),
            self.biases.capacity(),
            self.e8m0.capacity(),
        ]
    }
    /// Whether the shared converter finished, including before a later metadata refusal.
    pub fn completed(&self) -> bool {
        self.completed.is_some()
    }
    pub(crate) fn started(&self) -> bool {
        self.consumed || self.pair_attempted
    }
    pub(crate) fn mark_pair_attempted(&mut self) {
        self.pair_attempted = true;
    }
    pub(crate) fn fill(
        &mut self,
        raw: &[u8],
        descriptor: TensorDescriptorView<'_>,
        endian: Endian,
    ) -> CResult<()> {
        if self.consumed {
            return Err(ConversionDestinationError::Consumed);
        }
        self.consumed = true;
        let storage = Storage {
            binding: Some((self.descriptor.view(), self.endian)),
            shape1: Some(Slot::Fixed(self.shape1.loan())),
            shape2: Some(Slot::Fixed(self.shape2.loan())),
            bytes: Some(Slot::Fixed(self.bytes.loan())),
            words: Some(Slot::Fixed(self.words.loan())),
            scales: Some(Slot::Fixed(self.scales.loan())),
            biases: Some(Slot::Fixed(self.biases.loan())),
            e8m0: Some(Slot::Fixed(self.e8m0.loan())),
            scratch: Some(&mut self.scratch),
        };
        let output = convert_with_storage(descriptor, raw, endian, storage)?;
        self.completed = Some(output.stored_kind());
        Ok(())
    }
    pub(crate) fn finish(self) -> StoredConvertedTensor<F> {
        match self
            .completed
            .expect("only completed G2 emits its owning result")
        {
            StoredKind::Dense(dtype) => StoredConvertedTensor::Dense {
                shape: self.shape1,
                dtype,
                data: self.bytes,
            },
            StoredKind::IQuant { ggml_type, endian } => StoredConvertedTensor::IQuant {
                shape: self.shape1,
                ggml_type,
                endian,
                data: self.bytes,
            },
            StoredKind::MxFp4 => StoredConvertedTensor::MxFp4 {
                weight_shape: self.shape1,
                scale_shape: self.shape2,
                weights: self.words,
                scales: self.e8m0,
            },
            StoredKind::Affine { bits, group_size } => StoredConvertedTensor::Affine {
                weight_shape: self.shape1,
                scale_shape: self.shape2,
                bits,
                group_size,
                weights: self.words,
                scales: self.scales,
                biases: self.biases,
            },
        }
    }
}

/// The four ordinary conversion representations with intact supplied owners.
/// Shape, payload and emitted prefix are produced by the same shared equations.
#[derive(Debug)]
pub enum StoredConvertedTensor<F: StorageFamily> {
    Dense {
        shape: StoredBuffer<F, u64>,
        dtype: DenseDtype,
        data: StoredBuffer<F, u8>,
    },
    IQuant {
        shape: StoredBuffer<F, u64>,
        ggml_type: GgmlType,
        endian: Endian,
        data: StoredBuffer<F, u8>,
    },
    Affine {
        weight_shape: StoredBuffer<F, u64>,
        scale_shape: StoredBuffer<F, u64>,
        bits: u8,
        group_size: u32,
        weights: StoredBuffer<F, u32>,
        scales: StoredBuffer<F, u16>,
        biases: StoredBuffer<F, u16>,
    },
    MxFp4 {
        weight_shape: StoredBuffer<F, u64>,
        scale_shape: StoredBuffer<F, u64>,
        weights: StoredBuffer<F, u32>,
        scales: StoredBuffer<F, u8>,
    },
}

struct SuppliedPreparation<'a, F: StorageFamily, P> {
    provider: &'a mut P,
    shape1: &'a mut StoredBuffer<F, u64>,
    shape2: &'a mut StoredBuffer<F, u64>,
    bytes: &'a mut StoredBuffer<F, u8>,
    words: &'a mut StoredBuffer<F, u32>,
    scales: &'a mut StoredBuffer<F, u16>,
    biases: &'a mut StoredBuffer<F, u16>,
    e8m0: &'a mut StoredBuffer<F, u8>,
    limits: &'a mut [usize; 7],
}
impl<F: StorageFamily, P: StorageProvider<Family = F>> preparation::Preparation
    for SuppliedPreparation<'_, F, P>
{
    type Error = SuppliedStorageError<F::Error>;
    fn reserve(&mut self, index: usize, count: usize) -> std::result::Result<(), Self::Error> {
        match index {
            0 => self.shape1.prepare(self.provider, count, 0),
            1 => self.shape2.prepare(self.provider, count, 0),
            2 => self.bytes.prepare(self.provider, count, 0),
            3 => self.words.prepare(self.provider, count, 0),
            4 => self.scales.prepare(self.provider, count, 0),
            5 => self.biases.prepare(self.provider, count, 0),
            6 => self.e8m0.prepare(self.provider, count, 0),
            _ => unreachable!("closed conversion storage fields"),
        }?;
        self.limits[index] = count;
        Ok(())
    }
    fn shapes(&mut self) -> (Slot<'_, u64>, Slot<'_, u64>) {
        (
            Slot::Fixed(self.shape1.loan()),
            Slot::Fixed(self.shape2.loan()),
        )
    }
}
