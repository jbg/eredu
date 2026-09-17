//! Concrete final-vector storage for the shared conversion equations.
use super::*;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    ops::{Deref, DerefMut},
};

/// Failure of original conversion or its explicitly prepared storage.
#[derive(Debug)]
pub enum ConversionDestinationError {
    /// Original GGUF semantic error.
    Gguf(Error),
    /// Descriptor or byte order differs from the prepared binding.
    Binding,
    /// This destination has already attempted conversion.
    Consumed,
    /// A requested typed allocation is not representable.
    Layout,
    /// A real cold Vec reserve failed.
    Reserve(TryReserveError),
    /// A prepared vector cannot accept the requested write without growth.
    Capacity {
        /// Number of elements required by this operation.
        required: usize,
        /// Prepared element limit.
        limit: usize,
        /// Actual Vec capacity.
        capacity: usize,
    },
}
impl From<Error> for ConversionDestinationError {
    fn from(e: Error) -> Self {
        Self::Gguf(e)
    }
}
impl std::fmt::Display for ConversionDestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(e) => e.fmt(f),
            Self::Binding => f.write_str("GGUF conversion destination binding differs"),
            Self::Consumed => f.write_str("GGUF conversion destination was already consumed"),
            Self::Layout => f.write_str("GGUF conversion destination layout is not representable"),
            Self::Reserve(e) => e.fmt(f),
            Self::Capacity {
                required,
                limit,
                capacity,
            } => write!(
                f,
                "GGUF conversion requires {required} elements with limit {limit} and capacity {capacity}"
            ),
        }
    }
}
impl std::error::Error for ConversionDestinationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(e) => Some(e),
            Self::Reserve(e) => Some(e),
            _ => None,
        }
    }
}
impl ConversionDestinationError {
    pub(crate) fn ordinary(self) -> Error {
        match self {
            Self::Gguf(e) => e,
            _ => unreachable!("ordinary conversion has no fixed storage"),
        }
    }
}
pub(super) type CResult<T> = std::result::Result<T, ConversionDestinationError>;

/// Exact requested element layouts, distinct from actual capacities and allocator charge.
#[derive(Clone, Copy, Debug)]
pub struct ConversionLayouts {
    /// Shape one, shape two, bytes, words, f16 scales, f16 biases, E8M0 scales.
    pub vectors: [Layout; 7],
    /// Retained descriptor name bytes; its String header is inside owner.
    pub descriptor_name: Layout,
    /// Retained descriptor dimensions; its Vec header is inside owner.
    pub descriptor_dimensions: Layout,
    /// Actual owner type, including all Vec headers and inline Q8 scratch.
    pub owner: Layout,
}

/// Owns final conversion vector storage, a descriptor binding and Q8 scratch.
/// This is low-level storage, not source provenance or admission authority.
#[derive(Debug)]
pub struct PreparedConversion {
    descriptor: TensorDescriptor,
    endian: Endian,
    force_affine: bool,
    consumed: bool,
    limits: [usize; 7],
    shape1: Vec<u64>,
    shape2: Vec<u64>,
    bytes: Vec<u8>,
    words: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    e8m0: Vec<u8>,
    scratch: [u8; 32],
}
/// Retains successful preparation prefixes or partial conversion output and its cause.
#[derive(Debug)]
pub struct PreparedConversionFailure {
    cause: ConversionDestinationError,
    destination: PreparedConversion,
}
impl PreparedConversionFailure {
    /// Original semantic or storage cause.
    pub fn cause(&self) -> &ConversionDestinationError {
        &self.cause
    }
    /// Retained destination, including descriptor and actual capacities.
    pub fn destination(&self) -> &PreparedConversion {
        &self.destination
    }
    /// Moves the retained cause and destination without dropping either.
    pub fn into_parts(self) -> (ConversionDestinationError, PreparedConversion) {
        (self.cause, self.destination)
    }
}
impl std::fmt::Display for PreparedConversionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for PreparedConversionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

fn reserve<T>(v: &mut Vec<T>, len: usize) -> CResult<()> {
    Layout::array::<T>(len).map_err(|_| ConversionDestinationError::Layout)?;
    v.try_reserve_exact(len)
        .map_err(ConversionDestinationError::Reserve)
}
impl PreparedConversion {
    /// Prepares the ordinary dtype/endian dispatch for this owned descriptor.
    pub fn prepare(
        descriptor: TensorDescriptor,
        endian: Endian,
    ) -> std::result::Result<Self, PreparedConversionFailure> {
        Self::prepare_inner(descriptor, endian, false)
    }
    /// Prepares the explicit affine API, without changing ordinary source dispatch.
    pub fn prepare_affine(
        descriptor: TensorDescriptor,
        endian: Endian,
    ) -> std::result::Result<Self, PreparedConversionFailure> {
        Self::prepare_inner(descriptor, endian, true)
    }
    fn prepare_inner(
        descriptor: TensorDescriptor,
        endian: Endian,
        force_affine: bool,
    ) -> std::result::Result<Self, PreparedConversionFailure> {
        let mut out = Self {
            descriptor,
            endian,
            force_affine,
            consumed: false,
            limits: [0; 7],
            shape1: Vec::new(),
            shape2: Vec::new(),
            bytes: Vec::new(),
            words: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            e8m0: Vec::new(),
            scratch: [0; 32],
        };
        let result = out.prepare_storage();
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(PreparedConversionFailure {
                cause,
                destination: out,
            }),
        }
    }
    fn prepare_storage(&mut self) -> CResult<()> {
        super::preparation::prepare(
            self.descriptor.view(),
            self.endian,
            self.force_affine,
            &mut VecPreparation {
                shape1: &mut self.shape1,
                shape2: &mut self.shape2,
                bytes: &mut self.bytes,
                words: &mut self.words,
                scales: &mut self.scales,
                biases: &mut self.biases,
                e8m0: &mut self.e8m0,
                limits: &mut self.limits,
            },
        )
    }
    /// Requested vector and binding layouts; allocator overhead is excluded.
    pub fn layouts(&self) -> Option<ConversionLayouts> {
        super::plan::requested_layouts(self.descriptor.view(), self.limits)
    }
    /// Actual capacities in the same order as ConversionLayouts::vectors.
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
    /// Descriptor owned by this storage plan; not proof of input-byte provenance.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }
    /// Converts using the stored descriptor, retaining this owner on error.
    pub fn convert(
        mut self,
        raw: &[u8],
    ) -> std::result::Result<ConvertedTensor, PreparedConversionFailure> {
        match self.run(raw, None) {
            Ok(output) => Ok(output),
            Err(cause) => Err(PreparedConversionFailure {
                cause,
                destination: self,
            }),
        }
    }
    #[cfg(test)]
    pub(crate) fn fill(
        &mut self,
        raw: &[u8],
        descriptor: &TensorDescriptor,
        endian: Endian,
    ) -> CResult<ConvertedTensor> {
        self.run(raw, Some((descriptor.view(), endian)))
    }
    pub(crate) fn fill_view(
        &mut self,
        raw: &[u8],
        descriptor: TensorDescriptorView<'_>,
        endian: Endian,
    ) -> CResult<ConvertedTensor> {
        self.run(raw, Some((descriptor, endian)))
    }
    fn run(
        &mut self,
        raw: &[u8],
        actual: Option<(TensorDescriptorView<'_>, Endian)>,
    ) -> CResult<ConvertedTensor> {
        if self.consumed {
            return Err(ConversionDestinationError::Consumed);
        }
        self.consumed = true;
        let n = self.limits;
        let descriptor = actual.map(|x| x.0).unwrap_or(self.descriptor.view());
        let endian = actual.map(|x| x.1).unwrap_or(self.endian);
        let storage = Storage {
            binding: Some((self.descriptor.view(), self.endian)),
            shape1: Some(Slot::Vec {
                vec: &mut self.shape1,
                limit: n[0],
            }),
            shape2: Some(Slot::Vec {
                vec: &mut self.shape2,
                limit: n[1],
            }),
            bytes: Some(Slot::Vec {
                vec: &mut self.bytes,
                limit: n[2],
            }),
            words: Some(Slot::Vec {
                vec: &mut self.words,
                limit: n[3],
            }),
            scales: Some(Slot::Vec {
                vec: &mut self.scales,
                limit: n[4],
            }),
            biases: Some(Slot::Vec {
                vec: &mut self.biases,
                limit: n[5],
            }),
            e8m0: Some(Slot::Vec {
                vec: &mut self.e8m0,
                limit: n[6],
            }),
            scratch: Some(&mut self.scratch),
        };
        if self.force_affine {
            if raw.len() as u64 != descriptor.byte_len {
                return Err(Error::tensor(
                    descriptor.name,
                    "payload length does not match descriptor",
                )
                .into());
            }
            if actual.is_some() {
                // Reader always owns ordinary source dispatch. The separate
                // low-level affine API cannot override its selected variant.
                conversion_kind(descriptor.ggml_type, endian)?;
                return Err(ConversionDestinationError::Binding);
            }
            storage.validate(descriptor, endian)?;
            affine_with_storage(descriptor, raw, endian, storage)
                .map(ConversionOutput::Affine)
                .map(ConversionOutput::into_owned)
        } else {
            convert_with_storage(descriptor, raw, endian, storage).map(ConversionOutput::into_owned)
        }
    }
}

pub(super) enum Slot<'a, T> {
    Vec { vec: &'a mut Vec<T>, limit: usize },
    Fixed(crate::supplied_storage::FixedSlice<'a, T>),
}
impl<T> Slot<'_, T> {
    fn capacity(&self) -> usize {
        match self {
            Self::Vec { vec, .. } => vec.capacity(),
            Self::Fixed(s) => s.data.len(),
        }
    }
    fn limit(&self) -> usize {
        match self {
            Self::Vec { limit, .. } => *limit,
            Self::Fixed(s) => s.limit,
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Vec { vec, .. } => vec.clear(),
            Self::Fixed(s) => s.clear(),
        }
    }
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Vec { vec, .. } => vec,
            Self::Fixed(s) => s.as_slice(),
        }
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Vec { vec, .. } => vec,
            Self::Fixed(s) => s.as_mut_slice(),
        }
    }
    fn push(&mut self, value: T) -> CResult<()> {
        match self {
            Self::Vec { vec, .. } => {
                vec.push(value);
                Ok(())
            }
            Self::Fixed(s) => s.push(value),
        }
    }
}
pub(super) enum Vector<'a, T> {
    Ordinary(Vec<T>),
    Prepared(Slot<'a, T>),
}
impl<T> Deref for Vector<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Ordinary(v) => v,
            Self::Prepared(s) => s.as_slice(),
        }
    }
}
impl<T> DerefMut for Vector<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Ordinary(v) => v,
            Self::Prepared(s) => s.as_mut_slice(),
        }
    }
}
impl<T> Vector<'_, T> {
    fn check(&self, required: usize) -> CResult<()> {
        if let Self::Prepared(s) = self {
            if required > s.limit() || required > s.capacity() {
                return Err(ConversionDestinationError::Capacity {
                    required,
                    limit: s.limit(),
                    capacity: s.capacity(),
                });
            }
        }
        Ok(())
    }
    pub fn push(&mut self, value: T) -> CResult<()> {
        if let Self::Ordinary(v) = self {
            v.push(value);
            return Ok(());
        }
        let required = self
            .len()
            .checked_add(1)
            .ok_or(ConversionDestinationError::Layout)?;
        self.check(required)?;
        match self {
            Self::Ordinary(v) => {
                v.push(value);
                Ok(())
            }
            Self::Prepared(s) => s.push(value),
        }
    }
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Ordinary(v) => v,
            Self::Prepared(Slot::Vec { vec, .. }) => std::mem::take(vec),
            Self::Prepared(Slot::Fixed(_)) => {
                unreachable!("supplied result never detaches its owner")
            }
        }
    }
}
impl<T: Copy> Vector<'_, T> {
    pub fn resize(&mut self, len: usize, value: T) -> CResult<()> {
        self.check(len)?;
        match self {
            Self::Ordinary(v) => v.resize(len, value),
            Self::Prepared(Slot::Vec { vec, .. }) => vec.resize(len, value),
            Self::Prepared(Slot::Fixed(s)) => return s.resize(len, value),
        }
        Ok(())
    }
}
pub(super) fn capacity<T>(count: usize, slot: Option<Slot<'_, T>>) -> CResult<Vector<'_, T>> {
    match slot {
        None => Ok(Vector::Ordinary(Vec::with_capacity(count))),
        Some(mut s) => {
            if count > s.limit() || count > s.capacity() {
                return Err(ConversionDestinationError::Capacity {
                    required: count,
                    limit: s.limit(),
                    capacity: s.capacity(),
                });
            }
            s.clear();
            Ok(Vector::Prepared(s))
        }
    }
}
pub(super) fn shape<'a>(
    desc: TensorDescriptorView<'_>,
    slot: Option<Slot<'a, u64>>,
) -> CResult<Vector<'a, u64>> {
    match slot {
        None => Ok(Vector::Ordinary(desc.row_major_shape())),
        Some(slot) => {
            let mut out = capacity(desc.dimensions.len(), Some(slot))?;
            for x in desc.dimensions.iter().rev() {
                out.push(*x)?;
            }
            Ok(out)
        }
    }
}
pub(super) fn copied<'a>(raw: &[u8], slot: Option<Slot<'a, u8>>) -> CResult<Vector<'a, u8>> {
    match slot {
        None => Ok(Vector::Ordinary(raw.to_vec())),
        Some(slot) => {
            let mut out = capacity(raw.len(), Some(slot))?;
            for x in raw {
                out.push(*x)?;
            }
            Ok(out)
        }
    }
}
pub(super) struct Storage<'a> {
    pub binding: Option<(TensorDescriptorView<'a>, Endian)>,
    pub shape1: Option<Slot<'a, u64>>,
    pub shape2: Option<Slot<'a, u64>>,
    pub bytes: Option<Slot<'a, u8>>,
    pub words: Option<Slot<'a, u32>>,
    pub scales: Option<Slot<'a, u16>>,
    pub biases: Option<Slot<'a, u16>>,
    pub e8m0: Option<Slot<'a, u8>>,
    pub scratch: Option<&'a mut [u8; 32]>,
}
impl Storage<'_> {
    pub fn ordinary() -> Self {
        Self {
            binding: None,
            shape1: None,
            shape2: None,
            bytes: None,
            words: None,
            scales: None,
            biases: None,
            e8m0: None,
            scratch: None,
        }
    }
    pub fn validate(&self, descriptor: TensorDescriptorView<'_>, endian: Endian) -> CResult<()> {
        if self
            .binding
            .is_some_and(|(d, e)| d != descriptor || e != endian)
        {
            Err(ConversionDestinationError::Binding)
        } else {
            Ok(())
        }
    }
}
pub(super) struct AffineOutput<'a> {
    pub weight_shape: Vector<'a, u64>,
    pub scale_shape: Vector<'a, u64>,
    pub bits: u8,
    pub group_size: u32,
    pub weights: Vector<'a, u32>,
    pub scales: Vector<'a, u16>,
    pub biases: Vector<'a, u16>,
}
impl AffineOutput<'_> {
    pub fn finish(self) -> AffineTensor {
        AffineTensor {
            weight_shape: self.weight_shape.into_vec(),
            scale_shape: self.scale_shape.into_vec(),
            bits: self.bits,
            group_size: self.group_size,
            weights: self.weights.into_vec(),
            scales: self.scales.into_vec(),
            biases: self.biases.into_vec(),
        }
    }
}
pub(super) enum Codes<'a> {
    Ordinary(Vec<u8>),
    Prepared(&'a [u8; 32]),
}
impl Deref for Codes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Ordinary(v) => v,
            Self::Prepared(v) => v.as_slice(),
        }
    }
}
pub(super) fn q8_codes<'a>(bytes: &[u8], scratch: Option<&'a mut [u8; 32]>) -> Codes<'a> {
    let codes = bytes.iter().map(|x| x ^ 0x80);
    match scratch {
        None => Codes::Ordinary(codes.collect()),
        Some(s) => {
            for (out, x) in s.iter_mut().zip(codes) {
                *out = x;
            }
            Codes::Prepared(s)
        }
    }
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StoredKind {
    Dense(DenseDtype),
    IQuant { ggml_type: GgmlType, endian: Endian },
    Affine { bits: u8, group_size: u32 },
    MxFp4,
}
pub(super) struct MxFp4Output<'a> {
    pub weight_shape: Vector<'a, u64>,
    pub scale_shape: Vector<'a, u64>,
    pub weights: Vector<'a, u32>,
    pub scales: Vector<'a, u8>,
}
pub(super) enum ConversionOutput<'a> {
    Dense {
        shape: Vector<'a, u64>,
        dtype: DenseDtype,
        data: Vector<'a, u8>,
    },
    IQuant {
        shape: Vector<'a, u64>,
        ggml_type: GgmlType,
        endian: Endian,
        data: Vector<'a, u8>,
    },
    Affine(AffineOutput<'a>),
    MxFp4(MxFp4Output<'a>),
}
impl ConversionOutput<'_> {
    pub(super) fn into_owned(self) -> ConvertedTensor {
        match self {
            Self::Dense { shape, dtype, data } => ConvertedTensor::Dense(DenseTensor {
                shape: shape.into_vec(),
                dtype,
                data: data.into_vec(),
            }),
            Self::IQuant {
                shape,
                ggml_type,
                endian,
                data,
            } => ConvertedTensor::IQuant(IQuantTensor {
                shape: shape.into_vec(),
                ggml_type,
                endian,
                data: data.into_vec(),
            }),
            Self::Affine(v) => ConvertedTensor::Affine(v.finish()),
            Self::MxFp4(v) => ConvertedTensor::MxFp4(MxFp4Tensor {
                weight_shape: v.weight_shape.into_vec(),
                scale_shape: v.scale_shape.into_vec(),
                weights: v.weights.into_vec(),
                scales: v.scales.into_vec(),
            }),
        }
    }
    pub(super) fn stored_kind(self) -> StoredKind {
        match self {
            Self::Dense { dtype, .. } => StoredKind::Dense(dtype),
            Self::IQuant {
                ggml_type, endian, ..
            } => StoredKind::IQuant { ggml_type, endian },
            Self::Affine(v) => StoredKind::Affine {
                bits: v.bits,
                group_size: v.group_size,
            },
            Self::MxFp4(_) => StoredKind::MxFp4,
        }
    }
}

struct VecPreparation<'a> {
    shape1: &'a mut Vec<u64>,
    shape2: &'a mut Vec<u64>,
    bytes: &'a mut Vec<u8>,
    words: &'a mut Vec<u32>,
    scales: &'a mut Vec<u16>,
    biases: &'a mut Vec<u16>,
    e8m0: &'a mut Vec<u8>,
    limits: &'a mut [usize; 7],
}
impl super::preparation::Preparation for VecPreparation<'_> {
    type Error = ConversionDestinationError;
    fn reserve(&mut self, index: usize, count: usize) -> CResult<()> {
        match index {
            0 => reserve(self.shape1, count),
            1 => reserve(self.shape2, count),
            2 => reserve(self.bytes, count),
            3 => reserve(self.words, count),
            4 => reserve(self.scales, count),
            5 => reserve(self.biases, count),
            6 => reserve(self.e8m0, count),
            _ => unreachable!("closed conversion storage fields"),
        }?;
        self.limits[index] = count;
        Ok(())
    }
    fn shapes(&mut self) -> (Slot<'_, u64>, Slot<'_, u64>) {
        (
            Slot::Vec {
                vec: self.shape1,
                limit: self.limits[0],
            },
            Slot::Vec {
                vec: self.shape2,
                limit: self.limits[1],
            },
        )
    }
}
