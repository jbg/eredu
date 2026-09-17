// Conversion layout translated from MLX v0.32.0 `mlx/io/gguf_quants.cpp`
// (Apple Inc., MIT license) and Eredu's former in-tree MLX patch set. The resulting
// buffers intentionally match MLX affine quantization byte-for-byte.
use crate::{Endian, Error, GgmlType, Result, TensorDescriptor, TensorDescriptorView};
use half::f16;
mod destination;
mod plan;
mod preparation;
mod supplied;
use destination::{
    capacity, copied, q8_codes, shape, AffineOutput, CResult, ConversionOutput, MxFp4Output, Slot,
    Storage, Vector,
};
pub use destination::{
    ConversionDestinationError, ConversionLayouts, PreparedConversion, PreparedConversionFailure,
};
pub use plan::{ConversionOutputPlan, ConversionPlan};
pub use supplied::{StoredConversion, StoredConversionFailure, StoredConvertedTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseDtype {
    F32,
    F16,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensor {
    pub shape: Vec<u64>,
    pub dtype: DenseDtype,
    pub data: Vec<u8>,
}

/// One GGML tensor retained in its checkpoint-native block layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IQuantTensor {
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    pub endian: Endian,
    pub data: Vec<u8>,
}

impl IQuantTensor {
    /// Shape of the packed byte rows consumed by checkpoint-native runtimes.
    pub fn packed_shape(&self) -> Result<Vec<u64>> {
        iquant_packed_shape(&self.shape, self.ggml_type)
    }

    /// Canonically dequantize native GGML blocks for differential testing and
    /// generic execution backends. Model loading does not call this method.
    pub fn dequantize_f32(&self) -> Result<Vec<f32>> {
        if self.ggml_type.is_iq() {
            return crate::iquant::decode_f32(self.ggml_type, &self.data, self.endian);
        }
        let descriptor = TensorDescriptor {
            name: "<native blocks>".to_string(),
            dimensions: self.shape.iter().rev().copied().collect(),
            ggml_type: self.ggml_type,
            relative_offset: 0,
            data_offset: 0,
            byte_len: self.data.len() as u64,
        };
        Ok(convert_affine(&descriptor, &self.data, self.endian)?.dequantize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTensor {
    pub weight_shape: Vec<u64>,
    pub scale_shape: Vec<u64>,
    pub bits: u8,
    pub group_size: u32,
    pub weights: Vec<u32>,
    /// IEEE f16 bit patterns.
    pub scales: Vec<u16>,
    /// IEEE f16 bit patterns.
    pub biases: Vec<u16>,
}

/// Logical packed MXFP4 representation reconstructed from GGML type 39 blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MxFp4Tensor {
    pub weight_shape: Vec<u64>,
    pub scale_shape: Vec<u64>,
    pub weights: Vec<u32>,
    pub scales: Vec<u8>,
}

impl AffineTensor {
    /// Dequantize the affine representation using f16-rounded scales/biases.
    pub fn dequantize(&self) -> Vec<f32> {
        let count = self.scales.len() * self.group_size as usize;
        let mut out = Vec::with_capacity(count);
        let mask = (1u32 << self.bits) - 1;
        for index in 0..count {
            let bit = index * self.bits as usize;
            let word = bit / 32;
            let shift = bit % 32;
            let mut code = self.weights[word] >> shift;
            if shift + self.bits as usize > 32 {
                code |= self.weights[word + 1] << (32 - shift);
            }
            let group = index / self.group_size as usize;
            let scale = f16::from_bits(self.scales[group]).to_f32();
            let bias = f16::from_bits(self.biases[group]).to_f32();
            out.push(scale * (code & mask) as f32 + bias);
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertedTensor {
    Dense(DenseTensor),
    IQuant(IQuantTensor),
    Affine(AffineTensor),
    MxFp4(MxFp4Tensor),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConversionKind {
    Dense(DenseDtype),
    /// GGML blocks supported by checkpoint-native execution.
    IQuant,
    /// MXFP4 weights plus one E8M0 byte per 32-value group.
    MxFp4,
    Affine {
        bits: u8,
        group_size: u32,
    },
}

pub(crate) fn conversion_kind(ty: GgmlType, endian: Endian) -> Result<ConversionKind> {
    if let Some(dtype) = dense_dtype(ty) {
        return Ok(ConversionKind::Dense(dtype));
    }
    if ty.is_iq() || (endian == Endian::Little && ty.has_native_execution()) {
        return Ok(ConversionKind::IQuant);
    }
    if ty == GgmlType::MxFp4 {
        return Ok(ConversionKind::MxFp4);
    }
    let (bits, group_size) = match affine_config(ty) {
        Some(config) => config,
        None => return Err(Error::UnsupportedTensorType(ty.code())),
    };
    Ok(ConversionKind::Affine { bits, group_size })
}

fn affine_config(ty: GgmlType) -> Option<(u8, u32)> {
    Some(match ty {
        GgmlType::Q2K => (2, 16),
        GgmlType::Q3K => (3, 16),
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4K => (4, 32),
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5K => (5, 32),
        GgmlType::Q6K => (6, 16),
        GgmlType::Q8_0 => (8, 32),
        _ => return None,
    })
}

pub(crate) fn affine_shapes(
    desc: &TensorDescriptor,
    bits: u8,
    group_size: u32,
) -> Result<(Vec<u64>, Vec<u64>)> {
    affine_shapes_with_storage(desc.view(), bits, group_size, None, None)
        .map(|(a, b)| (a.into_vec(), b.into_vec()))
        .map_err(ConversionDestinationError::ordinary)
}
fn affine_shapes_with_storage<'a>(
    desc: TensorDescriptorView<'_>,
    bits: u8,
    group_size: u32,
    first: Option<Slot<'a, u64>>,
    second: Option<Slot<'a, u64>>,
) -> CResult<(Vector<'a, u64>, Vector<'a, u64>)> {
    let mut weight_shape = shape(desc, first)?;
    let last = weight_shape
        .last_mut()
        .ok_or_else(|| Error::tensor(desc.name, "quantized scalar is invalid"))?;
    if *last % u64::from(group_size) != 0 {
        return Err(Error::tensor(
            desc.name,
            format!("last dimension is not divisible by group size {group_size}"),
        )
        .into());
    }
    *last = last
        .checked_mul(u64::from(bits))
        .ok_or(Error::Overflow("affine packed dimension"))?
        / 32;
    let mut scale_shape = shape(desc, second)?;
    *scale_shape.last_mut().unwrap() /= u64::from(group_size);
    Ok((weight_shape, scale_shape))
}

pub(crate) fn iquant_packed_shape(shape: &[u64], ty: GgmlType) -> Result<Vec<u64>> {
    let (block_values, block_bytes) = ty.block_and_bytes()?;
    let mut packed = shape.to_vec();
    let columns = packed
        .last_mut()
        .ok_or_else(|| Error::tensor("<unnamed>", "IQ scalar is invalid"))?;
    if *columns % block_values != 0 {
        return Err(Error::tensor(
            "<unnamed>",
            format!("IQ row width is not divisible by block length {block_values}"),
        ));
    }
    *columns = columns
        .checked_div(block_values)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(Error::Overflow("IQ packed row width"))?;
    Ok(packed)
}

pub(crate) fn mxfp4_shapes(desc: &TensorDescriptor) -> Result<(Vec<u64>, Vec<u64>)> {
    mxfp4_shapes_with_storage(desc.view(), None, None)
        .map(|(a, b)| (a.into_vec(), b.into_vec()))
        .map_err(ConversionDestinationError::ordinary)
}
fn mxfp4_shapes_with_storage<'a>(
    desc: TensorDescriptorView<'_>,
    first: Option<Slot<'a, u64>>,
    second: Option<Slot<'a, u64>>,
) -> CResult<(Vector<'a, u64>, Vector<'a, u64>)> {
    let mut weight_shape = shape(desc, first)?;
    let columns = weight_shape
        .last_mut()
        .ok_or_else(|| Error::tensor(desc.name, "MXFP4 scalar is invalid"))?;
    if *columns % 32 != 0 {
        return Err(Error::tensor(desc.name, "MXFP4 row width is not divisible by 32").into());
    }
    *columns /= 8;
    let mut scale_shape = shape(desc, second)?;
    *scale_shape.last_mut().unwrap() /= 32;
    Ok((weight_shape, scale_shape))
}

pub(crate) fn convert(
    desc: &TensorDescriptor,
    raw: &[u8],
    endian: Endian,
) -> Result<ConvertedTensor> {
    convert_view(desc.view(), raw, endian)
}
pub(crate) fn convert_view(
    desc: TensorDescriptorView<'_>,
    raw: &[u8],
    endian: Endian,
) -> Result<ConvertedTensor> {
    convert_with_storage(desc, raw, endian, Storage::ordinary())
        .map(ConversionOutput::into_owned)
        .map_err(ConversionDestinationError::ordinary)
}
fn convert_with_storage<'a>(
    desc: TensorDescriptorView<'_>,
    raw: &[u8],
    endian: Endian,
    storage: Storage<'a>,
) -> CResult<ConversionOutput<'a>> {
    if raw.len() as u64 != desc.byte_len {
        return Err(Error::tensor(desc.name, "payload length does not match descriptor").into());
    }
    let kind = conversion_kind(desc.ggml_type, endian)?;
    storage.validate(desc, endian)?;
    match kind {
        ConversionKind::Dense(dtype) => {
            let shape = shape(desc, storage.shape1)?;
            let data = normalize_dense_with_storage(raw, dtype, endian, storage.bytes)?;
            return Ok(ConversionOutput::Dense { shape, dtype, data });
        }
        ConversionKind::IQuant => {
            let shape = shape(desc, storage.shape1)?;
            let data = copied(raw, storage.bytes)?;
            return Ok(ConversionOutput::IQuant {
                shape,
                ggml_type: desc.ggml_type,
                endian,
                data,
            });
        }
        ConversionKind::MxFp4 => {
            return mxfp4_with_storage(desc, raw, storage).map(ConversionOutput::MxFp4);
        }
        ConversionKind::Affine { .. } => {}
    }
    affine_with_storage(desc, raw, endian, storage).map(ConversionOutput::Affine)
}

fn mxfp4_with_storage<'a>(
    desc: TensorDescriptorView<'_>,
    raw: &[u8],
    storage: Storage<'a>,
) -> CResult<MxFp4Output<'a>> {
    let (weight_shape, scale_shape) =
        mxfp4_shapes_with_storage(desc, storage.shape1, storage.shape2)?;
    let blocks = raw.len() / 17;
    let mut weights = capacity(blocks * 4, storage.words)?;
    let mut scales = capacity(blocks, storage.e8m0)?;
    for block in raw.as_chunks::<17>().0 {
        scales.push(block[0])?;
        let quants = &block[1..];
        let mut values = [0u8; 32];
        for index in 0..16 {
            values[index] = quants[index] & 0x0f;
            values[index + 16] = quants[index] >> 4;
        }
        for group in values.as_chunks::<8>().0 {
            weights.push(
                group
                    .iter()
                    .enumerate()
                    .fold(0u32, |packed, (index, value)| {
                        packed | (u32::from(*value) << (index * 4))
                    }),
            )?;
        }
    }
    Ok(MxFp4Output {
        weight_shape,
        scale_shape,
        weights,
        scales,
    })
}

fn dense_dtype(ty: GgmlType) -> Option<DenseDtype> {
    Some(match ty {
        GgmlType::F32 => DenseDtype::F32,
        GgmlType::F16 => DenseDtype::F16,
        GgmlType::Bf16 => DenseDtype::Bf16,
        GgmlType::I8 => DenseDtype::I8,
        GgmlType::I16 => DenseDtype::I16,
        GgmlType::I32 => DenseDtype::I32,
        GgmlType::I64 => DenseDtype::I64,
        GgmlType::F64 => DenseDtype::F64,
        _ => return None,
    })
}

fn normalize_dense_with_storage<'a>(
    raw: &[u8],
    dtype: DenseDtype,
    endian: Endian,
    slot: Option<Slot<'a, u8>>,
) -> CResult<Vector<'a, u8>> {
    let width = match dtype {
        DenseDtype::I8 => 1,
        DenseDtype::F16 | DenseDtype::Bf16 | DenseDtype::I16 => 2,
        DenseDtype::F32 | DenseDtype::I32 => 4,
        DenseDtype::I64 | DenseDtype::F64 => 8,
    };
    if width == 1
        || (cfg!(target_endian = "little") && endian == Endian::Little)
        || (cfg!(target_endian = "big") && endian == Endian::Big)
    {
        return copied(raw, slot);
    }
    let mut out = copied(raw, slot)?;
    for chunk in out.chunks_exact_mut(width) {
        chunk.reverse();
    }
    Ok(out)
}

/// Converts a GGML affine-compatible encoding into the canonical packed
/// weight, scale, and bias representation.
///
/// Native-capable formats normally bypass this expansion. This explicit
/// conversion remains available to portable backends and differential tests.
pub fn convert_affine(desc: &TensorDescriptor, raw: &[u8], endian: Endian) -> Result<AffineTensor> {
    if raw.len() as u64 != desc.byte_len {
        return Err(Error::tensor(
            &desc.name,
            "payload length does not match descriptor",
        ));
    }
    affine_with_storage(desc.view(), raw, endian, Storage::ordinary())
        .map(AffineOutput::finish)
        .map_err(ConversionDestinationError::ordinary)
}

fn affine_counts(weight_shape: &[u64], scale_shape: &[u64]) -> Result<(u64, u64)> {
    let groups = scale_shape.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b)
            .ok_or(Error::Overflow("affine group count"))
    })?;
    let words = weight_shape.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b).ok_or(Error::Overflow("affine word count"))
    })?;
    Ok((groups, words))
}
fn affine_with_storage<'a>(
    desc: TensorDescriptorView<'_>,
    raw: &[u8],
    endian: Endian,
    mut storage: Storage<'a>,
) -> CResult<AffineOutput<'a>> {
    let (bits, group_size) = affine_config(desc.ggml_type)
        .ok_or_else(|| Error::tensor(desc.name, "tensor encoding has no affine conversion"))?;
    let (weight_shape, scale_shape) =
        affine_shapes_with_storage(desc, bits, group_size, storage.shape1, storage.shape2)?;
    let (groups, words) = affine_counts(&weight_shape, &scale_shape)?;
    let mut out = AffineOutput {
        weight_shape,
        scale_shape,
        bits,
        group_size,
        weights: capacity(words as usize, storage.words)?,
        scales: capacity(groups as usize, storage.scales)?,
        biases: capacity(groups as usize, storage.biases)?,
    };
    match desc.ggml_type {
        GgmlType::Q4_0 => {
            for b in raw.as_chunks::<18>().0 {
                let d = half(b, endian);
                out.scales.push(d)?;
                out.biases.push(hbits(-8.0 * f16::from_bits(d).to_f32()))?;
                let mut codes = [0; 32];
                for i in 0..16 {
                    codes[i] = b[2 + i] & 15;
                    codes[16 + i] = b[2 + i] >> 4;
                }
                pack(&codes, 4, &mut out.weights)?;
            }
        }
        GgmlType::Q4_1 => {
            for b in raw.as_chunks::<20>().0 {
                out.scales.push(half(b, endian))?;
                out.biases.push(half(&b[2..], endian))?;
                let mut codes = [0; 32];
                for i in 0..16 {
                    codes[i] = b[4 + i] & 15;
                    codes[16 + i] = b[4 + i] >> 4;
                }
                pack(&codes, 4, &mut out.weights)?;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            q5(raw, endian, desc.ggml_type == GgmlType::Q5_0, &mut out)?
        }
        GgmlType::Q8_0 => {
            for b in raw.as_chunks::<34>().0 {
                let d = half(b, endian);
                out.scales.push(d)?;
                out.biases
                    .push(hbits(-128.0 * f16::from_bits(d).to_f32()))?;
                let codes = q8_codes(&b[2..], storage.scratch.as_deref_mut());
                pack(&codes, 8, &mut out.weights)?;
            }
        }
        GgmlType::Q4K | GgmlType::Q5K => {
            q45k(raw, endian, desc.ggml_type == GgmlType::Q5K, &mut out)?
        }
        GgmlType::Q6K => q6k(raw, endian, &mut out)?,
        GgmlType::Q2K => q2k(raw, endian, &mut out)?,
        GgmlType::Q3K => q3k(raw, endian, &mut out)?,
        _ => unreachable!(),
    }
    if out.weights.len() as u64 != words
        || out.scales.len() as u64 != groups
        || out.biases.len() != out.scales.len()
    {
        return Err(Error::tensor(
            desc.name,
            "conversion produced an inconsistent affine shape",
        )
        .into());
    }
    Ok(out)
}

fn half(b: &[u8], e: Endian) -> u16 {
    e.u16([b[0], b[1]])
}
fn hbits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn pack(codes: &[u8], bits: u8, out: &mut Vector<'_, u32>) -> CResult<()> {
    let words = (codes.len() * bits as usize).div_ceil(32);
    let start = out.len();
    out.resize(start + words, 0)?;
    let mask = (1u32 << bits) - 1;
    for (i, &c) in codes.iter().enumerate() {
        let off = i * bits as usize;
        let w = off / 32;
        let s = off % 32;
        out[start + w] |= ((c as u32) & mask) << s;
        if s + bits as usize > 32 {
            out[start + w + 1] |= (c as u32) >> (32 - s);
        }
    }
    Ok(())
}

fn scale_min(s: &[u8], i: usize) -> (u8, u8) {
    if i < 4 {
        (s[i] & 63, s[i + 4] & 63)
    } else {
        (
            (s[i + 4] & 15) | ((s[i - 4] >> 6) << 4),
            (s[i + 4] >> 4) | ((s[i] >> 6) << 4),
        )
    }
}
fn q45k(raw: &[u8], e: Endian, is_q5: bool, out: &mut AffineOutput<'_>) -> CResult<()> {
    let size = if is_q5 { 176 } else { 144 };
    for b in raw.chunks_exact(size) {
        let d = f16::from_bits(half(b, e)).to_f32();
        let dm = f16::from_bits(half(&b[2..], e)).to_f32();
        let s = &b[4..16];
        let qh = if is_q5 { Some(&b[16..48]) } else { None };
        let qs = if is_q5 { &b[48..] } else { &b[16..] };
        for g in 0..8 {
            let (sc, m) = scale_min(s, g);
            out.scales.push(hbits(d * sc as f32))?;
            out.biases.push(hbits(-dm * m as f32))?;
            let mut c = [0; 32];
            for i in 0..32 {
                let p = qs[(g / 2) * 32 + i];
                let lo = if g % 2 == 0 { p & 15 } else { p >> 4 };
                let hi = if qh.is_some_and(|h| h[i] & (1 << g) != 0) {
                    16
                } else {
                    0
                };
                c[i] = lo | hi;
            }
            pack(&c, if is_q5 { 5 } else { 4 }, &mut out.weights)?;
        }
    }
    Ok(())
}

fn q6k(raw: &[u8], e: Endian, out: &mut AffineOutput<'_>) -> CResult<()> {
    for b in raw.as_chunks::<210>().0 {
        let d = f16::from_bits(half(&b[208..], e)).to_f32();
        for section in 0..2 {
            let ql = &b[section * 64..];
            let qh = &b[128 + section * 32..];
            let scales = &b[192 + section * 8..];
            let mut vals = [0; 128];
            for i in 0..32 {
                vals[i] = (ql[i] & 15) | ((qh[i] & 3) << 4);
                vals[i + 32] = (ql[i + 32] & 15) | (((qh[i] >> 2) & 3) << 4);
                vals[i + 64] = (ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4);
                vals[i + 96] = (ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4);
            }
            for g in 0..8 {
                let sc = d * (scales[g] as i8) as f32;
                out.scales.push(hbits(sc))?;
                out.biases.push(hbits(-32.0 * sc))?;
                pack(&vals[g * 16..g * 16 + 16], 6, &mut out.weights)?;
            }
        }
    }
    Ok(())
}

fn q2k(raw: &[u8], e: Endian, out: &mut AffineOutput<'_>) -> CResult<()> {
    for b in raw.as_chunks::<84>().0 {
        let d = f16::from_bits(half(&b[80..], e)).to_f32();
        let dm = f16::from_bits(half(&b[82..], e)).to_f32();
        let mut all = [0; 256];
        for g in 0..16 {
            let s = b[g];
            out.scales.push(hbits(d * (s & 15) as f32))?;
            out.biases.push(hbits(-dm * (s >> 4) as f32))?;
            let qo = (g / 8) * 32 + (g % 2) * 16;
            let shift = ((g % 8) / 2) * 2;
            for i in 0..16 {
                all[g * 16 + i] = (b[16 + qo + i] >> shift) & 3;
            }
        }
        pack(&all, 2, &mut out.weights)?;
    }
    Ok(())
}

fn q3k(raw: &[u8], e: Endian, out: &mut AffineOutput<'_>) -> CResult<()> {
    for b in raw.as_chunks::<110>().0 {
        let hm = &b[..32];
        let qs = &b[32..96];
        let src = &b[96..108];
        let mut enc = [0u8; 16];
        for i in 0..4 {
            enc[i] = (src[i] & 15) | ((src[8 + i] & 3) << 4);
            enc[4 + i] = (src[4 + i] & 15) | (((src[8 + i] >> 2) & 3) << 4);
            enc[8 + i] = (src[i] >> 4) | (((src[8 + i] >> 4) & 3) << 4);
            enc[12 + i] = (src[4 + i] >> 4) | (((src[8 + i] >> 6) & 3) << 4);
        }
        let d = f16::from_bits(half(&b[108..], e)).to_f32();
        let mut all = [0; 256];
        for g in 0..16 {
            let sc = d * (enc[g] as i32 - 32) as f32;
            out.scales.push(hbits(sc))?;
            out.biases.push(hbits(-4.0 * sc))?;
            let qo = (g / 8) * 32 + (g % 2) * 16;
            let shift = ((g % 8) / 2) * 2;
            let mask = 1 << (g / 2);
            for i in 0..16 {
                all[g * 16 + i] = ((qs[qo + i] >> shift) & 3)
                    | if hm[(g % 2) * 16 + i] & mask != 0 {
                        4
                    } else {
                        0
                    };
            }
        }
        pack(&all, 3, &mut out.weights)?;
    }
    Ok(())
}

fn q5(raw: &[u8], e: Endian, is_q5_0: bool, out: &mut AffineOutput<'_>) -> CResult<()> {
    let size = if is_q5_0 { 22 } else { 24 };
    for b in raw.chunks_exact(size) {
        let d = half(b, e);
        out.scales.push(d)?;
        if is_q5_0 {
            out.biases.push(hbits(-16.0 * f16::from_bits(d).to_f32()))?;
        } else {
            out.biases.push(half(&b[2..], e))?;
        }
        let qh_off = if is_q5_0 { 2 } else { 4 };
        let qs_off = if is_q5_0 { 6 } else { 8 };
        let qh = e.u32(b[qh_off..qh_off + 4].try_into().unwrap());
        let mut codes = [0u8; 32];
        for j in 0..16 {
            codes[j] = (b[qs_off + j] & 15) | (((qh >> j) as u8 & 1) << 4);
            codes[j + 16] = (b[qs_off + j] >> 4) | (((qh >> (j + 16)) as u8 & 1) << 4);
        }
        pack(&codes, 5, &mut out.weights)?;
    }
    Ok(())
}

mod parts;
pub use parts::{packed_iquant_shape, ConvertedParts};
