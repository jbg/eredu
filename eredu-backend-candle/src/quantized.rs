//! Native execution of admitted packed matrix parameters.

use std::{borrow::Cow, sync::Arc};

use candle_core::{
    quantized::{
        k_quants::{BlockQ4K, BlockQ5K, BlockQ5_0, BlockQ6K, BlockQ8_0},
        GgmlDType, QStorage, QTensor,
    },
    CpuStorage, DType, Device, Storage, Tensor as NativeTensor,
};
use eredu_checkpoint::{Endian, GgmlType, LinearFormat};
use eredu_gguf::packed_dot::PackedDotActivations;
use eredu_nn::{Error, LinearFormatSpec, Tensor};
use rayon::prelude::*;

use crate::{tensor::rounded_bf16_binary, CandleContext, CandleTensor};

/// The encoding of exact matrix bytes and their native realization on CUDA.
#[derive(Clone, Debug)]
pub(crate) struct PackedParameter {
    format: GgmlType,
    cuda: Option<Arc<QTensor>>,
}

pub(crate) fn import_matrix_bytes(
    bytes: &[u8],
    shape: &[usize],
    format: GgmlType,
    device: &Device,
) -> Result<CandleTensor, Error> {
    let [rows, row_bytes] = *shape else {
        return Err(Error::backend("packed matrix requires two physical axes"));
    };
    if rows.checked_mul(row_bytes) != Some(bytes.len()) {
        return Err(Error::backend(
            "packed matrix bytes differ from its binding extent",
        ));
    }
    let mut value = CandleTensor::wrap(if device.is_cuda() {
        // The tensor describes the binding geometry. The QTensor owns the
        // encoded allocation and is the only input used by CUDA kernels.
        packed_slot(shape, device)
    } else {
        NativeTensor::from_slice(bytes, shape, device)
    })?;
    let cuda = device
        .is_cuda()
        .then(|| bind_native_blocks(bytes, format, rows, row_bytes, device).map(Arc::new))
        .transpose()
        .map_err(Error::backend_source)?;
    value.packed = Some(PackedParameter { format, cuda });
    Ok(value)
}

pub(crate) fn matches_binding(slot: &CandleTensor, weight: &CandleTensor) -> bool {
    !slot.value.device().is_cuda()
        || slot.packed.as_ref().is_none_or(|expected| {
            weight
                .packed
                .as_ref()
                .is_none_or(|actual| actual.format == expected.format && actual.cuda.is_some())
        })
}

pub(crate) fn upload_matrix(
    value: &CandleTensor,
    device: &Device,
) -> Result<Option<CandleTensor>, Error> {
    let Some(parameter) = value.packed.as_ref().filter(|_| device.is_cuda()) else {
        return Ok(None);
    };
    let (storage, layout) = value.value.storage_and_layout();
    let (Storage::Cpu(CpuStorage::U8(bytes)), Some((start, end))) =
        (&*storage, layout.contiguous_offsets())
    else {
        return Err(Error::backend(
            "packed CUDA upload requires contiguous CPU bytes",
        ));
    };
    let bytes = bytes
        .get(start..end)
        .ok_or_else(|| Error::backend("packed CUDA byte view exceeds its storage"))?;
    import_matrix_bytes(bytes, value.value.dims(), parameter.format, device).map(Some)
}

fn bind_native_blocks(
    bytes: &[u8],
    format: GgmlType,
    rows: usize,
    row_bytes: usize,
    device: &Device,
) -> candle_core::Result<QTensor> {
    let Some((dtype, alignment)) = native_block(format) else {
        candle_core::bail!("unsupported native packed weights");
    };
    if !row_bytes.is_multiple_of(dtype.type_size())
        || !bytes.as_ptr().addr().is_multiple_of(alignment)
        || rows.checked_mul(row_bytes) != Some(bytes.len())
    {
        candle_core::bail!("invalid native packed block length or alignment");
    }
    let columns = (row_bytes / dtype.type_size())
        .checked_mul(dtype.block_size())
        .ok_or_else(|| candle_core::Error::Msg("native packed width overflows".into()))?;
    let storage = QStorage::from_data(Cow::Borrowed(bytes), device, dtype)?;
    let tensor = QTensor::new(storage, (rows, columns))?;
    if tensor.storage_size_in_bytes() != bytes.len() {
        candle_core::bail!("native packed allocation differs from its binding extent");
    }
    Ok(tensor)
}

fn packed_slot(shape: &[usize], device: &Device) -> candle_core::Result<NativeTensor> {
    NativeTensor::from_vec(vec![0u8], (), device)?.broadcast_as(shape)
}

/// The logical and encoded geometry of a packed matrix.
#[derive(Clone, Debug)]
pub(crate) struct PackedMatrix {
    rows: usize,
    columns: usize,
    row_bytes: usize,
    ggml_type: GgmlType,
}

fn native_block(ggml_type: GgmlType) -> Option<(GgmlDType, usize)> {
    use std::mem::align_of;
    Some(match ggml_type {
        GgmlType::Q5_0 => (GgmlDType::Q5_0, align_of::<BlockQ5_0>()),
        GgmlType::Q8_0 => (GgmlDType::Q8_0, align_of::<BlockQ8_0>()),
        GgmlType::Q5K => (GgmlDType::Q5K, align_of::<BlockQ5K>()),
        GgmlType::Q4K => (GgmlDType::Q4K, align_of::<BlockQ4K>()),
        GgmlType::Q6K => (GgmlDType::Q6K, align_of::<BlockQ6K>()),
        _ => return None,
    })
}

pub(crate) fn supports_packed(ggml_type: GgmlType) -> bool {
    cfg!(target_endian = "little")
        && eredu_gguf::BlockDecoder::new(ggml_type, Endian::Little).is_ok()
}

pub(crate) fn supports_packed_matrix(
    placement: crate::CandlePlacement,
    ggml_type: GgmlType,
    rank: usize,
) -> bool {
    supports_packed(ggml_type)
        && match placement {
            crate::CandlePlacement::Cpu | crate::CandlePlacement::Metal(_) => matches!(rank, 2 | 3),
            crate::CandlePlacement::Cuda(_) => {
                cfg!(feature = "cuda") && matches!(rank, 2 | 3) && native_block(ggml_type).is_some()
            }
        }
}

pub(crate) fn supports_activation_for_encoding(
    placement: crate::CandlePlacement,
    name: &str,
) -> bool {
    // The artifact contract reports GGML encodings by their canonical enum name.
    const BLOCK_FORMATS: [GgmlType; 21] = [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ3XXS,
        GgmlType::IQ1S,
        GgmlType::IQ4NL,
        GgmlType::IQ3S,
        GgmlType::IQ2S,
        GgmlType::IQ4XS,
        GgmlType::IQ1M,
        GgmlType::MxFp4,
        GgmlType::NvFp4,
    ];
    BLOCK_FORMATS
        .into_iter()
        .any(|format| supports_packed_matrix(placement, format, 2) && name == format!("{format:?}"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn optimized_metal(format: GgmlType) -> bool {
    matches!(
        format,
        GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    )
}

/// Clips dense scratch to a bounded output-row tile. The packed tensor remains
/// the resident authority for the whole projection.
pub(crate) fn project_decoded_tiles(
    input: &NativeTensor,
    rows: usize,
    columns: usize,
    mut decode: impl FnMut(std::ops::Range<usize>) -> candle_core::Result<NativeTensor>,
) -> candle_core::Result<NativeTensor> {
    project_row_tiles(input, rows, columns, |flat, range| {
        flat.matmul(&decode(range)?.t()?)
    })
}

pub(crate) fn project_row_tiles(
    input: &NativeTensor,
    rows: usize,
    columns: usize,
    mut project: impl FnMut(&NativeTensor, std::ops::Range<usize>) -> candle_core::Result<NativeTensor>,
) -> candle_core::Result<NativeTensor> {
    let token_count = input.elem_count() / columns;
    let mut shape = input.dims().to_vec();
    *shape.last_mut().expect("projection has a feature axis") = rows;
    if token_count == 0 {
        return NativeTensor::zeros(shape, DType::F32, input.device());
    }
    let flat = input.reshape((token_count, columns))?;
    let step = (65_536 / columns).max(1);
    let mut tiles = Vec::new();
    for start in (0..rows).step_by(step) {
        tiles.push(project(&flat, start..(start + step).min(rows))?);
    }
    NativeTensor::cat(&tiles, 1)?.reshape(shape)
}

pub(crate) fn read_tile<T: candle_core::WithDType>(
    value: &NativeTensor,
) -> candle_core::Result<Vec<T>> {
    let flat = value.flatten_all()?;
    if value.device().is_cpu() {
        flat.to_vec1()
    } else {
        // Read back only this tile rather than its whole backing bank.
        flat.force_contiguous()?.to_vec1()
    }
}

pub(crate) fn decode_block_rows(
    weight: &NativeTensor,
    rows: usize,
    columns: usize,
    format: GgmlType,
    endian: Endian,
) -> candle_core::Result<NativeTensor> {
    let decoder =
        eredu_gguf::BlockDecoder::new(format, endian).map_err(candle_core::Error::wrap)?;
    let bytes = read_tile::<u8>(weight)?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("decoded matrix tile extent overflows".into()))?;
    let mut decoded = vec![0f32; count];
    decoder
        .decode_into(&bytes, &mut decoded)
        .map_err(candle_core::Error::wrap)?;
    NativeTensor::from_vec(decoded, (rows, columns), weight.device())
}

pub(crate) fn matrix_slot(
    format: &LinearFormatSpec,
    rows: i32,
    columns: i32,
    context: &CandleContext,
) -> Result<(CandleTensor, Option<PackedMatrix>), Error> {
    format.validate()?;
    if rows <= 0 || columns <= 0 {
        return Err(Error::backend("invalid native matrix geometry"));
    }
    match format.encoding() {
        LinearFormat::Dense => Ok((CandleTensor::unloaded_f32(&[rows, columns], context)?, None)),
        encoding @ (LinearFormat::Affine(_) | LinearFormat::MxFp4)
            if context.placement().supports_word_matrix(2) =>
        {
            let width = encoding
                .weight_quantization()
                .expect("packed word encoding")
                .packed_word_geometry(columns as usize)
                .map_err(Error::backend_source)?
                .word_columns();
            Ok((
                CandleTensor::wrap(
                    NativeTensor::zeros((), DType::U32, &context.device)
                        .and_then(|value| value.broadcast_as((rows as usize, width))),
                )?,
                None,
            ))
        }
        encoding @ (LinearFormat::NvFp4 | LinearFormat::E4M3BlockFp8(_))
            if context.placement().supports_local_text_extensions() =>
        {
            let (width, dtype) = match encoding {
                LinearFormat::NvFp4 if columns % 16 == 0 => (columns / 8, DType::U32),
                LinearFormat::E4M3BlockFp8(_) => (columns, DType::U8),
                _ => return Err(Error::backend("block projection has an unaligned row")),
            };
            Ok((
                CandleTensor::wrap(
                    NativeTensor::zeros((), dtype, &context.device)
                        .and_then(|value| value.broadcast_as((rows as usize, width as usize))),
                )?,
                None,
            ))
        }
        LinearFormat::GgufIQuant {
            ggml_type,
            endian: Endian::Little,
        } if supports_packed_matrix(context.placement(), ggml_type, 2) => {
            let (block, bytes) = ggml_type.block_and_bytes().map_err(Error::backend_source)?;
            let (rows, columns) = (rows as usize, columns as usize);
            if !columns.is_multiple_of(block as usize) {
                return Err(Error::backend(
                    "native quantized matrix requires whole blocks per row",
                ));
            }
            let row_bytes = columns / block as usize * bytes as usize;
            i32::try_from(row_bytes).map_err(Error::backend_source)?;
            let mut slot = CandleTensor::wrap(packed_slot(&[rows, row_bytes], &context.device))?;
            if context.device.is_cuda() {
                slot.packed = Some(PackedParameter {
                    format: ggml_type,
                    cuda: None,
                });
            }
            Ok((
                slot,
                Some(PackedMatrix {
                    rows,
                    columns,
                    row_bytes,
                    ggml_type,
                }),
            ))
        }
        _ => Err(Error::backend("unsupported native matrix encoding")),
    }
}

impl PackedMatrix {
    fn with_blocks<R>(
        &self,
        value: &NativeTensor,
        rows: usize,
        use_blocks: impl FnOnce(&[u8]) -> candle_core::Result<R>,
    ) -> candle_core::Result<R> {
        if value.dims() != [rows, self.row_bytes] || !value.device().is_cpu() {
            candle_core::bail!("native packed parameter geometry differs from its slot");
        }
        let (storage, layout) = value.storage_and_layout();
        let (Storage::Cpu(CpuStorage::U8(bytes)), Some((start, end))) =
            (&*storage, layout.contiguous_offsets())
        else {
            candle_core::bail!("native packed parameter requires contiguous CPU bytes");
        };
        let Some(blocks) = bytes.get(start..end) else {
            candle_core::bail!("native packed parameter exceeds its storage");
        };
        use_blocks(blocks)
    }

    fn native_weights(&self, value: &NativeTensor, rows: usize) -> candle_core::Result<QTensor> {
        self.with_blocks(value, rows, |blocks| {
            bind_native_blocks(blocks, self.ggml_type, rows, self.row_bytes, value.device())
        })
    }

    pub(crate) fn linear(
        &self,
        input: &CandleTensor,
        weight: &CandleTensor,
        bias: Option<&CandleTensor>,
    ) -> Result<CandleTensor, Error> {
        self.project(input, weight, bias, true)
    }

    pub(crate) fn grouped_linear(
        &self,
        input: &CandleTensor,
        weight: &CandleTensor,
        bias: Option<&CandleTensor>,
    ) -> Result<CandleTensor, Error> {
        self.project(input, weight, bias, false)
    }

    fn project(
        &self,
        input: &CandleTensor,
        weight: &CandleTensor,
        bias: Option<&CandleTensor>,
        matrix_tiles: bool,
    ) -> Result<CandleTensor, Error> {
        CandleTensor::wrap((|| {
            if input.value.dim(candle_core::D::Minus1)? != self.columns {
                candle_core::bail!("native quantized input width differs from its weight");
            }
            let dtype = input.value.dtype();
            let input = input.value.to_dtype(DType::F32)?.contiguous()?;
            self.check_weight(weight, input.device())?;
            let output = if let Some(weights) = self.cuda_weights(weight) {
                let rows = input.elem_count() / self.columns;
                let mut shape = input.dims().to_vec();
                *shape.last_mut().expect("projection has a feature axis") = self.rows;
                if rows == 0 {
                    NativeTensor::zeros(shape, DType::F32, input.device())?
                } else {
                    input
                        .reshape((rows, self.columns))?
                        .apply_op1_no_bwd(weights)?
                        .reshape(shape)?
                }
            } else {
                self.project_native(&input, &weight.value, matrix_tiles)?
            }
            .to_dtype(dtype)?;
            match bias {
                Some(bias) => {
                    rounded_bf16_binary(&output, &bias.value, NativeTensor::broadcast_add)
                }
                None => Ok(output),
            }
        })())
    }

    fn check_weight(&self, weight: &CandleTensor, device: &Device) -> candle_core::Result<()> {
        if weight.value.dims() != [self.rows, self.row_bytes]
            || weight.value.dtype() != DType::U8
            || !weight.value.device().same_device(device)
        {
            candle_core::bail!("native packed parameter geometry differs from its slot");
        }
        Ok(())
    }

    fn cuda_weights<'a>(&self, weight: &'a CandleTensor) -> Option<&'a QTensor> {
        weight
            .packed
            .as_ref()
            .filter(|parameter| parameter.format == self.ggml_type)
            .and_then(|parameter| parameter.cuda.as_deref())
    }

    fn project_native(
        &self,
        input: &NativeTensor,
        weight: &NativeTensor,
        matrix_tiles: bool,
    ) -> candle_core::Result<NativeTensor> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if input.device().is_metal() && optimized_metal(self.ggml_type) {
            return crate::metal_quantized::project(
                input,
                weight,
                self.ggml_type,
                self.columns,
                self.rows,
                matrix_tiles,
            );
        }

        if input.device().is_cuda() {
            return project_row_tiles(input, self.rows, self.columns, |flat, rows| {
                let block = weight.narrow(0, rows.start, rows.len())?;
                let bytes = read_tile::<u8>(&block)?;
                let native = bind_native_blocks(
                    &bytes,
                    self.ggml_type,
                    rows.len(),
                    self.row_bytes,
                    input.device(),
                )?;
                let output = flat.apply_op1_no_bwd(&native);
                if let Err(error) = CandleContext::synchronize_device(input.device()) {
                    CandleContext::quarantine_device(
                        input.device(),
                        (native, flat.clone(), output.ok()),
                    );
                    return Err(error);
                }
                output
            });
        }

        if self.ggml_type == GgmlType::NvFp4 {
            if input.device().is_cpu() {
                return self.packed_matmul(input, weight, false);
            }
            let host = input.force_contiguous()?.to_device(&Device::Cpu)?;
            let values = host.flatten_all()?.to_vec1::<f32>()?;
            let activations = PackedDotActivations::new(self.ggml_type, self.columns, &values)
                .map_err(candle_core::Error::wrap)?;
            return project_row_tiles(input, self.rows, self.columns, |flat, range| {
                let bytes = read_tile::<u8>(&weight.narrow(0, range.start, range.len())?)?;
                let width = range.len();
                let row_count = flat.dim(0)?;
                let mut values = vec![0f32; row_count * width];
                values
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(i, value)| {
                        let packed = &bytes[i % width * self.row_bytes..][..self.row_bytes];
                        *value = activations
                            .dot(i / width, packed, false)
                            .map_err(candle_core::Error::wrap)?;
                        Ok::<_, candle_core::Error>(())
                    })?;
                NativeTensor::from_vec(values, (row_count, width), input.device())
            });
        }

        if native_block(self.ggml_type).is_none() || !input.device().is_cpu() {
            return project_decoded_tiles(input, self.rows, self.columns, |rows| {
                let block = weight.narrow(0, rows.start, rows.len())?;
                decode_block_rows(
                    &block,
                    rows.len(),
                    self.columns,
                    self.ggml_type,
                    Endian::Little,
                )
            });
        }

        // The ARM K-block reduction applies to the eight-output packed kernels.
        let arm = cfg!(target_arch = "aarch64");
        let arm_k = arm && self.rows.is_multiple_of(8);
        let ordered_dot = match self.ggml_type {
            GgmlType::Q8_0 => true,
            GgmlType::Q5_0 => arm,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => arm_k,
            _ => false,
        };
        if ordered_dot {
            self.packed_matmul(input, weight, matrix_tiles)
        } else {
            input.apply_op1_no_bwd(&self.native_weights(weight, self.rows)?)
        }
    }

    fn packed_matmul(
        &self,
        input: &NativeTensor,
        weight: &NativeTensor,
        matrix_tiles: bool,
    ) -> candle_core::Result<NativeTensor> {
        let (storage, layout) = input.storage_and_layout();
        let (Storage::Cpu(CpuStorage::F32(values)), Some((start, end))) =
            (&*storage, layout.contiguous_offsets())
        else {
            candle_core::bail!("native packed projection requires contiguous CPU activations");
        };
        let activations =
            PackedDotActivations::new(self.ggml_type, self.columns, &values[start..end])
                .map_err(candle_core::Error::wrap)?;
        let rows = input.elem_count() / self.columns;
        let batched_rows = if matrix_tiles { rows / 4 * 4 } else { 0 };
        let count = rows.checked_mul(self.rows).ok_or_else(|| {
            candle_core::Error::Msg("native packed projection output size overflows".into())
        })?;
        let mut result = vec![0.0; count];
        self.with_blocks(weight, self.rows, |blocks| {
            result
                .par_iter_mut()
                .with_min_len(256)
                .enumerate()
                .try_for_each(|(i, out)| {
                    let row = i / self.rows;
                    let col = i % self.rows;
                    let weight = &blocks[col * self.row_bytes..][..self.row_bytes];
                    *out = activations
                        .dot(row, weight, row < batched_rows)
                        .map_err(candle_core::Error::wrap)?;
                    Ok(())
                })
        })?;
        let mut shape = input.dims().to_vec();
        *shape
            .last_mut()
            .expect("projection input has a feature axis") = self.rows;
        NativeTensor::from_vec(result, shape, input.device())
    }

    pub(crate) fn embed(
        &self,
        input: &CandleTensor,
        weight: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        self.check_weight(weight, &context.device)
            .map_err(Error::backend_source)?;
        if let Some(weight) = self.cuda_weights(weight) {
            return CandleTensor::wrap((|| {
                let ids = crate::tensor::checked_gather_indices(&input.value, self.rows as i64)?;
                let ids = ids
                    .into_iter()
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                if ids.is_empty() {
                    let mut shape = input.value.dims().to_vec();
                    shape.push(self.columns);
                    return NativeTensor::zeros(shape, DType::F32, &context.device);
                }
                let ids = NativeTensor::from_vec(ids, input.value.shape(), &context.device)?;
                weight.embedding(&ids)
            })());
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if context.placement().is_metal() && optimized_metal(self.ggml_type) {
            let ids = crate::tensor::checked_gather_indices(&input.value, self.rows as i64)
                .map_err(Error::backend_source)?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(Error::backend_source)?;
            let mut shape = input.value.dims().to_vec();
            shape.push(self.columns);
            return CandleTensor::wrap(crate::metal_quantized::embedding(
                &weight.value,
                &ids,
                self.ggml_type,
                shape,
                self.columns,
                self.row_bytes,
            ));
        }
        let ids = crate::tensor::checked_gather_indices(&input.value, self.rows as i64)
            .map_err(Error::backend_source)?;
        let mut shape = input.value.dims().to_vec();
        shape.push(self.columns);
        if ids.is_empty() {
            return CandleTensor::wrap(NativeTensor::zeros(shape, DType::F32, &context.device));
        }
        if !context.device.is_cpu() {
            return CandleTensor::wrap((|| {
                let ids = ids
                    .into_iter()
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                let count = ids.len();
                let ids = NativeTensor::new(ids, &context.device)?;
                let selected = weight.value.index_select(&ids, 0)?;
                decode_block_rows(
                    &selected,
                    count,
                    self.columns,
                    self.ggml_type,
                    Endian::Little,
                )?
                .reshape(shape)
            })());
        }
        let decoder = eredu_gguf::BlockDecoder::new(self.ggml_type, Endian::Little)
            .map_err(Error::backend_source)?;
        let count = ids
            .len()
            .checked_mul(self.columns)
            .ok_or_else(|| Error::backend("decoded embedding size overflows"))?;
        let mut output = vec![0.0; count];
        if weight.value.is_contiguous() {
            self.with_blocks(&weight.value, self.rows, |encoded| {
                let rows = decoder
                    .packed_rows(encoded, self.rows, self.columns)
                    .map_err(candle_core::Error::wrap)?;
                for (id, row) in ids.into_iter().zip(output.chunks_exact_mut(self.columns)) {
                    rows.decode_row(id as usize, row)
                        .map_err(candle_core::Error::wrap)?;
                }
                Ok(())
            })
            .map_err(Error::backend_source)?;
        } else {
            // Materialize at most one encoded row when the view is strided.
            for (id, row) in ids.into_iter().zip(output.chunks_exact_mut(self.columns)) {
                let encoded = weight
                    .value
                    .narrow(0, id as usize, 1)
                    .and_then(|row| row.contiguous())
                    .map_err(Error::backend_source)?;
                self.with_blocks(&encoded, 1, |encoded| {
                    decoder
                        .decode_into(encoded, row)
                        .map_err(candle_core::Error::wrap)
                })
                .map_err(Error::backend_source)?;
            }
        }
        CandleTensor::wrap(NativeTensor::from_vec(output, shape, &context.device))
    }
}
