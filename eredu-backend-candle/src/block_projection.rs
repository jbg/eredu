//! Bounded block-float projections over native tensor tiles.

use std::ops::Range;

use candle_core::{DType, Tensor as NativeTensor};
use eredu_checkpoint::{
    fp8, nvfp4::NvFp4MatrixFormat, BlockFp8Format, BlockFp8ScaleEncoding, GgmlType, LinearFormat,
};
use eredu_nn::{Error, LinearFormatSpec, Parameter, Parameterized, Tensor};

use crate::{
    quantized::{self, read_tile},
    tensor::rounded_bf16_binary,
    CandleContext, CandleTensor,
};

#[derive(Debug, Clone, Copy)]
enum Encoding {
    Fp8(BlockFp8Format),
    NvFp4,
    GgufNvFp4,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub(crate) struct BlockProjection {
    scales: Option<Parameter<CandleTensor>>,
    global: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    encoding: Encoding,
    #[parameter(skip)]
    rows: usize,
    #[parameter(skip)]
    columns: usize,
    #[parameter(skip)]
    rows_per_partition: usize,
}

impl BlockProjection {
    pub(crate) fn new(
        spec: &LinearFormatSpec,
        rows: i32,
        columns: i32,
        groups: Option<i32>,
        context: &CandleContext,
    ) -> Result<Option<Self>, Error> {
        let (rows, columns) = (rows as usize, columns as usize);
        let partitions = spec.row_layout().partitions();
        let rows_per_partition = spec.row_layout().rows_per_partition(rows)?;
        let (encoding, scale) = match spec.encoding() {
            LinearFormat::NvFp4 => {
                let geometry = NvFp4MatrixFormat::new(rows, columns, partitions)
                    .map_err(Error::backend_source)?;
                (
                    Encoding::NvFp4,
                    Some((geometry.block_scale_shape(), DType::U8)),
                )
            }
            LinearFormat::E4M3BlockFp8(format) => {
                let scale_rows = spec
                    .row_layout()
                    .scale_rows(rows, format.block_rows as usize)?;
                let scale_columns = columns.div_ceil(format.block_columns as usize);
                let dtype = match format.scale_encoding {
                    BlockFp8ScaleEncoding::FloatingPoint => DType::F32,
                    BlockFp8ScaleEncoding::Ue8m0 => DType::U8,
                };
                (
                    Encoding::Fp8(format),
                    Some(([scale_rows, scale_columns], dtype)),
                )
            }
            LinearFormat::GgufIQuant {
                ggml_type: GgmlType::NvFp4,
                ..
            } if spec.global_scale().is_some() && partitions == 1 => (Encoding::GgufNvFp4, None),
            LinearFormat::GgufIQuant {
                ggml_type: GgmlType::NvFp4,
                ..
            } if spec.global_scale().is_some() => {
                return Err(Error::backend("scaled GGUF NVFP4 requires contiguous rows"))
            }
            _ => return Ok(None),
        };
        let slot = |parameter: &eredu_nn::ParameterSpec, mut shape: Vec<usize>, dtype| {
            if let Some(groups) = groups {
                shape.insert(0, groups as usize);
            }
            CandleTensor::wrap(
                NativeTensor::zeros((), dtype, &context.device)
                    .and_then(|value| value.broadcast_as(shape.as_slice())),
            )
            .map(|value| Parameter::new(parameter.clone(), value))
        };
        let scales = scale
            .map(|(shape, dtype)| {
                let spec = spec
                    .scale()
                    .ok_or_else(|| Error::backend("missing block scale declaration"))?;
                slot(spec, shape.to_vec(), dtype)
            })
            .transpose()?;
        let global = spec
            .global_scale()
            .map(|p| {
                let shape = if partitions == 1 {
                    Vec::new()
                } else {
                    vec![partitions]
                };
                slot(p, shape, DType::F32)
            })
            .transpose()?;
        Ok(Some(Self {
            scales,
            global,
            encoding,
            rows,
            columns,
            rows_per_partition,
        }))
    }

    pub(crate) fn input(
        &self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<Option<CandleTensor>, Error> {
        let Encoding::Fp8(format) = self.encoding else {
            return Ok(None);
        };
        if input.value.dims().last() != Some(&self.columns) {
            return Err(Error::backend(
                "block-FP8 activation width differs from its weights",
            ));
        }
        let mut values = input.to_f32_vec(context)?;
        eredu_nn::block_fp8::round_activation_rows(
            &mut values,
            self.columns,
            format.block_columns as usize,
        )?;
        CandleTensor::wrap(NativeTensor::from_vec(
            values,
            input.value.shape(),
            &context.device,
        ))
        .map(Some)
    }

    pub(crate) fn project(
        &self,
        input: &CandleTensor,
        quantized_input: Option<&CandleTensor>,
        weight: &CandleTensor,
        bias: Option<&CandleTensor>,
        packed: Option<&quantized::PackedMatrix>,
        group: Option<usize>,
    ) -> Result<CandleTensor, Error> {
        let dtype = input.value.dtype();
        CandleTensor::wrap((|| {
            if input.value.dims().last() != Some(&self.columns) {
                candle_core::bail!("block projection activation width differs from weights");
            }
            let (scales, globals) = self.companions(group)?;
            let output = match self.encoding {
                Encoding::GgufNvFp4 => {
                    let packed = packed.ok_or_else(|| {
                        candle_core::Error::Msg("missing GGUF NVFP4 geometry".into())
                    })?;
                    let wide = CandleTensor::wrap(input.value.to_dtype(DType::F32))
                        .map_err(candle_core::Error::wrap)?;
                    let projected = packed
                        .linear(&wide, weight, None)
                        .map_err(candle_core::Error::wrap)?;
                    projected.value.affine(f64::from(globals[0]), 0.)?
                }
                encoding => {
                    let scales = scales
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("missing block scale".into()))?;
                    let source = &quantized_input.unwrap_or(input).value;
                    quantized::project_row_tiles(source, self.rows, self.columns, |flat, rows| {
                        let decoded = self.decode_rows(&weight.value, scales, &globals, rows)?;
                        match encoding {
                            Encoding::NvFp4 => {
                                crate::matmul::project(flat, &decoded.to_dtype(dtype)?)
                            }
                            _ => flat.matmul(&decoded.t()?),
                        }
                    })?
                }
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

    pub(crate) fn embed(
        &self,
        ids: &CandleTensor,
        weight: &CandleTensor,
        packed: Option<&quantized::PackedMatrix>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        if matches!(self.encoding, Encoding::GgufNvFp4) {
            let matrix = packed.ok_or_else(|| Error::backend("missing GGUF NVFP4 geometry"))?;
            let value = matrix.embed(ids, weight, context)?;
            let (_, scales) = self.companions(None).map_err(Error::backend_source)?;
            return CandleTensor::wrap(value.value.affine(scales[0].into(), 0.));
        }
        CandleTensor::wrap((|| {
            let indices = crate::tensor::checked_gather_indices(&ids.value, self.rows as i64)?;
            let (scales, globals) = self.companions(None)?;
            let scales = scales
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing block scale".into()))?;
            let mut shape = ids.value.dims().to_vec();
            shape.push(self.columns);
            if indices.is_empty() {
                return NativeTensor::zeros(shape, DType::F32, &context.device);
            }
            let rows = indices
                .into_iter()
                .map(|row| {
                    self.decode_rows(
                        &weight.value,
                        scales,
                        &globals,
                        row as usize..row as usize + 1,
                    )
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            NativeTensor::cat(&rows, 0)?.reshape(shape)
        })())
    }

    fn companions(
        &self,
        group: Option<usize>,
    ) -> candle_core::Result<(Option<NativeTensor>, Vec<f32>)> {
        let select = |parameter: &Parameter<CandleTensor>| {
            let value = &parameter.as_ref().value;
            match group {
                Some(group) => value.narrow(0, group, 1)?.squeeze(0),
                None => Ok(value.clone()),
            }
        };
        let scales = self.scales.as_ref().map(select).transpose()?;
        let global = self.global.as_ref().map(select).transpose()?;
        let global = global
            .as_ref()
            .map(read_tile::<f32>)
            .transpose()?
            .unwrap_or_default();
        if self.global.is_some()
            && (global.len() != self.rows / self.rows_per_partition
                || global.iter().any(|x| !x.is_finite() || *x <= 0.))
        {
            candle_core::bail!("invalid block projection global scale");
        }
        Ok((scales, global))
    }

    fn decode_rows(
        &self,
        weight: &NativeTensor,
        scale: &NativeTensor,
        globals: &[f32],
        rows: Range<usize>,
    ) -> candle_core::Result<NativeTensor> {
        let tile = weight.narrow(0, rows.start, rows.len())?;
        let values = match self.encoding {
            Encoding::Fp8(format) => self.decode_fp8(&tile, scale, format, rows.clone())?,
            Encoding::NvFp4 => {
                let bytes: Vec<_> = read_tile::<u32>(&tile)?
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect();
                let scales = read_tile::<u8>(&scale.narrow(0, rows.start, rows.len())?)?;
                let mut values = Vec::with_capacity(rows.len() * self.columns);
                let mut start = rows.start;
                while start < rows.end {
                    let partition = start / self.rows_per_partition;
                    let end = rows.end.min((partition + 1) * self.rows_per_partition);
                    let local = start - rows.start..end - rows.start;
                    let slice = |width| local.start * width..local.end * width;
                    let geometry = NvFp4MatrixFormat::new(local.len(), self.columns, 1)
                        .map_err(candle_core::Error::wrap)?;
                    values.extend(
                        geometry
                            .decode_rows(
                                &bytes[slice(self.columns / 2)],
                                &scales[slice(self.columns / 16)],
                                &globals[partition..=partition],
                                0..local.len(),
                                local.len() * self.columns * 4,
                            )
                            .map_err(candle_core::Error::wrap)?,
                    );
                    start = end;
                }
                values
            }
            Encoding::GgufNvFp4 => unreachable!("GGUF NVFP4 uses the encoded dot"),
        };
        NativeTensor::from_vec(values, (rows.len(), self.columns), weight.device())
    }

    fn decode_fp8(
        &self,
        tile: &NativeTensor,
        scale: &NativeTensor,
        format: BlockFp8Format,
        rows: Range<usize>,
    ) -> candle_core::Result<Vec<f32>> {
        let block_rows = format.block_rows as usize;
        let block_columns = format.block_columns as usize;
        let partition_scales = self.rows_per_partition.div_ceil(block_rows);
        let scale_index = |row| {
            row / self.rows_per_partition * partition_scales
                + row % self.rows_per_partition / block_rows
        };
        let scale_columns = self.columns.div_ceil(block_columns);
        let first = scale_index(rows.start);
        let end = scale_index(rows.end - 1) + 1;
        let scale = scale.narrow(0, first, end - first)?;
        let scales = match format.scale_encoding {
            BlockFp8ScaleEncoding::FloatingPoint => read_tile::<f32>(&scale.to_dtype(DType::F32)?)?,
            BlockFp8ScaleEncoding::Ue8m0 => read_tile::<u8>(&scale)?
                .into_iter()
                .map(|exponent| f32::from_bits(u32::from(exponent) << 23))
                .collect(),
        };
        let bytes = read_tile::<u8>(tile)?;
        if bytes.len() != rows.len() * self.columns || scales.len() != (end - first) * scale_columns
        {
            candle_core::bail!("block-FP8 tile differs from admitted weight or scale shape");
        }
        let mut values = Vec::with_capacity(bytes.len());
        for (row, bytes) in rows.zip(bytes.chunks_exact(self.columns)) {
            let scales = &scales[(scale_index(row) - first) * scale_columns..][..scale_columns];
            for (block, &scale) in bytes.chunks(block_columns).zip(scales) {
                values.extend(block.iter().map(|&byte| fp8::decode_e4m3(byte) * scale));
            }
        }
        Ok(values)
    }
}
