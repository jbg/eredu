//! Native realization of packed words and separately bound companions.

use candle_core::{DType, Tensor as NativeTensor};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{Error, LinearFormatSpec, Parameter, Parameterized};

use crate::{quantized::read_tile, CandleContext, CandleTensor};

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub(crate) struct WordProjection {
    scale: Parameter<CandleTensor>,
    offset: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    encoding: WeightQuantization,
    #[parameter(skip)]
    rows: usize,
    #[parameter(skip)]
    columns: usize,
}

impl WordProjection {
    pub(crate) fn new(
        spec: &LinearFormatSpec,
        rows: i32,
        columns: i32,
        groups: Option<i32>,
        context: &CandleContext,
    ) -> Result<Option<Self>, Error> {
        let encoding = match spec.encoding() {
            LinearFormat::Affine(affine) => WeightQuantization::Affine(affine),
            LinearFormat::MxFp4 => WeightQuantization::MxFp4,
            _ => return Ok(None),
        };
        let geometry = encoding
            .packed_word_geometry(columns as usize)
            .map_err(Error::backend_source)?;
        let width = geometry.group_columns();
        let mut shape = vec![rows as usize, width];
        if let Some(groups) = groups {
            shape.insert(0, groups as usize);
        }
        let slot = |dtype| {
            CandleTensor::wrap(
                NativeTensor::zeros((), dtype, &context.device)
                    .and_then(|zero| zero.broadcast_as(shape.as_slice())),
            )
        };
        let scale_dtype = if encoding == WeightQuantization::MxFp4 {
            DType::U8
        } else {
            DType::F32
        };
        let scale_spec = spec
            .scale()
            .ok_or_else(|| Error::backend("missing packed scale declaration"))?;
        Ok(Some(Self {
            scale: Parameter::new(scale_spec.clone(), slot(scale_dtype)?),
            offset: spec
                .affine_bias()
                .map(|bias| slot(DType::F32).map(|tensor| Parameter::new(bias.clone(), tensor)))
                .transpose()?,
            encoding,
            rows: rows as usize,
            columns: columns as usize,
        }))
    }

    pub(crate) fn project(
        &self,
        input: &CandleTensor,
        weight: &CandleTensor,
        bias: Option<&CandleTensor>,
        group: Option<usize>,
    ) -> Result<CandleTensor, Error> {
        CandleTensor::wrap((|| {
            let input_dtype = input.value.dtype();
            if input.value.dim(candle_core::D::Minus1)? != self.columns {
                candle_core::bail!("packed projection activation width differs from its weights");
            }
            let input = input.value.to_dtype(DType::F32)?.contiguous()?;
            let values = self.group_values(group)?;
            let output =
                crate::quantized::project_decoded_tiles(&input, self.rows, self.columns, |rows| {
                    let count = rows.len();
                    let slice = |tensor: &NativeTensor| tensor.narrow(0, rows.start, count);
                    self.decode_rows(
                        &slice(&weight.value)?,
                        &slice(&values.0)?,
                        values.1.as_ref().map(slice).transpose()?,
                        count,
                    )
                })?
                .to_dtype(input_dtype)?;
            match bias {
                Some(bias) => crate::tensor::rounded_bf16_binary(
                    &output,
                    &bias.value,
                    NativeTensor::broadcast_add,
                ),
                None => Ok(output),
            }
        })())
    }

    pub(crate) fn embed(
        &self,
        ids: &CandleTensor,
        weight: &CandleTensor,
    ) -> Result<CandleTensor, Error> {
        CandleTensor::wrap((|| {
            let indices = crate::tensor::checked_gather_indices(&ids.value, self.rows as i64)?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let count = indices.len();
            let mut shape = ids.value.dims().to_vec();
            shape.push(self.columns);
            let device = weight.value.device();
            if count == 0 {
                return NativeTensor::zeros(shape, DType::F32, device);
            }
            let indices = NativeTensor::new(indices, device)?;
            let gather = |tensor: &NativeTensor| tensor.index_select(&indices, 0);
            let (scale, offset) = self.group_values(None)?;
            self.decode_rows(
                &gather(&weight.value)?,
                &gather(&scale)?,
                offset.as_ref().map(gather).transpose()?,
                count,
            )?
            .reshape(shape)
        })())
    }

    fn group_values(
        &self,
        group: Option<usize>,
    ) -> candle_core::Result<(NativeTensor, Option<NativeTensor>)> {
        let select = |value: &CandleTensor| match group {
            Some(group) => value.value.narrow(0, group, 1)?.squeeze(0),
            None => Ok(value.value.clone()),
        };
        Ok((
            select(self.scale.as_ref())?,
            self.offset
                .as_ref()
                .map(|value| select(value.as_ref()))
                .transpose()?,
        ))
    }

    fn decode_rows(
        &self,
        weight: &NativeTensor,
        scale: &NativeTensor,
        offset: Option<NativeTensor>,
        rows: usize,
    ) -> candle_core::Result<NativeTensor> {
        let geometry = self
            .encoding
            .packed_word_geometry(self.columns)
            .map_err(candle_core::Error::wrap)?;
        let words = geometry.word_columns();
        let groups = geometry.group_columns();
        let group = self.encoding.group_size() as usize;
        let bits = self.encoding.bits() as usize;
        if weight.dims() != [rows, words] || scale.dims() != [rows, groups] {
            candle_core::bail!("packed words and companion shapes differ from admission");
        }
        let weights = read_tile::<u32>(weight)?;
        let mut values = Vec::with_capacity(rows * self.columns);
        match self.encoding {
            WeightQuantization::MxFp4 => {
                let scales = read_tile::<u8>(scale)?;
                for (row, scales) in weights.chunks_exact(words).zip(scales.chunks_exact(groups)) {
                    for column in 0..self.columns {
                        let code = eredu_gguf::packed_words::unpack_code(row, column, bits) as u8;
                        values.push(eredu_gguf::packed_words::mxfp4_value(
                            code,
                            scales[column / group],
                        ));
                    }
                }
            }
            WeightQuantization::Affine(_) => {
                let offset = offset
                    .ok_or_else(|| candle_core::Error::Msg("missing packed offset".into()))?;
                if offset.dims() != scale.dims() {
                    candle_core::bail!("packed offset shape differs from scale");
                }
                let scales = read_tile::<f32>(&scale.to_dtype(DType::F32)?)?;
                let offsets = read_tile::<f32>(&offset.to_dtype(DType::F32)?)?;
                for ((row, scales), offsets) in weights
                    .chunks_exact(words)
                    .zip(scales.chunks_exact(groups))
                    .zip(offsets.chunks_exact(groups))
                {
                    for column in 0..self.columns {
                        let code = eredu_gguf::packed_words::unpack_code(row, column, bits);
                        values.push(code as f32 * scales[column / group] + offsets[column / group]);
                    }
                }
            }
            _ => unreachable!("word projections admit only affine and microscaling formats"),
        }
        NativeTensor::from_vec(values, (rows, self.columns), weight.device())
    }
}
