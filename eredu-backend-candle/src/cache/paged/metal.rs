//! Device-resident reduction lanes for ordered BF16 paged attention.

use super::{invalid, BackendFailure, CandleContext, CandleTensor, DType, Error, NativeTensor};
use candle_core::D;
use eredu_nn::arithmetic::Bf16DotGeometry;

pub(super) struct OrderedMetalAttention {
    end: i64,
    pass: usize,
    frontier: i64,
    row_shape: Option<[usize; 4]>,
    maxima: Option<NativeTensor>,
    mass: Option<NativeTensor>,
    values: Option<NativeTensor>,
}

impl OrderedMetalAttention {
    pub(super) fn new(end: i64) -> Self {
        Self {
            end,
            pass: 0,
            frontier: 0,
            row_shape: None,
            maxima: None,
            mass: None,
            values: None,
        }
    }

    pub(super) fn begin_pass(&mut self, pass: usize, end: i64) -> Result<(), BackendFailure> {
        if pass > 2 || pass != self.pass + 1 || self.frontier != end || end != self.end {
            return Err(invalid("paged attention sweep is incomplete"));
        }
        self.pass = pass;
        self.frontier = 0;
        Ok(())
    }

    pub(super) fn accumulate(
        &mut self,
        start: i64,
        scores: &NativeTensor,
        values: &NativeTensor,
    ) -> Result<(), Error> {
        self.accumulate_native(start, scores, values)
            .map_err(Error::backend_source)
    }

    fn accumulate_native(
        &mut self,
        start: i64,
        scores: &NativeTensor,
        values: &NativeTensor,
    ) -> candle_core::Result<()> {
        let (batch, heads, queries, width) = scores.dims4()?;
        let (value_batch, value_heads, value_tokens, features) = values.dims4()?;
        let row_shape = [batch, heads, queries, features];
        if self.row_shape.is_some_and(|shape| shape != row_shape)
            || width == 0
            || queries == 0
            || features == 0
            || scores.dtype() != DType::F32
            || values.dtype() != DType::F32
            || (batch, heads, width) != (value_batch, value_heads, value_tokens)
            || start != self.frontier
        {
            candle_core::bail!("paged attention sweep has noncontiguous or incompatible pages");
        }
        let frontier = start
            .checked_add(i64::try_from(width)?)
            .filter(|&next| next <= self.end)
            .ok_or_else(|| {
                candle_core::Error::Msg("attention sweep crosses its endpoint".into())
            })?;
        let width = u32::try_from(width)?;
        let start = u32::try_from(start)?;
        let end = u32::try_from(self.end)?;

        match self.pass {
            0 => {
                let maximum = scores.max_keepdim(D::Minus1)?;
                self.maxima = Some(if let Some(previous) = &self.maxima {
                    previous.maximum(&maximum)?
                } else {
                    maximum
                });
            }
            1 => {
                let maxima = self.maxima.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg(
                        "paged attention mass requires the maximum sweep".into(),
                    )
                })?;
                let shape = (batch, heads, queries, 4);
                let previous = match &self.mass {
                    Some(previous) => previous.clone(),
                    None => NativeTensor::zeros(shape, DType::F32, scores.device())?,
                };
                self.mass = Some(crate::metal_bf16::page_pass(
                    "bf16_page_mass",
                    [scores, maxima, &previous],
                    shape.into(),
                    DType::F32,
                    [width, start],
                )?);
            }
            2 => {
                let (Some(maxima), Some(mass)) = (&self.maxima, &self.mass) else {
                    candle_core::bail!("paged attention values require the mass sweep");
                };
                let probabilities = crate::metal_bf16::page_pass(
                    "bf16_page_probabilities",
                    [scores, maxima, mass],
                    scores.shape().clone(),
                    DType::BF16,
                    [width, end],
                )?;
                let reduction = Bf16DotGeometry {
                    rows: queries as u64,
                    width: end.into(),
                    columns: features as u64,
                }
                .value_reduction();
                let lanes = reduction.lanes();
                let page = reduction.page(start.into(), width as usize);
                let shape = (batch, heads, queries, features, lanes);
                let previous = match &self.values {
                    Some(previous) => previous.clone(),
                    None => NativeTensor::zeros(shape, DType::F32, scores.device())?,
                };
                self.values = Some(crate::metal_bf16::page_pass(
                    "bf16_page_values",
                    [&probabilities, values, &previous],
                    shape.into(),
                    DType::F32,
                    [
                        width,
                        page.first_lane as u32,
                        page.interleaved as u32,
                        u32::try_from(queries)?,
                        u32::try_from(features)?,
                        lanes as u32,
                    ],
                )?);
            }
            _ => candle_core::bail!("unknown paged attention sweep"),
        }
        self.row_shape = Some(row_shape);
        self.frontier = frontier;
        Ok(())
    }

    pub(super) fn scratch_bytes(&self) -> u64 {
        [&self.maxima, &self.mass, &self.values]
            .into_iter()
            .flatten()
            .map(|tensor| (tensor.elem_count() as u64).saturating_mul(4))
            .fold(0, u64::saturating_add)
            // A pass keeps its previous buffer until the new result settles.
            .saturating_mul(2)
    }

    pub(super) fn finish(
        self,
        end: i64,
        _: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        if self.pass != 2 || self.frontier != end {
            return Err(invalid("paged attention value sweep is incomplete"));
        }
        let values = self
            .values
            .ok_or_else(|| invalid("missing paged attention values"))?;
        CandleTensor::wrap((|| {
            let mut result = values.narrow(D::Minus1, 0, 1)?;
            for lane in 1..values.dim(D::Minus1)? {
                result = result.add(&values.narrow(D::Minus1, lane, 1)?)?;
            }
            result.squeeze(D::Minus1)?.to_dtype(DType::BF16)
        })())
        .map_err(BackendFailure::from_error)
    }
}
