//! Host-indexed operations over the regular Candle router projection.
use super::*;
use eredu_nn::routing_intervention::{RoutingMechanism, RoutingRows};
use eredu_nn::TopKGroupSelectionSpec;

pub(super) struct NativeRouting<'a> {
    pub selector: &'a mut CandleSelector,
    pub context: &'a CandleContext,
    pub input_dtype: DType,
}

impl NativeRouting<'_> {
    fn edit_scores(
        &self,
        value: &CandleTensor,
        selected: &[usize],
        edit: impl Fn(&mut [f32], DType),
    ) -> Result<CandleTensor, Error> {
        let mut scores = value.to_f32_vec(self.context)?;
        let width = value
            .value
            .dim(candle_core::D::Minus1)
            .map_err(Error::backend_source)?;
        let dtype = value.value.dtype();
        for &row in selected {
            edit(&mut scores[row * width..][..width], dtype);
        }
        CandleTensor::wrap(
            NativeTensor::from_vec(scores, value.value.shape(), &self.context.device)
                .and_then(|value| value.to_dtype(dtype)),
        )
    }
}

impl RoutingMechanism for NativeRouting<'_> {
    type Value = CandleTensor;
    type Rows = Vec<usize>;
    type Error = Error;

    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool), Error> {
        Ok((
            self.selector.spec.selection(),
            self.selector.coefficient_scale.is_some(),
        ))
    }

    fn token_rows(&self, input: &CandleTensor) -> Result<u64, Error> {
        let width = self.selector.spec.input_dimensions() as usize;
        if input.value.dims().last() != Some(&width) || width == 0 {
            return Err(Error::backend("invalid routing input width"));
        }
        Ok((input.value.elem_count() / width) as u64)
    }

    fn rows(&self, range: RoutingRows, tokens: u64) -> Result<Self::Rows, Error> {
        let first = usize::try_from(range.first).map_err(Error::backend_source)?;
        let end = usize::try_from(range.end).map_err(Error::backend_source)?;
        let stride = usize::try_from(range.stride).map_err(Error::backend_source)?;
        if tokens > i32::MAX as u64 {
            return Err(Error::backend("routing rows exceed native indexing"));
        }
        Ok((first..end).step_by(stride).collect())
    }

    fn project(&mut self, input: &CandleTensor) -> Result<CandleTensor, Error> {
        self.selector.project_logits(input, self.context)
    }

    fn transform(&self, raw: &CandleTensor) -> Result<CandleTensor, Error> {
        self.selector.score(raw, self.input_dtype, self.context)
    }

    fn ranking(&self, scores: &CandleTensor) -> Result<CandleTensor, Error> {
        self.selector.ranking(scores, self.context)
    }

    fn select(&self, ranking: &CandleTensor) -> Result<CandleTensor, Error> {
        self.selector.rank(ranking, self.context)
    }

    fn weights(
        &self,
        scores: &CandleTensor,
        ids: CandleTensor,
    ) -> Result<GroupSelection<CandleTensor>, Error> {
        self.selector
            .weights(scores, ids, self.input_dtype, self.context)
    }

    fn add_columns(
        &self,
        value: &CandleTensor,
        ids: &[u32],
        additions: &[f32],
        rows: &Self::Rows,
    ) -> Result<CandleTensor, Error> {
        self.edit_scores(value, rows, |row, dtype| {
            for (&id, &add) in ids.iter().zip(additions) {
                row[id as usize] += round_float(add, dtype);
            }
        })
    }

    fn fill_columns(
        &self,
        value: &CandleTensor,
        ids: &[u32],
        fill: f32,
        rows: &Self::Rows,
    ) -> Result<CandleTensor, Error> {
        self.edit_scores(value, rows, |row, _| {
            for &id in ids {
                row[id as usize] = fill;
            }
        })
    }

    fn replace_rows(
        &self,
        indices: &CandleTensor,
        ids: &[u32],
        rows: &Self::Rows,
    ) -> Result<CandleTensor, Error> {
        let mut positions = indices.to_i32_vec(self.context)?;
        let width = self.selector.spec.selection().top_k() as usize;
        for (&row, ids) in rows.iter().zip(ids.chunks_exact(width)) {
            for (out, &id) in positions[row * width..][..width].iter_mut().zip(ids) {
                *out = id as i32;
            }
        }
        CandleTensor::wrap(NativeTensor::from_vec(
            positions,
            indices.value.shape(),
            &self.context.device,
        ))
    }

    fn fill_gathered(
        &self,
        value: &CandleTensor,
        indices: &CandleTensor,
        ids: &[u32],
        fill: f32,
        rows: &Self::Rows,
    ) -> Result<CandleTensor, Error> {
        let indices = indices.to_i32_vec(self.context)?;
        let width = self.selector.spec.selection().top_k() as usize;
        let mut scores = value.to_f32_vec(self.context)?;
        for &row in rows {
            let span = row * width..(row + 1) * width;
            for (&index, score) in indices[span.clone()].iter().zip(&mut scores[span]) {
                if ids.contains(&(index as u32)) {
                    *score = fill;
                }
            }
        }
        CandleTensor::wrap(
            NativeTensor::from_vec(scores, value.value.shape(), &self.context.device)
                .and_then(|scores| scores.to_dtype(value.value.dtype())),
        )
    }

    fn excludes(
        &self,
        indices: &CandleTensor,
        ids: &[u32],
        rows: &Self::Rows,
    ) -> Result<bool, Error> {
        let indices = indices.to_i32_vec(self.context)?;
        let width = self.selector.spec.selection().top_k() as usize;
        Ok(rows.iter().all(|&row| {
            indices[row * width..][..width]
                .iter()
                .all(|&index| !ids.contains(&(index as u32)))
        }))
    }

    fn finite(&self, value: &CandleTensor) -> Result<bool, Error> {
        Ok(value
            .to_f32_vec(self.context)?
            .into_iter()
            .all(f32::is_finite))
    }

    fn nonnegative(&self, value: &CandleTensor) -> Result<bool, Error> {
        Ok(value
            .to_f32_vec(self.context)?
            .into_iter()
            .all(|value| value >= 0.))
    }

    fn positive_row_sums(&self, value: &CandleTensor) -> Result<bool, Error> {
        let width = self.selector.spec.selection().top_k() as usize;
        Ok(value
            .to_f32_vec(self.context)?
            .chunks_exact(width)
            .all(|row| row.iter().any(|&value| value > 0.)))
    }
}
