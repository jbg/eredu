//! Native controls reuse the ordinary selector's projection, top-k and weights.
use super::*;
use eredu_nn::routing_intervention::{
    execute_routing_intervention, GroupSelectionControl, RoutingMechanism, RoutingRows,
};
use safemlx::ops::indexing::{IntoStrideBy, TryIndexMutOp};

impl TopKGroupSelector {
    pub(crate) fn selection_spec(&self) -> Result<eredu_nn::TopKGroupSelectionSpec, Exception> {
        use eredu_nn::GroupScoring as S;
        eredu_nn::TopKGroupSelectionSpec::new(
            self.group_count,
            self.top_k,
            match self.score_function {
                TopKGroupScoring::Softmax => S::Softmax,
                TopKGroupScoring::SelectedSoftmax => S::SelectedSoftmax,
                TopKGroupScoring::Sigmoid => S::Sigmoid,
                TopKGroupScoring::SqrtSoftplus => S::SqrtSoftplus,
            },
            self.norm_topk_prob,
        )
        .and_then(|spec| spec.with_groups(self.n_group, self.topk_group))
        .and_then(|spec| {
            spec.with_weight_policy(self.normalization_epsilon, self.coefficient_scale)
        })
        .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn select_intervened(
        &mut self,
        input: &Array,
        control: &GroupSelectionControl,
        stream: &Stream,
    ) -> Result<(Option<GroupSelectionOutput>, GroupSelectionOutput), Exception> {
        let result = execute_routing_intervention(
            &mut NativeRouting {
                selector: self,
                stream,
                input_dtype: input.dtype(),
            },
            input,
            control,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let output = |selection: eredu_nn::GroupSelection<Array>| {
            let (indices, scores, weights) = selection.into_parts();
            Ok::<_, Exception>(GroupSelectionOutput {
                indices,
                scores,
                weights: weights.as_dtype(
                    routing_dtype(self.arithmetic.coefficients, input.dtype(), weights.dtype()),
                    stream,
                )?,
            })
        };
        Ok((
            result.original.map(output).transpose()?,
            output(result.effective)?,
        ))
    }
}

struct NativeRows {
    mask: Array,
    first: i32,
    end: i32,
    stride: i32,
    count: i32,
}
struct NativeRouting<'a> {
    selector: &'a mut TopKGroupSelector,
    stream: &'a Stream,
    input_dtype: Dtype,
}

impl NativeRouting<'_> {
    fn keep(&self, ids: &[u32]) -> Array {
        let mut keep = vec![true; self.selector.group_count as usize];
        for id in ids {
            keep[*id as usize] = false;
        }
        Array::from_slice(&keep, &[self.selector.group_count])
    }
    fn gathered_keep(
        &self,
        indices: &Array,
        ids: &[u32],
        rows: &NativeRows,
    ) -> Result<Array, Exception> {
        self.keep(ids)
            .take_axis(indices, 0, self.stream)?
            .logical_or(rows.mask.logical_not(self.stream)?, self.stream)
    }
    fn all(&self, predicate: Array) -> Result<bool, Exception> {
        Ok(predicate
            .all(false, self.stream)?
            .evaluated()?
            .as_slice::<bool>()[0])
    }
}

impl RoutingMechanism for NativeRouting<'_> {
    type Value = Array;
    type Rows = NativeRows;
    type Error = Exception;
    fn policy(&self) -> Result<(eredu_nn::TopKGroupSelectionSpec, bool), Exception> {
        Ok((
            self.selector.selection_spec()?,
            self.selector.learned_coefficient_scale.as_ref().is_some(),
        ))
    }
    fn token_rows(&self, input: &Array) -> Result<u64, Exception> {
        let shape = input.shape();
        if shape.is_empty() || shape.last() != Some(&self.selector.input_dims) {
            return Err(Exception::custom(
                "routing intervention input width mismatch",
            ));
        }
        shape[..shape.len() - 1]
            .iter()
            .try_fold(1u64, |n, size| {
                u64::try_from(*size)
                    .ok()
                    .and_then(|size| n.checked_mul(size))
            })
            .ok_or_else(|| Exception::custom("routing intervention token count overflow"))
    }
    fn rows(&self, selection: RoutingRows, tokens: u64) -> Result<NativeRows, Exception> {
        let convert = |n| {
            i32::try_from(n).map_err(|_| Exception::custom("routing index exceeds native indexing"))
        };
        let tokens = convert(tokens)?;
        let first = convert(selection.first)?;
        let end = convert(selection.end)?;
        let stride = convert(selection.stride)?;
        let count = convert((selection.end - selection.first).div_ceil(selection.stride))?;
        let rows = safemlx::ops::arange::<_, i32>(0, tokens, None, self.stream)?
            .reshape(&[tokens, 1], self.stream)?;
        let mask = rows
            .ge(Array::from(first), self.stream)?
            .logical_and(rows.lt(Array::from(end), self.stream)?, self.stream)?
            .logical_and(
                rows.subtract(Array::from(first), self.stream)?
                    .remainder(Array::from(stride), self.stream)?
                    .eq(Array::from(0i32), self.stream)?,
                self.stream,
            )?;
        Ok(NativeRows {
            mask,
            first,
            end,
            stride,
            count,
        })
    }
    fn project(&mut self, input: &Array) -> Result<Array, Exception> {
        self.selector.project_logits(input, self.stream)
    }
    fn transform(&self, raw: &Array) -> Result<Array, Exception> {
        self.selector
            .apply_scores(raw.clone(), self.input_dtype, self.stream)
    }
    fn ranking(&self, scores: &Array) -> Result<Array, Exception> {
        match self.selector.e_score_correction_bias.as_ref() {
            Some(bias) => scores.add(bias, self.stream),
            None => Ok(scores.clone()),
        }
    }
    fn select(&self, ranking: &Array) -> Result<Array, Exception> {
        self.selector.topk_indices(ranking, self.stream)
    }
    fn weights(
        &self,
        scores: &Array,
        ids: Array,
    ) -> Result<eredu_nn::GroupSelection<Array>, Exception> {
        let output = self
            .selector
            .weights_for_indices(scores, ids, self.stream)?;
        Ok(eredu_nn::GroupSelection::new(
            output.indices,
            output.scores,
            output.weights,
        ))
    }
    fn add_columns(
        &self,
        value: &Array,
        ids: &[u32],
        values: &[f32],
        rows: &NativeRows,
    ) -> Result<Array, Exception> {
        if !values.iter().all(|value_to_add| match value.dtype() {
            Dtype::Float16 => half::f16::from_f32(*value_to_add).is_finite(),
            Dtype::Bfloat16 => half::bf16::from_f32(*value_to_add).is_finite(),
            Dtype::Float32 => value_to_add.is_finite(),
            _ => false,
        }) {
            return Err(Exception::custom(
                "routing bias is not representable in the score dtype",
            ));
        }
        let mut bias = vec![0.0f32; self.selector.group_count as usize];
        for (id, value) in ids.iter().zip(values) {
            bias[*id as usize] = *value;
        }
        let bias = Array::from_slice(&bias, &[1, self.selector.group_count])
            .as_dtype(value.dtype(), self.stream)?;
        r#where(
            &rows.mask,
            value.add(bias, self.stream)?,
            value,
            self.stream,
        )
    }
    fn fill_columns(
        &self,
        value: &Array,
        ids: &[u32],
        fill: f32,
        rows: &NativeRows,
    ) -> Result<Array, Exception> {
        let keep = self
            .keep(ids)
            .reshape(&[1, self.selector.group_count], self.stream)?
            .logical_or(rows.mask.logical_not(self.stream)?, self.stream)?;
        r#where(
            keep,
            value,
            Array::from_f32(fill).as_dtype(value.dtype(), self.stream)?,
            self.stream,
        )
    }
    fn replace_rows(
        &self,
        indices: &Array,
        ids: &[u32],
        rows: &NativeRows,
    ) -> Result<Array, Exception> {
        let mut output = indices.clone();
        let forced = Array::from_slice(ids, &[rows.count, self.selector.top_k])
            .as_dtype(indices.dtype(), self.stream)?;
        output.try_index_mut_device(
            ((rows.first..rows.end).stride_by(rows.stride), ..),
            forced,
            self.stream,
        )?;
        Ok(output)
    }
    fn fill_gathered(
        &self,
        value: &Array,
        indices: &Array,
        ids: &[u32],
        fill: f32,
        rows: &NativeRows,
    ) -> Result<Array, Exception> {
        r#where(
            self.gathered_keep(indices, ids, rows)?,
            value,
            Array::from_f32(fill).as_dtype(value.dtype(), self.stream)?,
            self.stream,
        )
    }
    fn excludes(&self, indices: &Array, ids: &[u32], rows: &NativeRows) -> Result<bool, Exception> {
        self.all(self.gathered_keep(indices, ids, rows)?)
    }
    fn finite(&self, value: &Array) -> Result<bool, Exception> {
        self.all(value.is_finite(self.stream)?)
    }
    fn nonnegative(&self, value: &Array) -> Result<bool, Exception> {
        self.all(value.ge(Array::from_f32(0.0), self.stream)?)
    }
    fn positive_row_sums(&self, value: &Array) -> Result<bool, Exception> {
        self.all(
            value
                .sum_axis(-1, false, self.stream)?
                .gt(Array::from_f32(0.0), self.stream)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_routing_primitives_pass_shared_conformance() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut selector = TopKGroupSelector::new_with_quantization(
            TopKGroupSelectorConfig::new(
                2,
                4,
                4,
                TopKGroupScoring::Softmax,
                true,
                0.,
                1.,
                1,
                1,
                false,
                false,
                None,
                false,
                false,
            )
            .unwrap(),
            None,
            &stream,
        )
        .unwrap();
        selector.weight = PhysicalParam::new(Array::from_slice(
            &[
                1f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            &[4, 4],
        ));
        let input = Array::from_slice(&[0f32, 1., 2., 3., 0., 1., 2., 3.], &[2, 4]);
        eredu_evaluation::intervention::routing_conformance(
            &mut NativeRouting {
                selector: &mut selector,
                stream: &stream,
                input_dtype: input.dtype(),
            },
            &input,
            |v| {
                v.as_dtype(Dtype::Uint32, &stream)
                    .unwrap()
                    .contiguous(false, &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<u32>()
                    .to_vec()
            },
            |v| {
                v.as_dtype(Dtype::Float32, &stream)
                    .unwrap()
                    .contiguous(false, &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec()
            },
        );
        selector.score_function = TopKGroupScoring::SelectedSoftmax;
        selector.n_group = 2;
        selector.topk_group = 1;
        selector.normalization_epsilon = 0.25;
        selector.coefficient_scale = 1.5;
        selector.learned_coefficient_scale =
            PhysicalParam::new(Some(Array::from_slice(&[2f32, 3., 4., 5.], &[4])));
        let grouped_input = Array::from_slice(&[0f32, 2f32.ln(), 3f32.ln(), 4f32.ln()], &[1, 4]);
        eredu_evaluation::intervention::grouped_routing_conformance(
            &mut NativeRouting {
                selector: &mut selector,
                stream: &stream,
                input_dtype: input.dtype(),
            },
            &grouped_input,
            |v| {
                v.as_dtype(Dtype::Uint32, &stream)
                    .unwrap()
                    .contiguous(false, &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<u32>()
                    .to_vec()
            },
            |v| {
                v.as_dtype(Dtype::Float32, &stream)
                    .unwrap()
                    .contiguous(false, &stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec()
            },
        );
    }
}
