//! Eager numerical primitives for the shared routing-intervention driver.
use super::*;
use eredu_nn::routing_intervention::*;

pub(super) struct Mechanism<'a> {
    pub router: &'a mut NumericRouter,
    pub context: &'a NumericContext,
}
fn softmax(row: &[f32]) -> Vec<f32> {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values = row.iter().map(|v| (*v - maximum).exp()).collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    values.into_iter().map(|v| v / sum).collect()
}
impl RoutingMechanism for Mechanism<'_> {
    type Value = NumericTensor;
    type Rows = Vec<usize>;
    type Error = Error;
    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool), Error> {
        Ok((
            self.router.selection,
            self.router.coefficient_scale.is_some(),
        ))
    }
    fn token_rows(&self, input: &NumericTensor) -> Result<u64, Error> {
        Ok((input.data.len() / *input.shape.last().unwrap() as usize) as u64)
    }
    fn rows(&self, selection: RoutingRows, _: u64) -> Result<Vec<usize>, Error> {
        Ok((selection.first as usize..selection.end as usize)
            .step_by(selection.stride as usize)
            .collect())
    }
    fn project(&mut self, input: &NumericTensor) -> Result<NumericTensor, Error> {
        self.router
            .linear
            .forward(&self.router.transformed_input(input), self.context)
    }
    fn transform(&self, raw: &NumericTensor) -> Result<NumericTensor, Error> {
        let experts = self.router.selection.group_count() as usize;
        let mut data = Vec::with_capacity(raw.data.len());
        for row in raw.data.chunks(experts) {
            data.extend(match self.router.selection.scoring() {
                eredu_nn::GroupScoring::Softmax => softmax(row),
                eredu_nn::GroupScoring::SelectedSoftmax => row.to_vec(),
                eredu_nn::GroupScoring::Sigmoid => {
                    row.iter().map(|v| 1. / (1. + (-v).exp())).collect()
                }
                eredu_nn::GroupScoring::SqrtSoftplus => {
                    row.iter().map(|v| (1. + v.exp()).ln().sqrt()).collect()
                }
                _ => return Err(Error::backend("unsupported group scoring policy")),
            });
        }
        Ok(NumericTensor::new(
            vec![(data.len() / experts) as i32, experts as i32],
            data,
        ))
    }
    fn ranking(&self, scores: &NumericTensor) -> Result<NumericTensor, Error> {
        let mut result = scores.clone();
        if let Some((bias, _)) = &self.router.correction_bias {
            for (index, value) in result.data.iter_mut().enumerate() {
                *value += bias.data[index % bias.data.len()];
            }
        }
        Ok(result)
    }
    fn select(&self, ranking: &NumericTensor) -> Result<NumericTensor, Error> {
        let spec = self.router.selection;
        let experts = spec.group_count() as usize;
        let top_k = spec.top_k() as usize;
        let groups = spec.selection_partitions() as usize;
        let per_group = experts / groups;
        let mut ids = vec![];
        for row in ranking.data.chunks(experts) {
            let mut group_order = (0..groups).collect::<Vec<_>>();
            let score = |group: usize| {
                let mut values = row[group * per_group..(group + 1) * per_group].to_vec();
                values.sort_by(|a, b| b.total_cmp(a));
                values.into_iter().take(2).sum::<f32>()
            };
            group_order.sort_by(|a, b| score(*b).total_cmp(&score(*a)));
            let eligible = &group_order[..spec.selected_groups() as usize];
            let mut order = (0..experts)
                .filter(|id| eligible.contains(&(id / per_group)))
                .collect::<Vec<_>>();
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
            ids.extend(order[..top_k].iter().map(|id| *id as i32));
        }
        NumericTensor::from_i32_slice(
            &ids,
            &[(ids.len() / top_k) as i32, top_k as i32],
            self.context,
        )
    }
    fn weights(
        &self,
        scores: &NumericTensor,
        ids: NumericTensor,
    ) -> Result<GroupSelection<NumericTensor>, Error> {
        let spec = self.router.selection;
        let experts = spec.group_count() as usize;
        let top_k = spec.top_k() as usize;
        let tokens = scores.data.len() / experts;
        if ids.shape != [tokens as i32, top_k as i32] {
            return Err(Error::backend("caller-selected route geometry mismatch"));
        }
        let indices = ids.to_i32_vec(self.context)?;
        let mut selected_scores = NumericTensor::zeros(ids.shape.clone());
        let mut coefficients = NumericTensor::zeros(ids.shape.clone());
        for token in 0..tokens {
            let mut selected = Vec::with_capacity(top_k);
            for id in &indices[token * top_k..(token + 1) * top_k] {
                if *id < 0 || *id as usize >= experts {
                    return Err(Error::backend("caller-selected expert id is invalid"));
                }
                selected.push(scores.data[token * experts + *id as usize]);
            }
            if spec.scoring() == eredu_nn::GroupScoring::SelectedSoftmax {
                selected = softmax(&selected);
            }
            let sum = selected.iter().sum::<f32>();
            for (slot, value) in selected.into_iter().enumerate() {
                let index = token * top_k + slot;
                let expert = indices[index] as usize;
                selected_scores.data[index] = value;
                coefficients.data[index] = if spec.normalize_selected() {
                    value / (sum + spec.normalization_epsilon())
                } else {
                    value
                } * spec.coefficient_scale()
                    * self
                        .router
                        .coefficient_scale
                        .as_ref()
                        .map_or(1., |(scale, _)| scale.data[expert]);
            }
        }
        Ok(GroupSelection::new(
            NumericTensor::from_i32_slice(&indices, &ids.shape, self.context)?,
            selected_scores,
            coefficients,
        ))
    }
    fn add_columns(
        &self,
        value: &NumericTensor,
        ids: &[u32],
        values: &[f32],
        rows: &Vec<usize>,
    ) -> Result<NumericTensor, Error> {
        let mut result = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        for row in rows {
            for (id, v) in ids.iter().zip(values) {
                result.data[row * width + *id as usize] += v;
            }
        }
        Ok(result)
    }
    fn fill_columns(
        &self,
        value: &NumericTensor,
        ids: &[u32],
        fill: f32,
        rows: &Vec<usize>,
    ) -> Result<NumericTensor, Error> {
        let mut result = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        for row in rows {
            for id in ids {
                result.data[row * width + *id as usize] = fill;
            }
        }
        Ok(result)
    }
    fn replace_rows(
        &self,
        indices: &NumericTensor,
        ids: &[u32],
        rows: &Vec<usize>,
    ) -> Result<NumericTensor, Error> {
        let mut result = indices.to_i32_vec(self.context)?;
        let width = *indices.shape.last().unwrap() as usize;
        for (row, ids) in rows.iter().zip(ids.chunks(width)) {
            for (slot, id) in ids.iter().enumerate() {
                result[row * width + slot] = *id as i32;
            }
        }
        NumericTensor::from_i32_slice(&result, &indices.shape, self.context)
    }
    fn fill_gathered(
        &self,
        value: &NumericTensor,
        indices: &NumericTensor,
        ids: &[u32],
        fill: f32,
        rows: &Vec<usize>,
    ) -> Result<NumericTensor, Error> {
        let indices = indices.to_i32_vec(self.context)?;
        let mut result = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        for row in rows {
            for slot in 0..width {
                let index = row * width + slot;
                if ids.contains(&(indices[index] as u32)) {
                    result.data[index] = fill;
                }
            }
        }
        Ok(result)
    }
    fn excludes(
        &self,
        indices: &NumericTensor,
        ids: &[u32],
        rows: &Vec<usize>,
    ) -> Result<bool, Error> {
        let width = *indices.shape.last().unwrap() as usize;
        let indices = indices.to_i32_vec(self.context)?;
        Ok(rows.iter().all(|row| {
            indices[row * width..(row + 1) * width]
                .iter()
                .all(|id| !ids.contains(&(*id as u32)))
        }))
    }
    fn finite(&self, value: &NumericTensor) -> Result<bool, Error> {
        Ok(value.data.iter().all(|v| v.is_finite()))
    }
    fn nonnegative(&self, value: &NumericTensor) -> Result<bool, Error> {
        Ok(value.data.iter().all(|v| *v >= 0.))
    }
    fn positive_row_sums(&self, value: &NumericTensor) -> Result<bool, Error> {
        Ok(value
            .data
            .chunks(*value.shape.last().unwrap() as usize)
            .all(|row| row.iter().sum::<f32>() > 0.))
    }
}

#[test]
fn routing_controls_preserve_stage_weights_and_unselected_rows() {
    let context = NumericContext::default();
    let parameter = ParameterSpec::trainable("fixture.router").unwrap();
    let mut router = NumericRouter {
        linear: NumericLinear {
            weight: NumericTensor::new(vec![3, 1], vec![4f32.ln(), 2f32.ln(), 0.]),
            weight_metadata: ParameterMetadata::from_spec(&parameter, true),
            bias: None,
            execution_weight: None,
            format_companions: vec![],
        },
        selection: TopKGroupSelectionSpec::new(3, 2, eredu_nn::GroupScoring::Softmax, true)
            .unwrap(),
        correction_bias: None,
        input_transform: None,
        coefficient_scale: None,
    };
    let input = NumericTensor::new(vec![3, 1], vec![1., 1., 1.]);
    let ordinary = router.select(&input, &context).unwrap();
    let cases = [
        (
            GroupSelectionAction::Bias {
                stage: GroupScoreStage::RawLogits,
                ids: vec![2],
                values: vec![8f32.ln()],
            },
            [2, 0],
            [8. / 14., 4. / 14.],
            [2. / 3., 1. / 3.],
        ),
        (
            GroupSelectionAction::Bias {
                stage: GroupScoreStage::TransformedScores,
                ids: vec![2],
                values: vec![1.],
            },
            [2, 0],
            [8. / 7., 4. / 7.],
            [2. / 3., 1. / 3.],
        ),
        (
            GroupSelectionAction::Bias {
                stage: GroupScoreStage::RankingScores,
                ids: vec![2],
                values: vec![1.],
            },
            [2, 0],
            [1. / 7., 4. / 7.],
            [1. / 5., 4. / 5.],
        ),
        (
            GroupSelectionAction::Force(vec![2, 1]),
            [2, 1],
            [1. / 7., 2. / 7.],
            [1. / 3., 2. / 3.],
        ),
        (
            GroupSelectionAction::Exclude(vec![0]),
            [1, 2],
            [2. / 7., 1. / 7.],
            [2. / 3., 1. / 3.],
        ),
        (
            GroupSelectionAction::ZeroContribution(vec![0]),
            [0, 1],
            [4. / 7., 2. / 7.],
            [0., 1. / 3.],
        ),
    ];
    for (action, ids, scores, weights) in cases {
        let control = GroupSelectionControl {
            expected: router.selection,
            learned_coefficient_scale: false,
            first_row: 1,
            end_row: 3,
            row_stride: 2,
            action,
            capture_original: true,
        };
        let result = router
            .select_intervened(&input, &control, &context)
            .unwrap();
        let original = result.original.unwrap();
        assert_eq!(
            original.group_indices().to_i32_vec(&context).unwrap(),
            ordinary.group_indices().to_i32_vec(&context).unwrap()
        );
        assert_eq!(
            original.selected_scores().data,
            ordinary.selected_scores().data
        );
        assert_eq!(original.coefficients().data, ordinary.coefficients().data);
        assert_eq!(
            result
                .effective
                .group_indices()
                .to_i32_vec(&context)
                .unwrap(),
            [0, 1, ids[0], ids[1], 0, 1]
        );
        for row in [0, 2] {
            assert_eq!(
                result.effective.coefficients().data[row * 2..row * 2 + 2],
                ordinary.coefficients().data[row * 2..row * 2 + 2]
            );
            assert_eq!(
                result.effective.selected_scores().data[row * 2..row * 2 + 2],
                ordinary.selected_scores().data[row * 2..row * 2 + 2]
            );
        }
        for slot in 0..2 {
            assert!(
                (result.effective.selected_scores().data[2 + slot] - scores[slot]).abs() <= 1e-6
            );
            assert!((result.effective.coefficients().data[2 + slot] - weights[slot]).abs() <= 1e-6);
        }
    }
    router.selection = router.selection.with_weight_policy(0., 2.).unwrap();
    router.coefficient_scale = Some((
        NumericTensor::new(vec![3], vec![1., 2., 3.]),
        ParameterMetadata::from_spec(&parameter, true),
    ));
    let mut control = GroupSelectionControl {
        expected: router.selection,
        learned_coefficient_scale: true,
        first_row: 1,
        end_row: 2,
        row_stride: 1,
        action: GroupSelectionAction::Force(vec![2, 1]),
        capture_original: false,
    };
    let result = router
        .select_intervened(&input, &control, &context)
        .unwrap();
    assert!(result.original.is_none());
    for (actual, expected) in result.effective.coefficients().data[2..4]
        .iter()
        .zip([2., 8. / 3.])
    {
        assert!((*actual - expected).abs() <= 1e-6);
    }
    control.learned_coefficient_scale = false;
    assert!(router
        .select_intervened(&input, &control, &context)
        .is_err());
}
