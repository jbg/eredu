use super::*;
use eredu_nn::{GroupScoring, GroupSelection, TopKGroupSelectionSpec};
use std::io::{Error, Result};

#[derive(Clone, Debug)]
struct Value {
    shape: Vec<u64>,
    values: Vec<f32>,
    dtype: InterventionDtype,
}
impl Value {
    fn f32(shape: &[u64], values: Vec<f32>) -> Self {
        Self {
            shape: shape.to_vec(),
            values,
            dtype: InterventionDtype::Float32,
        }
    }
    fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let round = |v| match self.dtype {
            InterventionDtype::Float32 => v,
            InterventionDtype::Float16 => half::f16::from_f32(v).to_f32(),
            InterventionDtype::Bfloat16 => half::bf16::from_f32(v).to_f32(),
        };
        Self {
            values: self.values.iter().map(|v| round(f(*v))).collect(),
            ..self.clone()
        }
    }
}
struct Host {
    policy: TopKGroupSelectionSpec,
    learned: Option<Vec<f32>>,
    correction: Vec<f32>,
    projections: usize,
    selections: std::cell::Cell<usize>,
    transforms: std::cell::Cell<usize>,
    primitives: usize,
    bad_weights: bool,
}
impl Host {
    fn new() -> Self {
        Self {
            policy: TopKGroupSelectionSpec::new(4, 2, GroupScoring::Softmax, true).unwrap(),
            learned: None,
            correction: vec![0.; 4],
            projections: 0,
            selections: std::cell::Cell::new(0),
            transforms: std::cell::Cell::new(0),
            primitives: 0,
            bad_weights: false,
        }
    }
}
fn locations(shape: &[u64], slice: &ResolvedCaptureSlice) -> Vec<usize> {
    (0..shape.iter().product::<u64>() as usize)
        .filter(|index| {
            let mut index = *index as u64;
            (0..shape.len()).rev().all(|axis| {
                let coordinate = index % shape[axis];
                index /= shape[axis];
                coordinate >= slice.starts[axis]
                    && coordinate < slice.ends[axis]
                    && (coordinate - slice.starts[axis]).is_multiple_of(slice.strides[axis])
            })
        })
        .collect()
}
impl CaptureBackend for Host {
    type Tensor = Value;
    type Error = Error;
    fn shape(&self, v: &Value) -> Result<Vec<u64>> {
        Ok(v.shape.clone())
    }
    fn estimate(
        &self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> std::result::Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "direct primitive fixture has no capture transforms".into(),
        ))
    }
    fn transform(
        &mut self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload> {
        Err(Error::other(
            "capture transform not used by primitive conformance",
        ))
    }
}
impl InterventionBackend for Host {
    fn intervention_dtype(&self, value: &Value) -> Result<InterventionDtype> {
        Ok(value.dtype)
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> std::result::Result<(), CaptureError> {
        Ok(())
    }
    fn select_region(&mut self, value: &Value, slice: &ResolvedCaptureSlice) -> Result<Value> {
        self.primitives += 1;
        Ok(Value {
            shape: slice.shape.clone(),
            values: locations(&value.shape, slice)
                .iter()
                .map(|i| value.values[*i])
                .collect(),
            dtype: value.dtype,
        })
    }
    fn update_region(
        &mut self,
        value: &Value,
        slice: &ResolvedCaptureSlice,
        replacement: &Value,
    ) -> Result<Value> {
        let mut result = value.clone();
        for (i, v) in locations(&value.shape, slice)
            .iter()
            .zip(&replacement.values)
        {
            result.values[*i] = *v;
        }
        Ok(result)
    }
    fn zeros(&mut self, shape: &[u64], dtype: InterventionDtype) -> Result<Value> {
        Ok(Value {
            shape: shape.to_vec(),
            values: vec![0.; shape.iter().product::<u64>() as usize],
            dtype,
        })
    }
    fn scale(&mut self, value: &Value, factor: f32) -> Result<Value> {
        let scalar = Value {
            shape: vec![1],
            values: vec![factor],
            dtype: value.dtype,
        }
        .map(|v| v)
        .values[0];
        Ok(value.map(|v| v * scalar))
    }
    fn fill_masked(&mut self, value: &Value, keep: &[bool], fill: f32) -> Result<Value> {
        Ok(Value {
            values: value
                .values
                .iter()
                .zip(keep)
                .map(|(v, k)| if *k { *v } else { fill })
                .collect(),
            ..value.clone()
        })
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<Value> {
        let values = match &tensor.values {
            InterventionValues::Float32(v) => v.clone(),
            InterventionValues::Float16(v) => v
                .iter()
                .map(|v| half::f16::from_bits(*v).to_f32())
                .collect(),
            InterventionValues::Bfloat16(v) => v
                .iter()
                .map(|v| half::bf16::from_bits(*v).to_f32())
                .collect(),
        };
        Ok(Value {
            shape: tensor.shape.clone(),
            values,
            dtype: tensor.values.dtype(),
        })
    }
    fn add(&mut self, left: &Value, right: &Value) -> Result<Value> {
        Ok(Value {
            values: left
                .values
                .iter()
                .zip(&right.values)
                .map(|(a, b)| a + b)
                .collect(),
            ..left.clone()
        }
        .map(|v| v))
    }
    fn fill_columns(&mut self, value: &Value, ids: &[u32], fill: f32) -> Result<Value> {
        let width = *value.shape.last().unwrap() as usize;
        Ok(Value {
            values: value
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if ids.contains(&((i % width) as u32)) {
                        fill
                    } else {
                        *v
                    }
                })
                .collect(),
            ..value.clone()
        })
    }
}
fn selected(row: usize, rows: &RoutingRows) -> bool {
    row as u64 >= rows.first
        && (row as u64) < rows.end
        && (row as u64 - rows.first).is_multiple_of(rows.stride)
}
fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<_> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}
impl RoutingMechanism for Host {
    type Value = Value;
    type Rows = RoutingRows;
    type Error = Error;
    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool)> {
        Ok((self.policy, self.learned.is_some()))
    }
    fn token_rows(&self, input: &Value) -> Result<u64> {
        Ok(input.shape[0])
    }
    fn rows(&self, rows: RoutingRows, _: u64) -> Result<RoutingRows> {
        Ok(rows)
    }
    fn project(&mut self, input: &Value) -> Result<Value> {
        self.projections += 1;
        Ok(input.clone())
    }
    fn transform(&self, raw: &Value) -> Result<Value> {
        self.transforms.set(self.transforms.get() + 1);
        let width = self.policy.group_count() as usize;
        let values = raw
            .values
            .chunks(width)
            .flat_map(|row| match self.policy.scoring() {
                GroupScoring::Softmax => softmax(row),
                GroupScoring::SelectedSoftmax => row.to_vec(),
                GroupScoring::Sigmoid => row.iter().map(|v| 1. / (1. + (-v).exp())).collect(),
                GroupScoring::SqrtSoftplus => {
                    row.iter().map(|v| (1. + v.exp()).ln().sqrt()).collect()
                }
                _ => unreachable!(),
            })
            .collect();
        Ok(Value {
            values,
            ..raw.clone()
        })
    }
    fn ranking(&self, scores: &Value) -> Result<Value> {
        Ok(Value {
            values: scores
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| v + self.correction[i % self.correction.len()])
                .collect(),
            ..scores.clone()
        })
    }
    fn select(&self, ranking: &Value) -> Result<Value> {
        self.selections.set(self.selections.get() + 1);
        let width = self.policy.group_count() as usize;
        let per_group = width / self.policy.selection_partitions() as usize;
        let values = ranking
            .values
            .chunks(width)
            .flat_map(|row| {
                let mut groups: Vec<_> = row
                    .chunks(per_group)
                    .enumerate()
                    .map(|(i, values)| {
                        let mut values = values.to_vec();
                        values.sort_by(|a, b| b.total_cmp(a));
                        (i, values.iter().take(2).sum::<f32>())
                    })
                    .collect();
                groups.sort_by(|a, b| b.1.total_cmp(&a.1));
                let allowed: Vec<_> = groups
                    .iter()
                    .take(self.policy.selected_groups() as usize)
                    .map(|g| g.0)
                    .collect();
                let mut ids: Vec<_> = (0..width)
                    .filter(|id| allowed.contains(&(id / per_group)))
                    .collect();
                ids.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
                ids.truncate(self.policy.top_k() as usize);
                ids.into_iter().map(|id| id as f32).collect::<Vec<_>>()
            })
            .collect();
        Ok(Value::f32(
            &[ranking.shape[0], self.policy.top_k() as u64],
            values,
        ))
    }
    fn weights(&self, scores: &Value, ids: Value) -> Result<GroupSelection<Value>> {
        let width = self.policy.group_count() as usize;
        let k = self.policy.top_k() as usize;
        let mut values: Vec<_> = ids
            .values
            .iter()
            .enumerate()
            .map(|(i, id)| scores.values[(i / k) * width + *id as usize])
            .collect();
        if self.policy.scoring() == GroupScoring::SelectedSoftmax {
            values = values.chunks(k).flat_map(softmax).collect();
        }
        let selected_scores = Value::f32(&ids.shape, values.clone());
        for row in values.chunks_mut(k) {
            let denom = if self.policy.normalize_selected() {
                row.iter().sum::<f32>() + self.policy.normalization_epsilon()
            } else {
                1.
            };
            for v in row {
                *v = *v / denom * self.policy.coefficient_scale();
            }
        }
        if let Some(scales) = &self.learned {
            for (v, id) in values.iter_mut().zip(&ids.values) {
                *v *= scales[*id as usize];
            }
        }
        if self.bad_weights {
            values[0] = f32::NAN;
        }
        let weights = Value::f32(&ids.shape, values);
        Ok(GroupSelection::new(ids, selected_scores, weights))
    }
    fn add_columns(
        &self,
        value: &Value,
        ids: &[u32],
        values: &[f32],
        rows: &RoutingRows,
    ) -> Result<Value> {
        let width = *value.shape.last().unwrap() as usize;
        Ok(Value {
            values: value
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if selected(i / width, rows) {
                        *v + ids
                            .iter()
                            .position(|id| *id as usize == i % width)
                            .map_or(0., |j| values[j])
                    } else {
                        *v
                    }
                })
                .collect(),
            ..value.clone()
        })
    }
    fn fill_columns(
        &self,
        value: &Value,
        ids: &[u32],
        fill: f32,
        rows: &RoutingRows,
    ) -> Result<Value> {
        let width = *value.shape.last().unwrap() as usize;
        Ok(Value {
            values: value
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if selected(i / width, rows) && ids.contains(&((i % width) as u32)) {
                        fill
                    } else {
                        *v
                    }
                })
                .collect(),
            ..value.clone()
        })
    }
    fn replace_rows(&self, value: &Value, ids: &[u32], rows: &RoutingRows) -> Result<Value> {
        let mut output = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        let mut source = ids.iter();
        for (row, values) in output.values.chunks_mut(width).enumerate() {
            if selected(row, rows) {
                for v in values {
                    *v = *source.next().unwrap() as f32;
                }
            }
        }
        Ok(output)
    }
    fn fill_gathered(
        &self,
        value: &Value,
        indices: &Value,
        ids: &[u32],
        fill: f32,
        rows: &RoutingRows,
    ) -> Result<Value> {
        let width = *value.shape.last().unwrap() as usize;
        Ok(Value {
            values: value
                .values
                .iter()
                .zip(&indices.values)
                .enumerate()
                .map(|(i, (v, id))| {
                    if selected(i / width, rows) && ids.contains(&(*id as u32)) {
                        fill
                    } else {
                        *v
                    }
                })
                .collect(),
            ..value.clone()
        })
    }
    fn excludes(&self, indices: &Value, ids: &[u32], rows: &RoutingRows) -> Result<bool> {
        let width = *indices.shape.last().unwrap() as usize;
        Ok(indices
            .values
            .iter()
            .enumerate()
            .all(|(i, id)| !selected(i / width, rows) || !ids.contains(&(*id as u32))))
    }
    fn finite(&self, v: &Value) -> Result<bool> {
        Ok(v.values.iter().all(|v| v.is_finite()))
    }
    fn nonnegative(&self, v: &Value) -> Result<bool> {
        Ok(v.values.iter().all(|v| *v >= 0.))
    }
    fn positive_row_sums(&self, v: &Value) -> Result<bool> {
        Ok(v.values
            .chunks(*v.shape.last().unwrap() as usize)
            .all(|row| row.iter().sum::<f32>() > 0.))
    }
}

#[test]
fn host_primitives_pass_shared_activation_and_routing_conformance() {
    let mut native = Host::new();
    activation_conformance(&mut native, |v| v.values.clone());
    let input = Value::f32(&[2, 4], vec![0., 1., 2., 3., 0., 1., 2., 3.]);
    routing_conformance(
        &mut native,
        &input,
        |v| v.values.iter().map(|v| *v as u32).collect(),
        |v| v.values.clone(),
    );
    assert_eq!(
        native.projections, 8,
        "one projection per successful controlled call; invalid calls do no projection"
    );
    assert_eq!(
        native.selections.get(),
        15,
        "original selection is only computed when requested"
    );
}

#[test]
fn shared_driver_preserves_grouped_selected_softmax_scaling_and_rejects_bad_weights() {
    let mut native = Host::new();
    native.policy = TopKGroupSelectionSpec::new(4, 2, GroupScoring::SelectedSoftmax, true)
        .unwrap()
        .with_groups(2, 1)
        .unwrap()
        .with_weight_policy(0.25, 1.5)
        .unwrap();
    native.learned = Some(vec![2., 3., 4., 5.]);
    let input = Value::f32(&[1, 4], vec![0., 2f32.ln(), 3f32.ln(), 4f32.ln()]);
    grouped_routing_conformance(
        &mut native,
        &input,
        |v| v.values.iter().map(|v| *v as u32).collect(),
        |v| v.values.clone(),
    );
    let control = GroupSelectionControl {
        expected: native.policy,
        learned_coefficient_scale: true,
        first_row: 0,
        end_row: 1,
        row_stride: 1,
        action: GroupSelectionAction::Force(vec![0, 1]),
        capture_original: false,
    };
    native.bad_weights = true;
    assert!(execute_routing_intervention(&mut native, &input, &control).is_err());
}

#[test]
fn raw_bias_without_original_evidence_only_computes_the_effective_score_path() {
    for capture_original in [false, true] {
        let mut native = Host::new();
        let input = Value::f32(&[1, 4], vec![0., 1., 2., 3.]);
        let control = GroupSelectionControl {
            expected: native.policy,
            learned_coefficient_scale: false,
            first_row: 0,
            end_row: 1,
            row_stride: 1,
            action: GroupSelectionAction::Bias {
                stage: GroupScoreStage::RawLogits,
                ids: vec![0],
                values: vec![6.],
            },
            capture_original,
        };
        let result = execute_routing_intervention(&mut native, &input, &control).unwrap();
        assert_eq!(result.original.is_some(), capture_original);
        assert_eq!(native.projections, 1);
        let decisions = if capture_original { 2 } else { 1 };
        assert_eq!(native.selections.get(), decisions);
        assert_eq!(native.transforms.get(), decisions);
    }
}

#[test]
fn activation_half_payloads_preserve_bits_and_invalid_inputs_do_no_native_work() {
    let mut native = Host::new();
    let slice = ResolvedCaptureSlice {
        starts: vec![0],
        ends: vec![2],
        strides: vec![1],
        shape: vec![2],
    };
    for values in [
        InterventionValues::Float16(vec![0x3555, 0x8000]),
        InterventionValues::Bfloat16(vec![0x3eab, 0x8000]),
    ] {
        let tensor = InterventionTensor {
            shape: vec![2],
            values: values.clone(),
        };
        let source = native.realize_tensor(&tensor).unwrap();
        let output = eredu_runtime::intervention::apply_activation(
            &mut native,
            &source,
            &InterventionAction::Replace { tensor },
            &slice,
        )
        .unwrap();
        match values {
            InterventionValues::Float16(bits) => assert_eq!(
                output
                    .values
                    .iter()
                    .map(|v| half::f16::from_f32(*v).to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            InterventionValues::Bfloat16(bits) => assert_eq!(
                output
                    .values
                    .iter()
                    .map(|v| half::bf16::from_f32(*v).to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            _ => unreachable!(),
        }
        let before = native.primitives;
        assert!(eredu_runtime::intervention::apply_activation(
            &mut native,
            &source,
            &InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 1.
            },
            &slice
        )
        .is_err());
        assert_eq!(native.primitives, before);
    }
}
