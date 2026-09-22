use super::*;
use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};
use safemlx::{
    ops::indexing::{IntoStrideBy, TryIndexOp},
    Array, Device, DeviceType, Dtype, Stream,
};
use std::collections::BTreeMap;

fn floats(data: &[f32], shape: &[i32], dtype: Dtype, strided: bool, s: &Stream) -> MlxTensor {
    let values = if strided {
        data.iter().flat_map(|v| [*v, 0.125]).collect()
    } else {
        data.to_vec()
    };
    let mut a = Array::from_slice(&values, &[values.len() as i32])
        .as_dtype(dtype, s)
        .unwrap();
    if strided {
        a = a.try_index_device((..).stride_by(2), s).unwrap();
    }
    MlxTensor::from_array(a.reshape(shape, s).unwrap())
}
struct Bind<'a> {
    values: &'a BTreeMap<String, MlxTensor>,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        *value = self.values[metadata.id().as_str()].clone();
    }
}
fn dense_values(
    spec: &TopKGroupSelectorSpec,
    dtype: Dtype,
    strided: bool,
    ties: bool,
    s: &Stream,
) -> (BTreeMap<String, MlxTensor>, Vec<f32>) {
    let d = spec.input_dimensions() as usize;
    let p = spec.selection().group_count() as usize;
    let mut values = BTreeMap::new();
    let weights = (0..d * p)
        .map(|i| ((if ties { i % d * 3 } else { i * 11 + 7 }) % 37) as f32 / 64. - 0.2)
        .collect::<Vec<_>>();
    values.insert(
        "weight".into(),
        floats(&weights, &[p as i32, d as i32], dtype, strided, s),
    );
    for (name, size) in [
        ("bias", p),
        ("correction", p),
        ("input_scale", d),
        ("coefficient_scale", p),
    ] {
        let data = (0..size)
            .map(|i| match name {
                "bias" => {
                    if ties {
                        0.125
                    } else {
                        i as f32 / 97.
                    }
                }
                "correction" => {
                    if ties {
                        0.125
                    } else {
                        ((i * 7) % 11) as f32 / 32.
                    }
                }
                _ => 0.5 + ((i * 3) % 7) as f32 / 16.,
            })
            .collect::<Vec<_>>();
        values.insert(
            name.into(),
            floats(&data, &[size as i32], dtype, strided, s),
        );
    }
    (values, weights)
}
fn check(
    spec: &TopKGroupSelectorSpec,
    shape: &[i32],
    dtype: Dtype,
    strided: bool,
    ties: bool,
    supplied: bool,
    control: Option<&GroupSelectionControl>,
    s: &Stream,
) {
    let (op, _) = quote(spec, shape, supplied, control);
    let m = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let b = m.operation_bound(&op).unwrap().unwrap();
    let allowed = b.scratch_bytes
        + b.outputs
            .iter()
            .map(|storage| match storage {
                WorkspaceOutputStorage::Allocate(bytes)
                | WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum::<u64>();
    let d = spec.input_dimensions() as usize;
    let p = spec.selection().group_count() as usize;
    let k = spec.selection().top_k() as usize;
    let count = shape.iter().product::<i32>() as usize;
    let rows = count / d;
    let data = (0..count)
        .map(|i| ((i * 13 + 5) % 31) as f32 / 64. - 0.17)
        .collect::<Vec<_>>();
    let input = floats(&data, shape, dtype, strided, s);
    let (mut values, mut dense) = dense_values(spec, dtype, strided, ties, s);
    match spec.format().encoding() {
        LinearFormat::Affine(_) => {
            let words = (0..p * d / 8)
                .map(|i| 0x12345678u32.rotate_left((i % 8 * 4) as u32))
                .collect::<Vec<_>>();
            dense = (0..p * d)
                .map(|i| ((words[i / 8] >> (i % 8 * 4)) & 15) as f32 / 32. - 0.125)
                .collect();
            values.insert(
                "weight".into(),
                MlxTensor::from_array(Array::from_slice(&words, &[p as i32, d as i32 / 8])),
            );
            values.insert(
                "scales".into(),
                floats(
                    &vec![1. / 32.; p * d / 32],
                    &[p as i32, d as i32 / 32],
                    dtype,
                    strided,
                    s,
                ),
            );
            values.insert(
                "biases".into(),
                floats(
                    &vec![-0.125; p * d / 32],
                    &[p as i32, d as i32 / 32],
                    dtype,
                    strided,
                    s,
                ),
            );
        }
        LinearFormat::MxFp4 => {
            const FP4: [f32; 16] = [
                0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
            ];
            let words = vec![0x12349876u32; p * d / 8];
            dense = (0..p * d)
                .map(|i| FP4[((words[i / 8] >> (i % 8 * 4)) & 15) as usize] / 32.)
                .collect();
            values.insert(
                "weight".into(),
                MlxTensor::from_array(Array::from_slice(&words, &[p as i32, d as i32 / 8])),
            );
            values.insert(
                "scales".into(),
                MlxTensor::from_array(Array::from_slice(
                    &vec![122u8; p * d / 32],
                    &[p as i32, d as i32 / 32],
                )),
            );
        }
        LinearFormat::GgufIQuant { .. } => {
            let mut bytes = Vec::new();
            dense.clear();
            for block in 0..p * d / 32 {
                bytes.extend_from_slice(&half::f16::from_f32(1. / 32.).to_bits().to_le_bytes());
                for i in 0..32 {
                    let q = ((block * 3 + i * 7) % 17) as i8 - 8;
                    bytes.push(q as u8);
                    dense.push(q as f32 / 32.);
                }
            }
            values.insert(
                "weight".into(),
                MlxTensor::from_array(Array::from_slice(&bytes, &[p as i32, d as i32 / 32 * 34])),
            );
        }
        _ => {}
    }
    let mut selector = MlxNeuralBackend::top_k_group_selector(spec.clone(), s).unwrap();
    selector.visit_parameters_mut(&mut Bind { values: &values });
    let ids = supplied.then(|| {
        MlxTensor::from_array(Array::from_slice(
            &(0..rows * k).map(|i| (i % k) as u32).collect::<Vec<_>>(),
            &[rows as i32, k as i32],
        ))
    });
    let mut arrays = values.values().map(MlxTensor::as_array).collect::<Vec<_>>();
    arrays.push(input.as_array());
    arrays.extend(ids.iter().map(MlxTensor::as_array));
    safemlx::transforms::eval(arrays).unwrap();
    s.synchronize().unwrap();
    let before = safemlx::memory::active_memory().unwrap();
    safemlx::memory::reset_peak_memory().unwrap();
    let output = run(&mut selector, &input, ids.as_ref(), control, s);
    let arrays = outputs(&output);
    safemlx::transforms::eval(arrays.iter().map(MlxTensor::as_array)).unwrap();
    s.synchronize().unwrap();
    let observed = safemlx::memory::peak_memory()
        .unwrap()
        .saturating_sub(before) as u64;
    assert!(observed <= allowed, "{shape:?} {dtype:?} {:?} strided={strided} ties={ties} supplied={supplied} control={control:?}: {observed}>{allowed}", spec.format().encoding());
    if matches!(
        b.outputs.last(),
        Some(WorkspaceOutputStorage::AliasOutput(_))
    ) && !control.is_some_and(|c| matches!(c.action, GroupSelectionAction::ZeroContribution(_)))
    {
        let scores = output
            .effective
            .selected_scores()
            .as_array()
            .allocation_info()
            .unwrap();
        let weights = output
            .effective
            .coefficients()
            .as_array()
            .allocation_info()
            .unwrap();
        assert_eq!(scores, weights);
    }
    // All fixture reads occur after allocation measurement and compact views.
    let x = input.to_f32_vec(s).unwrap();
    let mut decoded = values
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "weight" | "scales" | "biases"))
        .map(|(key, value)| (key.clone(), value.to_f32_vec(s).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let dense = if spec.format().encoding() == LinearFormat::Dense {
        values["weight"].to_f32_vec(s).unwrap()
    } else {
        // Independently constructed reference fixture, outside native execution.
        eredu_core::HostTensorBuffer::new(dense, ())
    };
    decoded.insert("weight".into(), dense);
    if let Some(original) = &output.original {
        numeric(spec, original, &x, &decoded, rows, dtype, false, None, s);
    }
    numeric(
        spec,
        &output.effective,
        &x,
        &decoded,
        rows,
        dtype,
        supplied,
        control,
        s,
    );
}
fn numeric(
    spec: &TopKGroupSelectorSpec,
    output: &GroupSelection<MlxTensor>,
    x: &[f32],
    data: &BTreeMap<String, eredu_core::HostTensorBuffer<f32>>,
    rows: usize,
    dtype: Dtype,
    supplied: bool,
    control: Option<&GroupSelectionControl>,
    s: &Stream,
) {
    let (d, p, k) = (
        spec.input_dimensions() as usize,
        spec.selection().group_count() as usize,
        spec.selection().top_k() as usize,
    );
    let ids = output
        .group_indices()
        .as_array()
        .contiguous(false, s)
        .unwrap()
        .into_evaluated()
        .unwrap()
        .try_to_vec::<u32>()
        .unwrap();
    let actual_scores = output.selected_scores().to_f32_vec(s).unwrap();
    let actual_weights = output.coefficients().to_f32_vec(s).unwrap();
    let (atol, rtol) = if dtype == Dtype::Float32 {
        (4e-5, 4e-4)
    } else {
        (0.006, 0.06)
    };
    let softmax = |v: &mut Vec<f64>| {
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for x in v.iter_mut() {
            *x = (*x - max).exp();
        }
        let sum = v.iter().sum::<f64>();
        for x in v {
            *x /= sum;
        }
    };
    for row in 0..rows {
        let active = control.filter(|c| {
            row as u64 >= c.first_row
                && (row as u64) < c.end_row
                && (row as u64 - c.first_row) % c.row_stride == 0
        });
        let mut hidden = x[row * d..(row + 1) * d]
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>();
        if let Some(t) = spec.input_transform() {
            let inv = (hidden.iter().map(|v| v * v).sum::<f64>() / d as f64 + t.epsilon() as f64)
                .sqrt()
                .recip();
            for (i, v) in hidden.iter_mut().enumerate() {
                *v *= inv * data["input_scale"][i] as f64;
                if t.inverse_sqrt_dimensions() {
                    *v /= (d as f64).sqrt();
                }
            }
        }
        let mut scores = (0..p)
            .map(|g| {
                hidden
                    .iter()
                    .enumerate()
                    .map(|(i, h)| h * data["weight"][g * d + i] as f64)
                    .sum::<f64>()
                    + if spec.bias().is_some() {
                        data["bias"][g] as f64
                    } else {
                        0.
                    }
            })
            .collect::<Vec<_>>();
        let bias = |v: &mut Vec<f64>, stage| {
            if let Some(c) = active {
                if let GroupSelectionAction::Bias {
                    stage: selected,
                    ids,
                    values,
                } = &c.action
                {
                    if *selected == stage {
                        for (id, value) in ids.iter().zip(values) {
                            v[*id as usize] += *value as f64;
                        }
                    }
                }
            }
        };
        bias(&mut scores, GroupScoreStage::RawLogits);
        match spec.selection().scoring() {
            GroupScoring::Softmax => softmax(&mut scores),
            GroupScoring::SelectedSoftmax => {}
            GroupScoring::Sigmoid => scores.iter_mut().for_each(|v| *v = 1. / (1. + (-*v).exp())),
            GroupScoring::SqrtSoftplus => {
                scores.iter_mut().for_each(|v| *v = v.exp().ln_1p().sqrt())
            }
            _ => unreachable!(),
        }
        bias(&mut scores, GroupScoreStage::TransformedScores);
        let mut ranking = scores.clone();
        if spec.correction_bias().is_some() {
            for (g, v) in ranking.iter_mut().enumerate() {
                *v += data["correction"][g] as f64;
            }
        }
        bias(&mut ranking, GroupScoreStage::RankingScores);
        if let Some(c) = active {
            if let GroupSelectionAction::Exclude(ids) = &c.action {
                for id in ids {
                    ranking[*id as usize] = f64::NEG_INFINITY;
                }
            }
        }
        let chosen = &ids[row * k..(row + 1) * k];
        let forced = active.is_some_and(|c| matches!(c.action, GroupSelectionAction::Force(_)));
        let mut unique = std::collections::BTreeSet::new();
        for id in chosen {
            assert!((*id as usize) < p && unique.insert(*id));
        }
        if supplied {
            assert_eq!(chosen, &(0..k as u32).collect::<Vec<_>>());
        }
        if forced {
            let c = active.unwrap();
            if let GroupSelectionAction::Force(ids) = &c.action {
                let i = ((row as u64 - c.first_row) / c.row_stride) as usize;
                assert_eq!(chosen, &ids[i * k..(i + 1) * k]);
            }
        }
        if !supplied && !forced {
            let partitions = spec.selection().selection_partitions() as usize;
            let width = p / partitions;
            let mut eligibility = vec![true; p];
            if partitions > 1 {
                let group_scores = ranking
                    .chunks(width)
                    .map(|r| {
                        let mut v = r.to_vec();
                        v.sort_by(|a, b| b.total_cmp(a));
                        v.iter().take(2).sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let selected_partitions = chosen
                    .iter()
                    .map(|id| *id as usize / width)
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(selected_partitions.len() <= spec.selection().selected_groups() as usize);
                let mut ranked = group_scores.clone();
                ranked.sort_by(|a, b| b.total_cmp(a));
                let cutoff = ranked[spec.selection().selected_groups() as usize - 1];
                let tolerance = atol + rtol * cutoff.abs();
                for &partition in &selected_partitions {
                    assert!(
                        group_scores[partition] + tolerance >= cutoff,
                        "partition cutoff"
                    );
                }
                for (g, eligible) in eligibility.iter_mut().enumerate() {
                    // A tied cutoff permits several legal partition choices.
                    // Every strictly higher partition must remain eligible.
                    *eligible = selected_partitions.contains(&(g / width))
                        || group_scores[g / width] > cutoff + tolerance;
                }
            }
            let cutoff = chosen
                .iter()
                .map(|id| ranking[*id as usize])
                .fold(f64::INFINITY, f64::min);
            for (g, v) in ranking.iter().enumerate() {
                if eligibility[g] && !unique.contains(&(g as u32)) {
                    assert!(
                        *v <= cutoff + atol + rtol * cutoff.abs(),
                        "routing cutoff {v}>{cutoff}"
                    );
                }
            }
        }
        let mut selected = chosen
            .iter()
            .map(|id| scores[*id as usize])
            .collect::<Vec<_>>();
        if spec.selection().scoring() == GroupScoring::SelectedSoftmax {
            softmax(&mut selected);
        }
        let mut weights = selected.clone();
        if spec.selection().normalize_selected() {
            let sum = weights.iter().sum::<f64>() + spec.selection().normalization_epsilon() as f64;
            for w in &mut weights {
                *w /= sum;
            }
        }
        for (i, w) in weights.iter_mut().enumerate() {
            *w *= spec.selection().coefficient_scale() as f64;
            if spec.coefficient_scale().is_some() {
                *w *= data["coefficient_scale"][chosen[i] as usize] as f64;
            }
            if let Some(c) = active {
                match &c.action {
                    GroupSelectionAction::ZeroContribution(ids) if ids.contains(&chosen[i]) => {
                        *w = 0.
                    }
                    GroupSelectionAction::Exclude(ids) => assert!(!ids.contains(&chosen[i])),
                    _ => {}
                }
            }
        }
        for (actual, expected) in actual_scores[row * k..(row + 1) * k]
            .iter()
            .zip(selected)
            .chain(actual_weights[row * k..(row + 1) * k].iter().zip(weights))
        {
            assert!(
                (*actual as f64 - expected).abs() <= atol + rtol * expected.abs(),
                "{dtype:?} {:?}: {actual}!={expected}",
                spec.selection()
            );
        }
    }
}

#[test]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_topk_routing_workspace_bounds_match_nonzero_equations_and_interventions() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut cases = 0;
    for scoring in [
        GroupScoring::Softmax,
        GroupScoring::SelectedSoftmax,
        GroupScoring::Sigmoid,
        GroupScoring::SqrtSoftplus,
    ] {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for rich in [false, true] {
                for strided in [false, true] {
                    for (shape, d, p, k, partitions, ties) in [
                        (&[2, 2, 8][..], 8, 8, 2, 2, false),
                        (&[1, 64][..], 64, 4097, 1, 1, true),
                        (&[2, 5][..], 5, 7, 7, 1, false),
                        (&[2, 32][..], 32, 8, 3, 4, false),
                        (&[2, 32][..], 32, 4, 2, 4, false),
                        (&[0, 8][..], 8, 8, 2, 2, false),
                    ] {
                        check(
                            &spec(d, p, k, partitions, scoring, rich, LinearFormat::Dense),
                            shape,
                            dtype,
                            strided,
                            ties,
                            false,
                            None,
                            &s,
                        );
                        cases += 1;
                    }
                    check(
                        &spec(8, 8, 2, 2, scoring, rich, LinearFormat::Dense),
                        &[2, 2, 8],
                        dtype,
                        strided,
                        false,
                        true,
                        None,
                        &s,
                    );
                    cases += 1;
                }
            }
            let spec = spec(8, 8, 2, 2, scoring, true, LinearFormat::Dense);
            for capture in [false, true] {
                for control in controls(&spec, capture) {
                    check(
                        &spec,
                        &[2, 2, 8],
                        dtype,
                        true,
                        false,
                        false,
                        Some(&control),
                        &s,
                    );
                    cases += 1;
                }
            }
            for encoding in [
                LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
                LinearFormat::MxFp4,
                LinearFormat::GgufIQuant {
                    ggml_type: eredu_gguf::GgmlType::Q8_0,
                    endian: eredu_gguf::Endian::Little,
                },
            ] {
                for strided in [false, true] {
                    check(
                        &super::spec(64, 8, 2, 2, scoring, true, encoding),
                        &[2, 2, 64],
                        dtype,
                        strided,
                        false,
                        false,
                        None,
                        &s,
                    );
                    cases += 1;
                }
            }
        }
    }
    eprintln!("TOPK_ROUTING_WORKSPACE_NATIVE_CASES={cases}");
}
