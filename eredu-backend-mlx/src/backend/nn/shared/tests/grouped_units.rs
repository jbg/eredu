//! Independent scalar checks of the selected-unit boundary, not family equations.
use super::*;
use eredu_nn::{
    GroupedGatedProductOperator, GroupedRelu2Operator, GroupedUnitBatch, GroupedUnitError,
    GroupedUnitObserver, TensorParallelGroupedGatedProductOperator,
    TensorParallelGroupedRelu2Operator,
};
use std::collections::{BTreeMap, BTreeSet};

const GROUPS: usize = 4;
const HIDDEN: usize = 3;
const UNITS: usize = 4;

fn input_value(token: usize, channel: usize) -> f32 {
    ((token * 3 + channel) % 17) as f32 / 16.0 - 0.5
}
fn read_value(expert: usize, row: usize, channel: usize) -> f32 {
    ((expert * 11 + row * 5 + channel * 3) % 19) as f32 / 16.0 - 0.5625
}
fn read_bias(expert: usize, row: usize) -> f32 {
    (expert as f32 - row as f32) / 32.0
}
fn write_value(expert: usize, channel: usize, unit: usize) -> f32 {
    ((expert * 7 + channel * 3 + unit * 5) % 13) as f32 / 8.0 - 0.75
}
fn write_bias(expert: usize, channel: usize) -> f32 {
    (expert as f32 + channel as f32 - 1.5) / 16.0
}
fn route(token: usize, slot: usize) -> usize {
    [[2, 0], [1, 2], [0, 0]][token % 3][slot]
}
fn coefficient(token: usize, slot: usize) -> f32 {
    if slot == 0 {
        0.25 + (token % 3) as f32 / 8.0
    } else {
        0.625
    }
}
fn activation(token: usize, expert: usize, unit: usize, gated: bool) -> f64 {
    let read = |row| {
        (0..HIDDEN)
            .map(|c| input_value(token, c) as f64 * read_value(expert, row, c) as f64)
            .sum::<f64>()
            + if gated {
                read_bias(expert, row) as f64
            } else {
                0.0
            }
    };
    let gate = read(unit);
    if gated {
        gate / (1.0 + (-gate).exp()) * read(unit + UNITS)
    } else {
        gate.max(0.0).powi(2)
    }
}
#[derive(Clone, Copy, Debug)]
enum Edit {
    None,
    Zero,
    Scale,
    Add,
    Replace,
    Keep,
}
fn edited(value: f64, token: usize, expert: usize, unit: usize, edit: Edit) -> f64 {
    if token % 3 != 1 {
        return value;
    }
    if matches!(edit, Edit::Keep) {
        return if (expert == 0 && unit == 2) || (expert == 2 && unit == 1) {
            value
        } else {
            0.0
        };
    }
    if expert != 2 || unit != 1 {
        return value;
    }
    match edit {
        Edit::None | Edit::Keep => value,
        Edit::Zero => 0.0,
        Edit::Scale => value * -1.5,
        Edit::Add => value + 0.375,
        Edit::Replace => -0.625,
    }
}
fn reference(tokens: usize, gated: bool, edit: Edit) -> Vec<f32> {
    (0..tokens)
        .flat_map(|token| {
            (0..HIDDEN).map(move |channel| {
                (0..2)
                    .map(|slot| {
                        let expert = route(token, slot);
                        coefficient(token, slot) as f64
                            * ((0..UNITS)
                                .map(|unit| {
                                    edited(
                                        activation(token, expert, unit, gated),
                                        token,
                                        expert,
                                        unit,
                                        edit,
                                    ) * write_value(expert, channel, unit) as f64
                                })
                                .sum::<f64>()
                                + if gated {
                                    write_bias(expert, channel) as f64
                                } else {
                                    0.0
                                })
                    })
                    .sum::<f64>() as f32
            })
        })
        .collect()
}
fn inputs(tokens: usize) -> (MlxTensor, GroupSelection<MlxTensor>) {
    let values: Vec<_> = (0..tokens)
        .flat_map(|t| (0..HIDDEN).map(move |c| input_value(t, c)))
        .collect();
    let ids: Vec<_> = (0..tokens)
        .flat_map(|t| (0..2).map(move |s| route(t, s) as i32))
        .collect();
    let weights: Vec<_> = (0..tokens)
        .flat_map(|t| (0..2).map(move |s| coefficient(t, s)))
        .collect();
    let coefficients = MlxTensor::from_array(Array::from_slice(&weights, &[tokens as i32, 2]));
    (
        MlxTensor::from_array(Array::from_slice(
            &values,
            &[1, tokens as i32, HIDDEN as i32],
        )),
        GroupSelection::new(
            MlxTensor::from_array(Array::from_slice(&ids, &[tokens as i32, 2])),
            coefficients.clone(),
            coefficients,
        ),
    )
}
fn projection(name: &str, bias: bool) -> GroupedProjectionSpec {
    GroupedProjectionSpec::new(
        ParameterSpec::trainable(name).unwrap(),
        bias.then(|| ParameterSpec::trainable(format!("{name}_bias")).unwrap()),
        eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
    )
    .unwrap()
}
fn bank(gated: bool, range: std::ops::Range<usize>, stream: &safemlx::Stream) -> Bank {
    let width = range.len();
    let mut down = Vec::new();
    let mut up = Vec::new();
    let mut up_bias = Vec::new();
    let mut down_bias = Vec::new();
    for expert in 0..GROUPS {
        for row in (0..if gated { 2 } else { 1 })
            .flat_map(|half| range.clone().map(move |unit| unit + half * UNITS))
        {
            up_bias.push(read_bias(expert, row));
            for channel in 0..HIDDEN {
                up.push(read_value(expert, row, channel));
            }
        }
        for channel in 0..HIDDEN {
            down_bias.push(write_bias(expert, channel));
            for unit in range.clone() {
                down.push(write_value(expert, channel, unit));
            }
        }
    }
    let rows = width * if gated { 2 } else { 1 };
    let mut bindings = BTreeMap::from([
        (
            (if gated { "gate_up_proj" } else { "up_proj" }).into(),
            Array::from_slice(&up, &[GROUPS as i32, rows as i32, HIDDEN as i32]),
        ),
        (
            "down_proj".into(),
            Array::from_slice(&down, &[GROUPS as i32, HIDDEN as i32, width as i32]),
        ),
    ]);
    if gated {
        bindings.insert(
            "gate_up_proj_bias".into(),
            Array::from_slice(&up_bias, &[GROUPS as i32, rows as i32]),
        );
        bindings.insert(
            "down_proj_bias".into(),
            Array::from_slice(&down_bias, &[GROUPS as i32, HIDDEN as i32]),
        );
        let mut bank = MlxNeuralBackend::grouped_gated_product(
            GroupedGatedProductSpec::new(
                GROUPS as i32,
                HIDDEN as i32,
                width as i32,
                HIDDEN as i32,
                eredu_nn::GatedProductPolicy::ordinary_silu(),
                GatedProductGroupLayout::Packed {
                    gate_up: projection("read", true),
                    down: projection("write", true),
                },
            )
            .unwrap(),
            stream,
        )
        .unwrap();
        bank.bind_local_parameters(bindings).unwrap();
        Bank::Gated(bank)
    } else {
        let mut bank = MlxNeuralBackend::grouped_relu2(
            GroupedRelu2Spec::new(
                GROUPS as i32,
                HIDDEN as i32,
                width as i32,
                projection("read", false),
                projection("write", false),
            )
            .unwrap(),
            stream,
        )
        .unwrap();
        bank.bind_local_parameters(bindings).unwrap();
        Bank::Relu(bank)
    }
}
enum Bank {
    Gated(MlxGroupedGatedProduct),
    Relu(MlxGroupedRelu2),
}
impl Bank {
    fn forward(
        &mut self,
        input: &MlxTensor,
        routes: &GroupSelection<MlxTensor>,
        stream: &safemlx::Stream,
        observer: Option<&mut dyn GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        match self {
            Self::Gated(bank) => {
                bank.forward_grouped_with_unit_observer(input, routes, stream, observer)
            }
            Self::Relu(bank) => {
                bank.forward_grouped_with_unit_observer(input, routes, stream, observer)
            }
        }
    }
    fn ordinary(
        &mut self,
        input: &MlxTensor,
        routes: &GroupSelection<MlxTensor>,
        stream: &safemlx::Stream,
    ) -> MlxTensor {
        match self {
            Self::Gated(bank) => bank.forward_grouped(input, routes, stream).unwrap(),
            Self::Relu(bank) => bank.forward_grouped(input, routes, stream).unwrap(),
        }
    }
    fn partial(
        &mut self,
        input: &MlxTensor,
        routes: &GroupSelection<MlxTensor>,
        stream: &safemlx::Stream,
        observer: &mut dyn GroupedUnitObserver<MlxTensor>,
    ) -> TensorParallelGroupedOutput<MlxTensor> {
        match self {
            Self::Gated(bank) => bank
                .forward_grouped_tensor_parallel_with_unit_observer(
                    input,
                    routes,
                    2,
                    stream,
                    Some(observer),
                )
                .unwrap(),
            Self::Relu(bank) => bank
                .forward_grouped_tensor_parallel_with_unit_observer(
                    input,
                    routes,
                    2,
                    stream,
                    Some(observer),
                )
                .unwrap(),
        }
    }
}
struct Observer {
    tokens: usize,
    gated: bool,
    start: usize,
    edit: Edit,
    before: BTreeSet<(usize, usize)>,
    after: BTreeSet<(usize, usize)>,
    chunks: Vec<usize>,
}
impl Observer {
    fn new(tokens: usize, gated: bool, start: usize, edit: Edit) -> Self {
        Self {
            tokens,
            gated,
            start,
            edit,
            before: BTreeSet::new(),
            after: BTreeSet::new(),
            chunks: Vec::new(),
        }
    }
    fn check(&mut self, b: &GroupedUnitBatch<'_, MlxTensor>, effective: bool) {
        assert_eq!(b.total_token_count, self.tokens);
        assert_eq!(b.group_count, GROUPS);
        let ids = b.group_indices.as_array().evaluated().unwrap();
        let slots = b.selection_indices.as_array().evaluated().unwrap();
        let token_ids = b.token_indices.as_array().evaluated().unwrap();
        let coefficients = b.coefficients.as_array().evaluated().unwrap();
        let values = b.values.as_array().evaluated().unwrap();
        let rows = b.values.shape()[0] as usize;
        let width = b.values.shape()[1] as usize;
        assert_eq!(rows, b.coefficients.shape()[0] as usize * 2);
        for row in 0..rows {
            let selection = slots.as_slice::<i32>()[row] as usize;
            let token = token_ids.as_slice::<i32>()[row] as usize + b.token_offset;
            let slot = selection % 2;
            let expert = ids.as_slice::<i32>()[row] as usize;
            assert_eq!(selection / 2 + b.token_offset, token);
            assert_eq!(expert, route(token, slot));
            assert_ne!(expert, 3, "inactive expert must not acquire synthetic rows");
            assert_eq!(
                coefficients.as_slice::<f32>()[selection],
                coefficient(token, slot)
            );
            assert!(
                (if effective {
                    &mut self.after
                } else {
                    &mut self.before
                })
                .insert((token, slot)),
                "duplicate route evidence"
            );
            for local in 0..width {
                let unit = self.start + local;
                let baseline = activation(token, expert, unit, self.gated);
                let expected = if effective {
                    edited(baseline, token, expert, unit, self.edit)
                } else {
                    baseline
                };
                let actual = values.as_slice::<f32>()[row * width + local] as f64;
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "token={token} expert={expert} unit={unit}: {actual} vs {expected}"
                );
            }
        }
    }
    fn finish(&self) {
        assert_eq!(self.before.len(), self.tokens * 2);
        assert_eq!(self.before, self.after);
        assert_eq!(
            self.chunks,
            if self.gated && self.tokens > 64 {
                vec![0, 32, 64]
            } else {
                vec![0]
            }
        );
    }
}
impl GroupedUnitObserver<MlxTensor> for Observer {
    fn observe(&mut self, b: &GroupedUnitBatch<'_, MlxTensor>) -> Result<(), ComputeError> {
        self.chunks.push(b.token_offset);
        self.check(b, false);
        Ok(())
    }
    fn intervene(
        &mut self,
        b: &GroupedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, ComputeError> {
        if matches!(self.edit, Edit::None) {
            return Ok(None);
        }
        let ids = b.group_indices.as_array().evaluated().unwrap();
        let tokens = b.token_indices.as_array().evaluated().unwrap();
        let values = b.values.as_array().evaluated().unwrap();
        let width = b.values.shape()[1] as usize;
        let edited_values: Vec<_> = values
            .as_slice::<f32>()
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let row = index / width;
                edited(
                    value as f64,
                    tokens.as_slice::<i32>()[row] as usize + b.token_offset,
                    ids.as_slice::<i32>()[row] as usize,
                    self.start + index % width,
                    self.edit,
                ) as f32
            })
            .collect();
        Ok(Some(MlxTensor::from_array(Array::from_slice(
            &edited_values,
            b.values.shape(),
        ))))
    }
    fn observe_effective(
        &mut self,
        b: &GroupedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), ComputeError> {
        self.check(b, true);
        Ok(())
    }
}
fn verify(device: DeviceType) {
    let execution = ExecutionContext::new(Device::new(device, 0));
    let stream = execution.stream();
    for gated in [true, false] {
        for tokens in [5, 67] {
            let (input, routes) = inputs(tokens);
            let mut full = bank(gated, 0..UNITS, stream);
            let baseline = full.ordinary(&input, &routes, stream);
            close(&baseline, &reference(tokens, gated, Edit::None), 2e-5);
            close(
                &full.forward(&input, &routes, stream, None).unwrap(),
                baseline.as_array().evaluated().unwrap().as_slice::<f32>(),
                0.0,
            );
            for edit in [
                Edit::None,
                Edit::Zero,
                Edit::Scale,
                Edit::Add,
                Edit::Replace,
                Edit::Keep,
            ] {
                let mut observer = Observer::new(tokens, gated, 0, edit);
                let actual = full
                    .forward(&input, &routes, stream, Some(&mut observer))
                    .unwrap();
                assert_eq!(actual.shape(), input.shape());
                let expected = reference(tokens, gated, edit);
                close(&actual, &expected, 2e-5);
                if matches!(edit, Edit::None) {
                    close(
                        &actual,
                        baseline.as_array().evaluated().unwrap().as_slice::<f32>(),
                        0.0,
                    );
                }
                observer.finish();
                if !matches!(edit, Edit::None) {
                    assert!(expected
                        .iter()
                        .zip(reference(tokens, gated, Edit::None))
                        .any(|(a, b)| (a - b).abs() > 1e-4));
                }
                let mut total: Option<MlxTensor> = None;
                let mut bias: Option<Vec<f32>> = None;
                for range in [0..2, 2..4] {
                    let mut observer = Observer::new(tokens, gated, range.start, edit);
                    let (partial, post) = bank(gated, range, stream)
                        .partial(&input, &routes, stream, &mut observer)
                        .into_parts();
                    total = Some(match total {
                        Some(total) => total.add(&partial, stream).unwrap(),
                        None => partial,
                    });
                    if let Some(post) = post {
                        if let Some(previous) = &bias {
                            close(&post, previous, 0.0)
                        } else {
                            bias = Some(
                                post.as_array()
                                    .evaluated()
                                    .unwrap()
                                    .as_slice::<f32>()
                                    .to_vec(),
                            )
                        }
                    }
                    observer.finish();
                }
                let total = total.unwrap();
                let total = if let Some(bias) = bias {
                    total
                        .add(
                            &MlxTensor::from_array(Array::from_slice(&bias, input.shape())),
                            stream,
                        )
                        .unwrap()
                } else {
                    total
                };
                close(&total, &expected, 2e-5);
            }
            close(
                &full.ordinary(&input, &routes, stream),
                &reference(tokens, gated, Edit::None),
                2e-5,
            );
        }
    }
}
#[test]
#[ignore = "explicit native selected-unit validation; run outside the sandbox"]
fn grouped_unit_hooks_cpu() {
    verify(DeviceType::Cpu)
}
#[test]
#[ignore = "explicit native selected-unit validation; run outside the sandbox"]
fn grouped_unit_hooks_metal() {
    verify(DeviceType::Gpu)
}

#[derive(Debug, thiserror::Error)]
#[error("observer sentinel")]
struct Sentinel;
struct Reject<'a> {
    stage: u8,
    observed: usize,
    effective: usize,
    stream: &'a safemlx::Stream,
}
impl GroupedUnitObserver<MlxTensor> for Reject<'_> {
    fn observe(&mut self, _: &GroupedUnitBatch<'_, MlxTensor>) -> Result<(), ComputeError> {
        self.observed += 1;
        if self.stage == 0 {
            Err(ComputeError::backend_retained_source(Sentinel))
        } else {
            Ok(())
        }
    }
    fn intervene(
        &mut self,
        b: &GroupedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, ComputeError> {
        match self.stage {
            1 => Err(ComputeError::backend_retained_source(Sentinel)),
            2 => Ok(Some(MlxTensor::from_array(Array::from_slice(
                &[0_f32],
                &[1],
            )))),
            3 => Ok(Some(MlxTensor::from_array(
                b.values
                    .as_array()
                    .as_dtype(Dtype::Int32, self.stream)
                    .unwrap(),
            ))),
            _ => Ok(None),
        }
    }
    fn observe_effective(
        &mut self,
        _: &GroupedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), ComputeError> {
        self.effective += 1;
        Err(ComputeError::backend_retained_source(Sentinel))
    }
}
#[test]
#[ignore = "explicit native selected-unit failure validation; run outside the sandbox"]
fn grouped_unit_failures_preserve_sources_and_stop_before_projection() {
    use std::error::Error as _;
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let (input, routes) = inputs(67);
    for gated in [true, false] {
        let mut full = bank(gated, 0..UNITS, stream);
        // An invalid downstream matrix is a sentinel: every observer rejection
        // must occur before its multiplication is even constructed.
        match &mut full {
            Bank::Gated(b) => {
                b.module.down_proj =
                    crate::module::PhysicalParam::new(Array::from_slice(&[0_f32], &[1]))
            }
            Bank::Relu(b) => {
                b.module.down_proj =
                    crate::module::PhysicalParam::new(Array::from_slice(&[0_f32], &[1]))
            }
        }
        for stage in 0..5 {
            let mut observer = Reject {
                stage,
                observed: 0,
                effective: 0,
                stream,
            };
            let error = full
                .forward(&input, &routes, stream, Some(&mut observer))
                .unwrap_err();
            let source = error
                .source()
                .expect("original source retained across native callback");
            match stage {
                2 => assert!(matches!(
                    source.downcast_ref::<GroupedUnitError>(),
                    Some(GroupedUnitError::ReplacementShape { .. })
                )),
                3 => assert_eq!(
                    source.downcast_ref::<GroupedUnitError>(),
                    Some(&GroupedUnitError::ReplacementDtype)
                ),
                _ => assert!(source.is::<Sentinel>()),
            }
            assert_eq!(observer.observed, 1);
            assert_eq!(observer.effective, usize::from(stage == 4));
        }
    }
}

struct ProviderCapture(Observer);
impl eredu_runtime::RoutedUnitObserver<MlxTensor> for ProviderCapture {
    fn observe(
        &mut self,
        b: &eredu_runtime::RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), ComputeError> {
        assert!(b.global_groups.is_none());
        assert!(b.origins.is_none());
        assert_eq!(b.provider_token_offset, 0);
        let source = b.source_groups.as_array().evaluated().unwrap();
        let indices = b.units.selection_indices.as_array().evaluated().unwrap();
        let tokens = b.units.token_indices.as_array().evaluated().unwrap();
        for row in 0..b.units.values.shape()[0] as usize {
            let slot = indices.as_slice::<i32>()[row] as usize % 2;
            let origin = b
                .route_origin(tokens.as_slice::<i32>()[row] as usize, slot)
                .unwrap();
            assert_eq!(origin.source_peer, None);
            assert_eq!(
                source.as_slice::<i32>()[origin.token * 2 + origin.slot] as usize,
                route(origin.token, origin.slot)
            );
        }
        self.0.observe(&b.units)
    }
    fn intervene(
        &mut self,
        b: &eredu_runtime::RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<Option<MlxTensor>, ComputeError> {
        self.0.intervene(&b.units)
    }
    fn observe_effective(
        &mut self,
        b: &eredu_runtime::RoutedUnitBatch<'_, MlxTensor>,
    ) -> Result<(), ComputeError> {
        self.0.observe_effective(&b.units)
    }
}
impl eredu_runtime::ActivationObserver<MlxTensor, ComputeError> for ProviderCapture {
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), ComputeError> {
        Ok(())
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<MlxTensor>>, ComputeError> {
        assert_eq!(path, "fixture.bank");
        Ok(Some(self))
    }
}
fn verify_provider(device: DeviceType) {
    use eredu_runtime::{RoutedExpertProvider, TensorParallelRoutedExpertProvider};
    let execution = ExecutionContext::new(Device::new(device, 0));
    let stream = execution.stream();
    let (input, routes) = inputs(67);
    for gated in [true, false] {
        for tp in [false, true] {
            for edit in [Edit::None, Edit::Keep, Edit::Add] {
                let mut module = bank(gated, 0..UNITS, stream);
                let mut capture = ProviderCapture(Observer::new(67, gated, 0, edit));
                let id = eredu_runtime::RoutedBankId::new(37);
                let mut banks = eredu_runtime::RoutedBankProviders::new([(
                    id,
                    Box::new(eredu_runtime::ResidentExpertProvider),
                )])
                .unwrap();
                let mut provider = eredu_runtime::ObservedExpertProvider::<_, _, ComputeError>::new(
                    &mut banks,
                    &mut capture,
                    eredu_runtime::RoutedObservationPoints::new(id, format_args!("{}", "fixture.bank"), GROUPS as i32, None).unwrap(),
                );
                let request = eredu_runtime::RoutedExpertRequest {
                    bank: id,
                    layer: 0,
                    input: &input,
                    routes: &routes,
                    pass: eredu_runtime::ExpertPass::Prefill,
                    unit_observer: None,
                };
                let actual = match (&mut module, tp) {
                    (Bank::Gated(bank), false) => {
                        RoutedExpertProvider::<MlxNeuralBackend>::forward_grouped(
                            &mut provider,
                            bank,
                            request,
                            stream,
                        )
                        .unwrap()
                    }
                    (Bank::Relu(bank), false) => {
                        RoutedExpertProvider::<MlxNeuralBackend>::forward_relu2_routed(
                            &mut provider,
                            bank,
                            request,
                            stream,
                        )
                        .unwrap()
                    }
                    (bank, true) => {
                        let output = match bank {
                            Bank::Gated(bank) => TensorParallelRoutedExpertProvider::<
                                MlxNeuralBackend,
                            >::forward_grouped_tensor_parallel(
                                &mut provider,
                                bank,
                                request,
                                1,
                                stream,
                            )
                            .unwrap(),
                            Bank::Relu(bank) => TensorParallelRoutedExpertProvider::<
                                MlxNeuralBackend,
                            >::forward_relu2_routed_tensor_parallel(
                                &mut provider,
                                bank,
                                request,
                                1,
                                stream,
                            )
                            .unwrap(),
                        };
                        match output {
                            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(output) => {
                                output
                            }
                            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(output) => {
                                let (v, b) = output.into_parts();
                                if let Some(b) = b {
                                    v.add(&b, stream).unwrap()
                                } else {
                                    v
                                }
                            }
                        }
                    }
                };
                close(&actual, &reference(67, gated, edit), 2e-5);
                capture.0.finish();
            }
        }
    }
}
#[test]
#[ignore = "explicit native routed-unit provider validation; run outside the sandbox"]
fn grouped_unit_provider_cpu() {
    verify_provider(DeviceType::Cpu)
}
#[test]
#[ignore = "explicit native routed-unit provider validation; run outside the sandbox"]
fn grouped_unit_provider_metal() {
    verify_provider(DeviceType::Gpu)
}
