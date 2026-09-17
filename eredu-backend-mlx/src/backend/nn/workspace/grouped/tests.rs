use super::*;
use eredu_checkpoint::{AffineQuantization, BlockFp8Format};
use eredu_nn::{
    GatedProductPolicy, GroupSelection, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedLinearOperator, GroupedLinearSpec, GroupedNeuralBackend, GroupedRelu2Operator,
    GroupedRelu2Spec, GroupedUnitBatch, GroupedUnitObserver, ParameterSpec, Tensor,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedRelu2Operator,
};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn projection(name: &str, encoding: LinearFormat, bias: bool) -> GroupedProjectionSpec {
    let p = |suffix: &str| ParameterSpec::trainable(format!("{name}.{suffix}")).unwrap();
    let format = match encoding {
        LinearFormat::Affine(_) => LinearFormatSpec::affine(encoding, p("scales"), p("biases")),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => {
            LinearFormatSpec::scaled(encoding, p("scales"))
        }
        _ => LinearFormatSpec::unscaled(encoding),
    }
    .unwrap();
    GroupedProjectionSpec::new(p("weight"), bias.then(|| p("bias")), format).unwrap()
}
#[derive(Clone, Copy, Debug)]
enum Kind {
    Linear(bool),
    Gated(u8, bool),
    Relu2,
}
fn specification(
    kind: Kind,
    groups: i32,
    width: i32,
    units: i32,
    encoding: LinearFormat,
    bias: bool,
) -> WorkspaceGroupedBank {
    match kind {
        Kind::Linear(activate) => WorkspaceGroupedBank::Linear(
            GroupedLinearSpec::new(
                groups,
                width,
                units,
                if activate {
                    GroupedLinearActivation::Silu
                } else {
                    GroupedLinearActivation::Identity
                },
                projection("read", encoding, bias),
            )
            .unwrap(),
        ),
        Kind::Gated(policy, sequential) => WorkspaceGroupedBank::GatedProduct(
            GroupedGatedProductSpec::new(
                groups,
                width,
                units,
                width,
                match policy {
                    0 => GatedProductPolicy::ordinary_silu(),
                    1 => GatedProductPolicy::ordinary_gelu_approximate(),
                    _ => GatedProductPolicy::new(
                        eredu_nn::GatedProductActivation::Silu,
                        Some(0.2),
                        Some(0.3),
                        1.7,
                        0.1,
                    )
                    .unwrap(),
                },
                GatedProductGroupLayout::Packed {
                    gate_up: projection("read", encoding, bias),
                    down: projection("write", encoding, bias),
                },
            )
            .unwrap()
            .with_reduction(if sequential {
                GroupReduction::SequentialGroupOrder
            } else {
                GroupReduction::Sum
            }),
        ),
        Kind::Relu2 => WorkspaceGroupedBank::Relu2(
            GroupedRelu2Spec::new(
                groups,
                width,
                units,
                projection("read", encoding, false),
                projection("write", encoding, false),
            )
            .unwrap(),
        ),
    }
}
enum Executable<B: GroupedNeuralBackend> {
    Linear(B::LinearGroups),
    Gated(B::GatedProductGroups),
    Relu2(B::Relu2Groups),
}
impl<B: GroupedNeuralBackend> Executable<B>
where
    B::GatedProductGroups: TensorParallelGroupedGatedProductOperator<B::Tensor>,
    B::Relu2Groups: TensorParallelGroupedRelu2Operator<B::Tensor>,
{
    fn new(spec: &WorkspaceGroupedBank, c: &<B::Tensor as Tensor>::Context) -> Self {
        match spec {
            WorkspaceGroupedBank::Linear(s) => {
                Self::Linear(B::grouped_linear_bank(s.clone(), c).unwrap())
            }
            WorkspaceGroupedBank::GatedProduct(s) => {
                Self::Gated(B::grouped_gated_product(s.clone(), c).unwrap())
            }
            WorkspaceGroupedBank::Relu2(s) => Self::Relu2(B::grouped_relu2(s.clone(), c).unwrap()),
        }
    }
    fn forward(
        &mut self,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: Option<usize>,
        observer: Option<&mut dyn GroupedUnitObserver<B::Tensor>>,
        c: &<B::Tensor as Tensor>::Context,
    ) -> Vec<B::Tensor> {
        let out = match self {
            Self::Linear(m) => return vec![m.forward_grouped(input, routes, c).unwrap()],
            Self::Gated(m) => {
                if let Some(p) = partitions {
                    m.forward_grouped_tensor_parallel_with_unit_observer(
                        input, routes, p, c, observer,
                    )
                    .unwrap()
                } else {
                    return vec![
                        m.forward_grouped_with_unit_observer(input, routes, c, observer)
                            .unwrap(),
                    ];
                }
            }
            Self::Relu2(m) => {
                if let Some(p) = partitions {
                    m.forward_grouped_tensor_parallel_with_unit_observer(
                        input, routes, p, c, observer,
                    )
                    .unwrap()
                } else {
                    return vec![
                        m.forward_grouped_with_unit_observer(input, routes, c, observer)
                            .unwrap(),
                    ];
                }
            }
        };
        let (partial, bias) = out.into_parts();
        std::iter::once(partial).chain(bias).collect()
    }
}
#[derive(Debug, PartialEq)]
struct Delivery {
    phase: u8,
    offset: usize,
    total: usize,
    units: Vec<i32>,
    coefficients: Vec<i32>,
}
struct Scale<'a, T: Tensor> {
    c: &'a T::Context,
    deliveries: Vec<Delivery>,
}
impl<T: Tensor> Scale<'_, T> {
    fn record(&mut self, phase: u8, batch: &GroupedUnitBatch<'_, T>) {
        self.deliveries.push(Delivery {
            phase,
            offset: batch.token_offset,
            total: batch.total_token_count,
            units: batch.values.shape().to_vec(),
            coefficients: batch.coefficients.shape().to_vec(),
        });
    }
}
impl<T: Tensor> GroupedUnitObserver<T> for Scale<'_, T> {
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error> {
        self.record(0, batch);
        assert_eq!(batch.values.shape()[0], batch.group_indices.shape()[0]);
        assert!(
            batch.token_offset + batch.coefficients.shape()[0] as usize <= batch.total_token_count
        );
        Ok(())
    }
    fn intervene(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        self.record(1, batch);
        Ok(Some(batch.values.multiply_scalar(0.5, self.c)?))
    }
    fn observe_effective(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error> {
        self.record(2, batch);
        Ok(())
    }
}
fn quote(
    spec: &WorkspaceGroupedBank,
    shape: &[i32],
    k: i32,
    parallel: bool,
    observe: bool,
) -> WorkspaceTraceReport {
    quote_with_deliveries(spec, shape, k, parallel, observe).0
}
fn quote_with_deliveries(
    spec: &WorkspaceGroupedBank,
    shape: &[i32],
    k: i32,
    parallel: bool,
    observe: bool,
) -> (WorkspaceTraceReport, Vec<Delivery>) {
    let c = WorkspaceContext::new(selected());
    let mut executable = Executable::<WorkspaceBackend>::new(spec, &c);
    let x = WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        &c,
    )
    .unwrap();
    let rows = shape.iter().product::<i32>() / shape.last().unwrap();
    let ids = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[rows, k], WorkspaceDtype::Uint32).unwrap(),
        &c,
    )
    .unwrap();
    let coefficients = WorkspaceTensor::unloaded_f32(&[rows, k], &c).unwrap();
    let routes = GroupSelection::new(ids, coefficients.clone(), coefficients);
    let mut observer = Scale {
        c: &c,
        deliveries: vec![],
    };
    c.begin_span();
    let output = executable.forward(
        &x,
        &routes,
        parallel.then_some(2),
        observe.then_some(&mut observer),
        &c,
    );
    (c.report(&output).unwrap(), observer.deliveries)
}
fn formats() -> Vec<LinearFormat> {
    let mut formats = vec![LinearFormat::Dense, LinearFormat::MxFp4];
    for group_size in [16, 32, 64, 128] {
        for bits in [2, 3, 4, 5, 6, 8] {
            formats.push(LinearFormat::Affine(
                AffineQuantization::new(group_size, bits).unwrap(),
            ));
        }
    }
    for ggml_type in [
        eredu_gguf::GgmlType::Q4K,
        eredu_gguf::GgmlType::Q5K,
        eredu_gguf::GgmlType::Q6K,
        eredu_gguf::GgmlType::Q5_1,
        eredu_gguf::GgmlType::Q8_0,
        eredu_gguf::GgmlType::IQ2XXS,
        eredu_gguf::GgmlType::IQ2XS,
        eredu_gguf::GgmlType::IQ3XXS,
        eredu_gguf::GgmlType::IQ1S,
        eredu_gguf::GgmlType::IQ4NL,
        eredu_gguf::GgmlType::IQ3S,
        eredu_gguf::GgmlType::IQ2S,
        eredu_gguf::GgmlType::IQ4XS,
        eredu_gguf::GgmlType::IQ1M,
    ] {
        for endian in [eredu_gguf::Endian::Little, eredu_gguf::Endian::Big] {
            formats.push(LinearFormat::GgufIQuant { ggml_type, endian });
        }
    }
    for scale_encoding in [
        BlockFp8ScaleEncoding::FloatingPoint,
        BlockFp8ScaleEncoding::Ue8m0,
    ] {
        formats.push(LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, scale_encoding).unwrap(),
        ));
    }
    formats
}

#[test]
fn grouped_workspace_composes_packed_formats_chunks_bias_partials_and_unit_boundaries() {
    let mut cases = 0;
    for encoding in formats() {
        for kind in [
            Kind::Linear(false),
            Kind::Linear(true),
            Kind::Gated(0, false),
            Kind::Gated(1, true),
            Kind::Gated(2, false),
            Kind::Relu2,
        ] {
            if matches!(kind, Kind::Relu2) && matches!(encoding, LinearFormat::E4M3BlockFp8(_)) {
                continue;
            }
            for tokens in [0, 1, 64, 65, 129] {
                let spec = specification(kind, 4, 256, 256, encoding, true);
                let parallel = !matches!(kind, Kind::Linear(_));
                let report = quote(&spec, &[tokens, 256], 2, parallel, false);
                assert!(
                    report.total_bytes.is_some(),
                    "{kind:?} {encoding:?} tokens={tokens} {:?}",
                    report.unpriced_operations
                );
                let expected = 0;
                assert_eq!(report.host_workspace_bytes, Some(expected));
                cases += 1;
                if parallel {
                    let report = quote(&spec, &[tokens, 256], 2, true, true);
                    assert!(
                        report.total_bytes.is_some(),
                        "units {kind:?} {encoding:?} {tokens}: {:?}",
                        report.unpriced_operations
                    );
                    assert!(report.operations.iter().any(|op| matches!(
                        op.kind,
                        WorkspaceOperationKind::Grouped {
                            phase: WorkspaceGroupedPhase::Units,
                            ..
                        }
                    )));
                    cases += 1;
                }
            }
        }
    }
    eprintln!("GROUPED_WORKSPACE_COLD_CASES={cases}");
}

#[test]
fn grouped_workspace_prices_chunked_observers_and_preserves_independent_provider_gap() {
    let s = specification(Kind::Gated(0, false), 4, 32, 32, LinearFormat::Dense, false);
    let report = quote(&s, &[65, 32], 2, false, true);
    assert!(report.total_bytes.is_some());
    assert!(report.unpriced_operations.is_empty());
    let independent = GroupedGatedProductSpec::new(
        2,
        32,
        32,
        32,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Independent(
            (0..2)
                .map(|i| {
                    eredu_nn::GatedProductGroupParameters::new(
                        projection(&format!("g{i}.gate"), LinearFormat::Dense, false),
                        projection(&format!("g{i}.up"), LinearFormat::Dense, false),
                        projection(&format!("g{i}.down"), LinearFormat::Dense, false),
                    )
                })
                .collect(),
        ),
    )
    .unwrap();
    assert!(
        quote(
            &WorkspaceGroupedBank::GatedProduct(independent),
            &[1, 32],
            2,
            false,
            false
        )
        .total_bytes
        .is_none()
    );
}

#[test]
fn grouped_workspace_validates_native_parameter_and_selection_geometry() {
    let s = specification(Kind::Gated(0, false), 4, 32, 32, LinearFormat::Dense, true);
    let report = quote(&s, &[2, 32], 2, true, false);
    let mut op = report.operations[0].clone();
    op.inputs.pop();
    assert!(selected().operation_bound(&op).is_err());
    let mut op = report.operations[0].clone();
    op.inputs[1] = WorkspaceLayout::new(&[3, 2], WorkspaceDtype::Uint32).unwrap();
    assert!(selected().operation_bound(&op).is_err());
    let mut op = report.operations[0].clone();
    op.inputs[0] = WorkspaceLayout::new(&[2, 32], WorkspaceDtype::Int32).unwrap();
    assert!(selected().operation_bound(&op).unwrap().is_none());
    let mut op = report.operations[0].clone();
    op.outputs.pop();
    assert!(selected().operation_bound(&op).is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native;
