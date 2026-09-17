use super::*;
use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, Index, LinearFormatSpec, LinearOperator, LinearSpec,
    NeuralBackend, ParameterMetadata, ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
};

#[derive(Clone, Copy, Debug)]
enum Case {
    Matmul,
    Linear(bool),
    Projection(bool),
    Tied,
    LastTied,
}
fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn meta(shape: &[i32], context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        context,
    )
    .unwrap()
}
struct Bind<'w, T> {
    weight: &'w T,
    bias: &'w T,
}
impl<'a, T: Tensor + 'a> ParameterVisitorMut<'a, T> for Bind<'_, T> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut T) {
        *value = if metadata.id.as_str() == "matrix.bias" {
            self.bias
        } else {
            self.weight
        }
        .clone();
    }
}
struct Product<B: NeuralBackend> {
    case: Case,
    projection: Option<B::Linear>,
    embedding: Option<B::Embedding>,
}
impl<B: NeuralBackend> Product<B> {
    fn new(
        case: Case,
        weight: &B::Tensor,
        bias: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Self {
        let format = || LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
        let mut bind = Bind { weight, bias };
        let projection = if let Case::Projection(biased) = case {
            let mut value = B::linear(
                LinearSpec {
                    input: weight.shape()[1],
                    output: weight.shape()[0],
                    weight: ParameterSpec::trainable("matrix.weight").unwrap(),
                    bias: biased.then(|| ParameterSpec::trainable("matrix.bias").unwrap()),
                    format: format(),
                },
                context,
            )
            .unwrap();
            value.visit_parameters_mut(&mut bind);
            Some(value)
        } else {
            None
        };
        let embedding = if matches!(case, Case::Tied | Case::LastTied) {
            let mut value = B::embedding(
                EmbeddingSpec {
                    vocabulary: weight.shape()[0],
                    dimensions: weight.shape()[1],
                    weight: ParameterSpec::trainable("matrix.weight").unwrap(),
                    format: format(),
                },
                context,
            )
            .unwrap();
            value.visit_parameters_mut(&mut bind);
            Some(value)
        } else {
            None
        };
        Self {
            case,
            projection,
            embedding,
        }
    }
    fn forward(
        &mut self,
        input: &B::Tensor,
        weight: &B::Tensor,
        bias: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self.case {
            Case::Matmul => B::Tensor::matmul(input, weight, context),
            Case::Linear(biased) => {
                B::Tensor::linear(input, weight, biased.then_some(bias), context)
            }
            Case::Projection(_) => self.projection.as_mut().unwrap().forward(input, context),
            Case::Tied => self.embedding.as_mut().unwrap().as_linear(input, context),
            Case::LastTied => {
                let rank = input.shape().len();
                let mut index = vec![Index::Full; rank];
                let length = input.shape()[rank - 2];
                index[rank - 2] = Index::Range(length - 1, length);
                self.embedding
                    .as_mut()
                    .unwrap()
                    .as_linear(&input.index(&index, context)?, context)
            }
        }
    }
}

fn matmul_shapes() -> Vec<(Vec<i32>, Vec<i32>)> {
    [
        (&[7][..], &[7][..]),
        (&[7], &[7, 13]),
        (&[3, 7], &[7]),
        (&[1, 128], &[128, 7]),
        (&[3, 128], &[128, 1]),
        (&[2, 5], &[5, 3]),
        (&[2, 127], &[127, 3]),
        (&[2, 128], &[128, 3]),
        (&[2, 129], &[129, 3]),
        (&[17, 1025], &[1025, 33]),
        (&[2, 8192], &[8192, 3]),
        (&[2, 131073], &[131073, 2]),
        (&[2, 3, 32], &[32, 7]),
        (&[32], &[2, 32, 7]),
        (&[2, 3, 32], &[1, 32, 7]),
        (&[2, 3, 128], &[1, 128, 7]),
        (&[2, 1, 3, 129], &[1, 3, 129, 5]),
        (&[2, 3, 0], &[0, 7]),
        (&[0, 3, 32], &[32, 7]),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_vec(), b.to_vec()))
    .collect()
}
fn projection_shapes() -> Vec<Vec<i32>> {
    vec![
        vec![32],
        vec![1, 32],
        vec![7, 32],
        vec![2, 3, 32],
        vec![3, 129],
        vec![2, 1025],
        vec![2, 8192],
        vec![0, 32],
    ]
}

#[test]
fn dense_equations_trace_vectors_broadcasts_tied_selection_and_empty_products() {
    for (left, right) in matmul_shapes() {
        let context = WorkspaceContext::new(mechanisms());
        let a = meta(&left, &context);
        let b = meta(&right, &context);
        let bias = meta(&[1], &context);
        let output = Product::<WorkspaceBackend>::new(Case::Matmul, &b, &bias, &context)
            .forward(&a, &b, &bias, &context)
            .unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(
            report.tensor_buffers.total_bytes.unwrap()
                >= report.tensor_buffers.retained_bytes.unwrap()
        );
        assert!(report.tensor_buffers.retained_bytes.unwrap() > 0);
    }
    for shape in projection_shapes() {
        for case in [
            Case::Linear(false),
            Case::Linear(true),
            Case::Projection(false),
            Case::Projection(true),
            Case::Tied,
        ] {
            let context = WorkspaceContext::new(mechanisms());
            let a = meta(&shape, &context);
            let w = meta(&[7, *shape.last().unwrap()], &context);
            let b = meta(&[7], &context);
            let output = Product::<WorkspaceBackend>::new(case, &w, &b, &context)
                .forward(&a, &w, &b, &context)
                .unwrap();
            assert!(context
                .report(&[output])
                .unwrap()
                .tensor_buffers
                .total_bytes
                .is_some());
        }
    }
    let quote = |last| {
        let context = WorkspaceContext::new(mechanisms());
        let input = meta(&[2, 3000, 32], &context);
        let weight = meta(&[248320, 32], &context);
        let bias = meta(&[248320], &context);
        let case = if last { Case::LastTied } else { Case::Tied };
        let output = Product::<WorkspaceBackend>::new(case, &weight, &bias, &context)
            .forward(&input, &weight, &bias, &context)
            .unwrap();
        assert_eq!(output.shape(), [2, if last { 1 } else { 3000 }, 248320]);
        context
            .report(&[output])
            .unwrap()
            .tensor_buffers
            .total_bytes
            .unwrap()
    };
    assert!(quote(true) < quote(false) / 50);
}

#[test]
fn split_k_prices_odd_small_nax_and_unbounded_large_nax_partition_counts() {
    for (m, n, k, expected) in [
        (1, 2, 131073, 0),
        (2, 1, 131073, 0),
        (2, 3, 4, 0),
        (2, 2, 5, 3),
        (2, 2, 128, 8),
        (2, 2, 129, 8),
        (32, 32, 512, 32),
        (512, 512, 512, 2),
        (1024, 1024, 2048, 0),
        (1024, 1024, 2049, 2),
        (2, 2, 131073, 33),
        (2048, 2048, 8192, 2),
    ] {
        assert_eq!(split_partitions(m, n, k).unwrap(), expected, "{m}/{n}/{k}");
    }
    assert!(split_partitions(u64::MAX, 2, 128).is_err());
}

#[test]
fn malformed_dense_and_packed_or_unpriced_collective_products_cannot_acquire_authority() {
    let f = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    let mut operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Matmul,
        inputs: vec![f(&[3, 7]), f(&[7, 13])],
        outputs: vec![f(&[3, 12])],
    };
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.outputs[0] = f(&[3, 13]);
    operation.inputs[1] = f(&[8, 13]);
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.kind = WorkspaceOperationKind::Projection(
        LinearFormatSpec::scaled(
            LinearFormat::MxFp4,
            ParameterSpec::trainable("matrix.scales").unwrap(),
        )
        .unwrap(),
    );
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.kind = WorkspaceOperationKind::RowParallelProjection(
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        2,
    );
    assert!(mechanisms().operation_bound(&operation).unwrap().is_none());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    #[derive(Clone, Copy, Debug)]
    enum Layout {
        Contiguous,
        Strided,
        Transposed,
    }

    fn tensor(
        shape: &[i32],
        dtype: Dtype,
        layout: Layout,
        seed: usize,
        stream: &Stream,
    ) -> MlxTensor {
        let count = elements(shape).unwrap() as usize;
        let values = (0..count
            * if matches!(layout, Layout::Strided) {
                2
            } else {
                1
            })
            .map(|i| (((i * 7 + seed) % 17) as i32 - 8) as f32 / 64.0)
            .collect::<Vec<_>>();
        let mut value = Array::from_slice(&values, &[values.len() as i32])
            .as_dtype(dtype, stream)
            .unwrap();
        if matches!(layout, Layout::Strided) {
            value = value.try_index_device((..).stride_by(2), stream).unwrap();
        }
        if matches!(layout, Layout::Transposed) {
            let reversed = shape.iter().rev().copied().collect::<Vec<_>>();
            let axes = (0..shape.len() as i32).rev().collect::<Vec<_>>();
            MlxTensor::from_array(
                value
                    .reshape(&reversed, stream)
                    .unwrap()
                    .transpose_axes(&axes, stream)
                    .unwrap(),
            )
        } else {
            MlxTensor::from_array(value.reshape(shape, stream).unwrap())
        }
    }
    // Independent scalar dot products with explicit broadcast indexing. Values
    // are binary fractions representable in all tested dtypes, including BF16.
    fn reference(
        case: Case,
        a: &MlxTensor,
        b: &MlxTensor,
        bias: &MlxTensor,
        stream: &Stream,
    ) -> Vec<f64> {
        let av = a.to_f32_vec(stream).unwrap();
        let bv = b.to_f32_vec(stream).unwrap();
        if !matches!(case, Case::Matmul) {
            let k = *a.shape().last().unwrap() as usize;
            let n = b.shape()[0] as usize;
            let bias = bias.to_f32_vec(stream).unwrap();
            let rows = elements(&a.shape()[..a.shape().len() - 1]).unwrap() as usize;
            return (0..rows)
                .filter(|row| {
                    !matches!(case, Case::LastTied)
                        || row % a.shape()[a.shape().len() - 2] as usize
                            == a.shape()[a.shape().len() - 2] as usize - 1
                })
                .flat_map(|row| {
                    let av = &av;
                    let bv = &bv;
                    let bias = &bias;
                    (0..n).map(move |col| {
                        let dot = (0..k)
                            .map(|j| av[row * k + j] as f64 * bv[col * k + j] as f64)
                            .sum::<f64>();
                        dot + if matches!(case, Case::Linear(true) | Case::Projection(true)) {
                            bias[col] as f64
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
        }
        let g = Geometry::new(a.shape(), b.shape()).unwrap();
        let mut ashape = a.shape().to_vec();
        if ashape.len() == 1 {
            ashape.insert(0, 1);
        }
        let mut bshape = b.shape().to_vec();
        if bshape.len() == 1 {
            bshape.push(1);
        }
        let rank = ashape.len().max(bshape.len()) - 2;
        let batch_shape = g.output.dimensions().take(rank).collect::<Vec<_>>();
        let offset = |shape: &[i32], mut batch: usize| {
            let mut index = 0;
            let mut stride = 1;
            for axis in (0..rank).rev() {
                let coord = batch % batch_shape[axis] as usize;
                batch /= batch_shape[axis] as usize;
                if axis + shape.len() >= rank + 2 {
                    let size = shape[axis + shape.len() - rank - 2] as usize;
                    if size != 1 {
                        index += coord * stride;
                    }
                    stride *= size;
                }
            }
            index * (shape[shape.len() - 2] * shape[shape.len() - 1]) as usize
        };
        let mut result = Vec::new();
        for batch in 0..g.batches as usize {
            for row in 0..g.m as usize {
                for col in 0..g.n as usize {
                    result.push(
                        (0..g.k as usize)
                            .map(|j| {
                                av[offset(&ashape, batch) + row * g.k as usize + j] as f64
                                    * bv[offset(&bshape, batch) + j * g.n as usize + col] as f64
                            })
                            .sum(),
                    );
                }
            }
        }
        result
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_dense_products_fit_bounds_and_independent_dot_products() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = matmul_shapes()
            .into_iter()
            .map(|(a, b)| (Case::Matmul, a, b))
            .collect::<Vec<_>>();
        for shape in projection_shapes() {
            for case in [
                Case::Linear(false),
                Case::Linear(true),
                Case::Projection(false),
                Case::Projection(true),
                Case::Tied,
            ] {
                cases.push((case, shape.clone(), vec![7, *shape.last().unwrap()]));
            }
        }
        cases.push((Case::LastTied, vec![2, 5, 32], vec![13, 32]));
        for shape in [vec![0], vec![2, 3, 0]] {
            for biased in [false, true] {
                cases.push((Case::Linear(biased), shape.clone(), vec![7, 0]));
            }
        }
        for (dtype, weight_dtype) in [
            (Dtype::Float32, Dtype::Float32),
            (Dtype::Float16, Dtype::Float16),
            (Dtype::Bfloat16, Dtype::Bfloat16),
            (Dtype::Float16, Dtype::Bfloat16),
            (Dtype::Bfloat16, Dtype::Float32),
        ] {
            for layout in [Layout::Contiguous, Layout::Strided, Layout::Transposed] {
                for (case, ashape, bshape) in &cases {
                    let context = WorkspaceContext::new(selected);
                    let ma = meta(ashape, &context);
                    let mb = meta(bshape, &context);
                    let n = if matches!(case, Case::Matmul) {
                        1
                    } else {
                        bshape[0]
                    };
                    let mc = meta(&[n], &context);
                    let output = Product::<WorkspaceBackend>::new(*case, &mb, &mc, &context)
                        .forward(&ma, &mb, &mc, &context)
                        .unwrap();
                    let allowed = context
                        .report(&[])
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .unwrap();
                    let a = tensor(ashape, dtype, layout, 3, &stream);
                    let b = tensor(bshape, weight_dtype, layout, 5, &stream);
                    let bias = tensor(&[n], weight_dtype, layout, 7, &stream);
                    let mut product = Product::<MlxNeuralBackend>::new(*case, &b, &bias, &stream);
                    safemlx::transforms::eval([a.as_array(), b.as_array(), bias.as_array()])
                        .unwrap();
                    let expected = reference(*case, &a, &b, &bias, &stream);
                    // Retire input preparation and oracle conversion buffers
                    // before taking the paid-existing-residency baseline.
                    stream.synchronize().unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let actual = product.forward(&a, &b, &bias, &stream).unwrap();
                    safemlx::transforms::eval([actual.as_array()]).unwrap();
                    stream.synchronize().unwrap();
                    let observed = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    assert_eq!(actual.shape(), output.shape());
                    assert!(
                        observed <= allowed,
                        "{case:?} {ashape:?}/{bshape:?} peak {observed} exceeds {allowed}"
                    );
                    let values = actual.to_f32_vec(&stream).unwrap();
                    assert_eq!(values.len(), expected.len());
                    let mut max_error = 0f64;
                    for (actual, expected) in values.iter().zip(expected) {
                        let error = (*actual as f64 - expected).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error <= 0.002 + 0.01 * expected.abs(),
                            "{case:?}: {actual} != {expected}"
                        );
                    }
                    eprintln!("matrix dtype={dtype:?}/{weight_dtype:?} layout={layout:?} a={ashape:?} b={bshape:?} case={case:?} observed={observed} bound={allowed} max_error={max_error}");
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn source_representation_controls_dense_projection_and_owned_borrowed_facts() {
    use WorkspaceFloatingType as F;
    use eredu_nn::ParameterId;

    #[derive(Clone, Copy, Debug)]
    enum Source {
        Missing,
        Qualified,
        Contradictory,
        WrongShape,
        Noncontiguous,
        WidenedInput,
        WidenedMissing,
        WidenedNoncontiguous,
    }
    let quote = |source: Source, width: i32, outputs: i32| {
        let selected = mechanisms();
        let context = WorkspaceContext::new_recording_facts(selected);
        let row = |name: &str, shape: &[i32], dtype, contiguous| {
            WorkspaceParameterRepresentation::new(
                ParameterId::new(name).unwrap(),
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, contiguous))),
            )
        };
        // The shared embedding/linear constructors must consume the actual
        // installed rows. No replacement tensor is bound over their placeholders.
        let mut rows = vec![row("source.embedding", &[5, width], F::Bfloat16, true)];
        if !matches!(source, Source::Missing | Source::WidenedMissing) {
            let weight_shape = if matches!(source, Source::WrongShape) {
                [outputs, width * 2]
            } else {
                [outputs, width]
            };
            rows.push(row(
                "matrix.weight",
                &weight_shape,
                F::Bfloat16,
                !matches!(source, Source::Noncontiguous | Source::WidenedNoncontiguous),
            ));
        }
        if matches!(source, Source::Contradictory) {
            rows.push(row("matrix.weight", &[outputs, width], F::Float32, true));
        }
        context.install_parameter_representations(rows).unwrap();
        let format = || LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
        let mut embedding = WorkspaceBackend::embedding(
            EmbeddingSpec {
                vocabulary: 5,
                dimensions: width,
                weight: ParameterSpec::trainable("source.embedding").unwrap(),
                format: format(),
            },
            &context,
        )
        .unwrap();
        let mut projection = WorkspaceBackend::linear(
            LinearSpec {
                input: width,
                output: outputs,
                weight: ParameterSpec::trainable("matrix.weight").unwrap(),
                bias: None,
                format: format(),
            },
            &context,
        )
        .unwrap();
        let tokens = WorkspaceTensor::full_i32(2, &[1, 1], &context).unwrap();
        let input = embedding.forward(&tokens, &context).unwrap();
        let input = WorkspaceBackend::silu(input, &context).unwrap();
        assert_eq!(
            input.layout().representation().unwrap().dtype(),
            F::Bfloat16
        );
        let widened = matches!(
            source,
            Source::WidenedInput | Source::WidenedMissing | Source::WidenedNoncontiguous
        );
        let input = if widened {
            // This is the real shared scalar operation, whose native input is
            // an explicit F32 array. A logical Float32 layout alone cannot
            // distinguish this activation from the prior BF16 value.
            input.multiply_scalar(0.5, &context).unwrap()
        } else {
            input
        };
        let expected_input = if widened { F::Float32 } else { F::Bfloat16 };
        assert_eq!(
            input.layout().representation().unwrap().dtype(),
            expected_input
        );
        let output = projection.forward(&input, &context).unwrap();
        assert_eq!(output.shape(), &[1, 1, outputs]);
        let report = context.finish_report(&[output]).unwrap();
        let operation = report.operations.last().unwrap();
        assert!(matches!(
            operation.kind,
            WorkspaceOperationKind::Projection(_)
        ));
        let weight = operation.inputs[1].representation();
        match source {
            Source::Missing
            | Source::WidenedMissing
            | Source::Contradictory
            | Source::WrongShape => {
                assert_eq!(weight, None)
            }
            Source::Noncontiguous | Source::WidenedNoncontiguous => {
                assert!(!weight.unwrap().row_contiguous())
            }
            Source::Qualified | Source::WidenedInput => assert!(weight.unwrap().row_contiguous()),
        }

        // Count, caller-owned emission and owning adaptation consume the same
        // traced operation; compare complete effects and text, not a copied
        // implementation formula or a hard-coded byte allowance.
        let counted = selected
            .operation_facts(operation.as_view())
            .unwrap()
            .unwrap();
        let owned = selected.operation_bound(operation).unwrap().unwrap();
        let mut outputs =
            vec![WorkspaceOutputEffect::AliasInput(usize::MAX); counted.layout.outputs];
        let mut aliases = vec![usize::MAX; counted.layout.aliases];
        let mut assumptions = vec![0xFF; counted.layout.assumption_bytes];
        assert_eq!(
            selected
                .write_operation_facts(
                    operation.as_view(),
                    WorkspaceEffectDestination {
                        outputs: &mut outputs,
                        aliases: &mut aliases,
                        assumptions: &mut assumptions,
                    }
                )
                .unwrap(),
            Some(counted)
        );
        assert_eq!(counted.scratch_bytes, owned.scratch_bytes);
        assert_eq!(assumptions, owned.assumptions.as_bytes());
        assert_eq!(outputs.len(), owned.outputs.len());
        for (flat, old) in outputs.iter().zip(&owned.outputs) {
            assert_eq!(flat.as_view(&aliases), Some(old.as_view()));
        }
        (owned.scratch_bytes, owned.outputs)
    };
    let generic = quote(Source::Missing, 32, 128);
    let qualified = quote(Source::Qualified, 32, 128);
    assert_eq!(qualified.1, generic.1);
    assert!(
        qualified.0 < generic.0,
        "proved BF16 source should select its actual row worker"
    );
    let mixed = quote(Source::WidenedInput, 32, 128);
    assert_eq!(mixed.1, generic.1);
    assert!(
        mixed.0 < generic.0,
        "proved mixed matrix layout excludes the second weight compaction"
    );
    assert!(
        mixed.0 > qualified.0,
        "mixed projection must retain its full dtype cast and ordinary matmul temporaries"
    );
    for source in [
        Source::Contradictory,
        Source::WrongShape,
        Source::Noncontiguous,
        Source::WidenedMissing,
        Source::WidenedNoncontiguous,
    ] {
        assert_eq!(
            quote(source, 32, 128),
            generic,
            "{source:?} must preserve the generic union"
        );
    }
    for (width, outputs) in [(32, 1), (1, 128)] {
        assert_eq!(
            quote(Source::WidenedInput, width, outputs),
            quote(Source::WidenedMissing, width, outputs),
            "unused singleton strides need more evidence than row-contiguity",
        );
    }
}
