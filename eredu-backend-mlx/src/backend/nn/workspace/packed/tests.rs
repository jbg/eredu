use super::*;
use eredu_checkpoint::{AffineQuantization, BlockFp8Format};
use eredu_gguf::{Endian, GgmlType};
use eredu_nn::{
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, LinearOperator, LinearSpec,
    NeuralBackend, ParameterMetadata, ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn format(encoding: LinearFormat) -> LinearFormatSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    match encoding {
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            encoding,
            parameter("matrix.scales"),
            parameter("matrix.biases"),
        ),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => {
            LinearFormatSpec::scaled(encoding, parameter("matrix.scales"))
        }
        _ => LinearFormatSpec::unscaled(encoding),
    }
    .unwrap()
}
fn linear<B: NeuralBackend>(
    encoding: LinearFormat,
    k: i32,
    n: i32,
    bias: bool,
    c: &<B::Tensor as Tensor>::Context,
) -> B::Linear {
    B::linear(
        LinearSpec {
            input: k,
            output: n,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: bias.then(|| ParameterSpec::trainable("matrix.bias").unwrap()),
            format: format(encoding),
        },
        c,
    )
    .unwrap()
}
fn embedding<B: NeuralBackend>(
    encoding: LinearFormat,
    k: i32,
    n: i32,
    c: &<B::Tensor as Tensor>::Context,
) -> B::Embedding {
    B::embedding(
        EmbeddingSpec {
            dimensions: k,
            vocabulary: n,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            format: format(encoding),
        },
        c,
    )
    .unwrap()
}
fn ggml_types() -> [GgmlType; 14] {
    [
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ3XXS,
        GgmlType::IQ1S,
        GgmlType::IQ4NL,
        GgmlType::IQ3S,
        GgmlType::IQ2S,
        GgmlType::IQ4XS,
        GgmlType::IQ1M,
    ]
}
fn encodings() -> Vec<LinearFormat> {
    let mut formats = Vec::new();
    for group in [16, 32, 64, 128] {
        for bits in [2, 3, 4, 5, 6, 8] {
            formats.push(LinearFormat::Affine(
                AffineQuantization::new(group, bits).unwrap(),
            ));
        }
    }
    formats.push(LinearFormat::MxFp4);
    for ggml_type in ggml_types() {
        for endian in [Endian::Little, Endian::Big] {
            formats.push(LinearFormat::GgufIQuant { ggml_type, endian });
        }
    }
    for scales in [
        BlockFp8ScaleEncoding::FloatingPoint,
        BlockFp8ScaleEncoding::Ue8m0,
    ] {
        formats.push(LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, scales).unwrap(),
        ));
    }
    formats
}

#[test]
#[cfg(not(feature = "cuda"))]
fn packed_formats_preserve_complete_projection_lookup_and_tied_readout_bounds() {
    for encoding in encodings() {
        for shape in [
            vec![1024],
            vec![1, 1024],
            vec![2, 7, 1024],
            vec![65, 1024],
            vec![0, 1024],
        ] {
            let c = WorkspaceContext::new(selected());
            let input = WorkspaceTensor::existing(
                WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                &c,
            )
            .unwrap();
            let mut projection = linear::<WorkspaceBackend>(encoding, 1024, 37, true, &c);
            c.begin_span();
            let out = projection.forward(&input, &c).unwrap();
            let report = c.report(&[out]).unwrap();
            assert!(
                report.total_bytes.is_some(),
                "{encoding:?}: {:?}",
                report.unpriced_operations
            );
            let expected_host = 0;
            assert_eq!(report.host_workspace_bytes, Some(expected_host));
            assert_eq!(
                report.total_bytes,
                report.tensor_buffers.total_bytes.map(|v| v + expected_host)
            );
            if matches!(encoding, LinearFormat::E4M3BlockFp8(_)) {
                continue;
            }
            let mut table = embedding::<WorkspaceBackend>(encoding, 1024, 37, &c);
            for policy in [
                EmbeddingLookupPolicy::Strict,
                EmbeddingLookupPolicy::ZeroSentinel(-1),
            ] {
                c.begin_span();
                let ids = WorkspaceTensor::existing(
                    WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Int32).unwrap(),
                    &c,
                )
                .unwrap();
                let out = table.lookup(&ids, policy, &c).unwrap();
                let tied = table.as_linear(&input, &c).unwrap();
                let report = c.report(&[out, tied]).unwrap();
                assert!(report.total_bytes.is_some(), "{encoding:?}");
                assert_eq!(report.host_workspace_bytes, Some(0));
            }
        }
    }
}

#[test]
fn packed_geometry_validates_companions_and_split_k_reduction_storage() {
    let encoding = LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap());
    let c = WorkspaceContext::new(selected());
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[32, 16384], WorkspaceDtype::Float32).unwrap(),
        &c,
    )
    .unwrap();
    let mut projection = linear::<WorkspaceBackend>(encoding, 16384, 1, true, &c);
    c.begin_span();
    projection.forward(&input, &c).unwrap();
    let op = c.report(&[]).unwrap().operations.remove(0);
    let r = |n| capacity(selected().allocation, n).unwrap();
    assert_eq!(
        quantized_split_scratch(32, 1, 16384, 32, selected().allocation).unwrap(),
        r(512 * 32) + r(32 * 32)
    );
    assert_eq!(
        quantized_split_scratch(32, 1, 16384, 16, selected().allocation).unwrap(),
        0
    );
    for index in 1..op.inputs.len() {
        let mut wrong = op.clone();
        wrong.inputs[index] = WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap();
        assert!(selected().operation_bound(&wrong).is_err());
        assert!(selected().host_workspace_bound(&wrong).is_err());
    }
    let mut missing = op.clone();
    missing.inputs.remove(2);
    assert!(selected().operation_bound(&missing).is_err());
    let mut unrelated = op;
    unrelated.kind = WorkspaceOperationKind::Elementwise("unpriced_packed_transform");
    assert!(selected().operation_bound(&unrelated).unwrap().is_none());
    assert!(
        selected()
            .host_workspace_bound(&unrelated)
            .unwrap()
            .is_none()
    );
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native;
