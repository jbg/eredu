use super::operation_component_tests::exercise_component;
use super::*;
use crate::{
    MlxTensor,
    backend::nn::shared::MlxNeuralBackend,
    backend::nn::workspace::{MlxMetalWorkspaceMechanisms, OriginalComponentTestPlan},
};
use eredu_checkpoint::{AffineQuantization, LinearFormat};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceTensor};
use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, LinearFormatSpec, NeuralBackend, ParameterMetadata,
    ParameterSpec, ParameterVisitorMut, Parameterized,
};
use safemlx::{Array, Dtype, ops};

fn spec(encoding: LinearFormat) -> EmbeddingSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let format = match encoding {
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            encoding,
            parameter("embedding.scales"),
            parameter("embedding.biases"),
        ),
        LinearFormat::MxFp4 => LinearFormatSpec::scaled(encoding, parameter("embedding.scales")),
        _ => unreachable!(),
    }
    .unwrap();
    EmbeddingSpec {
        vocabulary: 4,
        dimensions: 128,
        weight: parameter("embedding.weight"),
        format,
    }
}
fn plan(encoding: LinearFormat) -> OriginalComponentTestPlan {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let input = WorkspaceTensor::existing(
        context.layout(&[1, 3], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    let mut embedding = WorkspaceBackend::embedding(spec(encoding), &context).unwrap();
    context.begin_span();
    let output = embedding.forward(&input, &context).unwrap();
    OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap())
}
struct Bind<'a> {
    weight: &'a MlxTensor,
    scales: &'a MlxTensor,
    biases: Option<&'a MlxTensor>,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        *value = match metadata.id().as_str() {
            "embedding.weight" => self.weight,
            "embedding.scales" => self.scales,
            "embedding.biases" => self.biases.unwrap(),
            name => panic!("unexpected binding {name}"),
        }
        .clone();
    }
}

#[test]
fn original_packed_embedding_rows_match_ordinary_with_retained_callback() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    // Real cold operation descriptors cover every admitted affine bit/group
    // pair; the three executions below select distinct companion/cast workers.
    for group in [16, 32, 64, 128] {
        for bits in [2, 3, 4, 5, 6, 8] {
            let _ = plan(LinearFormat::Affine(
                AffineQuantization::new(group, bits).unwrap(),
            ));
        }
    }
    let _ = plan(LinearFormat::MxFp4);
    let input = MlxTensor::from_array(Array::from_slice(&[2i32, 0, 3], &[1, 3]));
    for (group, bits, mode, scale_dtype, bias_dtype) in [
        (
            64,
            4,
            ops::QuantizationMode::Affine,
            Dtype::Float32,
            Dtype::Float32,
        ),
        (
            32,
            3,
            ops::QuantizationMode::Affine,
            Dtype::Bfloat16,
            Dtype::Float16,
        ),
        (
            32,
            4,
            ops::QuantizationMode::MxFp4,
            Dtype::Uint8,
            Dtype::Uint8,
        ),
    ] {
        let encoding = match mode {
            ops::QuantizationMode::Affine => {
                LinearFormat::Affine(AffineQuantization::new(group, bits).unwrap())
            }
            ops::QuantizationMode::MxFp4 => LinearFormat::MxFp4,
        };
        let values: Vec<f32> = (0..4 * 128)
            .map(|i| ((i * 13 % 47) as f32 - 23.0) / 32.0)
            .collect();
        let dense = Array::from_slice(&values, &[4, 128]);
        let packed = ops::quantize_with_mode(&dense, group, bits, mode, &stream).unwrap();
        let weight = MlxTensor::from_array(packed.weight);
        let scales = MlxTensor::from_array(packed.scales.as_dtype(scale_dtype, &stream).unwrap());
        let biases = packed
            .biases
            .map(|bias| MlxTensor::from_array(bias.as_dtype(bias_dtype, &stream).unwrap()));
        let mut embedding = MlxNeuralBackend::embedding(spec(encoding), &stream).unwrap();
        embedding.visit_parameters_mut(&mut Bind {
            weight: &weight,
            scales: &scales,
            biases: biases.as_ref(),
        });
        let ordinary = embedding.forward(&input, &stream).unwrap().into_array();
        assert_eq!(ordinary.shape(), &[1, 3, 128]);
        assert_eq!(ordinary.dtype(), Dtype::Float32);
        let expected = ordinary.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        // Execute the real strict operator. Its retained predicate is an exact
        // second completion root, checked before the numerical result escapes.
        let mut leaves = vec![input.as_array(), weight.as_array(), scales.as_array()];
        leaves.extend(biases.iter().map(MlxTensor::as_array));
        exercise_component(&mut prepared, &plan(encoding), &leaves, &expected, || {
            embedding.forward(&input, &stream).unwrap().into_array()
        });
    }
    drop((input, stream));
    prepared.finish();
}
