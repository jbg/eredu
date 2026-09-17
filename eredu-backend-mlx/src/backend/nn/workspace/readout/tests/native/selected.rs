//! Native selected readout uses the same tied/untied physical modules as models.
use super::*;
use crate::backend::nn::linear::{unloaded_embedding,PhysicalEmbedding,PhysicalLinear};
use eredu_architectures::readout::execute_readout;
use eredu_checkpoint::{AffineQuantization,LinearFormat};
use eredu_core::OutputDemand;
use eredu_nn::{LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec};

const WIDTH:i32=64;
const VOCAB:i32=32;
const POSITIONS:i32=5;
const GROUP:i32=32;
const BITS:i32=4;

enum Projection {Tied(PhysicalEmbedding),Untied(PhysicalLinear)}
impl Projection {
    fn new(tied:bool,format:LinearFormat,weight:&Array,scales:Option<&Array>,
        biases:Option<&Array>,stream:&Stream)->Self {
        if tied {
            let mut module=unloaded_embedding(VOCAB,WIDTH,format.weight_quantization(),stream).unwrap();
            match &mut module {
                PhysicalEmbedding::Dense(module)=>module.weight.value=weight.clone(),
                PhysicalEmbedding::Quantized(module)=>{
                    module.inner.weight.value=weight.clone();
                    module.scales.value=scales.cloned();module.biases.value=biases.cloned();
                },
            }
            Self::Tied(module)
        } else {
            let mut module=PhysicalLinear::unloaded(WIDTH,VOCAB,false,format,stream).unwrap();
            module.weight.value=weight.clone();module.scales.value=scales.cloned();
            module.biases.value=biases.cloned();Self::Untied(module)
        }
    }
    fn forward(&mut self,hidden:&MlxTensor,stream:&Stream)->MlxTensor {
        MlxTensor::from_array(match self {
            Self::Tied(module)=>module.as_linear(hidden.as_array(),stream),
            Self::Untied(module)=>module.forward(hidden.as_array(),stream),
        }.unwrap())
    }
}

#[test]
#[ignore="requires Metal; native shared readout dtype and actual affine storage parity"]
fn native_selected_readout_preserves_half_and_affine_tied_untied_projection() {
    let stream=Stream::new_with_device(&Device::new(DeviceType::Gpu,0));
    let hidden_layout=WorkspaceLayout::new(&[2,POSITIONS,WIDTH],WorkspaceDtype::Float32).unwrap();
    let weight_layout=WorkspaceLayout::new(&[VOCAB,WIDTH],WorkspaceDtype::Float32).unwrap();
    let mut cases=0;
    for dtype in [Dtype::Float32,Dtype::Float16,Dtype::Bfloat16] {
        let dense=floating(&weight_layout,1,dtype,false,&stream);
        for affine in [false,true] {
            let format=if affine {LinearFormat::Affine(AffineQuantization::new(GROUP,BITS).unwrap())}
                else {LinearFormat::Dense};
            let (weight,scales,biases)=if affine {
                let encoded=safemlx::ops::quantize_with_mode(&dense,GROUP,BITS,
                    safemlx::ops::QuantizationMode::Affine,&stream).unwrap();
                let scales=encoded.scales.as_dtype(dtype,&stream).unwrap();
                let biases=encoded.biases.expect("actual affine bias companion").as_dtype(dtype,&stream).unwrap();
                assert_eq!(encoded.weight.dtype(),Dtype::Uint32);
                assert_eq!(encoded.weight.shape(),[VOCAB,WIDTH*BITS/32]);
                assert_eq!(scales.shape(),[VOCAB,WIDTH/GROUP]);assert_eq!(biases.shape(),scales.shape());
                assert_eq!(scales.dtype(),dtype);assert_eq!(biases.dtype(),dtype);
                (encoded.weight,Some(scales),Some(biases))
            }else{(dense.clone(),None,None)};
            for tied in [false,true] {
                let mut projection=Projection::new(tied,format,&weight,scales.as_ref(),biases.as_ref(),&stream);
                for strided in [false,true] {
                    let hidden=MlxTensor::from_array(floating(&hidden_layout,0,dtype,strided,&stream));
                    assert_eq!(hidden.as_array().dtype(),dtype);
                    let reference=projection.forward(&hidden,&stream);
                    assert_eq!(reference.shape(),[2,POSITIONS,VOCAB]);
                    assert_eq!(reference.as_array().dtype(),dtype);
                    let reference_values=values(&reference,&stream);
                    assert!(reference_values.iter().all(|value|value.is_finite()));
                    assert!(reference_values.iter().any(|value|value.abs()>1e-5));
                    assert_cold_projection_scalar(format, &hidden, &weight, scales.as_ref(),
                        biases.as_ref(), &reference);
                    let reference_last=reference.narrow_axis(1,POSITIONS-1,POSITIONS,&stream).unwrap();

                    let mut calls=Vec::new();
                    let mut project=|input:&MlxTensor| {
                        calls.push(input.shape().to_vec());
                        assert_eq!(input.as_array().dtype(),dtype);
                        Ok(projection.forward(input,&stream))
                    };
                    assert!(execute_readout(&hidden,OutputDemand::StateOnly,1,&stream,&mut project)
                        .unwrap().is_none());
                    let sequence=execute_readout(&hidden,OutputDemand::Sequence,1,&stream,&mut project)
                        .unwrap().unwrap();
                    let last=execute_readout(&hidden,OutputDemand::LastPosition,1,&stream,&mut project)
                        .unwrap().unwrap();
                    drop(project);
                    assert_eq!(calls,[vec![2,POSITIONS,WIDTH],vec![2,1,WIDTH]],
                        "StateOnly skips projection; Last selects hidden rows before native projection");
                    assert_eq!(sequence.shape(),[2,POSITIONS,VOCAB]);
                    assert_eq!(last.shape(),[2,1,VOCAB]);
                    assert_eq!(sequence.as_array().dtype(),dtype);assert_eq!(last.as_array().dtype(),dtype);
                    compare(&values(&sequence,&stream),&reference_values,dtype);
                    compare(&values(&last,&stream),&values(&reference_last,&stream),dtype);
                    eprintln!("SELECTED_READOUT dtype={dtype:?} affine={affine} tied={tied} strided={strided}");
                    cases+=1;
                }
            }
        }
    }
    assert_eq!(cases,24);
}

// Compare the ordinary parameter-construction/forward source with the actual
// native result above, including nonzero strided inputs and reduced precision.
fn assert_cold_projection_scalar(
    encoding: LinearFormat, input: &MlxTensor, weight: &Array,
    scales: Option<&Array>, biases: Option<&Array>, actual: &MlxTensor,
) {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::with_source_count(&context,
        3 + usize::from(scales.is_some()) + usize::from(biases.is_some())).unwrap();
    let input = projection.project(input.as_array()).unwrap();
    let weight_spec = ParameterSpec::trainable("projection.weight").unwrap();
    let mut parameters = vec![WorkspaceParameterRepresentation::new(weight_spec.id.clone(),
        projection.project(weight).unwrap().layout().clone())];
    let format = if let (Some(scales), Some(biases)) = (scales, biases) {
        let scale = ParameterSpec::trainable("projection.scales").unwrap();
        let bias = ParameterSpec::trainable("projection.biases").unwrap();
        parameters.push(WorkspaceParameterRepresentation::new(scale.id.clone(),
            projection.project(scales).unwrap().layout().clone()));
        parameters.push(WorkspaceParameterRepresentation::new(bias.id.clone(),
            projection.project(biases).unwrap().layout().clone()));
        LinearFormatSpec::affine(encoding, scale, bias).unwrap()
    } else { LinearFormatSpec::unscaled(encoding).unwrap() };
    let expected = projection.project(actual.as_array()).unwrap();
    assert!(projection.is_complete());
    context.install_parameter_representations(parameters).unwrap();
    let mut linear = WorkspaceBackend::linear(LinearSpec {
        input: WIDTH, output: VOCAB, weight: weight_spec, bias: None, format,
    }, &context).unwrap();
    let output = linear.forward(&input, &context).unwrap();
    assert_eq!(output.shape(), actual.shape());
    assert_eq!(output.layout().representation().unwrap().dtype(),
        expected.layout().representation().unwrap().dtype());
    assert!(!output.layout().representation().unwrap().row_contiguous());
}
