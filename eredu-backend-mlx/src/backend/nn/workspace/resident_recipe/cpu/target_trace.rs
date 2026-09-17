//! The public external target geometry through the same resident family driver.
//! This diagnoses the remaining CPU producers; it grants no public admission.
use super::*;
use eredu_architectures::gemma4::{self, DecoderInputPart, FamilyConfig, ModelInput};
use eredu_nn::{CpuMatmulImplementation, ParameterMetadata, ParameterVisitorMut, Parameterized};
use eredu_nn::workspace::WorkspaceBackend;
use eredu_runtime::{ArchitectureStateFactory, DeviceState, LayeredArchitecture, ResidentRuntime};
use eredu_runtime::working_memory::{WorkspaceResidentLayerState, WorkspaceResidentStateFactory};
use std::num::NonZeroU32;

type State=DeviceState<WorkspaceBackend,WorkspaceResidentLayerState>;
type Architecture=gemma4::LayeredModel<WorkspaceBackend>;
struct Sources<'a>(&'a WorkspaceContext,WorkspaceFloatingType);
impl<'a> ParameterVisitorMut<'a,WorkspaceTensor> for Sources<'_> {
    fn visit_mut(&mut self,_:ParameterMetadata,value:&'a mut WorkspaceTensor) {
        let layout=value.layout().clone().with_representation(Some(
            WorkspaceRepresentation::new(self.1,true)));
        *value=WorkspaceTensor::existing(layout,self.0).unwrap();
    }
}
#[test]
fn cpu_public_gemma_target_trace_names_remaining_prefill_and_cached_producers() {
    target_trace(WorkspaceFloatingType::Float32);
}
#[test]
fn cpu_public_gemma_half_target_trace_names_remaining_prefill_and_cached_producers() {
    for dtype in [WorkspaceFloatingType::Bfloat16,WorkspaceFloatingType::Float16] {target_trace(dtype);}
}
// This cold fixture describes the selected checkpoint parameter representation;
// it does not claim a released checkpoint or native execution was validated.
fn target_trace(dtype:WorkspaceFloatingType) {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
        assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
    };
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    let context=WorkspaceContext::new(cpu);
    let args=FamilyConfig::from_hf_json(br#"{"model_type":"gemma4","text_config":{
        "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":2,
        "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
        "rms_norm_eps":0.00001,"vocab_size":64,"max_position_embeddings":128,
        "tie_word_embeddings":false,"attention_k_eq_v":false,"num_kv_shared_layers":1,
        "layer_types":["full_attention","full_attention"]}}"#).unwrap();
    let layout=gemma4::state_layout(&args.text).unwrap();
    let architecture=Architecture::new(args,&context).unwrap();
    let mut runtime=ResidentRuntime::<Architecture,WorkspaceBackend,State>::new(architecture,&context).unwrap();
    <Architecture as LayeredArchitecture<WorkspaceBackend,State>>::static_modules_mut(runtime.architecture_mut())
        .visit_parameters_mut(&mut Sources(&context,dtype));
    for group in runtime.units_mut() {for unit in group {unit.visit_parameters_mut(&mut Sources(&context,dtype));}}
    let mut state=WorkspaceResidentStateFactory::new(NonZeroU32::new(1).unwrap(),NonZeroU32::new(128).unwrap(),&context)
        .unwrap().realize(&layout).unwrap();
    let mut cached=0usize;
    for (step,positions) in [3,2,1,1].into_iter().enumerate() {
        let tokens=WorkspaceTensor::existing(context.layout(&[1,positions],WorkspaceDtype::Int32).unwrap(),&context).unwrap();
        context.begin_span();
        let parts=[DecoderInputPart::Text(&tokens)];
        let output=runtime.forward(ModelInput{parts:&parts,vision:None,audio:None,per_layer_tokens:None,mask:None},
            &mut state,&context).unwrap();
        let report=context.report(&[output]).unwrap();
        assert!(!report.operations.is_empty());
        let mut missing=Vec::new();
        for (index,operation) in report.operations.iter().enumerate() {
            let known=cpu.plan(operation.as_view()).unwrap().is_some();
            if dtype==WorkspaceFloatingType::Float32||!known {
                println!("CPU_TARGET_OPERATION step={step} cached={cached} positions={positions} dtype={dtype:?} index={index} known={known} kind={:?} inputs={:?}",operation.kind,operation.inputs);
            }
            if !known {missing.push(index);}
        }
        println!("CPU_TARGET_MISSING step={step} positions={positions} dtype={dtype:?} operations={} indices={missing:?}",report.operations.len());
        cached+=positions as usize;
    }
    assert_eq!(cached,7);
}
