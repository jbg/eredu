//! The existing one-layer Gemma DraftStep, used to close CPU sources in order.
use super::*;
use crate::backend::nn::workspace::{MlxCpuMatmulMechanism,MlxCpuWorkspaceMechanisms};
use eredu_architectures::gemma4::{AssistantState,assistant::invocation::{DraftStep,DraftStepArguments}};
use eredu_nn::{ParameterVisitorMut,ParameterMetadata,Tensor,CpuMatmulImplementation};
use eredu_nn::workspace::{WorkspaceDtype,WorkspaceMechanisms};
struct Bind<'a>(&'a WorkspaceContext);
impl<'a> ParameterVisitorMut<'a,WorkspaceTensor> for Bind<'_> {
    fn visit_mut(&mut self,_:eredu_nn::ParameterMetadataView<'_>,value:&'a mut WorkspaceTensor) {
        *value=represented(value.shape(),self.0);
    }
}
fn represented(shape:&[i32],context:&WorkspaceContext)->WorkspaceTensor {
    WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),context).unwrap()
}
#[test]
fn cpu_gemma_assistant_actual_step_lists_unqualified_sources() {
    // Same existing native materialization fixture: un-ordered one-layer Gemma,
    // 32-wide target/assistant, 4 query heads, 2 shared KV heads and width 8.
    let config=eredu_architectures::gemma4::AssistantConfig::from_json(
        super::super::super::external_materialization_tests::ASSISTANT_CONFIG.as_bytes()).unwrap();
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
        assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
    };
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    // This is an inventory trace, not CPU admission. Actual CPU source facts
    // propagate represented outputs through qualified producers; later missing
    // sources remain visible without inventing a row layout for them.
    let context=WorkspaceContext::new(cpu);
    let mut module=DraftStep::workspace_module(&config,&context).unwrap();
    module.visit_parameters_mut(&mut Bind(&context));
    let embedding=represented(&[1,1,32],&context);
    let mut state=AssistantState {evidence:None,
        shared_kv:std::collections::HashMap::from([(eredu_core::AttentionPolicy::Full,
            (represented(&[1,2,3,8],&context),represented(&[1,2,3,8],&context)))]).into(),
        kv_offset:3,hidden:represented(&[1,1,32],&context)};
    let arguments=DraftStepArguments{embedding:&embedding,state:&mut state};
    let mut projected=DraftStep::project(&arguments,|value|Ok(value.clone()),&context).unwrap();
    let mut opening=Vec::new();DraftStep::visit_projected(&projected,&mut |v|opening.push(v.clone()));
    context.begin_state_span(&opening).unwrap();
    let output=DraftStep::trace(&mut module,&mut projected,&context).unwrap();
    assert_eq!(output.shape(),[1,1,32]);
    let report=context.report(&[output]).unwrap();
    assert!(!report.operations.is_empty());
    let mut missing=0;
    for (index,operation) in report.operations.iter().enumerate() {
        let supported=cpu.operation_bound(operation).unwrap().is_some();
        if !supported {missing+=1;}
        println!("CPU_GEMMA_OPERATION {index} supported={supported} kind={:?} inputs={:?} outputs={:?}",
            operation.kind,operation.inputs,operation.outputs);
    }
    println!("CPU_GEMMA_SOURCE_INVENTORY operations={} missing={missing}",report.operations.len());
    assert_eq!(state.kv_offset,3,"cold tracing does not consume the real assistant state");
    assert_eq!(missing,0,"every actual Gemma DraftStep operation needs its own CPU source");
    assert!(report.tensor_buffers.total_bytes.is_some());
    // This proves the actual family trace; native source/role execution remains
    // separately required before any public cross-device admission opens.
}
