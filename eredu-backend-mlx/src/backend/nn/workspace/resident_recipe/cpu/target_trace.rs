//! The public external target geometry through the same resident family driver.
//! This diagnoses the remaining CPU producers; it grants no public admission.
use super::*;
use eredu_architectures::composite_execution::CompositeArchitecture;
use eredu_architectures::gemma4::{self, DecoderInputPart, ModelInput};
use eredu_architectures::prepared_execution::*;
use eredu_architectures::replicated_text::{
    CompositeTextArchitectureVisitor, PreparedCompositeTextArchitecture,
    PreparedRoutedCompositeTextArchitecture,
};
use eredu_core::ModelLoadingBackend as _;
use eredu_nn::workspace::WorkspaceBackend;
use eredu_nn::{CpuMatmulImplementation, ParameterMetadata, ParameterVisitorMut, Parameterized};
use eredu_runtime::working_memory::{WorkspaceResidentLayerState, WorkspaceResidentStateFactory};
use eredu_runtime::{ArchitectureStateFactory, DeviceState, LayeredArchitecture, ResidentRuntime};
use std::num::NonZeroU32;

type State = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
type Architecture = gemma4::LayeredModel<WorkspaceBackend>;
struct Sources<'a>(&'a WorkspaceContext, WorkspaceFloatingType);
impl<'a> ParameterVisitorMut<'a, WorkspaceTensor> for Sources<'_> {
    fn visit_mut(
        &mut self,
        _: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut WorkspaceTensor,
    ) {
        let layout = value
            .layout()
            .clone()
            .with_representation(Some(WorkspaceRepresentation::new(self.1, true)));
        *value = WorkspaceTensor::existing(layout, self.0).unwrap();
    }
}
#[test]
fn cpu_public_gemma_target_trace_names_remaining_prefill_and_cached_producers() {
    if !crate::tests::support::native_process::enter("qualified-recipe") {
        return;
    }
    target_trace(WorkspaceFloatingType::Float32);
}
#[test]
fn cpu_public_gemma_half_target_trace_names_remaining_prefill_and_cached_producers() {
    if !crate::tests::support::native_process::enter("qualified-recipe") {
        return;
    }
    for dtype in [
        WorkspaceFloatingType::Bfloat16,
        WorkspaceFloatingType::Float16,
    ] {
        target_trace(dtype);
    }
}
// This cold fixture describes the selected checkpoint parameter representation;
// it does not claim a released checkpoint or native execution was validated.
fn target_trace(dtype: WorkspaceFloatingType) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let Some(selected) = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles)
    else {
        assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());
        return;
    };
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    // This standalone census owns ordinary cold declarations. A funded
    // architecture quote consumes an existing prepared source through the shared
    // admission driver; this diagnostic does not authenticate such a source.
    let context = WorkspaceContext::new(cpu);
    let configuration = serde_json::json!({"model_type":"gemma4","text_config":{
        "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":2,
        "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
        "rms_norm_eps":0.00001,"vocab_size":64,"max_position_embeddings":128,
        "tie_word_embeddings":false,"attention_k_eq_v":false,"num_kv_shared_layers":1,
        "layer_types":["full_attention","full_attention"]}});
    let artifact =
        crate::composition::mlx::replicated_text::tests::tiny_heterogeneous_artifact(configuration);
    let streams =
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &pool,
            safemlx::DeviceType::Cpu,
        )
        .unwrap()
        .unwrap();
    let identity = crate::backend::MlxDeviceIdentity::from_realized_device(
        &safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
        None,
    )
    .unwrap();
    let backend = crate::backend::MlxBackend::for_prepared_execution_plan(streams, identity);
    let inspection =
        eredu_core::inspect_artifact(artifact.path(), backend.configuration_resolver()).unwrap();
    let source = eredu_core::prepare_inspected_model_config(
        &backend,
        inspection,
        crate::MlxLoadRequest::default(),
    )
    .unwrap();
    let layout = source
        .prepared_sources()
        .selected()
        .text_realization()
        .state()
        .layout()
        .clone();
    let architecture = construct_prepared_execution(
        source.prepared_sources().clone(),
        None::<()>,
        PreparedExecutionRoutes::new().with_composite(
            CompositeRoute::<WorkspaceBackend, State, _>::new(
                &context,
                &context,
                SelectedArchitecture,
            ),
        ),
        ArchitectureAssembly,
    )
    .unwrap()
    .downcast::<Architecture>()
    .ok()
    .expect("selected Gemma architecture");
    let architecture = *architecture;
    let mut runtime = ResidentRuntime::<Architecture, WorkspaceBackend, State>::new_workspace(
        architecture,
        &context,
    )
    .unwrap();
    <Architecture as LayeredArchitecture<WorkspaceBackend, State>>::static_modules_mut(
        runtime.architecture_mut(),
    )
    .visit_parameters_mut(&mut Sources(&context, dtype));
    for group in runtime.units_mut() {
        for unit in group {
            unit.visit_parameters_mut(&mut Sources(&context, dtype));
        }
    }
    let mut state = WorkspaceResidentStateFactory::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(128).unwrap(),
        &context,
    )
    .unwrap()
    .realize(&layout)
    .unwrap();
    let mut cached = 0usize;
    for (step, positions) in [3, 2, 1, 1].into_iter().enumerate() {
        let tokens = WorkspaceTensor::existing(
            context
                .layout(&[1, positions], WorkspaceDtype::Int32)
                .unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let parts = [DecoderInputPart::Text(&tokens)];
        let output = runtime
            .forward(
                ModelInput {
                    parts: &parts,
                    vision: None,
                    audio: None,
                    per_layer_tokens: None,
                    mask: None,
                },
                &mut state,
                &context,
            )
            .unwrap();
        let report = context.finish_report(&[output]).unwrap();
        assert!(!report.operations.is_empty());
        let mut missing = Vec::new();
        for (index, operation) in report.operations.iter().enumerate() {
            let known = cpu.plan(operation.as_view()).unwrap().is_some();
            if dtype == WorkspaceFloatingType::Float32 || !known {
                println!(
                    "CPU_TARGET_OPERATION step={step} cached={cached} positions={positions} dtype={dtype:?} index={index} known={known} kind={:?} inputs={:?}",
                    operation.kind, operation.inputs
                );
            }
            if !known {
                missing.push(index);
            }
        }
        println!(
            "CPU_TARGET_MISSING step={step} positions={positions} dtype={dtype:?} operations={} indices={missing:?}",
            report.operations.len()
        );
        cached += positions as usize;
    }
    assert_eq!(cached, 7);
}

// The ordinary total construction driver authenticates the graph source before
// this fixture extracts its selected typed architecture for a cold operator census.
struct SelectedArchitecture;
impl CompositeTextArchitectureVisitor<WorkspaceBackend, State> for SelectedArchitecture {
    type Output = Box<dyn std::any::Any>;
    type Error = eredu_nn::Error;
    fn construction_started(&mut self) {}
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, State, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        Ok(Box::new(prepared.into_parts().0.into_inner()))
    }
    fn visit_routed<A>(
        self,
        _: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, State, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        panic!("fixture selects a dense composite source")
    }
}
struct ArchitectureAssembly;
impl PreparedExecutableAssembler<()> for ArchitectureAssembly {
    type Executable = Box<dyn std::any::Any>;
    type Output = Self::Executable;
    type Error = eredu_nn::Error;
    fn floating_state_dtype(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Self::Error> {
        Ok(eredu_runtime::StateStorageDtype::F32)
    }
    fn validate_communication(
        &mut self,
        _: &eredu_runtime::CommunicationManifest,
        _: &(),
    ) -> Result<(), Self::Error> {
        panic!("fixture selects local construction")
    }
    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, ()>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(parts.into_executable())
    }
}
