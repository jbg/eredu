//! Fault injection lives under prepared_sources so invalid pairings never need a public API.

use std::{cell::RefCell, num::NonZeroU8, rc::Rc};

use eredu_core::{ArtifactInspection, ModelConfigurationResolver};
use eredu_runtime::{CommunicationManifest, NormalizedLoadRequest};

use super::*;
use crate::{
    preparation_selection::{
        select_preparation,
        tests::{
            composite_config, inspected_config, inspected_llama, parallel_request,
            prediction_config, routed_config, BoundedIndependentAdapter,
        },
        PreparationMechanismProvider,
    },
    prepared_execution::*,
};

type Events = Rc<RefCell<Vec<&'static str>>>;

struct AssemblyProbe {
    events: Events,
    reject_width: bool,
    wrong_dtype: bool,
}

impl AssemblyProbe {
    fn new(events: &Events) -> Self {
        Self {
            events: Rc::clone(events),
            reject_width: false,
            wrong_dtype: false,
        }
    }
}

impl PreparedExecutableAssembler<CommunicationManifest> for AssemblyProbe {
    type Executable = &'static str;
    type Output = (&'static str, NonZeroU8, bool);
    type Error = &'static str;

    fn floating_state_dtype(
        &mut self,
        source: &crate::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Self::Error> {
        self.events.borrow_mut().push("native-width");
        assert_eq!(source.dtype(), &eredu_core::checkpoint::TensorDtype::F32);
        if self.reject_width {
            Err("unsupported native floating width")
        } else {
            // I32 occupies the same four bytes, but is not the admitted F32 representation.
            Ok(if self.wrong_dtype {
                eredu_runtime::StateStorageDtype::I32
            } else {
                eredu_runtime::StateStorageDtype::F32
            })
        }
    }

    fn validate_communication(
        &mut self,
        manifest: &CommunicationManifest,
        communication: &CommunicationManifest,
    ) -> Result<(), Self::Error> {
        self.events.borrow_mut().push("communication-check");
        if communication == manifest {
            Ok(())
        } else {
            Err("native communication does not match selected manifest")
        }
    }

    fn finish(
        self,
        mut parts: PreparedExecutableParts<Self::Executable, CommunicationManifest>,
    ) -> Result<Self::Output, Self::Error> {
        self.events.borrow_mut().push("publication");
        let width = parts.floating_state_bytes();
        let partitioned = parts.take_communication().is_some();
        assert!(parts.take_processor().is_none());
        Ok((parts.into_executable(), width, partitioned))
    }
}

struct RouteProbe(Events, &'static str);

impl crate::prepared_execution::sealed::Sealed for RouteProbe {}

impl<S> PreparedExecutionRoute<S, CommunicationManifest, &'static str, &'static str>
    for RouteProbe
{
    fn construct(
        self,
        _: PreparedConstructionBranch<S, CommunicationManifest>,
    ) -> Result<&'static str, PreparedExecutionError<&'static str>> {
        self.0.borrow_mut().push(self.1);
        Ok(self.1)
    }
}

/// Construction-order probe with raw image facts; no media work is executed.
struct RawImageAdapter(BoundedIndependentAdapter);

impl PreparationMechanismProvider for RawImageAdapter {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        self.0.preparation_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        self.0.supports_grouped_operation(requirement)
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        self.0.replicated_text_capabilities(requirements, request)
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        use eredu_runtime::ProcessorPrimitive::*;
        let modalities = [
            eredu_core::InputModality::Text,
            eredu_core::InputModality::Image,
        ];
        eredu_runtime::MediaPrimitiveCapabilities::new(
            modalities,
            modalities,
            modalities,
            [
                RgbResizeBicubic,
                RgbNormalize,
                TensorF32,
                TensorI32,
                TensorU32,
            ],
            u64::MAX,
        )
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        self.0.speculative_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        self.0.communication_capabilities()
    }
}

fn prepare(
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    request: NormalizedLoadRequest,
) -> PreparedModelSources {
    prepare_with_mechanisms(inspection, request, &BoundedIndependentAdapter::default())
}

fn prepare_with_mechanisms(
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    request: NormalizedLoadRequest,
    mechanisms: &impl PreparationMechanismProvider,
) -> PreparedModelSources {
    let selected = select_preparation(&inspection, &request, mechanisms).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        request.preparation_policy().unwrap(),
        selected.session_capabilities(),
    )
    .unwrap();
    prepare_model_sources(plan, selected).unwrap()
}

#[test]
fn prediction_role_mismatch_stops_before_width_construction_or_publication() {
    let (_root, inspection) = inspected_config(prediction_config());
    let mut sources = prepare(inspection, NormalizedLoadRequest::default());
    assert!(sources.graph.extension.take().is_some());
    let events = Events::default();
    let routes =
        PreparedExecutionRoutes::new().with_routed(RouteProbe(Rc::clone(&events), "constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    assert!(matches!(
        result,
        Err(PreparedExecutionError::PredictionSourceMismatch)
    ));
    assert!(events.borrow().is_empty());
}

#[test]
fn unselected_prediction_source_stops_before_width_construction_or_publication() {
    let (_root, inspection) = inspected_llama();
    let mut sources = prepare(inspection, NormalizedLoadRequest::default());
    sources.graph.extension = Some(Arc::clone(sources.graph.target()));
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    assert!(matches!(
        result,
        Err(PreparedExecutionError::PredictionSourceMismatch)
    ));
    assert!(events.borrow().is_empty());
}

#[test]
fn missing_partition_communication_stops_before_native_facts_or_construction() {
    let (_root, inspection) = inspected_llama();
    let sources = prepare(inspection, parallel_request());
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_partitioned_dense(RouteProbe(Rc::clone(&events), "constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    assert!(matches!(
        result,
        Err(PreparedExecutionError::MissingCommunication)
    ));
    assert!(events.borrow().is_empty());
}

#[test]
fn ordinary_communication_is_rejected_in_release_semantics_too() {
    let (_root, inspection) = inspected_llama();
    let sources = prepare(inspection, NormalizedLoadRequest::default());
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "constructor"));
    let communication = CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap();
    let result = construct_prepared_execution(
        sources,
        Some(communication),
        routes,
        AssemblyProbe::new(&events),
    );
    assert!(matches!(
        result,
        Err(PreparedExecutionError::UnexpectedCommunication)
    ));
    assert!(events.borrow().is_empty());
}

#[test]
fn wrong_native_communication_stops_before_width_construction_or_publication() {
    let (_root, inspection) = inspected_llama();
    let sources = prepare(inspection, parallel_request());
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_partitioned_dense(RouteProbe(Rc::clone(&events), "constructor"));
    let communication = CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap();
    let result = construct_prepared_execution(
        sources,
        Some(communication),
        routes,
        AssemblyProbe::new(&events),
    );
    assert!(matches!(result, Err(PreparedExecutionError::Backend(_))));
    assert_eq!(*events.borrow(), ["communication-check"]);
}

#[test]
fn invalid_architecture_floating_source_stops_before_native_width_or_construction() {
    let (_root, inspection) = inspected_llama();
    let mut sources = prepare(inspection, NormalizedLoadRequest::default());
    let other = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":16, "hidden_size":8,
            "intermediate_size":12, "num_hidden_layers":2,
            "hybrid_override_pattern":"--", "num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"mamba_num_heads":2,
            "n_groups":1,"mamba_head_dim":4,"ssm_state_size":3,"conv_kernel":3,
            "n_routed_experts":4,"n_shared_experts":1,"moe_intermediate_size":6,
            "moe_shared_expert_intermediate_size":6,"num_experts_per_tok":2,
            "n_group":2,"topk_group":1,"num_nextn_predict_layers":0
        }))
        .unwrap();
    // Nemotron's declared embedding is absent from the retained Llama catalog.
    sources.graph.architecture = other.architecture_plan().clone();
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    let Err(PreparedExecutionError::Architecture(message)) = result else {
        panic!("expected architecture floating-source failure");
    };
    assert!(message.contains("floating-state dtype source"));
    assert!(events.borrow().is_empty());
}

#[test]
fn native_dtype_must_match_admission_even_when_the_byte_width_is_equal() {
    let (_root, inspection) = inspected_llama();
    let sources = prepare(inspection, NormalizedLoadRequest::default());
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "constructor"));
    let mut assembler = AssemblyProbe::new(&events);
    assembler.wrong_dtype = true;
    let result = construct_prepared_execution(sources, None, routes, assembler);
    assert!(
        matches!(result, Err(PreparedExecutionError::Architecture(message))
        if message.contains("differs from the admitted storage dtype"))
    );
    assert_eq!(*events.borrow(), ["native-width"]);
}

#[test]
fn native_floating_width_failure_stops_before_construction_and_publication() {
    let (_root, inspection) = inspected_llama();
    let sources = prepare(inspection, NormalizedLoadRequest::default());
    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "constructor"));
    let mut assembler = AssemblyProbe::new(&events);
    assembler.reject_width = true;
    let result = construct_prepared_execution(sources, None, routes, assembler);
    assert!(matches!(result, Err(PreparedExecutionError::Backend(_))));
    assert_eq!(*events.borrow(), ["native-width"]);
}

#[test]
fn removed_selected_raw_processor_stops_before_width_construction_or_publication() {
    let mut config = composite_config();
    config["image_token_id"] = 60.into();
    config["boi_token_id"] = 61.into();
    config["eoi_token_id"] = 62.into();
    config["vision_config"] = serde_json::json!({
        "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
        "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
        "rms_norm_eps":0.000001
    });
    let (_root, inspection) = inspected_config(config);
    let mut sources = prepare_with_mechanisms(
        inspection,
        NormalizedLoadRequest::default(),
        &RawImageAdapter(BoundedIndependentAdapter::default()),
    );
    assert!(sources
        .selected
        .execution()
        .processor()
        .unwrap()
        .raw_media());
    assert!(
        crate::processor_execution::PreparedProcessor::from_artifact(sources.graph.architecture())
            .is_some()
    );

    // This internal enrichment method changes only the retained processor.
    // The already admitted family, selected raw-media proof, exact checkpoint
    // sources, and original inspection remain otherwise untouched.
    sources.graph.architecture = sources
        .graph
        .architecture
        .clone()
        .with_safetensors_processors(b"{}", None, None, None)
        .unwrap();
    assert!(!sources.graph.architecture.has_processor());
    assert!(sources.inspection.architecture_plan().has_processor());
    assert!(sources
        .selected
        .execution()
        .processor()
        .unwrap()
        .raw_media());

    let events = Events::default();
    let routes = PreparedExecutionRoutes::new()
        .with_composite(RouteProbe(Rc::clone(&events), "constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    let Err(PreparedExecutionError::Architecture(message)) = result else {
        panic!("expected retained raw-media processor rejection");
    };
    assert_eq!(
        message,
        "selected raw-media execution has no retained architecture processor"
    );
    assert!(events.borrow().is_empty());
}

#[test]
fn absent_optional_route_stops_before_construction_and_publication() {
    let (_root, inspection) = inspected_config(routed_config());
    let sources = prepare(inspection, NormalizedLoadRequest::default());
    let events = Events::default();
    // An ordinary-only backend supplies no routed mechanism or optional backend bound.
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(RouteProbe(Rc::clone(&events), "wrong-constructor"));
    let result = construct_prepared_execution(sources, None, routes, AssemblyProbe::new(&events));
    assert!(matches!(
        result,
        Err(PreparedExecutionError::UnavailableExecution)
    ));
    assert_eq!(*events.borrow(), ["native-width"]);
}

#[test]
fn all_six_selected_classes_dispatch_once_then_publish_checked_parts() {
    for (class, config, partitioned) in [
        ("replicated", None, false),
        ("routed", Some(routed_config()), false),
        ("composite", Some(composite_config()), false),
        ("partitioned-dense", None, true),
        ("partitioned-routed", Some(routed_config()), true),
        ("partitioned-composite", Some(composite_config()), true),
    ] {
        let (_root, inspection) = match config {
            Some(config) => inspected_config(config),
            None => inspected_llama(),
        };
        let request = if partitioned {
            parallel_request()
        } else {
            NormalizedLoadRequest::default()
        };
        let sources = prepare(inspection, request);
        let communication = sources.selected().communication_manifest().cloned();
        let events = Events::default();
        let route = |name| RouteProbe(Rc::clone(&events), name);
        let routes = PreparedExecutionRoutes::new()
            .with_replicated(route("replicated"))
            .with_routed(route("routed"))
            .with_composite(route("composite"))
            .with_partitioned_dense(route("partitioned-dense"))
            .with_partitioned_routed(route("partitioned-routed"))
            .with_partitioned_composite(route("partitioned-composite"));
        let output = construct_prepared_execution(
            sources,
            communication,
            routes,
            AssemblyProbe::new(&events),
        )
        .unwrap();
        assert_eq!(output, (class, NonZeroU8::new(4).unwrap(), partitioned));
        let mut expected = Vec::new();
        if partitioned {
            expected.push("communication-check");
        }
        expected.extend(["native-width", class, "publication"]);
        assert_eq!(*events.borrow(), expected);
    }
}

// This generic body must type-check without BlockwiseAttentionBackend,
// DistributedNeuralBackend, or any other optional mechanism. A concrete broad
// mock backend alone would not prove the absence of those hidden bounds.
#[allow(dead_code)]
fn minimal_key_value_route_needs_only_base_neural_traits<B, S, V, A>(
    sources: PreparedModelSources,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    assembler: A,
) -> Result<A::Output, PreparedExecutionError<A::Error>>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
    V: crate::replicated_text::ReplicatedTextArchitectureVisitor<
        B,
        S,
        Output = A::Executable,
        Error = A::Error,
    >,
    A: PreparedExecutableAssembler<()>,
{
    construct_prepared_execution(
        sources,
        None,
        PreparedExecutionRoutes::new()
            .with_replicated(KeyValueRoute::<B, S, V>::new(context, visitor)),
        assembler,
    )
}

// A second compile-time boundary: gated-only adapters need not implement the
// ReLU-squared production-visitor contract to enter the total driver.
#[allow(dead_code)]
fn gated_partition_route_does_not_require_relu2_visitor<B, S, PS, GF, TF, G, T, A>(
    sources: PreparedModelSources,
    communication: (),
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    gated: GF,
    pooling: TF,
    assembler: A,
) -> Result<A::Output, PreparedExecutionError<A::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: eredu_runtime::LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    GF: FnOnce(PreparedPartitionResources<()>) -> G,
    TF: FnOnce(PreparedPartitionResources<()>) -> T,
    G: crate::partitioned_execution::RoutedPartitionedProductionVisitor<
        B,
        S,
        Output = A::Executable,
        Error = A::Error,
    >,
    T: crate::partitioned_execution::RoutedPartitionedProductionVisitor<
        B,
        PS,
        Output = A::Executable,
        Error = A::Error,
    >,
    A: PreparedExecutableAssembler<()>,
{
    construct_prepared_execution(
        sources,
        Some(communication),
        PreparedExecutionRoutes::new().with_partitioned_routed(PartitionedRoutedRoute::<
            B,
            S,
            PS,
            _,
            _,
        >::new(
            context, context, gated, pooling
        )),
        assembler,
    )
}
