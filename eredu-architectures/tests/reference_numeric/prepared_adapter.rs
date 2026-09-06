//! Native scalar mechanisms entering the same complete construction path as MLX.

use super::*;
use eredu_architectures::{prepared_execution::*, prepared_sources::PreparedModelSources};

pub(super) struct NumericPreparationProvider {
    pub(super) addressable: bool,
}

impl eredu_architectures::PreparationMechanismProvider for NumericPreparationProvider {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        eredu_core::PreparationMechanismCapabilities::new(true, true)
            .with_input_modalities(eredu_core::InputModalities {
                text: true,
                image: true,
                audio: true,
                video: true,
            })
            .with_safetensors_quantization(true, true)
            .with_gguf_quantized_loading(true)
            .with_exact_completion(true)
            .with_session(eredu_core::SessionCapabilities::new(true, true, true))
            .with_parallel_axis(eredu_core::ParallelAxis::Tensor, true)
            .with_parallel_axis(eredu_core::ParallelAxis::Pipeline, true)
            .with_parallel_axis(eredu_core::ParallelAxis::Expert, true)
            .with_residency(eredu_core::ResidencyRequest::FullyResident, true)
            .with_residency(
                eredu_core::ResidencyRequest::AddressableParameterBanks,
                self.addressable,
            )
    }

    fn supports_grouped_operation(&self, _: eredu_runtime::GroupedOperationRequirement) -> bool {
        true
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        eredu_runtime::synthesize_replicated_text_capabilities(
            requirements,
            request,
            &NumericMechanismSupport {
                persistent_session: true,
                addressable: self.addressable,
            },
        )
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        let modalities = [
            eredu_core::InputModality::Text,
            eredu_core::InputModality::Image,
            eredu_core::InputModality::Audio,
            eredu_core::InputModality::Video,
        ];
        eredu_runtime::MediaPrimitiveCapabilities::new(
            [],
            modalities,
            modalities,
            [],
            i32::MAX as u64,
        )
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        eredu_runtime::SpeculativeMechanismCapabilities::new([])
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        numeric_partition_capabilities()
    }
}

pub(super) fn plan(
    quantization: Option<eredu_core::QuantizationRequest>,
) -> eredu_core::ExecutionPlan {
    let transform = match quantization {
        None => eredu_core::WeightTransformationPlan::PreserveCheckpoint,
        Some(eredu_core::QuantizationRequest::Affine { group_size, bits }) => {
            eredu_core::WeightTransformationPlan::Affine {
                group_size: group_size.try_into().unwrap(),
                bits: bits.into(),
            }
        }
        Some(eredu_core::QuantizationRequest::MxFp4) => eredu_core::WeightTransformationPlan::MxFp4,
        Some(request) => panic!("unsupported synthetic quantization fixture {request:?}"),
    };
    eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("scalar-client", "cpu:0").unwrap(),
    )
    .with_weight_transformation(transform)
}

pub(super) fn prepare(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    plan: &eredu_core::ExecutionPlan,
    mechanisms: &NumericPreparationProvider,
) -> Result<PreparedModelSources, String> {
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        plan,
        eredu_runtime::ResidencyDiagnostics::new(false, false),
        None,
    )
    .map_err(|error| error.to_string())?;
    let selected = eredu_architectures::select_preparation(inspection, &request, mechanisms)
        .map_err(|error| error.to_string())?;
    let plan = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection.clone(),
        selected.admission(),
    )
    .map_err(|error| error.to_string())?;
    eredu_architectures::prepared_sources::prepare_model_sources(plan, selected)
        .map_err(|error| error.to_string())
}

struct RunAssembler;
impl PreparedExecutableAssembler<()> for RunAssembler {
    type Executable = NumericReplicatedRun;
    type Output = NumericReplicatedRun;
    type Error = String;

    fn floating_state_bytes(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<std::num::NonZeroU8, String> {
        // The scalar mechanism computes all floating activations in f32.
        Ok(std::num::NonZeroU8::new(4).unwrap())
    }

    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("ordinary scalar adapter has no communication mechanism".into())
    }

    fn finish(
        self,
        mut parts: PreparedExecutableParts<NumericReplicatedRun, ()>,
    ) -> Result<NumericReplicatedRun, String> {
        if parts.take_communication().is_some() || parts.take_processor().is_some() {
            return Err("ordinary scalar adapter received optional native resources".into());
        }
        assert_eq!(parts.floating_state_bytes().get(), 4);
        Ok(parts.into_executable())
    }
}

pub(super) fn replicated(
    sources: PreparedModelSources,
    context: &NumericContext,
    tokens: &NumericTensor,
) -> Result<NumericReplicatedRun, String> {
    let visitor = NumericReplicatedVisitor {
        context,
        tokens,
        construction_started: false,
    };
    let routes =
        PreparedExecutionRoutes::new().with_replicated(ReplicatedRoute::<NumericBackend, _>::new(
            context,
            context,
            SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(visitor),
        ));
    construct_prepared_execution(sources, None, routes, RunAssembler)
        .map_err(|error| error.to_string())
}

pub(super) fn routed(
    sources: PreparedModelSources,
    context: &NumericContext,
    tokens: &NumericTensor,
) -> Result<NumericReplicatedRun, String> {
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let routes = PreparedExecutionRoutes::new().with_routed(RoutedRoute::<
        NumericBackend,
        State,
        State,
        _,
        _,
        _,
    >::new(
        context,
        context,
        NumericRoutedVisitor { context, tokens },
        NumericRelu2RoutedVisitor { context, tokens },
        NumericRoutedVisitor { context, tokens },
    ));
    construct_prepared_execution(sources, None, routes, RunAssembler)
        .map_err(|error| error.to_string())
}

pub(super) fn composite(
    sources: PreparedModelSources,
    context: &NumericContext,
    input: &eredu_runtime::PreparedModelInput<NumericTensor>,
    observer: Option<std::rc::Rc<std::cell::RefCell<CompositeObservation>>>,
) -> Result<NumericReplicatedRun, String> {
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let routes =
        PreparedExecutionRoutes::new().with_composite(
            CompositeRoute::<NumericBackend, State, _>::new(
                context,
                context,
                NumericCompositeVisitor {
                    context,
                    input,
                    construction_started: false,
                    observer,
                },
            ),
        );
    construct_prepared_execution(sources, None, routes, RunAssembler)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
type ParameterBits = BTreeMap<String, (Vec<i32>, Vec<u32>)>;

#[cfg(test)]
pub(super) fn payload_fixture(head_scale: f32) -> (tempfile::TempDir, ParameterBits) {
    payload_fixture_config(&config("llama", false), head_scale)
}

#[cfg(test)]
pub(super) fn payload_fixture_config(
    config: &serde_json::Value,
    head_scale: f32,
) -> (tempfile::TempDir, ParameterBits) {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = required_safetensors_parameters(config)
        .into_iter()
        .map(|(name, dimensions)| {
            let spec = ParameterSpec::trainable(name.clone()).unwrap();
            let dimensions = dimensions
                .into_iter()
                .map(|x| i32::try_from(x).unwrap())
                .collect::<Vec<_>>();
            let mut value = parameter(&spec, dimensions, name.contains("norm"));
            if name == "lm_head.weight" {
                value = value.map(|value| value * head_scale);
            }
            (name, value)
        })
        .collect::<BTreeMap<_, _>>();
    let expected = tensors
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                (
                    value.shape.clone(),
                    value.data.iter().map(|value| value.to_bits()).collect(),
                ),
            )
        })
        .collect();
    let encoded = tensors
        .into_iter()
        .map(|(name, value)| {
            (
                name,
                (
                    value
                        .shape
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    value
                        .data
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let views = encoded
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
    (root, expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_bound(head_scale: f32, key_value_only: bool) -> NumericReplicatedRun {
        let (root, expected) = payload_fixture(head_scale);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let sources = prepare(
            &inspection,
            &plan(None),
            &NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let context = NumericContext {
            bind_checkpoint_values: true,
            ..NumericContext::default()
        };
        let tokens = NumericTensor::token_ids(&[1, 3, 2]);
        reset_reference_stage_evidence("SafeTensors");
        let run = if key_value_only {
            let visitor = NumericReplicatedVisitor {
                context: &context,
                tokens: &tokens,
                construction_started: false,
            };
            let routes = PreparedExecutionRoutes::new().with_replicated(KeyValueRoute::<
                NumericBackend,
                DeviceState<NumericBackend, NumericHybridLayerState>,
                _,
            >::new(
                &context, visitor
            ));
            construct_prepared_execution(sources, None, routes, RunAssembler).unwrap()
        } else {
            replicated(sources, &context, &tokens).unwrap()
        };
        assert_eq!(
            last_reference_stage_evidence().bound_parameters,
            expected,
            "every executed module parameter must equal its exact selected checkpoint payload"
        );
        run
    }

    #[test]
    fn prepared_execution_binds_exact_payload_values_and_head_changes_only_logits() {
        let ordinary = execute_bound(1.0, false);
        let key_value = execute_bound(1.0, true);
        let altered = execute_bound(2.0, false);
        for ((original, narrow), changed) in ordinary
            .outputs
            .iter()
            .zip(&key_value.outputs)
            .zip(&altered.outputs)
        {
            assert_tensor_exact(original, narrow, "full versus minimal key/value route");
            assert!(original.data.iter().any(|value| value.abs() > 1e-7));
            assert_tensor_close(
                changed,
                &original.map(|value| 2.0 * value),
                "payload output-head perturbation",
            );
        }
        assert_state_exact(
            &ordinary.state,
            &altered.state,
            ordinary.state.layout().len(),
            "head payload leaves recurrent state unchanged",
        );
    }
}
