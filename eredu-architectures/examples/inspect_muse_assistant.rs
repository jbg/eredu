//! Inspects exact released assistant sources without allocating backend tensors or hashing payloads.
use eredu_architectures::{
    external_assistant::{
        ExternalAssistantArchitecture, ExternalAssistantPreparationVisitor,
        ExternalAssistantTargetProfile, PreparedExternalAssistantSource,
    },
    external_draft::ExternalDraftPreparation,
    muse::DecoderConfig,
};
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_runtime::WeightLoweringKind;

struct Inspect;
impl ExternalAssistantPreparationVisitor for Inspect {
    type Output<A: ExternalAssistantArchitecture> = ();
    type Error = Box<dyn std::error::Error>;
    fn visit<A: ExternalAssistantArchitecture>(
        self,
        prepared: PreparedExternalAssistantSource<A>,
    ) -> Result<(), Self::Error> {
        let model_type = A::configuration_model_type(prepared.config()).to_owned();
        let tokenizer = A::tokenizer_model_type(prepared.config());
        let requirement = prepared.parameter_residency_requirement()?;
        let memory_topology = A::memory_topology(prepared.module_config())?;
        let units = prepared
            .parameter_units()
            .iter()
            .map(|unit| {
                let owner = match unit.owner() {
                    eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                        serde_json::json!({"group":group, "unit":unit})
                    }
                    eredu_runtime::ReplicatedTextParameterOwner::StaticRole(role) => {
                        serde_json::json!({"static_role":role})
                    }
                    eredu_runtime::ReplicatedTextParameterOwner::StaticUnitConsumers {
                        role,
                        consumers,
                    } => serde_json::json!({"static_role":role, "consumers":consumers}),
                    owner => serde_json::json!({"unrecognized_owner":format!("{owner:?}")}),
                };
                serde_json::json!({
                    "owner":owner,
                    "task_count":unit.task_indices().len(),
                    "outputs":unit.outputs(),
                    "parameter_bytes":unit.parameter_bytes(),
                })
            })
            .collect::<Vec<_>>();
        let (source, _, _, _, _, _, tasks) = prepared.into_parts();
        let tensors = tasks.iter().map(|task| {
            let metadata = source.source_metadata(task.name())?;
            Ok(serde_json::json!({"name":task.name(), "shape":metadata.logical_shape,
                "dtype":format!("{:?}", metadata.stored_dtype), "role":format!("{:?}", task.role())}))
        }).collect::<Result<Vec<_>, eredu_checkpoint::store::StoreError>>()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model_type": model_type, "tokenizer_model_type": tokenizer,
                "tensor_count":tensors.len(), "tensors":tensors,
                "parameter_units":units,
                "conditioned_memory_topology":memory_topology,
                "device_parameter_requirement": {
                    "static_bytes":requirement.static_bytes,
                    "window_bytes":requirement.window_bytes,
                    "required_bytes":requirement.required_bytes,
                    "depth":requirement.depth,
                    "includes_workspace_or_state":false,
                    "includes_optional_conversions":false,
                },
                "payload_hash_resolved":false, "backend_execution_verified":false,
            }))?
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: inspect_muse_assistant ASSISTANT_DIRECTORY TARGET_CONFIG_JSON".into());
    }
    let target = DecoderConfig::from_hf_value(&serde_json::from_slice(&std::fs::read(&args[1])?)?)?;
    let ExternalDraftPreparation::Assistant(preparation) =
        eredu_architectures::prepare_external_draft(&args[0])?
    else {
        return Err("artifact is an ordinary target rather than a trained assistant".into());
    };
    let selected = preparation.select_materialization(None, 1, |descriptor, transforms| {
        (!transforms
            && descriptor.executable() == LinearFormat::Dense
            && matches!(
                descriptor.source(),
                SourceTensorEncoding::Safetensors(
                    StoredDtype::BF16 | StoredDtype::F16 | StoredDtype::F32 | StoredDtype::I64
                )
            ))
        .then_some(WeightLoweringKind::Direct)
    })?;
    let compatible =
        selected.prove_target_compatibility(&ExternalAssistantTargetProfile::Muse(target))?;
    eprintln!("capture: {:?}", compatible.capture());
    compatible.prepare_source(1)?.visit(Inspect)?;
    Ok(())
}
