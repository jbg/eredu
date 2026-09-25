//! Architecture inspection and facade forecast parity with cold mock facts.
use eredu::api::*;
use eredu_architectures::inspect_model_metadata;
use eredu_architectures::PreparationMechanismProvider;
use eredu_core::{
    MediaFeatureAvailability, ModelConfigurationResolver, PreparationMechanismCapabilities,
    SessionCapabilities,
};
use eredu_runtime::*;
use std::collections::BTreeMap;

struct Mechanisms;
impl ReplicatedTextMechanismSupport for Mechanisms {
    fn facts(&self, _: &CacheResidencyPolicy) -> BackendMechanismFacts {
        BackendMechanismFacts::new(
            eredu_nn::NeuralOperatorCapabilities::ALL,
            [
                WeightResidencyMechanism::Resident,
                WeightResidencyMechanism::Windowed,
                WeightResidencyMechanism::DiskStreamed,
            ],
            StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_prompt_cache(true)
                .with_observation_retention(true),
        )
        .with_exact_completion(true)
        .with_session(SessionCapabilities::new(true, true, true))
        .with_chunked_prefill(true)
    }
    fn supports_direct(&self, _: &WeightLoweringDescriptor) -> bool {
        true
    }
    fn supports_transform(&self, _: &WeightLoweringDescriptor) -> bool {
        true
    }
    fn floating_state_dtype(
        &self,
        _: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<StateStorageDtype> {
        Some(StateStorageDtype::F32)
    }
    fn supports_state_component(
        &self,
        _: &eredu_core::cache::StateComponentPolicy,
        _: StateStorageDtype,
        _: StateComponentPlacement,
    ) -> bool {
        true
    }
}
impl PreparationMechanismProvider for Mechanisms {
    fn preparation_capabilities(&self) -> PreparationMechanismCapabilities {
        let mut caps = PreparationMechanismCapabilities::new(true, true)
            .with_safetensors_quantization(true, true)
            .with_gguf_quantized_loading(true)
            .with_input_modalities(eredu_core::InputModalities {
                text: true,
                image: false,
                audio: false,
                video: false,
            })
            .with_exact_completion(true)
            .with_session(SessionCapabilities::new(true, true, true));
        for residency in [
            eredu_core::ResidencyRequest::FullyResident,
            eredu_core::ResidencyRequest::LayerwiseHost,
            eredu_core::ResidencyRequest::DenseDiskStream,
        ] {
            caps = caps.with_residency(residency, true);
        }
        caps
    }
    fn supports_grouped_operation(&self, _: GroupedOperationRequirement) -> bool {
        true
    }
    fn replicated_text_capabilities(
        &self,
        r: &ReplicatedTextRequirements,
        q: &ReplicatedTextSelectionRequest,
    ) -> BackendMechanismCapabilities {
        synthesize_replicated_text_capabilities(r, q, self)
    }
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
        MediaPrimitiveCapabilities::new([], [], [], [], u64::MAX)
    }
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
        SpeculativeMechanismCapabilities::new([])
    }
    fn communication_capabilities(&self) -> CommunicationCapabilities {
        CommunicationCapabilities::new([]).unwrap()
    }
    fn recipe_materialization_workspace(
        &self,
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<u64, String> {
        recipe
            .peak_materialization_bytes(source)
            .map_err(|e| e.to_string())
    }
}
const MEDIA: MediaFeatureAvailability = MediaFeatureAvailability {
    image: false,
    audio: false,
};
fn provenance() -> MetadataProvenance {
    MetadataProvenance {
        source: "not-a-local-checkpoint".into(),
        revision: "immutable-fixture-digest".into(),
    }
}
fn config() -> serde_json::Value {
    serde_json::json!({"model_type":"llama", "hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"rms_norm_eps":0.00001,"vocab_size":32,"max_position_embeddings":4096,"rope_theta":10000.0,"tie_word_embeddings":false})
}
fn safetensors_fixture(split: bool) -> (tempfile::TempDir, ArtifactMetadata) {
    safetensors_fixture_with_config(split, config())
}
fn safetensors_fixture_with_config(
    split: bool,
    config: serde_json::Value,
) -> (tempfile::TempDir, ArtifactMetadata) {
    let root = tempfile::tempdir().unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let plan = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let constraints = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter_map(|g| g.variants.first())
                .flat_map(|v| &v.tensors),
        )
        .collect::<Vec<_>>();
    let mut headers = Vec::new();
    let mut weight_map = BTreeMap::new();
    for shard in 0..if split { 2 } else { 1 } {
        let member = if split {
            format!("weights-{shard}.safetensors")
        } else {
            "model.safetensors".into()
        };
        let mut offset = 0;
        let mut header = serde_json::Map::new();
        for (i, tensor) in constraints.iter().enumerate() {
            if split && i % 2 != shard {
                continue;
            }
            let dtype = if matches!(
                tensor.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U32
                )
            ) {
                "U32"
            } else {
                "F32"
            };
            let size = tensor.shape.iter().product::<usize>() * 4;
            header.insert(tensor.key.clone(), serde_json::json!({"dtype":dtype, "shape":tensor.shape,"data_offsets":[offset,offset+size]}));
            weight_map.insert(tensor.key.clone(), member.clone());
            offset += size;
        }
        let json = serde_json::to_vec(&header).unwrap();
        let bytes = [
            (json.len() as u64).to_le_bytes().as_slice(),
            json.as_slice(),
        ]
        .concat();
        let mut full = bytes.clone();
        full.resize(bytes.len() + offset, 0);
        std::fs::write(root.path().join(&member), &full).unwrap();
        headers.push(SafetensorsHeader {
            member,
            bytes,
            file_len: full.len() as u64,
        });
    }
    let index = serde_json::to_vec(&serde_json::json!({"weight_map":weight_map})).unwrap();
    let config = serde_json::to_vec(&config).unwrap();
    if split {
        std::fs::write(root.path().join("model.safetensors.index.json"), &index).unwrap();
    }
    std::fs::write(root.path().join("config.json"), &config).unwrap();
    (
        root,
        ArtifactMetadata {
            provenance: provenance(),
            checkpoint: CheckpointMetadata::SafeTensors {
                config,
                index: split.then_some(index),
                headers,
            },
            sidecars: BTreeMap::new(),
        },
    )
}
fn assert_parity(local: ModelInspectionOutcome, remote: ModelInspectionOutcome) {
    assert!(local.report().is_loadable(), "{:?}", local.report().issues);
    assert!(
        remote.report().is_compatible(),
        "{:?}",
        remote.report().issues
    );
    assert!(!remote.report().is_loadable());
    assert_eq!(
        remote.report().metadata_provenance.as_ref(),
        Some(&provenance())
    );
    assert_eq!(
        local.report().resources.materialized_parameter_bytes,
        remote.report().resources.materialized_parameter_bytes
    );
    assert_eq!(
        local.report().preparation_admission,
        remote.report().preparation_admission
    );
    for positions in [4, 128, 2000] {
        let mut options = GenerationMemoryOptions::new(
            eredu_core::InputTokenCount::text(positions),
            GenerationMemoryPlacement::Unified,
        );
        options.max_output_tokens = Some(32);
        options.budget.available_bytes = Some(1 << 30);
        let calibration = eredu_runtime::memory_forecast::ForecastCalibration::default();
        let local = forecast_inspected_generation(&local, &options, &calibration).unwrap();
        let remote = forecast_inspected_generation(&remote, &options, &calibration).unwrap();
        assert_eq!(local.request, remote.request);
        assert!(remote
            .request
            .domains
            .iter()
            .flat_map(|domain| &domain.executions)
            .filter_map(|execution| execution.execution_topology.as_ref())
            .all(|topology| topology.projection_storage.is_empty()));
        assert_eq!(local.estimate, remote.estimate);
        assert_eq!(local.execution, remote.execution);
    }
    let selected = remote.clone().into_selected().unwrap();
    let (inspection, preparation) = selected.into_parts();
    assert!(matches!(
        eredu_core::ModelPreparationPlan::from_retained_admission(
            inspection.clone(),
            preparation.admission()
        ),
        Err(eredu_core::artifact::ArtifactError::MetadataOnly)
    ));
    assert!(matches!(
        eredu_core::plan_model_preparation(inspection, Default::default(), Default::default()),
        Err(eredu_core::artifact::ArtifactError::MetadataOnly)
    ));
    assert!(eredu_architectures::prepare_inspected_model_sources(remote).is_err());
}
#[test]
fn supplied_safetensors_headers_match_local_forecasts_without_files() {
    for split in [false, true] {
        for layers in [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(Default::default()),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 2, 1).unwrap(),
            ),
        ] {
            let (root, metadata) = safetensors_fixture(split);
            let request = NormalizedLoadRequest::default()
                .with_weight_residency(WeightResidency::with_layers(layers));
            let local =
                eredu_architectures::inspect_model(root.path(), &request, &Mechanisms, MEDIA);
            root.close().unwrap();
            let remote = inspect_model_metadata(&metadata, &request, &Mechanisms, MEDIA);
            assert_parity(local, remote);
        }
    }
}
#[test]
fn malformed_metadata_returns_diagnostics_without_selection() {
    let (root, mut metadata) = safetensors_fixture(true);
    root.close().unwrap();
    let CheckpointMetadata::SafeTensors { headers, .. } = &mut metadata.checkpoint else {
        unreachable!()
    };
    headers.pop();
    let outcome = inspect_model_metadata(
        &metadata,
        &NormalizedLoadRequest::default(),
        &Mechanisms,
        MEDIA,
    );
    assert!(outcome.selected().is_none());
    assert!(!outcome.report().is_compatible());
    assert!(!outcome.report().issues.is_empty());
    assert_eq!(
        outcome.report().metadata_provenance.as_ref(),
        Some(&provenance())
    );
}

fn gguf_fixture(
    encoding: eredu_gguf::GgmlType,
    split: bool,
) -> (tempfile::TempDir, std::path::PathBuf, ArtifactMetadata) {
    use eredu_gguf::{GgmlType, MetadataValue as V, TensorInput, Writer};
    let root = tempfile::tempdir().unwrap();
    let args: eredu_architectures::llama::ModelArgs =
        eredu_architectures::llama::model_args_from_config_value(&config()).unwrap();
    let plan = eredu_architectures::llama::gguf_plan(&args).unwrap();
    let tensors = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter_map(|g| g.variants.first())
                .flat_map(|v| &v.tensors),
        )
        .map(|t| {
            let dtype = if t.shape.len() == 2 {
                encoding
            } else {
                GgmlType::F32
            };
            let (block, bytes) = dtype.block_and_bytes().unwrap();
            let bytes =
                vec![0; t.shape.iter().product::<usize>() / block as usize * bytes as usize];
            (
                t.key.clone(),
                t.shape.iter().rev().map(|n| *n as u64).collect::<Vec<_>>(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("llama".into())),
        ("llama.block_count".into(), V::Uint32(2)),
        ("llama.embedding_length".into(), V::Uint32(32)),
        ("llama.feed_forward_length".into(), V::Uint32(64)),
        ("llama.attention.head_count".into(), V::Uint32(2)),
        ("llama.attention.head_count_kv".into(), V::Uint32(1)),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            V::Float32(1e-5),
        ),
        ("llama.vocab_size".into(), V::Uint32(32)),
        ("llama.context_length".into(), V::Uint32(4096)),
        ("llama.rope.freq_base".into(), V::Float32(10000.0)),
    ]);
    let mut headers = Vec::new();
    let count = if split { 2 } else { 1 };
    for shard in 0..count {
        let member = if split {
            format!("model-{:05}-of-00002.gguf", shard + 1)
        } else {
            "model.gguf".into()
        };
        let inputs = tensors
            .iter()
            .enumerate()
            .filter(|(i, _)| i % count == shard)
            .map(|(_, (name, dimensions, dtype, data))| TensorInput {
                name,
                dimensions,
                ggml_type: *dtype,
                data,
            })
            .collect::<Vec<_>>();
        let mut metadata = metadata.clone();
        if split {
            metadata.extend([
                ("split.no".into(), V::Uint16(shard as u16)),
                ("split.count".into(), V::Uint16(count as u16)),
                (
                    "split.tensors.count".into(),
                    V::Uint64(tensors.len() as u64),
                ),
            ]);
        }
        let mut bytes = std::io::Cursor::new(Vec::new());
        Writer::default()
            .write(&mut bytes, &metadata, &inputs)
            .unwrap();
        let mut bytes = bytes.into_inner();
        std::fs::write(root.path().join(&member), &bytes).unwrap();
        let reader = eredu_gguf::Reader::new(std::io::Cursor::new(&bytes)).unwrap();
        let offset = reader
            .tensors()
            .iter()
            .map(|t| t.data_offset)
            .min()
            .unwrap() as usize;
        let file_len = bytes.len() as u64;
        bytes.truncate(offset);
        headers.push(GgufHeader {
            member,
            bytes,
            file_len,
        });
    }
    let first = root.path().join(&headers[0].member);
    (
        root,
        first,
        ArtifactMetadata {
            provenance: provenance(),
            checkpoint: CheckpointMetadata::Gguf {
                headers,
                companions: Vec::new(),
            },
            sidecars: BTreeMap::new(),
        },
    )
}

#[test]
fn supplied_gguf_headers_match_local_forecasts_without_files() {
    use eredu_gguf::GgmlType::*;
    for dtype in [F32, F16, Bf16, Q4_0, Q8_0] {
        for split in [false, true] {
            let (root, first, metadata) = gguf_fixture(dtype, split);
            let request = NormalizedLoadRequest::default();
            let local = eredu_architectures::inspect_model(&first, &request, &Mechanisms, MEDIA);
            root.close().unwrap();
            let remote = inspect_model_metadata(&metadata, &request, &Mechanisms, MEDIA);
            assert_parity(local, remote);
        }
    }
}

#[test]
#[ignore = "requires EREDU_METADATA_GGUF pointing to a pinned released text checkpoint"]
fn released_gguf_metadata_forecast_parity() {
    use std::io::Read;
    let path = std::path::PathBuf::from(
        std::env::var_os("EREDU_METADATA_GGUF").expect("EREDU_METADATA_GGUF"),
    );
    let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
    let headers = checkpoint
        .shards()
        .iter()
        .map(|shard| {
            let prefix = shard
                .tensors()
                .iter()
                .map(|t| t.descriptor().data_offset)
                .min()
                .unwrap();
            let mut file = std::fs::File::open(shard.path()).unwrap();
            let file_len = file.metadata().unwrap().len();
            let mut bytes = vec![0; prefix as usize];
            file.read_exact(&mut bytes).unwrap();
            GgufHeader {
                member: shard
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
                bytes,
                file_len,
            }
        })
        .collect();
    let metadata = ArtifactMetadata {
        provenance: provenance(),
        checkpoint: CheckpointMetadata::Gguf {
            headers,
            companions: Vec::new(),
        },
        sidecars: BTreeMap::new(),
    };
    let request = NormalizedLoadRequest::default();
    assert_parity(
        eredu_architectures::inspect_model(&path, &request, &Mechanisms, MEDIA),
        inspect_model_metadata(&metadata, &request, &Mechanisms, MEDIA),
    );
}

#[test]
fn supplied_affine_safetensors_forecasts_include_quantization_companions() {
    let mut config = config();
    config["quantization"] = serde_json::json!({"group_size":32,"bits":4});
    for split in [false, true] {
        let (root, metadata) = safetensors_fixture_with_config(split, config.clone());
        let request = NormalizedLoadRequest::default();
        let local = eredu_architectures::inspect_model(root.path(), &request, &Mechanisms, MEDIA);
        root.close().unwrap();
        assert_parity(
            local,
            inspect_model_metadata(&metadata, &request, &Mechanisms, MEDIA),
        );
    }
}
