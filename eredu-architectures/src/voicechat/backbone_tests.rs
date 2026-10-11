use super::*;
use crate::voicechat::{artifact::tests as fixtures, checkpoint::GgufLayout};
use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{MemoryWeightStore, TensorSelection},
};
use std::{collections::BTreeSet, sync::Arc};

fn released_args() -> ModelArgs {
    // The published split backbone carries the same released geometry and
    // schedule. Reuse its pinned metadata instead of copying another config.
    let checkpoint = fixtures::header_checkpoint(GgufLayout::Backbone);
    let metadata = checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    nemotron_h::model_args_from_gguf_catalog(&checkpoint, &metadata).unwrap()
}

#[test]
fn released_layouts_project_to_shared_backbone_names_and_layer_groups() {
    let args = released_args();
    let official = fixtures::official_parameters();
    let complete = fixtures::admitted(GgufLayout::Complete);
    let split = fixtures::admitted(GgufLayout::Backbone);
    let mut reference_names = None;
    for parameters in [&official, complete.parameters(), split.parameters()] {
        let projected = project_parameters(&args, parameters).unwrap();
        assert_eq!(projected.len(), 341);
        let names = projected.keys().cloned().collect::<BTreeSet<_>>();
        if let Some(reference) = &reference_names {
            assert_eq!(&names, reference);
        } else {
            reference_names = Some(names.clone());
        }
        // These are the names consumed by shared static, Mamba, MLP and
        // attention modules, including the layer-normalization sibling.
        for (runtime, canonical) in [
            ("model.embeddings.weight", "stt_model.embed_tokens.weight"),
            ("model.norm_f.weight", "stt_model.llm.norm_f.weight"),
            ("lm_head.weight", "stt_model.lm_head.weight"),
            (
                "model.layers.0.mamba.A_log",
                "stt_model.llm.layers.0.mixer.A_log",
            ),
            (
                "model.layers.0.mamba.in_proj.weight",
                "stt_model.llm.layers.0.mixer.in_proj.weight",
            ),
            (
                "model.layers.0.norm.weight",
                "stt_model.llm.layers.0.norm.weight",
            ),
            (
                "model.layers.1.mlp.up_proj.weight",
                "stt_model.llm.layers.1.mixer.up_proj.weight",
            ),
            (
                "model.layers.14.attention.q_proj.weight",
                "stt_model.llm.layers.14.mixer.q_proj.weight",
            ),
        ] {
            assert_eq!(projected[runtime].name, canonical);
        }
        // Grouping is metadata-only; this empty source cannot authorize any
        // released weight access and is never used by these recipe queries.
        let prepared = PreparedBackboneParameters {
            args: args.clone(),
            source: Arc::new(MemoryWeightStore::default()),
            parameters: projected,
        };
        let mut grouped = prepared.static_recipes();
        assert_eq!(
            grouped.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "lm_head.weight",
                "model.embeddings.weight",
                "model.norm_f.weight",
            ]
        );
        for index in 0..56 {
            let layer = prepared.layer_recipes(index).unwrap();
            assert!(!layer.is_empty(), "missing released layer {index}");
            assert!(layer
                .keys()
                .all(|name| name.starts_with(&format!("model.layers.{index}."))));
            for (name, recipe) in layer {
                assert!(grouped.insert(name, recipe).is_none());
            }
        }
        assert_eq!(grouped.keys().cloned().collect::<BTreeSet<_>>(), names);
        assert!(prepared.layer_recipes(56).is_err());
    }
}

#[test]
fn mamba_transition_recipes_restore_split_storage_exactly_once() {
    let args = released_args();
    let official = fixtures::official_parameters();
    let complete = fixtures::admitted(GgufLayout::Complete);
    let split = fixtures::admitted(GgufLayout::Backbone);
    let runtime = "model.layers.0.mamba.A_log";
    let canonical = "stt_model.llm.layers.0.mixer.A_log";
    for parameters in [&official, complete.parameters()] {
        let projected = project_parameters(&args, parameters).unwrap();
        assert_eq!(
            projected[runtime].recipe,
            DerivedWeightRecipe::source(canonical, TensorSelection::Full)
        );
        assert_eq!(projected[runtime].output.shape, [128]);
    }
    let projected = project_parameters(&args, split.parameters()).unwrap();
    // The converter stores -exp(A_log) with a singleton trailing dimension.
    // A literal complete expression detects both skipped and double NegLog.
    assert_eq!(
        projected[runtime].recipe,
        DerivedWeightRecipe::NegLog {
            input: Box::new(DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source(
                    "blk.0.ssm_a",
                    TensorSelection::Full
                )),
                shape: vec![128],
            }),
        }
    );
    assert_eq!(projected[runtime].output.shape, [128]);
    assert_eq!(projected[runtime].output.dtype, RecipeDtype::F32);
    assert_eq!(projected[runtime].output.byte_len, 128 * 4);
    assert!(matches!(
        project_parameters(&args, &BTreeMap::new()),
        Err(BackbonePreparationError::MissingComponent)
    ));
}

#[test]
fn released_gguf_backbone_recipes_pass_normalized_catalog_admission_without_reads() {
    use eredu_checkpoint::{
        gguf_store::GgufCatalog,
        store::{
            CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
            WeightStoreDiagnostics,
        },
        validation::resolve_gguf_plan,
    };

    struct HeaderSource(GgufCatalog);
    impl CheckpointSource for HeaderSource {
        fn source_keys(&self) -> Vec<String> {
            self.0.keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0.metadata(key)
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("normalized backbone admission must not read checkpoint payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    let config = super::super::config::VoiceChatConfig::from_json(include_str!(
        "fixtures/official-inference-config.json"
    ))
    .unwrap();
    for layout in [GgufLayout::Complete, GgufLayout::Backbone] {
        let checkpoint = fixtures::header_checkpoint(layout);
        let plan = super::super::checkpoint::gguf_plan(&config, layout).unwrap();
        let resolution = resolve_gguf_plan(&checkpoint, &plan).unwrap();
        let mapping = checkpoint
            .tensors()
            .flat_map(|tensor| {
                tensor
                    .outputs()
                    .iter()
                    .map(move |output| eredu_gguf::TranslatedTensorLayout {
                        physical_name: tensor.descriptor().name.clone(),
                        original_name: output.name.clone(),
                        layout: output.clone(),
                    })
            })
            .collect::<Vec<_>>();
        let source = Arc::new(HeaderSource(
            GgufCatalog::from_resolved_checkpoint(&checkpoint, &resolution, &mapping).unwrap(),
        ));
        let admitted = fixtures::admitted(layout);
        let prepared = PreparedBackboneParameters::from_parameters(
            released_args(),
            source,
            admitted.parameters(),
        )
        .unwrap();
        assert_eq!(prepared.parameters().len(), 341);
        assert_eq!(prepared.static_recipes().len(), 3);
        let projected = &prepared.parameters()["model.layers.0.mamba.A_log"];
        let original = &admitted.parameters()["stt_model.llm.layers.0.mixer.A_log"];
        let expected = if original.output.dtype == RecipeDtype::F32 {
            original.recipe.clone()
        } else {
            DerivedWeightRecipe::Cast {
                input: Box::new(original.recipe.clone()),
                dtype: RecipeDtype::F32,
            }
        };
        assert_eq!(projected.recipe, expected);
        assert_eq!(projected.output.dtype, RecipeDtype::F32);
    }
}

#[test]
fn resident_backbone_bindings_keep_transforms_and_restrict_physical_sources() {
    use eredu_checkpoint::store::{ReadPolicy, TensorReadRequest};
    use eredu_nn::{Parameter, ParameterSpec};

    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "blk.0.ssm_a".into(),
                safetensors::Dtype::F32,
                vec![2, 1],
                [-1.0f32, -2.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
            (
                "blk.1.ssm_a".into(),
                safetensors::Dtype::F32,
                vec![1],
                (-3.0f32).to_le_bytes().to_vec(),
            ),
        ])
        .unwrap(),
    );
    let name = "model.layers.0.mamba.A_log";
    let recipes = BTreeMap::from([(
        name.into(),
        DerivedWeightRecipe::NegLog {
            input: Box::new(DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source(
                    "blk.0.ssm_a",
                    TensorSelection::Full,
                )),
                shape: vec![2],
            }),
        },
    )]);
    let slot = Parameter::new(ParameterSpec::trainable(name).unwrap(), ());
    let (restricted, bindings) = construction::prepare_bindings(
        source.clone(),
        recipes.clone(),
        &slot,
        BackboneUnit::Layer(0),
    )
    .unwrap();
    assert_eq!(restricted.source_keys(), ["blk.0.ssm_a"]);
    assert!(restricted
        .acquire_lease(TensorReadRequest {
            key: "blk.1.ssm_a".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .is_err());
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name(), name);
    assert_eq!(bindings[0].recipe(), Some(&recipes[name]));
    // Exact module identity is admitted before a backend can read any payload.
    // This catches stale recipes or accidental reuse of the neighboring unit.
    let wrong = Parameter::new(
        ParameterSpec::trainable("model.layers.1.mamba.A_log").unwrap(),
        (),
    );
    assert!(matches!(
        construction::prepare_bindings(source, recipes, &wrong, BackboneUnit::Layer(1)),
        Err(BackboneConstructionError::Parameters {
            unit: BackboneUnit::Layer(1),
            missing,
            unexpected,
        }) if missing == ["model.layers.1.mamba.A_log"]
            && unexpected == ["model.layers.0.mamba.A_log"]
    ));
}

#[test]
fn released_backbone_residency_counts_static_heads_and_largest_loading_unit_without_reads() {
    use crate::voicechat::component_residency::{BackboneResources, VoiceChatResidencyError};
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    };

    struct HeaderSource(BTreeMap<String, TensorMetadata>);
    impl CheckpointSource for HeaderSource {
        fn source_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("component residency admission must not read checkpoint payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }
    let args = released_args();
    let source = Arc::new(HeaderSource(
        crate::voicechat::checkpoint::tests::official()
            .tensors()
            .clone(),
    ));
    let mut prepared =
        PreparedBackboneParameters::from_parameters(args, source, &fixtures::official_parameters())
            .unwrap();
    let resources = BackboneResources::from_prepared(&prepared).unwrap();
    // Independent sums of the pinned SafeTensors data_offsets: the static unit
    // contains two 131072 x 4480 F32 matrices and a 4480-element F32 norm.
    assert_eq!(resources.parameter_bytes, 35_552_909_312);
    assert_eq!(resources.static_parameter_bytes, 4_697_638_400);
    assert_eq!(resources.layer_parameter_bytes, 30_855_270_912);
    // Mamba block 0 is the largest released unit; its physical header spans
    // 589802496 bytes. Direct F32 recipes reserve source plus output storage.
    assert_eq!(resources.largest_layer_bytes, 589_802_496);
    assert_eq!(resources.layer_materialization_peak_bytes, 1_179_604_992);
    assert_eq!(resources.static_materialization_peak_bytes, 9_395_276_800);
    assert_eq!(resources.materialization_peak_bytes, 9_395_276_800);

    prepared.args.tie_word_embeddings = true;
    assert!(matches!(
        BackboneResources::from_prepared(&prepared),
        Err(VoiceChatResidencyError::UnexpectedTiedHead)
    ));
    prepared.parameters.remove("lm_head.weight");
    let resources = BackboneResources::from_prepared(&prepared).unwrap();
    // Tied readout uses the single embedding owner, with no duplicate head slot.
    assert_eq!(resources.parameter_bytes, 33_204_099_072);
    assert_eq!(resources.static_parameter_bytes, 2_348_828_160);
    assert_eq!(resources.layer_parameter_bytes, 30_855_270_912);
    assert_eq!(resources.materialization_peak_bytes, 4_697_656_320);
}
