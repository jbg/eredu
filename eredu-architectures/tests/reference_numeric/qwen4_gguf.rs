//! Independent miniature published layout, never generated from production recipes.
use super::*;
use eredu_architectures::qwen4_exp::checkpoint::{
    gguf_text::{GgufTextPlan, GgufTextWeights},
    recipes::ParameterScope,
};
use eredu_gguf::{Checkpoint, GgmlType, MetadataValue as V};

use eredu_evaluation::qwen4_exp::metadata;
pub(super) use eredu_evaluation::qwen4_exp::Fixture;

fn bind_text(path: &std::path::Path) -> GgufTextWeights {
    let plan = GgufTextPlan::prepare(&Checkpoint::open(path).unwrap()).unwrap();
    let source = open_text_source(&plan);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    for key in plan.catalog().keys() {
        assert_eq!(
            plan.catalog().metadata(&key).unwrap(),
            source.source_metadata(&key).unwrap()
        );
        assert_eq!(
            plan.catalog().source_provenance(&key).unwrap(),
            source.source_provenance(&key).unwrap()
        );
    }
    let scopes = [
        ParameterScope::Static,
        ParameterScope::Target(0),
        ParameterScope::Target(1),
        ParameterScope::Lexical(0),
    ];
    let cold_recipes: Vec<_> = scopes
        .iter()
        .map(|scope| plan.parameter_recipes(*scope).unwrap())
        .collect();
    let cold_catalog = plan.catalog().clone();
    let prepared = plan.bind(source.clone()).unwrap();
    for (scope, recipes) in scopes.into_iter().zip(cold_recipes) {
        let owner = prepared.parameters(scope).unwrap();
        assert_eq!(owner.recipes(), &recipes);
        for recipe in recipes.values() {
            assert_eq!(
                recipe.infer(&cold_catalog).unwrap(),
                recipe.infer(owner.source().as_ref()).unwrap()
            );
        }
    }
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    prepared
}

pub(super) fn open_text_source(
    plan: &GgufTextPlan,
) -> eredu_checkpoint::store::SharedCheckpointSource {
    Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .add_resolved_checkpoint(plan.checkpoint().clone(), plan.resolution(), plan.mapping())
            .unwrap()
            .build()
            .unwrap(),
    )
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 2e-7, "{a} != {b}");
    }
}

#[test]
fn published_text_recipes_invert_exporter_and_bound_each_owner() {
    let f = Fixture::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("text.gguf");
    f.write(&path);
    let prepared = bind_text(&path);
    let mut seen = BTreeSet::new();
    for scope in [
        ParameterScope::Static,
        ParameterScope::Target(0),
        ParameterScope::Target(1),
        ParameterScope::Lexical(0),
    ] {
        let owner = prepared.parameters(scope).unwrap();
        if scope == ParameterScope::Static {
            let d = owner.source().source_diagnostics().unwrap();
            assert_eq!(d.physical_reads, 0);
            assert!(d.payload_shard_paths.is_empty());
        }
        assert!(owner
            .source()
            .source_metadata("per_layer_token_embd.weight")
            .is_err());
        assert!(owner
            .source()
            .source_keys()
            .iter()
            .all(|k| !k.contains(".experts.")));
        for (key, recipe) in owner.recipes() {
            let actual =
                payload::recipe_value(recipe, owner.source().as_ref(), &NumericContext::default())
                    .unwrap();
            if key.ends_with("conv1d.weight") {
                assert_eq!(actual.shape.len(), 3);
                assert_eq!(actual.shape[1], 1);
            }
            assert_close(&actual.data, &f.expected[key]);
            seen.insert(key.clone());
        }
    }
    assert_eq!(
        seen.len(),
        f.expected
            .keys()
            .filter(|k| !k.contains(".experts."))
            .count()
    );
    for layer in 0..2 {
        for expert in [0, 2] {
            let owner = prepared.expert(layer, expert).unwrap();
            assert_eq!(owner.source().source_keys().len(), 3);
            let before = owner.source().source_diagnostics().unwrap();
            for (key, recipe) in owner.recipes() {
                let root = format!("model.layers.{layer}.mlp.experts");
                let value = payload::recipe_value(
                    recipe,
                    owner.source().as_ref(),
                    &NumericContext::default(),
                )
                .unwrap();
                let slice = |name: &str| {
                    f.expected[&format!("{root}.{name}")][expert * 1024..(expert + 1) * 1024]
                        .to_vec()
                };
                let expected = if key.ends_with("gate_up_proj") {
                    [slice("gate_proj"), slice("up_proj")].concat()
                } else {
                    slice("down_proj")
                };
                assert_close(&value.data, &expected);
                assert_eq!(value.shape[0], 1);
                for source in recipe.source_keys() {
                    assert!(source.contains(".experts."));
                }
            }
            let after = owner.source().source_diagnostics().unwrap();
            assert_eq!(after.physical_reads - before.physical_reads, 3);
            assert_eq!(
                after.physical_read_bytes - before.physical_read_bytes,
                3 * 1024 * 4
            );
        }
    }
    assert!(prepared.expert(0, 3).is_err());
    assert!(prepared.parameters(ParameterScope::Prediction(0)).is_err());
}

#[test]
fn quantized_head_crossing_decodes_one_matrix_and_reorders_exact_upstream_values() {
    let mut f = Fixture::new();
    let hex = |s: &str| {
        s.as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
    };
    let line = include_str!("../../../eredu-gguf/tests/fixtures/scalar-block-rows.txt")
        .lines()
        .find(|l| l.starts_with("12|"))
        .unwrap();
    let parts: Vec<_> = line.split('|').collect();
    let raw = hex(parts[1]);
    let oracle: Vec<f32> = hex(parts[2])
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let tensor = f
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ssm_out.weight")
        .unwrap();
    tensor.ty = GgmlType::Q4K;
    tensor.data = raw.repeat(32);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quant.gguf");
    f.write(&path);
    let prepared = bind_text(&path);
    let key = "model.layers.0.linear_attn.out_proj.weight";
    assert_eq!(
        prepared.format(key),
        Some(eredu_checkpoint::LinearFormat::Dense)
    );
    let owner = prepared.parameters(ParameterScope::Target(0)).unwrap();
    let before = owner.source().source_diagnostics().unwrap();
    let value = payload::recipe_value(
        &owner.recipes()[key],
        owner.source().as_ref(),
        &NumericContext::default(),
    )
    .unwrap();
    let expected: Vec<f32> = (0..32)
        .flat_map(|_| {
            [0usize, 2, 4, 1, 3, 5]
                .into_iter()
                .flat_map(|head| oracle[head * 128..(head + 1) * 128].iter().copied())
        })
        .collect();
    assert_eq!(value.data, expected);
    let after = owner.source().source_diagnostics().unwrap();
    assert_eq!(after.physical_reads - before.physical_reads, 1);
    assert_eq!(
        after.physical_read_bytes - before.physical_read_bytes,
        32 * 3 * 144
    );
    assert_eq!(
        owner.source().source_metadata(key).unwrap().logical_shape,
        [32, 768]
    );
    assert!(!owner
        .recipes()
        .contains_key("model.layers.0.linear_attn.out_proj.scales"));
}

#[test]
fn missing_and_malformed_text_descriptors_reject_before_materialization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.gguf");
    let mut f = Fixture::new();
    f.tensors
        .retain(|t| t.name != "blk.1.indexer.k_proj.weight");
    f.write(&path);
    assert!(GgufTextPlan::prepare(&Checkpoint::open(&path).unwrap()).is_err());
    let mut f = Fixture::new();
    let t = f
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ssm_conv1d.weight")
        .unwrap();
    t.shape = vec![400, 6];
    f.write(&path);
    assert!(GgufTextPlan::prepare(&Checkpoint::open(&path).unwrap()).is_err());
}

#[test]
fn heterogeneous_expert_pair_decodes_only_selected_expert_and_promotes_dense_member() {
    let mut f = Fixture::new();
    let gate = f
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight")
        .unwrap();
    let mut row = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    row.extend((0..16).map(|i| i | ((15 - i) << 4)));
    gate.ty = GgmlType::Q4_0;
    gate.data = row.repeat(96);
    let up = f
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight")
        .unwrap();
    up.ty = GgmlType::F16;
    up.data = up
        .data
        .chunks_exact(4)
        .flat_map(|b| half::f16::from_f32(f32::from_le_bytes(b.try_into().unwrap())).to_le_bytes())
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.gguf");
    f.write(&path);
    let prepared = bind_text(&path);
    let key = "model.layers.0.mlp.experts.gate_up_proj";
    assert_eq!(
        prepared.format(key),
        Some(eredu_checkpoint::LinearFormat::Dense)
    );
    let owner = prepared.expert(0, 2).unwrap();
    let actual = payload::recipe_value(
        &owner.recipes()[key],
        owner.source().as_ref(),
        &NumericContext::default(),
    )
    .unwrap();
    let mut expected: Vec<_> = (0..1024)
        .map(|i| ((if i % 32 < 16 { i % 16 } else { 31 - i % 32 }) as f32 - 8.) * 0.5)
        .collect();
    expected.extend_from_slice(&f.expected["model.layers.0.mlp.experts.up_proj"][2048..3072]);
    assert_eq!(actual.data, expected);
    let d = owner.source().source_diagnostics().unwrap();
    assert_eq!(d.physical_reads, 2);
    assert_eq!(d.physical_read_bytes, 32 * 18 + 1024 * 2);
}

#[test]
fn encoded_recurrent_rows_reorder_every_affine_companion() {
    use eredu_checkpoint::store::{CheckpointLease, MemoryWeightStore};
    let mut f = Fixture::new();
    let matrix = f
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ssm_beta.weight")
        .unwrap();
    matrix.ty = GgmlType::Q4_0;
    matrix.data.clear();
    for scale in [1., 4., 2., 5., 3., 6.] {
        matrix.data.extend(half::f16::from_f32(scale).to_le_bytes());
        matrix.data.extend([0x98; 16]);
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("companions.gguf");
    f.write(&path);
    let prepared = bind_text(&path);
    let owner = prepared.parameters(ParameterScope::Target(0)).unwrap();
    let root = "model.layers.0.linear_attn.in_proj_b";
    assert!(matches!(
        prepared.format(&format!("{root}.weight")),
        Some(eredu_checkpoint::LinearFormat::Affine(_))
    ));
    for (suffix, multiplier) in [("scales", 1.), ("biases", -8.)] {
        let key = format!("{root}.{suffix}");
        let lease = owner
            .source()
            .acquire_lease(TensorReadRequest {
                key: key.clone(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let CheckpointLease::Gguf(lease) = lease else {
            panic!("GGUF")
        };
        let materialized = lease.materialize_portable().unwrap();
        let eredu_gguf::ConvertedTensor::Affine(a) = materialized.converted() else {
            panic!("affine")
        };
        // Feed actual converted companion bytes through the shared neutral recipe evaluator.
        let bits = if suffix == "scales" {
            &a.scales
        } else {
            &a.biases
        };
        let source = MemoryWeightStore::from_safetensors([(
            key.clone(),
            safetensors::Dtype::F16,
            vec![6, 1],
            bits.iter().flat_map(|n| n.to_le_bytes()).collect(),
        )])
        .unwrap();
        let actual =
            payload::recipe_value(&owner.recipes()[&key], &source, &NumericContext::default())
                .unwrap();
        assert_eq!(
            actual.data,
            (1..=6).map(|n| n as f32 * multiplier).collect::<Vec<_>>()
        );
    }
}

#[test]
fn tied_readout_squeezed_gate_and_oversized_projection_headers() {
    let mut f = Fixture::new();
    f.tensors.retain(|t| t.name != "output.weight");
    f.tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_gate_inp_shexp.weight")
        .unwrap()
        .shape = vec![32];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tied.gguf");
    f.write(&path);
    let prepared = bind_text(&path);
    let owner = prepared.parameters(ParameterScope::Static).unwrap();
    assert_eq!(
        owner.recipes()["lm_head.weight"],
        owner.recipes()["model.embed_tokens.weight"]
    );
    assert_eq!(
        prepared.format("lm_head.weight"),
        prepared.format("model.embed_tokens.weight")
    );
    let owner = prepared.parameters(ParameterScope::Target(0)).unwrap();
    let recipe = &owner.recipes()["model.layers.0.mlp.shared_expert_gate.weight"];
    assert_eq!(
        recipe.infer(owner.source().as_ref()).unwrap().shape,
        [1, 32]
    );
    assert!(matches!(
        prepared.parameters(ParameterScope::Prediction(0)),
        Err(eredu_architectures::qwen4_exp::prepared::PreparationError::MissingPrediction)
    ));
    let mut metadata = metadata();
    metadata.insert(
        "qwen4exp.attention.head_count".into(),
        V::Uint32(2_000_000_000),
    );
    f.write_metadata(&path, &metadata);
    assert!(GgufTextPlan::prepare(&Checkpoint::open(&path).unwrap()).is_err());
}

#[test]
fn released_text_header_plan_retains_bounded_fallbacks_without_a_readable_artifact() {
    let mut fixture = Fixture::new();
    let mut block = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    block.extend([0x98; 16]);
    fixture.replace_encoding(
        "blk.0.ffn_gate_exps.weight",
        GgmlType::Q4_0,
        block.repeat(96),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("headers-only.gguf");
    fixture.write(&path);
    let checkpoint = Checkpoint::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let plan = GgufTextPlan::prepare(&checkpoint).unwrap();
    let key = "model.layers.0.mlp.experts.gate_up_proj";
    assert_eq!(
        plan.format(key),
        Some(eredu_checkpoint::LinearFormat::Dense)
    );
    let expert = plan.expert_recipes(0, 2).unwrap();
    let metadata = expert[key].infer(plan.catalog()).unwrap();
    assert_eq!(metadata.shape, [1, 64, 32]);
    assert_eq!(metadata.dtype, eredu_checkpoint::recipe::RecipeDtype::F32);
    assert!(matches!(
        plan.parameter_recipes(ParameterScope::Prediction(0)),
        Err(eredu_architectures::qwen4_exp::prepared::PreparationError::MissingPrediction)
    ));
}
