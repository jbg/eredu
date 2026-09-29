//! Independently exported miniature weights through the ordinary selected sessions.
use super::super::qwen4_gguf::Fixture;
use super::*;
use eredu_architectures::qwen4_exp::prepared::{
    GgufTargetPlan, PreparedTarget, TargetExecutionPlan,
};
use eredu_checkpoint::store::CheckpointSource;
use eredu_runtime::*;

use eredu_evaluation::qwen4_exp::limits;

pub(super) fn fixtures(quantized: bool) -> (tempfile::TempDir, PreparedTarget, PreparedTarget) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new();
    if quantized {
        let decode_hex = |s: &str| {
            s.as_bytes()
                .chunks_exact(2)
                .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
                .collect::<Vec<_>>()
        };
        let fixture = include_str!("../../../../eredu-gguf/tests/fixtures/scalar-block-rows.txt")
            .lines()
            .find(|l| l.starts_with("12|"))
            .unwrap();
        let columns: Vec<_> = fixture.split('|').collect();
        let bytes = decode_hex(columns[1]);
        let values = decode_hex(columns[2]);
        let raw = [&bytes[..288], &bytes[..144]].concat();
        let values: Vec<_> = [&values[..2048], &values[..1024]]
            .concat()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        f.replace_encoding(
            "blk.0.ssm_out.weight",
            eredu_gguf::GgmlType::Q4K,
            raw.repeat(32),
        );
        let original = (0..32)
            .flat_map(|_| {
                [0usize, 2, 4, 1, 3, 5]
                    .into_iter()
                    .flat_map(|h| values[h * 128..(h + 1) * 128].iter().copied())
            })
            .collect();
        f.expected.insert(
            "model.layers.0.linear_attn.out_proj.weight".into(),
            original,
        );
    }
    let prepared = f.prepare(dir.path()).unwrap();
    (dir, prepared.gguf, prepared.safetensors)
}
pub(super) fn plan(target: &PreparedTarget) -> TargetExecutionPlan {
    target
        .execution_plan(
            stream_bindings(target.spec()),
            eredu_runtime::SelectedRowLookupPlans::select(
                target
                    .row_lookups(
                        RowLookupLimits {
                            requests: 128,
                            rows_per_acquisition: 2,
                            acquisition_bytes: 128,
                            host_bytes: 1 << 16,
                            output_bytes: 32768,
                        },
                        eredu_core::residency::ResidencyPolicy::Cacheable,
                    )
                    .unwrap()
                    .descriptors()
                    .clone(),
                ParameterBankLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
                    1 << 20,
                    1 << 20,
                )
                .unwrap(),
                0,
                &cold::Support,
            )
            .unwrap(),
        )
        .unwrap()
}
pub(super) fn state_from_layout(layout: &eredu_runtime::StateLayout) -> Result<State, Error> {
    DeviceState::create(layout.clone(), |_, policy| {
        let mut state = NumericHybridLayerState::new(policy);
        for p in policy.append_streams() {
            for lane in 0..2 {
                let element = match p.dtype() {
                    eredu_core::cache::StateTensorDtype::Int32 => eredu_nn::TensorElementType::I32,
                    _ => eredu_nn::TensorElementType::F32,
                };
                state.streams.push((
                    p.slot(),
                    lane,
                    ResidentAppendStream::new(
                        AppendStreamSpec {
                            slot: p.slot(),
                            width: p.width(),
                            element,
                        },
                        AppendStreamLimits {
                            entries: 64,
                            page_entries: 2,
                            read_entries: 2,
                        },
                        65536,
                        65536,
                        65536,
                    )
                    .unwrap(),
                ));
            }
        }
        Ok(state)
    })
}
fn run(
    target: &PreparedTarget,
    residency: LayerWeightResidency,
    chunks: &[usize],
) -> (Vec<NumericTensor>, State) {
    run_with_header(target, residency, chunks, None)
}
enum ColdHeader {
    Gguf(GgufTargetPlan),
    Safetensors(eredu_architectures::qwen4_exp::prepared::SafetensorsTargetPlan),
}
fn run_with_header(
    target: &PreparedTarget,
    residency: LayerWeightResidency,
    chunks: &[usize],
    header: Option<ColdHeader>,
) -> (Vec<NumericTensor>, State) {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let normalized = super::load_policy::request(residency);
    let plan = target
        .execution_plan_for_load(&normalized, &cold::Support)
        .unwrap();
    let capabilities = cold::capabilities(plan.requirements(), None);
    let before = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let request = plan.load_selection_request().unwrap().clone();
    let expected_sources = plan
        .requirements()
        .text()
        .parameters()
        .iter()
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    let delegated_sources = target
        .artifact()
        .source_keys()
        .into_iter()
        .filter(|key| !expected_sources.contains(key))
        .collect::<Vec<_>>();
    let control_reads = if matches!(header, Some(ColdHeader::Safetensors(_))) {
        3
    } else {
        0
    };
    let selected = if let Some(ColdHeader::Gguf(header)) = header {
        let cold = header
            .execution_plan_for_load(
                &normalized,
                eredu_nn::TensorElementType::F32,
                &cold::Support,
            )
            .unwrap();
        assert_eq!(cold.requirements(), plan.requirements());
        assert_eq!(cold.load_selection_request(), plan.load_selection_request());
        cold.select(&request, &capabilities, None)
            .unwrap()
            .bind(target.artifact().clone(), None)
            .unwrap()
    } else if let Some(ColdHeader::Safetensors(header)) = header {
        let cold = header
            .execution_plan_for_load(
                &normalized,
                eredu_nn::TensorElementType::F32,
                &cold::Support,
                super::safetensors_admission::physical_sources(target.artifact().as_ref()),
            )
            .unwrap();
        assert_eq!(cold.requirements(), plan.requirements());
        assert_eq!(cold.load_selection_request(), plan.load_selection_request());
        cold.select(&request, &capabilities, None)
            .unwrap()
            .bind(target.artifact().clone(), None)
            .unwrap()
    } else {
        plan.select(&request, &capabilities, None).unwrap()
    };
    let after_binding = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    assert_eq!(after_binding, before + control_reads);
    let workspace = selected
        .parameter_materialization_workspace(
            &super::super::prepared_adapter::NumericPreparationProvider { addressable: true },
        )
        .unwrap();
    assert!(workspace.ordinary_recipe_peak_bytes >= 32 * 768 * 4);
    assert!(workspace.ordinary_native_peak_bytes >= workspace.ordinary_recipe_peak_bytes);
    let (prepared, source) = selected.prepare::<NumericBackend, State>(&ctx).unwrap();
    assert_eq!(
        source.source_keys().into_iter().collect::<BTreeSet<_>>(),
        expected_sources
    );
    for key in delegated_sources {
        assert!(
            source.source_metadata(&key).is_err(),
            "separate role remains inaccessible: {key}"
        );
        assert!(
            source.acquire_lease(key.as_str().into()).is_err(),
            "separate role cannot acquire payload through execution view: {key}"
        );
    }

    assert_eq!(
        target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        after_binding,
        "cold construction reads no payload"
    );
    let mut mechanisms = if residency.is_fully_resident() {
        NumericReplicatedMechanisms::with_bound_checkpoint(source)
    } else {
        NumericReplicatedMechanisms::with_bounded_checkpoint(source)
    };
    mechanisms.fixture_state_factory = Some(state_from_layout);
    macro_rules! exercise {
        ($session:expr) => {{
            let mut session = $session;
            let ids = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            let mut offset = 0;
            let mut last = None;
            for &length in chunks {
                let token = NumericTensor::from_i32_slice(
                    &ids[offset..offset + length],
                    &[1, length as i32],
                    &ctx,
                )
                .unwrap();
                last = Some(session.prefill(&token, None, &ctx).unwrap());
                offset += length;
            }
            assert_eq!(offset, ids.len());
            let mut outputs = vec![last.unwrap()];
            for id in 0..16 {
                let token = NumericTensor::from_i32_slice(&[id], &[1, 1], &ctx).unwrap();
                outputs.push(session.decode(&token, &ctx).unwrap());
            }
            let saved = session.checkpoint(&ctx).unwrap();
            let token = NumericTensor::from_i32_slice(&[5], &[1, 1], &ctx).unwrap();
            let a = session.decode(&token, &ctx).unwrap();
            let reads = target
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads;
            session.rollback(saved.clone(), &ctx).unwrap();
            assert_eq!(
                target
                    .artifact()
                    .source_diagnostics()
                    .unwrap()
                    .physical_reads,
                reads
            );
            session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &saved);
            assert_tensor_exact(&session.decode(&token, &ctx).unwrap(), &a, "GGUF rollback");
            Ok::<_, String>((outputs, saved))
        }};
    }
    eredu_architectures::prepared_execution::construct_selected_routed_session(prepared,mechanisms,&ctx,|_,_,rows| {
        let rows=rows.unwrap();
        let providers=rows.prepared().bind(rows.prepared().entries().iter().map(|(id,e)|(id.clone(),super::super::row_bank::SourceRows::new(e))).collect()).unwrap();
        Ok::<_,String>((BTreeMap::<RoutedBankId,(NumericGroupedBankMechanism,NumericIndexedMovement)>::new(),Some(providers)))
    },(),|_,s,_|exercise!(s),|_,s,_|exercise!(s)).unwrap()
}
#[test]
fn miniature_target_matches_frozen_neutral_trajectory() {
    let (_dir, _gguf, st) = fixtures(false);
    let checkpoint = eredu_gguf::Checkpoint::open(_dir.path().join("target.gguf")).unwrap();
    let mut config =
        eredu_architectures::qwen4_exp::config::Config::from_gguf(&checkpoint).unwrap();
    config.ngram.source = eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        vocabulary_base: 5,
        vocabulary_alignment: 1,
        shards: 1,
        seed: 1,
    };
    let st_header = eredu_architectures::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
        st.artifact().as_ref(),
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
    )
    .unwrap();
    let reference = run_with_header(
        &st,
        LayerWeightResidency::FullyResident,
        &[19],
        Some(ColdHeader::Safetensors(st_header.clone())),
    );
    let rows: Vec<Vec<f32>> = reference
        .0
        .iter()
        .map(|value| value.data[value.data.len() - 16..].to_vec())
        .collect();
    let expected: serde_json::Value =
        serde_json::from_str(eredu_evaluation::qwen4_exp::DENSE_TRAJECTORY_JSON).unwrap();
    let expected_rows = expected["logits"].as_array().unwrap();
    assert_eq!(rows.len(), expected_rows.len());
    let absolute = expected["absolute_tolerance"].as_f64().unwrap() as f32;
    let relative = expected["relative_tolerance"].as_f64().unwrap() as f32;
    for (step, (row, expected)) in rows.iter().zip(expected_rows).enumerate() {
        let expected = expected.as_array().unwrap();
        assert_eq!(row.len(), expected.len());
        for (actual, expected) in row.iter().zip(expected) {
            let expected = expected.as_f64().unwrap() as f32;
            assert!(
                actual.is_finite()
                    && (actual - expected).abs() <= absolute + relative * expected.abs(),
                "frozen neutral trajectory step {step}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn gguf_target_uses_shared_selection_and_sessions_across_residency() {
    let (_dir, gguf, st) = fixtures(false);
    assert_eq!(
        gguf.artifact().source_diagnostics().unwrap().physical_reads,
        0
    );
    assert_eq!(gguf.spec().units.len(), 3);
    assert!(gguf
        .artifact()
        .source_keys()
        .iter()
        .all(|n| !n.contains("ngram_embedding")));
    let checkpoint = eredu_gguf::Checkpoint::open(_dir.path().join("target.gguf")).unwrap();
    let mut config =
        eredu_architectures::qwen4_exp::config::Config::from_gguf(&checkpoint).unwrap();
    config.ngram.source = eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        vocabulary_base: 5,
        vocabulary_alignment: 1,
        shards: 1,
        seed: 1,
    };
    let st_header = eredu_architectures::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
        st.artifact().as_ref(),
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
    )
    .unwrap();
    let reference = run(&st, LayerWeightResidency::FullyResident, &[19]);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let header = GgufTargetPlan::prepare(
            &eredu_gguf::Checkpoint::open(_dir.path().join("target.gguf")).unwrap(),
        )
        .unwrap();
        let actual = run_with_header(
            &gguf,
            residency,
            &[1, 2, 4, 3, 9],
            Some(ColdHeader::Gguf(header)),
        );
        let st_actual = run_with_header(
            &st,
            residency,
            &[1, 2, 4, 3, 9],
            Some(ColdHeader::Safetensors(st_header.clone())),
        );
        for (a, b) in st_actual.0.iter().zip(&reference.0) {
            assert_tensor_close(a, b, "SafeTensors normalized cold cached session");
        }
        for (a, b) in actual.0.iter().zip(&reference.0) {
            assert_tensor_close(a, b, "GGUF/SafeTensors shared cached session");
        }
        // Float exp/log inversion can change low bits. Integer controls must be exact.
        for state in [&actual.1, &st_actual.1] {
            assert_eq!(state.layout(), reference.1.layout());
            for (a, b) in state.as_ref().iter().zip(reference.1.as_ref()) {
                let a: Vec<_> = RuntimeLayerState::retained_values(a).collect();
                let b: Vec<_> = RuntimeLayerState::retained_values(b).collect();
                assert_eq!(a.len(), b.len());
                for (a, b) in a.into_iter().zip(b) {
                    assert_tensor_close(a, b, "GGUF/SafeTensors retained state");
                    assert_eq!(a.exact_i32, b.exact_i32);
                }
            }
        }
    }
}

#[test]
fn gguf_scalar_view_target_matches_independently_decoded_weights() {
    let (dir, _, _) = fixtures(true);
    let a = super::registry::run(
        &dir.path().join("target.gguf"),
        LayerWeightResidency::FullyResident,
    );
    let b = super::registry::run(
        &dir.path().join("safetensors"),
        LayerWeightResidency::FullyResident,
    );
    for (a, b) in a.0.iter().zip(&b.0) {
        assert_tensor_close(a, b, "GGUF scalar source view session");
    }
}

#[test]
fn gguf_packed_target_retains_logical_geometry_companions_and_bounded_experts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.gguf");
    let mut fixture = Fixture::new();
    let mut block = half::f16::from_f32(0.125).to_le_bytes().to_vec();
    block.extend([0x98; 16]);
    for (name, rows) in [
        ("token_embd.weight", 16),
        ("blk.0.ssm_beta.weight", 6),
        ("blk.1.indexer.q_proj.weight", 16),
        ("blk.1.indexer.k_proj.weight", 8),
        ("blk.0.ffn_gate_exps.weight", 96),
        ("blk.0.ffn_up_exps.weight", 96),
    ] {
        fixture.replace_encoding(name, eredu_gguf::GgmlType::Q4_0, block.repeat(rows));
    }
    // Unlike concatenated gate/up recipes, down projections retain direct GGUF
    // sources. Exercise native byte blocks as well as unpacked affine companions.
    fixture.replace_encoding(
        "blk.0.ffn_down_exps.weight",
        eredu_gguf::GgmlType::IQ4NL,
        block.repeat(96),
    );
    fixture.replace_encoding(
        "blk.1.ffn_down_exps.weight",
        eredu_gguf::GgmlType::Q4_0,
        block.repeat(96),
    );
    fixture.write(&path);
    let header = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(path).unwrap()).unwrap();
    let source = super::super::qwen4_gguf::open_text_source(header.text_plan());
    let cold = super::gguf_admission::cold_plan(header.clone());
    let target = header.bind(source, limits()).unwrap();
    let plan = plan(&target);
    assert_eq!(cold.requirements(), plan.requirements());
    let parameters = plan.requirements().text().parameters();
    for (name, shape) in [
        ("model.embed_tokens.weight", vec![16, 32]),
        ("model.layers.0.linear_attn.in_proj_b.weight", vec![6, 32]),
        (
            "model.layers.1.self_attn.indexer.index_qk_proj.weight",
            vec![24, 32],
        ),
        ("model.layers.0.mlp.experts.gate_up_proj", vec![3, 64, 32]),
    ] {
        let p = parameters.iter().find(|p| p.name() == name).unwrap();
        assert_eq!(p.logical_shape(), shape);
        assert!(matches!(
            p.native_executable(),
            eredu_checkpoint::LinearFormat::Affine(_)
        ));
    }
    for name in [
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.gate_up_proj_biases",
    ] {
        let p = parameters.iter().find(|p| p.name() == name).unwrap();
        assert_eq!(p.role(), ReplicatedTextParameterRole::FormatCompanion);
    }
    let owner = target.expert(1, 2).unwrap();
    for (name, recipe) in owner.recipes() {
        assert_eq!(
            recipe.infer(owner.source().as_ref()).unwrap().shape[0],
            1,
            "{name}"
        );
        recipe.preflight_bounded(owner.source().as_ref()).unwrap();
    }
    let capabilities = cold::capabilities(plan.requirements(), None);
    let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        ),
        WeightResidency::default(),
    )
    .unwrap();
    let selected = cold
        .select(&request, &capabilities, None)
        .unwrap()
        .bind(target.artifact().clone(), None)
        .unwrap();
    selected
        .prepare::<NumericBackend, State>(&NumericContext::default())
        .unwrap();
}

#[test]
fn gguf_target_rows_and_ordinary_parameters_share_retained_reads() {
    let (_dir, target, _) = fixtures(false);
    let table = &target.tables()[&0];
    let source = target.artifact();
    let table_key = "per_layer_token_embd.weight";
    assert_eq!(table.rows.source_keys(), [table_key]);
    assert_eq!(
        table.rows.source_provenance(table_key).unwrap(),
        source.source_provenance(table_key).unwrap()
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let rows = table
        .rows
        .plan(
            &[1, 2, 1],
            eredu_checkpoint::rows::RowReadLimits {
                requests: 3,
                rows_per_read: 2,
            },
        )
        .unwrap();
    assert_eq!(rows.reads().len(), 1);
    let value = payload::recipe_value(
        rows.reads()[0].recipe(),
        rows.source(),
        &NumericContext::default(),
    )
    .unwrap();
    assert_eq!(value.shape, [2, table.lookup.dimensions]);
    let after_rows = source.source_diagnostics().unwrap();
    assert_eq!(after_rows.physical_reads, 1);
    assert_eq!(
        after_rows.physical_read_bytes,
        2 * table.lookup.dimensions as u64 * 4
    );
    assert_eq!(table.rows.source_diagnostics().unwrap(), after_rows);
    let ordinary = target.static_parameters();
    assert!(ordinary.source().source_metadata(table_key).is_err());
    let recipe = &ordinary.recipes()["model.embed_tokens.weight"];
    payload::recipe_value(
        recipe,
        ordinary.source().as_ref(),
        &NumericContext::default(),
    )
    .unwrap();
    let after_ordinary = source.source_diagnostics().unwrap();
    assert_eq!(after_ordinary.physical_reads, 2);
    assert_eq!(after_ordinary.cache_misses, 1);
    assert!(after_ordinary.cache_hits >= 1);
    assert_eq!(after_ordinary.currently_cached_shards, 1);
    assert_eq!(
        ordinary.source().source_diagnostics().unwrap(),
        after_ordinary
    );
    assert_eq!(table.rows.source_diagnostics().unwrap(), after_ordinary);
}
