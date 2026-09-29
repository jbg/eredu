//! Rank-local binding consumes retained prepared recipes for both published containers.
use super::*;
use eredu_architectures::qwen4_exp::prepared::{
    GgufTargetPlan, PreparedParameters, PreparedTarget, PreparedTensorTarget, SafetensorsTargetPlan,
};
use eredu_checkpoint::store::{SafetensorsWeightStore, SharedCheckpointSource};
use eredu_runtime::*;

type Runtime = LayerwiseRuntime<
    TargetModel<NumericBackend>,
    NumericBackend,
    State,
    ResidentUnitWindow<Unit<NumericBackend>>,
>;

pub(super) struct Bind<'a> {
    pub(super) ordinary: &'a PreparedParameters,
    pub(super) experts: Vec<PreparedParameters>,
    pub(super) artifact: &'a SharedCheckpointSource,
    pub(super) ctx: &'a NumericContext,
}
impl Bind<'_> {
    fn read(&self, owner: &PreparedParameters, name: &str) -> NumericTensor {
        let recipe = &owner.recipes()[name];
        for key in recipe.source_keys() {
            assert_eq!(
                owner.source().source_provenance(key).unwrap(),
                self.artifact.source_provenance(key).unwrap(),
                "rank-local recipes retain the admitted physical source"
            );
        }
        payload::recipe_value(recipe, owner.source().as_ref(), self.ctx).unwrap()
    }
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let payload = if self.ordinary.recipes().contains_key(name) {
            self.read(self.ordinary, name)
        } else {
            let pieces: Vec<_> = self.experts.iter().map(|e| self.read(e, name)).collect();
            NumericTensor::concatenate(&pieces, 0, self.ctx).unwrap()
        };
        // Checkpoint depthwise kernels have a singleton group axis.
        let payload = if name.ends_with("conv1d.weight") && payload.shape.len() == 3 {
            payload.reshape(&value.shape, self.ctx).unwrap()
        } else {
            payload
        };
        assert_eq!(
            payload.shape, value.shape,
            "prepared local parameter {name}"
        );
        *value = payload;
    }
}
pub(super) fn runtime(
    target: &PreparedTarget,
    local: &PreparedTensorTarget,
    parallel: bool,
    ctx: &NumericContext,
) -> Runtime {
    let mut model = if parallel {
        TargetModel::<NumericBackend>::new_tensor_parallel(
            local.source_bound_spec().clone(),
            local.partition().clone(),
            ctx,
        )
    } else {
        assert_eq!(local.partition().ranks(), 1);
        TargetModel::<NumericBackend>::new(local.source_bound_spec().clone(), ctx)
    }
    .unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model)
        .visit_parameters_mut(&mut Bind { ordinary: local.static_parameters(), experts: vec![], artifact: target.artifact(), ctx });
    let units = (0..target.spec().units.len())
        .map(|ordinal| {
            let mut unit = model.construct_unit(ordinal, ctx).unwrap();
            let experts = if matches!(target.spec().units[ordinal], UnitSpec::Decoder { .. }) {
                (0..target.spec().configuration().experts.count as usize)
                    .map(|expert| local.expert(ordinal, expert).unwrap())
                    .collect()
            } else {
                vec![]
            };
            unit.visit_parameters_mut(&mut Bind {
                ordinary: local.unit(ordinal).unwrap(),
                experts,
                artifact: target.artifact(),
                ctx,
            });
            unit
        })
        .collect();
    LayerwiseRuntime::new(model, ResidentUnitWindow::new(units))
}

fn run(target: &PreparedTarget, ranks: usize) -> Vec<(Vec<NumericTensor>, State)> {
    let before = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let local: Vec<_> = (0..ranks)
        .map(|rank| target.tensor_partition(rank, ranks).unwrap())
        .collect();
    assert_eq!(
        target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        before,
        "rank preparation is metadata-only"
    );
    let group = NumericParallelGroup::new(ranks);
    std::thread::scope(|scope| {
        let handles: Vec<_> = local
            .iter()
            .enumerate()
            .map(|(rank, local)| {
                let group = group.clone();
                scope.spawn(move || {
                    let ctx = NumericContext {
                        bind_checkpoint_values: true,
                        ..Default::default()
                    };
                    let parallel = NumericParallelContext::new(rank, group);
                    let mut runtime = runtime(target, local, true, &ctx);
                    let mut state = state(local.partition().local_spec());
                    let prepared = local
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
                        .unwrap();
                    let banks = if rank == 0 {
                        prepared
                            .entries()
                            .iter()
                            .map(|(id, e)| (id.clone(), super::super::row_bank::SourceRows::new(e)))
                            .collect()
                    } else {
                        BTreeMap::new()
                    };
                    let mut provider = ParameterProviders {
                        grouped: ResidentExpertProvider,
                        rows: prepared
                            .bind_tensor_parallel::<NumericBackend, _, _>(
                                banks, &parallel, &parallel, rank, 0,
                            )
                            .unwrap(),
                    };
                    let vocabulary = target.spec().configuration().vocabulary as u64;
                    let ids: Vec<_> = (0..2)
                        .flat_map(|lane| {
                            (0..13).map(move |t| ((t * 3 + lane * 5) as u64) % vocabulary)
                        })
                        .collect();
                    let mut outputs = vec![];
                    for (start, end) in [(0, 3), (3, 8), (8, 13)] {
                        let ids: Vec<_> = (0..2)
                            .flat_map(|lane| {
                                ids[lane * 13 + start..lane * 13 + end].iter().copied()
                            })
                            .collect();
                        outputs.push(
                            runtime
                                .forward_parallel_with_provider_and_observer(
                                    TargetInput {
                                        ids: Some(OriginalTokenIds::Host(&ids)),
                                        batch: 2,
                                        tokens: (end - start) as i32,
                                        embeddings: None,
                                        visible: None,
                                        rotary: None,
                                        position_delta: None,
                                    },
                                    &mut state,
                                    ExpertPass::Prefill,
                                    &mut provider,
                                    &parallel,
                                    &ctx,
                                    &mut Observe::default(),
                                )
                                .unwrap(),
                        );
                    }
                    let mut logits = vec![NumericTensor::concatenate(&outputs, 1, &ctx).unwrap()];
                    for step in 0..3 {
                        let ids = [
                            ((step + 3) as u64) % vocabulary,
                            ((step + 9) as u64) % vocabulary,
                        ];
                        logits.push(
                            runtime
                                .forward_parallel_with_provider_and_observer(
                                    TargetInput {
                                        ids: Some(OriginalTokenIds::Host(&ids)),
                                        batch: 2,
                                        tokens: 1,
                                        embeddings: None,
                                        visible: None,
                                        rotary: None,
                                        position_delta: None,
                                    },
                                    &mut state,
                                    ExpertPass::Decode,
                                    &mut provider,
                                    &parallel,
                                    &ctx,
                                    &mut Observe::default(),
                                )
                                .unwrap(),
                        );
                    }
                    (logits, state)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
fn compare(target: &PreparedTarget, expected: &(Vec<NumericTensor>, State), ranks: usize) {
    let ctx = NumericContext::default();
    for (rank, (logits, state)) in run(target, ranks).into_iter().enumerate() {
        for (actual, expected) in logits.iter().zip(&expected.0) {
            assert_tensor_close(actual, expected, "prepared TP cached logits");
        }
        super::tensor_parallel::compare_state(
            &state,
            &expected.1,
            target.spec(),
            rank,
            ranks,
            &ctx,
        );
    }
}

#[test]
fn prepared_tp2_safetensors_and_published_gguf_bind_retained_rank_sources() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::Fixture::new()
        .prepare(directory.path())
        .unwrap();
    let baseline = run(&fixture.safetensors, 1).remove(0);
    assert!(baseline.0[0].data.iter().any(|x| x.abs() > 0.01));
    compare(&fixture.safetensors, &baseline, 2);
    compare(&fixture.gguf, &baseline, 2);
}

pub(super) fn small_safetensors(directory: &std::path::Path, fp8: bool) -> PreparedTarget {
    let mut config = configuration();
    config.attention.heads = 4;
    config.attention.kv_heads = 2;
    config.recurrent.key_heads = 4;
    config.recurrent.value_heads = 8;
    config.experts.intermediate = if fp8 { 256 } else { 4 };
    let spec = specification_for(config.clone());
    let ctx = NumericContext::default();
    let mut model = TargetModel::<NumericBackend>::new(bind_spec(spec.clone()), &ctx).unwrap();
    let mut parameters = Parameters::default();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut parameters);
    for ordinal in 0..spec.units.len() {
        model
            .construct_unit(ordinal, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let mut values = table_values();
    for (name, tensor) in parameters.0 {
        let mut shape: Vec<_> = tensor.shape.iter().map(|n| *n as usize).collect();
        if name.ends_with("conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        values.push((
            name,
            safetensors::Dtype::F32,
            shape,
            tensor.data.into_iter().flat_map(f32::to_le_bytes).collect(),
        ));
    }
    let excluded: Vec<_> = values
        .iter()
        .filter(|(name, _, _, _)| !name.contains(".mlp.experts."))
        .map(|(name, _, _, _)| name.strip_suffix(".weight").unwrap_or(name).to_owned())
        .collect();
    let encoding_json = if fp8 {
        let mut scales = Vec::new();
        for (name, dtype, shape, bytes) in &mut values {
            if name.contains(".mlp.experts.") || name.contains(".ngram_embedding.shard_") {
                *dtype = safetensors::Dtype::F8_E4M3;
                *bytes = (0..shape.iter().product())
                    .map(|i: usize| 0x30 + (i % 8) as u8)
                    .collect();
                if name.contains(".mlp.experts.") {
                    let shape = vec![shape[0], shape[1].div_ceil(128), shape[2].div_ceil(128)];
                    let bytes = (0..shape.iter().product())
                        .flat_map(|i: usize| (1. + i as f32 / 16.).to_le_bytes())
                        .collect();
                    scales.push((
                        format!("{name}_scale_inv"),
                        safetensors::Dtype::F32,
                        shape,
                        bytes,
                    ));
                }
            }
        }
        values.extend(scales);
        values.push((
            "model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale".into(),
            safetensors::Dtype::F32,
            vec![1],
            0.5f32.to_le_bytes().to_vec(),
        ));
        serde_json::json!({"quantization_config": {
            "quant_method":"fp8", "activation_scheme":"dynamic", "weight_block_size":[128,128],
            "weight_per_tensor":false, "act_per_tensor":false,
            "modules_to_not_convert":excluded, "modules_to_convert":["ple.ple_embedding.ngram_embedding"]
        }})
    } else {
        serde_json::json!({})
    };
    safetensors::tensor::serialize_to_file(
        values.iter().map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.join("model.safetensors"),
    )
    .unwrap();
    let source: SharedCheckpointSource = Arc::new(SafetensorsWeightStore::open(directory).unwrap());
    let header = SafetensorsTargetPlan::prepare(
        source.as_ref(),
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &encoding_json,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    header.bind(source, spec.limits).unwrap()
}

#[test]
fn prepared_tp4_safetensors_preserves_replicated_kv_and_exact_controls() {
    let directory = tempfile::tempdir().unwrap();
    let target = small_safetensors(directory.path(), false);
    let baseline = run(&target, 1).remove(0);
    compare(&target, &baseline, 4);
}

#[test]
fn prepared_tp2_encoded_gguf_reads_only_owned_projection_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("encoded.gguf");
    let mut fixture = eredu_evaluation::qwen4_exp::Fixture::new();
    let line = include_str!("../../../../eredu-gguf/tests/fixtures/scalar-block-rows.txt")
        .lines()
        .find(|line| line.starts_with("8|"))
        .unwrap();
    let hex = line.split('|').nth(1).unwrap();
    let blocks: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect();
    let rows: Vec<_> = (0..64)
        .flat_map(|row| blocks[(row % 3) * 34..(row % 3 + 1) * 34].iter().copied())
        .collect();
    fixture.replace_encoding(
        "blk.1.attn_q.weight",
        eredu_gguf::GgmlType::Q8_0,
        rows.clone(),
    );
    fixture.write(&path);
    let header = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(&path).unwrap()).unwrap();
    let source = super::super::qwen4_gguf::open_text_source(header.text_plan());
    let target = header
        .bind(source.clone(), eredu_evaluation::qwen4_exp::limits())
        .unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let ordinal = target
        .spec()
        .units
        .iter()
        .position(|unit| matches!(unit, UnitSpec::Decoder { layer: 1, .. }))
        .unwrap();
    let name = "model.layers.1.self_attn.q_proj.weight";
    for rank in 0..2 {
        let local = target.tensor_partition(rank, 2).unwrap();
        let owner = local.unit(ordinal).unwrap();
        let recipe = &owner.recipes()[name];
        let metadata = recipe.infer(owner.source().as_ref()).unwrap();
        assert_eq!(metadata.shape, [32, 34]);
        assert_eq!(metadata.byte_len, 32 * 34);
        for key in recipe.source_keys() {
            assert_eq!(
                owner.source().source_provenance(key).unwrap(),
                source.source_provenance(key).unwrap()
            );
        }
        let before = source.source_diagnostics().unwrap();
        let read = recipe
            .prepare_encoded_read(owner.source().as_ref())
            .unwrap()
            .unwrap();
        let mut actual = vec![0; metadata.byte_len as usize];
        read.read_into(&mut actual).unwrap();
        assert_eq!(actual, rows[rank * 32 * 34..(rank + 1) * 32 * 34]);
        let after = source.source_diagnostics().unwrap();
        assert_eq!(
            after.physical_read_bytes - before.physical_read_bytes,
            32 * 34,
            "only the rank-owned encoded rows are read"
        );
    }
}

// Execute only byte-preserving recipe operations, independently of native binding.
// A column slice can read one complete expert before narrowing its local output.
fn encoded_recipe_bytes(
    recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    maximum_source_bytes: usize,
) -> Vec<u8> {
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe as R,
        store::{EncodedTensorLease, ReadPolicy, TensorReadRequest, TensorSelection as S},
    };
    match recipe {
        R::Source { key, selection } => {
            let lease = source
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert!(lease.bounded_read_proof().physically_bounded);
            let bytes = lease.encoded_bytes().unwrap();
            assert!(
                bytes.len() <= maximum_source_bytes,
                "a source read may not expand to the whole expert bank"
            );
            bytes.to_vec()
        }
        R::View { input, .. } | R::Reshape { input, .. } => {
            encoded_recipe_bytes(input, source, maximum_source_bytes)
        }
        R::Select { input, selection } => {
            let metadata = input.infer(source).unwrap();
            let bytes = encoded_recipe_bytes(input, source, maximum_source_bytes);
            let element = metadata.byte_len as usize / metadata.shape.iter().product::<usize>();
            match selection {
                S::Full => bytes,
                S::Contiguous {
                    offset_elements,
                    shape,
                } => bytes[offset_elements * element
                    ..(offset_elements + shape.iter().product::<usize>()) * element]
                    .to_vec(),
                S::Range { axis, start, end } => {
                    let outer = metadata.shape[..*axis].iter().product::<usize>();
                    let stride = metadata.shape[axis + 1..].iter().product::<usize>() * element;
                    (0..outer)
                        .flat_map(|o| {
                            bytes[(o * metadata.shape[*axis] + start) * stride
                                ..(o * metadata.shape[*axis] + end) * stride]
                                .iter()
                                .copied()
                        })
                        .collect()
                }
                S::Indices { axis, indices } => {
                    let outer = metadata.shape[..*axis].iter().product::<usize>();
                    let stride = metadata.shape[axis + 1..].iter().product::<usize>() * element;
                    let mut output = Vec::new();
                    for o in 0..outer {
                        for i in indices {
                            output.extend_from_slice(
                                &bytes[(o * metadata.shape[*axis] + i) * stride
                                    ..(o * metadata.shape[*axis] + i + 1) * stride],
                            );
                        }
                    }
                    output
                }
            }
        }
        R::Concatenate { axis, inputs } => {
            let metadata = recipe.infer(source).unwrap();
            let outer = metadata.shape[..*axis].iter().product::<usize>();
            let parts: Vec<_> = inputs
                .iter()
                .map(|input| encoded_recipe_bytes(input, source, maximum_source_bytes))
                .collect();
            (0..outer)
                .flat_map(|o| {
                    parts.iter().flat_map(move |part| {
                        part[o * part.len() / outer..(o + 1) * part.len() / outer]
                            .iter()
                            .copied()
                    })
                })
                .collect()
        }
        other => panic!("unexpected encoded fixture recipe {other:?}"),
    }
}

#[test]
fn prepared_tp2_fp8_expert_and_inverse_scales_read_exact_rank_members() {
    let directory = tempfile::tempdir().unwrap();
    let target = small_safetensors(directory.path(), true);
    let source = target.artifact();
    for rank in 0..2 {
        let before_partition = source.source_diagnostics().unwrap().physical_reads;
        let local = target.tensor_partition(rank, 2).unwrap();
        let expert = local.expert(0, 1).unwrap();
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            before_partition
        );
        for (suffix, shape, expected) in [
            (
                "gate_up_proj",
                vec![1, 256, 2],
                [
                    128 * rank..128 * (rank + 1),
                    256 + 128 * rank..256 + 128 * (rank + 1),
                ]
                .into_iter()
                .flat_map(|range| {
                    range.flat_map(|row| {
                        (0..2).map(move |column| 0x30 + ((512 * 2 + row * 2 + column) % 8) as u8)
                    })
                })
                .collect::<Vec<_>>(),
            ),
            (
                "gate_up_proj_scales",
                vec![1, 2, 1],
                [4 + rank, 6 + rank]
                    .into_iter()
                    .flat_map(|i| (1. + i as f32 / 16.).to_le_bytes())
                    .collect(),
            ),
            (
                "down_proj",
                vec![1, 2, 128],
                (0..2)
                    .flat_map(|row| {
                        (rank * 128..(rank + 1) * 128)
                            .map(move |column| 0x30 + ((2 * 256 + row * 256 + column) % 8) as u8)
                    })
                    .collect(),
            ),
            (
                "down_proj_scales",
                vec![1, 1, 1],
                (1. + (2 + rank) as f32 / 16.).to_le_bytes().to_vec(),
            ),
        ] {
            let name = format!("model.layers.0.mlp.experts.{suffix}");
            let recipe = &expert.recipes()[&name];
            let metadata = recipe.infer(expert.source().as_ref()).unwrap();
            assert_eq!(metadata.shape, shape, "{name}");
            let before = source.source_diagnostics().unwrap();
            let actual = encoded_recipe_bytes(recipe, expert.source().as_ref(), expected.len() * 2);
            assert_eq!(actual, expected, "rank{rank} {name}");
            let after = source.source_diagnostics().unwrap();
            assert!(after.physical_read_bytes > before.physical_read_bytes);
            assert_eq!(actual.len(), metadata.byte_len as usize);
        }
    }
}

#[test]
fn prepared_row_binding_rejects_wrong_native_rank_and_nonowner_banks() {
    let directory = tempfile::tempdir().unwrap();
    let target = small_safetensors(directory.path(), false);
    let rows = target
        .row_lookups(
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 65536,
                output_bytes: 32768,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let before = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let owner = NumericParallelContext::new(0, NumericParallelGroup::new(2));
    let peer = NumericParallelContext::new(1, NumericParallelGroup::new(2));
    let banks = || {
        rows.entries()
            .iter()
            .map(|(id, entry)| (id.clone(), super::super::row_bank::SourceRows::new(entry)))
            .collect()
    };
    assert!(matches!(
        rows.bind_tensor_parallel::<NumericBackend, super::super::row_bank::SourceRows, _>(
            BTreeMap::new(),
            &owner,
            &owner,
            1,
            0
        ),
        Err(RowLookupError::Geometry)
    ));
    assert!(matches!(
        rows.bind_tensor_parallel::<NumericBackend, super::super::row_bank::SourceRows, _>(
            BTreeMap::new(),
            &owner,
            &owner,
            0,
            0
        ),
        Err(RowLookupError::Missing(_))
    ));
    assert!(matches!(
        rows.bind_tensor_parallel::<NumericBackend, _, _>(banks(), &peer, &peer, 1, 0),
        Err(RowLookupError::Specification(_))
    ));
    assert!(rows
        .bind_tensor_parallel::<NumericBackend, super::super::row_bank::SourceRows, _>(
            BTreeMap::new(),
            &peer,
            &peer,
            1,
            0
        )
        .is_ok());
    assert!(rows
        .bind_tensor_parallel::<NumericBackend, _, _>(banks(), &owner, &owner, 0, 0)
        .is_ok());
    assert_eq!(owner.next_sequence.load(Ordering::Relaxed), 0);
    assert_eq!(peer.next_sequence.load(Ordering::Relaxed), 0);
    assert_eq!(
        target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        before
    );
}
