//! Published split projector recipes and an independent PyTorch vision oracle.
use super::*;
use eredu_architectures::qwen::vision::{
    VisionBlock, VisionInput, VisionMode, VisionStatic, VisionTower,
};
use eredu_architectures::qwen4_exp::{
    config::MediaTokens,
    prepared::{
        GgufVisionPlan, PreparationError, PreparedParameters, PreparedTarget, PreparedVision,
        VisionPlan,
    },
};
use eredu_gguf::{GgmlType, MetadataArray, MetadataValue as V, TensorInput, Writer};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("../../fixtures/qwen4/vision-reference.json")).unwrap()
}
fn media() -> MediaTokens {
    MediaTokens {
        image: 12,
        video: 13,
        start: 14,
        end: 15,
    }
}
fn values() -> Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> {
    fixture()["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, shape)| {
            let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
            let name = format!("model.visual.{name}");
            let seed = name.bytes().map(usize::from).sum::<usize>() % 17;
            let data = (0..shape.iter().product())
                .flat_map(|i| {
                    let v = ((i * 7 + seed) % 37) as f32 / 256. - 18. / 256.
                        + if name.contains("norm") && name.ends_with("weight") {
                            1.
                        } else {
                            0.
                        };
                    v.to_le_bytes()
                })
                .collect();
            (name, safetensors::Dtype::F32, shape, data)
        })
        .collect()
}
fn physical(name: &str) -> String {
    name.strip_prefix("model.visual.")
        .unwrap()
        .replace("pos_embed", "v.position_embd")
        .replace("patch_embed.proj", "v.patch_embd")
        .replace("merger.norm", "v.post_ln")
        .replace("merger.linear_fc1", "mm.0")
        .replace("merger.linear_fc2", "mm.2")
        .replace("blocks.", "v.blk.")
        .replace(".norm1.", ".ln1.")
        .replace(".norm2.", ".ln2.")
        .replace(".attn.qkv.", ".attn_qkv.")
        .replace(".attn.proj.", ".attn_out.")
        .replace(".mlp.linear_fc1.", ".ffn_up.")
        .replace(".mlp.linear_fc2.", ".ffn_down.")
}
fn write(path: &std::path::Path, omit_patch: bool, quantized: bool, output: u32, mixed: bool) {
    let mut tensors = Vec::new();
    for (name, _, shape, data) in values() {
        let name = physical(&name);
        if name == "v.patch_embd.weight" {
            for t in 0..if omit_patch { 1 } else { 2 } {
                let bytes: Vec<u8> = data
                    .chunks_exact(32)
                    .flat_map(|pair| pair[t * 16..t * 16 + 16].iter().copied())
                    .collect();
                let (ty, bytes) = if mixed && t == 1 {
                    (
                        GgmlType::F16,
                        bytes
                            .chunks_exact(4)
                            .flat_map(|v| {
                                half::f16::from_f32(f32::from_le_bytes(v.try_into().unwrap()))
                                    .to_le_bytes()
                            })
                            .collect(),
                    )
                } else {
                    (GgmlType::F32, bytes)
                };
                tensors.push((
                    if t == 0 {
                        name.clone()
                    } else {
                        format!("{name}.1")
                    },
                    vec![8, 3, 2, 2],
                    bytes,
                    ty,
                ));
            }
        } else if quantized && name == "mm.2.weight" {
            let mut block = half::f16::from_f32(0.125).to_le_bytes().to_vec();
            block.extend([0x98; 16]);
            tensors.push((name, shape, block.repeat(32), GgmlType::Q4_0));
        } else {
            tensors.push((name, shape, data, GgmlType::F32));
        }
    }
    let mut metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("clip".into())),
        (
            "clip.projector_type".into(),
            V::String("qwen3vl_merger".into()),
        ),
        ("clip.use_gelu".into(), V::Bool(true)),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            V::Float32(1e-6),
        ),
        (
            "clip.vision.is_deepstack_layers".into(),
            V::Array(MetadataArray::Bool(vec![false, false])),
        ),
    ]);
    for (name, n) in [
        ("embedding_length", 8),
        ("feed_forward_length", 12),
        ("attention.head_count", 2),
        ("block_count", 2),
        ("patch_size", 2),
        ("spatial_merge_size", 2),
        ("projection_dim", output),
    ] {
        metadata.insert(format!("clip.vision.{name}"), V::Uint32(n));
    }
    metadata.insert(
        "clip.vision.image_mean".into(),
        V::Array(MetadataArray::Float32(vec![0.; 3])),
    );
    metadata.insert(
        "clip.vision.image_std".into(),
        V::Array(MetadataArray::Float32(vec![1.; 3])),
    );
    metadata.insert("clip.vision.image_min_pixels".into(), V::Uint32(16));
    metadata.insert("clip.vision.image_max_pixels".into(), V::Uint32(64));
    let dims: Vec<Vec<u64>> = tensors
        .iter()
        .map(|(_, s, _, _)| s.iter().rev().map(|n| *n as u64).collect())
        .collect();
    let inputs: Vec<_> = tensors
        .iter()
        .zip(&dims)
        .map(|((name, _, data, ty), dimensions)| TensorInput {
            name,
            dimensions,
            data,
            ggml_type: *ty,
        })
        .collect();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}
fn gguf_vision_source(plan: &VisionPlan) -> SharedCheckpointSource {
    let source = plan.gguf_source().unwrap();
    Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .add_resolved_checkpoint(
                source.checkpoint().clone(),
                source.resolution(),
                source.mapping(),
            )
            .unwrap()
            .build()
            .unwrap(),
    )
}
fn gguf_vision(
    target: &PreparedTarget,
    checkpoint: &eredu_gguf::Checkpoint,
    media: MediaTokens,
) -> Result<PreparedVision, PreparationError> {
    let plan = GgufVisionPlan::prepare(target.spec().configuration(), checkpoint)
        .and_then(|plan| plan.bind_media_tokens(media, None))?;
    let source = gguf_vision_source(&plan);
    plan.bind(source)
}
struct Bind<'a>(&'a PreparedParameters, &'a NumericContext);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let recipe = &self.0.recipes()[meta.id.as_str()];
        *value =
            super::super::payload::recipe_value(recipe, self.0.source().as_ref(), self.1).unwrap();
    }
}

#[test]
fn gguf_vision_projector_matches_independent_image_video_oracle_and_streamed_blocks() {
    let (dir, target, text_st) = super::gguf::fixtures(false);
    let path = dir.path().join("vision.gguf");
    write(&path, false, false, 32, false);
    let prepared = gguf_vision(
        &target,
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
        media(),
    )
    .unwrap();
    assert_eq!(prepared.config().mode, VisionMode::DeepStack);
    assert_eq!(
        prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        0
    );
    assert!(prepared.block(2).is_err());
    assert!(prepared
        .block(0)
        .unwrap()
        .source()
        .source_metadata("model.visual.blocks.1.norm1.weight")
        .is_err());
    assert!(prepared
        .static_parameters()
        .source()
        .source_metadata("model.visual.blocks.0.norm1.weight")
        .is_err());
    let (_st_dir, vision_source) = transforms::fixture(&values());
    let mut config = eredu_architectures::qwen4_exp::config::Config::from_gguf(
        &eredu_gguf::Checkpoint::open(dir.path().join("target.gguf")).unwrap(),
    )
    .unwrap();
    config.ngram.source = eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        vocabulary_base: 5,
        vocabulary_alignment: 1,
        shards: 1,
        seed: 1,
    };
    config.vision = Some(prepared.config().clone());
    config.media = Some(media());
    let composite = Arc::new(
        eredu_checkpoint::store::CompositeCheckpointSource::new([
            text_st.artifact().clone(),
            vision_source,
        ])
        .unwrap(),
    );
    let st = PreparedTarget::safetensors(
        composite,
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
        target.spec().limits,
    )
    .unwrap()
    .vision()
    .unwrap();
    let ctx = NumericContext::default();
    let mut cold_block =
        VisionBlock::<NumericBackend>::new_with_root(prepared.config(), "model.visual", 0, &ctx)
            .unwrap();
    let before = prepared
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_read_bytes;
    cold_block.visit_parameters_mut(&mut Bind(prepared.block(0).unwrap(), &ctx));
    let bytes = prepared
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_read_bytes
        - before;
    let bound: u64 = prepared
        .block(0)
        .unwrap()
        .recipes()
        .values()
        .map(|r| {
            r.infer(prepared.block(0).unwrap().source().as_ref())
                .unwrap()
                .byte_len
        })
        .sum();
    assert_eq!(
        bytes, bound,
        "cold block acquisition reads exactly its parameters"
    );
    drop(cold_block);
    for case in fixture()["cases"].as_array().unwrap() {
        let grid: Vec<(i32, i32, i32)> = serde_json::from_value(case["grid"].clone()).unwrap();
        let count: i32 = grid.iter().map(|(t, h, w)| t * h * w).sum();
        let pixels = NumericTensor::new(
            [count, 24],
            (0..count * 24)
                .map(|i| ((i * 11 % 53) as f32 - 26.) / 64.)
                .collect(),
        );
        let expected = NumericTensor::new(
            serde_json::from_value::<Vec<i32>>(case["shape"].clone()).unwrap(),
            serde_json::from_value(case["values"].clone()).unwrap(),
        );
        for owner in [&prepared, &st] {
            let mut tower = VisionTower::<NumericBackend>::new_with_root(
                owner.config().clone(),
                "model.visual",
                &ctx,
            )
            .unwrap();
            tower
                .static_modules
                .visit_parameters_mut(&mut Bind(owner.static_parameters(), &ctx));
            for (i, block) in tower.blocks.iter_mut().enumerate() {
                block.visit_parameters_mut(&mut Bind(owner.block(i).unwrap(), &ctx));
            }
            let full = tower
                .forward(
                    VisionInput {
                        pixels: &pixels,
                        grid: &grid,
                    },
                    &ctx,
                )
                .unwrap();
            assert!(full.deepstack_features.is_empty());
            assert_tensor_close(&full.embeddings, &expected, "pinned vision oracle");
            let mut statics = VisionStatic::<NumericBackend>::new_with_root(
                owner.config().clone(),
                "model.visual",
                &ctx,
            )
            .unwrap();
            statics.visit_parameters_mut(&mut Bind(owner.static_parameters(), &ctx));
            let (mut hidden, mut state) = statics
                .begin(
                    VisionInput {
                        pixels: &pixels,
                        grid: &grid,
                    },
                    &ctx,
                )
                .unwrap();
            for i in 0..owner.config().layer_count() {
                let mut block = VisionBlock::<NumericBackend>::new_with_root(
                    owner.config(),
                    "model.visual",
                    i,
                    &ctx,
                )
                .unwrap();
                let before = owner
                    .artifact()
                    .source_diagnostics()
                    .unwrap()
                    .physical_read_bytes;
                block.visit_parameters_mut(&mut Bind(owner.block(i).unwrap(), &ctx));
                let after = owner
                    .artifact()
                    .source_diagnostics()
                    .unwrap()
                    .physical_read_bytes;
                let size: u64 = owner
                    .block(i)
                    .unwrap()
                    .recipes()
                    .values()
                    .map(|r| {
                        r.infer(owner.block(i).unwrap().source().as_ref())
                            .unwrap()
                            .byte_len
                    })
                    .sum();
                assert!(
                    after - before <= size,
                    "one block may not read another owner"
                );
                hidden = statics
                    .forward_block(&mut block, i, &hidden, &mut state, &ctx)
                    .unwrap();
            }
            let streamed = statics.finish(&hidden, &mut state, &ctx).unwrap();
            assert_tensor_exact(
                &streamed.embeddings,
                &full.embeddings,
                "streamed vision blocks",
            );
        }
    }
    write(&path, false, false, 32, true);
    let mixed = gguf_vision(
        &target,
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
        media(),
    )
    .unwrap();
    let weight = "model.visual.patch_embed.proj.weight";
    let a = super::super::payload::recipe_value(
        &mixed.static_parameters().recipes()[weight],
        mixed.static_parameters().source().as_ref(),
        &ctx,
    )
    .unwrap();
    let b = super::super::payload::recipe_value(
        &st.static_parameters().recipes()[weight],
        st.static_parameters().source().as_ref(),
        &ctx,
    )
    .unwrap();
    assert_tensor_exact(&a, &b, "mixed temporal source dtypes");
}

#[test]
fn gguf_vision_projector_rejects_missing_and_mismatched_sources_and_retains_packed_formats() {
    let (dir, target, _) = super::gguf::fixtures(false);
    let path = dir.path().join("vision.gguf");
    assert!(matches!(
        target.vision(),
        Err(PreparationError::MissingVision)
    ));
    write(&path, true, false, 32, false);
    assert!(gguf_vision(
        &target,
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
        media()
    )
    .is_err());
    write(&path, false, false, 31, false);
    assert!(matches!(
        gguf_vision(
            &target,
            &eredu_gguf::Checkpoint::open(&path).unwrap(),
            media()
        ),
        Err(PreparationError::VisionMismatch {
            field: "output width"
        })
    ));
    write(&path, false, true, 32, false);
    let cp = eredu_gguf::Checkpoint::open(&path).unwrap();
    let mut bad = media();
    bad.video = bad.image;
    assert!(matches!(
        gguf_vision(&target, &cp, bad),
        Err(PreparationError::VisionMismatch {
            field: "media token IDs"
        })
    ));
    let prepared = gguf_vision(&target, &cp, media()).unwrap();
    assert!(matches!(
        prepared.config().linear_format("merger.linear_fc2.weight"),
        eredu_checkpoint::LinearFormat::Affine(_)
    ));
    for suffix in ["weight", "scales", "biases"] {
        let name = format!("model.visual.merger.linear_fc2.{suffix}");
        prepared.static_parameters().recipes()[&name]
            .preflight_bounded(prepared.static_parameters().source().as_ref())
            .unwrap();
    }
    assert_eq!(
        prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        0
    );
}

#[path = "vision/ingress.rs"]
mod ingress;

#[path = "vision/header.rs"]
mod header;

#[path = "vision/registry.rs"]
mod registry;
