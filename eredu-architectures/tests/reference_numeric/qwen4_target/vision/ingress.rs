//! Token provenance, real image/video projection, rotary oracle and target caching.
use super::*;
#[path = "attention_parallel.rs"]
mod attention_parallel;
#[path = "merger_parallel.rs"]
mod merger_parallel;
#[path = "storage.rs"]
mod storage;
use eredu_architectures::qwen4_exp::media::{MediaIngress, MediaInputError};
use eredu_core::residency::ResidencyPolicy;
use eredu_core::{checkpoint::TensorDtype, InputMetadataKey as K, InputModality as M};
use eredu_runtime::{
    PreparedInputInspector, PreparedInputPart, PreparedInputPayload as P, PreparedModelInput,
    RowLookupLimits,
};

struct Inspector(std::cell::Cell<usize>);
impl PreparedInputInspector<NumericTensor> for Inspector {
    fn identity(
        &self,
        t: &NumericTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::InputTensorIdentity::new(
            match t.element_type() {
                Some(eredu_nn::TensorElementType::I32) => TensorDtype::I32,
                Some(eredu_nn::TensorElementType::U32) => TensorDtype::U32,
                _ => TensorDtype::F32,
            },
            t.shape().iter().map(|n| *n as usize).collect(),
        )
    }
    fn i32_values(&self, t: &NumericTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        self.0.set(self.0.get() + 1);
        t.to_i32_vec(&NumericContext::default())
            .map_err(|e| eredu_core::CapabilityError::Observation(e.to_string()))
    }
    fn bool_values(&self, _: &NumericTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        unreachable!()
    }
}
fn input(
    parts: Vec<PreparedInputPart<NumericTensor>>,
    i: &Inspector,
) -> PreparedModelInput<NumericTensor> {
    PreparedModelInput::new(parts, |t| i.identity(t)).unwrap()
}
fn text(ids: &[i32]) -> PreparedInputPart<NumericTensor> {
    PreparedInputPart::new(
        M::Text,
        P::TokenIds(
            NumericTensor::from_i32_slice(ids, &[1, ids.len() as i32], &NumericContext::default())
                .unwrap()
                .with_dtype(TensorDtype::U32),
        ),
        [],
    )
    .unwrap()
}
fn grid(time: i32, height: i32, width: i32) -> NumericTensor {
    NumericTensor::from_i32_slice(&[time, height, width], &[1, 3], &NumericContext::default())
        .unwrap()
}
fn pixels(count: i32) -> NumericTensor {
    NumericTensor::new(
        [count, 24],
        (0..count * 24)
            .map(|i| ((i * 11 % 53) as f32 - 26.) / 64.)
            .collect(),
    )
}
fn setup() -> (tempfile::TempDir, PreparedTarget, MediaIngress) {
    let (dir, target, _) = super::super::gguf::fixtures(false);
    let path = dir.path().join("vision.gguf");
    write(&path, false, false, 32, false);
    let vision = gguf_vision(
        &target,
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
        media(),
    )
    .unwrap();
    let ingress = target.media_ingress(vision).unwrap();
    (dir, target, ingress)
}
fn tower(ingress: &MediaIngress, ctx: &NumericContext) -> VisionTower<NumericBackend> {
    let prepared = ingress.vision();
    let mut tower =
        VisionTower::new_with_root(prepared.config().clone(), "model.visual", ctx).unwrap();
    tower
        .static_modules
        .visit_parameters_mut(&mut Bind(prepared.static_parameters(), ctx));
    for (i, b) in tower.blocks.iter_mut().enumerate() {
        b.visit_parameters_mut(&mut Bind(prepared.block(i).unwrap(), ctx));
    }
    tower
}
fn prompt(i: &Inspector) -> PreparedModelInput<NumericTensor> {
    // Each video frame is separated by timestamp/control text, as in the processor.
    input(
        vec![
            text(&[3, 14]),
            PreparedInputPart::new(
                M::Image,
                P::Tensor(pixels(16)),
                [(K::PatchGrid, grid(1, 4, 4))],
            )
            .unwrap(),
            text(&[15, 4, 14]),
            PreparedInputPart::new(
                M::Video,
                P::Tensor(pixels(8)),
                [(K::PatchGrid, grid(1, 4, 2))],
            )
            .unwrap(),
            text(&[15, 5, 14]),
            PreparedInputPart::new(
                M::Video,
                P::Tensor(pixels(8)),
                [(K::PatchGrid, grid(1, 4, 2))],
            )
            .unwrap(),
            text(&[15, 6]),
        ],
        i,
    )
}
fn embedding(ids: &NumericTensor) -> Result<NumericTensor, Error> {
    Ok(NumericTensor::new(
        [1, ids.dim(1), 32],
        ids.data
            .iter()
            .flat_map(|id| (0..32).map(move |d| (*id + d as f32) / 64.))
            .collect(),
    ))
}
#[test]
fn original_ids_and_rotary_survive_projection_wire_roundtrip_and_arbitrary_slices() {
    let (_dir, target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let before = target
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let admitted = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap();
    let inspected = inspector.0.get();
    let prepared = admitted.prepare(&ctx).unwrap();
    let replay = admitted.prepare(&ctx).unwrap();

    assert_eq!(
        inspector.0.get(),
        inspected,
        "native preparation must reuse host proof"
    );
    assert_tensor_exact(prepared.token_ids(), replay.token_ids(), "admission replay");
    assert_eq!(
        target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        before
    );
    assert_eq!(prepared.admission().decoder_positions(), 18);
    let mut tower = tower(&ingress, &ctx);
    let projected = prepared
        .with_vision_input(|v| tower.forward(v.unwrap(), &ctx))
        .unwrap()
        .embeddings;
    let result = prepared
        .assemble(
            0..prepared.token_ids().dim(1),
            Some(&projected),
            embedding,
            &ctx,
        )
        .unwrap();
    let expected_ids = [
        3, 14, 12, 12, 12, 12, 15, 4, 14, 13, 13, 15, 5, 14, 13, 13, 15, 6,
    ];
    assert_eq!(result.token_ids().to_i32_vec(&ctx).unwrap(), expected_ids);
    assert_eq!(
        result.token_ids().element_type(),
        Some(eredu_nn::TensorElementType::I32)
    );
    assert_eq!(result.rotary_delta(), -2);
    assert_eq!(admitted.token_ids(), expected_ids);
    assert_eq!(admitted.rotary_delta(), -2);
    // Manual pinned-reference axes; section width one selects T,H,W once each.
    let axes = [
        [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 9, 10, 11, 12, 12, 14, 15],
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11, 12, 12, 14, 15],
    ];
    assert_eq!(admitted.position_ids(), &axes.map(|axis| axis.to_vec()));
    result.with_target_input(|input| {
        let RotaryPosition::Embeddings { cosine, sine } = input.rotary.unwrap() else {
            panic!()
        };
        for token in 0..18 {
            for d in 0..6 {
                let lane = d % 3;
                let angle = axes[lane][token] as f32 / 10000f32.powf((2 * lane) as f32 / 6.);
                assert!((cosine.data[token * 6 + d] - angle.cos()).abs() < 1e-6);
                assert!((sine.data[token * 6 + d] - angle.sin()).abs() < 1e-6);
            }
        }
    });
    for range in [0..1, 1..4, 4..9, 9..18] {
        let sliced = result.slice(range.clone(), &ctx).unwrap();
        assert_eq!(
            sliced.token_ids().to_i32_vec(&ctx).unwrap(),
            expected_ids[range.start as usize..range.end as usize]
        );
    }
    for range in (0..18)
        .map(|i| i..i + 1)
        .chain([0..5, 3..10, 8..16, 11..18])
    {
        let mut looked_up = Vec::new();
        let span = prepared
            .assemble(
                range.clone(),
                Some(&projected),
                |ids| {
                    looked_up.extend(ids.to_i32_vec(&ctx)?);
                    embedding(ids)
                },
                &ctx,
            )
            .unwrap();
        let expected_text = expected_ids[range.start as usize..range.end as usize]
            .iter()
            .copied()
            .filter(|id| *id != 12 && *id != 13)
            .collect::<Vec<_>>();
        assert_eq!(looked_up, expected_text, "embed only text in {range:?}");
        let expected = result.slice(range, &ctx).unwrap();
        span.with_target_input(|actual| {
            expected.with_target_input(|expected| {
                assert_tensor_exact(
                    actual.embeddings.unwrap(),
                    expected.embeddings.unwrap(),
                    "span embeddings",
                );
                let RotaryPosition::Embeddings { cosine: a, sine: b } = actual.rotary.unwrap()
                else {
                    panic!()
                };
                let RotaryPosition::Embeddings { cosine: c, sine: d } = expected.rotary.unwrap()
                else {
                    panic!()
                };
                assert_tensor_exact(a, c, "span cosine");
                assert_tensor_exact(b, d, "span sine");
            })
        });
        assert_tensor_exact(span.token_ids(), expected.token_ids(), "span original IDs");
        assert_eq!(span.rotary_delta(), -2);
    }
    for range in [-1..2, 0..0, 1..19, 5..3] {
        assert!(matches!(
            prepared.assemble(
                range,
                Some(&projected),
                |_| panic!("invalid span must fail before embedding"),
                &ctx
            ),
            Err(MediaInputError::Geometry(_))
        ));
    }
    assert!(result.slice(0..19, &ctx).is_err());
    // Projected media retains explicit IDs and the same PatchGrid; no tower runs.
    let projected_part = PreparedInputPart::new(
        M::Image,
        P::Embeddings(
            projected
                .index(&[Index::Full, Index::Range(0, 4), Index::Full], &ctx)
                .unwrap(),
        ),
        [
            (K::PatchGrid, grid(1, 4, 4)),
            (
                K::OriginalTokenIds,
                NumericTensor::from_i32_slice(&[12, 12, 12, 12], &[1, 4], &ctx).unwrap(),
            ),
        ],
    )
    .unwrap();
    let projected_input = input(
        vec![text(&[3, 14]), projected_part, text(&[15])],
        &inspector,
    );
    let identity = eredu_core::PreparedInputIdentity::decode_words(
        &projected_input.identity().encode_words().unwrap(),
    )
    .unwrap();
    let wire = PreparedModelInput::from_identity_wire_values(
        identity,
        projected_input.wire_values().into_iter().cloned().collect(),
        |t| inspector.identity(t),
    )
    .unwrap();
    let prepared = ingress
        .admission_config()
        .admit(&wire, &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .unwrap();
    prepared.with_vision_input(|v| assert!(v.is_none()));
    let output = prepared
        .assemble(0..prepared.token_ids().dim(1), None, embedding, &ctx)
        .unwrap();
    assert_eq!(
        output.token_ids().to_i32_vec(&ctx).unwrap(),
        [3, 14, 12, 12, 12, 12, 15]
    );
}

#[test]
fn projected_inputs_require_exact_bounded_original_identity_before_execution() {
    let (_dir, _target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let projected = NumericTensor::new([1, 3, 32], vec![0.5; 96]);
    let build = |ids: Option<NumericTensor>| {
        input(
            vec![PreparedInputPart::new(
                M::Text,
                P::Embeddings(projected.clone()),
                ids.map(|ids| (K::OriginalTokenIds, ids)),
            )
            .unwrap()],
            &inspector,
        )
    };
    assert!(matches!(
        ingress
            .admission_config()
            .admit(&build(None), &inspector)
            .and_then(|admitted| admitted.prepare(&ctx)),
        Err(MediaInputError::Metadata(
            eredu_core::PreparedInputError::MissingMetadata {
                key: K::OriginalTokenIds,
                ..
            }
        ))
    ));
    for (ids, expected) in [
        (NumericTensor::new([1, 3], vec![1., 2., 3.]), 0),
        (
            NumericTensor::from_i32_slice(&[1, 2], &[1, 2], &ctx).unwrap(),
            1,
        ),
        (
            NumericTensor::from_i32_slice(&[1, 2, 16], &[1, 3], &ctx).unwrap(),
            2,
        ),
    ] {
        let Err(MediaInputError::Tokens(error)) = ingress
            .admission_config()
            .admit(&build(Some(ids)), &inspector)
            .and_then(|admitted| admitted.prepare(&ctx))
        else {
            panic!()
        };
        assert!(matches!(
            (expected, error),
            (0, eredu_runtime::TokenInputError::ScalarType)
                | (1, eredu_runtime::TokenInputError::Geometry)
                | (2, eredu_runtime::TokenInputError::Vocabulary)
        ));
    }
    let values = NumericTensor::from_i32_slice(&[3, 0, 9], &[1, 3], &ctx).unwrap();
    let valid = ingress
        .admission_config()
        .admit(&build(Some(values)), &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .unwrap()
        .assemble(
            0..3,
            None,
            |_| panic!("projected text must not embed"),
            &ctx,
        )
        .unwrap();
    assert_eq!(valid.token_ids().to_i32_vec(&ctx).unwrap(), [3, 0, 9]);
    let excessive = ingress.admission_config().maximum_request_tokens() + 1;
    let huge = input(vec![text(&vec![3; excessive])], &inspector);
    let before = inspector.0.get();
    assert!(matches!(
        ingress
            .admission_config()
            .admit(&huge, &inspector)
            .and_then(|admitted| admitted.prepare(&ctx)),
        Err(MediaInputError::Geometry(_))
    ));
    assert_eq!(
        inspector.0.get(),
        before,
        "over-budget token IDs are not transferred"
    );
    let malformed = input(
        vec![PreparedInputPart::new(
            M::Image,
            P::Tensor(pixels(4)),
            [(
                K::PatchGrid,
                NumericTensor::from_i32_slice(
                    &vec![1; excessive * 3],
                    &[excessive as i32, 3],
                    &ctx,
                )
                .unwrap(),
            )],
        )
        .unwrap()],
        &inspector,
    );
    assert!(ingress
        .admission_config()
        .admit(&malformed, &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .is_err());
    assert_eq!(
        inspector.0.get(),
        before,
        "over-budget grid metadata is not transferred"
    );
    let unused = input(
        vec![PreparedInputPart::new(
            M::Image,
            P::Tensor(pixels(4)),
            [
                (K::PatchGrid, grid(1, 2, 2)),
                (
                    K::PatchPositions,
                    NumericTensor::from_i32_slice(
                        &vec![1; excessive * 3],
                        &[excessive as i32, 3],
                        &ctx,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap()],
        &inspector,
    );
    assert!(ingress
        .admission_config()
        .admit(&unused, &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .is_err());
    assert_eq!(
        inspector.0.get(),
        before,
        "unused metadata is rejected before shared inspection"
    );
    let video = input(
        vec![PreparedInputPart::new(
            M::Video,
            P::Tensor(pixels(16)),
            [(K::PatchGrid, grid(2, 4, 2))],
        )
        .unwrap()],
        &inspector,
    );
    assert!(matches!(
        ingress
            .admission_config()
            .admit(&video, &inspector)
            .and_then(|admitted| admitted.prepare(&ctx)),
        Err(MediaInputError::Geometry(_))
    ));
}

struct TargetBind<'a> {
    ordinary: &'a PreparedParameters,
    experts: Vec<PreparedParameters>,
    ctx: &'a NumericContext,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for TargetBind<'_> {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str();
        let read = |owner: &PreparedParameters| {
            super::super::super::payload::recipe_value(
                &owner.recipes()[name],
                owner.source().as_ref(),
                self.ctx,
            )
            .unwrap()
            .data
        };
        value.data = if self.ordinary.recipes().contains_key(name) {
            read(self.ordinary)
        } else {
            self.experts.iter().flat_map(read).collect()
        };
        assert_eq!(
            value.data.len(),
            value.shape.iter().product::<i32>() as usize,
            "{name}"
        );
    }
}
#[test]
fn vision_ingress_target_prefill_cached_decode_and_restore_keep_original_ngram_state() {
    use eredu_nn::{EmbeddingOperator, RotaryPosition};
    use eredu_runtime::{LayerRuntimeState, RuntimeLayerState};
    let (_dir, target, ingress) = setup();
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let prepared = ingress
        .admission_config()
        .admit(&prompt(&inspector), &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .unwrap();
    let mut tower = tower(&ingress, &ctx);
    let projected = prepared
        .with_vision_input(|v| tower.forward(v.unwrap(), &ctx))
        .unwrap()
        .embeddings;
    let run = |chunks: &[i32], automatic: bool| {
        let mut model =
            TargetModel::<NumericBackend>::new(target.bound_spec().unwrap(), &ctx).unwrap();
        <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut TargetBind{ordinary:target.static_parameters(),experts:vec![],ctx:&ctx});
        let units = (0..target.spec().units.len())
            .map(|i| {
                let mut unit = model.construct_unit(i, &ctx).unwrap();
                let experts = if matches!(target.spec().units[i], UnitSpec::Decoder { .. }) {
                    (0..3).map(|e| target.expert(i, e).unwrap()).collect()
                } else {
                    vec![]
                };
                unit.visit_parameters_mut(&mut TargetBind {
                    ordinary: target.unit(i).unwrap(),
                    experts,
                    ctx: &ctx,
                });
                unit
            })
            .collect();
        let rows = target
            .row_lookups(
                RowLookupLimits {
                    requests: 128,
                    rows_per_acquisition: 2,
                    acquisition_bytes: 128,
                    host_bytes: 65536,
                    output_bytes: 32768,
                },
                ResidencyPolicy::Cacheable,
            )
            .unwrap();
        let rows = rows
            .bind(
                rows.entries()
                    .iter()
                    .map(|(id, e)| (id.clone(), crate::row_bank::SourceRows::new(e)))
                    .collect(),
            )
            .unwrap();
        let mut provider = ParameterProviders {
            grouped: ResidentExpertProvider,
            rows,
        };
        let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
        let mut state =
            super::super::gguf::state_from_layout(&target.spec().state_layout().unwrap()).unwrap();
        let mut offset = 0;
        let mut prefill = Vec::new();
        for &length in chunks {
            let slice = prepared
                .assemble(
                    offset..offset + length,
                    Some(&projected),
                    |ids| {
                        assert!(
                            ids.dim(1) <= length,
                            "embedding workspace exceeds this target span"
                        );
                        <TargetModel<NumericBackend> as LayeredArchitecture<
                            NumericBackend,
                            State,
                        >>::static_modules_mut(runtime.architecture_mut())
                        .embeddings
                        .forward(ids, &ctx)
                    },
                    &ctx,
                )
                .unwrap();
            prefill.push(
                slice
                    .with_target_input(|input| {
                        runtime.forward_with_provider_and_observer_and_context(
                            input,
                            &mut state,
                            ExpertPass::Prefill,
                            &mut provider,
                            &ctx,
                            &mut Observe::default(),
                        )
                    })
                    .unwrap()
                    .0,
            );
            offset += length;
        }
        assert_eq!(offset, prepared.token_ids().dim(1));
        let mut outputs = vec![NumericTensor::concatenate(&prefill, 1, &ctx).unwrap()];
        for step in 0..16 {
            let id = [(step % 12) as u64];
            let causal = state.layer(0).unwrap().position();
            let rotary = causal.checked_sub(2).unwrap();
            let checkpoint = state.clone();
            let mut forward = |state: &mut State| {
                runtime
                    .forward_with_provider_and_observer_and_context(
                        TargetInput {
                            position_delta: None,
                            ids: Some(OriginalTokenIds::Host(&id)),
                            batch: 1,
                            tokens: 1,
                            embeddings: None,
                            visible: None,
                            rotary: (!automatic).then_some(RotaryPosition::Offset(rotary)),
                        },
                        state,
                        ExpertPass::Decode,
                        &mut provider,
                        &ctx,
                        &mut Observe::default(),
                    )
                    .unwrap()
                    .0
            };
            let expected = forward(&mut state);
            if step == 3 {
                state = checkpoint;
                assert_tensor_exact(&forward(&mut state), &expected, "media restore/replay");
            }
            outputs.push(expected);
        }
        (outputs, state)
    };
    let whole = run(&[18], false);
    assert!(whole.0[0].data.iter().any(|v| v.abs() > 0.01));
    for chunks in [vec![1, 3, 2, 3, 9], vec![1; 18]] {
        let split = run(&chunks, true);
        for (a, b) in whole.0.iter().zip(&split.0) {
            assert_tensor_close(a, b, "media target chunk/decode parity");
        }
        for (a, b) in whole.1.as_ref().iter().zip(split.1.as_ref()) {
            let a: Vec<_> = RuntimeLayerState::retained_values(a).collect();
            let b: Vec<_> = RuntimeLayerState::retained_values(b).collect();
            assert_eq!(a.len(), b.len());
            for (a, b) in a.into_iter().zip(b) {
                assert_tensor_close(a, b, "media target retained state");
                assert_eq!(a.exact_i32, b.exact_i32);
            }
        }
    }
}

#[path = "position_session.rs"]
mod position_session;

#[path = "joint.rs"]
mod joint;

#[path = "admission.rs"]
mod admission;

#[test]
fn prepared_request_preserves_resources_and_bounds_low_level_assembly_before_embedding() {
    let (_dir, _target, ingress) = setup();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = input(vec![text(&[3; 33])], &inspector);
    let admitted = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap();
    let prepared = admitted.prepare(&ctx).unwrap();

    let calls = std::cell::Cell::new(0);
    assert!(matches!(
        prepared.assemble(
            0..33,
            None,
            |ids| {
                calls.set(calls.get() + 1);
                embedding(ids)
            },
            &ctx
        ),
        Err(MediaInputError::Geometry(
            "target span exceeds admitted assembly bound"
        ))
    ));
    assert_eq!(calls.get(), 0);
    let span = prepared.assemble(0..32, None, embedding, &ctx).unwrap();
    assert_eq!(span.token_ids().dim(1), 32);
    let tail = prepared.assemble(32..33, None, embedding, &ctx).unwrap();
    assert_eq!(tail.token_ids().dim(1), 1);
}
