use super::*;

// This local cache models the native contiguous contract exactly: retain W-1
// past tokens after returning the complete attention view. NumericCache's global
// W-token retained representation is unchanged for all existing callers.
#[derive(Clone)]
struct NativeTail {
    cache: NumericCache,
    masks: Vec<Option<NumericTensor>>,
}
impl NativeTail {
    fn new() -> Self {
        Self {
            cache: NumericCache::new(Some(4)),
            masks: Vec::new(),
        }
    }
}
impl AttentionCache<NumericTensor> for NativeTail {
    fn uses_blockwise_attention(&self) -> bool {
        self.cache.uses_blockwise_attention()
    }
    fn offset(&self) -> i32 {
        self.cache.offset()
    }
    fn max_size(&self) -> Option<i32> {
        self.cache.max_size()
    }
    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        let result = self.cache.update_for_attention(keys, values, context)?;
        for slot in [&mut self.cache.keys, &mut self.cache.values] {
            let value = slot.as_ref().unwrap();
            let end = value.shape[2] as usize;
            *slot = Some(value.axis_slice(2, end.saturating_sub(3), end));
        }
        Ok(result)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.masks.push(request.mask.cloned());
        self.cache.attention(request, context)
    }
}
fn modules(f: &Fixture, context: &NumericContext) -> [gemma4::Attention<NumericBackend>; 2] {
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, tensor: &'a mut NumericTensor) {
            let source = self.0.get(metadata.id().as_str()).unwrap();
            assert_eq!(tensor.shape, source.shape);
            tensor.data.clone_from(&source.data);
            nonzero(tensor);
        }
    }
    // The actual configuration's first sliding publisher and shared consumer.
    [0, 2].map(|index| {
        let mut module = gemma4::Attention::new(
            &f.args.text,
            index,
            f.args.text.layer_policy(index).unwrap(),
            context,
        )
        .unwrap();
        module.visit_parameters_mut(&mut Populate(&f.parameters));
        module
    })
}
fn hidden(start: usize, count: usize) -> NumericTensor {
    NumericTensor::new(
        [1, count as i32, 8],
        (start * 8..(start + count) * 8)
            .map(|i| ((i * 7 % 43) as f32 - 21.) * 0.07)
            .collect(),
    )
}
fn pair(
    modules: &mut [gemma4::Attention<NumericBackend>; 2],
    cache: &mut NativeTail,
    input: &NumericTensor,
    mask: Option<&NumericTensor>,
    context: &NumericContext,
) -> [NumericTensor; 2] {
    let start = cache.offset();
    let mut shared = gemma4::SharedAttentionStates::new();
    let publisher = modules[0]
        .forward(
            gemma4::AttentionInput {
                hidden: input,
                mask,
                cache: Some(&mut *cache),
                shared: &mut shared,
                rotary_position: None,
            },
            context,
        )
        .unwrap();
    let end = cache.offset();
    assert_eq!(end, start + input.shape[1]);
    assert_eq!(shared.len(), 1);
    let consumer = modules[1]
        .forward(
            gemma4::AttentionInput {
                hidden: &publisher,
                mask,
                cache: Some(&mut *cache),
                shared: &mut shared,
                rotary_position: None,
            },
            context,
        )
        .unwrap();
    assert_eq!(cache.offset(), end, "consumer must not append shared state");
    nonzero(&publisher);
    nonzero(&consumer);
    [publisher, consumer]
}
fn same_state(a: &NativeTail, b: &NativeTail, a_chunk: usize, b_chunk: usize) {
    assert_eq!(a.cache.offset, b.cache.offset);
    assert_eq!(a.cache.window, b.cache.window);
    for (a, b) in [
        (&a.cache.keys, &b.cache.keys),
        (&a.cache.values, &b.cache.values),
    ] {
        let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
        assert!(a.shape[2] <= 3);
        assert_tensor_close(a, b, "complete native-style retained tail");
        nonzero(a);
    }
    assert_eq!(
        a.cache.attention_history.is_some(),
        b.cache.attention_history.is_some()
    );
    if let (Some((ak, av)), Some((bk, bv))) =
        (&a.cache.attention_history, &b.cache.attention_history)
    {
        let end = a.offset() as usize;
        let an = (end - a_chunk).min(3) + a_chunk;
        let bn = (end - b_chunk).min(3) + b_chunk;
        for (a, b) in [(ak, bk), (av, bv)] {
            assert_eq!(a.shape[2] as usize, an);
            assert_eq!(b.shape[2] as usize, bn);
            assert_tensor_close(
                a,
                &b.axis_slice(2, bn - an, bn),
                "entire cache-owned invocation history",
            );
        }
    }
}

#[test]
fn gemma4_native_tail_and_cache_owned_publishers_preserve_uneven_rows_and_decode() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, false, false, false));
        for cache_owned_attention in [false, true] {
            let context = NumericContext {
                cache_owned_attention,
                ..Default::default()
            };
            let mut modules = modules(&f, &context);
            let mut prefix = NativeTail::new();
            pair(&mut modules, &mut prefix, &hidden(0, 2), None, &context);
            let mut full = prefix.clone();
            let expected = pair(&mut modules, &mut full, &hidden(2, 5), None, &context);
            let mut chunked = prefix.clone();
            let mut assembled = [Vec::new(), Vec::new()];
            let mut consumed = 0;
            for count in [2, 1, 2] {
                let output = pair(
                    &mut modules,
                    &mut chunked,
                    &hidden(2 + consumed, count),
                    None,
                    &context,
                );
                consumed += count;
                let mut oracle = prefix.clone();
                pair(
                    &mut modules,
                    &mut oracle,
                    &hidden(2, consumed),
                    None,
                    &context,
                );
                same_state(&chunked, &oracle, count, consumed);
                for (values, tensor) in assembled.iter_mut().zip(output) {
                    values.extend(tensor.data);
                }
            }
            for (values, expected) in assembled.into_iter().zip(expected) {
                assert_tensor_close(
                    &NumericTensor::new(expected.shape.clone(), values),
                    &expected,
                    "actual publisher and shared-consumer rows",
                );
            }
            same_state(&chunked, &full, 2, 5);
            for position in 7..10 {
                let a = pair(
                    &mut modules,
                    &mut chunked,
                    &hidden(position, 1),
                    None,
                    &context,
                );
                let b = pair(
                    &mut modules,
                    &mut full,
                    &hidden(position, 1),
                    None,
                    &context,
                );
                for (a, b) in a.iter().zip(b) {
                    assert_tensor_close(a, &b, "same next decode");
                }
                same_state(&chunked, &full, 1, 1);
            }
            assert_eq!(
                context.sliding_attention_calls.get() > 0,
                !cache_owned_attention
            );
            assert!(chunked.masks.iter().all(Option::is_none));
        }
    }
}

#[test]
fn gemma4_explicit_sliding_masks_keep_the_existing_cache_request() {
    let f = fixture_for(configuration(false, false, false, false));
    for cache_owned_attention in [false, true] {
        let context = NumericContext {
            cache_owned_attention,
            ..Default::default()
        };
        let mut modules = modules(&f, &context);
        let mut prefix = NativeTail::new();
        pair(&mut modules, &mut prefix, &hidden(0, 2), None, &context);
        let mut plain = prefix.clone();
        let expected = pair(&mut modules, &mut plain, &hidden(2, 2), None, &context);
        let mut masked = prefix;
        let mut mask = NumericBackend::causal_mask(2, 2, Some(3), &context).unwrap();
        mask.data[0] = -1.0e9;
        mask.data[5] = -1.0e9;
        let before = context.sliding_attention_calls.get();
        let output = pair(
            &mut modules,
            &mut masked,
            &hidden(2, 2),
            Some(&mask),
            &context,
        );
        assert_eq!(context.sliding_attention_calls.get(), before);
        for delivered in masked.masks.iter().rev().take(2) {
            assert_tensor_exact(
                delivered.as_ref().unwrap(),
                &mask,
                "unchanged explicit caller mask",
            );
        }
        assert!(output[1]
            .data
            .iter()
            .zip(&expected[1].data)
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }
}

#[test]
fn gemma4_prepared_workspace_records_actual_sliding_mechanism_and_full_attention() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, false, true, false));
        let inspection =
            eredu_architectures::configuration::inspect_artifact(f.artifact.path()).unwrap();
        let sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let before = sources.target().source_diagnostics().unwrap();
        let facts = Facts::default();
        let context = WorkspaceContext::new(facts.clone());
        let state = WorkspaceResidentStateFactory::new(
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(256).unwrap(),
            &context,
        )
        .unwrap()
        .realize(sources.selected().text_realization().state().layout())
        .unwrap();
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 7,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: OutputDemand::Sequence,
        };
        let report = sources
            .inference_blueprint()
            .quote_replicated_resident_text(geometry, &state, &context)
            .unwrap();
        assert!(matches!(report.transient(), WorkspaceBound::Bounded { .. }));
        assert!(report.retained_peak_bytes().is_some());
        assert!(state.as_ref().iter().all(|layer| layer.position() == 0));
        let operations = facts.operations.lock().unwrap();
        assert!(operations.iter().any(|operation| matches!(operation.kind,
            WorkspaceOperationKind::Attention { causal:true, window:Some((4, start)), .. } if start >= 4)));
        assert!(operations.iter().any(|operation| matches!(
            operation.kind,
            WorkspaceOperationKind::Attention { window: None, .. }
        ) && operation.inputs.len() > 3));
        assert_eq!(
            before.physical_reads,
            sources
                .target()
                .source_diagnostics()
                .unwrap()
                .physical_reads
        );
    }
}

fn uncached_pair(
    modules: &mut [gemma4::Attention<NumericBackend>; 2],
    input: &NumericTensor,
    rotary_position: Option<RotaryPosition<'_, NumericTensor>>,
    context: &NumericContext,
) -> [NumericTensor; 2] {
    let mut shared = gemma4::SharedAttentionStates::new();
    let publisher = modules[0]
        .forward(
            gemma4::AttentionInput::<_, NativeTail> {
                hidden: input,
                mask: None,
                cache: None,
                shared: &mut shared,
                rotary_position,
            },
            context,
        )
        .unwrap();
    assert_eq!(shared.len(), 1);
    for (keys, values) in shared.values() {
        assert_eq!(keys.shape[2], input.shape[1]);
        assert_eq!(values.shape[2], input.shape[1]);
        nonzero(keys);
        nonzero(values);
    }
    let consumer = modules[1]
        .forward(
            gemma4::AttentionInput::<_, NativeTail> {
                hidden: &publisher,
                mask: None,
                cache: None,
                shared: &mut shared,
                rotary_position,
            },
            context,
        )
        .unwrap();
    nonzero(&publisher);
    nonzero(&consumer);
    [publisher, consumer]
}

#[test]
fn gemma4_uncached_shared_attention_is_causal_windowed_and_preserves_explicit_rotary() {
    for sparse in [false, true] {
        let f = fixture_for(configuration(sparse, false, false, false));
        let context = NumericContext::default();
        let mut modules = modules(&f, &context);
        let input = hidden(0, 7);
        let ordinary = uncached_pair(&mut modules, &input, None, &context);
        let cached = pair(&mut modules, &mut NativeTail::new(), &input, None, &context);
        for (a, b) in ordinary.iter().zip(cached) {
            assert_tensor_close(a, &b, "uncached and initial native-tail attention");
        }
        // Nonuniform explicit positions are rotary data, not cache coordinates.
        let angles: Vec<_> = (0..14)
            .map(|i| 0.13 * ((i / 2 + 3) * (i / 2 + 3)) as f32 / (i % 2 + 1) as f32)
            .collect();
        let cosine = NumericTensor::new([7, 2], angles.iter().map(|x| x.cos()).collect());
        let sine = NumericTensor::new([7, 2], angles.iter().map(|x| x.sin()).collect());
        let explicit = uncached_pair(
            &mut modules,
            &input,
            Some(RotaryPosition::Embeddings {
                cosine: &cosine,
                sine: &sine,
            }),
            &context,
        );
        for (a, b) in ordinary.iter().zip(&explicit) {
            assert!(a
                .data
                .iter()
                .zip(&b.data)
                .any(|(a, b)| (a - b).abs() > 1e-6));
        }
        for count in [2, 4, 5] {
            let cosine = cosine.axis_slice(0, 0, count);
            let sine = sine.axis_slice(0, 0, count);
            let prefix = uncached_pair(
                &mut modules,
                &input.axis_slice(1, 0, count),
                Some(RotaryPosition::Embeddings {
                    cosine: &cosine,
                    sine: &sine,
                }),
                &context,
            );
            for (a, b) in prefix.iter().zip(&explicit) {
                assert_tensor_close(a, &b.axis_slice(1, 0, count), "no future-row visibility");
            }
        }
        let mut changed = input.clone();
        for (i, value) in changed.data[..8].iter_mut().enumerate() {
            *value += (i as f32 + 1.) * 0.19;
        }
        let changed = uncached_pair(
            &mut modules,
            &changed,
            Some(RotaryPosition::Embeddings {
                cosine: &cosine,
                sine: &sine,
            }),
            &context,
        );
        for (a, b) in changed.iter().zip(&explicit) {
            assert_tensor_close(
                &a.axis_slice(1, 6, 7),
                &b.axis_slice(1, 6, 7),
                "outside-window keys remain invisible",
            );
        }
    }
}
