use super::*;
use crate::generation::{payload_copy_count, SamplerProjectionError};
use crate::working_memory::sampling::request::quote_sampling_workspace_legacy;

type Trace = Rc<RefCell<Vec<(String, Vec<WorkspaceLayout>, Vec<WorkspaceLayout>)>>>;

#[derive(Debug)]
struct ProjectionFacts {
    trace: Trace,
    missing_penalties: bool,
    alias_output: bool,
}
impl WorkspaceMechanisms for ProjectionFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        self.trace.borrow_mut().push((
            format!("{:?}", op.kind),
            op.inputs.clone(),
            op.outputs.clone(),
        ));
        let outputs = op
            .outputs
            .iter()
            .map(|out| {
                let bytes = out.bytes()?.checked_add(16).unwrap();
                Ok(
                    if self.alias_output
                        && matches!(
                            op.kind,
                            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Greedy)
                        )
                    {
                        WorkspaceOutputStorage::AllocateOrAliasInputs {
                            bytes,
                            inputs: vec![0],
                        }
                    } else {
                        WorkspaceOutputStorage::Allocate(bytes)
                    },
                )
            })
            .collect::<Result<_, Error>>()?;
        Ok(Some(WorkspaceOperationBound {
            outputs,
            scratch_bytes: 11,
            assumptions: "padded outputs and fixed workspace".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        if self.missing_penalties
            && matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Penalties { .. })
            )
        {
            return Ok(None);
        }
        let history = match op.kind {
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Penalties {
                history_positions,
                ..
            }) => u64::try_from(history_positions)
                .unwrap()
                .checked_mul(8)
                .unwrap(),
            _ => 0,
        };
        Ok(Some(WorkspaceHostBound {
            bytes: 5 + history,
            assumptions: "full history-position staging plus fixed host workspace".into(),
        }))
    }
}

fn projection_source(
    adaptive: bool,
    cleared: bool,
    window: i32,
    filtered: bool,
) -> ConfiguredTextSampler {
    let mut source = if adaptive {
        let mut sampler = MirostatV2Sampler::new(3.7, 0.23)
            .unwrap()
            .penalties(1.19, window, 0.13, 0.41);
        for (token, probability) in [3, 11, 7, 19, 5]
            .into_iter()
            .zip([0.2, 0.3, 0.1, 0.4, 0.05])
        {
            sampler.accept_token(token, probability).unwrap();
        }
        ConfiguredTextSampler::MirostatV2(sampler)
    } else {
        let mut sampler = GenerationSampler::new()
            .top_k(if filtered { 7 } else { 0 })
            .top_p(if filtered { 0.83 } else { 1.0 })
            .min_p(if filtered { 0.07 } else { 0.0 })
            .penalties(1.13, window, 0.17, 0.29);
        for token in [3, 11, 7, 19, 5] {
            sampler.accept_token(token);
        }
        ConfiguredTextSampler::Standard(sampler)
    };
    if cleared {
        match &mut source {
            ConfiguredTextSampler::Standard(s) => s.clear_generated_tokens(),
            ConfiguredTextSampler::MirostatV2(s) => s.reset(),
        }
    }
    assert_eq!(source.history_capacity(), 8);
    source
}

fn inspected(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    steps: u64,
    projected: bool,
    missing: bool,
    optional: bool,
) -> (SamplingWorkspaceReport, Trace) {
    let trace = Rc::new(RefCell::new(Vec::new()));
    let context = WorkspaceContext::new(ProjectionFacts {
        trace: trace.clone(),
        missing_penalties: missing,
        alias_output: temperature == 0.0,
    });
    let random = (temperature > 0.0).then(|| {
        WorkspaceSamplingRandomState::from_key(
            WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
                &WorkspaceExistingStorage::new(Some(64), &context),
                &context,
            )
            .unwrap(),
        )
        .unwrap()
    });
    let filter = if optional {
        TextFilterWorkspace::OptionalMask {
            max_mask_positions: 37,
            mask_capacity_bytes: 64,
        }
    } else {
        TextFilterWorkspace::Exact(&TokenFilter::All)
    };
    let logits = layout();
    let input = WorkspaceSamplingInput {
        layout: &logits,
        backing_capacity_bytes: Some(256),
    };
    let report = if projected {
        quote_sampling_workspace(
            sampler,
            temperature,
            random.as_ref(),
            input,
            filter,
            steps,
            &context,
        )
    } else {
        quote_sampling_workspace_legacy(
            sampler,
            temperature,
            random.as_ref(),
            input,
            filter,
            steps,
            &context,
        )
    }
    .unwrap();
    (report, trace)
}

fn assert_report(left: &SamplingWorkspaceReport, right: &SamplingWorkspaceReport) {
    assert_eq!(left.output_width, right.output_width);
    assert_eq!(left.steps, right.steps);
    assert_eq!(left.peak, right.peak);
    assert_eq!(left.tensor_peak_bytes, right.tensor_peak_bytes);
    assert_eq!(left.host_peak_bytes, right.host_peak_bytes);
    assert_eq!(left.first_gap, right.first_gap);
    assert_eq!(left.final_history_bytes, right.final_history_bytes);
}

#[test]
fn populated_projection_matches_real_sampler_operation_shapes_and_complete_bounds() {
    for adaptive in [false, true] {
        for cleared in [false, true] {
            for window in [-1, 0, 3] {
                for filtered in [false, true] {
                    let source = projection_source(adaptive, cleared, window, filtered);
                    for steps in [0, 4, 11, 27] {
                        for temperature in if adaptive {
                            &[0.73][..]
                        } else {
                            &[0.0, 0.73][..]
                        } {
                            let (projected, actual_trace) =
                                inspected(&source, *temperature, steps, true, false, filtered);
                            let (actual, expected_trace) =
                                inspected(&source, *temperature, steps, false, false, filtered);
                            assert_report(&projected, &actual);
                            assert_eq!(*actual_trace.borrow(), *expected_trace.borrow());
                        }
                    }
                    assert_eq!(source.history_len(), if cleared { 0 } else { 5 });
                    assert_eq!(source.history_capacity(), 8);
                }
            }
        }
    }
}

#[test]
fn cold_projection_and_full_quote_never_copy_populated_history() {
    let mut policy = GenerationSampler::new().penalties(1.2, -1, 0.2, 0.3);
    for token in 0..8191 {
        policy.accept_token(token % 37);
    }
    let source = ConfiguredTextSampler::Standard(policy);
    let before = payload_copy_count();
    let projection = source.workspace_projection();
    assert_eq!(projection.history_len(), 8191);
    assert_eq!(projection.history_capacity(), 8192);
    projection.validate_steps(2).unwrap();
    let (report, _) = inspected(&source, 0.0, 2, true, false, false);
    assert_eq!(report.final_history_bytes, 16384 * 4);
    assert_eq!(payload_copy_count(), before);
    let (reference, _) = inspected(&source, 0.0, 2, false, false, false);
    assert_eq!(
        payload_copy_count(),
        before + 1,
        "the real history-copy worker is observed"
    );
    assert_report(&report, &reference);
    let ConfiguredTextSampler::Standard(policy) = &source else {
        unreachable!()
    };
    assert_eq!(policy.generated_tokens().len(), 8191);
    assert_eq!(&policy.generated_tokens()[8188..], &[11, 12, 13]);
}

#[test]
fn source_spare_capacity_and_future_overlap_are_preserved_without_history_storage() {
    for cleared in [false, true] {
        let mut odd = GenerationSampler::new().with_generated_tokens([3, 11, 7, 19, 5]);
        if cleared {
            odd.clear_generated_tokens();
        }
        for source in [
            projection_source(false, cleared, -1, false),
            ConfiguredTextSampler::Standard(odd),
        ] {
            let initial_capacity = source.history_capacity();
            assert!(matches!(initial_capacity, 5 | 8));
            let mut cursor = source.workspace_projection();
            let mut actual = source.clone();
            for _ in 0..32 {
                cursor.advance(None).unwrap();
                let ConfiguredTextSampler::Standard(real) = &mut actual else {
                    unreachable!()
                };
                real.accept_token(29);
                assert_eq!(cursor.history_len(), real.generated_tokens().len());
                assert_eq!(cursor.history_capacity(), actual.history_capacity());
            }
            for steps in [0, 1, 4, 11, 27] {
                let (projected, _) = inspected(&source, 0.0, steps, true, false, false);
                let (real, _) = inspected(&source, 0.0, steps, false, false, false);
                assert_report(&projected, &real);
            }
            assert_eq!(source.history_capacity(), initial_capacity);
        }
    }
}

#[test]
fn adaptive_scalar_projection_uses_shared_probability_update_and_rejects_without_progress() {
    let source = projection_source(true, false, 3, false);
    let mut cursor = source.workspace_projection();
    let mut actual = source.clone();
    let source_mu = cursor.mirostat_mu().unwrap();
    for probability in [0.125, 0.75, 0.04, 1.0, 0.5] {
        cursor.advance(Some(probability)).unwrap();
        let ConfiguredTextSampler::MirostatV2(real) = &mut actual else {
            unreachable!()
        };
        real.accept_token(23, probability).unwrap();
        assert_eq!(cursor.mirostat_mu().unwrap().to_bits(), real.mu().to_bits());
        assert_eq!(cursor.history_len(), real.generated_tokens().len());
        assert_eq!(cursor.history_capacity(), actual.history_capacity());
    }
    let before = (
        cursor.history_len(),
        cursor.history_capacity(),
        cursor.mirostat_mu().unwrap().to_bits(),
    );
    assert!(matches!(
        cursor.advance(Some(0.0)),
        Err(SamplerProjectionError::Sampling(_))
    ));
    assert_eq!(
        before,
        (
            cursor.history_len(),
            cursor.history_capacity(),
            cursor.mirostat_mu().unwrap().to_bits()
        )
    );
    let ConfiguredTextSampler::MirostatV2(original) = &source else {
        unreachable!()
    };
    assert_eq!(original.mu().to_bits(), source_mu.to_bits());
}

#[test]
fn incomplete_penalty_host_facts_remain_unknown_with_full_extent_descriptors() {
    for adaptive in [false, true] {
        for window in [-1, 0, 3] {
            let source = projection_source(adaptive, false, window, true);
            let (projected, trace) = inspected(&source, 0.73, 4, true, true, true);
            let (actual, reference_trace) = inspected(&source, 0.73, 4, false, true, true);
            assert_report(&projected, &actual);
            assert_eq!(*trace.borrow(), *reference_trace.borrow());
            assert_eq!(projected.peak.bytes(), None);
            assert_eq!(projected.first_gap, Some(0));
            assert_eq!(source.history_len(), 5);
        }
    }
}

#[test]
fn overflowing_future_extent_rejects_before_operations_or_history_copy_with_typed_cause() {
    let source = projection_source(false, false, -1, true);
    let (context, calls) = setup(false);
    let before = payload_copy_count();
    let error = quote_sampling_workspace(
        &source,
        0.0,
        None,
        &layout(),
        &TokenFilter::All,
        u64::MAX,
        &context,
    )
    .unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(error) = cause.downcast_ref::<SamplerProjectionError>() {
            assert_eq!(*error, SamplerProjectionError::HistoryOverflow);
            break;
        }
        cause = cause
            .source()
            .expect("typed projection error remains in source chain");
    }
    assert!(calls.borrow().is_empty());
    assert_eq!(payload_copy_count(), before);
    assert_eq!(source.history_len(), 5);
    assert_eq!(source.history_capacity(), 8);
}

#[test]
fn static_policy_comparison_allows_new_limits_without_resetting_history_or_adaptive_mu() {
    use eredu_core::{ResolvedGenerationConfig, TextGenerationConfig, TextInferencePolicy};
    let controls = ResolvedGenerationConfig {
        do_sample: true,
        temperature: 0.73,
        top_k: 7,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 3,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(4),
    };
    let initial = TextGenerationConfig::new(controls);
    let mut standard = ConfiguredTextSampler::from_config(initial).unwrap();
    let ConfiguredTextSampler::Standard(policy) = &mut standard else {
        unreachable!()
    };
    for id in [3, 11, 7, 19, 5] {
        policy.accept_token(id);
    }
    let before = payload_copy_count();
    let mut future = controls;
    future.max_new_tokens = Some(32);
    future.temperature = 0.5;
    let future_config = TextGenerationConfig::new(future)
        .with_seed(99)
        .with_inference_policy(TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(1),
            managed_memory_capacity_bytes: Some(9999),
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        });
    assert!(standard.matches_config_policy(future_config));
    for index in 0..7 {
        let mut changed = future;
        match index {
            0 => changed.top_k += 1,
            1 => changed.top_p = 0.5,
            2 => changed.min_p = 0.2,
            3 => changed.repetition_penalty = 1.5,
            4 => changed.repeat_last_n = -1,
            5 => changed.frequency_penalty = 0.3,
            _ => changed.presence_penalty = 0.4,
        }
        assert!(!standard.matches_config_policy(TextGenerationConfig::new(changed)));
    }
    let adaptive_config = initial.with_mirostat_v2(3.7, 0.23).unwrap();
    let mut adaptive = ConfiguredTextSampler::from_config(adaptive_config).unwrap();
    let ConfiguredTextSampler::MirostatV2(policy) = &mut adaptive else {
        unreachable!()
    };
    policy.accept_token(11, 0.125).unwrap();
    let mu = policy.mu().to_bits();
    // Standard filters are not selected by Mirostat; its penalties still are.
    future.top_k = 1;
    future.top_p = 0.3;
    future.min_p = 0.9;
    let candidate = TextGenerationConfig::new(future)
        .with_mirostat_v2(3.7, 0.23)
        .unwrap();
    assert!(adaptive.matches_config_policy(candidate));
    assert!(!standard.matches_config_policy(candidate));
    assert!(!adaptive.matches_config_policy(initial));
    assert!(!adaptive.matches_config_policy(initial.with_mirostat_v2(4.7, 0.23).unwrap()));
    assert!(!adaptive.matches_config_policy(initial.with_mirostat_v2(3.7, 0.33).unwrap()));
    future.frequency_penalty = 0.5;
    assert!(!adaptive.matches_config_policy(
        TextGenerationConfig::new(future)
            .with_mirostat_v2(3.7, 0.23)
            .unwrap()
    ));
    let ConfiguredTextSampler::MirostatV2(policy) = &adaptive else {
        unreachable!()
    };
    assert_eq!(policy.mu().to_bits(), mu);
    assert_eq!(policy.generated_tokens(), &[11]);
    assert_eq!(standard.history_len(), 5);
    assert_eq!(payload_copy_count(), before);
}
