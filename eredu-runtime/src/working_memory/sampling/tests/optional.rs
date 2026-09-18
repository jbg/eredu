use super::*;

#[derive(Debug, Clone, Copy)]
enum OutputBacking {
    Independent,
    Input,
    InputAndKey,
}

#[derive(Debug)]
struct OptionalFacts {
    calls: Rc<RefCell<Vec<WorkspaceSamplingOperation>>>,
    missing_native_filter: bool,
    missing_host_filter: bool,
    output: OutputBacking,
    key_capacity: u64,
}

impl WorkspaceMechanisms for OptionalFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        use WorkspaceSamplingOperation as S;
        if let WorkspaceOperationKind::Sampling(kind) = &op.kind {
            self.calls.borrow_mut().push(kind.clone());
        }
        if self.missing_native_filter
            && matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(S::OptionalTokenFilter)
            )
        {
            return Ok(None);
        }
        let outputs = op
            .outputs
            .iter()
            .map(|out| {
                Ok(match &op.kind {
                    WorkspaceOperationKind::Sampling(S::TokenFilter) => {
                        WorkspaceOutputStorage::Allocate(512)
                    }
                    WorkspaceOperationKind::Sampling(S::OptionalTokenFilter) => {
                        WorkspaceOutputStorage::AllocateOrAliasInputs {
                            bytes: 512,
                            inputs: vec![0],
                        }
                    }
                    WorkspaceOperationKind::Sampling(S::SplitRandomKey) => {
                        WorkspaceOutputStorage::Allocate(self.key_capacity)
                    }
                    WorkspaceOperationKind::Sampling(S::Greedy | S::Categorical) => {
                        match self.output {
                            OutputBacking::Independent => WorkspaceOutputStorage::Allocate(64),
                            OutputBacking::Input => WorkspaceOutputStorage::AliasInput(0),
                            OutputBacking::InputAndKey => {
                                WorkspaceOutputStorage::AllocateOrAliasInputs {
                                    bytes: 64,
                                    inputs: (0..op.inputs.len()).collect(),
                                }
                            }
                        }
                    }
                    WorkspaceOperationKind::Sampling(S::ReadToken | S::SelectRandomKey { .. })
                    | WorkspaceOperationKind::Index { .. } | WorkspaceOperationKind::StaticSlice { .. } => WorkspaceOutputStorage::AliasInput(0),
                    _ if !op.inputs.is_empty() && out == &op.inputs[0] => {
                        // Intermediate transforms may preserve the filtered score's
                        // backing, so later token aliases reach all original roots.
                        WorkspaceOutputStorage::AllocateOrAliasInputs {
                            bytes: out.bytes()?,
                            inputs: vec![0],
                        }
                    }
                    _ => WorkspaceOutputStorage::Allocate(out.bytes()?),
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Some(WorkspaceOperationBound {
            outputs,
            scratch_bytes: 7,
            assumptions:
                "padded filtered scores and keys; optional filtering retains both branch roots"
                    .into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        if self.missing_host_filter
            && matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::OptionalTokenFilter)
            )
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceHostBound {
            bytes: 3,
            assumptions: "three bytes of host staging per executed metadata primitive".into(),
        }))
    }
}

fn optional(capacity: u64) -> TextFilterWorkspace<'static> {
    TextFilterWorkspace::OptionalMask {
        max_mask_positions: 37,
        mask_capacity_bytes: capacity,
    }
}

fn optional_quote(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    steps: u64,
    filter: TextFilterWorkspace<'_>,
    input_capacity: Option<u64>,
    facts: OptionalFacts,
) -> SamplingWorkspaceReport {
    let context = WorkspaceContext::new(facts);
    let random =
        (temperature > 0.0).then(|| WorkspaceSamplingRandomState::from_seed(&context).unwrap());
    let layout = layout();
    quote_sampling_workspace(
        sampler,
        temperature,
        random.as_ref(),
        WorkspaceSamplingInput {
            layout: &layout,
            backing_capacity_bytes: input_capacity,
        },
        filter,
        steps,
        &context,
    )
    .unwrap()
}

fn facts(output: OutputBacking) -> OptionalFacts {
    OptionalFacts {
        calls: Rc::new(RefCell::new(Vec::new())),
        missing_native_filter: false,
        missing_host_filter: false,
        output,
        key_capacity: 128,
    }
}

#[test]
fn optional_filtering_uses_every_configured_sampler_and_prices_only_emitted_masks() {
    for (sampler, temperature, _) in retained_output_modes() {
        for steps in [0, 1, 4] {
            let small_facts = facts(OutputBacking::Independent);
            let calls = small_facts.calls.clone();
            let small = optional_quote(
                &sampler,
                temperature,
                steps,
                optional(37),
                Some(2048),
                small_facts,
            );
            let padded = optional_quote(
                &sampler,
                temperature,
                steps,
                optional(73),
                Some(2048),
                facts(OutputBacking::Independent),
            );
            assert_eq!(small.first_gap, None);
            assert_eq!(small.steps, steps);
            assert_eq!(small.output_width, 37);
            assert_eq!(padded.tensor_peak_bytes, small.tensor_peak_bytes);
            assert_eq!(padded.final_history_bytes, small.final_history_bytes);
            assert_eq!(
                padded.peak.bytes().unwrap() - small.peak.bytes().unwrap(),
                if steps == 0 { 0 } else { 36 }
            );
            let calls = calls.borrow();
            assert_eq!(
                calls
                    .iter()
                    .filter(|kind| matches!(kind, WorkspaceSamplingOperation::OptionalTokenFilter))
                    .count(),
                steps as usize
            );
            assert_eq!(
                calls
                    .iter()
                    .filter(|kind| matches!(
                        kind,
                        WorkspaceSamplingOperation::Greedy
                            | WorkspaceSamplingOperation::Categorical
                    ))
                    .count(),
                steps as usize
            );
            assert!(!calls
                .iter()
                .any(|kind| matches!(kind, WorkspaceSamplingOperation::TokenFilter)));
            if matches!(sampler, ConfiguredTextSampler::MirostatV2(_)) {
                assert_eq!(
                    calls
                        .iter()
                        .filter(|kind| matches!(kind, WorkspaceSamplingOperation::TokenProbability))
                        .count(),
                    steps as usize
                );
            }
            assert_eq!(sampler.history_len(), 0);
        }
    }
}

#[test]
fn optional_filter_facts_are_required_for_nonzero_allowance_but_not_initialization() {
    for missing_native in [false, true] {
        for steps in [0, 3] {
            let mut missing = facts(OutputBacking::Independent);
            missing.missing_native_filter = missing_native;
            missing.missing_host_filter = !missing_native;
            let calls = missing.calls.clone();
            let report = optional_quote(&plain_sampler(), 0.7, steps, optional(37), None, missing);
            if steps == 0 {
                let all = optional_quote(
                    &plain_sampler(),
                    0.7,
                    0,
                    (&TokenFilter::All).into(),
                    None,
                    facts(OutputBacking::Independent),
                );
                assert_eq!(report.peak.bytes(), all.peak.bytes());
                assert_eq!(report.tensor_peak_bytes, all.tensor_peak_bytes);
                assert_eq!(report.host_peak_bytes, all.host_peak_bytes);
                assert_eq!(report.first_gap, None);
                assert!(!calls
                    .borrow()
                    .iter()
                    .any(|kind| matches!(kind, WorkspaceSamplingOperation::OptionalTokenFilter)));
            } else {
                assert_eq!(report.first_gap, Some(0));
                assert_eq!(report.peak.bytes(), None);
                if missing_native {
                    assert_eq!(report.tensor_peak_bytes, None);
                } else {
                    assert_eq!(report.host_peak_bytes, None);
                }
            }
        }
    }
}

#[test]
fn fresh_score_backings_survive_optional_output_aliases_at_actual_capacity() {
    for steps in [1, 2, 4] {
        let quote = |capacity| {
            optional_quote(
                &plain_sampler(),
                0.0,
                steps,
                optional(37),
                capacity,
                facts(OutputBacking::Input),
            )
        };
        let compact = quote(Some(148));
        let padded = quote(Some(2048));
        let broadcast = quote(Some(4));
        assert_eq!(
            padded.tensor_peak_bytes.unwrap() - compact.tensor_peak_bytes.unwrap(),
            (steps - 1) * (2048 - 148)
        );
        assert_eq!(
            compact.tensor_peak_bytes.unwrap() - broadcast.tensor_peak_bytes.unwrap(),
            (steps - 1) * (148 - 4)
        );
        // Every previous optional output retains both a possible new 512-byte
        // masked buffer and that step's distinct 2048-byte source. The current
        // input belongs to equations; current filter allocation/scratch is ours.
        assert_eq!(
            padded.tensor_peak_bytes,
            Some((steps - 1) * (512 + 2048) + 512 + 3 * 7)
        );
        let unknown = quote(None);
        assert_eq!(unknown.first_gap, if steps == 1 { None } else { Some(1) });
        assert_eq!(unknown.tensor_peak_bytes.is_some(), steps == 1);
    }
}

#[test]
fn optional_aliases_keep_prior_key_and_score_roots_without_counting_shared_views_twice() {
    for steps in [1, 2, 4] {
        let quote = |input_capacity, key_capacity| {
            let mut selected = facts(OutputBacking::InputAndKey);
            selected.key_capacity = key_capacity;
            optional_quote(
                &plain_sampler(),
                0.7,
                steps,
                optional(37),
                Some(input_capacity),
                selected,
            )
        };
        let compact = quote(148, 128);
        let padded_keys = quote(148, 256);
        let padded_scores = quote(2048, 128);
        assert_eq!(
            padded_keys.tensor_peak_bytes.unwrap() - compact.tensor_peak_bytes.unwrap(),
            steps * 128
        );
        assert_eq!(
            padded_scores.tensor_peak_bytes.unwrap() - compact.tensor_peak_bytes.unwrap(),
            (steps - 1) * (2048 - 148)
        );
        assert_eq!(padded_scores.host_peak_bytes, compact.host_peak_bytes);
        assert_eq!(padded_keys.first_gap, None);
    }
}

fn concrete_schedule_peak(schedule: &[bool], capacity: u64) -> u64 {
    let context = WorkspaceContext::new(facts(OutputBacking::Input));
    let layout = layout();
    let mask = TokenFilter::allowed(vec![true; 37]).unwrap();
    let mut sampler = plain_sampler();
    let mut emitted = Vec::new();
    let mut peak = 0;
    for forced in schedule {
        context.begin_state_span(emitted.iter()).unwrap();
        let scores = WorkspaceTensor::existing_with_storage(
            layout.clone(),
            &WorkspaceExistingStorage::new(Some(capacity), &context),
            &context,
        )
        .unwrap();
        let filtered = WorkspaceSamplingBackend::apply_token_filter(
            &scores,
            if *forced { &mask } else { &TokenFilter::All },
            &context,
        )
        .unwrap();
        let token = Sampler::<WorkspaceSamplingBackend>::sample(
            &mut sampler,
            &filtered,
            0.0,
            None,
            &context,
        )
        .unwrap();
        emitted.push(token);
        let report = context.report(&[]).unwrap();
        peak = peak.max(
            report.tensor_buffers.total_bytes.unwrap()
                + report.state.unwrap().displaced_bytes.unwrap(),
        );
    }
    peak
}

#[test]
fn optional_trace_covers_mixed_schedules_and_exact_aliases_use_fresh_inputs_too() {
    let report = optional_quote(
        &plain_sampler(),
        0.0,
        4,
        optional(37),
        Some(2048),
        facts(OutputBacking::Input),
    );
    for schedule in [[false, true, false, true], [true, false, true, false]] {
        let concrete = concrete_schedule_peak(&schedule, 2048);
        assert!(report.tensor_peak_bytes.unwrap() >= concrete);
        assert!(
            concrete > 2048,
            "earlier unfiltered score storage must survive later steps"
        );
    }
    let late_filter = concrete_schedule_peak(&[false, false, false, true], 2048);
    let homogeneous =
        concrete_schedule_peak(&[false; 4], 2048).max(concrete_schedule_peak(&[true; 4], 2048));
    assert!(
        late_filter > homogeneous,
        "three escaped unfiltered scores overlap the final filtered allocation"
    );
    assert!(report.tensor_peak_bytes.unwrap() >= late_filter);
    let unfiltered = optional_quote(
        &plain_sampler(),
        0.0,
        4,
        (&TokenFilter::All).into(),
        Some(2048),
        facts(OutputBacking::Input),
    );
    assert_eq!(unfiltered.tensor_peak_bytes, Some(3 * 2048 + 2 * 7));
    assert_eq!(
        unfiltered.tensor_peak_bytes,
        Some(concrete_schedule_peak(&[false; 4], 2048))
    );
}

#[test]
fn exact_filter_compatibility_and_packed_input_contract_remain_explicit() {
    let layout = layout();
    let descriptor = WorkspaceSamplingInput::from(&layout);
    assert_eq!(descriptor.backing_capacity_bytes, Some(148));
    let mut mask = Vec::with_capacity(71);
    mask.extend([true; 37]);
    let mask = TokenFilter::allowed(mask).unwrap();
    let context = WorkspaceContext::new(facts(OutputBacking::Independent));
    let old =
        quote_sampling_workspace(&plain_sampler(), 0.0, None, &layout, &mask, 0, &context).unwrap();
    let exact = optional_quote(
        &plain_sampler(),
        0.0,
        0,
        TextFilterWorkspace::Exact(&mask),
        Some(148),
        facts(OutputBacking::Independent),
    );
    let optional = optional_quote(
        &plain_sampler(),
        0.0,
        0,
        optional(71),
        Some(148),
        facts(OutputBacking::Independent),
    );
    assert_eq!(old.peak.bytes(), exact.peak.bytes());
    assert_eq!(
        exact.host_peak_bytes.unwrap() - optional.host_peak_bytes.unwrap(),
        71
    );

    for filter in [
        TextFilterWorkspace::OptionalMask {
            max_mask_positions: 0,
            mask_capacity_bytes: 0,
        },
        TextFilterWorkspace::OptionalMask {
            max_mask_positions: 37,
            mask_capacity_bytes: 36,
        },
        TextFilterWorkspace::OptionalMask {
            max_mask_positions: 37,
            mask_capacity_bytes: u64::MAX,
        },
    ] {
        let selected = facts(OutputBacking::Independent);
        let calls = selected.calls.clone();
        let context = WorkspaceContext::new(selected);
        assert!(quote_sampling_workspace(
            &plain_sampler(),
            0.0,
            None,
            &layout,
            filter,
            1,
            &context
        )
        .is_err());
        assert!(calls.borrow().is_empty());
    }
}
