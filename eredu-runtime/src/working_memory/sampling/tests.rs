use super::*;
use crate::{ConfiguredTextSampler, GenerationSampler, MirostatV2Sampler, Sampler};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

mod held_history;
mod optional;
mod projection;

#[derive(Debug)]
struct Facts {
    calls: Rc<RefCell<Vec<WorkspaceSamplingOperation>>>,
    missing_host_penalties: bool,
    missing_host_create: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if let WorkspaceOperationKind::Sampling(kind) = &op.kind {
            self.calls.borrow_mut().push(kind.clone());
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|out| Ok(WorkspaceOutputStorage::Allocate(out.bytes()?)))
                .collect::<Result<_, Error>>()?,
            scratch_bytes: 7,
            assumptions: "test mechanism retains every output and seven scratch bytes".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        if self.missing_host_create
            && matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::CreateRandomKey)
            )
        {
            return Ok(None);
        }
        if self.missing_host_penalties
            && matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Penalties { .. })
            )
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceHostBound {
            bytes: 3,
            assumptions: "test mechanism three-byte staging".into(),
        }))
    }
}
fn setup(
    missing: bool,
) -> (
    WorkspaceContext,
    Rc<RefCell<Vec<WorkspaceSamplingOperation>>>,
) {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let context = WorkspaceContext::new(Facts {
        calls: calls.clone(),
        missing_host_penalties: missing,
        missing_host_create: false,
    });
    (context, calls)
}

#[test]
fn configured_sampling_initial_key_scratch_and_unknown_host_cannot_disappear_between_spans() {
    let sampler = ConfiguredTextSampler::Standard(GenerationSampler::new());
    for missing in [false, true] {
        let context = WorkspaceContext::new(Facts {
            calls: Rc::new(RefCell::new(Vec::new())),
            missing_host_penalties: false,
            missing_host_create: missing,
        });
        let random = WorkspaceSamplingRandomState::from_seed(&context).unwrap();
        for steps in [0, 3] {
            let report = quote_sampling_workspace(
                &sampler,
                0.7,
                Some(&random),
                &layout(),
                &TokenFilter::All,
                steps,
                &context,
            )
            .unwrap();
            assert_eq!(report.output_width, 37);
            if missing {
                assert_eq!(report.peak.bytes(), None);
                assert_eq!(report.first_gap, Some(0));
                assert_eq!(report.host_peak_bytes, None);
            } else {
                assert!(report.peak.bytes().is_some());
                if steps == 0 {
                    assert_eq!(report.tensor_peak_bytes, Some(8 + 7));
                    assert_eq!(
                        report.host_peak_bytes,
                        Some(std::mem::size_of::<ConfiguredTextSampler>() as u64 + 3)
                    );
                }
            }
        }
    }
}
fn layout() -> WorkspaceLayout {
    WorkspaceLayout::new(&[1, 1, 37], WorkspaceDtype::Float32).unwrap()
}

#[test]
fn configured_sampling_quote_runs_all_history_growth_edges_without_advancing_source_state() {
    let (context, calls) = setup(false);
    let sampler =
        ConfiguredTextSampler::Standard(GenerationSampler::new().penalties(1.2, 3, 0.1, 0.2));
    let random = WorkspaceSamplingRandomState::from_seed(&context).unwrap();
    let original_key = random.key().layout().clone();
    let report = quote_sampling_workspace(
        &sampler,
        0.7,
        Some(&random),
        &layout(),
        &TokenFilter::Allowed(vec![false, true]),
        9,
        &context,
    )
    .unwrap();
    assert!(report.peak.bytes().unwrap() > 0);
    assert_eq!(report.steps, 9);
    assert_eq!(report.final_history_bytes, 16 * 4);
    assert_eq!(sampler.history_len(), 0);
    assert_eq!(sampler.history_capacity(), 0);
    assert_eq!(random.key().layout(), &original_key);
    let calls = calls.borrow();
    assert_eq!(
        calls
            .iter()
            .filter(|kind| matches!(kind, WorkspaceSamplingOperation::Categorical))
            .count(),
        9
    );
    assert_eq!(
        calls
            .iter()
            .filter(|kind| matches!(kind, WorkspaceSamplingOperation::TokenFilter))
            .count(),
        9
    );
    let histories = calls
        .iter()
        .filter_map(|kind| match kind {
            WorkspaceSamplingOperation::Penalties {
                history_positions, ..
            } => Some(*history_positions),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(histories, [1, 2, 3, 3, 3, 3, 3, 3]);
    assert_eq!(report.first_gap, None);
}

#[test]
fn configured_sampling_missing_late_host_fact_or_existing_key_capacity_stays_unknown() {
    let (context, _) = setup(true);
    let sampler =
        ConfiguredTextSampler::Standard(GenerationSampler::new().penalties(1.1, -1, 0.0, 0.0));
    let report = quote_sampling_workspace(
        &sampler,
        0.0,
        None,
        &layout(),
        &TokenFilter::All,
        7,
        &context,
    )
    .unwrap();
    assert_eq!(report.first_gap, Some(1));
    assert_eq!(report.peak.bytes(), None);
    assert!(report.tensor_peak_bytes.is_some());
    assert_eq!(report.host_peak_bytes, None);
    assert_eq!(report.final_history_bytes, 32);
    let (context, _) = setup(false);
    let unknown = WorkspaceSamplingRandomState::from_key(
        WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
            &WorkspaceExistingStorage::new(None, &context),
            &context,
        )
        .unwrap(),
    )
    .unwrap();
    for steps in [0, 1, 9] {
        let report = quote_sampling_workspace(
            &sampler,
            0.7,
            Some(&unknown),
            &layout(),
            &TokenFilter::All,
            steps,
            &context,
        )
        .unwrap();
        assert_eq!(report.peak.bytes(), None);
    }
}

#[test]
fn configured_sampling_adaptive_policy_prices_probability_commit_and_preserves_mu() {
    let (context, calls) = setup(false);
    let mut source = MirostatV2Sampler::new(5.0, 0.1)
        .unwrap()
        .penalties(1.2, -1, 0.2, 0.0);
    source.accept_token(5, 0.125).unwrap();
    let sampler = ConfiguredTextSampler::MirostatV2(source.clone());
    let random = WorkspaceSamplingRandomState::from_seed(&context).unwrap();
    let report = quote_sampling_workspace(
        &sampler,
        0.9,
        Some(&random),
        &layout(),
        &TokenFilter::All,
        5,
        &context,
    )
    .unwrap();
    assert!(report.peak.bytes().is_some());
    assert_eq!(sampler.history_len(), 1);
    let ConfiguredTextSampler::MirostatV2(after) = sampler else {
        unreachable!()
    };
    assert_eq!(after.mu(), source.mu());
    let calls = calls.borrow();
    assert_eq!(
        calls
            .iter()
            .filter(|kind| matches!(kind, WorkspaceSamplingOperation::TokenProbability))
            .count(),
        5
    );
    assert_eq!(
        calls
            .iter()
            .filter(|kind| matches!(kind, WorkspaceSamplingOperation::MirostatCutoff))
            .count(),
        5
    );
    assert!(!calls.iter().any(|kind| matches!(
        kind,
        WorkspaceSamplingOperation::TopK { .. }
            | WorkspaceSamplingOperation::TopP
            | WorkspaceSamplingOperation::MinP
    )));
}

#[test]
fn configured_sampling_rejects_wrong_context_or_geometry_and_keeps_history_capacity_on_clone() {
    let (context, _) = setup(false);
    let (foreign, _) = setup(false);
    let random = WorkspaceSamplingRandomState::from_seed(&foreign).unwrap();
    let mut sampler = ConfiguredTextSampler::Standard(GenerationSampler::new());
    assert!(quote_sampling_workspace(
        &sampler,
        0.7,
        Some(&random),
        &layout(),
        &TokenFilter::All,
        1,
        &context
    )
    .is_err());
    for shape in [vec![2, 37], vec![0], vec![1, 1, 1, 37]] {
        assert!(quote_sampling_workspace(
            &sampler,
            0.0,
            None,
            &WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
            &TokenFilter::All,
            1,
            &context
        )
        .is_err());
    }
    assert!(quote_sampling_workspace(
        &sampler,
        0.0,
        None,
        &layout(),
        &TokenFilter::All,
        u64::MAX,
        &context
    )
    .is_err());
    let logits = WorkspaceTensor::existing(layout(), &context).unwrap();
    for _ in 0..5 {
        Sampler::<WorkspaceSamplingBackend>::sample(&mut sampler, &logits, 0.0, None, &context)
            .unwrap();
    }
    assert_eq!(sampler.history_capacity(), 8);
    assert_eq!(sampler.clone().history_capacity(), 8);
    let quoted = quote_sampling_workspace(
        &sampler,
        0.0,
        None,
        &layout(),
        &TokenFilter::All,
        4,
        &context,
    )
    .unwrap();
    assert_eq!(quoted.final_history_bytes, 64);
    assert_eq!(sampler.history_len(), 5);
}

#[derive(Debug, Clone, Copy)]
enum TokenBacking {
    Allocated(u64),
    KeyAlias { possible_copy: Option<u64> },
}

#[derive(Debug)]
struct RetainedOutputFacts {
    token: TokenBacking,
    missing_at: Option<usize>,
    emissions: Rc<Cell<usize>>,
}

impl WorkspaceMechanisms for RetainedOutputFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let mut outputs = op
            .outputs
            .iter()
            .map(|output| Ok(WorkspaceOutputStorage::Allocate(output.bytes()?)))
            .collect::<Result<Vec<_>, Error>>()?;
        match &op.kind {
            WorkspaceOperationKind::Sampling(
                WorkspaceSamplingOperation::Greedy | WorkspaceSamplingOperation::Categorical,
            ) => {
                let index = self.emissions.get();
                self.emissions.set(index + 1);
                if self.missing_at == Some(index) {
                    return Ok(None);
                }
                outputs[0] = match self.token {
                    TokenBacking::Allocated(bytes) => WorkspaceOutputStorage::Allocate(bytes),
                    TokenBacking::KeyAlias { possible_copy } => {
                        assert!(matches!(
                            op.kind,
                            WorkspaceOperationKind::Sampling(
                                WorkspaceSamplingOperation::Categorical
                            )
                        ));
                        match possible_copy {
                            Some(bytes) => WorkspaceOutputStorage::AllocateOrAliasInputs {
                                bytes,
                                inputs: vec![1],
                            },
                            None => WorkspaceOutputStorage::AliasInput(1),
                        }
                    }
                };
            }
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SplitRandomKey) => {
                outputs[0] = WorkspaceOutputStorage::Allocate(128);
            }
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SelectRandomKey { .. })
            | WorkspaceOperationKind::Index { .. }
            | WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::ReadToken) => {
                outputs[0] = WorkspaceOutputStorage::AliasInput(0);
            }
            _ => {}
        }
        Ok(Some(WorkspaceOperationBound {
            outputs,
            scratch_bytes: 7,
            assumptions: "padded key/token allocations; indexing and token reads share backing"
                .into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 3,
            assumptions: "three bytes of disjoint host staging per operation".into(),
        }))
    }
}

fn plain_sampler() -> ConfiguredTextSampler {
    ConfiguredTextSampler::Standard(GenerationSampler::new().top_k(0).top_p(1.0).min_p(0.0))
}

fn retained_output_modes() -> Vec<(ConfiguredTextSampler, f32, TokenFilter)> {
    vec![
        (plain_sampler(), 0.0, TokenFilter::All),
        (
            ConfiguredTextSampler::Standard(
                GenerationSampler::new()
                    .top_k(5)
                    .top_p(0.8)
                    .min_p(0.2)
                    .penalties(1.2, 3, 0.1, 0.2),
            ),
            0.7,
            TokenFilter::allowed(vec![true; 37]).unwrap(),
        ),
        (
            ConfiguredTextSampler::MirostatV2(MirostatV2Sampler::new(5.0, 0.1).unwrap()),
            0.9,
            TokenFilter::All,
        ),
    ]
}

fn retained_output_quote(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    filter: &TokenFilter,
    steps: u64,
    token: TokenBacking,
    missing_at: Option<usize>,
) -> Result<(SamplingWorkspaceReport, usize), Error> {
    let emissions = Rc::new(Cell::new(0));
    let context = WorkspaceContext::new(RetainedOutputFacts {
        token,
        missing_at,
        emissions: emissions.clone(),
    });
    let random = (temperature > 0.0)
        .then(|| WorkspaceSamplingRandomState::from_seed(&context))
        .transpose()?;
    let report = quote_sampling_workspace(
        sampler,
        temperature,
        random.as_ref(),
        &layout(),
        filter,
        steps,
        &context,
    )?;
    Ok((report, emissions.get()))
}

#[test]
fn retained_outputs_use_native_capacity_for_every_configured_sampling_mode() {
    for (sampler, temperature, filter) in retained_output_modes() {
        for steps in [0, 1, 5] {
            let quote = |capacity| {
                retained_output_quote(
                    &sampler,
                    temperature,
                    &filter,
                    steps,
                    TokenBacking::Allocated(capacity),
                    None,
                )
                .unwrap()
            };
            let (compact, compact_emissions) = quote(4);
            let (padded, padded_emissions) = quote(64);
            assert_eq!(compact_emissions, steps as usize);
            assert_eq!(padded_emissions, steps as usize);
            assert_eq!(padded.first_gap, None);
            assert_eq!(
                padded.tensor_peak_bytes.unwrap() - compact.tensor_peak_bytes.unwrap(),
                steps * (64 - 4),
                "each retained output keeps its complete independent capacity"
            );
            assert_eq!(
                padded.peak.bytes().unwrap() - compact.peak.bytes().unwrap(),
                steps * (64 - 4)
            );
            assert_eq!(padded.host_peak_bytes, compact.host_peak_bytes);
            assert_eq!(padded.final_history_bytes, compact.final_history_bytes);
            assert_eq!(sampler.history_len(), 0);
            if temperature == 0.0 {
                assert_eq!(
                    padded.tensor_peak_bytes,
                    Some(if steps == 0 { 0 } else { steps * 64 + 14 }),
                    "scalar read aliases the emitted token; only current scratch overlaps"
                );
            }
            if steps == 0 {
                assert_eq!(padded.final_history_bytes, 0);
                assert_eq!(
                    padded.tensor_peak_bytes,
                    Some(if temperature == 0.0 { 0 } else { 8 + 7 })
                );
            }
        }
    }
}

#[test]
fn retained_output_aliases_share_key_backing_and_preserve_possible_copy_roots() {
    for possible_copy in [None, Some(64)] {
        for steps in [0, 1, 2, 4] {
            let (report, emitted) = retained_output_quote(
                &plain_sampler(),
                0.7,
                &TokenFilter::All,
                steps,
                TokenBacking::KeyAlias { possible_copy },
                None,
            )
            .unwrap();
            // Each split has 128-byte backing shared by both key views and the
            // emitted token. Old splits survive through those outputs. A
            // possible token copy adds its own capacity without duplicating the
            // shared split. Only the first step still overlaps the initial key.
            let expected = if steps == 0 {
                8 + 7
            } else {
                steps * (128 + possible_copy.unwrap_or(0))
                    + 37 * 4
                    + 6 * 7
                    + if steps == 1 { 8 } else { 0 }
            };
            assert_eq!(report.tensor_peak_bytes, Some(expected));
            assert_eq!(emitted, steps as usize);
            assert_eq!(report.first_gap, None);
        }
    }
}

#[test]
fn missing_retained_output_capacity_stays_unknown_after_later_known_outputs() {
    for (sampler, temperature, filter) in retained_output_modes() {
        let (report, emitted) = retained_output_quote(
            &sampler,
            temperature,
            &filter,
            5,
            TokenBacking::Allocated(64),
            Some(1),
        )
        .unwrap();
        assert_eq!(emitted, 5);
        assert_eq!(report.first_gap, Some(1));
        assert_eq!(report.tensor_peak_bytes, None);
        assert_eq!(report.peak.bytes(), None);
        assert!(report.host_peak_bytes.is_some());
        assert_eq!(report.final_history_bytes, 32);
    }
}

#[test]
fn retained_output_capacity_overflow_is_reported_across_spans() {
    let quote = |steps| {
        retained_output_quote(
            &plain_sampler(),
            0.0,
            &TokenFilter::All,
            steps,
            TokenBacking::Allocated(u64::MAX / 2 + 1),
            None,
        )
    };
    assert!(quote(1).unwrap().0.peak.bytes().is_some());
    assert!(quote(2).is_err());
}


#[test]
fn prepared_logit_policy_matches_actual_policy_and_refuses_changed_history_before_work() {
    use crate::generation::{LogitProgramError, PreparedLogitPolicyError};
    use crate::{DefaultSampler, SpeculativeSampler};
    let history = [3, 5, 3, 7];
    let mut standard = GenerationSampler::default();
    standard.repeat_penalty = 1.1;
    standard.frequency_penalty = 0.3;
    standard.repeat_last_n = 3;
    let mut policies = [
        ConfiguredTextSampler::Standard(standard),
        ConfiguredTextSampler::MirostatV2(MirostatV2Sampler::default()),
    ];
    for policy in &mut policies {
        let prepared = <ConfiguredTextSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
            ::prepared_logit_policy(policy).unwrap().bind(0.7, history.len()).unwrap();
        prepared.validate_layout(&[1, 37]).unwrap();
        let trace = |prepared_path: bool, policy: &mut ConfiguredTextSampler| {
            let (context, calls) = setup(false);
            let source = WorkspaceTensor::existing(layout(), &context).unwrap();
            context.begin_span();
            let output = if prepared_path {
                prepared.process::<WorkspaceSamplingBackend>(&source, &history, &context).unwrap()
            } else {
                <ConfiguredTextSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
                    ::process_logits(policy, &source, 0.7, &history, &context).unwrap()
            };
            let report = context.finish_report(&[output]).unwrap();
            let operations = report.operations.iter().map(|op| op.kind.clone()).collect::<Vec<_>>();
            let calls = calls.borrow().clone();
            (operations, calls, report.tensor_handle_clones)
        };
        let ordinary = trace(false, policy);
        let projected = trace(true, policy);
        // Operation declarations include families without structural PartialEq.
        // Compare the actual emitted scalar specifications, backend calls and
        // retained handle census from the two executions.
        assert_eq!(format!("{:?}", ordinary.0), format!("{:?}", projected.0));
        assert_eq!(ordinary.1, projected.1);
        assert_eq!(ordinary.2, projected.2);
        assert!(!projected.0.is_empty());
        let (context, calls) = setup(false);
        let source = WorkspaceTensor::existing(layout(), &context).unwrap();
        context.begin_span();
        let refused = prepared.process::<WorkspaceSamplingBackend>(&source, &history[..3], &context);
        assert!(matches!(refused, Err(LogitProgramError::Policy(PreparedLogitPolicyError::History))));
        assert!(calls.borrow().is_empty());
        assert!(context.finish_report(&[]).unwrap().operations.is_empty());
    }
    let identity = <DefaultSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
        ::prepared_logit_policy(&DefaultSampler).unwrap().bind(-0.0, history.len()).unwrap();
    let (context, _) = setup(false);
    let source = WorkspaceTensor::existing(layout(), &context).unwrap();
    context.begin_span();
    let output = identity.process::<WorkspaceSamplingBackend>(&source, &history, &context).unwrap();
    assert!(context.finish_report(&[output]).unwrap().operations.is_empty());

    struct Callback;
    impl SpeculativeSampler<WorkspaceSamplingBackend> for Callback {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
        fn process_logits(&mut self, _: &WorkspaceTensor, _: f32, _: &[u32], _: &WorkspaceContext)
            -> Result<WorkspaceTensor, Error> { panic!("unqualified callback must not be invoked") }
    }
    assert!(Callback.prepared_logit_policy().is_none());
}

#[test]
fn prepared_greedy_choice_is_exact_and_unknown_or_adaptive_commit_stays_closed() {
    use crate::{DefaultSampler, SpeculativeSampler};
    use crate::generation::PreparedGreedyError;
    let mut standard = GenerationSampler::default().with_generated_tokens([2, 3]);
    let original_history = standard.generated_tokens().to_vec();
    let choice = <GenerationSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
        ::prepared_greedy_policy(&standard).unwrap().greedy(-0.0).unwrap();
    use crate::speculative::numerical::{SpeculativeNumericalProgram, SpeculativeNumericalKind, SpeculativeNumericalError};
    assert!(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Greedy(choice), &[1,1,37]).is_ok());
    assert_eq!(SpeculativeNumericalProgram::new(SpeculativeNumericalKind::Greedy(choice), &[2,37]),
        Err(SpeculativeNumericalError::GreedyRows));
    let (context, _) = setup(false);
    let source = WorkspaceTensor::existing(layout(), &context).unwrap();
    context.begin_span();
    let token = choice.construct::<WorkspaceSamplingBackend>(&source, &context).unwrap();
    let report = context.finish_report(&[token]).unwrap();
    assert_eq!(report.operations.len(), 1);
    assert!(matches!(report.operations[0].kind,
        WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Greedy)));
    assert_eq!(report.operations[0].outputs[0].shape(), &[1, 1]);
    assert_eq!(report.operations[0].outputs[0].dtype(), WorkspaceDtype::Uint32);
    <GenerationSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
        ::prepared_greedy_policy(&standard).unwrap().commit_without_mutation();
    <GenerationSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
        ::commit_token(&mut standard, &source, 3, &context).unwrap();
    assert_eq!(standard.generated_tokens(), original_history);
    for temperature in [0.1, -0.1, f32::NAN] {
        assert!(matches!(<DefaultSampler as SpeculativeSampler<WorkspaceSamplingBackend>>
            ::prepared_greedy_policy(&DefaultSampler).unwrap().greedy(temperature),
            Err(PreparedGreedyError::Temperature)));
    }
    assert!(<MirostatV2Sampler as SpeculativeSampler<WorkspaceSamplingBackend>>
        ::prepared_greedy_policy(&MirostatV2Sampler::default()).is_none());
    struct Override;
    impl SpeculativeSampler<WorkspaceSamplingBackend> for Override {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
        fn process_logits(&mut self, _: &WorkspaceTensor, _: f32, _: &[u32], _: &WorkspaceContext)
            -> Result<WorkspaceTensor, Error> { panic!("unqualified callback") }
        fn commit_token(&mut self, _: &WorkspaceTensor, _: u32, _: &WorkspaceContext)
            -> Result<(), Error> { panic!("unqualified commit") }
    }
    assert!(Override.prepared_greedy_policy().is_none());
}

#[test]
fn prepared_random_phases_keep_full_split_tables_and_both_categorical_roots() {
    use crate::{DefaultSampler, SpeculativeSampler};
    use crate::speculative::numerical::{SpeculativeNumericalKind as Kind, SpeculativeNumericalProgram as Program};
    let (context, _) = setup(false);
    context.begin_span();
    let key = WorkspaceSamplingRandomState::from_seed(&context).unwrap().into_key();
    let report = context.finish_report(&[key.clone()]).unwrap();
    assert_eq!(report.operations.len(), 1);
    assert!(matches!(report.operations[0].kind, WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::CreateRandomKey)));
    assert_eq!(Program::new(Kind::CreateKey { seed: 71 }, &[2]).unwrap().source_count(), 0);
    for position in [0, 1, 7, 128] {
        context.begin_span();
        let selected = WorkspaceSamplingRandomState::key_at(&key, position, &context).unwrap();
        let report = context.finish_report(&[selected]).unwrap();
        assert_eq!(report.operations[0].outputs[0].shape(), &[position as i32 + 1, 2]);
        assert!(matches!(report.operations[0].kind, WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SplitRandomKey)));
        assert_eq!(report.operations.len(), 2);
        assert!(matches!(report.operations[1].kind, WorkspaceOperationKind::Sampling(
            WorkspaceSamplingOperation::SelectRandomKey { index }) if index == position));
        assert_eq!(Program::new(Kind::KeyAt { position }, &[2]).unwrap().source_count(), 1);
    }
    assert!(Program::new(Kind::KeyAt { position: i32::MAX as u32 }, &[2]).is_err());
    let scores = WorkspaceTensor::existing(layout(), &context).unwrap();
    let choice = <DefaultSampler as SpeculativeSampler<WorkspaceSamplingBackend>>::prepared_categorical_policy(&DefaultSampler)
        .unwrap().bind(0.7).unwrap();
    assert_eq!(Program::new(Kind::Categorical(choice), scores.shape()).unwrap().source_count(), 2);
    context.begin_span();
    let mut random = WorkspaceSamplingRandomState::from_key(key.clone()).unwrap();
    let token = choice.construct::<WorkspaceSamplingBackend>(&scores, &mut random, &context).unwrap();
    let advanced = random.into_key();
    assert_eq!(advanced.shape(), &[2]);
    assert_eq!(token.shape(), &[1,1]);
    let report = context.finish_report(&[token,advanced]).unwrap();
    assert!(matches!(report.operations[0].kind, WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SplitRandomKey)));
    assert!(matches!(report.operations.last().unwrap().kind, WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Categorical)));
    assert_eq!(report.operations[0].outputs[0].shape(), &[2,2]);
    let (ordinary_context, ordinary_calls) = setup(false);
    let ordinary_scores = WorkspaceTensor::existing(layout(), &ordinary_context).unwrap();
    let ordinary_key = WorkspaceSamplingRandomState::from_seed(&ordinary_context).unwrap();
    ordinary_calls.borrow_mut().clear();
    let mut ordinary_key = ordinary_key;
    ordinary_context.begin_span();
    let token = WorkspaceSamplingBackend::sample_processed(&ordinary_scores,0.7,Some(&mut ordinary_key),&ordinary_context).unwrap();
    let ordinary = ordinary_context.finish_report(&[token,ordinary_key.into_key()]).unwrap();
    assert_eq!(format!("{:?}",report.operations.iter().map(|op|&op.kind).collect::<Vec<_>>()),
        format!("{:?}",ordinary.operations.iter().map(|op|&op.kind).collect::<Vec<_>>()));
}

#[test]
fn prepared_uniform_draw_keeps_sequential_key_and_scalar_roots() {
    use crate::speculative::numerical::{SpeculativeNumericalKind as Kind, SpeculativeNumericalProgram as Program};
    let (context, _) = setup(false);
    let key = WorkspaceTensor::existing(
        context.layout(&[2], WorkspaceDtype::Uint32).unwrap(), &context).unwrap();
    let mut random = WorkspaceSamplingRandomState::from_key(key).unwrap();
    for _ in 0..2 {
        context.begin_span();
        let draw = random.uniform_unit_interval(&context).unwrap();
        assert_eq!(draw.shape(), &[1]);
        assert_eq!(draw.layout().dtype(), WorkspaceDtype::Float32);
        assert_eq!(random.key().shape(), &[2]);
        let report = context.finish_report(&[draw, random.key().clone()]).unwrap();
        assert_eq!(report.operations.len(), 4);
        assert!(matches!(report.operations[0].kind,
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SplitRandomKey)));
        assert_eq!(report.operations[0].outputs[0].shape(), &[2,2]);
        assert!(matches!(report.operations[3].kind,
            WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::UniformUnitInterval)));
        assert_eq!(report.operations[3].inputs[0].shape(), &[2]);
    }
    assert_eq!(Program::new(Kind::UniformUnitInterval, &[2]).unwrap().source_count(), 1);
    for shape in [&[][..], &[1][..], &[4][..], &[1,2][..]] {
        assert!(Program::new(Kind::UniformUnitInterval, shape).is_err());
    }
}
