use super::*;
mod ordinary_output;
use eredu_core::{GenerationSequenceConsumerLayout, TextGenerationDriver};
fn layout() -> GenerationSequenceConsumerLayout {
    // Neutral actual fixture types; no claim that this is the facade's layout.
    GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
}

#[test]
fn retained_provider_copy_has_independent_budget_state_and_output_lifetime() {
    use crate::execution_control::{PreparedTextHostCopy, TextHostCopyError};

    let (mut actual, state) = runtime(Mode {
        copy_headroom: 65_536,
        ..Mode::default()
    });
    let mut sequence = with_consumer(&mut actual, 1, None, &layout())
        .unwrap()
        .prepare_storage()
        .unwrap();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let pool = state.borrow().pool.clone();
    let used = pool.used_bytes().unwrap();
    let bytes = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            u64::MAX,
        )
        .unwrap()
        .storage_bytes()
        .unwrap();
    assert!(bytes > 0);
    let short = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            used + bytes - 1,
        )
        .unwrap()
        .copy(bytes);
    assert!(matches!(
        short,
        Err(TextHostCopyError::Admission(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(sequence.tokens(), &[5]);
    assert_eq!(sequence.finish_reason(), None);

    let mut copied = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            used + bytes,
        )
        .unwrap()
        .copy(bytes)
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), used + bytes);
    assert_ne!(sequence.tokens().as_ptr(), copied.tokens().as_ptr());
    assert_eq!(copied.tokens(), &[5]);
    assert_eq!(
        copied
            .commit(7, TokenTerminalSignals::default())
            .unwrap()
            .finish_reason,
        Some(eredu_core::FinishReason::Eos)
    );
    assert_eq!(
        sequence
            .commit(6, TokenTerminalSignals::default())
            .unwrap()
            .finish_reason,
        Some(eredu_core::FinishReason::MaxTokens)
    );
    drop(sequence);
    retire_request(&state);
    drop(actual);
    assert_eq!(pool.used_bytes().unwrap(), bytes);

    // Resume's logical branch lease follows the actual independent provider,
    // including aliases that survive both its cursor and its machine H handle.
    use crate::execution_control::{PendingSnapshotResumeRetention, SnapshotBudget};
    use eredu_core::execution_control::{SnapshotEstimate, SnapshotLimits, SnapshotResourceKind};
    let logical = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0,
        max_branches: 1,
        retained_bytes: 40,
        cumulative_copy_bytes: 120,
    });
    let cost = SnapshotEstimate {
        retained_bytes: 40,
        copy_bytes: 40,
    };
    let reserve = || {
        PendingSnapshotResumeRetention::reserve(&logical, SnapshotResourceKind::Branch, cost)
            .unwrap()
    };
    let resume_bytes = pool
        .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            u64::MAX,
        )
        .unwrap()
        .storage_bytes()
        .unwrap();
    let short = pool
        .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            bytes + resume_bytes - 1,
        )
        .unwrap()
        .copy_original_resume(40, reserve());
    assert!(matches!(
        short,
        Err(TextHostCopyError::Admission(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(logical.usage().branches, 0);
    assert_eq!(logical.usage().retained_bytes, 0);
    assert_eq!(logical.usage().cumulative_copy_bytes, 40);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    for host_last in [true, false] {
        let (branch, host) = pool
            .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
                &copied,
                bytes + resume_bytes,
            )
            .unwrap()
            .copy_original_resume(40, reserve())
            .unwrap();
        let ids = branch.into_token_ids();
        assert_eq!(&*ids, &[5, 7]);
        if host_last {
            drop(ids);
            assert_eq!(logical.usage().branches, 1);
            assert_eq!(pool.used_bytes().unwrap(), bytes + resume_bytes);
            drop(host);
        } else {
            let alias = ids.clone();
            drop(ids);
            drop(host);
            assert_eq!(logical.usage().branches, 1);
            assert_eq!(logical.usage().retained_bytes, 40);
            assert!(
                PendingSnapshotResumeRetention::reserve(
                    &logical,
                    SnapshotResourceKind::Branch,
                    cost
                )
                .is_err()
            );
            std::thread::spawn(move || drop(alias)).join().unwrap();
        }
        assert_eq!(logical.usage().branches, 0);
        assert_eq!(logical.usage().retained_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
    }
    assert_eq!(logical.usage().cumulative_copy_bytes, 120);

    let plan = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            u64::MAX,
        )
        .unwrap();
    let later_bytes = plan.storage_bytes().unwrap();
    let later = plan.copy(later_bytes).unwrap();
    assert_eq!(later.tokens(), &[5, 7]);
    assert_eq!(later.finish_reason(), Some(eredu_core::FinishReason::Eos));
    drop(copied);
    let ids = later.into_token_ids();
    let alias = ids.clone();
    drop(ids);
    assert_eq!(&*alias, &[5, 7]);
    assert_eq!(pool.used_bytes().unwrap(), later_bytes);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn semantic_and_cursor_copy_share_exact_admission_and_nonrefundable_branch_custody() {
    use crate::execution_control::{PreparedTextHostCopy, TextHostCopyError, PendingSnapshotResumeRetention, SnapshotBudget};
    use crate::working_memory::PreparedSemanticSource;
    use eredu_core::{FinishReason, generation::SemanticEvent, execution_control::{SnapshotEstimate, SnapshotLimits, SnapshotResourceKind}};
    use eredu_text::{tokenizer_storage::TokenizerPlan, stop_storage::StopCompilePlan};
    let (mut actual, state) = runtime(Mode { copy_headroom: 64 << 20, ..Mode::default() });
    let mut sequence = with_consumer(&mut actual, 1, None, &layout()).unwrap().prepare_storage().unwrap();
    sequence.commit(0, TokenTerminalSignals::default()).unwrap();
    let pool = state.borrow().pool.clone();
    let execution = state.borrow().active.as_ref().unwrap().request.request()
        .memory_reservation().unwrap().0.execution.clone();
    let input = br#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel",
        "add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],
        "model":{"type":"BPE","vocab":{"h":0,"i":1},"merges":[]}}"#;
    let tokenizer = pool.compile_tokenizer(TokenizerPlan::prepare_json(input).unwrap()).unwrap();
    let stops = pool.compile_stop_source(StopCompilePlan::prepare_refs(&[]).unwrap()).unwrap();
    let prepared = PreparedSemanticSource::new(&tokenizer, &execution, pool.effective_capacity().unwrap()).unwrap();
    let mut semantic = prepared.prepare(&stops, 2, std::num::NonZeroUsize::MIN, true).unwrap();
    semantic.push_token(0).unwrap();
    let foreign = PreparedSemanticSource::new(&tokenizer, &InferenceExecutionIdentity::default(), pool.effective_capacity().unwrap()).unwrap()
        .prepare(&stops, 2, std::num::NonZeroUsize::MIN, true).unwrap();
    assert!(matches!(pool.prepare_semantic_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError, _>(
        &sequence, &foreign, u64::MAX, ()), Err(WorkingMemoryError::IdentityMismatch)));
    drop(foreign);
    let used = pool.used_bytes().unwrap();
    let bytes = pool.prepare_semantic_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError, _>(
        &sequence, &semantic, u64::MAX, ()).unwrap().storage_bytes().unwrap();
    let logical = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0, max_branches: 1, retained_bytes: bytes,
        cumulative_copy_bytes: bytes * 3,
    });
    let reserve = || PendingSnapshotResumeRetention::reserve(&logical, SnapshotResourceKind::Branch,
        SnapshotEstimate { retained_bytes: bytes, copy_bytes: bytes }).unwrap();
    let short = pool.prepare_semantic_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError, _>(
        &sequence, &semantic, used + bytes - 1, ()).unwrap().copy_original_resume(bytes, reserve());
    assert!(matches!(short, Err(TextHostCopyError::Admission(WorkingMemoryError::BudgetExceeded { .. }))));
    assert_eq!(logical.usage().cumulative_copy_bytes, bytes);
    assert_eq!(logical.usage().branches, 0);
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(sequence.tokens(), &[0]);
    let exact = pool.prepare_semantic_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError, _>(
        &sequence, &semantic, used + bytes, ()).unwrap().copy_original_resume(bytes, reserve()).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), used + bytes);
    drop(exact);
    assert_eq!(pool.used_bytes().unwrap(), used);
    // Retained accounts keep their shared-domain ceiling. Continuing generation
    // needs the original execution capacity, beyond the exact copy-only fit.
    let ((mut copied, mut semantic_copy, ()), host) = pool.prepare_semantic_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError, _>(
        &sequence, &semantic, pool.effective_capacity().unwrap(), ()).unwrap().copy_original_resume(bytes, reserve()).unwrap();
    let mut original_events = Vec::new();
    let mut copied_events = Vec::new();
    semantic.publish_events(&mut |event| original_events.push(event));
    semantic_copy.publish_events(&mut |event| copied_events.push(event));
    assert_eq!(copied_events, original_events);
    assert_eq!(copied_events, [SemanticEvent::TextDelta("h".into())]);
    semantic.push_token(0).unwrap();
    semantic_copy.push_token(1).unwrap();
    semantic.finish(FinishReason::MaxTokens).unwrap();
    semantic_copy.finish(FinishReason::MaxTokens).unwrap();
    semantic.publish_events(&mut |event| original_events.push(event));
    semantic_copy.publish_events(&mut |event| copied_events.push(event));
    let visible = |events: &[SemanticEvent]| events.iter().filter_map(|event| match event {
        SemanticEvent::TextDelta(text) => Some(text.as_str()), _ => None,
    }).collect::<String>();
    assert_eq!(visible(&original_events), "hh");
    assert_eq!(visible(&copied_events), "hi");
    sequence.commit(0, TokenTerminalSignals::default()).unwrap();
    copied.commit(1, TokenTerminalSignals::default()).unwrap();
    assert_eq!(sequence.tokens(), &[0, 0]);
    assert_eq!(copied.tokens(), &[0, 1]);
    let ids = copied.into_token_ids();
    drop(host);
    drop(ids);
    assert_eq!(logical.usage().branches, 1, "semantic copy retains the same branch lease");
    drop(semantic_copy);
    assert_eq!(logical.usage().branches, 0);
    assert_eq!(logical.usage().retained_bytes, 0);
    assert_eq!(logical.usage().cumulative_copy_bytes, bytes * 3);
    // A journal contributes its actual physical producer separately from logical
    // retention, while sharing the parser/cursor account and one resume lease.
    use crate::execution_control::PreparedTextHostJournal;
    let calls = std::cell::Cell::new(0);
    let journal = |fail| CopyJournal { source: b"journal", copies: &calls, fail };
    let plan = |capacity, fail| pool.prepare_semantic_resume_provider_for_test::<
        RetainedGenerationSequence, TextHostCopyError, _>(
            &sequence, &semantic, capacity, journal(fail)).unwrap();
    let logical_bytes = plan(u64::MAX, false).storage_bytes().unwrap();
    let physical_bytes = logical_bytes + journal(false).preparation_bytes().unwrap() as u64
        - journal(false).storage_bytes().unwrap();
    assert!(physical_bytes > logical_bytes);
    let journal_budget = SnapshotBudget::prepare(SnapshotLimits {
        max_snapshots: 0, max_branches: 1, retained_bytes: logical_bytes,
        cumulative_copy_bytes: logical_bytes * 3,
    }, prepared.metadata_funding()).unwrap();
    let reserve_journal = || PendingSnapshotResumeRetention::reserve(&journal_budget,
        SnapshotResourceKind::Branch, SnapshotEstimate {
            retained_bytes: logical_bytes, copy_bytes: logical_bytes,
        }).unwrap();
    let before = pool.used_bytes().unwrap();
    assert!(matches!(plan(before + physical_bytes - 1, false)
        .copy_original_resume(logical_bytes, reserve_journal()),
        Err(TextHostCopyError::Admission(WorkingMemoryError::BudgetExceeded { .. }))));
    assert_eq!(calls.get(), 0, "refusal precedes the journal producer");
    assert_eq!(journal_budget.usage().branches, 0);
    let ((cursor, parser, journal_copy), host) = plan(u64::MAX, false)
        .copy_original_resume(logical_bytes, reserve_journal()).unwrap();
    assert_eq!(journal_copy.bytes, b"journal");
    assert_eq!(pool.used_bytes().unwrap(), before + physical_bytes);
    drop((cursor, parser, host));
    assert_eq!(journal_budget.usage().branches, 1, "escaped journal retains same branch lease");
    drop(journal_copy);
    assert_eq!(journal_budget.usage().branches, 0);
    assert_eq!(pool.used_bytes().unwrap(), before);
    let error = plan(u64::MAX, true).copy_original_resume(logical_bytes, reserve_journal()).unwrap_err();
    assert_eq!(calls.get(), 2);
    assert_eq!(journal_budget.usage().branches, 1, "partial journal error retains destination");
    assert_eq!(pool.used_bytes().unwrap(), before + physical_bytes);
    drop(error);
    assert_eq!(journal_budget.usage().branches, 0);
    assert_eq!(journal_budget.usage().cumulative_copy_bytes, logical_bytes * 3);
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop(journal_budget);
    drop((sequence, semantic, prepared, tokenizer, stops, original_events, copied_events));
    retire_request(&state);
    drop(actual);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

struct CopyJournal<'a> {
    source: &'a [u8],
    copies: &'a std::cell::Cell<usize>,
    fail: bool,
}
#[derive(Debug)]
struct CopiedJournal {
    bytes: Vec<u8>,
    host: eredu_core::HostPreparationAuthority,
}
impl std::fmt::Display for CopiedJournal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("test journal rejected after destination construction")
    }
}
impl std::error::Error for CopiedJournal {}
impl crate::execution_control::PreparedTextHostJournal for CopyJournal<'_> {
    type Copied = CopiedJournal;
    fn storage_bytes(&self) -> Option<u64> { Some(self.source.len() as u64) }
    fn preparation_bytes(&self) -> Option<usize> {
        [
            self.source.len(), std::mem::size_of::<Self>(),
            std::mem::size_of::<CopiedJournal>(), std::mem::size_of::<Vec<u8>>(),
            std::mem::size_of::<std::collections::TryReserveError>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<CopiedJournal>()?,
        ].into_iter().try_fold(0usize, usize::checked_add)
    }
    fn copy(self, _: u64, host: &eredu_core::HostPreparationAuthority)
        -> Result<Self::Copied, crate::execution_control::TextHostCopyError>
    {
        self.copies.set(self.copies.get() + 1);
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(self.source.len()).expect("small journal fixture");
        bytes.extend_from_slice(self.source);
        let copied = CopiedJournal { bytes, host: host.clone() };
        if self.fail {
            Err(crate::execution_control::TextHostCopyError::Source(
                eredu_core::BackendFailure::from_error(copied)))
        } else { Ok(copied) }
    }
}

fn with_consumer(
    runtime: &mut ModelRuntime<Backend>,
    route: usize,
    options: Option<TextPreparationOptions>,
    consumer: &GenerationSequenceConsumerLayout,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let request = GenerationSequenceRequest::new(2, &[9, 7, 9]).with_consumer(consumer);
    let input = TextGenerationInput::TokenIds(vec![2]);
    let map = |e| match e {
        eredu_core::ControlledTextGenerationError::Preparation(e) => e,
        _ => panic!("unexpected preparation failure"),
    };
    let sequence = match route {
        0 => {
            let mut run = TextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(2),
                TokenFilter::All,
                options,
                request,
            )?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            sequence
        }
        1 => {
            let mut run = ControlledTextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(2),
                Controller,
                options,
                request,
            )
            .map_err(map)?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            sequence
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut run = driver
                .start_input_with_sequence(input, config(2), Controller, options, request)
                .map_err(map)?;
            let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
            assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
            sequence
        }
    };
    Ok(sequence)
}

#[test]
fn original_consumer_controls_add_exact_r_and_preserve_admission_phases_for_all_drivers() {
    let consumer = layout();
    for route in 0..3 {
        for capture in [false, true] {
            let mode = Mode {
                capture,
                ..Mode::default()
            };
            let options = || {
                capture.then(|| TextPreparationOptions {
                    interventions: None, capture: Some(capture_source()),
                })
            };
            let (mut baseline, base) = runtime(mode);
            let ordinary = extract(&mut baseline, 2, &[9, 7, 9], false, false, options()).unwrap();
            let base_r = base.borrow().r;
            drop(ordinary);
            retire_request(&base);
            drop(baseline);
            let (mut actual, state) = runtime(mode);
            let sequence = with_consumer(&mut actual, route, options(), &consumer).unwrap();
            assert_eq!(sequence.consumer_layout(), Some(&consumer));
            let facts = state.borrow();
            assert_eq!(facts.r - base_r, consumer.retention_peak_bytes() as u64);
            assert_eq!(facts.held, facts.p + 51 + facts.r);
            assert_eq!(
                facts.order,
                ["admit", "bind", "extract", "prompt", "sampling"]
            );
            assert_eq!(
                facts.votes,
                if capture {
                    vec![
                        (Stage::Admission, Status::Ready),
                        (Stage::Prompt, Status::Ready),
                        (Stage::Sampling, Status::Ready),
                        (Stage::Instrumentation, Status::Ready),
                    ]
                } else {
                    vec![
                        (Stage::Admission, Status::Ready),
                        (Stage::Prompt, Status::Ready),
                        (Stage::Sampling, Status::Ready),
                    ]
                }
            );
            assert!(
                facts
                    .active
                    .as_ref()
                    .unwrap()
                    .owner
                    .borrow_mut()
                    .take_generation_sequence_bank()
                    .is_none()
            );
            drop(facts);
            let (pool, held) = retire_request(&state);
            drop(actual);
            assert_eq!(pool.used_bytes().unwrap(), held);
            assert!(matches!(
                pool.pin_registered_storage([(1u32, 64)]),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            drop(sequence);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn original_consumer_exact_minus_one_rejects_original_width_one_before_binding_or_work() {
    let consumer = layout();
    for capture in [false, true] {
        let (mut actual, state) = runtime(Mode {
            capture,
            short: true,
            ..Mode::default()
        });
        let options = capture.then(|| TextPreparationOptions {
            interventions: None, capture: Some(capture_source()),
        });
        let error = with_consumer(&mut actual, 0, options, &consumer).unwrap_err();
        // The harness's constant fixture has only width one, so there is no stale
        // geometry return on fallback and this must be the actual budget cause.
        let mut source: &(dyn std::error::Error + 'static) = &error;
        let mut budget = false;
        loop {
            if let Some(WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: available,
            }) = source.downcast_ref::<WorkingMemoryError>()
            {
                assert_eq!(*required, *available + 1);
                budget = true;
            }
            match source.source() {
                Some(next) => source = next,
                None => break,
            }
        }
        assert!(
            budget,
            "exact minus one must preserve its typed budget cause: {error:?}"
        );
        let facts = state.borrow();
        assert_eq!(facts.order, ["admit"]);
        assert_eq!(facts.votes, [(Stage::Admission, Status::Failed)]);
        assert_eq!(facts.pool.used_bytes().unwrap(), 64);
        assert!(facts.active.is_none());
    }
}

#[test]
fn original_consumer_result_and_partial_iterator_keep_same_explicit_source_transition() {
    let consumer = layout();
    let (mut actual, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let sequence = with_consumer(&mut actual, 1, None, &consumer).unwrap();
    let mut sequence = sequence.prepare_storage().unwrap();
    let pointer = sequence.tokens().as_ptr();
    sequence.commit(4, TokenTerminalSignals::default()).unwrap();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let (pool, held) = retire_request(&state);
    drop(actual);
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    let tokens = sequence.into_token_ids();
    assert_eq!(tokens.as_ptr(), pointer);
    assert_eq!(tokens.as_slice(), [4, 5]);
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let alias = tokens.clone();
    let mut iter = tokens.into_iter();
    assert_eq!(iter.next(), Some(4));
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(iter);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
