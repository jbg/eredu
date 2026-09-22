//! Resume-time sampler changes use the public recorded branch options. Direct
//! prepared-chat restore settings deliberately preserve the saved sampler.
use super::*;
use eredu_core::{FinishReason, SamplingOverride, SamplingStateFacts};
use std::ops::ControlFlow;

struct ChildStart {
    change: Option<SamplingOverride>,
    before: SamplingStateFacts,
    after: SamplingStateFacts,
    parent: String,
    child: String,
    ids: Vec<u32>,
    semantics: Vec<SemanticEvent>,
}

#[derive(Default)]
struct Trace {
    semantics: Vec<SemanticEvent>,
    child: Option<ChildStart>,
}
impl Trace {
    fn accept(&mut self, record: ControlledGenerationRecord) -> ControlFlow<()> {
        match &record.event {
            ControlledGenerationEvent::BranchStarted {
                lineage,
                inherited_token_ids,
                inherited_semantics,
                ..
            } => {
                assert!(self.child.is_none(), "one prospective child sampler");
                self.child = Some(ChildStart {
                    change: lineage.sampling_override,
                    before: lineage.sampling_before,
                    after: lineage.sampling_after,
                    parent: lineage.parent.output.run_id.clone(),
                    child: lineage.run_id.clone(),
                    ids: inherited_token_ids.clone(),
                    semantics: inherited_semantics.clone(),
                });
            }
            ControlledGenerationEvent::Progress {
                event: ObservedGenerationEvent::Semantic { event, .. },
            } => self.semantics.push(event.clone()),
            _ => {}
        }
        // Keep only caller-owned scalar/semantic diagnostics, so delivered
        // snapshot records do not keep a snapshot slot occupied after its drop.
        ControlFlow::Continue(())
    }
}

pub(super) fn check_recorded_child(
    model: &mut LoadedModel<eredu_backend_mlx::backend::MlxBackend<'static>>,
    chat: &eredu::runtime::chat::PreparedChat,
    settings: PreparedChatGenerationSettings,
    script: &[u32],
    expected_ids: &[u32],
    expected_finish: FinishReason,
    expected_events: &[SemanticEvent],
) {
    let trace_limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1 << 20,
    };
    let mut trace = Trace::default();
    let mut session = model
        .start_controlled_chat(
            PreparedChatRequest::new(chat, settings),
            trace_limits,
            Default::default(),
            |record| trace.accept(record),
        )
        .unwrap_or_else(fail)
        .unwrap();
    for _ in 0..3 {
        session
            .step(|record| trace.accept(record))
            .unwrap_or_else(fail);
    }
    assert_eq!(session.token_ids(), &script[..3]);
    let prefix = trace.semantics.clone();
    let parent_sampling = session.sampling_state().unwrap_or_else(fail);
    assert_eq!(parent_sampling.temperature, 0.0);
    let parent_run = session
        .output_checkpoint()
        .unwrap_or_else(fail)
        .run_id
        .clone();
    session.force_next_token(script[3]).unwrap_or_else(fail);
    session
        .enable_snapshots(
            SnapshotLimits {
                max_snapshots: 1,
                max_branches: 1,
                retained_bytes: CAPACITY,
                cumulative_copy_bytes: CAPACITY * 8,
            },
            native_limits(CAPACITY),
            WorkspaceCopyLimits::new(native_limits(CAPACITY)),
        )
        .unwrap_or_else(fail);
    let saved = session
        .snapshot(|record| trace.accept(record))
        .unwrap_or_else(fail);
    let before_fork = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    let change = SamplingOverride {
        temperature: Some(0.5),
        reseed: Some(711),
    };
    let mut parent = session
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits,
                capture_limits: None,
                sampling: Some(change),
                intervention: None,
            },
            |record| trace.accept(record),
        )
        .unwrap_or_else(fail);
    let lineage = trace.child.as_ref().expect("child sampler provenance");
    assert_eq!(lineage.change, Some(change));
    assert_eq!(lineage.before, parent_sampling);
    assert_eq!(lineage.after.temperature, 0.5);
    assert!(lineage.after.has_rng);
    assert_eq!(lineage.parent, parent_run);
    assert_ne!(lineage.child, parent_run);
    assert_eq!(lineage.ids, script[..3]);
    assert_eq!(lineage.semantics, prefix);
    let child_sampling = lineage.after;
    let child_run = lineage.child.clone();
    assert_eq!(
        session.sampling_state().unwrap_or_else(fail),
        parent_sampling
    );
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    assert_eq!(saved.token_ids(), &script[..3]);
    assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > before_fork);
    drop(saved);

    session
        .exchange(&mut parent, |record| trace.accept(record))
        .unwrap_or_else(fail);
    assert_eq!(
        session.output_checkpoint().unwrap_or_else(fail).run_id,
        child_run
    );
    assert_eq!(
        session.sampling_state().unwrap_or_else(fail),
        child_sampling
    );
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    let changed = session
        .snapshot(|record| trace.accept(record))
        .unwrap_or_else(fail);
    for manual in [true, false] {
        if !manual {
            let spent = session.snapshot_usage().unwrap().cumulative_copy_bytes;
            session
                .restore(&changed, |record| trace.accept(record))
                .unwrap_or_else(fail);
            assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > spent);
            assert_eq!(session.token_ids(), &script[..3]);
            assert_eq!(session.pending_forced_token(), Some(script[3]));
            assert_eq!(
                session.sampling_state().unwrap_or_else(fail),
                child_sampling
            );
        }
        let start = trace.semantics.len();
        if manual {
            while session.finish_reason().is_none() {
                session
                    .step(|record| trace.accept(record))
                    .unwrap_or_else(fail);
            }
        } else {
            session
                .run(|record| trace.accept(record))
                .unwrap_or_else(fail);
        }
        assert_eq!(session.token_ids(), expected_ids);
        assert_eq!(session.finish_reason(), Some(expected_finish));
        assert_eq!(
            session.sampling_state().unwrap_or_else(fail),
            child_sampling
        );
        assert_eq!(session.pending_forced_token(), None);
        assert_eq!(changed.token_ids(), &script[..3]);
        let actual = prefix
            .iter()
            .chain(&trace.semantics[start..])
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_events, "changed child manual={manual}");
    }
    drop(changed);
    session
        .exchange(&mut parent, |record| trace.accept(record))
        .unwrap_or_else(fail);
    assert_eq!(
        session.output_checkpoint().unwrap_or_else(fail).run_id,
        parent_run
    );
    assert_eq!(
        session.sampling_state().unwrap_or_else(fail),
        parent_sampling
    );
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    let start = trace.semantics.len();
    session
        .run(|record| trace.accept(record))
        .unwrap_or_else(fail);
    assert_eq!(session.token_ids(), expected_ids);
    assert_eq!(session.finish_reason(), Some(expected_finish));
    let actual = prefix
        .iter()
        .chain(&trace.semantics[start..])
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(actual, expected_events, "parent continuation was preserved");
    let spent = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    assert!(spent > before_fork);
    drop(parent);
    assert_eq!(session.snapshot_usage().unwrap().branches, 0);
    assert_eq!(session.snapshot_usage().unwrap().snapshots, 0);
    assert_eq!(
        session.snapshot_usage().unwrap().cumulative_copy_bytes,
        spent
    );
    drop(session);
    model.synchronize().unwrap_or_else(fail);
}
