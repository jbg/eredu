use super::*;
use eredu_nn::workspace::{WorkspaceReportInputs, WorkspaceReportNode};
const NODE_COUNT: usize = 5;
const EDGE_COUNT: usize = 2;
const ROOT_COUNT: usize = 5;
struct Contribution {
    nodes: [WorkspaceReportNode; NODE_COUNT],
    edges: [usize; EDGE_COUNT],
    opening: [usize; 1],
    allocations: [usize; 3],
    closing: [usize; 1],
}
pub(super) fn recipe() -> report_workspace::ReportRecipe {
    report_workspace::ReportRecipe {
        nodes: NODE_COUNT,
        edges: EDGE_COUNT,
        roots: ROOT_COUNT,
    }
}
pub(super) fn controls() -> usize {
    std::mem::size_of::<Contribution>() + std::mem::size_of::<WorkspaceReportInputs<'_>>()
}
// Actual state/compact/transient buffers of the closed numerical producer. The
// fifth zero-byte node joins the two retained components and owns two aliases.
// No native graph or arbitrary complete report is manufactured by this fixture.
pub(super) fn observe(
    f: &Fixture,
    state: &State,
    first_encoder: bool,
) -> Result<(), WorkingMemoryError> {
    let compact = state
        .compact
        .as_ref()
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    let fixed =
        (std::mem::size_of_val(&state.value) + std::mem::size_of_val(&state.history)) as u64;
    let compact_bytes = std::mem::size_of_val(compact) as u64;
    let activations = std::mem::size_of::<([f32; 2], [f32; 2])>() as u64;
    let encoder = (std::mem::size_of_val(compact) + std::mem::size_of::<[f32; 2]>()) as u64;
    let node = |bytes| WorkspaceReportNode {
        bytes: Some(bytes),
        alias_start: 0,
        alias_count: 0,
    };
    let nodes = [
        node(fixed),
        node(compact_bytes),
        node(activations),
        node(encoder),
        WorkspaceReportNode {
            bytes: Some(0),
            alias_start: 0,
            alias_count: 2,
        },
    ];
    let source = Contribution {
        nodes,
        edges: [0, 1],
        opening: [if first_encoder { 0 } else { 4 }],
        allocations: if first_encoder { [1, 2, 3] } else { [2, 0, 0] },
        closing: [4],
    };
    let report = state
        .owner
        .sources
        .report
        .as_ref()
        .ok_or(WorkingMemoryError::UnknownBound)?
        .report(
            &state.owner.sources,
            &source.nodes,
            &source.edges,
            WorkspaceReportInputs {
                opening: Some(&source.opening),
                allocations: &source.allocations[..if first_encoder { 3 } else { 1 }],
                closing: &source.closing,
                borrowed: None,
                scratch: 0,
                host_workspace: Some(0),
                tensor_complete: true,
            },
        )
        .map_err(|error| error.working_memory())?;
    let values = (
        report.total_bytes.unwrap(),
        report.retained_bytes.unwrap(),
        report.transient_bytes.unwrap(),
        report.state.unwrap().retained_bytes.unwrap(),
    );
    let mut facts = f.facts.lock().unwrap();
    facts.report_count += 1;
    if facts.first_report.is_none() {
        facts.first_report = Some(values);
    }
    facts.last_report = Some(values);
    Ok(())
}

#[test]
fn original_report_source_destinations_fail_at_actual_reserves_under_same_q() {
    for site in 8..11 {
        let (mut runtime, f) = setup(false, None);
        let baseline = f.pool.used_bytes().unwrap();
        let q = probe_q(false);
        diagnostics::fail_destination_for_test(site);
        let error = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .err()
        .unwrap();
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<ConstructionFailure<Sources>>()
            .unwrap();
        assert_eq!(source.destination(), Some(site));
        assert!(source.prefix_bytes() > 0);
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(source);
        let mut reserve = false;
        while let Some(e) = cause {
            reserve |= e.is::<std::collections::TryReserveError>();
            cause = e.source();
        }
        assert!(reserve);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        assert!(f.pool.0.usage.lock().unwrap().pending_original.is_none());
        assert_eq!(f.facts.lock().unwrap().report_count, 0);
        assert_eq!(f.facts.lock().unwrap().prepared, 0);
        // This error owns its actual prefix, but no longer monopolizes the
        // global pending slot; a fresh genuine request can complete construction.
        let fresh = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .unwrap();
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + 2 * q);
        drop(fresh);
        drop(error);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}

#[test]
fn original_report_aliases_preserve_source_identity_and_retire_payload_before_q() {
    let (mut runtime, f) = setup(false, None);
    let baseline = f.pool.used_bytes().unwrap();
    let q = probe_q(false);
    let mut run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .unwrap();
    let token = run.next().unwrap().unwrap();
    let owner = token.owner.sources.report.as_ref().unwrap().clone();
    assert!(owner.bytes() > 0);
    let aliases: [report_workspace::ReportOwner; 8] = std::array::from_fn(|_| owner.clone());
    let (_foreign_runtime, foreign) = setup(false, None);
    let empty = WorkspaceReportInputs {
        opening: None,
        allocations: &[],
        closing: &[],
        borrowed: None,
        scratch: 0,
        host_workspace: Some(0),
        tensor_complete: true,
    };
    assert_eq!(
        owner
            .report(&foreign.sources, &[], &[], empty)
            .unwrap_err()
            .working_memory(),
        WorkingMemoryError::IdentityMismatch
    );
    let retained = owner.bytes();
    assert!(retained > 0);
    drop(token);
    drop(run);
    drop(owner);
    let pool = f.pool.clone();
    let source_bytes = f.sources.input.original_bytes() + f.sources.selected_model.original_bytes();
    drop(runtime);
    drop(f);
    assert_eq!(pool.used_bytes().unwrap(), baseline + q);
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for alias in aliases {
            let barrier = &barrier;
            let pool = &pool;
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(alias.bytes(), retained);
                assert_eq!(pool.used_bytes().unwrap(), baseline + q);
                drop(alias);
            });
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), baseline - source_bytes);
}

#[test]
fn original_report_reduction_errors_retain_same_q_and_reuse_exact_destinations() {
    let (mut runtime, f) = setup(false, None);
    let baseline = f.pool.used_bytes().unwrap();
    let q = probe_q(false);
    let mut run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        prompt(&f),
        f.config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(OUTPUTS, &EOS),
    )
    .unwrap();
    let token = run.next().unwrap().unwrap();
    let sources = &token.owner.sources;
    let owner = sources.report.as_ref().unwrap();
    let retained = owner.bytes();
    let input = WorkspaceReportInputs {
        opening: Some(&[]),
        allocations: &[0, 1],
        closing: &[],
        borrowed: Some(&[]),
        scratch: 0,
        host_workspace: Some(0),
        tensor_complete: true,
    };
    let nodes = [
        WorkspaceReportNode {
            bytes: Some(u64::MAX),
            alias_start: 0,
            alias_count: 0,
        },
        WorkspaceReportNode {
            bytes: Some(1),
            alias_start: 0,
            alias_count: 0,
        },
    ];
    assert_eq!(
        owner
            .report(sources, &nodes, &[], input)
            .unwrap_err()
            .working_memory(),
        WorkingMemoryError::Overflow
    );
    assert_eq!(owner.bytes(), retained);
    assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
    let malformed = [WorkspaceReportNode {
        alias_count: 1,
        ..nodes[0]
    }];
    assert_eq!(
        owner
            .report(sources, &malformed, &[], input)
            .unwrap_err()
            .working_memory(),
        WorkingMemoryError::IdentityMismatch
    );
    assert_eq!(owner.bytes(), retained);
    let valid = [
        WorkspaceReportNode {
            bytes: Some(7),
            ..nodes[0]
        },
        nodes[1],
    ];
    let scalar = owner.report(sources, &valid, &[], input).unwrap();
    assert_eq!(scalar.total_bytes, Some(8));
    assert_eq!(scalar.residual.unwrap().total_bytes, Some(8));
    assert_eq!(owner.bytes(), retained);
    drop(token);
    drop(run);
    assert_eq!(f.pool.used_bytes().unwrap(), baseline);
}

#[test]
fn original_report_nonblocking_busy_and_poison_errors_retain_buffers_and_q() {
    for poison in [false, true] {
        let (mut runtime, f) = setup(false, None);
        let baseline = f.pool.used_bytes().unwrap();
        let q = probe_q(false);
        let mut run = TextGeneration::from_input_with_sequence(
            &mut runtime,
            prompt(&f),
            f.config(),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(OUTPUTS, &EOS),
        )
        .unwrap();
        let token = run.next().unwrap().unwrap();
        let owner = token.owner.sources.report.as_ref().unwrap().clone();
        let input = WorkspaceReportInputs {
            opening: Some(&[]),
            allocations: &[],
            closing: &[],
            borrowed: None,
            scratch: 0,
            host_workspace: Some(0),
            tensor_complete: true,
        };
        let error = if poison {
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _held = owner.hold_for_test();
                panic!("poison after source/scratch construction");
            }));
            assert!(unwind.is_err());
            let error = owner
                .report(&token.owner.sources, &[], &[], input)
                .unwrap_err();
            assert_eq!(error.cause(), report_workspace::ReportUseCause::Poisoned);
            error
        } else {
            let held = owner.hold_for_test();
            let error = std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        owner
                            .report(&token.owner.sources, &[], &[], input)
                            .unwrap_err()
                    })
                    .join()
                    .unwrap()
            });
            assert_eq!(error.cause(), report_workspace::ReportUseCause::Busy);
            drop(held);
            assert_eq!(
                owner
                    .report(&token.owner.sources, &[], &[], input)
                    .unwrap()
                    .total_bytes,
                Some(0)
            );
            error
        };
        drop(token);
        drop(run);
        drop(owner);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline + q);
        let mut chain: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut cause_found = false;
        while let Some(cause) = chain {
            cause_found |= cause.is::<report_workspace::ReportUseCause>();
            chain = cause.source();
        }
        assert!(cause_found);
        drop(error);
        assert_eq!(f.pool.used_bytes().unwrap(), baseline);
    }
}
