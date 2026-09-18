use super::*;
use eredu_checkpoint::artifact::ArtifactFileReadError;
use std::{error::Error as _, fs::File, io::Write};

const JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"post_processor":{"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"added_tokens":[{"id":4,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2,"Ġ":3},"merges":[["h","i"]]}}"#;

fn file(input: &[u8]) -> File {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(input).unwrap();
    file
}
fn read(input: &[u8]) -> PreparedArtifactFileRead {
    PreparedArtifactFileRead::new(file(input)).unwrap()
}
fn facts() -> (u64, u64) {
    (
        WorkingMemoryPool::tokenizer_file_required_bytes(&read(JSON.as_bytes())).unwrap(),
        WorkingMemoryPool::tokenizer_required_bytes(
            &TokenizerPlan::prepare_json(JSON.as_bytes()).unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn exact_file_i_then_fresh_c_coexist_and_only_the_last_source_owner_retires_c() {
    let (i, c) = facts();
    let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
    let source = pool
        .compile_tokenizer_file_with(
            read(JSON.as_bytes()),
            |_| {
                assert_eq!(pool.used_bytes().unwrap(), i);
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
            },
            || {
                assert_eq!(pool.used_bytes().unwrap(), i);
                assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
                drop(pool.acquire_unquoted().unwrap());
            },
            |pool, plan| {
                assert_eq!(
                    WorkingMemoryPool::tokenizer_required_bytes(&plan).unwrap(),
                    c
                );
                pool.compile_tokenizer_with(plan, || {
                    assert_eq!(pool.used_bytes().unwrap(), i + c);
                    assert!(matches!(
                        pool.acquire_unquoted(),
                        Err(WorkingMemoryError::ReservedWorkActive)
                    ));
                })
            },
        )
        .unwrap();
    assert_eq!(source.token_id("hi"), Some(2));
    assert_eq!(source.spelling(4), Some("<S>"));
    assert_eq!(source.original_bytes(), c);
    assert_eq!(pool.used_bytes().unwrap(), c);
    assert_eq!(pool.peak_bytes().unwrap(), i + c);
    let alias = source.clone();
    assert!(source.same_source(&alias));
    let witness = pool.clone();
    drop((pool, source));
    assert_eq!(witness.used_bytes().unwrap(), c);
    assert_eq!(alias.spelling(2), Some("hi"));
    let peer = alias.clone();
    std::thread::scope(|threads| {
        threads.spawn(move || drop(alias));
        threads.spawn(move || drop(peer));
    });
    assert_eq!(witness.used_bytes().unwrap(), 0);
}

#[test]
fn one_short_i_never_reserves_and_one_short_c_keeps_actual_file_bytes_in_its_error() {
    let (i, c) = facts();
    let pool = WorkingMemoryPool::new(i - 1, 0).unwrap();
    let error = pool
        .compile_tokenizer_file_with(
            read(JSON.as_bytes()),
            |_| panic!("I rejected"),
            || panic!("I rejected"),
            |_, _| panic!("I rejected"),
        )
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == i && *available_bytes == i - 1)
    );
    assert_eq!(error.input_bytes(), 0);
    assert_eq!(error.input_capacity(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(i + c - 1, 0).unwrap();
    let error = pool
        .compile_tokenizer_file(read(JSON.as_bytes()))
        .unwrap_err();
    assert!(
        matches!(error.compiler_failure().unwrap().accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == c && *available_bytes == c - 1)
    );
    assert_eq!(error.compiler_failure().unwrap().retained_bytes(), 0);
    assert_eq!(error.filled_bytes(), JSON.len());
    assert!(error.input_capacity() >= JSON.len());
    assert_eq!(error.input_bytes(), i);
    assert_eq!(pool.used_bytes().unwrap(), i);
    let error = error.into_backend_failure();
    assert_eq!(error.kind(), BackendFailureKind::ResourceExhausted);
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<OriginalTokenizerInputError>()
            .unwrap()
            .input_bytes(),
        i
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn actual_input_target_overflow_os_read_failure_and_changed_handle_keep_original_i() {
    let (i, c) = facts();
    let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
    let error = pool
        .compile_tokenizer_file_with(
            read(JSON.as_bytes()),
            |requested| *requested = usize::MAX,
            || panic!("reserve failed"),
            |_, _| panic!("reserve failed"),
        )
        .unwrap_err();
    assert!(error.source().unwrap().is::<TryReserveError>());
    assert_eq!(error.input_capacity(), 0);
    assert_eq!(error.input_bytes(), i);
    assert_eq!(pool.used_bytes().unwrap(), i);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    std::fs::write(&path, JSON).unwrap();
    let write_only = File::options().write(true).open(&path).unwrap();
    let error = pool
        .compile_tokenizer_file(PreparedArtifactFileRead::new(write_only).unwrap())
        .unwrap_err();
    let failure = error.read_failure().unwrap();
    assert!(
        failure
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .raw_os_error()
            .is_some()
    );
    assert_eq!(error.filled_bytes(), 0);
    assert!(error.input_capacity() >= JSON.len());
    assert_eq!(error.kind(), BackendFailureKind::Io);
    let error = error.into_backend_failure();
    assert_eq!(pool.used_bytes().unwrap(), i);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let file = file(JSON.as_bytes());
    let writer = file.try_clone().unwrap();
    let read = PreparedArtifactFileRead::new(file).unwrap();
    let error = pool
        .compile_tokenizer_file_with(
            read,
            |_| writer.set_len(2).unwrap(),
            || panic!("changed file"),
            |_, _| panic!("changed file"),
        )
        .unwrap_err();
    assert!(matches!(
        error.read_failure().unwrap().cause(),
        ArtifactFileReadError::Changed
    ));
    assert_eq!(error.input_bytes(), i);
    assert_eq!(error.filled_bytes(), 0);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn real_aggregate_partial_frontiers_keep_both_original_allowances_in_one_core_envelope() {
    let (i, c) = facts();
    for stage in 0..11 {
        let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
        let error = pool
            .compile_tokenizer_file_with(
                read(JSON.as_bytes()),
                |_| {},
                || {},
                |pool, plan| {
                    let plan = match stage {
                        0 => plan.fail_model_reservation(3),
                        1..=3 => plan.fail_pipeline_reservation(stage - 1),
                        4..=7 => plan.fail_added_reservation(stage - 4),
                        _ => plan.fail_decode_reservation(stage - 8),
                    };
                    pool.compile_tokenizer(plan)
                },
            )
            .unwrap_err();
        assert_eq!(error.input_bytes(), i);
        assert_eq!(error.filled_bytes(), JSON.len());
        let compiler = error.compiler_failure().unwrap();
        assert_eq!(compiler.retained_bytes(), c);
        assert_eq!(
            compiler.compiler_failure().unwrap().completed_tokenizer(),
            stage >= 8
        );
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut reserve = false;
        while let Some(error) = cause {
            reserve |= error.is::<TryReserveError>();
            cause = error.source();
        }
        assert!(reserve, "actual target frontier {stage}");
        drop(pool.acquire_unquoted().unwrap());
        let error = error.into_backend_failure();
        assert_eq!(pool.used_bytes().unwrap(), i + c);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn root_rejection_empty_and_late_semantics_keep_input_without_reclassifying_it_as_c() {
    // These formerly unqualified regex producers now use fresh construction.
    // Preserve both as real C-admission refusals while I remains retained.
    for supported in [
        JSON.replace("\"use_regex\":false", "\"use_regex\":true"),
        JSON.replace(
            r#"{"type":"Digits","individual_digits":true}"#,
            r#"{"type":"Whitespace"}"#,
        ),
    ] {
        assert!(TokenizerPlan::prepare_json(supported.as_bytes()).is_ok());
        let input = read(supported.as_bytes());
        let i = WorkingMemoryPool::tokenizer_file_required_bytes(&input).unwrap();
        let pool = WorkingMemoryPool::new(i, 0).unwrap();
        let error = pool.compile_tokenizer_file(input).unwrap_err();
        assert!(error.planning_failure().is_none());
        assert!(matches!(
            error.compiler_failure().unwrap().accounting_failure(),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(error.filled_bytes(), supported.len());
        assert_eq!(pool.used_bytes().unwrap(), i);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    for bytes in [b"".as_slice(), b"invalid json".as_slice()] {
        let read = read(bytes);
        let i = WorkingMemoryPool::tokenizer_file_required_bytes(&read).unwrap();
        let pool = WorkingMemoryPool::new(i, 0).unwrap();
        let error = pool.compile_tokenizer_file(read).unwrap_err();
        assert!(error.planning_failure().is_some());
        assert!(error.compiler_failure().is_none());
        assert_eq!(error.filled_bytes(), bytes.len());
        assert_eq!(pool.used_bytes().unwrap(), i);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let invalid = JSON.replace("\"i\":1", "\"i\":0");
    let read = read(invalid.as_bytes());
    let i = WorkingMemoryPool::tokenizer_file_required_bytes(&read).unwrap();
    let c = WorkingMemoryPool::tokenizer_required_bytes(
        &TokenizerPlan::prepare_json(invalid.as_bytes()).unwrap(),
    )
    .unwrap();
    let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
    let error = pool.compile_tokenizer_file(read).unwrap_err();
    assert!(
        error
            .compiler_failure()
            .unwrap()
            .compiler_failure()
            .is_some()
    );
    assert_eq!(pool.used_bytes().unwrap(), i + c);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unwind_destroys_file_destinations_before_healthy_i_refund() {
    let (i, c) = facts();
    for phase in 0..3 {
        let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.compile_tokenizer_file_with(
                read(JSON.as_bytes()),
                |_| {
                    if phase == 0 {
                        panic!("installed I before first reserve");
                    }
                },
                || {
                    if phase == 1 {
                        panic!("filled I before C");
                    }
                },
                |pool, plan| pool.compile_tokenizer_with(plan, || panic!("installed original C")),
            )
        }));
        assert!(unwind.is_err());
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    }
}

#[test]
fn concurrent_real_reads_and_compilers_preserve_each_active_and_idle_charge() {
    let (i, c) = facts();
    let pool = WorkingMemoryPool::new(2 * (i + c), 0).unwrap();
    let reads = [read(JSON.as_bytes()), read(JSON.as_bytes())];
    let (outputs, observed) = std::thread::scope(|threads| {
        let mut receivers = Vec::new();
        let mut releases = Vec::new();
        let mut workers = Vec::new();
        for read in reads {
            let (notify, receiver) = std::sync::mpsc::channel::<bool>();
            let (release, resume) = std::sync::mpsc::channel::<()>();
            receivers.push(receiver);
            releases.push(release);
            let pool = &pool;
            workers.push(threads.spawn(move || {
                let reached = std::cell::Cell::new(0);
                let pause = || {
                    reached.set(reached.get() + 1);
                    let _ = notify.send(true);
                    // Parent disconnection always releases this worker.
                    let _ = resume.recv();
                };
                let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool.compile_tokenizer_file_with(
                        read,
                        |_| pause(),
                        pause,
                        |pool, plan| pool.compile_tokenizer_with(plan, pause),
                    )
                }));
                // An early admission failure or panic cannot strand a receiver.
                for _ in reached.get()..3 {
                    let _ = notify.send(false);
                }
                output
            }));
        }
        let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut observed = Vec::new();
            for _ in 0..3 {
                let arrivals: Vec<_> = receivers.iter().map(|r| r.recv()).collect();
                let used = pool.used_bytes();
                let reservations = pool.0.usage.lock().unwrap().reservations;
                let idle = match pool.acquire_unquoted() {
                    Ok(owner) => {
                        drop(owner);
                        true
                    }
                    Err(WorkingMemoryError::ReservedWorkActive) => false,
                    Err(error) => panic!("unexpected account result: {error}"),
                };
                observed.push((arrivals, used, reservations, idle));
                for release in &releases {
                    let _ = release.send(());
                }
            }
            observed
        }));
        drop(releases);
        let outputs: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
        (outputs, observed)
    });
    let observed = observed.unwrap();
    for (phase, (arrivals, used, reservations, idle)) in observed.into_iter().enumerate() {
        assert!(arrivals.iter().all(|a| matches!(a, Ok(true))));
        assert_eq!(used.unwrap(), if phase == 2 { 2 * (i + c) } else { 2 * i });
        assert_eq!(reservations, if phase == 1 { 0 } else { 2 });
        assert_eq!(idle, phase == 1);
    }
    let sources: Vec<_> = outputs
        .into_iter()
        .map(|o| o.unwrap().unwrap().unwrap())
        .collect();
    assert!(!sources[0].same_source(&sources[1]));
    assert_eq!(pool.used_bytes().unwrap(), 2 * c);
    drop(sources);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn completed_aggregate_on_poison_keeps_both_original_allowances_quarantined() {
    let (i, c) = facts();
    let pool = WorkingMemoryPool::new(i + c, 0).unwrap();
    let error = pool
        .compile_tokenizer_file_with(
            read(JSON.as_bytes()),
            |_| {},
            || {},
            |pool, plan| {
                pool.compile_tokenizer_with(plan, || {
                    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _usage = pool.0.usage.lock().unwrap();
                        panic!("actual original account poisoned before completed publication");
                    }));
                    assert!(poisoned.is_err());
                })
            },
        )
        .unwrap_err();
    let compiler = error.compiler_failure().unwrap();
    assert_eq!(compiler.retained_bytes(), c);
    assert_eq!(
        compiler._completed.as_ref().unwrap().token_id("hi"),
        Some(2)
    );
    assert_eq!(error.input_bytes(), i);
    assert_eq!(error.kind(), BackendFailureKind::InvalidSession);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, i + c);
    assert_eq!(usage.reservations, 1); // I became idle; only C failed settlement.
    drop(usage);
    drop(error.into_backend_failure());
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, i + c); // Poison never certifies a refund.
}

#[test]
fn retained_configuration_uses_fresh_compiler_preserves_flags_and_keeps_input_failures() {
    use std::str::FromStr;
    let mut selected = eredu_text::tokenizer::Tokenizer::from_str(JSON).unwrap();
    selected.set_encode_special_tokens(true);
    let pool = WorkingMemoryPool::new(64 * 1024 * 1024, 0).unwrap();
    let source = pool
        .compile_tokenizer_source_for_generation(OriginalTokenizerInput::Configuration(&selected))
        .unwrap();
    assert!(source.matches_configuration(&selected));
    assert!(source.generation_domain().is_some());
    let peak = pool.peak_bytes().unwrap();
    let compiled = source.original_bytes();
    assert!(peak > compiled);
    assert_eq!(pool.used_bytes().unwrap(), compiled);
    let alias = source.clone();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), compiled);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let short = WorkingMemoryPool::new(peak - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_source_for_generation(OriginalTokenizerInput::Configuration(&selected))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(
        error.compiler_failure().is_some(),
        "complete admitted input reaches the same C comparison"
    );
    assert!(error.filled_bytes() > 0);
    assert!(error.input_capacity() >= error.filled_bytes());
    assert_eq!(short.used_bytes().unwrap(), error.input_bytes());
    drop(error);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let tiny = WorkingMemoryPool::new(1, 0).unwrap();
    let error = tiny
        .compile_tokenizer_source_for_generation(OriginalTokenizerInput::Configuration(&selected))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(error.input_capacity(), 0);
    assert_eq!(tiny.used_bytes().unwrap(), 0);
}

