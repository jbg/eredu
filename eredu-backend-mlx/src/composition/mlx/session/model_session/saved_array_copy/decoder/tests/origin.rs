use super::*;

fn saved_pair(runtime: &mut ModelRuntime<MlxBackend<'static>>) -> CopiedTextComponents {
    let mut driver = TextGenerationDriver::new(runtime);
    let mut state = start_variant(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let saved =
            PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
                .unwrap()
                .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
                .unwrap();
        let before = (
            accounting(runtime.backend().memory_pool()),
            paths::snapshot(),
            copies(),
        );
        saved.validate_resume_origin(runtime).unwrap();
        assert_eq!(
            (
                accounting(runtime.backend().memory_pool()),
                paths::snapshot(),
                copies()
            ),
            before
        );
        saved
    };
    drop((outputs, state, driver));
    reclaim();
    saved
}

#[test]
fn saved_origin_survives_model_retirement_without_becoming_foreign_resume_authority() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut original, original_artifact) = runtime(&pool);
    // Loading is unquoted work. Both executables must finish loading before
    // the immutable saved account reserves work in their shared domain.
    let (mut foreign, foreign_artifact) = runtime(&pool);
    let saved = saved_pair(&mut original);
    let saved_layout = saved
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .shared_layout()
        .unwrap()
        .clone();
    let foreign_layout = foreign
        .session()
        .payload
        .model
        .erased()
        .prepare_resident_decoder_copy()
        .unwrap()
        .shared_layout()
        .unwrap()
        .clone();
    assert_eq!(
        saved_layout.as_ref(),
        foreign_layout.as_ref(),
        "equal geometry does not identify an executable"
    );
    let before = (accounting(&pool), paths::snapshot(), copies());
    let error = saved.validate_resume_origin(&foreign).unwrap_err();
    assert!(
        error.to_string().contains("different executable"),
        "{error}"
    );
    assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
    saved.validate_resume_origin(&original).unwrap();
    let original_payload = original.session().payload.retirement_probe();
    drop((original, original_artifact));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        original_payload()
    });
    assert!(original_payload(), "saved origin must not retain the model");
    // Duplication after model retirement is data copying. Its current destination
    // must not replace the saved origin with its own executable identity.
    let duplicate = PreparedTextComponentsCopy::prepare_saved(&foreign, &saved)
        .unwrap()
        .copy(&mut foreign, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_independent(&saved, &duplicate);
    for value in [&saved, &duplicate] {
        let before = (accounting(&pool), paths::snapshot(), copies());
        let error = value.validate_resume_origin(&foreign).unwrap_err();
        assert!(
            error.to_string().contains("different executable"),
            "{error}"
        );
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
    }
    drop((
        saved,
        duplicate,
        foreign,
        foreign_artifact,
        saved_layout,
        foreign_layout,
    ));
    settle(&pool, 0);
}

#[test]
fn parameter_origin_invalidation_rejects_before_copy_and_survives_immutable_duplication() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let saved = saved_pair(&mut runtime);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
    saved.validate_resume_origin(&runtime).unwrap();
    // Exercise the same exact-origin rotation used after parameter publication;
    // no equality of layout or numeric parameter-epoch values can undo it.
    runtime
        .session_mut()
        .payload
        .get_mut()
        .unwrap()
        .model
        .erased_mut()
        .invalidate_parameter_snapshots();
    let before = (accounting(&pool), paths::snapshot(), copies());
    let error = saved.validate_resume_origin(&runtime).unwrap_err();
    assert!(
        error.to_string().contains("different executable"),
        "{error}"
    );
    assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
    let duplicate = PreparedTextComponentsCopy::prepare_saved(&runtime, &saved)
        .unwrap()
        .copy(&mut runtime, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_independent(&saved, &duplicate);
    let before = (accounting(&pool), paths::snapshot(), copies());
    assert!(duplicate.validate_resume_origin(&runtime).is_err());
    assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
    drop((saved, duplicate, runtime, artifact));
    settle(&pool, 0);
}
