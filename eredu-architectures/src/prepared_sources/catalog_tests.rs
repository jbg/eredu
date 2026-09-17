use super::*;
use eredu_checkpoint::store::{
    ReadPolicy, SelectedGgufConversionPlan, TensorReadRequest, TensorSelection,
};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};

fn selected(
    path: &std::path::Path,
) -> (
    ModelPreparationPlan<ArtifactArchitecturePlan>,
    SelectedPreparation,
) {
    let inspection = crate::configuration::inspect_artifact(path).unwrap();
    let selected = crate::select_preparation(
        &inspection,
        &eredu_runtime::NormalizedLoadRequest::default(),
        &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
    )
    .unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
        selected.session_capabilities(),
    )
    .unwrap();
    (plan, selected)
}

#[test]
fn retained_catalog_sources_reach_both_artifact_routes_and_last_opaque_identity() {
    let directory = super::tests::gemma4_gguf_fixture();
    let path = directory.path().join("model.gguf");
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (plan, selection) = selected(&path);
    let result = prepare_model_sources_with_catalog_pool(plan, selection, &pool);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_CATALOG").is_some()
        || std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_SOURCE").is_some()
        || std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_COMPOSITE").is_some()
        || std::env::var_os("EREDU_REQUIRE_QUALIFIED_RETAINED_SOURCE").is_some()
    {
        assert!(
            result.is_ok(),
            "mandatory pinned catalog qualification: {:?}",
            result.as_ref().err()
        );
    }
    let sources = match result {
        Ok(sources) => sources,
        Err(PreparedModelSourcesError::GgufCatalog(error))
            if matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ) =>
        {
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(PreparedModelSourcesError::GgufSourceConstructor(error))
            if matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ) =>
        {
            drop(error); // The refused source still owns its admitted catalog.
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(PreparedModelSourcesError::GgufCompositeConstructor(error))
            if matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ) =>
        {
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(PreparedModelSourcesError::SourceErasure(error))
            if matches!(error.accounting_failure(), WorkingMemoryError::UnknownBound) =>
        {
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(error) => panic!("unexpected actual source preparation outcome: {error:?}"),
    };
    let total = pool.used_bytes().unwrap();
    assert!(total > 0);
    let (plan, selection) = selected(&path);
    let ordinary = prepare_model_sources(plan, selection).unwrap();
    assert_eq!(
        pool.used_bytes().unwrap(),
        total,
        "ordinary sources receive no catalog promotion"
    );
    let role = GgufCompanionRole::MediaProjector;
    assert_eq!(
        sources.primary().source_keys(),
        ordinary.primary().source_keys()
    );
    assert_eq!(
        sources.companion(&role).unwrap().source_keys(),
        ordinary.companion(&role).unwrap().source_keys()
    );
    assert_eq!(
        sources.target().source_keys(),
        ordinary.target().source_keys()
    );
    for source in [
        sources.primary(),
        sources.companion(&role).unwrap(),
        sources.target(),
    ] {
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }
    let leaf_bytes = WorkingMemoryPool::gguf_source_erasure_required_bytes().unwrap();
    let union_bytes = WorkingMemoryPool::gguf_composite_erasure_required_bytes().unwrap();
    let primary_identity = sources.primary().identity();
    let companion_identity = sources.companion(&role).unwrap().identity();
    let union_identity = sources.complete().identity();
    for source in [
        sources.primary(),
        sources.companion(&role).unwrap(),
        sources.complete(),
    ] {
        pool.validate_retained_source_controls(source).unwrap();
    }
    assert_eq!(
        pool.validate_retained_source_controls(ordinary.primary()),
        Err(WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        pool.validate_retained_source_controls(ordinary.complete()),
        Err(WorkingMemoryError::UnknownBound)
    );
    let key = sources.primary().source_keys().into_iter().next().unwrap();
    let retained = SelectedGgufConversionPlan::query_retained(
        sources.primary().clone(),
        TensorReadRequest {
            key,
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        },
    )
    .unwrap()
    .unwrap();
    let complete_key = sources
        .companion(&role)
        .unwrap()
        .source_keys()
        .into_iter()
        .next()
        .unwrap();
    let complete_route = SelectedGgufConversionPlan::query_retained(
        sources.complete().clone(),
        TensorReadRequest {
            key: complete_key,
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        },
    )
    .unwrap()
    .unwrap();
    drop(sources);
    assert_eq!(
        pool.used_bytes().unwrap(),
        total,
        "complete route retains actual union and both sources"
    );
    drop(complete_route);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "the retained built-in route owns the primary catalog"
    );
    assert!(
        pool.used_bytes().unwrap() < total,
        "the independent companion catalog has retired"
    );
    drop(retained);
    assert_eq!(pool.used_bytes().unwrap(), 2 * leaf_bytes + union_bytes);
    drop(primary_identity);
    assert_eq!(pool.used_bytes().unwrap(), leaf_bytes + union_bytes);
    drop(companion_identity);
    assert_eq!(pool.used_bytes().unwrap(), union_bytes);
    drop(union_identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(ordinary);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
