use super::*;
use crate::backend::runtime::checkpoint::store::{
    CacheHandle, OriginalGgufMissFixture, PreparedMaterializationStreams,
};
use crate::backend::runtime::residency::manager::ResidencyError;
use eredu_checkpoint::store::{SharedCheckpointSource, TensorSelection};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_runtime::residency::{OffloadUnit, WeightBinding};
use eredu_runtime::working_memory::{WorkingMemoryPool, WorkingMemoryReservation};

fn manager(
    source: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
    stream: &safemlx::Stream,
    cache: Option<CacheHandle>,
    pool: &WorkingMemoryPool,
) -> ResidencyManager {
    let source = source.into();
    let id = OffloadUnitId::new("cache.window").unwrap();
    let binding = WeightBinding::new("weight", "bank.weight", TensorSelection::Full, 32).unwrap();
    let unit = OffloadUnit::new(id.clone(), vec![binding]).unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        [OffloadUnitSpec::new(id, 32, ResidencyPolicy::Windowed, MemoryTier::Disk).unwrap()],
    )
    .unwrap();
    match cache {
        Some(cache) => ResidencyManager::new_shared_sources_with_cache(
            source,
            Default::default(),
            plan,
            [unit],
            stream.clone(),
            stream.clone(),
            cache,
            pool,
        ),
        None => ResidencyManager::new_shared(source, plan, [unit], stream.clone(), stream.clone()),
    }
    .unwrap()
}
fn source_plan(
    manager: &ResidencyManager,
    runtime: &safemlx::PreparedInputRuntime,
) -> SourceArenaPlan {
    let ids = [OffloadUnitId::new("cache.window").unwrap()];
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![eredu_runtime::ExecutionGroupSpec::root("all")],
        "all",
    )
    .unwrap();
    let layout = eredu_runtime::execution::ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let selected = manager
        .prepare_original_operation_source(&ids, &layout, 1)
        .unwrap();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        input_positions: 1,
        cached_positions: 0,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let population =
        ResidencyPopulation::from_windows(selected.controller_units, selected.windows(), geometry)
            .unwrap();
    SourceArenaPlan::prepare(manager, selected.windows(), &ids, population, runtime).unwrap()
}
// Zero-work reservations are used only to compare actual pool identities.
fn reservation(pool: &WorkingMemoryPool) -> WorkingMemoryReservation {
    use eredu_core::{
        Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
        InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless cache-origin comparison");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry: InferenceGeometry {
            batch_size: 1,
            input_positions: 1,
            cached_positions: 0,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: OutputDemand::StateOnly,
        },
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    pool.reserve(
        &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
        &Admission {
            requested_positions: 1,
            state,
            incremental_required_bytes: 0,
            available_memory_bytes: None,
        },
    )
    .unwrap()
}
#[test]
fn selected_source_plan_retains_cache_origin_without_promoting_ordinary_facts() {
    let required = CacheHandle::required_storage_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_CACHE_CONTEXT").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    let bytes = match required {
        Ok(bytes) => bytes,
        Err(WorkingMemoryError::UnknownBound) => return,
        Err(error) => panic!("unexpected cache qualification: {error}"),
    };
    let fixture = OriginalGgufMissFixture::new();
    let (source, runtime, stream) = fixture.source_planning_inputs();
    let stream_bytes = PreparedMaterializationStreams::required_bytes(&stream, &stream).unwrap();
    // Two independently prepared managers below each own a separate stream pair.
    let all_stream_bytes = stream_bytes.checked_mul(2).unwrap();
    let total = bytes
        .checked_mul(2)
        .unwrap()
        .checked_add(all_stream_bytes)
        .unwrap();
    let pool = WorkingMemoryPool::new(total, 0).unwrap();
    let foreign = WorkingMemoryPool::new(bytes, 0).unwrap();
    let cache = CacheHandle::prepare(&pool).unwrap();
    let selected = manager(source.clone(), &stream, Some(cache.clone()), &pool);
    let ordinary = manager(source.clone(), &stream, None, &pool);
    let unrelated = manager(
        source,
        &stream,
        Some(CacheHandle::prepare(&pool).unwrap()),
        &pool,
    );
    assert_eq!(pool.used_bytes().unwrap(), total);
    let plan = source_plan(&selected, &runtime);
    let ordinary_plan = source_plan(&ordinary, &runtime);
    assert_eq!(plan.facts(), ordinary_plan.facts());
    assert_eq!(plan.host_facts(), ordinary_plan.host_facts());
    let local = reservation(&pool);
    let other = reservation(&foreign);
    plan.validate_cache_origin(&selected, &local).unwrap();
    assert!(matches!(
        plan.validate_cache_origin(&selected, &other),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    ));
    assert!(matches!(
        ordinary_plan.validate_cache_origin(&ordinary, &local),
        Err(Error::PrefillControl(WorkingMemoryError::UnknownBound))
    ));
    assert!(matches!(
        plan.validate_cache_origin(&ordinary, &local),
        Err(Error::Residency(ResidencyError::OriginalOperationDomain))
    ));
    assert!(matches!(
        plan.validate_cache_origin(&unrelated, &local),
        Err(Error::Residency(ResidencyError::OriginalOperationDomain))
    ));
    drop((
        cache,
        selected,
        ordinary,
        unrelated,
        ordinary_plan,
        local,
        other,
    ));
    // Reclaim outside every manager/cache loan before isolating the plan's
    // surviving cache ownership. Ordinary Stream drops may already drain it.
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(plan);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn selected_source_plan_retains_actual_catalog_origin_through_manager_erasure() {
    use eredu_checkpoint::{
        gguf_store::GgufCatalogPlan,
        schema::{
            CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
            TensorOperation,
        },
        store::CheckpointSource,
        validation::resolve_gguf_plan,
    };
    use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog-origin.gguf");
    let bytes: Vec<_> = [1.5_f32, -2.5, 3.5, 4.5, -5.5, 6.5, 7.5, -8.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &Default::default(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[4, 2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    let checkpoint = Checkpoint::open(path).unwrap();
    let schema = GgufCheckpointPlan::new(
        "catalog origin fixture",
        vec![GgufTensorConstraint::required(
            "bank.weight",
            vec![2, 4],
            GgufTypeConstraint::OperationClass(TensorOperation::Dense),
        )],
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap();
    let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    let input = GgufCatalogPlan::new(checkpoint, &resolution, &mapping, 1);
    let required = WorkingMemoryPool::gguf_catalog_required_bytes(&input);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_CATALOG").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    let catalog_bytes = match required {
        Ok(bytes) => bytes,
        Err(WorkingMemoryError::UnknownBound) => return,
        Err(error) => panic!("unexpected catalog qualification: {error}"),
    };
    // Native setup, reader shells and ordinary fixture storage are separate
    // baseline owners. This case exercises the actual selected catalog route.
    let fixture = OriginalGgufMissFixture::new();
    let (ordinary_source, runtime, stream) = fixture.source_planning_inputs();
    let cache_bytes = CacheHandle::required_storage_bytes().unwrap();
    let retained_bytes = catalog_bytes.checked_add(cache_bytes).unwrap();
    let stream_bytes = PreparedMaterializationStreams::required_bytes(&stream, &stream).unwrap();
    let total = retained_bytes.checked_add(stream_bytes).unwrap();
    let pool = WorkingMemoryPool::new(total, 0).unwrap();
    let foreign = WorkingMemoryPool::new(total, 0).unwrap();
    let source = pool
        .compile_gguf_catalog(input)
        .unwrap()
        .into_prepared()
        .build()
        .unwrap();
    let cache = CacheHandle::prepare(&pool).unwrap();
    let selected = manager(
        std::sync::Arc::new(source.clone()),
        &stream,
        Some(cache),
        &pool,
    );
    let ordinary = manager(ordinary_source, &stream, None, &pool);
    let plan = source_plan(&selected, &runtime);
    let ordinary_plan = source_plan(&ordinary, &runtime);
    let local = reservation(&pool);
    let other = reservation(&foreign);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    plan.validate_catalog_origins(&local).unwrap();
    assert!(matches!(
        plan.validate_catalog_origins(&other),
        Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
    ));
    assert!(matches!(
        ordinary_plan.validate_catalog_origins(&local),
        Err(Error::PrefillControl(WorkingMemoryError::UnknownBound))
    ));
    assert_eq!(pool.used_bytes().unwrap(), total);
    drop((source, selected, ordinary, ordinary_plan, local, other));
    safemlx::reclaim_allocation_owners(); // no manager/cache loan remains
    assert_eq!(pool.used_bytes().unwrap(), retained_bytes); // exact catalog + cache
    drop(plan);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn retained_reader_root_reaches_manager_registration_and_last_opaque_identity() {
    use eredu_checkpoint::{
        gguf_store::GgufCatalogPlan,
        schema::{
            CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
            TensorOperation,
        },
        store::CheckpointSource,
        validation::resolve_gguf_plan,
    };
    use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reader-origin.gguf");
    let bytes: Vec<_> = [1.5_f32, -2.5, 3.5, 4.5, -5.5, 6.5, 7.5, -8.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &Default::default(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[4, 2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    let checkpoint = Checkpoint::open(path).unwrap();
    let schema = GgufCheckpointPlan::new(
        "reader registration fixture",
        vec![GgufTensorConstraint::required(
            "bank.weight",
            vec![2, 4],
            GgufTypeConstraint::OperationClass(TensorOperation::Dense),
        )],
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap();
    let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    let input = || GgufCatalogPlan::new(checkpoint.clone(), &resolution, &mapping, 1);
    let ordinary_catalog = input().compile(()).unwrap();
    let required = WorkingMemoryPool::gguf_source_required_bytes(&ordinary_catalog);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_SOURCE").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    let source_bytes = match required {
        Ok(bytes) => bytes,
        Err(WorkingMemoryError::UnknownBound) => return,
        Err(error) => panic!("unexpected source qualification: {error}"),
    };
    let catalog_bytes = WorkingMemoryPool::gguf_catalog_required_bytes(&input()).unwrap();
    // Physical inventory does not depend on the custody type. Its authentic
    // prepaid origin is still issued only by compile_gguf_source below.
    let (physical, prepaid) = ordinary_catalog
        .source_storage_request::<()>()
        .unwrap()
        .inventory_bytes();
    drop(ordinary_catalog);
    assert!(prepaid > 0 && physical >= prepaid);
    let residual = physical - prepaid;
    let erasure_bytes = WorkingMemoryPool::gguf_source_erasure_required_bytes().unwrap();
    let constructor = catalog_bytes
        .checked_add(source_bytes)
        .unwrap()
        .checked_add(erasure_bytes)
        .unwrap();
    let pool = WorkingMemoryPool::new(constructor.checked_add(residual).unwrap(), 0).unwrap();
    let foreign = WorkingMemoryPool::new(physical, 0).unwrap();
    let source = pool
        .compile_gguf_source(pool.compile_gguf_catalog(input()).unwrap().into_prepared())
        .unwrap();
    let source = pool.retain_gguf_source(source).unwrap();
    let root_identity = source.identity();
    pool.validate_retained_source_controls(&source).unwrap();
    assert_eq!(
        foreign.validate_retained_source_controls(&source),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(pool.used_bytes().unwrap(), constructor);

    // The existing native fixture owns unrelated runtime/stream setup. This
    // component's ordinary manager adds no admitted cache/stream allocation.
    let fixture = OriginalGgufMissFixture::new();
    let (_, _runtime, stream) = fixture.source_planning_inputs();
    let selected = manager(source.clone(), &stream, None, &pool);
    let inventory = selected.retained_storage().unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), Some(physical));
    let mut source_entries = inventory.source_storage().capacities();
    let (key, declared) = source_entries.next().unwrap();
    assert!(source_entries.next().is_none());
    assert_eq!(declared, physical);
    drop(source_entries);
    let first = inventory.register(&pool).unwrap();
    let second = selected
        .retained_storage()
        .unwrap()
        .register(&pool)
        .unwrap();
    let foreign_owner = selected
        .retained_storage()
        .unwrap()
        .register(&foreign)
        .unwrap();
    assert_eq!(first.bytes(), physical);
    assert_eq!(second.bytes(), physical);
    assert_eq!(foreign_owner.bytes(), physical);
    assert_eq!(pool.used_bytes().unwrap(), constructor + residual);
    assert_eq!(foreign.used_bytes().unwrap(), physical);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);

    drop((selected, source));
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.used_bytes().unwrap(), constructor + residual);
    drop(first);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.used_bytes().unwrap(), constructor + residual);
    drop(second);
    crate::backend::ordinary_retirement::reclaim_all();
    // B still owns the same source, but A's canonical registration retired.
    assert_eq!(pool.used_bytes().unwrap(), constructor);
    assert_eq!(foreign.used_bytes().unwrap(), physical);
    drop(foreign_owner);
    assert_eq!(foreign.used_bytes().unwrap(), physical); // queued host retirement
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes + erasure_bytes);
    drop(key);
    assert_eq!(pool.used_bytes().unwrap(), erasure_bytes);
    drop(root_identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
