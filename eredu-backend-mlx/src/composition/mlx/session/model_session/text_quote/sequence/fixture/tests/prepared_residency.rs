//! Exact native factory owners shared by the serial source/residency fixtures.
use super::*;

pub(in crate::composition::mlx::session::model_session::text_quote) struct PreparedResidencyFixture {
    // Keep the actual initialized source/worker owners live across model loads.
    // This target is never materialized, so it contains no numerical reference.
    target: eredu_core::ExecutionPlanTarget<MlxBackend<'static>>,
    _artifact: tempfile::TempDir,
    pub pool: WorkingMemoryPool,
    pub baseline: u64,
}
impl PreparedResidencyFixture {
    pub fn new() -> Self {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (target, artifact) = target(0, None);
        disk::reclaim();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0,
            "retained cold target must own only its qualified stream/source accounts");
        let baseline = pool.used_bytes().unwrap();
        Self { target, _artifact: artifact, pool, baseline }
    }
    pub fn stream(&self) -> &Stream { self.target.backend().stream() }
    /// Reads detached payload telemetry without retaining another source owner.
    /// The public residency report describes the metadata catalog separately.
    pub fn disk_read_bytes(&self, runtime: &Runtime, residency: usize) -> Option<u64> {
        if residency != 2 { return None; }
        Some(Self::actual_disk_read_bytes(runtime))
    }
    /// Borrows the selected foreground payload reader, without a fixture owner.
    pub fn actual_disk_read_bytes(runtime: &Runtime) -> u64 {
        let workspace = runtime.session().payload.model.layerwise_workspace().unwrap().unwrap();
        let mut total = workspace.detached_physical_read_bytes(0)
            .expect("disk fixture must retain an actual detached payload source");
        let mut ordinal = 1;
        while let Some(bytes) = workspace.detached_physical_read_bytes(ordinal) {
            total = total.checked_add(bytes).unwrap();
            ordinal += 1;
        }
        total
    }
    #[track_caller]
    pub fn assert_disk_read_progress(&self, runtime: &Runtime, residency: usize, before: Option<u64>) {
        match (before, self.disk_read_bytes(runtime, residency)) {
            (Some(before), Some(after)) => {
                use eredu_core::residency::{MemoryTier, ResidencyPolicy};
                let report = runtime.session().residency_report().unwrap().unwrap();
                let layers = report.units().iter().filter(|unit| unit.planned_tier() == MemoryTier::Disk)
                    .collect::<Vec<_>>();
                assert_eq!(layers.len(), 3);
                let depth = runtime.session().payload.model.inference_blueprint().unwrap()
                    .selected().text_realization().residency().device_depth(layers.len());
                assert!(depth < layers.len());
                assert!(layers.iter().all(|unit| !unit.host_resident()));
                let live = layers.iter().filter(|unit| unit.device_resident()).count();
                assert!(live > 0 && live <= depth, "retained disk units exceed the selected device window");
                assert!(report.offload().tier_evictions(MemoryTier::Device).count() > 0);
                let pinned = report.units().iter().filter(|unit| unit.policy() == ResidencyPolicy::Pinned).count();
                assert!(report.offload().peak_resident_units().get(MemoryTier::Device) <= pinned + depth);
                assert!(after > before,
                    "each disk request must perform physical payload I/O: before={before}, after={after}");
            },
            (None, None) => {},
            _ => panic!("detached source telemetry changed availability across a request"),
        }
    }
    #[track_caller]
    pub fn load(&self, residency: usize, maximum_positions: Option<usize>)
        -> (Runtime, tempfile::TempDir, u64) {
        let before_target = self.pool.used_bytes().unwrap();
        assert!(before_target >= self.baseline);
        assert_eq!(self.pool.unquoted_owner_count().unwrap(), 0);
        let (target, artifact) = target(residency, maximum_positions);
        // A factory target creates actual GPU/CPU process stream and worker
        // registrations. Their native process nodes retain initialization
        // custody after the C wrappers retire. Measure that source birth here,
        // before any model allocation, not after a generation has run.
        let target_baseline = self.pool.used_bytes().unwrap();
        let source_birth = target_baseline.checked_sub(before_target).unwrap();
        assert!(source_birth > 0, "new prepared target must retain its actual process source accounts");
        assert_eq!(self.pool.unquoted_owner_count().unwrap(), 0,
            "qualified target source birth precedes all model materialization");
        let runtime = target.into_runtime().unwrap();
        runtime.backend().validate_original_stream_owners().unwrap();
        assert!(runtime.backend().memory_pool().same_domain(&self.pool));
        assert!(runtime.session().payload.model.has_published_idle_storage(),
            "loaded residency {residency} must have published initial storage and retired its loading authority");
        let caller = std::panic::Location::caller();
        let started = std::time::Instant::now();
        let mut reported = false;
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            let unquoted = self.pool.unquoted_owner_count().unwrap();
            if unquoted != 0 && !reported && started.elapsed().as_secs() >= 9 {
                eprintln!("prepared residency load at {caller}: residency={residency}, used={}, baseline={}, unquoted={unquoted}",
                    self.pool.used_bytes().unwrap(), self.baseline);
                reported = true;
            }
            unquoted == 0
        });
        assert_eq!(runtime.session().payload.model.erased().state_snapshot().len(), 3);
        if residency != 0 {
            use eredu_core::residency::{MemoryTier, ResidencyPolicy};
            let workspace = runtime.session().payload.model.layerwise_workspace().unwrap().unwrap();
            assert_eq!(workspace.layout().len(), 3);
            // Canonical loading retains the prepared host/foreground source.
            // The legacy ordinary disk receipt is not that source's witness.
            assert!(workspace.disk_receipt().is_none());
            let mut native_reads = 0usize;
            assert!(workspace.visit_native_copy_layouts(|rank, _dtype| {
                assert!(rank > 0, "fixture weight reads have actual tensor geometry");
                native_reads += 1;
                Ok::<_, std::convert::Infallible>(())
            }).unwrap(), "selected source must expose its qualified native read/copy plans");
            assert!(native_reads > 0, "selected source retains actual native parameter reads");
            let report = runtime.session().residency_report().unwrap().unwrap();
            let tier = if residency == 2 { MemoryTier::Disk } else { MemoryTier::Host };
            let layers = report.units().iter().filter(|unit| unit.planned_tier() == tier).collect::<Vec<_>>();
            assert_eq!(layers.len(), 3);
            assert_eq!(layers.iter().map(|unit| unit.id()).collect::<std::collections::BTreeSet<_>>().len(), 3);
            let pinned = report.units().iter().filter(|unit| unit.policy() == ResidencyPolicy::Pinned).collect::<Vec<_>>();
            assert!(!pinned.is_empty());
            assert!(pinned.iter().all(|unit| unit.planned_tier() == MemoryTier::Device
                && unit.device_resident() && unit.expected_bytes() > 0));
            assert_eq!(report.units().len(), layers.len() + pinned.len());
            if residency == 2 {
                assert!(layers.iter().all(|unit| !unit.host_resident() && !unit.device_resident()));
                // These reports belong to the metadata-only detached catalogs,
                // not to the payload reader whose counters advance below.
                for catalog in std::iter::once(report.weight_store()).chain(report.unit_sources().values()) {
                    assert_eq!(catalog.backend, eredu_checkpoint::store::WeightStoreBackend::Memory);
                    assert_eq!(catalog.physical_reads, 0);
                    assert_eq!(catalog.physical_read_bytes, 0);
                    assert!(catalog.payload_shard_paths.is_empty());
                }
                assert!(self.disk_read_bytes(&runtime, residency).is_some());
                assert!(report.units().iter().map(|unit| unit.expected_bytes()).sum::<u64>() < 1 << 30);
                assert!(workspace.materialization().bytes().unwrap() > 0);
            } else {
                assert!(layers.iter().all(|unit| unit.policy() == ResidencyPolicy::Windowed));
            }
        }
        (runtime, artifact, target_baseline)
    }
}
fn target(residency: usize, maximum_positions: Option<usize>)
    -> (eredu_core::ExecutionPlanTarget<MlxBackend<'static>>, tempfile::TempDir) {
    use eredu_core::ResidencyPlan;
    let artifact = if residency == 2 { disk::artifact() } else { host::artifact("llama") };
    if let Some(maximum) = maximum_positions {
        let path = artifact.path().join("config.json");
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["max_position_embeddings"] = maximum.into();
        std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
    let placement = match residency {
        0 => ResidencyPlan::FullyResident,
        1 => ResidencyPlan::LayerwiseHost {
            device_layer_window: 1, device_budget_bytes: Some(u64::MAX), host_budget_bytes: Some(u64::MAX),
        },
        2 => ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 30, host_budget_bytes: 0, host_lookahead: 0, background_queue: 0,
        },
        _ => unreachable!(),
    };
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
    ).with_residency(placement);
    let inspection = eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(artifact.path()).unwrap();
    let factory = crate::MlxBackendFactory::default();
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    target.backend().validate_original_stream_owners().unwrap();
    (target, artifact)
}
