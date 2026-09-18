use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::runtime::checkpoint::binding::binding_bytes;
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe, LinearFormat, SourceTensorEncoding, StoredDtype,
};
use eredu_runtime::{
    preflight_realtime_materialization_tasks, realtime_task_binding_plan, ExecutionGroupId,
    ParameterGroupOwner, RealtimeIdentity, RealtimeMaterializationComponent,
    RealtimeMaterializationTask, RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, WeightLoweringDescriptor, WeightLoweringKind,
};

fn identity(s: &str) -> RealtimeIdentity {
    RealtimeIdentity::new(s).unwrap()
}

fn tasks(source: &dyn CheckpointSource) -> Vec<RealtimeMaterializationTask> {
    [
        (0, "original.norm", "original.norm", "a"),
        (0, "original.unrelated", "original.unrelated", "b"),
        (1, "later.norm", "original.norm", "a"),
        (2, "last.norm", "later.norm", "a"),
        (2, "last.alias", "last.norm", "a"),
    ]
    .into_iter()
    .map(|(unit, target, owner, key)| {
        let recipe = DerivedWeightRecipe::source(key, TensorSelection::Full);
        let output = recipe.infer(source).unwrap();
        let component = RealtimeWeightComponentRequirement::new(
            identity(target),
            Some(identity(owner)),
            Some(identity(&format!("original-recipe-{key}"))),
            Some(recipe),
            Some(output),
            [identity(key)],
            vec![2],
            RealtimeWeightComponentRole::Primary,
        )
        .unwrap();
        let lowering = RealtimeWeightLoweringRequirement::new(
            identity(target),
            [component.clone()],
            WeightLoweringDescriptor::new(
                SourceTensorEncoding::Safetensors(StoredDtype::I32),
                LinearFormat::Dense,
                vec![2],
                vec![2],
                None,
            )
            .unwrap(),
            WeightLoweringKind::Derived,
        )
        .unwrap();
        RealtimeMaterializationTask::new(
            lowering,
            ParameterGroupOwner::execution_unit(ExecutionGroupId::new("units").unwrap(), unit),
            [RealtimeMaterializationComponent::new(
                component,
                [source.source_provenance(key).unwrap()],
            )
            .unwrap()],
        )
        .unwrap()
    })
    .collect()
}

fn replica_manager(
    source: Arc<SafetensorsWeightStore>,
    tier: MemoryTier,
    capacity: u64,
) -> ResidencyManager {
    let tasks = tasks(source.as_ref());
    let original = tasks.clone();
    preflight_realtime_materialization_tasks::<MlxNeuralBackend>(&tasks, source.as_ref()).unwrap();
    let (pinned, mut bindings) = realtime_task_binding_plan(&tasks, source.as_ref())
        .unwrap()
        .into_parts();
    assert_eq!(tasks, original);
    assert!(pinned.is_empty());
    let mut specs = Vec::new();
    let mut units = Vec::new();
    for (ordinal, name) in [(0, "original"), (1, "later"), (2, "last")] {
        let owner =
            ParameterGroupOwner::execution_unit(ExecutionGroupId::new("units").unwrap(), ordinal);
        let values = bindings.remove(&owner).unwrap();
        specs.push(spec(
            name,
            binding_bytes(&values).unwrap(),
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        ));
        units.push(unit(name, values));
    }
    assert!(bindings.is_empty());
    // Exact native capacities; do not use the legacy fixture's logical-budget
    // conversion. An absent opposite-tier limit is not a zero-byte claim.
    let config = OffloadConfig::new(
        (tier == MemoryTier::Device).then_some(capacity),
        (tier == MemoryTier::Host).then_some(capacity),
        1,
    )
    .unwrap();
    ResidencyManager::new(
        source,
        OffloadPlan::new(config, specs).unwrap(),
        units,
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap()
}

fn allocation(lease: &ResidentUnitLease, name: &str, tier: MemoryTier) -> safemlx::AllocationInfo {
    allocation_values(lease, name, tier, &[1, 2])
}

fn allocation_values(
    lease: &ResidentUnitLease,
    name: &str,
    tier: MemoryTier,
    expected: &[i32],
) -> safemlx::AllocationInfo {
    match tier {
        MemoryTier::Host => {
            assert_eq!(host_i32(lease, name), expected);
            lease.host_value(name).unwrap().allocation_info().unwrap()
        }
        MemoryTier::Device => {
            let array = lease.device_value(name).unwrap();
            assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), expected);
            array.allocation_info().unwrap().unwrap()
        }
        _ => unreachable!(),
    }
}

fn no_original_owner(manager: &ResidencyManager) {
    let report = manager.report().unwrap();
    let original = state(&report, "original");
    assert!(!original.host_resident() && !original.device_resident());
    assert_eq!(original.host_pins(), 0);
    assert_eq!(original.device_pins(), 0);
    assert!(manager
        .inner
        .state
        .lock()
        .unwrap()
        .alias_owner_pins
        .is_empty());
}

#[test]
fn realtime_unit_replicas_charge_actual_windows_without_whole_owner_pins() {
    for tier in [MemoryTier::Host, MemoryTier::Device] {
        let single_capacity = if tier == MemoryTier::Host {
            host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64
        } else {
            8
        };
        let (_dir, source) = fixture_store();
        let rejected = replica_manager(source.clone(), tier, single_capacity - 1);
        rejected.initialize().unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(matches!(
            rejected.acquire(&id("later"), tier),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::BudgetExhausted { .. }
            ))
        ));
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        assert_eq!(
            rejected.retained_storage().unwrap().byte_bound().unwrap(),
            Some(0)
        );
        no_original_owner(&rejected);
        drop(rejected);

        let exact = replica_manager(source.clone(), tier, single_capacity);
        exact.initialize().unwrap();
        let first = exact.acquire(&id("later"), tier).unwrap();
        assert_eq!(
            allocation(&first, "later.norm", tier).bytes() as u64,
            single_capacity
        );
        assert_eq!(
            exact.retained_storage().unwrap().byte_bound().unwrap(),
            Some(single_capacity)
        );
        no_original_owner(&exact);
        drop(first);
        assert!(exact.evict(&id("later"), tier).unwrap());
        let next = exact.acquire(&id("last"), tier).unwrap();
        let owner = allocation(&next, "last.norm", tier);
        let alias = allocation(&next, "last.alias", tier);
        assert_eq!(
            owner.identity(),
            alias.identity(),
            "same-unit aliases remain shared"
        );
        assert_eq!(
            exact.retained_storage().unwrap().byte_bound().unwrap(),
            Some(single_capacity)
        );
        assert_eq!(
            exact.report().unwrap().offload().resident_bytes().get(tier),
            single_capacity
        );
        no_original_owner(&exact);
        if tier == MemoryTier::Host {
            let promoted = exact.acquire(&id("last"), MemoryTier::Device).unwrap();
            let owner = allocation(&promoted, "last.norm", MemoryTier::Device);
            let alias = allocation(&promoted, "last.alias", MemoryTier::Device);
            assert_eq!(owner.identity(), alias.identity());
            assert_eq!(owner.bytes(), 8);
            assert_eq!(
                exact.retained_storage().unwrap().byte_bound().unwrap(),
                Some(single_capacity + owner.bytes() as u64)
            );
            no_original_owner(&exact);
            drop(promoted);
            assert!(exact.evict(&id("last"), MemoryTier::Device).unwrap());
        }
        drop(next);
        assert!(exact.evict(&id("last"), tier).unwrap());
        drop(exact);

        // Two simultaneous unit lifetimes count two real copies from the same
        // source tensor. Direct recipe reads fill each native destination and
        // do not populate the shard cache.
        let both = replica_manager(source.clone(), tier, single_capacity * 2);
        both.initialize().unwrap();
        #[cfg(not(feature = "cuda"))]
        let reads_before = source.source_diagnostics().unwrap();
        let first = both.acquire(&id("later"), tier).unwrap();
        let second = both.acquire(&id("last"), tier).unwrap();
        let first_allocation = allocation(&first, "later.norm", tier);
        let second_allocation = allocation(&second, "last.norm", tier);
        assert_ne!(first_allocation.identity(), second_allocation.identity());
        let actual = (first_allocation.bytes() + second_allocation.bytes()) as u64;
        assert_eq!(actual, single_capacity * 2);
        assert_eq!(
            both.retained_storage().unwrap().byte_bound().unwrap(),
            Some(actual)
        );
        assert_eq!(
            both.report().unwrap().offload().resident_bytes().get(tier),
            actual
        );
        no_original_owner(&both);
        #[cfg(not(feature = "cuda"))]
        {
            let reads_after = source.source_diagnostics().unwrap();
            assert_eq!(
                reads_after.physical_read_bytes,
                reads_before.physical_read_bytes + 16,
                "each independent replica reads the selected eight-byte tensor once"
            );
            assert!(reads_after.physical_reads >= reads_before.physical_reads + 2);
            assert_eq!(reads_after.currently_cached_shards, 0);
        }
        // CUDA retains the existing ordinary lease path: DirectRecipeRead is
        // deliberately unavailable there.
        #[cfg(feature = "cuda")]
        assert_eq!(
            source.source_diagnostics().unwrap().currently_cached_shards,
            1
        );
        drop((first, second));
    }

    // Preserve the existing global-owner route too. Promotion must consume its
    // already prepared shared array rather than copy the alias's host buffer.
    let (_dir, source) = fixture_store();
    let owner = unit(
        "owner",
        [binding("weight", "a", TensorSelection::Full, 8)
            .with_logical_target("shared.weight")
            .unwrap()],
    );
    let alias = unit(
        "alias",
        [
            WeightBinding::alias("shared", "shared.weight", 8).unwrap(),
            binding("local", "b", TensorSelection::Full, 8),
        ],
    );
    let host_capacity =
        host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
    let global = ResidencyManager::new(
        source,
        OffloadPlan::new(
            OffloadConfig::new(Some(16), Some(host_capacity * 2), 1).unwrap(),
            [
                spec("owner", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
        )
        .unwrap(),
        [owner, alias],
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap();
    global.initialize().unwrap();
    let host = global.acquire(&id("alias"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&host, "shared"), [1, 2]);
    assert_eq!(host_i32(&host, "local"), [3, 4]);
    let device = global.acquire(&id("alias"), MemoryTier::Device).unwrap();
    let original = global.acquire(&id("owner"), MemoryTier::Device).unwrap();
    assert_eq!(
        allocation(&device, "shared", MemoryTier::Device).identity(),
        allocation(&original, "weight", MemoryTier::Device).identity(),
    );
    assert_eq!(
        device
            .device_value("local")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        [3, 4]
    );
    assert_eq!(
        global.retained_storage().unwrap().byte_bound().unwrap(),
        Some(host_capacity * 2 + 16)
    );
    assert_eq!(
        global
            .report()
            .unwrap()
            .offload()
            .resident_bytes()
            .get(MemoryTier::Device),
        16
    );
}

#[test]
fn valid_cross_unit_cycle_materializes_canonical_rows_once_then_binds_both_aliases() {
    for tier in [MemoryTier::Host, MemoryTier::Device] {
        let (_dir, source) = fixture_store();
        let first = unit(
            "first",
            [
                binding("own_a", "a", TensorSelection::Full, 8)
                    .with_logical_target("first.owner")
                    .unwrap(),
                WeightBinding::alias("other_b", "second.owner", 8).unwrap(),
            ],
        );
        let second = unit(
            "second",
            [
                binding("own_b", "b", TensorSelection::Full, 8)
                    .with_logical_target("second.owner")
                    .unwrap(),
                WeightBinding::alias("other_a", "first.owner", 8).unwrap(),
            ],
        );
        let host = fixture_host_capacity(2);
        let manager = ResidencyManager::new(
            source,
            OffloadPlan::new(
                OffloadConfig::new(Some(16), Some(host), 1).unwrap(),
                [
                    spec("first", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                    spec("second", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                ],
            )
            .unwrap(),
            [first, second],
            cpu_stream(),
            cpu_stream(),
        )
        .unwrap();
        manager.initialize().unwrap();
        // Resolve individual tensor aliases to their canonical roots without
        // recursing through the two logical units.
        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("first"), 1)], tier)
            .unwrap();
        transfer.synchronize().unwrap();
        let first = &transfer.leases()[0];
        let second = manager.acquire(&id("second"), tier).unwrap();
        assert_eq!(
            allocation(first, "own_a", tier).identity(),
            allocation(&second, "other_a", tier).identity()
        );
        assert_eq!(
            allocation_values(first, "other_b", tier, &[3, 4]).identity(),
            allocation_values(&second, "own_b", tier, &[3, 4]).identity()
        );
        match tier {
            MemoryTier::Host => {
                assert_eq!(host_i32(first, "own_a"), [1, 2]);
                assert_eq!(host_i32(first, "other_b"), [3, 4]);
            }
            MemoryTier::Device => {
                assert_eq!(
                    first
                        .device_value("own_a")
                        .unwrap()
                        .evaluated()
                        .unwrap()
                        .as_slice::<i32>(),
                    [1, 2]
                );
                assert_eq!(
                    first
                        .device_value("other_b")
                        .unwrap()
                        .evaluated()
                        .unwrap()
                        .as_slice::<i32>(),
                    [3, 4]
                );
            }
            MemoryTier::Disk => unreachable!(),
        }
        assert_eq!(
            manager
                .report()
                .unwrap()
                .offload()
                .resident_bytes()
                .get(tier),
            if tier == MemoryTier::Host { host } else { 16 }
        );
        let again = manager.acquire(&id("first"), tier).unwrap();
        assert_eq!(
            allocation_values(first, "other_b", tier, &[3, 4]).identity(),
            allocation_values(&again, "other_b", tier, &[3, 4]).identity()
        );
    }
}
