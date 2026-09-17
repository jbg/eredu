use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeMetadata},
    store::{CheckpointSource, MemoryWeightStore, TensorSelection},
};
use eredu_runtime::{
    preflight_realtime_materialization_tasks, realtime_task_binding_plan, ExecutionGroupId,
    ParameterGroupOwner, RealtimeIdentity, RealtimeMaterializationComponent,
    RealtimeMaterializationTask, RealtimeModelContractError, RealtimeWeightComponentRequirement,
    RealtimeWeightComponentRole, RealtimeWeightLoweringRequirement, WeightBindingPlan,
    WeightLoweringDescriptor,
};

fn identity(s: &str) -> RealtimeIdentity {
    RealtimeIdentity::new(s).unwrap()
}

fn owner(unit: usize) -> ParameterGroupOwner {
    ParameterGroupOwner::execution_unit(ExecutionGroupId::new("depth").unwrap(), unit)
}

fn source() -> CountingSource {
    CountingSource {
        inner: MemoryWeightStore::from_safetensors(["physical", "foreign"].map(|name| {
            (
                name.to_owned(),
                safetensors::Dtype::F32,
                vec![2, 2],
                [0.3_f32, 0.7, 1.1, 1.9]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )
        }))
        .unwrap(),
        leases: Arc::new(AtomicUsize::new(0)),
    }
}

fn task(
    source: &dyn CheckpointSource,
    target: &str,
    recipe_owner: &str,
    key: &str,
    unit: usize,
    kind: WeightLoweringKind,
) -> RealtimeMaterializationTask {
    let component = RealtimeWeightComponentRequirement::new(
        identity(target),
        Some(identity(recipe_owner)),
        Some(identity("shared-source-recipe")),
        Some(DerivedWeightRecipe::source(key, TensorSelection::Full)),
        Some(RecipeMetadata {
            shape: vec![2, 2],
            dtype: RecipeDtype::F32,
            byte_len: 16,
        }),
        [identity(key)],
        vec![2, 2],
        RealtimeWeightComponentRole::Primary,
    )
    .unwrap();
    let lowering = RealtimeWeightLoweringRequirement::new(
        identity(target),
        [component.clone()],
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::F32),
            LinearFormat::Dense,
            vec![2, 2],
            vec![2, 2],
            Some(1),
        )
        .unwrap(),
        kind,
    )
    .unwrap();
    RealtimeMaterializationTask::new(
        lowering,
        owner(unit),
        [RealtimeMaterializationComponent::new(
            component,
            [source.source_provenance(key).unwrap()],
        )
        .unwrap()],
    )
    .unwrap()
}

#[test]
fn realtime_alias_chains_materialize_independent_units_with_local_sharing() {
    let source = source();
    let entry = |target, recipe_owner, unit| {
        task(
            &source,
            target,
            recipe_owner,
            "physical",
            unit,
            WeightLoweringKind::Derived,
        )
    };
    // Alias-first ordering and a chain which leaves and re-enters a partition.
    let tasks = vec![
        entry("later.second", "later.first", 1),
        entry("later.first", "original", 1),
        entry("original.alias", "later.second", 0),
        entry("original", "original", 0),
    ];
    let unchanged = tasks.clone();
    preflight_realtime_materialization_tasks::<FakeBackend>(&tasks, &source).unwrap();
    let (pinned, mut units) = realtime_task_binding_plan(&tasks, &source)
        .unwrap()
        .into_parts();
    assert!(pinned.is_empty());
    assert_eq!(
        tasks, unchanged,
        "logical owner and original source are retained"
    );
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);

    // The later unit is genuinely materializable before the original unit.
    // Its same-unit aliases share one owner; neither binding needs unit zero.
    let later = units.remove(&owner(1)).unwrap();
    let plan = WeightBindingPlan::new(&later).unwrap();
    assert_eq!(plan.owners().count(), 1);
    assert_eq!(plan.aliases().count(), 1);
    assert_eq!(
        plan.owners()
            .map(WeightBinding::expected_bytes)
            .sum::<u64>(),
        16
    );
    let values = materialize_bindings::<FakeBackend>(&source, &later, &()).unwrap();
    assert_eq!(values.len(), 2);
    for name in ["later.first", "later.second"] {
        assert!(values.contains(&eredu_nn::ParameterId::new(name).unwrap()));
    }
    drop(values);
    let original = units.remove(&owner(0)).unwrap();
    let plan = WeightBindingPlan::new(&original).unwrap();
    assert_eq!(plan.owners().next().unwrap().name(), "original");
    assert_eq!(plan.aliases().count(), 1);
    assert_eq!(
        materialize_bindings::<FakeBackend>(&source, &original, &())
            .unwrap()
            .len(),
        2
    );
    assert!(units.is_empty());
    let repeated = materialize_bindings::<FakeBackend>(&source, &later, &()).unwrap();
    assert_eq!(repeated.len(), 2, "later reacquisition owns its own recipe");
    assert_eq!(
        original
            .iter()
            .chain(&later)
            .filter(|b| !b.is_alias())
            .map(WeightBinding::expected_bytes)
            .sum::<u64>(),
        32,
        "two real binding lifetimes, despite one sixteen-byte checkpoint tensor"
    );
}

#[test]
fn realtime_alias_contract_rejects_foreign_missing_cyclic_and_unpublished_sources() {
    let source = source();
    let entry = |target, recipe_owner, key, unit| {
        task(
            &source,
            target,
            recipe_owner,
            key,
            unit,
            WeightLoweringKind::Derived,
        )
    };
    let root = entry("original", "original", "physical", 0);
    let cases = [
        vec![root.clone(), entry("alias", "absent", "physical", 1)],
        vec![
            entry("a", "b", "physical", 0),
            entry("b", "a", "physical", 1),
        ],
        vec![root.clone(), entry("alias", "original", "foreign", 1)],
        vec![root.clone(), root.clone()],
    ];
    for tasks in cases {
        assert!(matches!(
            preflight_realtime_materialization_tasks::<FakeBackend>(&tasks, &source),
            Err(RealtimeModelContractError::BindingPlan { .. })
        ));
        assert!(matches!(
            realtime_task_binding_plan(&tasks, &source),
            Err(RealtimeModelContractError::BindingPlan { .. })
        ));
        assert_eq!(source.leases.load(Ordering::SeqCst), 0);
    }

    let mut provenance = root.components()[0].source_provenance().to_vec();
    provenance[0].physical_tensor = "equal-sized-foreign-owner".into();
    let foreign = RealtimeMaterializationTask::new(
        root.lowering().clone(),
        root.owner().clone(),
        [RealtimeMaterializationComponent::new(
            root.components()[0].requirement().clone(),
            provenance,
        )
        .unwrap()],
    )
    .unwrap();
    assert!(
        preflight_realtime_materialization_tasks::<FakeBackend>(&[foreign.clone()], &source)
            .is_err()
    );
    assert!(realtime_task_binding_plan(&[foreign], &source).is_err());

    // Same byte extent and key, different actual shape: inferred metadata,
    // not just owner names or byte totals, must match the selected component.
    let reshaped = CountingSource {
        inner: MemoryWeightStore::from_safetensors([(
            "physical".to_owned(),
            safetensors::Dtype::F32,
            vec![1, 4],
            [0.3_f32, 0.7, 1.1, 1.9]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )])
        .unwrap(),
        leases: source.leases.clone(),
    };
    assert!(
        preflight_realtime_materialization_tasks::<FakeBackend>(&[root.clone()], &reshaped)
            .is_err()
    );
    assert!(realtime_task_binding_plan(&[root], &reshaped).is_err());
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);

    let transformed = [task(
        &source,
        "unpublished",
        "unpublished",
        "physical",
        0,
        WeightLoweringKind::DerivedTransform,
    )];
    preflight_realtime_materialization_tasks::<FakeBackend>(&transformed, &source).unwrap();
    assert!(matches!(
        realtime_task_binding_plan(&transformed, &source),
        Err(RealtimeModelContractError::BindingPlan { .. })
    ));
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

struct PublishedOverlay<'a> {
    original: &'a dyn CheckpointSource,
    output: MemoryWeightStore,
    authoritative: bool,
    change_passthrough: bool,
    reads: AtomicUsize,
}
impl CheckpointSource for PublishedOverlay<'_> {
    fn source_keys(&self) -> Vec<String> {
        self.original
            .source_keys()
            .into_iter()
            .chain(self.output.source_keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        if self.output.source_keys().iter().any(|k| k == key) {
            self.output.source_metadata(key)
        } else {
            self.original.source_metadata(key)
        }
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        let mut actual = if self.output.source_keys().iter().any(|k| k == key) {
            self.output.source_provenance(key)?
        } else {
            self.original.source_provenance(key)?
        };
        if self.change_passthrough && key == "foreign" {
            actual.physical_tensor = "same-sized-foreign-source".into();
        }
        Ok(actual)
    }
    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.output.source_keys().iter().any(|k| k == &request.key) {
            self.output.acquire_lease(request)
        } else {
            self.original.acquire_lease(request)
        }
    }
    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        self.original.source_diagnostics()
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.authoritative && self.output.source_keys().iter().any(|k| k == key)
    }
}

fn packed_source() -> CountingSource {
    CountingSource {
        inner: MemoryWeightStore::from_safetensors([
            (
                "weight".to_owned(),
                safetensors::Dtype::F32,
                vec![2, 32],
                (0..64)
                    .flat_map(|n| (0.1f32 + n as f32 / 64.0).to_le_bytes())
                    .collect(),
            ),
            (
                "foreign".to_owned(),
                safetensors::Dtype::F32,
                vec![2, 2],
                [0.3f32, 0.7, 1.1, 1.9]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
        ])
        .unwrap(),
        leases: Arc::new(AtomicUsize::new(0)),
    }
}

fn packed_tasks(source: &dyn CheckpointSource, affine: bool) -> Vec<RealtimeMaterializationTask> {
    let mut components = vec![RealtimeWeightComponentRequirement::new(
        identity("weight"),
        Some(identity("weight")),
        Some(identity("original-dense-recipe")),
        Some(DerivedWeightRecipe::source("weight", TensorSelection::Full)),
        Some(RecipeMetadata {
            shape: vec![2, 32],
            dtype: RecipeDtype::F32,
            byte_len: 256,
        }),
        [identity("weight")],
        vec![2, 32],
        RealtimeWeightComponentRole::Primary,
    )
    .unwrap()];
    for (target, role) in [("weight.scales", RealtimeWeightComponentRole::Scale)]
        .into_iter()
        .chain(affine.then_some(("weight.biases", RealtimeWeightComponentRole::AffineBias)))
    {
        components.push(
            RealtimeWeightComponentRequirement::new(
                identity(target),
                None,
                None,
                None,
                None,
                [],
                vec![2, 1],
                role,
            )
            .unwrap(),
        );
    }
    let format = if affine {
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap())
    } else {
        LinearFormat::MxFp4
    };
    let lowering = RealtimeWeightLoweringRequirement::new(
        identity("weight"),
        components.clone(),
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::F32),
            format,
            vec![2, 32],
            vec![2, 32],
            Some(1),
        )
        .unwrap(),
        WeightLoweringKind::Transform,
    )
    .unwrap();
    let components = components
        .into_iter()
        .map(|c| {
            let provenance = if c.role() == RealtimeWeightComponentRole::Primary {
                vec![source.source_provenance("weight").unwrap()]
            } else {
                Vec::new()
            };
            RealtimeMaterializationComponent::new(c, provenance).unwrap()
        })
        .collect::<Vec<_>>();
    vec![
        RealtimeMaterializationTask::new(lowering, owner(0), components).unwrap(),
        task(
            source,
            "side",
            "side",
            "foreign",
            1,
            WeightLoweringKind::Derived,
        ),
    ]
}

fn published<'a>(
    original: &'a dyn CheckpointSource,
    affine: bool,
    scale_dtype: safetensors::Dtype,
    bad_primary_shape: bool,
    bad_primary_dtype: bool,
    bad_scale_shape: bool,
) -> PublishedOverlay<'a> {
    let primary_dtype = if bad_primary_dtype {
        safetensors::Dtype::F32
    } else {
        safetensors::Dtype::U32
    };
    let primary_shape = if bad_primary_shape {
        vec![4, 2]
    } else {
        vec![2, 4]
    };
    let scale_shape = if bad_scale_shape {
        vec![1, 2]
    } else {
        vec![2, 1]
    };
    let mut outputs = vec![(
        "weight".to_owned(),
        primary_dtype,
        primary_shape,
        [0x76543210u32; 8]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    )];
    for name in ["weight.scales"]
        .into_iter()
        .chain(affine.then_some("weight.biases"))
    {
        outputs.push((
            name.to_owned(),
            scale_dtype,
            scale_shape.clone(),
            vec![1u8; 2 * scale_dtype.size()],
        ));
    }
    PublishedOverlay {
        original,
        output: MemoryWeightStore::from_safetensors(outputs).unwrap(),
        authoritative: true,
        change_passthrough: false,
        reads: AtomicUsize::new(0),
    }
}

#[test]
fn retained_realtime_plan_accepts_authoritative_packed_replacement_and_precision() {
    let source = packed_source();
    for (affine, dtype) in [
        (true, safetensors::Dtype::F16),
        (true, safetensors::Dtype::BF16),
        (true, safetensors::Dtype::F32),
        (false, safetensors::Dtype::U8),
    ] {
        let tasks = packed_tasks(&source, affine);
        let unchanged = tasks.clone();
        let prepared =
            preflight_realtime_materialization_tasks::<FakeBackend>(&tasks, &source).unwrap();
        let output = published(&source, affine, dtype, false, false, false);
        assert_ne!(
            source.source_provenance("weight").unwrap(),
            output.source_provenance("weight").unwrap()
        );
        // A transformed catalog cannot substitute for the original preflight.
        assert!(realtime_task_binding_plan(&tasks, &output).is_err());
        let (pinned, units) = prepared.materialized(&output).unwrap().into_parts();
        assert!(pinned.is_empty());
        assert_eq!(units[&owner(0)].len(), if affine { 3 } else { 2 });
        let weight = units[&owner(0)]
            .iter()
            .find(|b| b.name() == "weight")
            .unwrap();
        assert_eq!(weight.expected_bytes(), 32);
        assert_eq!(units[&owner(1)][0].expected_bytes(), 16);
        assert_eq!(tasks, unchanged);
        assert_eq!(output.reads.load(Ordering::SeqCst), 0);
    }
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn retained_realtime_plan_rejects_unpublished_wrong_geometry_encoding_and_passthrough() {
    let source = packed_source();
    let tasks = packed_tasks(&source, true);
    for case in 0..6 {
        let prepared =
            preflight_realtime_materialization_tasks::<FakeBackend>(&tasks, &source).unwrap();
        let mut output = published(
            &source,
            true,
            if case == 4 {
                safetensors::Dtype::U32
            } else {
                safetensors::Dtype::F32
            },
            case == 1,
            case == 2,
            case == 3,
        );
        output.authoritative = case != 0;
        output.change_passthrough = case == 5;
        assert!(
            matches!(
                prepared.materialized(&output),
                Err(RealtimeModelContractError::BindingPlan { .. })
            ),
            "case {case}"
        );
        assert_eq!(output.reads.load(Ordering::SeqCst), 0);
    }
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn retained_realtime_plan_uses_final_group_geometry_for_source_backed_companions() {
    // This checks the neutral post-transform contract. It does not enable
    // native packed-to-packed transcoding, whose preflight still rejects it.
    let packed = || {
        [0x76543210u32; 16]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let scalar = |count| {
        (0..count)
            .flat_map(|n| (0.25f32 + n as f32).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let source = CountingSource {
        inner: MemoryWeightStore::from_safetensors([
            (
                "weight".to_owned(),
                safetensors::Dtype::U32,
                vec![2, 8],
                packed(),
            ),
            (
                "weight.scales".to_owned(),
                safetensors::Dtype::F32,
                vec![2, 2],
                scalar(4),
            ),
            (
                "weight.biases".to_owned(),
                safetensors::Dtype::F32,
                vec![2, 2],
                scalar(4),
            ),
        ])
        .unwrap(),
        leases: Arc::new(AtomicUsize::new(0)),
    };
    let requirements = [
        (
            "weight",
            RealtimeWeightComponentRole::Primary,
            vec![2, 8],
            RecipeDtype::U32,
            64,
        ),
        (
            "weight.scales",
            RealtimeWeightComponentRole::Scale,
            vec![2, 2],
            RecipeDtype::F32,
            16,
        ),
        (
            "weight.biases",
            RealtimeWeightComponentRole::AffineBias,
            vec![2, 2],
            RecipeDtype::F32,
            16,
        ),
    ]
    .map(|(name, role, shape, dtype, byte_len)| {
        RealtimeWeightComponentRequirement::new(
            identity(name),
            Some(identity(name)),
            Some(identity(name)),
            Some(DerivedWeightRecipe::source(name, TensorSelection::Full)),
            Some(RecipeMetadata {
                shape: shape.clone(),
                dtype,
                byte_len,
            }),
            [identity(name)],
            shape,
            role,
        )
        .unwrap()
    });
    let lowering = RealtimeWeightLoweringRequirement::new(
        identity("weight"),
        requirements.clone(),
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(64, 4).unwrap()),
            vec![2, 8],
            vec![2, 64],
            Some(1),
        )
        .unwrap(),
        WeightLoweringKind::Transform,
    )
    .unwrap();
    let components = requirements
        .into_iter()
        .map(|requirement| {
            let provenance = source
                .source_provenance(requirement.target().as_str())
                .unwrap();
            RealtimeMaterializationComponent::new(requirement, [provenance]).unwrap()
        })
        .collect::<Vec<_>>();
    let tasks = [RealtimeMaterializationTask::new(lowering, owner(0), components).unwrap()];
    for stale_group_geometry in [false, true] {
        let prepared =
            preflight_realtime_materialization_tasks::<FakeBackend>(&tasks, &source).unwrap();
        let columns = if stale_group_geometry { 2 } else { 1 };
        let output = PublishedOverlay {
            original: &source,
            output: MemoryWeightStore::from_safetensors([
                (
                    "weight".to_owned(),
                    safetensors::Dtype::U32,
                    vec![2, 8],
                    packed(),
                ),
                (
                    "weight.scales".to_owned(),
                    safetensors::Dtype::F32,
                    vec![2, columns],
                    scalar(2 * columns),
                ),
                (
                    "weight.biases".to_owned(),
                    safetensors::Dtype::F32,
                    vec![2, columns],
                    scalar(2 * columns),
                ),
            ])
            .unwrap(),
            authoritative: true,
            change_passthrough: false,
            reads: AtomicUsize::new(0),
        };
        let result = prepared.materialized(&output);
        if stale_group_geometry {
            assert!(matches!(
                result,
                Err(RealtimeModelContractError::BindingPlan { .. })
            ));
        } else {
            let (pinned, units) = result.unwrap().into_parts();
            assert!(pinned.is_empty());
            let bindings = &units[&owner(0)];
            assert_eq!(bindings.len(), 3);
            for binding in bindings {
                assert_eq!(
                    binding.expected_bytes(),
                    if binding.name() == "weight" { 64 } else { 8 }
                );
            }
        }
        assert_eq!(output.reads.load(Ordering::SeqCst), 0);
    }
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}
