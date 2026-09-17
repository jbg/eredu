use super::*;
use eredu_runtime::{WeightBinding, WeightBindingPlan, WeightResidency};
use std::{borrow::Cow, marker::PhantomData};

pub(super) fn weight_policy(mode: ExecutionResidency) -> WeightResidency {
    match mode {
        ExecutionResidency::FullyResident => WeightResidency::fully_resident(),
        ExecutionResidency::LayerwiseHost => WeightResidency::layerwise_host(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(None, None, 1).unwrap(),
            )
            .with_max_cached_shards(1),
        ),
        ExecutionResidency::DenseDiskStream => WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0)
                .unwrap()
                .with_max_cached_shards(1),
        ),
        _ => panic!("unhandled execution residency in Moshi fixture"),
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Evidence {
    mode: Option<ExecutionResidency>,
    unit_count: usize,
    pinned_parameters: usize,
    host_payloads: usize,
    host_payload_bytes: usize,
    cold_execution_units: usize,
    live: usize,
    peak: usize,
    acquired: Vec<usize>,
    evicted: Vec<usize>,
    // Actual prepared source diagnostics, measured separately from labels.
    unit_reads: Vec<(usize, u64, u64)>,
    reads_after_construction: u64,
    bytes_after_construction: u64,
    reads_after_execution: u64,
    bytes_after_execution: u64,
}
impl Evidence {
    pub(super) fn construction_complete(
        &mut self,
        source: &RetainedCheckpointSource,
    ) -> Result<(), Error> {
        assert_eq!(self.live, 0);
        self.cold_execution_units = self.acquired.len();
        let d = source.source_diagnostics().map_err(Error::backend)?;
        self.reads_after_construction = d.physical_reads;
        self.bytes_after_construction = d.physical_read_bytes;
        Ok(())
    }
    pub(super) fn execution_complete(
        &mut self,
        source: &RetainedCheckpointSource,
    ) -> Result<(), Error> {
        assert_eq!(
            self.live, 0,
            "canonical completion/abort must retire every unit"
        );
        assert_eq!(self.acquired, self.evicted);
        let d = source.source_diagnostics().map_err(Error::backend)?;
        self.reads_after_execution = d.physical_reads;
        self.bytes_after_execution = d.physical_read_bytes;
        Ok(())
    }
}

// Actual shared selected-task preflight, followed by the existing explicit
// source and inferred-output assertions. No physical payload is read here.
pub(super) fn validate_tasks(
    tasks: &[RealtimeMaterializationTask],
    source: &RetainedCheckpointSource,
) -> Result<(), Error> {
    eredu_runtime::preflight_realtime_materialization_tasks::<NumericBackend>(
        tasks,
        source.as_ref(),
    )
    .map_err(Error::backend)?;
    for task in tasks {
        for component in task.components() {
            for admitted in component.source_provenance() {
                assert_eq!(
                    &source
                        .source_provenance(&admitted.catalog_key)
                        .map_err(Error::backend)?,
                    admitted
                );
            }
            if let Some(recipe) = component.recipe() {
                assert_eq!(
                    Some(&recipe.infer(source.as_ref()).map_err(Error::backend)?),
                    component.recipe_output()
                );
            }
        }
    }
    Ok(())
}

type Values = BTreeMap<String, NumericTensor>;
pub(super) fn materialize(
    bindings: &[WeightBinding],
    source: &RetainedCheckpointSource,
    context: &NumericContext,
) -> Result<Values, Error> {
    let plan = WeightBindingPlan::new(bindings).map_err(Error::backend)?;
    let mut values = Values::new();
    for binding in plan.owners() {
        let recipe = binding.source_recipe();
        assert_eq!(
            recipe
                .infer(source.as_ref())
                .map_err(Error::backend)?
                .byte_len(),
            binding.expected_bytes()
        );
        let value = payload::recipe_value(&recipe, source.as_ref(), context)?;
        assert_eq!(value.data.len() as u64 * 4, binding.expected_bytes());
        assert!(values.insert(binding.name().to_owned(), value).is_none());
    }
    for (alias, owner) in plan.aliases() {
        let value = values
            .get(owner.name())
            .expect("canonical physical owner")
            .clone();
        assert!(values.insert(alias.name().to_owned(), value).is_none());
    }
    Ok(values)
}

enum UnitStorage {
    // Populated once before execution, independent of executable U values.
    Host(Values),
    // No NumericTensor or executable U is retained in this branch.
    Disk(Vec<WeightBinding>),
}
pub(super) struct Policy<U> {
    source: RetainedCheckpointSource,
    units: Vec<(ExecutionUnitAddress, UnitStorage)>,
    seen: BTreeSet<String>,
    bound: Rc<Cell<usize>>,
    evidence: Rc<RefCell<Evidence>>,
    _unit: PhantomData<U>,
}
// The field after unit records retirement only after U's fields have dropped.
struct Retired {
    ordinal: usize,
    evidence: Rc<RefCell<Evidence>>,
}
impl Drop for Retired {
    fn drop(&mut self) {
        let mut e = self.evidence.borrow_mut();
        assert_eq!(e.live, 1);
        e.live -= 1;
        e.evicted.push(self.ordinal);
    }
}
pub(super) struct Lease<U> {
    unit: U,
    _retired: Retired,
}
impl<U> Deref for Lease<U> {
    type Target = U;
    fn deref(&self) -> &U {
        &self.unit
    }
}
impl<U> DerefMut for Lease<U> {
    fn deref_mut(&mut self) -> &mut U {
        &mut self.unit
    }
}
impl<U> Policy<U> {
    pub(super) fn prepare<A>(
        architecture: &mut A,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        source: RetainedCheckpointSource,
        context: &NumericContext,
        bound: Rc<Cell<usize>>,
        evidence: Rc<RefCell<Evidence>>,
    ) -> Result<Self, Error>
    where
        A: LayeredArchitecture<NumericBackend, State, Unit = U>,
        A::Error: std::fmt::Display,
    {
        validate_tasks(tasks, &source)?;
        let (pinned, mut unit_bindings) =
            eredu_runtime::realtime_task_binding_plan(tasks, source.as_ref())
                .map_err(Error::backend)?
                .into_parts();
        let values = materialize(&pinned, &source, context)?;
        let mut seen = BTreeSet::new();
        architecture
            .visit_static_parameters_mut(&mut Bind {
                values: &values,
                seen: &mut seen,
            })
            .map_err(Error::backend)?;
        assert_eq!(seen, values.keys().cloned().collect());
        let mode = match selected.residency() {
            LayerWeightResidency::LayerwiseHost(_) => ExecutionResidency::LayerwiseHost,
            LayerWeightResidency::DenseDiskStream(_) => ExecutionResidency::DenseDiskStream,
            LayerWeightResidency::FullyResident => panic!("resident does not enter bounded policy"),
            _ => panic!("unhandled layer residency in Moshi fixture"),
        };
        let mut units = Vec::new();
        let mut host_payload_bytes = 0;
        for ordinal in 0..selected.execution_units().len() {
            let address = selected.execution_units().address(ordinal).unwrap();
            let group = selected
                .execution_units()
                .group_id(address.group())
                .unwrap()
                .clone();
            let owner = eredu_runtime::ParameterGroupOwner::execution_unit(group, address.index());
            let bindings = unit_bindings
                .remove(&owner)
                .expect("selected exact unit bindings");
            assert!(!bindings.is_empty());
            let storage = if mode == ExecutionResidency::LayerwiseHost {
                let values = materialize(&bindings, &source, context)?;
                host_payload_bytes += values.values().map(|v| v.data.len() * 4).sum::<usize>();
                UnitStorage::Host(values)
            } else {
                UnitStorage::Disk(bindings)
            };
            units.push((address, storage));
        }
        assert!(unit_bindings.is_empty());
        bound.set(seen.len());
        *evidence.borrow_mut() = Evidence {
            mode: Some(mode),
            unit_count: units.len(),
            pinned_parameters: seen.len(),
            host_payloads: if mode == ExecutionResidency::LayerwiseHost {
                units.len()
            } else {
                0
            },
            host_payload_bytes,
            ..Default::default()
        };
        Ok(Self {
            source,
            units,
            seen,
            bound,
            evidence,
            _unit: PhantomData,
        })
    }
}
impl<U: Parameterized<NumericTensor>> LayerwisePolicy<NumericBackend, U> for Policy<U> {
    type Lease = Lease<U>;
    type Error = Error;
    fn begin(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Error> {
        Ok(())
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<U, E>,
    {
        let (expected, storage) = self.units.get(ordinal).expect("original selected ordinal");
        assert_eq!(*expected, address);
        assert_eq!(self.evidence.borrow().live, 0, "one-unit execution window");
        let unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        {
            let mut e = self.evidence.borrow_mut();
            e.live += 1;
            e.peak = e.peak.max(e.live);
            e.acquired.push(ordinal);
        }
        let mut lease = Lease {
            unit,
            _retired: Retired {
                ordinal,
                evidence: self.evidence.clone(),
            },
        };
        let before = self
            .source
            .source_diagnostics()
            .map_err(Error::backend)
            .map_err(LayerwiseAcquireError::Policy)?;
        let values = match storage {
            UnitStorage::Host(values) => Cow::Borrowed(values),
            UnitStorage::Disk(bindings) => Cow::Owned(
                materialize(bindings, &self.source, context)
                    .map_err(LayerwiseAcquireError::Policy)?,
            ),
        };
        let mut bound = BTreeSet::new();
        lease.visit_parameters_mut(&mut Bind {
            values: &values,
            seen: &mut bound,
        });
        assert_eq!(bound, values.keys().cloned().collect());
        self.seen.extend(bound);
        self.bound.set(self.seen.len());
        let after = self
            .source
            .source_diagnostics()
            .map_err(Error::backend)
            .map_err(LayerwiseAcquireError::Policy)?;
        self.evidence.borrow_mut().unit_reads.push((
            ordinal,
            after.physical_reads - before.physical_reads,
            after.physical_read_bytes - before.physical_read_bytes,
        ));
        Ok(lease)
    }
    fn complete<'a, S, C>(
        &mut self,
        _: usize,
        _: ExecutionUnitAddress,
        lease: Self::Lease,
        _: &'a NumericTensor,
        _: S,
        _: C,
        _: &NumericContext,
    ) -> Result<(), Error>
    where
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        drop(lease); // Scalar computation is synchronous; the shared abort path also drops this lease.
        Ok(())
    }
    fn finish(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Error> {
        Ok(())
    }
}

pub(super) fn same_report(a: &Report, b: &Report) {
    assert_eq!(a.status, b.status);
    assert_eq!(a.committed_work, b.committed_work);
    assert_eq!(a.completion_calls, b.completion_calls);
    assert_eq!(a.executions, b.executions);
    assert_eq!(a.frames.len(), b.frames.len());
    for (a, b) in a.frames.iter().zip(&b.frames) {
        same_snapshot(&a.snapshot, &b.snapshot);
        same_values(&a.values, &b.values);
        same_outputs(&a.outputs, &b.outputs);
        same_outputs(&a.diagnostics, &b.diagnostics);
    }
    same_snapshot(&a.final_state, &b.final_state);
    same_values(&a.failed_values, &b.failed_values);
    match (&a.retired_state, &b.retired_state) {
        (Some(a), Some(b)) => same_state(a, b),
        (None, None) => (),
        _ => panic!("different canonical retirement state"),
    }
}
pub(super) fn storage_proof(report: &Report, mode: ExecutionResidency) {
    let e = &report.residency;
    assert_eq!(e.mode, Some(mode));
    assert!(e.unit_count >= 4 && e.pinned_parameters > 0);
    assert_eq!(
        e.cold_execution_units, 0,
        "no eager executable bounded units"
    );
    assert_eq!(e.live, 0);
    assert_eq!(e.peak, 1);
    assert_eq!(e.acquired, e.evicted);
    assert_eq!(e.acquired.len(), e.unit_reads.len());
    assert!(e.acquired.len() > e.unit_count);
    match mode {
        ExecutionResidency::LayerwiseHost => {
            assert_eq!(e.host_payloads, e.unit_count);
            assert!(e.host_payload_bytes > 0);
            assert_eq!(e.reads_after_execution, e.reads_after_construction);
            assert_eq!(e.bytes_after_execution, e.bytes_after_construction);
            assert!(e
                .unit_reads
                .iter()
                .all(|(_, reads, bytes)| *reads == 0 && *bytes == 0));
        }
        ExecutionResidency::DenseDiskStream => {
            assert_eq!(e.host_payloads, 0);
            assert_eq!(e.host_payload_bytes, 0);
            assert!(e.reads_after_execution > e.reads_after_construction);
            assert!(e.bytes_after_execution > e.bytes_after_construction);
            // Every reacquisition follows actual executable eviction and has a
            // fresh bounded physical read; no retained whole-model payload map.
            assert!(e
                .unit_reads
                .iter()
                .all(|(_, reads, bytes)| *reads > 0 && *bytes > 0));
            assert!(
                e.unit_reads
                    .iter()
                    .filter(|(ordinal, _, _)| *ordinal == 0)
                    .count()
                    > 1
            );
        }
        ExecutionResidency::FullyResident => unreachable!(),
        _ => panic!("unhandled execution residency in Moshi storage proof"),
    }
}

// Error adaptation only; lease ownership and rollback remain in ResidentUnitWindow.
pub(super) struct Resident<U>(pub(super) ResidentUnitWindow<U>);
impl<U> LayerwisePolicy<NumericBackend, U> for Resident<U> {
    type Lease = <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::Lease;
    type Error = Error;
    fn retained_value_slot_bound(&self) -> Option<usize>
    where
        U: Parameterized<NumericTensor>,
    {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::retained_value_slot_bound(
            &self.0,
        )
    }
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&NumericTensor)) -> bool
    where
        U: Parameterized<NumericTensor>,
    {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::visit_retained_values(
            &self.0, visitor,
        )
    }
    fn begin(&mut self, initial: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::begin(
            &mut self.0,
            initial,
            context,
        )
        .map_err(Error::backend)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<U, E>,
    {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::acquire(
            &mut self.0,
            ordinal,
            address,
            build,
            context,
        )
        .map_err(|e| match e {
            LayerwiseAcquireError::Architecture(e) => LayerwiseAcquireError::Architecture(e),
            LayerwiseAcquireError::Policy(e) => LayerwiseAcquireError::Policy(Error::backend(e)),
        })
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &NumericContext,
    ) {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::abort(
            &mut self.0,
            active,
            context,
        )
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        forward: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::complete(
            &mut self.0,
            ordinal,
            address,
            lease,
            output,
            state,
            forward,
            context,
        )
        .map_err(Error::backend)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <ResidentUnitWindow<U> as LayerwisePolicy<NumericBackend, U>>::finish(
            &mut self.0,
            output,
            context,
        )
        .map_err(Error::backend)
    }
}
