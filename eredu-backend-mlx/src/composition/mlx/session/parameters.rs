//! Native loaded-slot operations. Architecture names and geometry come from traversal.
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::composition::mlx::replicated_text::{
    ParameterOwnerCounts, ParameterOwnerRole, ParameterOwnerSourceError,
};
use eredu_core::{capture::*, intervention::InterventionDtype, parameters::*};
use eredu_nn::{ParameterMetadata, ParameterSlotVisitor};
use eredu_runtime::parameter_operations::ParameterReplacementValues;
use safemlx::ops::indexing::{ArrayIndex, ArrayIndexOp};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
#[path = "parameters/completed.rs"]
mod completed;
#[path = "parameters/encoding.rs"]
mod encoding;
#[path = "parameters/numerical.rs"]
mod numerical;
#[path = "parameters/partition.rs"]
mod partition;
#[path = "parameters/publication.rs"]
mod publication;
#[path = "parameters/result.rs"]
mod result;
use crate::backend::nn::workspace::CompletedParameterSources;
use encoding::EffectiveLayout;
#[cfg(test)]
#[path = "parameters/partition_owner_tests.rs"]
mod partition_owner_tests;
#[cfg(test)]
#[path = "parameters/storage_tests.rs"]
mod storage_tests;

#[derive(Default)]
pub(super) struct NativeParameterState {
    pub(super) model_identity: Option<eredu_runtime::parameter_operations::ParameterModelIdentity>,
    pub(super) active: Option<String>,
    /// Conservative state precision after native promotion by edited parameters.
    pub(super) floating_state_dtype_bytes: Option<std::num::NonZeroU8>,
    originals: ParameterReplacementValues<MlxTensor>,
    published: ParameterReplacementValues<MlxTensor>,
    original_sources: CompletedParameterSources,
    published_sources: CompletedParameterSources,
    baseline_transforms: Vec<(String, ProjectionInputTransform)>,
    metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
    reset_estimate: Option<eredu_core::execution_control::SnapshotEstimate>,
    #[cfg(test)]
    reject_publication: bool,
    epoch: u64,
    usage: Rc<Cell<CaptureUsage>>,
}

impl NativeParameterState {
    /// Borrows only these actual immutable maps. This is not idle-session,
    /// native readiness, publication or submission authority.
    pub(super) fn parameter_sources(&self) -> DisplacedParameterSource<'_> {
        DisplacedParameterSource { state: self }
    }

    /// Retained numerical payload from reversible parameter publication.
    ///
    /// Originals may no longer be installed in the model; published values can
    /// alias its current slots. Merge this inventory with the model inventory
    /// before counting or registering storage so both cases retain exact backing
    /// identities. Inspection never materializes lazy or unrecognized arrays;
    /// those owners remain retained with an unknown bound.
    ///
    /// The remaining fields describe identity, input arithmetic, precision,
    /// snapshot estimates and usage. They retain no numerical storage or erased
    /// resource owner. In particular model_identity does not own a coordinator,
    /// and baseline transforms contain enum/scalar metadata rather than tensors.
    pub(super) fn retained_storage(&self) -> Result<RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub(super) fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        for values in [&self.originals, &self.published] {
            for value in values.values() {
                storage.include_array(value.as_array())?;
            }
        }
        Ok(())
    }
}

pub(super) struct DisplacedParameterSource<'source> {
    state: &'source NativeParameterState,
}
pub(super) struct CountedDisplacedParameterSource<'source> {
    source: DisplacedParameterSource<'source>,
    counts: ParameterOwnerCounts,
}
impl<'source> DisplacedParameterSource<'source> {
    pub(super) fn count(
        self,
        guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<CountedDisplacedParameterSource<'source>, ParameterOwnerSourceError> {
        let mut counts = ParameterOwnerCounts::default();
        counts.observe_replacements(
            ParameterOwnerRole::DisplacedOriginal,
            None,
            &self.state.originals,
            guard,
        )?;
        counts.observe_replacements(
            ParameterOwnerRole::PublishedOverlay,
            None,
            &self.state.published,
            guard,
        )?;
        Ok(CountedDisplacedParameterSource {
            source: self,
            counts,
        })
    }
    pub(super) fn state(&self) -> &'source NativeParameterState {
        self.state
    }
}
impl<'source> CountedDisplacedParameterSource<'source> {
    pub(super) const fn counts(&self) -> ParameterOwnerCounts {
        self.counts
    }
    pub(super) fn source(&self) -> &DisplacedParameterSource<'source> {
        &self.source
    }
}

/// An independently borrowed counter permits charging metadata, native loans
/// and delivery without retaining or mutably borrowing the model payload.
#[derive(Clone)]
struct NativeParameterBudget {
    total: Rc<Cell<CaptureUsage>>,
    limit: CaptureUsage,
    operation: Option<(Rc<Cell<CaptureUsage>>, CaptureUsage)>,
}
impl CaptureReservation for NativeParameterBudget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let next = self.total.get().checked_add(cost)?;
        if let Some(budget) = next.exceeded(self.limit) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        if let Some((used, limit)) = &self.operation {
            let next_operation = used.get().checked_add(cost)?;
            if let Some(budget) = next_operation.exceeded(*limit) {
                return Err(CaptureError::Limit {
                    budget,
                    cumulative: false,
                });
            }
            used.set(next_operation);
        }
        self.total.set(next);
        Ok(None)
    }
}

fn failure(error: Error) -> ParameterError {
    BackendFailure::new(BackendFailureKind::Other, error).into()
}
fn environment_failure(cause: crate::backend::OriginalCopyEnvironmentError) -> ParameterError {
    BackendFailure::new(BackendFailureKind::Other, cause).into()
}
fn dtype(value: Dtype) -> Option<InterventionDtype> {
    match value {
        Dtype::Float32 => Some(InterventionDtype::Float32),
        Dtype::Float16 => Some(InterventionDtype::Float16),
        Dtype::Bfloat16 => Some(InterventionDtype::Bfloat16),
        _ => None,
    }
}
fn native_indices(region: &ParameterRegion) -> Vec<ArrayIndexOp<'static>> {
    region
        .starts
        .iter()
        .zip(&region.shape)
        .map(|(start, count)| (*start as i32..(start + count) as i32).index_op())
        .collect()
}

struct Catalog {
    layouts: BTreeMap<String, EffectiveLayout>,
    companions: BTreeSet<String>,
    physical: BTreeMap<String, (Dtype, Vec<u64>)>,
    slots: Vec<LoadedParameter>,
}
impl ParameterSlotVisitor<MlxTensor> for Catalog {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &MlxTensor) {
        self.record(
            metadata.to_owned(),
            value.as_array().dtype(),
            value.shape().iter().map(|n| *n as u64).collect(),
        );
    }
}
impl Catalog {
    fn bind_bank_loan_costs(
        &mut self,
        prepared: &[eredu_runtime::parameter_operations::PreparedParameterSlot],
    ) -> Result<(), ParameterError> {
        for (id, layout) in &mut self.layouts {
            let mut ids = BTreeSet::from([id.clone()]);
            layout.extend_dependencies(&mut ids);
            for slot in prepared
                .iter()
                .filter(|slot| ids.contains(slot.parameter.id.as_str()))
            {
                layout.bank_loan = layout.bank_loan.checked_add(slot.bank_loan_usage()?)?;
            }
        }
        Ok(())
    }
    fn record(&mut self, metadata: ParameterMetadata, actual_dtype: Dtype, actual_shape: Vec<u64>) {
        let id = metadata.id.as_str().to_string();
        if let Some(role) = metadata.linear_companion {
            self.companions.insert(id.clone());
            if let Some(layout) = metadata
                .linear_companion_of
                .as_ref()
                .and_then(|owner| self.layouts.get_mut(owner.as_str()))
            {
                layout.bind_companion(role, id.clone());
            }
        }
        let layout = self.layouts.entry(id.clone()).or_insert_with(|| {
            EffectiveLayout::new(eredu_checkpoint::LinearFormat::Dense, actual_shape.clone())
        });
        layout.row_layout = metadata.linear_row_layout;
        self.slots.push(LoadedParameter {
            shared_id: metadata
                .alias_of
                .as_ref()
                .map_or_else(|| id.clone(), |alias| alias.as_str().to_string()),
            id: id.clone(),
            shape: layout.shape.clone(),
            dtype: dtype(actual_dtype),
            supported: false,
            access: None,
            condition: String::new(),
            input_transform: ProjectionInputTransform::Unspecified,
        });
        self.physical.insert(id, (actual_dtype, actual_shape));
    }
    fn finish(
        &mut self,
        originals: &ParameterReplacementValues<MlxTensor>,
    ) -> Result<(), ParameterError> {
        for slot in &mut self.slots {
            let layout = &self.layouts[&slot.id];
            let (physical_dtype, shape) = &self.physical[&slot.id];
            slot.supported =
                !self.companions.contains(&slot.id) && layout.supported(shape, *physical_dtype);
            if slot.supported {
                slot.input_transform = if originals.contains_key(&slot.id) {
                    ProjectionInputTransform::Identity
                } else {
                    layout.input_transform(*physical_dtype)
                };
                slot.dtype = Some(dtype(*physical_dtype).unwrap_or(InterventionDtype::Float32));
                slot.condition = if layout.format == eredu_checkpoint::LinearFormat::Dense {
                    "Loaded dense values after materialization; edits preserve the native floating dtype".into()
                } else {
                    format!(
                        "Effective {:?} values; queries decode with F32 arithmetic; editing retains the original packed storage and publishes a complete F32 copy of each affected parameter only, with no requantization",
                        layout.format
                    )
                };
            } else {
                slot.condition = "Effective access requires a floating parameter or a supported packed matrix with its declared companions; companion slots are not independently editable".into();
            }
        }
        // Module implementations may traverse hash-backed parameter maps.
        // Public discovery remains deterministic across idle inspections.
        self.slots
            .sort_unstable_by(|left, right| left.id.cmp(&right.id));
        let mut ids = BTreeSet::new();
        if self.slots.iter().any(|slot| !ids.insert(&slot.id)) {
            return Err(ParameterError::Invalid(
                "duplicate loaded parameter identity".into(),
            ));
        }
        Ok(())
    }
}

impl MlxModelSession {
    #[cfg(test)]
    pub(crate) fn reject_next_parameter_publication_for_test(&mut self) {
        self.payload
            .get_mut()
            .expect("idle test session")
            .parameter_state
            .reject_publication = true;
    }
    fn parameter_operation_binding(
        &self,
    ) -> Result<eredu_runtime::parameter_operations::ParameterOperationBinding, ParameterError>
    {
        let identity = self
            .payload
            .parameter_state
            .model_identity
            .as_ref()
            .ok_or_else(|| {
                ParameterError::Unsupported(
                    "loaded model has no bounded parameter control identity".into(),
                )
            })?;
        let prepared = self
            .capture_discovery
            .as_ref()
            .ok_or_else(|| ParameterError::Unsupported("no retained prepared identity".into()))?;
        identity.binding(
            &prepared.capture()?.artifact_identity,
            prepared.execution_identity(),
            self.payload.parameter_state.epoch,
        )
    }
    /// Pure comparison for host sequence-bank preflight; no error allocation.
    pub(super) fn parameter_epoch_matches(&self, expected: u64) -> bool {
        self.payload.parameter_state.epoch == expected
    }

    pub(super) fn validate_parameter_epoch(&self, saved: &mut Option<u64>) -> Result<(), Error> {
        let current = self.payload.parameter_state.epoch;
        if saved.is_some_and(|epoch| epoch != current) {
            return Err(eredu_core::BackendError::Execution {
                session: "text-generation".into(), operation: "validate parameter version".into(),
                message: "generation state belongs to an earlier parameter version; prepare a fresh exact-token prefill".into(),
            }.into());
        }
        *saved = Some(current);
        Ok(())
    }
    pub(super) fn ensure_parameter_cache_compatible(&self) -> Result<(), Error> {
        if self.payload.parameter_state.active.is_some() {
            return Err(eredu_core::cache::PromptCacheError::Incompatible(
                "persistent prompt caches do not yet encode active parameter overlays; use a fresh exact-token prefill".into()).into());
        }
        Ok(())
    }
    fn parameter_facts(&mut self) -> Result<ParameterDiscovery, ParameterError> {
        if self.payload.distributed.is_some() {
            return self
                .partition_parameter_catalog(None)
                .map(|catalog| catalog.discovery);
        }
        self.parameter_facts_and_layouts().map(|(facts, _)| facts)
    }
    fn parameter_facts_and_layouts(
        &mut self,
    ) -> Result<(ParameterDiscovery, BTreeMap<String, EffectiveLayout>), ParameterError> {
        self.ensure_no_submission_in_flight().map_err(failure)?;
        // Local prepared owners also serve partitioned executions. Public
        // operations require global assembly and peer transaction admission;
        // exposing a local catalog here would silently describe partial values.
        if self.payload.distributed.is_some()
            || self
                .payload
                .model
                .erased()
                .partition_parameter_description()
                .is_some()
        {
            return Err(ParameterError::Unsupported(
                "partitioned parameter operations require the retained collective communication owner".into(),
            ));
        }
        let artifact_identity = self
            .capture_discovery
            .as_ref()
            .ok_or_else(|| ParameterError::Unsupported("no retained prepared identity".into()))?
            .capture()?
            .artifact_identity;
        let payload = self
            .payload
            .get_mut()
            .ok_or_else(|| ParameterError::Unsupported("native payload still retained".into()))?;
        let layouts = payload
            .model
            .erased()
            .parameter_materialization_tasks()
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name().to_string())
                    .chain(task.aliases().iter().cloned())
                    .map(move |name| {
                        (
                            name,
                            EffectiveLayout::new(
                                task.executable(),
                                task.logical_shape().iter().map(|n| *n as u64).collect(),
                            ),
                        )
                    })
            })
            .collect();
        let mut catalog = Catalog {
            layouts,
            companions: BTreeSet::new(),
            physical: BTreeMap::new(),
            slots: vec![],
        };
        if !payload
            .model
            .erased_mut()
            .visit_loaded_parameters(&mut catalog)
        {
            let prepared = payload.model.erased().prepared_parameter_slots();
            if prepared.is_empty() {
                return Err(ParameterError::Unsupported(
                    "selected execution has no prepared parameter access mechanism".into(),
                ));
            }
            for slot in prepared {
                let physical_dtype = crate::backend::runtime::checkpoint::recipe::mlx_dtype(
                    &slot.materialized.dtype,
                )
                .map_err(|error| ParameterError::Unsupported(error.to_string()))?;
                catalog.record(
                    slot.parameter.clone(),
                    physical_dtype,
                    slot.materialized.shape.iter().map(|n| *n as u64).collect(),
                );
            }
        }
        catalog.bind_bank_loan_costs(payload.model.erased().prepared_parameter_slots())?;
        catalog.finish(&payload.parameter_state.originals)?;
        Ok((
            ParameterDiscovery {
                identity: format!(
                    "{}:parameters:{}",
                    self.intervention_session_identity, payload.parameter_state.epoch
                ),
                artifact_identity,
                overlay_identity: payload.parameter_state.active.clone(),
                parameters: catalog.slots,
                usage: payload.parameter_state.usage.get(),
                coordination_usage: Default::default(),
            },
            catalog.layouts,
        ))
    }
    fn reserve_parameters(
        &mut self,
        usage: CaptureUsage,
        limits: CaptureUsage,
    ) -> Result<(), ParameterError> {
        NativeParameterBudget {
            total: Rc::clone(&self.payload.parameter_state.usage),
            limit: limits,
            operation: None,
        }
        .reserve_quota(usage)?;
        Ok(())
    }
}

impl ParameterBackend for MlxBackend<'_> {
    fn parameter_discovery(
        runtime: &mut ModelRuntime<Self>,
    ) -> Result<ParameterDiscovery, ParameterError> {
        runtime.session_mut().parameter_facts()
    }

    fn query_parameter(
        runtime: &mut ModelRuntime<Self>,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
    ) -> Result<SharedParameterValues, ParameterError> {
        let (backend, session) = runtime.parts_mut();
        if session.payload.distributed.is_some() {
            let environment = backend
                .original_copy_environment()
                .map_err(environment_failure)?;
            return session.query_partition_parameter(
                identity,
                parameter,
                region,
                limits,
                &environment,
            );
        }
        let (discovery, layouts) = session.parameter_facts_and_layouts()?;
        if identity != discovery.identity {
            return Err(ParameterError::StaleIdentity);
        }
        let descriptor = discovery
            .parameters
            .iter()
            .find(|p| p.id == parameter)
            .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
        if !descriptor.supported {
            return Err(ParameterError::Unsupported(descriptor.condition.clone()));
        }
        let layout = &layouts[parameter];
        let count = region.validate(&descriptor.shape)?;
        let source = elements(&descriptor.shape)?;
        if source > i32::MAX as u64 {
            return Err(ParameterError::Unsupported(
                "native indexing exceeds i32".into(),
            ));
        }
        session.reserve_parameters(
            CaptureUsage {
                captures: 1,
                retained_bytes: add(
                    mul(
                        source,
                        if layout.format == eredu_checkpoint::LinearFormat::Dense {
                            8
                        } else {
                            32
                        },
                    )?,
                    add(
                        add(mul(count, 32)?, 4096)?,
                        layout.conversion_and_loan_bytes()?,
                    )?,
                )?,
                host_bytes: add(add(mul(count, 16)?, 256)?, layout.host_conversion_bytes()?)?,
                encoded_bytes: add(
                    mul(count, 32)?,
                    add(2048, mul((parameter.len() + identity.len()) as u64, 6)?)?,
                )?,
            },
            limits,
        )?;
        let result = result::PreparedResult::query(
            &session.payload.memory_ledger,
            identity,
            parameter,
            &region,
        )?;
        let completed =
            session.query_completed_parameter(parameter, &region, result.host_authority())?;
        let resident = if completed.is_some() {
            completed
        } else {
            match backend.original_copy_environment() {
                Ok(environment) => session
                    .read_resident_effective_parameter(
                        parameter,
                        layout,
                        numerical::Request::Read(&region),
                        &environment,
                    )
                    .map_err(failure)?,
                Err(cause) => {
                    return Err(completed::environment_failure(
                        cause,
                        result.host_authority(),
                    ))
                }
            }
        };
        let values = resident.ok_or_else(|| {
            failure(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ParameterError::Invalid(
                "non-finite effective parameter".into(),
            ));
        }
        result.finish_query(ParameterValues {
            identity: identity.into(),
            parameter: parameter.into(),
            dtype: descriptor.dtype.expect("supported floating dtype"),
            region,
            values,
            usage: session.payload.parameter_state.usage.get(),
        })
    }

    fn project_parameter(
        runtime: &mut ModelRuntime<Self>,
        identity: &str,
        parameter: &str,
        projection: ParameterProjection,
        limits: CaptureUsage,
    ) -> Result<SharedParameterProjectionValues, ParameterError> {
        let (backend, session) = runtime.parts_mut();
        if session.payload.distributed.is_some() {
            let environment = backend
                .original_copy_environment()
                .map_err(environment_failure)?;
            return session.project_partition_parameter(
                identity,
                parameter,
                projection,
                limits,
                &environment,
            );
        }
        let (discovery, layouts) = session.parameter_facts_and_layouts()?;
        if identity != discovery.identity {
            return Err(ParameterError::StaleIdentity);
        }
        let descriptor = discovery
            .parameters
            .iter()
            .find(|p| p.id == parameter)
            .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
        if !descriptor.supported {
            return Err(ParameterError::Unsupported(descriptor.condition.clone()));
        }
        let layout = &layouts[parameter];
        let shape = projection.output_shape(&descriptor.shape)?;
        let source = elements(&descriptor.shape)?;
        let selected = elements(&projection.region.shape)?;
        let output = elements(&shape)?;
        let directions = projection.coefficients.len() as u64;
        if source > i32::MAX as u64 || output > i32::MAX as u64 {
            return Err(ParameterError::Unsupported(
                "native projection exceeds i32 indexing".into(),
            ));
        }
        // Retained source, contiguous F32 selection, uploaded directions, multiplication
        // output and conservative conversion/workspace allowance. Host work scales with
        // directions plus the reduced output, never with the selected weight rectangle.
        session.reserve_parameters(
            CaptureUsage {
                captures: 1,
                retained_bytes: add(
                    add(4096, layout.conversion_and_loan_bytes()?)?,
                    add(
                        mul(
                            source,
                            if layout.format == eredu_checkpoint::LinearFormat::Dense {
                                8
                            } else {
                                32
                            },
                        )?,
                        add(
                            mul(selected, 16)?,
                            add(mul(directions, 8)?, mul(output, 16)?)?,
                        )?,
                    )?,
                )?,
                host_bytes: add(
                    add(256, add(mul(directions, 8)?, mul(output, 16)?)?)?,
                    layout.host_conversion_bytes()?,
                )?,
                encoded_bytes: add(
                    mul(output, 32)?,
                    add(2048, mul((parameter.len() + identity.len()) as u64, 6)?)?,
                )?,
            },
            limits,
        )?;
        let result = result::PreparedResult::projection(
            &session.payload.memory_ledger,
            identity,
            parameter,
            &shape,
            shape.capacity(),
        )?;
        let resident = match backend.original_copy_environment() {
            Ok(environment) => match session
                .project_resident_parameter(parameter, &projection, &environment)
                .map_err(failure)?
            {
                Some(values) => Some(values),
                None => session
                    .read_resident_effective_parameter(
                        parameter,
                        layout,
                        numerical::Request::Project(&projection),
                        &environment,
                    )
                    .map_err(failure)?,
            },
            // Ordinary sources keep their existing exclusion until their
            // selected materialization producer supplies complete native facts.
            Err(crate::backend::OriginalCopyEnvironmentError::Memory(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )) => None,
            Err(cause) => {
                return Err(completed::environment_failure(
                    cause,
                    result.host_authority(),
                ))
            }
        };
        let values = resident.ok_or_else(|| {
            failure(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ParameterError::Invalid(
                "non-finite effective parameter projection".into(),
            ));
        }
        result.finish_projection(ParameterProjectionValues {
            identity: identity.into(),
            parameter: parameter.into(),
            source_dtype: descriptor.dtype.expect("supported floating dtype"),
            shape,
            values,
            usage: session.payload.parameter_state.usage.get(),
        })
    }

    fn activate_parameter_overlay(
        runtime: &mut ModelRuntime<Self>,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let (backend, session) = runtime.parts_mut();
        let environment = backend
            .original_copy_environment()
            .map_err(environment_failure)?;
        if session.payload.distributed.is_some() {
            return session.activate_partition_parameter_overlay(overlay, limits, &environment);
        }
        let (mut discovery, layouts) = session.parameter_facts_and_layouts()?;
        overlay.validate(&discovery)?;
        if discovery.overlay_identity.is_some() {
            return Err(ParameterError::Unsupported(
                "remove the active overlay before installing another transaction".into(),
            ));
        }
        let epoch = session
            .payload
            .parameter_state
            .epoch
            .checked_add(1)
            .ok_or(ParameterError::Overflow)?;
        let shared: BTreeSet<_> = overlay.shared_targets().iter().collect();
        let targets: Vec<_> = discovery
            .parameters
            .iter()
            .filter(|p| shared.contains(&p.shared_id))
            .collect();
        let mut cost = CaptureUsage {
            captures: 1,
            retained_bytes: 4096,
            host_bytes: 4096,
            encoded_bytes: 4096,
        };
        for target in &targets {
            let n = elements(&target.shape)?;
            if !target.supported || n > i32::MAX as u64 {
                return Err(ParameterError::Unsupported(
                    "shared target cannot be edited by the selected native mechanism".into(),
                ));
            }
            for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
                if shared == &target.shared_id {
                    // Each immutable slice update may own another full destination;
                    // reserve it even when the allocator reuses a completed buffer.
                    cost.retained_bytes = add(cost.retained_bytes, mul(n, 8)?)?;
                }
                if shared == &target.shared_id
                    && (edit.parameter_shape != target.shape || Some(edit.dtype) != target.dtype)
                {
                    return Err(ParameterError::Invalid(
                        "shared parameter slots have incompatible geometry or dtype".into(),
                    ));
                }
            }
            // Original, complete affected-parameter copy, casting and index-update temporaries.
            cost.retained_bytes = add(cost.retained_bytes, mul(n, 48)?)?;
            cost.retained_bytes = add(
                cost.retained_bytes,
                layouts[&target.id].conversion_and_loan_bytes()?,
            )?;
            cost.host_bytes = add(
                cost.host_bytes,
                layouts[&target.id].host_conversion_bytes()?,
            )?;
        }
        for edit in &overlay.plan().edits {
            cost.host_bytes = add(cost.host_bytes, mul(edit.update.values().len() as u64, 8)?)?;
            cost.retained_bytes = add(
                cost.retained_bytes,
                mul(edit.update.values().len() as u64, 16)?,
            )?;
        }
        session.reserve_parameters(cost, limits)?;
        let prepared = numerical::PreparedOperation::new(session, &environment).map_err(failure)?;
        let context = &prepared.context;
        let mut originals = context
            .metadata_vec(targets.len())
            .map_err(|cause| failure(Error::Neural(cause)))?;
        let mut replacements = context
            .metadata_vec(targets.len())
            .map_err(|cause| failure(Error::Neural(cause)))?;
        for target in &targets {
            let mut edits = context
                .metadata_vec(overlay.plan().edits.len())
                .map_err(|cause| failure(Error::Neural(cause)))?;
            for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
                if shared == &target.shared_id {
                    edits.push((&edit.region, &edit.update));
                }
            }
            let (original, replacement) = session
                .prepare_parameter_update(&target.id, &layouts[&target.id], &edits, &prepared)
                .map_err(failure)?;
            context
                .charge_metadata(
                    target
                        .id
                        .len()
                        .checked_mul(2)
                        .ok_or(ParameterError::Overflow)?,
                )
                .map_err(|cause| failure(Error::Neural(cause.into())))?;
            originals.push((target.id.clone(), original));
            replacements.push((target.id.clone(), replacement));
        }
        let originals = parameter_rows(originals, &prepared)?;
        let replacements = parameter_rows(replacements, &prepared)?;
        let sources = CompletedParameterSources::from_prepared(
            prepared.take_sources(),
            context,
            prepared.funding.clone(),
        )
        .map_err(failure)?;
        let original_sources = sources
            .select(&originals, context, prepared.funding.clone())
            .map_err(failure)?;
        let published_sources = sources
            .select(&replacements, context, prepared.funding.clone())
            .map_err(failure)?;
        context
            .charge_metadata(overlay.identity().len())
            .map_err(|cause| failure(Error::Neural(cause.into())))?;
        let active = overlay.identity().to_owned();
        let (mut publication, mut reset) = session
            .prepare_parameter_publication(
                replacements.clone(),
                published_sources.clone(),
                true,
                &prepared,
            )
            .map_err(failure)?;
        session
            .commit_parameter_publication(&mut publication, &mut reset)
            .map_err(failure)?;
        let state = &mut session
            .payload
            .get_mut()
            .expect("completed parameter transaction")
            .parameter_state;
        state.originals = originals;
        state.published = replacements;
        state.original_sources = original_sources;
        state.published_sources = published_sources;
        state.metadata = Some(prepared.funding.clone());
        state.active = Some(active);
        // Dense replacements preserve their effective dtype; packed replacements
        // are F32. Their outputs can promote later KV/recurrent state. The backend
        // uses an upper bound without reconstructing family execution branches.
        state.floating_state_dtype_bytes = targets
            .iter()
            .filter_map(|target| match target.dtype {
                Some(InterventionDtype::Float32) => std::num::NonZeroU8::new(4),
                Some(InterventionDtype::Float16 | InterventionDtype::Bfloat16) => {
                    std::num::NonZeroU8::new(2)
                }
                None => None,
            })
            .max();
        state.epoch = epoch;
        discovery.identity = format!(
            "{}:parameters:{epoch}",
            session.intervention_session_identity
        );
        for parameter in &mut discovery.parameters {
            if shared.contains(&parameter.shared_id) {
                parameter.input_transform = ProjectionInputTransform::Identity;
            }
        }
        discovery.overlay_identity = state.active.clone();
        discovery.usage = state.usage.get();
        Ok(discovery)
    }

    fn remove_parameter_overlay(
        runtime: &mut ModelRuntime<Self>,
        identity: &str,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let (backend, session) = runtime.parts_mut();
        let environment = backend
            .original_copy_environment()
            .map_err(environment_failure)?;
        if session.payload.distributed.is_some() {
            return session.remove_partition_parameter_overlay(identity, &environment);
        }
        let (mut discovery, layouts) = session.parameter_facts_and_layouts()?;
        if discovery.identity != identity || discovery.overlay_identity.is_none() {
            return Err(ParameterError::StaleIdentity);
        }
        let epoch = session
            .payload
            .parameter_state
            .epoch
            .checked_add(1)
            .ok_or(ParameterError::Overflow)?;
        let prepared = numerical::PreparedOperation::new(session, &environment).map_err(failure)?;
        let originals = session.payload.parameter_state.originals.clone();
        let original_sources = session.payload.parameter_state.original_sources.clone();
        let (mut publication, mut reset) = session
            .prepare_parameter_publication(originals.clone(), original_sources, false, &prepared)
            .map_err(failure)?;
        session
            .commit_parameter_publication(&mut publication, &mut reset)
            .map_err(failure)?;
        let state = &mut session
            .payload
            .get_mut()
            .expect("completed parameter transaction")
            .parameter_state;
        for parameter in &mut discovery.parameters {
            if let Some(original) = originals.get(&parameter.id) {
                parameter.input_transform =
                    layouts[&parameter.id].input_transform(original.as_array().dtype());
            }
        }
        state.originals = Default::default();
        state.published = Default::default();
        state.original_sources = Default::default();
        state.published_sources = Default::default();
        state.active = None;
        state.floating_state_dtype_bytes = None;
        state.epoch = epoch;
        discovery.identity = format!(
            "{}:parameters:{epoch}",
            session.intervention_session_identity
        );
        discovery.overlay_identity = None;
        discovery.usage = state.usage.get();
        Ok(discovery)
    }
}

#[cfg(test)]
#[path = "parameters/owner_source_tests.rs"]
mod owner_source_tests;

struct PreparedParameterReset {
    native: Box<dyn std::any::Any>,
    covered_publication: bool,
    sources: CompletedParameterSources,
}
fn parameter_rows(
    rows: Vec<(String, MlxTensor)>,
    prepared: &numerical::PreparedOperation<'_>,
) -> Result<ParameterReplacementValues<MlxTensor>, ParameterError> {
    ParameterReplacementValues::from_prepared_rows(
        rows,
        prepared.funding.clone(),
        &prepared.context,
    )
    .map_err(|cause| failure(Error::Neural(prepared.context.metadata_source(cause))))
}
impl MlxModelSession {
    /// The session inventory retains both installed and displaced parameter roots.
    pub(super) fn native_storage_mechanism(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>,
        Error,
    > {
        Ok(self
            .payload
            .model
            .native_storage_mechanism()?
            .map(|mechanism| {
                mechanism.with_displaced_parameter_sources(
                    self.payload.parameter_state.original_sources.clone(),
                )
            }))
    }

    fn prepare_parameter_publication(
        &mut self,
        values: ParameterReplacementValues<MlxTensor>,
        sources: CompletedParameterSources,
        active: bool,
        prepared: &numerical::PreparedOperation<'_>,
    ) -> Result<
        (
            publication::PreparedNativeParameterPublication,
            PreparedParameterReset,
        ),
        Error,
    > {
        self.original_model_source()
            .map_err(Error::PrefillControl)?;
        let payload = self.payload.get_mut().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::ReservedWorkActive,
        ))?;
        prepared.context.charge_metadata(std::mem::size_of::<PreparedParameterReset>().checked_add(
            crate::backend::runtime::residency::storage::RetainedStoragePublication::coverage_control_bytes()
                .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow))?)
            .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow))?)
            .map_err(|cause|Error::Neural(cause.into()))?;
        let covered_publication = payload
            .nonstate_publication
            .get_mut()
            .as_ref()
            .is_some_and(|value| payload.model.covers_nonstate_publication(value));
        let preparation = prepared.preparation(&[]);
        let publication = publication::PreparedNativeParameterPublication::prepare(
            payload.model.erased_mut(),
            values,
            active,
            &preparation,
        )?;
        let reset = payload
            .model
            .erased_mut()
            .prepare_parameter_reset_state(&preparation)?;
        Ok((
            publication,
            PreparedParameterReset {
                native: reset,
                covered_publication,
                sources,
            },
        ))
    }
    fn commit_parameter_publication(
        &mut self,
        publication: &mut publication::PreparedNativeParameterPublication,
        reset: &mut PreparedParameterReset,
    ) -> Result<(), Error> {
        self.original_model_source()
            .map_err(Error::PrefillControl)?;
        let payload = self.payload.get_mut().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::ReservedWorkActive,
        ))?;
        publication.exchange_reset(payload.model.erased_mut(), reset.native.as_mut())?;
        crate::composition::mlx::replicated_text::exchange_parameter_reset_memory(
            reset.native.as_mut(),
            &mut payload.state_memory,
            payload.nonstate_publication.get_mut(),
            reset.covered_publication,
        )
        .expect("reset type validated by the same prepared publication");
        payload.model.exchange_parameter_sources(&mut reset.sources);
        payload.model.erased_mut().finalize_parameter_publication();
        Ok(())
    }
}

#[cfg(test)]
fn fixture_rows<const N: usize>(
    values: [(&str, MlxTensor); N],
) -> ParameterReplacementValues<MlxTensor> {
    let ledger = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let execution = eredu_runtime::working_memory::InferenceExecutionIdentity::default();
    let funding = ledger
        .prepare_workspace_metadata(&execution, ledger.configured_limits().clone())
        .unwrap();
    let context = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        funding.clone(),
    )
    .unwrap();
    let mut rows = context.metadata_vec(N).unwrap();
    for (name, value) in values {
        context.charge_metadata(name.len()).unwrap();
        rows.push((name.to_owned(), value));
    }
    ParameterReplacementValues::from_prepared_rows(rows, funding, &context).unwrap()
}
