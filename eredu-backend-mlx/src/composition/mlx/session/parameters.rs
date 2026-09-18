//! Native loaded-slot operations. Architecture names and geometry come from traversal.
use super::*;
use crate::backend::nn::shared::visit_parameter_map;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::composition::mlx::replicated_text::{
    ParameterOwnerCounts, ParameterOwnerRole, ParameterOwnerSourceError,
};
use eredu_core::{capture::*, intervention::InterventionDtype, parameters::*};
use eredu_nn::{ParameterMetadata, ParameterSlotVisitor};
use safemlx::ops::indexing::{ArrayIndex, ArrayIndexOp, TryIndexMutOp};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
#[path = "parameters/encoding.rs"]
mod encoding;
#[path = "parameters/partition.rs"]
mod partition;
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
    originals: BTreeMap<String, MlxTensor>,
    published: BTreeMap<String, MlxTensor>,
    baseline_transforms: BTreeMap<String, ProjectionInputTransform>,
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
            visit_parameter_map(values, |_, value| storage.include_array(value.as_array()))?;
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
        counts.observe_map(
            ParameterOwnerRole::DisplacedOriginal,
            None,
            &self.state.originals,
            guard,
        )?;
        counts.observe_map(
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

fn apply_effective_update(
    result: &mut Array,
    region: &ParameterRegion,
    update: &ParameterUpdate,
    stream: &Stream,
) -> Result<Result<(), ParameterError>, Error> {
    let shape: Vec<_> = region.shape.iter().map(|n| *n as i32).collect();
    let indices = native_indices(region);
    let values = Array::from_slice(update.values(), &shape);
    let effective = match update {
        ParameterUpdate::Replace { .. } => values,
        ParameterUpdate::Add { .. } => result
            .try_index_device(indices.as_slice(), stream)?
            .as_dtype(Dtype::Float32, stream)?
            .add(values, stream)?,
    }
    .as_dtype(result.dtype(), stream)?;
    if !effective
        .is_finite(stream)?
        .all(false, stream)?
        .evaluated()?
        .as_slice::<bool>()[0]
    {
        return Ok(Err(ParameterError::Invalid(
            "parameter edit overflows target dtype".into(),
        )));
    }
    result.try_index_mut_device(indices.as_slice(), &effective, stream)?;
    result.evaluated()?;
    Ok(Ok(()))
}
struct Catalog {
    layouts: BTreeMap<String, EffectiveLayout>,
    companions: BTreeSet<String>,
    physical: BTreeMap<String, (Dtype, Vec<u64>)>,
    slots: Vec<LoadedParameter>,
}
impl ParameterSlotVisitor<MlxTensor> for Catalog {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &mut MlxTensor) {
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
    fn finish(&mut self, originals: &BTreeMap<String, MlxTensor>) -> Result<(), ParameterError> {
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
struct Select<'a> {
    ids: &'a BTreeSet<String>,
    values: BTreeMap<String, MlxTensor>,
}
impl ParameterSlotVisitor<MlxTensor> for Select<'_> {
    fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &mut MlxTensor) {
        if self.ids.contains(metadata.id().as_str()) {
            self.values
                .insert(metadata.id().as_str().into(), value.clone());
        }
    }
}

fn with_selected_parameter_values<T>(
    model: &mut dyn crate::composition::mlx::replicated_text::ErasedReplicatedTextExecutable,
    ids: &BTreeSet<String>,
    stream: &Stream,
    operation: impl FnOnce(&BTreeMap<String, MlxTensor>) -> Result<T, Error>,
) -> Result<T, Error> {
    use eredu_runtime::parameter_operations::PreparedParameterLocation;
    let prepared = model.prepared_parameter_slots();
    let mut owner = None;
    for id in ids {
        let slot = prepared
            .iter()
            .find(|slot| slot.parameter.id.as_str() == id)
            .ok_or_else(|| {
                Error::ArchitectureModel(format!("selected parameter {id} has no prepared owner"))
            })?;
        if let Some(previous) = &owner {
            let compatible = match (previous, &slot.location) {
                (
                    PreparedParameterLocation::Static { .. },
                    PreparedParameterLocation::Static { .. },
                ) => true,
                (left, right) => left == right,
            };
            if !compatible {
                return Err(Error::ArchitectureModel(
                    "one parameter operation spans unrelated residency units".into(),
                ));
            }
        }
        owner = Some(slot.location.clone());
    }
    let owner =
        owner.ok_or_else(|| Error::ArchitectureModel("empty parameter operation".into()))?;
    let mut operation = Some(operation);
    let mut output = None;
    let available = model.with_parameter_slots(
        &owner,
        ids,
        &mut |visit| {
            let mut selected = Select {
                ids,
                values: BTreeMap::new(),
            };
            visit(&mut selected);
            if selected.values.len() != ids.len() {
                return Err(Error::ArchitectureModel(
                    "prepared parameter topology changed".into(),
                ));
            }
            output = Some(operation.take().expect("one parameter loan")(
                &selected.values,
            )?);
            Ok(())
        },
        stream,
    )?;
    if !available {
        return Err(Error::ArchitectureModel(
            "selected parameter loan is unavailable".into(),
        ));
    }
    output.ok_or_else(|| Error::ArchitectureModel("parameter loan did not run".into()))
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
    ) -> Result<ParameterValues, ParameterError> {
        let stream = runtime.backend().stream().clone();
        let session = runtime.session_mut();
        if session.payload.distributed.is_some() {
            return session.query_partition_parameter(identity, parameter, region, limits, &stream);
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
        let mut ids = BTreeSet::from([parameter.to_string()]);
        layout.extend_dependencies(&mut ids);
        let values = session
            .with_model_operation(|model| {
                with_selected_parameter_values(model.erased_mut(), &ids, &stream, |selected| {
                    let tensor = layout.effective(parameter, selected, &stream)?;
                    encoding::read_effective(&tensor, &region, &stream)
                })
            })
            .map_err(failure)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ParameterError::Invalid(
                "non-finite effective parameter".into(),
            ));
        }
        Ok(ParameterValues {
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
    ) -> Result<ParameterProjectionValues, ParameterError> {
        let stream = runtime.backend().stream().clone();
        let session = runtime.session_mut();
        if session.payload.distributed.is_some() {
            return session
                .project_partition_parameter(identity, parameter, projection, limits, &stream);
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
        let mut ids = BTreeSet::from([parameter.to_string()]);
        layout.extend_dependencies(&mut ids);
        let values = session
            .with_model_operation(|model| {
                with_selected_parameter_values(model.erased_mut(), &ids, &stream, |selected| {
                    let tensor = layout.effective(parameter, selected, &stream)?;
                    encoding::project_effective(&tensor, &projection, &stream)
                })
            })
            .map_err(failure)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ParameterError::Invalid(
                "non-finite effective parameter projection".into(),
            ));
        }
        Ok(ParameterProjectionValues {
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
        let stream = runtime.backend().stream().clone();
        let session = runtime.session_mut();
        if session.payload.distributed.is_some() {
            return session.activate_partition_parameter_overlay(overlay, limits, &stream);
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
        let originals = session
            .with_model_operation(|model| {
                let mut originals = BTreeMap::new();
                let mut replacements = BTreeMap::new();
                for target in &targets {
                    let mut ids = BTreeSet::from([target.id.clone()]);
                    layouts[&target.id].extend_dependencies(&mut ids);
                    let candidate = with_selected_parameter_values(
                        model.erased_mut(),
                        &ids,
                        &stream,
                        |selected| {
                            let original = &selected[&target.id];
                            original.as_array().evaluated()?;
                            let mut result =
                                layouts[&target.id].effective(&target.id, selected, &stream)?;
                            for (edit, shared) in
                                overlay.plan().edits.iter().zip(overlay.shared_targets())
                            {
                                if shared != &target.shared_id {
                                    continue;
                                }
                                if let Err(error) = apply_effective_update(
                                    &mut result,
                                    &edit.region,
                                    &edit.update,
                                    &stream,
                                )? {
                                    return Ok(Err(error));
                                }
                            }
                            Ok(Ok((original.clone(), MlxTensor::from_array(result))))
                        },
                    )?;
                    let (original, replacement) = match candidate {
                        Ok(value) => value,
                        Err(error) => return Ok(Err(error)),
                    };
                    originals.insert(target.id.clone(), original);
                    replacements.insert(target.id.clone(), replacement);
                }
                // Every fallible native replacement is ready before clearing incompatible state.
                model.erased_mut().reset_cache()?;
                // Slot publication performs only handle moves/clones; no native work can fail.
                if !model
                    .erased_mut()
                    .publish_parameter_replacements(&replacements, true)?
                {
                    return Err(Error::ArchitectureModel(
                        "selected parameter publication is unavailable".into(),
                    ));
                }
                model.erased_mut().invalidate_parameter_snapshots();
                Ok(Ok(originals))
            })
            .map_err(failure)??;
        let state = &mut session
            .payload
            .get_mut()
            .expect("completed parameter transaction")
            .parameter_state;
        state.originals = originals;
        state.active = Some(overlay.identity().into());
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
        let session = runtime.session_mut();
        if session.payload.distributed.is_some() {
            return session.remove_partition_parameter_overlay(identity);
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
        // Cloned handles retain existing reservation; removal performs no parameter copies.
        let originals = session.payload.parameter_state.originals.clone();
        session
            .with_model_operation(|model| {
                model.erased_mut().reset_cache()?;
                if !model
                    .erased_mut()
                    .publish_parameter_replacements(&originals, false)?
                {
                    return Err(Error::ArchitectureModel(
                        "selected parameter restoration is unavailable".into(),
                    ));
                }
                model.erased_mut().invalidate_parameter_snapshots();
                Ok(())
            })
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
        state.originals.clear();
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
