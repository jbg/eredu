//! The real prepared-leaf equation under a borrowed, authenticated Original bank.
use super::*;
use crate::backend::{
    Error,
    runtime::{
        checkpoint::{
            recipe::{WeightRecipeError, prepare_original_recipe_from_leaves},
            store::{MaterializationView, OriginalMaterializationSlots},
        },
        residency::{manager::OriginalMaterializedLoan, storage::filled_host},
    },
    submission_recovery::native_role::{self, NativeRoleCapacity},
};
use eredu_nn::workspace::WorkspaceContext;
use safemlx::{
    Array, OperationEvent, OriginalBufferBudget, OriginalScopeObserver, PreparedHostCopyDestination,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

/// Independent child scopes use their own paid Graph/Record arenas. Their
/// numerical allocations draw from the enclosing accepted native buffer bank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Population {
    pub(crate) recipes: usize,
    pub(crate) bytes: u64,
    pub(crate) births: usize,
    pub(crate) controls: usize,
    pub(crate) validations: usize,
}
impl Population {
    pub(crate) fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            recipes: self.recipes.checked_add(rhs.recipes)?,
            bytes: self.bytes.checked_add(rhs.bytes)?,
            births: self.births.checked_add(rhs.births)?,
            controls: self.controls.checked_add(rhs.controls)?,
            validations: self.validations.checked_add(rhs.validations)?,
        })
    }
    pub(crate) fn checked_mul(self, n: usize) -> Option<Self> {
        Some(Self {
            recipes: self.recipes.checked_mul(n)?,
            bytes: self.bytes.checked_mul(u64::try_from(n).ok()?)?,
            births: self.births.checked_mul(n)?,
            controls: self.controls.checked_mul(n)?,
            validations: self.validations.checked_mul(n)?,
        })
    }
    pub(crate) fn maximum(self, rhs: Self) -> Self {
        Self {
            recipes: self.recipes.max(rhs.recipes),
            bytes: self.bytes.max(rhs.bytes),
            births: self.births.max(rhs.births),
            controls: self.controls.max(rhs.controls),
            validations: self.validations.max(rhs.validations),
        }
    }
    pub(crate) fn minimum(self, rhs: Self) -> Self {
        Self {
            recipes: self.recipes.min(rhs.recipes),
            bytes: self.bytes.min(rhs.bytes),
            births: self.births.min(rhs.births),
            controls: self.controls.min(rhs.controls),
            validations: self.validations.min(rhs.validations),
        }
    }
}
pub(super) fn population(
    plan: &read_source_plan::MaterializedReadPlan,
    runtime: &safemlx::PreparedInputRuntime,
) -> Result<Population, WorkingMemoryError> {
    let (_, own, role) = controls(plan, runtime)?;
    Ok(Population {
        recipes: 1,
        bytes: plan.original_numerical.storage.mutable_bytes(),
        births: plan.original_numerical.storage.maximum_births(),
        controls: own.checked_add(role).ok_or_else(overflow)?,
        validations: plan.validations,
    })
}

pub(super) struct PreparedRecipeSource {
    pub(super) ordinal: usize,
    pub(super) plan: Arc<read_source_plan::MaterializedReadPlan>,
    pub(super) leaves: Vec<publication::PublishedHostSource>,
    pub(super) output: Option<publication::Pending>,
}
struct Resources {
    source: PreparedRecipeSource,
    arrays: Vec<Array>,
    events: Vec<OperationEvent>,
    destination: Option<PreparedHostCopyDestination>,
}
type Invocation = Rc<RefCell<Resources>>;

fn missing() -> WorkingMemoryError {
    WorkingMemoryError::UnknownBound
}
fn overflow() -> WorkingMemoryError {
    WorkingMemoryError::Overflow
}
fn capacity(
    plan: &read_source_plan::MaterializedReadPlan,
    runtime: &safemlx::PreparedInputRuntime,
) -> Result<NativeRoleCapacity, WorkingMemoryError> {
    let recipe = plan.original_numerical;
    let layout = OriginalBufferBudget::population_layout(
        runtime,
        usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| overflow())?,
        recipe.storage.maximum_births(),
    )
    .map_err(|_| missing())?;
    Ok(NativeRoleCapacity {
        graph: recipe.graph_capacity,
        records: recipe.record_capacity,
        backing: layout.capacity(),
    })
}

/// The worker's exact metadata and child arenas; immutable Host sources have
/// their independent source-bank quotation and are not counted a second time.
pub(super) fn controls(
    plan: &read_source_plan::MaterializedReadPlan,
    runtime: &safemlx::PreparedInputRuntime,
) -> Result<(NativeRoleCapacity, usize, usize), WorkingMemoryError> {
    let capacity = capacity(plan, runtime)?;
    let own = [
        WorkspaceContext::metadata_rc_bytes::<RefCell<Resources>>().ok_or_else(overflow)?,
        Layout::array::<Array>(plan.leaves.len().checked_add(1).ok_or_else(overflow)?)
            .map_err(|_| overflow())?
            .size(),
        Layout::array::<OperationEvent>(plan.leaves.len().checked_add(1).ok_or_else(overflow)?)
            .map_err(|_| overflow())?
            .size(),
        filled_host::native_copy_control_bytes().ok_or_else(overflow)?,
        OperationEvent::nested_submission_control_bytes().ok_or_else(missing)?,
        usize::try_from(plan.original_numerical.controls).map_err(|_| overflow())?,
        crate::backend::runtime::checkpoint::recipe::prepared_recipe_entry_control_bytes()
            .ok_or_else(overflow)?,
        size_of::<Resources>(),
        size_of::<Invocation>(),
        size_of::<PreparedRecipeSource>(),
        size_of::<safemlx::PreparedResidentGraph>(),
        size_of::<Result<publication::Pending, eredu_core::BackendFailure>>(),
        size_of::<Option<publication::Pending>>(),
        size_of::<OriginalMaterializedLoan<'_>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), WeightRecipeError>>(),
        size_of::<(
            &PreparedRecipeSource,
            &MaterializationView<'_>,
            &mut OriginalMaterializationSlots<'_>,
            &OriginalScopeObserver,
        )>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(overflow)?;
    let role = native_role::control_bytes::<Invocation, OriginalOperationMetadataCustody>(
        capacity,
        Some(safemlx::PreparedPipelineCachePlan::new(
            plan.original_numerical.kernels,
        )),
    )
    .map_err(|_| missing())?;
    // Actual callback captures the borrowed materialization context and slots;
    // its result is returned only after the same native child has settled.
    let callback = native_role::callback_control_bytes::<(), Error>(size_of::<(
        MaterializationView<'_>,
        &mut OriginalMaterializationSlots<'_>,
    )>())
    .ok_or_else(overflow)?;
    Ok((
        capacity,
        own,
        role.checked_add(callback).ok_or_else(overflow)?,
    ))
}

pub(super) fn execute(
    source: PreparedRecipeSource,
    context: MaterializationView<'_>,
    slots: &mut OriginalMaterializationSlots<'_>,
    loan: OriginalMaterializedLoan<'_>,
    parent: &OriginalScopeObserver,
    runtime: &safemlx::PreparedInputRuntime,
    custody: &OriginalOperationMetadataCustody,
) -> Result<publication::Pending, eredu_core::BackendFailure> {
    let bad = || Error::PrefillControl(WorkingMemoryError::IdentityMismatch).into_backend_failure();
    if source.leaves.len() != source.plan.leaves.len() || source.output.is_none() {
        return Err(bad());
    }
    let (capacity, own, _) = controls(&source.plan, runtime)
        .map_err(|e| Error::PrefillControl(e).into_backend_failure())?;
    loan.funding
        .reserve_metadata(own)
        .map_err(|e| Error::WorkspacePlanning(e).into_backend_failure())?;
    let mut arrays = Vec::new();
    let mut events = Vec::new();
    let count = source.leaves.len().checked_add(1).ok_or_else(bad)?;
    arrays
        .try_reserve_exact(count)
        .map_err(|e| eredu_core::BackendFailure::from_error(e))?;
    events
        .try_reserve_exact(count)
        .map_err(|e| eredu_core::BackendFailure::from_error(e))?;
    let recipe = source.plan.original_numerical;
    let invocation = Rc::new(RefCell::new(Resources {
        source,
        arrays,
        events,
        destination: None,
    }));
    native_role::run_with_prepared_budget(
        invocation.clone(),
        capacity,
        Some(safemlx::PreparedPipelineCachePlan::new(recipe.kernels)),
        loan.budget,
        parent,
        custody,
        loan.funding,
        None,
        |invocation, role| {
            let mut graph =
                OperationEvent::prepare_resident_graph(recipe.completion.graph, role.observer())?;
            if recipe.completion.nested_completions != 0 {
                graph.configure_nested_completions(
                    &recipe
                        .completion
                        .nested_traversal()
                        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?,
                    recipe.completion.nested_completions,
                )?;
            }
            let mut resources = invocation.borrow_mut();
            let Resources {
                source,
                arrays,
                events,
                destination,
            } = &mut *resources;
            let mut ordinal = 0usize;
            let prepared = {
                let mut produce = |key: &str,
                                   selection: &eredu_checkpoint::store::TensorSelection,
                                   stream: &Stream| {
                    let expected = source
                        .plan
                        .leaves
                        .get(ordinal)
                        .ok_or(WeightRecipeError::ShapeMismatch)?;
                    if expected.key != key || &expected.selection != selection {
                        return Err(WeightRecipeError::ShapeMismatch);
                    }
                    let buffer = source
                        .leaves
                        .get(ordinal)
                        .ok_or(WeightRecipeError::ShapeMismatch)?;
                    let (array, event) = buffer
                        .buffer()
                        .copy_to_array_in_original_scope(stream, role.observer())?;
                    arrays.push(array.try_clone_handle()?);
                    events.push(event.into());
                    ordinal += 1;
                    Ok(array)
                };
                prepare_original_recipe_from_leaves(
                    &source.plan.recipe,
                    context,
                    &mut produce,
                    slots,
                    role.observer(),
                )
            };
            let (value, pending) = prepared?.into_parts();
            if ordinal != source.plan.leaves.len() || !pending.is_empty() {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            arrays.push(value);
            let value = arrays.last().expect("retained output");
            let event = OperationEvent::submit_nested(value, context.source_stream())?;
            events.push(event);
            events.last().expect("retained completion").synchronize()?;
            source
                .output
                .as_mut()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
                .completed_mut()
                .copy_from_array(value, context.source_stream(), role.observer(), destination)
                .map_err(|cause| {
                    Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
                })?;
            drop(graph);
            Ok(Ok::<(), Error>(()))
        },
    )?
    .map_err(Error::into_backend_failure)?;
    let result = invocation.borrow_mut().source.output.take().ok_or_else(bad);
    result
}
