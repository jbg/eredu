//! Same resident executor/slot with the exact AR or Embedded role account.
use super::*;
use crate::backend::nn::shared::SubmissionPreparationError;
use crate::backend::nn::workspace::{
    AutoregressiveEquationRecipe, EmbeddedEquationRecipe, ResidentNativeRecipe,
};
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalOperationMetadataCustody, OriginalSpeculativeRole,
};
use eredu_runtime::{
    speculative::external_occurrence::ExternalInvocation,
    working_memory::OriginalExternalSpeculativeRole,
};
use safemlx::{OriginalScopeObserver, SubmissionScope};

type Prepared = PreparedNeuralSubmission;
type PreparationError =
    SubmissionPreparationError<OriginalOperationMetadataCustody, OriginalScopeObserver>;

#[derive(Clone)]
pub(super) struct Projection {
    value: Weak<Bank>,
    // A stale weak projection cannot retire its header after the account.
    role: SpeculativeOperationRole,
}
impl Projection {
    pub(super) fn is_live(&self) -> bool {
        self.value.strong_count() != 0
    }
    fn access(&self, stream: &Stream) -> Result<Rc<Bank>, Error> {
        let bank = self.value.upgrade().ok_or_else(identity)?;
        if !bank.active.get()
            || !bank.role.same_role(&self.role)
            || !bank.stream.matches_source(stream)
            || !bank
                .observer
                .same_scope(&OriginalScopeObserver::require_current()?)
        {
            return Err(identity());
        }
        Ok(bank)
    }
    pub(super) fn uses_shared_executor(&self, stream: &Stream) -> Result<bool, Error> {
        self.access(stream)
            .map(|_| true)
            .map_err(|cause| neural::boundary_error(cause, &self.role))
    }
    pub(super) fn submit(
        &self,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<OrderedNeuralCompletion, Error> {
        let result = (|| {
            let bank = self.access(stream)?;
            let prepared = bank
                .prepared
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .checkout()
                .map_err(|_| Error::PrefillScopeUnavailable)?;
            prepared
                .submit_nested(value, bank.observer.clone(), stream)
                .map(|completion| OrderedNeuralCompletion::Speculative {
                    completion,
                    controls: self.role.clone(),
                })
                .map_err(|failure| Error::from(failure.into_cause()))
        })();
        result.map_err(|cause| neural::boundary_error(cause, &self.role))
    }
}
struct Bank {
    prepared: RefCell<PreparedOperationBank<Prepared>>,
    observer: OriginalScopeObserver,
    stream: safemlx::StreamCopyPlan<()>,
    active: Cell<bool>,
    role: SpeculativeOperationRole,
}
struct Owner<U: 'static> {
    bank: Rc<Bank>,
    slot: ResidentNeuralSlot<U>,
}
impl<U: 'static> ErasedOwner for Owner<U> {
    fn selected_residency_access(
        &self,
        role: &SpeculativeOperationRole,
    ) -> Result<OriginalSelectedResidencyAccess, Error> {
        if !self.bank.role.same_role(role) {
            return Err(identity());
        }
        OriginalSelectedResidencyAccess::resident(SelectedResidencyProjection {
            value: Rc::downgrade(&self.bank),
            role: role.clone(),
        })
    }
}
impl<U: 'static> Drop for Owner<U> {
    fn drop(&mut self) {
        let previous = self.slot.0.resident.take();
        if previous.as_ref().is_some_and(|view| {
            matches!(view, super::Projection::Speculative(view)
                if Weak::ptr_eq(&view.value, &Rc::downgrade(&self.bank)))
        }) {
            self.slot.0.resident.set(None);
            drop(previous);
        } else {
            self.slot.0.resident.set(previous);
        }
        self.bank.active.set(false);
    }
}
/// The installed bank retires on return or unwind before the enclosing native
/// role seals its Scope. Submitted completions retain their own role aliases.
pub(crate) struct SpeculativeNeuralOwner {
    pub(in crate::backend::runtime::execution::generic::original_operations) value:
        Box<dyn ErasedOwner>,
    pub(in crate::backend::runtime::execution::generic::original_operations) role:
        SpeculativeOperationRole,
}
impl SpeculativeNeuralOwner {
    /// Explicit loan from this actual installed operation bank. The private
    /// implementation authenticates the role and current observer; no source
    /// is selected by the current thread or by matching tensor geometry.
    pub(crate) fn selected_residency_access(
        &self,
    ) -> Result<OriginalSelectedResidencyAccess, Error> {
        self.value.selected_residency_access(&self.role)
    }
    pub(crate) fn during<T>(self, run: impl FnOnce() -> T) -> T {
        let result = run();
        drop(self);
        result
    }
}

impl<U: 'static> ResidentNeuralPlan<U> {
    pub(crate) fn bind_speculative_recipe(
        &self,
        recipe: &mut AutoregressiveEquationRecipe,
    ) -> Result<(), Error> {
        recipe.with_native_recipe(|recipe| self.bind_role_recipe(recipe))
    }
    pub(crate) fn bind_embedded_recipe(
        &self,
        recipe: &mut EmbeddedEquationRecipe,
    ) -> Result<(), Error> {
        if self.geometry != recipe.workspace().geometry() {
            return Err(identity());
        }
        recipe.with_native_recipe(|recipe| self.bind_role_recipe(recipe))
    }
    pub(crate) fn bind_external_recipe(
        &self,
        recipe: &mut ResidentNativeRecipe,
    ) -> Result<(), Error> {
        if self.geometry != recipe.plan().geometry() || recipe.records().len() != 1 {
            return Err(identity());
        }
        self.bind_role_recipe(recipe)
    }
    pub(crate) fn external_control_bytes(&self, recipe: &ResidentNativeRecipe) -> Option<u64> {
        (recipe.records().len() == 1)
            .then(|| self.role_control_bytes(recipe))
            .flatten()
    }
    fn bind_role_recipe(&self, recipe: &mut ResidentNativeRecipe) -> Result<(), Error> {
        recipe.bind_ordinary_neural_calls(
            self.geometry,
            self.neural.per_forward,
            self.neural.shape.consumers(),
        )?;
        if self.neural.submissions == 0 {
            return Ok(());
        }
        recipe.bind_neural_boundaries(
            self.geometry,
            self.neural.per_forward,
            self.neural.shape.consumers(),
        )
    }
    pub(crate) fn speculative_control_bytes(
        &self,
        recipe: &AutoregressiveEquationRecipe,
    ) -> Option<u64> {
        self.role_control_bytes(recipe.native_recipe())
    }
    pub(crate) fn embedded_control_bytes(&self, recipe: &EmbeddedEquationRecipe) -> Option<u64> {
        if self.geometry != recipe.workspace().geometry() {
            return None;
        }
        self.role_control_bytes(recipe.native_recipe())
    }
    fn role_control_bytes(&self, recipe: &ResidentNativeRecipe) -> Option<u64> {
        if self.geometry != recipe.plan().geometry() {
            return None;
        }
        let addressable = recipe
            .records()
            .iter()
            .any(|row| row.addressable().is_some());
        if self.neural.submissions == 0 && !addressable {
            return Some(0);
        }
        if self.neural.submissions != 0
            && !recipe.matches_neural_boundaries(
                self.geometry,
                self.neural.submissions,
                self.neural.shape.consumers(),
            )
        {
            return None;
        }
        self.speculative_bank_control_bytes(self.neural, addressable)
    }
    pub(crate) fn speculative_span_control_bytes(
        &self,
        recipe: &AutoregressiveEquationRecipe,
    ) -> Option<u64> {
        self.speculative_control_bytes(recipe)?;
        self.speculative_bank_control_bytes(
            NeuralPopulation {
                submissions: self.neural.per_forward,
                ..self.neural
            },
            recipe
                .native_recipe()
                .records()
                .iter()
                .any(|row| row.addressable().is_some()),
        )
    }
    fn speculative_bank_control_bytes(
        &self,
        population: NeuralPopulation,
        addressable: bool,
    ) -> Option<u64> {
        if population.submissions == 0 && !addressable {
            return Some(0);
        }
        let factory = neural::factory::<OriginalOperationMetadataCustody, OriginalScopeObserver>(
            population.shape,
            None,
        );
        let bank = bank_layout(population, &factory)?;
        let fixed = [
            size_of::<Self>(),
            size_of::<bool>(),
            size_of::<SpeculativeNeuralOwner>(),
            size_of::<Owner<U>>(),
            size_of::<Box<dyn ErasedOwner>>(),
            size_of::<Bank>(),
            size_of::<Projection>(),
            size_of::<super::Projection>(),
            size_of::<Option<super::Projection>>(),
            size_of::<Result<Option<SpeculativeNeuralOwner>, Error>>(),
            size_of::<SpeculativeOperationRole>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<OrderedNeuralCompletion>(),
            size_of::<Result<OrderedNeuralCompletion, Error>>(),
            size_of::<std::cell::RefMut<'static, PreparedOperationBank<Prepared>>>(),
            size_of::<
                crate::backend::nn::shared::OriginalSubmissionFailure<
                    OriginalOperationMetadataCustody,
                >,
            >(),
            size_of::<
                Result<
                    crate::backend::nn::shared::OriginalNeuralSubmissionCompletion<
                        OriginalOperationMetadataCustody,
                    >,
                    crate::backend::nn::shared::OriginalSubmissionFailure<
                        OriginalOperationMetadataCustody,
                    >,
                >,
            >(),
            size_of::<eredu_runtime::LayerwiseRuntimeError<eredu_nn::Error, Error>>(),
            size_of::<eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<Option<SpeculativeSourcePartition<'_>>>(),
            size_of::<Option<eredu_runtime::working_memory::OriginalHostSourceBank>>(),
            size_of::<Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, Error>>(
            ),
            size_of::<Result<(), Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        bank.checked_add(
            u64::try_from(
                fixed
                    .checked_add(rc_layout::<Bank>()?)?
                    .checked_add(size_of::<Owner<U>>())?
                    .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                        neural::NeuralBoundaryFailure<SpeculativeOperationRole>,
                    >()?)?,
            )
            .ok()?,
        )?
        .checked_add(neural::clone_error_control_bytes()?)?
        .checked_add(NeuralProducerFit::Recipe(self.fit.requirement()).additional_control_bytes()?)?
        .checked_add(u64::try_from(self.selected_stream?.source_comparison_control_bytes()?).ok()?)
    }
    /// Exact source/recipe and actual active Scope are supplied by the enclosing
    /// invocation compiler. This constructor does not create a Scope or infer
    /// completion; it consumes the role's one group-bank claim before storage.
    pub(crate) fn prepare_speculative(
        self,
        recipe: &AutoregressiveEquationRecipe,
        role: OriginalSpeculativeRole,
        scope: &SubmissionScope,
        partition: Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let role = SpeculativeOperationRole::Autoregressive(role);
        let validation = (|| {
            let SpeculativeOperationRole::Autoregressive(actual) = &role else {
                unreachable!()
            };
            actual.validate_plan(recipe.plan()).map_err(memory)?;
            actual
                .validate_invocation(recipe.invocation())
                .map_err(memory)?;
            let bytes = self.speculative_control_bytes(recipe).ok_or_else(unknown)?;
            role.claim_neural_bank(bytes).map_err(memory)
        })();
        validation.map_err(|cause| neural::boundary_error(cause, &role))?;
        if let Some(partition) = partition {
            let source = role.take_host_source_constructions().map_err(memory)?;
            if partition(source)?.is_some() {
                return Err(neural::boundary_error(identity(), &role));
            }
        }
        let population = self.neural;
        self.prepare_speculative_bank(
            role,
            scope,
            population,
            recipe
                .native_recipe()
                .records()
                .iter()
                .any(|row| row.addressable().is_some()),
        )
    }
    pub(crate) fn prepare_embedded(
        self,
        recipe: &EmbeddedEquationRecipe,
        role: OriginalEmbeddedSpeculativeRole,
        scope: &SubmissionScope,
        partition: Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let role = SpeculativeOperationRole::Embedded(role);
        let validation = (|| {
            let SpeculativeOperationRole::Embedded(actual) = &role else {
                unreachable!()
            };
            actual.validate_plan(recipe.plan()).map_err(memory)?;
            actual
                .validate_invocation(recipe.workspace().invocation())
                .map_err(memory)?;
            if actual.geometry() != recipe.workspace().geometry() {
                return Err(identity());
            }
            let bytes = self.embedded_control_bytes(recipe).ok_or_else(unknown)?;
            role.claim_neural_bank(bytes).map_err(memory)
        })();
        validation.map_err(|cause| neural::boundary_error(cause, &role))?;
        if let Some(partition) = partition {
            let source = role.take_host_source_constructions().map_err(memory)?;
            if partition(source)?.is_some() {
                return Err(neural::boundary_error(identity(), &role));
            }
        }
        let population = self.neural;
        self.prepare_speculative_bank(
            role,
            scope,
            population,
            recipe
                .native_recipe()
                .records()
                .iter()
                .any(|row| row.addressable().is_some()),
        )
    }
    pub(crate) fn prepare_external(
        self,
        recipe: &ResidentNativeRecipe,
        invocation: ExternalInvocation,
        role: OriginalExternalSpeculativeRole,
        scope: &SubmissionScope,
        partition: Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let tagged = SpeculativeOperationRole::External(role);
        let validation = (|| {
            let SpeculativeOperationRole::External(role) = &tagged else {
                unreachable!()
            };
            role.validate_plan(recipe.plan()).map_err(memory)?;
            role.validate_invocation(invocation).map_err(memory)?;
            let controls = self.external_control_bytes(recipe).ok_or_else(unknown)?;
            tagged.claim_neural_bank(controls).map_err(memory)
        })();
        validation.map_err(|cause| neural::boundary_error(cause, &tagged))?;
        if let Some(partition) = partition {
            let source = tagged.take_host_source_constructions().map_err(memory)?;
            if partition(source)?.is_some() {
                return Err(neural::boundary_error(identity(), &tagged));
            }
        }
        let population = self.neural;
        self.prepare_speculative_bank(
            tagged,
            scope,
            population,
            recipe
                .records()
                .iter()
                .any(|row| row.addressable().is_some()),
        )
    }
    pub(crate) fn prepare_speculative_span(
        self,
        recipe: &AutoregressiveEquationRecipe,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        scope: &SubmissionScope,
        partition: Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let role = SpeculativeOperationRole::Autoregressive(span.role().clone());
        let validation = (|| {
            span.validate_plan(recipe.plan()).map_err(memory)?;
            span.role()
                .validate_invocation(recipe.invocation())
                .map_err(memory)?;
            let bytes = self
                .speculative_span_control_bytes(recipe)
                .ok_or_else(unknown)?;
            span.claim_neural_bank(bytes).map_err(memory)
        })();
        validation.map_err(|cause| neural::boundary_error(cause, &role))?;
        if let Some(partition) = partition {
            let source = span.take_host_source_constructions().map_err(memory)?;
            if partition(source)?.is_some() {
                return Err(neural::boundary_error(identity(), &role));
            }
        }
        let population = NeuralPopulation {
            submissions: self.neural.per_forward,
            ..self.neural
        };
        self.prepare_speculative_bank(
            role,
            scope,
            population,
            recipe
                .native_recipe()
                .records()
                .iter()
                .any(|row| row.addressable().is_some()),
        )
    }
    fn prepare_speculative_bank(
        self,
        role: SpeculativeOperationRole,
        scope: &SubmissionScope,
        population: NeuralPopulation,
        addressable: bool,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let result = (|| {
            if population.submissions == 0 && !addressable {
                return Ok(None);
            }
            let observer = OriginalScopeObserver::require_current()?;
            if !observer.belongs_to(scope) {
                return Err(identity());
            }
            let previous = self.slot.0.resident.take();
            let busy = previous.as_ref().is_some_and(super::Projection::is_live);
            self.slot.0.resident.set(previous);
            if busy {
                return Err(Error::PrefillScopeReentrant);
            }
            let prepared = neural::prepare_with_custody(population, role.budget_custody().into())?;
            let bank = Rc::new(Bank {
                prepared: RefCell::new(prepared),
                observer,
                stream: self.selected_stream.ok_or_else(identity)?,
                active: Cell::new(true),
                role: role.clone(),
            });
            let replaced = self
                .slot
                .0
                .resident
                .replace(Some(super::Projection::Speculative(Projection {
                    value: Rc::downgrade(&bank),
                    role: role.clone(),
                })));
            drop(replaced);
            Ok(Some(SpeculativeNeuralOwner {
                value: Box::new(Owner {
                    bank,
                    slot: self.slot,
                }),
                role: role.clone(),
            }))
        })();
        result.map_err(|cause| neural::boundary_error(cause, &role))
    }
}
fn bank_layout<F>(population: NeuralPopulation, _factory: &F) -> Option<u64>
where
    F: FnMut(usize) -> Result<Prepared, PreparationError>,
{
    PreparedOperationBank::<Prepared>::layout::<F, PreparationError>(
        population.submissions,
        Prepared::control_bytes(population.shape)?,
    )
    .map(|layout| layout.total_control_bytes)
}

/// Weak source projection pins the actual bank header but cannot keep its
/// installation active after the resident owner retires.
#[derive(Clone)]
pub(in crate::backend::runtime::execution::generic::original_operations) struct SelectedResidencyProjection
{
    value: Weak<Bank>,
    role: SpeculativeOperationRole,
}
impl Projection {
    pub(super) fn selected_residency(
        &self,
        stream: &Stream,
    ) -> Result<SelectedResidencyProjection, Error> {
        self.access(stream)?;
        Ok(SelectedResidencyProjection {
            value: self.value.clone(),
            role: self.role.clone(),
        })
    }
}
impl SelectedResidencyProjection {
    pub(in crate::backend::runtime::execution::generic::original_operations) fn authenticate(
        &self,
    ) -> Result<OriginalScopeObserver, Error> {
        let observer = OriginalScopeObserver::require_current()?;
        self.authenticate_observer(&observer)?;
        Ok(observer)
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn authenticate_observer(
        &self,
        observer: &OriginalScopeObserver,
    ) -> Result<(), Error> {
        let bank = self.value.upgrade().ok_or_else(identity)?;
        if !bank.active.get()
            || !bank.role.same_role(&self.role)
            || !bank.observer.same_scope(observer)
        {
            return Err(identity());
        }
        Ok(())
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn same_source(
        &self,
        other: &Self,
    ) -> bool {
        Weak::ptr_eq(&self.value, &other.value) && self.role.same_role(&other.role)
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn custody(
        &self,
    ) -> eredu_runtime::working_memory::OriginalHostSourceCustody {
        self.role.budget_custody().into()
    }
}
