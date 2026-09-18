//! Resident graph boundaries share the finite neural producer and scope registry.
//! No unit loading, transfer, source-copy allowance or native child is inferred.
use super::neural::OrderedNeuralCompletion;
use super::*;
use super::neural::PreparedNeuralSubmission;
use eredu_runtime::SubmissionBackend;

#[derive(Clone)]
pub(super) struct TextProjection {
    value: Weak<ResidentBank>,
    controls: OriginalTextControlGuard,
}
#[derive(Clone)]
pub(super) enum Projection {
    Text(TextProjection),
    Speculative(speculative::Projection),
    Realtime(realtime::Projection),
}
impl Projection {
    fn require_idle(&self) -> Result<(), Error> {
        match self {
            Self::Text(value) => match value.value.upgrade() {
                Some(bank) => {
                    bank.registry.require_idle()?;
                    drop(bank.prepared.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?);
                    Ok(())
                }
                None => Ok(()),
            },
            _ if self.is_live() => Err(Error::PrefillScopeReentrant),
            _ => Ok(()),
        }
    }
    fn is_live(&self) -> bool {
        match self {
            Self::Text(value) => value.value.strong_count() != 0,
            Self::Speculative(value) => value.is_live(),
            Self::Realtime(value) => value.is_live(),
        }
    }
}
mod speculative;
mod realtime;
pub(crate) use realtime::{RealtimeNeuralPlan,QualifiedRealtimeNeuralPlan,RealtimeNeuralOwner};
pub(crate) use speculative::SpeculativeNeuralOwner;
pub(super) use speculative::SelectedResidencyProjection;
pub(crate) struct ResidentNeuralSlot<U: 'static>(InstallSlot<U>);
impl<U: 'static> Clone for ResidentNeuralSlot<U> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<U: 'static> ResidentNeuralSlot<U> {
    /// Explicit installed-slot loan for addressable acquisition. No current
    /// observer is used to choose or reconstruct the model's source owner.
    pub(crate) fn selected_residency_access(&self,stream:&Stream)->Result<OriginalSelectedResidencyAccess,Error> {
        match self.projection().ok_or_else(identity)? {
            Projection::Text(source)=>{
                let bank=source.value.upgrade().ok_or_else(identity)?;
                if !bank.selected_stream.matches_source(stream){return Err(identity());}
                OriginalSelectedResidencyAccess::registered(Rc::clone(&bank.registry),OperationControls::Text(source.controls))
            },
            Projection::Speculative(source)=>OriginalSelectedResidencyAccess::resident(source.selected_residency(stream)?),
            Projection::Realtime(_)=>Err(identity()),
        }
    }

    pub(crate) fn from_slot(slot: InstallSlot<U>) -> Self {
        Self(slot)
    }
    pub(crate) fn plan(
        &self,
        layout: &ExecutionUnitLayout,
        geometry: eredu_core::InferenceGeometry,
        groups: eredu_runtime::GroupSubmissionMechanism,
    ) -> Result<ResidentNeuralPlan<U>, Error> {
        let neural = NeuralPopulation::from_execution(layout, geometry, groups, false)?;
        Ok(ResidentNeuralPlan {
            slot: self.clone(),
            geometry,
            neural,
            scopes: operation_counts(0, geometry)?.scopes,
            fit: NeuralProducerFit::pending(neural).ok_or_else(overflow)?,
            selected_stream: None,
            addressable: false,
        })
    }
    fn projection(&self) -> Option<Projection> {
        let view = self.0.resident.take();
        let result = view.clone();
        self.0.resident.set(view);
        result
    }
    pub(in crate::backend::runtime::execution::generic) fn uses_shared_executor(
        &self,
        stream: &Stream,
    ) -> Result<bool, Error> {
        let projection = match self.projection() {
            None => return Ok(false),
            Some(Projection::Text(value)) => value,
            Some(Projection::Speculative(value)) => return value.uses_shared_executor(stream),
            Some(Projection::Realtime(value)) => return value.uses_shared_executor(stream),
        };
        let result = (|| {
            let bank = projection
                .value
                .upgrade()
                .ok_or(Error::PrefillScopeUnavailable)?;
            projection
                .controls
                .validate_reservation(
                    bank.registry
                        .request
                        .memory_reservation()
                        .ok_or_else(identity)?,
                )
                .map_err(memory)?;
            bank.registry.authenticate()?;
            if !bank.selected_stream.matches_source(stream) {
                return Err(identity());
            }
            Ok(true)
        })();
        result.map_err(|cause| neural::boundary_error(cause, &projection.controls))
    }
    pub(in crate::backend::runtime::execution::generic) fn submit(
        &self,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<OrderedNeuralCompletion, Error> {
        let projection = match self.projection() {
            None => {
                return MlxNeuralBackend::submit(stream, [value])
                    .map(OrderedNeuralCompletion::Ordinary)
                    .map_err(Error::from)
            }
            Some(Projection::Text(value)) => value,
            Some(Projection::Speculative(projection)) => return projection.submit(stream, value),
            Some(Projection::Realtime(projection)) => return projection.submit(stream, value),
        };
        let result = (|| {
            let bank = projection
                .value
                .upgrade()
                .ok_or(Error::PrefillScopeUnavailable)?;
            projection
                .controls
                .validate_reservation(
                    bank.registry
                        .request
                        .memory_reservation()
                        .ok_or_else(identity)?,
                )
                .map_err(memory)?;
            let observer = bank.registry.authenticate()?;
            if !bank.selected_stream.matches_source(stream) {
                return Err(identity());
            }
            let prepared = bank
                .prepared
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .checkout()
                .map_err(|_| Error::PrefillScopeUnavailable)?;
            neural::submit_prepared(prepared, observer, value, stream, &OperationControls::Text(projection.controls.clone()))
        })();
        result.map_err(|cause| neural::boundary_error(cause, &projection.controls))
    }
}

pub(crate) struct ResidentNeuralPlan<U: 'static> {
    slot: ResidentNeuralSlot<U>,
    geometry: eredu_core::InferenceGeometry,
    neural: NeuralPopulation,
    scopes: usize,
    fit: NeuralProducerFit,
    selected_stream: Option<safemlx::StreamCopyPlan<()>>,
    addressable: bool,
}
impl<U: 'static> ResidentNeuralPlan<U> {
    pub(crate) fn with_selected_stream(mut self, stream: &Stream) -> Result<Self, Error> {
        self.selected_stream =
            Some(safemlx::StreamCopyPlan::capture(stream).map_err(|_| identity())?);
        Ok(self)
    }

    pub(crate) fn bind_neural_recipe(
        &self,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(), Error> {
        if self.neural.submissions == 0 {
            return Ok(());
        }
        recipe.bind_neural_boundaries(
            self.geometry,
            self.neural.per_forward,
            self.neural.shape.consumers(),
        )
    }
    pub(crate) fn with_neural_recipe(
        mut self,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
    ) -> Result<Self, Error> {
        // A sequential graph can have no neural boundaries and still execute
        // addressable cache acquisitions. They require the same registered
        // request/stream owner, with an exactly empty neural submission bank.
        self.addressable = recipe.is_some_and(|recipe| recipe.records().iter()
            .any(|row| row.addressable().is_some()));
        if self.neural.submissions == 0 {
            return Ok(self);
        }
        if !recipe.is_some_and(|recipe| {
            recipe.matches_neural_boundaries(
                self.geometry,
                self.neural.submissions,
                self.neural.shape.consumers(),
            )
        }) {
            return Err(identity());
        }
        self.fit = NeuralProducerFit::Recipe(self.fit.requirement());
        Ok(self)
    }
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        if self.neural.submissions == 0 && !self.addressable {
            // This branch does not enter the neural bank's measured layout,
            // but its population source still passed the selected mechanism.
            return selected_plan_control_bytes::<U>()?.checked_add(u64::try_from(
                NeuralPopulation::group_source_control_bytes()?).ok()?);
        }
        let scopes = Layout::array::<safemlx::OriginalScopeObserver>(self.scopes)
            .ok()?
            .size();
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ResidentNeuralSlot<U>>(),
            size_of::<Projection>(),
            size_of::<Option<Projection>>(),
            size_of::<ResidentBank>(),
            size_of::<OwnedResidentBank<U>>(),
            size_of::<OriginalOperationBankOwner>(),
            size_of::<Box<dyn ErasedOwner>>(),
            size_of::<Registry>(),
            size_of::<Rc<Registry>>(),
            size_of::<Rc<ResidentBank>>(),
            size_of::<Vec<safemlx::OriginalScopeObserver>>(),
            size_of::<OriginalTextControlGuard>(),
            size_of::<OriginalOperationRegistration>(),
            size_of::<Result<OriginalOperationBankOwner, Error>>(),
            size_of::<Result<Option<OriginalOperationBankOwner>, Error>>(),
            size_of::<std::cell::RefMut<'static, PreparedOperationBank<PreparedNeuralSubmission>>>(
            ),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<PreparationFailure>(),
            size_of::<(PreparedNeuralSubmission, safemlx::OriginalScopeObserver)>(),
            size_of::<&ResidentBank>(), size_of::<&OwnedResidentBank<U>>(),
            size_of::<Option<Rc<ResidentBank>>>(), size_of::<TextProjection>(),
            size_of::<Option<&Projection>>(), size_of::<Result<(), Error>>(),
            OriginalOperationActivation::control_bytes()?,
        ]
        .into_iter()
        .try_fold(scopes, usize::checked_add)?;
        let allocations = rc_layout::<ResidentBank>()?
            .checked_add(rc_layout::<Registry>()?)?
            .checked_add(size_of::<OwnedResidentBank<U>>())?
            .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                PreparationFailure,
            >()?)?;
        self.neural
            .rust_control_bytes()?
            .checked_add(
                u64::try_from(self.selected_stream?.source_comparison_control_bytes()?).ok()?,
            )?
            .checked_add(selected_plan_control_bytes::<U>()?)?
            .checked_add(if self.neural.submissions == 0 { 0 }
                else { self.fit.additional_control_bytes()? })?
            .checked_add(u64::try_from(fixed.checked_add(allocations)?).ok()?)
    }
    pub(crate) fn prepare_install(
        self,
        original: &OriginalTextPrefillScopeSet,
        step: &InferenceTextStep,
        registration: OriginalOperationRegistration,
        controls: OriginalTextControlGuard,
        host_destinations: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> Result<Option<OriginalOperationBankOwner>, Error> {
        original.validate_request(step.request()).map_err(memory)?;
        controls
            .validate_reservation(step.request().memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        if original.facts().plan().geometry() != self.geometry
            || original.facts().operation_control_bytes()
                != Some(self.control_bytes().ok_or_else(unknown)?)
            || original.facts().source_construction_facts().is_some()
            || original.facts().host_destination_facts().is_some()
            || host_destinations.is_some()
        {
            return Err(identity());
        }
        if self.neural.submissions == 0 && !self.addressable {
            return Ok(None);
        }
        let previous = self.slot.0.resident.take();
        let idle = previous.as_ref().map_or(Ok(()), Projection::require_idle);
        self.slot.0.resident.set(previous);
        idle?;
        let mut scopes = Vec::new();
        scopes
            .try_reserve_exact(self.scopes)
            .map_err(|cause| reserve_error(cause, &controls))?;
        let prepared = neural::prepare(self.neural, &controls)?;
        let registry = Rc::new(Registry {
            request: step.request().clone().into(),
            scopes: RefCell::new(scopes),
            scope_limit: self.scopes,
            registered_scopes: Cell::new(0),
            active: Cell::new(true),
            entered: Cell::new(false),
            retirement_failure: Cell::new(None),
        });
        let bank = Rc::new(ResidentBank {
            selected_stream: self.selected_stream.ok_or_else(identity)?,
            prepared: RefCell::new(prepared),
            registry,
        });
        registration.install(&bank.registry)?;
        let replaced = self
            .slot
            .0
            .resident
            .replace(Some(Projection::Text(TextProjection {
                value: Rc::downgrade(&bank),
                controls: controls.clone(),
            })));
        drop(replaced);
        Ok(Some(OriginalOperationBankOwner {
            value: Box::new(OwnedResidentBank {
                bank,
                slot: self.slot,
                registration,
            }),
            controls: OperationControls::Text(controls),
        }))
    }
}
struct ResidentBank {
    selected_stream: safemlx::StreamCopyPlan<()>,
    prepared: RefCell<PreparedOperationBank<PreparedNeuralSubmission>>,
    registry: Rc<Registry>,
}
struct OwnedResidentBank<U: 'static> {
    bank: Rc<ResidentBank>,
    slot: ResidentNeuralSlot<U>,
    registration: OriginalOperationRegistration,
}
impl<U: 'static> ErasedOwner for OwnedResidentBank<U> {
    fn activate(&self, controls: &OperationControls) -> Result<OriginalOperationActivation, Error> {
        let OperationControls::Text(controls) = controls else { return Err(identity()); };
        controls.validate_reservation(self.bank.registry.request.memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        self.bank.registry.require_idle()?;
        drop(self.bank.prepared.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?);
        let previous = self.slot.0.resident.take();
        let idle = previous.as_ref().map_or(Ok(()), Projection::require_idle);
        self.slot.0.resident.set(previous);
        idle?;
        let previous = self.slot.0.resident.replace(Some(Projection::Text(TextProjection {
            value: Rc::downgrade(&self.bank), controls: controls.clone(),
        })));
        let guard = self.bank.registry.enter();
        drop(previous);
        Ok(guard)
    }
}
impl<U: 'static> Drop for OwnedResidentBank<U> {
    fn drop(&mut self) {
        let previous = self.slot.0.resident.take();
        if previous
            .as_ref()
            .is_some_and(|view| matches!(view, Projection::Text(view) if Weak::ptr_eq(&view.value, &Rc::downgrade(&self.bank))))
        {
            self.slot.0.resident.set(None);
            drop(previous);
        } else {
            self.slot.0.resident.set(previous);
        }
        self.bank.registry.active.set(false);
        self.registration.clear(&self.bank.registry);
        // Outer OriginalOperationBankOwner controls outlive bank, registry and Box.
    }
}
