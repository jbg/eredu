use crate::composition::mlx::capability::MlxAllocatorObservations;
use eredu_core::session_authority::{RetainedExecutionPayload, RetainedPayloadOperation};
use eredu_core::speculative::assistant::ExternalAssistantBinder;

use super::*;
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use crate::{
    backend::runtime::checkpoint::{
        recipe::native_recipe_workspace, store::MlxParameterMaterializationContext,
    },
    native::MlxComponentBinder,
};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Reports only generic mechanisms available to neutral speculative selection.
///
/// Architecture identity, proposal mode, capture paths, and assistant family
/// never enter this report.
pub(crate) fn speculative_mechanism_capabilities() -> SpeculativeMechanismCapabilities {
    SpeculativeMechanismCapabilities::new([
        SpeculativeMechanism::TensorOperations,
        SpeculativeMechanism::NeuralOperations,
        SpeculativeMechanism::GroupedNeuralOperations,
        SpeculativeMechanism::HyperNeuralOperations,
        SpeculativeMechanism::PayloadMaterialization,
        SpeculativeMechanism::LogitsProcessing,
        SpeculativeMechanism::Sampling,
        SpeculativeMechanism::Randomness,
        SpeculativeMechanism::StateStorage,
        SpeculativeMechanism::StorageResidency,
        SpeculativeMechanism::ExactCompletion,
        SpeculativeMechanism::Observation,
        SpeculativeMechanism::Timing,
        SpeculativeMechanism::QueueBinding,
        SpeculativeMechanism::Communication,
        SpeculativeMechanism::Agreement,
        SpeculativeMechanism::Publication,
        SpeculativeMechanism::SameDeviceHandoff,
        SpeculativeMechanism::CrossDeviceTransfer,
    ])
}

/// Native checkpoint and stream binding for an admitted external assistant.
pub struct MlxAssistantBinder {
    pub(super) stream: Stream,
    pub(super) weights_stream: Stream,
}

impl ExternalAssistantBinder for MlxAssistantBinder {
    type Backend = MlxNeuralBackend;
    type Storage = MlxComponentBinder;
    type Logits = Array;
    type ObservationError = Exception;
    type Error = Error;

    fn context(&self) -> &Stream {
        &self.stream
    }

    fn storage(&self) -> MlxComponentBinder {
        MlxComponentBinder::new(&self.weights_stream, &self.stream)
    }

    type Parameters = MlxNeuralBackend;

    fn recipe_workspace(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<u64, Error> {
        native_recipe_workspace(recipe, source).map_err(Error::ArchitectureModel)
    }

    type ParameterReads = MlxParameterMaterializationContext;

    fn parameter_reads(&self) -> Self::ParameterReads {
        MlxParameterMaterializationContext::new(&self.weights_stream, &self.stream)
    }
}

/// Exact MLX submission and retirement resources protecting an opaque payload.
pub struct MlxDraftStorage<P: 'static> {
    payload: Rc<OrdinaryRetirement<DrafterPayload<P>>>,
    poisoned: Rc<Cell<bool>>,
    _retirement: ordinary_retirement::ReclaimOnDrop,
}

struct DrafterPayload<P> {
    payload: P,
    stream: Stream,
}

struct DrafterRetention<P: 'static> {
    _payload: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload<P>>>>>>,
    poisoned: Rc<Cell<bool>>,
}

impl<P: 'static> Retention for DrafterRetention<P> {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.poisoned.set(true);
        }
    }
}

#[doc(hidden)]
pub struct DrafterOperation<'a, P: 'static> {
    drafter: &'a mut MlxDraftStorage<P>,
    retained: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload<P>>>>>>,
    recovery: Recovery<DrafterRetention<P>>,
    completed: bool,
}

impl<P: 'static> DrafterOperation<'_, P> {
    fn payload(&mut self) -> &mut DrafterPayload<P> {
        Rc::get_mut(&mut self.drafter.payload).expect("healthy drafter owns its native payload")
    }

    fn finish<T>(mut self, result: Result<T, Error>) -> Result<T, Error> {
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        // One nonblocking poll may leave healthy CPU work or native recovery
        // records pending. A successful synchronous operation must wait for
        // completion; failures retain their nonblocking recovery path.
        let status = if result.is_ok() {
            self.recovery.wait()
        } else {
            self.recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            self.drafter.poisoned.set(true);
            let message = match &result {
                Ok(_) => "native drafter work failed or remains unresolved".to_owned(),
                Err(error) => format!("native drafter work failed or remains unresolved: {error}"),
            };
            return Err(Error::Speculative(message));
        }
        let value = result?;
        self.completed = true;
        self.retained.take();
        Ok(value)
    }
}

impl<P: 'static> RetainedPayloadOperation<Error> for DrafterOperation<'_, P> {
    type Payload = P;

    fn payload_mut(&mut self) -> &mut Self::Payload {
        &mut self.payload().payload
    }

    fn settle<T>(self, result: Result<T, Error>) -> Result<T, Error> {
        self.finish(result)
    }
}

impl<P: 'static> Drop for DrafterOperation<'_, P> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        self.drafter.poisoned.set(true);
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        let status = self.recovery.progress();
        if status.failed || status.blocked || std::thread::panicking() {
            self.drafter.poisoned.set(true);
        }
    }
}

impl<P: 'static> RetainedExecutionPayload for MlxDraftStorage<P> {
    type Payload = P;
    type Queue = Stream;
    type Error = Error;
    type Operation<'a> = DrafterOperation<'a, P>;

    fn payload(&self) -> &P {
        &self.payload.payload
    }

    fn retain_queue(&self) -> Stream {
        self.payload.stream.clone()
    }

    fn begin_operation(&mut self) -> Result<Self::Operation<'_>, Error> {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        if self.poisoned.get() {
            return Err(Error::Speculative(
                "native drafter is poisoned by failed or unresolved work".into(),
            ));
        }
        // A healthy terminal scope can still retain its payload while another
        // thread owns the native runtime. Terminal evidence is not exclusive
        // mutation authority: reject until retirement releases that owner.
        if Rc::get_mut(&mut self.payload).is_none() {
            return Err(Error::Speculative(
                "native drafter payload is still retained by prior work".into(),
            ));
        }
        let retained = Rc::new(RefCell::new(None));
        let recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&self.poisoned),
        })?;
        Ok(DrafterOperation {
            drafter: self,
            retained,
            recovery,
            completed: false,
        })
    }
}

impl<P: 'static> eredu_core::capability::NativePlacementMemory for MlxDraftStorage<P> {
    type Memory = MlxAllocatorObservations;

    fn host_execution(&self) -> Result<bool, eredu_core::BackendFailure> {
        self.payload
            .stream
            .get_device()
            .and_then(|device| device.get_type())
            .map(|kind| kind == safemlx::DeviceType::Cpu)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn scratch_memory_upper_bound(&self) -> Option<u64> {
        None
    }
}

/// Native construction scope retained while portable code builds a selected draft.
#[doc(hidden)]
pub struct MlxDraftBinding<Dispatch: 'static, P: 'static> {
    recovery: Recovery<DrafterRetention<P>>,
    retained: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload<P>>>>>>,
    poisoned: Rc<Cell<bool>>,
    backend: crate::backend::MlxBackend<'static, Dispatch>,
}

impl<Dispatch: 'static, P: 'static> MlxDraftBinding<Dispatch, P> {
    pub(in crate::composition::mlx) fn begin(
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        ordinary_retirement::reclaim();
        let poisoned = Rc::new(Cell::new(false));
        let retained = Rc::new(RefCell::new(None));
        let recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&poisoned),
        })?;
        Ok(Self {
            recovery,
            retained,
            poisoned,
            backend: crate::backend::MlxBackend::from_streams(stream, weights_stream),
        })
    }
}

impl<Dispatch: 'static, P: 'static> eredu_core::automatic::DraftBindingResources
    for MlxDraftBinding<Dispatch, P>
{
    type Backend = crate::backend::MlxBackend<'static, Dispatch>;
    type AssistantBinder = MlxAssistantBinder;
    type Payload = P;
    type Resources = MlxDraftStorage<P>;

    fn assistant_binder(&self) -> Self::AssistantBinder {
        MlxAssistantBinder {
            stream: self.backend.stream().clone(),
            weights_stream: self.backend.weights_stream().clone(),
        }
    }

    fn model_binding(
        &self,
    ) -> (
        &Self::Backend,
        <Self::Backend as eredu_core::ModelLoadingBackend>::Binding,
    ) {
        (
            &self.backend,
            super::super::loading::MlxModelPlacement::direct(None),
        )
    }

    fn finish_binding(
        mut self,
        value: P,
    ) -> Result<Self::Resources, eredu_core::AutomaticPlanningError> {
        let stream = self.backend.stream();
        let payload = Rc::new(OrdinaryRetirement::new(DrafterPayload {
            payload: value,
            stream: stream.clone(),
        }));
        self.retained.replace(Some(Rc::clone(&payload)));
        self.recovery.seal();
        let status = self.recovery.wait();
        if !status.settled || status.failed || status.blocked {
            return Err(eredu_core::AutomaticPlanningError::Backend {
                operation: "realize_external_drafter",
                message: "native drafter materialization failed or remains unresolved".into(),
            });
        }
        drop(self.recovery);
        self.retained.borrow_mut().take();
        let retained = MlxDraftStorage {
            payload,
            poisoned: self.poisoned,
            _retirement: ordinary_retirement::ReclaimOnDrop,
        };
        stream
            .synchronize()
            .map_err(|error| eredu_core::AutomaticPlanningError::Backend {
                operation: "complete_external_drafter_load",
                message: error.to_string(),
            })?;
        Ok(retained)
    }
}
