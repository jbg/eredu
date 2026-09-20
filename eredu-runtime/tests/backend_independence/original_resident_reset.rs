//! Actual shared-session source mint plus genuine core reset claim. This fixture
//! tests provisional construction only; no native publication is simulated.
use super::*;
use eredu_core::{
    BackendDescriptor, BackendFailure, BackendProvider, BackendSession, DeviceCapabilities,
    DeviceDescriptor, ModelRuntime, ObservationSet, PendingTextInput, PreparedModel,
    SessionCapabilities, SessionResetLimits, TextGenerationBackend, TextGenerationConfig,
    TextPreparationInput, TextStepContext, TokenFilterController, TokenOutput,
};
use eredu_runtime::working_memory::{
    HostSlotStorageKey, InferenceStateRetention, PreparedResidentKvReset, ResidentKvResetLayer,
    ResidentTableResetState, ResidentResetSession, ResidentResetSource, WorkingMemoryError,
    WorkingMemoryPool, WorkingMemoryStorage,
};
type ActualSession =
    ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;
type State = DeviceState<FakeBackend, FakeLayerState>;
impl ResidentKvResetLayer for FakeLayerState {
    fn matches_resident_reset(&self, policy: &LayerCachePolicy) -> bool {
        matches!(policy, LayerCachePolicy::KeyValue { .. })
    }
    fn empty_resident_reset(_: &LayerCachePolicy) -> Self {
        Self::new(0)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(eredu_runtime::HostMetadataKey);
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&eredu_runtime::HostMetadataKey> {
        Some(&self.0)
    }
}
struct Backend;
struct Session {
    inner: ActualSession,
    pool: WorkingMemoryPool,
    registration: WorkingMemoryStorage<Key>,
    replacement: Rc<RefCell<Option<State>>>,
}
impl ResidentResetSession<State> for Session {
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, State>,
    ) -> Result<(), WorkingMemoryError> {
        if self.inner.resident_reset_source()?.same_source(source) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
#[derive(Clone)]
struct Token;
struct Done;
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        unreachable!("cold fixture")
    }
}
impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        unreachable!("cold fixture")
    }
    fn wait(&self) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
}
impl BackendProvider for Backend {
    type ModelConfig = Session;
    type Model = Session;
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("actual-selected-reset-construction", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(vec![])
    }
    fn prepare_model(&self, session: Session) -> Result<PreparedModel<Session>, Self::Error> {
        Ok(PreparedModel::new(session, SessionCapabilities::default()))
    }
    fn create_session(&self, prepared: PreparedModel<Session>) -> Result<Session, Self::Error> {
        Ok(prepared.into_inner())
    }
    fn session_capability_mismatch(
        &self,
        _: SessionCapabilities,
        _: SessionCapabilities,
    ) -> Self::Error {
        WorkingMemoryError::IdentityMismatch
    }
}
impl BackendSession<Backend> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &Backend,
        _: Vec<u32>,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
    fn decode(
        &mut self,
        _: &Backend,
        _: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<ObservationSet, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
}
impl TextGenerationBackend for Backend {
    fn reset_session_admitted(
        _: &Self,
        session: &mut Session,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        let source = session
            .inner
            .resident_reset_source()
            .map_err(BackendFailure::from_error)?;
        let plan = PreparedResidentKvReset::prepare(source, &session.registration)
            .map_err(BackendFailure::from_error)?;
        let replacement = plan
            .construct(&*session, claim, &session.pool)
            .map_err(BackendFailure::from_error)?;
        *session.replacement.borrow_mut() = Some(replacement);
        Ok(())
    }

    fn reset_session(
        _: &Self,
        _: &mut Session,
        _claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    type TextPreparation = ();
    type TextPreparationControl = ();
    type TextStepPermit = ();
    type Prompt = Vec<u32>;
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Done;
    fn begin_text_step<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &(),
        _: &(),
        _: &C,
        _: PendingTextInput<&Vec<u32>, &Token>,
        _: &TextStepContext,
    ) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn finish_text_step(_: ()) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Vec<u32>>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<(), BackendFailure> {
        unreachable!("cold fixture")
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn prepare_text_prompt(_: &Self, _: Vec<u32>) -> Result<Vec<u32>, Self::Error> {
        unreachable!("cold fixture")
    }
    fn submit_text_prefill(
        _: &mut ModelRuntime<Self>,
        _: Vec<u32>,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        unreachable!("cold fixture")
    }
    fn submit_text_decode(
        _: &mut ModelRuntime<Self>,
        _: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        unreachable!("cold fixture")
    }
}

#[test]
fn actual_selected_session_borrow_constructs_provisional_empty_state_under_real_core_claim() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
        LayerWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (mut inner, counters) = super::prepared_session_observation::session(residency);
        // Canonical ordinary execution makes the source nonzero before capture.
        // Disk here proves the selected neutral bounded driver, not file I/O.
        assert_eq!(
            inner.forward(&FakeTensor(vec![3, 7]), None, &()).unwrap(),
            FakeTensor(vec![3, 7, 5])
        );
        let report = inner.report().unwrap();
        assert_eq!(report.execution(), residency.execution_residency());
        assert_eq!(
            report.execution_report(),
            &(residency, !residency.is_fully_resident())
        );
        assert_eq!(report.state_report(), &[1]);
        let counts = counters.snapshot();
        assert_eq!(counts.forward_calls, 1);
        assert_eq!(counts.completion_attempts, 1);
        assert_eq!(counts.publications, 1);
        assert_eq!(
            (counts.resident_policies, counts.bounded_policies),
            if residency.is_fully_resident() {
                (1, 0)
            } else {
                (0, 1)
            }
        );
        assert_eq!(
            counts.unit_constructions,
            1 + usize::from(!residency.is_fully_resident())
        );
        let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
        let source = inner.resident_reset_source().unwrap();
        let state = source.state();
        let table = state.resident_reset_layers().metadata();
        let layout = state.resident_reset_layout();
        let registration = pool
            .register_storage([
                (
                    Key(table.identity().registry_key().clone()),
                    table.capacity_bytes().unwrap(),
                ),
                (
                    Key(layout.identity().registry_key().clone()),
                    layout.capacity_bytes().unwrap(),
                ),
            ])
            .unwrap();
        let plan = PreparedResidentKvReset::prepare(source, &registration).unwrap();
        let required = plan.required_bytes();
        let original_layout = state.resident_reset_layout().clone();
        let original_revision = state.inference_retention().revision().clone();
        drop(plan);
        let replacement = Rc::new(RefCell::new(None));
        let mut runtime = ModelRuntime::prepare(
            Backend,
            Session {
                inner,
                pool: pool.clone(),
                registration,
                replacement: replacement.clone(),
            },
        )
        .unwrap();
        runtime
            .reset_admitted(SessionResetLimits {
                capacity_bytes: 10_000_000,
                application_memory_budget_bytes: Some(required - 1),
                safety_reserve_bytes: 0,
            })
            .unwrap_err();
        assert!(replacement.borrow().is_none());
        runtime
            .reset_admitted(SessionResetLimits::new(10_000_000))
            .unwrap();
        let state = replacement.borrow_mut().take().unwrap();
        assert!(state
            .as_ref()
            .iter()
            .all(|layer| layer.0 == 0 && layer.1.is_none()));
        assert!(state.resident_reset_layout().same_storage(&original_layout));
        assert!(state.inference_retention().is_empty());
        assert_ne!(state.inference_retention().revision(), &original_revision);
        let identity = state.resident_reset_layers().metadata().identity().clone();
        let remaining = original_layout.capacity_bytes().unwrap() + required;
        drop(state);
        drop(runtime);
        drop(original_layout);
        drop(original_revision);
        assert_eq!(pool.used_bytes().unwrap(), remaining);
        drop(identity);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
