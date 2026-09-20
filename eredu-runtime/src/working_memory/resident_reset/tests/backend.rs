use super::*;
pub(super) struct Session(pub(super) Rc<RefCell<Data>>);
#[derive(Clone)]
pub(super) struct Token;
pub(super) struct Done;
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
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("resident-reset-host", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(vec![])
    }
    fn prepare_model(&self, _: ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> Result<Session, Self::Error> {
        Ok(Session(self.data.clone()))
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
        backend: &Self,
        session: &mut Session,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        backend.construct(session, claim)
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
