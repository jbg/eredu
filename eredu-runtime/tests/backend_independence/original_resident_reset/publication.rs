//! Uses the same selected source, metadata registration and prepared publication
//! for ordinary and partitioned fixture sessions.
use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::{
    parameter_operations::{
        ParameterPublication, ParameterReplacementValues, PreparedParameterPublication,
    },
    working_memory::{
        InferenceExecutionIdentity, ResidentResetPublicationCustody,
        ResidentResetPublicationProfile,
    },
};

pub(crate) trait ControlSession {
    fn source(&self) -> Result<ResidentResetSource<'_, State>, WorkingMemoryError>;
    fn execution(&self) -> &InferenceExecutionIdentity;
    fn publication(
        &mut self,
        visitor: &mut dyn ParameterPublication<FakeTensor>,
    ) -> Result<bool, &'static str>;
    fn finalize(&mut self);
}
macro_rules! control_session {
    ($session:ty) => {
        impl ControlSession for $session {
            fn source(&self) -> Result<ResidentResetSource<'_, State>, WorkingMemoryError> {
                self.resident_reset_source()
            }
            fn execution(&self) -> &InferenceExecutionIdentity {
                self.inference_execution_identity()
            }
            fn publication(
                &mut self,
                visitor: &mut dyn ParameterPublication<FakeTensor>,
            ) -> Result<bool, &'static str> {
                self.visit_parameter_publication(visitor)
            }
            fn finalize(&mut self) {
                self.finalize_parameter_publication();
            }
        }
    };
}
control_session!(ActualSession);
control_session!(ReferencePartitionedSession);
struct Source<'a, S>(&'a S);
impl<S: ControlSession> ResidentResetSession<State> for Source<'_, S> {
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, State>,
    ) -> Result<(), WorkingMemoryError> {
        if self.0.source()?.same_source(source) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
pub(crate) struct Publication {
    _custody: ResidentResetPublicationCustody,
}
impl ResidentResetPublicationProfile for Publication {
    fn control_bytes() -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>()).ok()
    }
    fn prepare(custody: ResidentResetPublicationCustody) -> Self {
        Self { _custody: custody }
    }
}
pub(crate) struct ResetCustody {
    _registration: WorkingMemoryStorage<Key>,
    _publication: Publication,
}
pub(crate) fn construct_reset(
    session: &impl ControlSession,
    pool: &MemoryLedger,
) -> (
    eredu_runtime::working_memory::ResidentResetInstallation<State>,
    ResetCustody,
) {
    let source = session.source().unwrap();
    let table = source.state().resident_reset_layers().metadata();
    let layout = source.state().resident_reset_layout();
    let registration = pool
        .register_host_storage([
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
    let plan = PreparedResidentKvReset::prepare(source, &registration, pool).unwrap();
    let (installation, publication) = plan
        .construct_for_parameter_publication::<_, Publication>(
            &Source(session),
            session.execution(),
            pool.configured_limits(),
            pool,
        )
        .unwrap();
    (
        installation,
        ResetCustody {
            _registration: registration,
            _publication: publication,
        },
    )
}

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        unreachable!("metadata-only parameter publication")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}
pub(crate) fn prepare_generation(
    session: &mut impl ControlSession,
    pool: &MemoryLedger,
) -> PreparedParameterPublication<FakeTensor> {
    let funding = pool
        .prepare_workspace_metadata(session.execution(), pool.configured_limits().clone())
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    PreparedParameterPublication::prepare(
        ParameterReplacementValues::default(),
        true,
        |visitor| session.publication(visitor),
        |_| unreachable!("fixture modules contain no tensor parameters"),
        &context,
        funding,
    )
    .unwrap()
}
pub(crate) fn exchange_generation(
    session: &mut impl ControlSession,
    generation: &mut PreparedParameterPublication<FakeTensor>,
) {
    generation
        .validate(|visitor| session.publication(visitor), |a, b| Ok(a == b))
        .unwrap();
    generation.exchange(|visitor| assert!(session.publication(visitor).unwrap()));
    session.finalize();
}
pub(crate) fn publish_generation(session: &mut impl ControlSession, pool: &MemoryLedger) {
    let mut prepared = prepare_generation(session, pool);
    exchange_generation(session, &mut prepared);
}
