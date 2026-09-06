//! Generic loading uses a normalized policy and a separate native target token.
//! Neither a backend-local policy facade nor a portable request forwarder is needed.

use super::*;
use eredu_core::{BackendProvider, BackendSession, ModelLoadingBackend};

#[derive(Clone)]
pub(super) struct Target {
    device: Device,
    owner: Rc<()>,
}
pub(super) struct Provider {
    target: Target,
    counts: Rc<Counts>,
    fail_publication: bool,
    realized: eredu_core::SessionCapabilities,
}
impl Provider {
    pub(super) fn new(runtime: RuntimeShape, counts: Rc<Counts>, fail_publication: bool) -> Self {
        Self {
            target: Target {
                device: Device {
                    runtime,
                    ordinal: 0,
                },
                owner: Rc::new(()),
            },
            counts,
            fail_publication,
            realized: eredu_core::SessionCapabilities::new(true, true, false),
        }
    }
    pub(super) fn target(&self) -> Target {
        self.target.clone()
    }
    fn validate_target(&self, target: &Target) -> Result<(), Error> {
        if self.target.device != target.device || !Rc::ptr_eq(&self.target.owner, &target.owner) {
            return Err(Error::backend(
                "native target belongs to another scalar client",
            ));
        }
        Ok(())
    }
    fn validate_client(&self, client: &Client) -> Result<(), Error> {
        if client.device != self.target.device || !Rc::ptr_eq(&client.owner, &self.target.owner) {
            return Err(Error::backend("session belongs to another scalar client"));
        }
        Ok(())
    }
}

pub(super) struct Selected {
    selection: eredu_architectures::SelectedPreparation,
    target: Target,
}
pub(super) struct Configuration {
    sources: eredu_architectures::prepared_sources::PreparedModelSources,
    target: Target,
}

impl ModelLoadingBackend for Provider {
    type LoadOptions = (eredu_runtime::NormalizedLoadRequest, Target);
    type SelectedPreparation = Selected;
    type ConfigurationResolver = eredu_architectures::configuration::ModelConfigurations;

    fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
        &eredu_architectures::configuration::MODEL_CONFIGURATIONS
    }
    fn select_preparation(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        (request, target): &Self::LoadOptions,
    ) -> Result<Selected, Error> {
        self.validate_target(target)?;
        increment(&self.counts.selection);
        let support = Support {
            runtime: target.device.runtime,
            counts: Rc::clone(&self.counts),
        };
        let selection = eredu_architectures::select_preparation(inspection, request, &support)
            .map_err(Error::backend)?;
        Ok(Selected {
            selection,
            target: target.clone(),
        })
    }
    fn selected_preparation_admission(
        &self,
        selected: &Selected,
    ) -> eredu_core::PreparationAdmission {
        selected.selection.admission()
    }
    fn model_config(
        &self,
        selected: eredu_core::SelectedModelPreparation<Self>,
    ) -> Result<Configuration, Error> {
        let (plan, Selected { selection, target }) = selected.into_parts();
        self.validate_target(&target)?;
        increment(&self.counts.sources);
        let sources = eredu_architectures::prepared_sources::prepare_model_sources(plan, selection)
            .map_err(Error::backend)?;
        Ok(Configuration { sources, target })
    }
}

impl BackendProvider for Provider {
    type ModelConfig = Configuration;
    type Model = Session;
    type Session = Session;
    type Error = Error;

    fn descriptor(&self) -> eredu_core::BackendDescriptor {
        eredu_core::BackendDescriptor::new("scalar-client-proof", "test")
    }
    fn devices(
        &self,
    ) -> Result<Vec<(eredu_core::DeviceDescriptor, eredu_core::DeviceCapabilities)>, Error> {
        Ok(vec![(
            eredu_core::DeviceDescriptor::new(
                "scalar:0",
                "Synthetic scalar client",
                "scalar",
                None,
            ),
            eredu_core::DeviceCapabilities::new(true, false, false),
        )])
    }
    fn prepare_model(
        &self,
        config: Configuration,
    ) -> Result<eredu_core::PreparedModel<Session>, Error> {
        self.validate_target(&config.target)?;
        let capabilities = config.sources.selected().admission().session_capabilities();
        let mut session = materialize(
            config.sources,
            config.target.device,
            Rc::clone(&config.target.owner),
            Rc::clone(&self.counts),
            self.fail_publication,
        )
        .map_err(Error::backend)?;
        session.reported_capabilities = self.realized;
        Ok(eredu_core::PreparedModel::new(session, capabilities))
    }
    fn create_session(
        &self,
        prepared: eredu_core::PreparedModel<Session>,
    ) -> Result<Session, Error> {
        let session = prepared.into_inner();
        self.validate_client(&session.client)?;
        Ok(session)
    }
    fn session_capability_mismatch(
        &self,
        admitted: eredu_core::SessionCapabilities,
        realized: eredu_core::SessionCapabilities,
    ) -> Error {
        Error::backend(format!(
            "exact scalar session mismatch: admitted {admitted:?}, realized {realized:?}"
        ))
    }
}

impl BackendSession<Provider> for Session {
    type PrefillInput = NumericTensor;
    type DecodeInput = NumericTensor;
    type Output = NumericTensor;
    type Completion = NativeCompletion;
    fn capabilities(&self) -> eredu_core::SessionCapabilities {
        self.reported_capabilities
    }
    fn prefill(
        &mut self,
        provider: &Provider,
        input: NumericTensor,
    ) -> Result<eredu_core::Submission<NumericTensor, NativeCompletion>, Error> {
        provider.validate_client(&self.client)?;
        self.submit(&input, true).map_err(Error::backend)
    }
    fn decode(
        &mut self,
        provider: &Provider,
        input: NumericTensor,
    ) -> Result<eredu_core::Submission<NumericTensor, NativeCompletion>, Error> {
        provider.validate_client(&self.client)?;
        self.submit(&input, false).map_err(Error::backend)
    }
    fn observe_output(
        &self,
        provider: &Provider,
        output: &NumericTensor,
    ) -> Result<eredu_core::ObservationSet, Error> {
        provider.validate_client(&self.client)?;
        self.authority.require_idle().map_err(Error::backend)?;
        let value = eredu_core::TensorObservation::new(
            output.shape.iter().map(|n| *n as usize).collect(),
            eredu_core::TensorObservationData::F32(output.data.clone()),
        )
        .map_err(Error::backend)?;
        let mut observations = eredu_core::ObservationSet::new();
        observations
            .insert(
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                eredu_core::ObservationValue::Tensor(value),
            )
            .map_err(Error::backend)?;
        Ok(observations)
    }
}

#[test]
fn generic_provider_executes_and_observes_exact_materialized_values() {
    let (root, expected_parameters) = prepared_adapter::payload_fixture(1.0);
    let counts = Rc::new(Counts::default());
    let provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        &prepared_adapter::plan(None),
        eredu_runtime::ResidencyDiagnostics::default(),
        None,
    )
    .unwrap();
    reset_reference_stage_evidence("SafeTensors");
    let prepared =
        eredu_core::load_model(&provider, root.path(), (request, provider.target())).unwrap();
    let mut runtime = eredu_core::ModelRuntime::from_prepared(provider, prepared).unwrap();
    let submission = runtime
        .prefill(NumericTensor::token_ids(&[1, 3, 2]))
        .unwrap();
    assert!(runtime.decode(NumericTensor::token_ids(&[4])).is_err());
    assert!(runtime.observe_output(&submission.output).is_err());
    submission.completion.wait().unwrap();
    let observations = runtime.observe_output(&submission.output).unwrap();
    let observed = match observations
        .get(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        .unwrap()
    {
        eredu_core::ObservationValue::Tensor(value) => value,
        _ => panic!("logits must be tensor values"),
    };
    assert_eq!(
        observed.data(),
        &eredu_core::TensorObservationData::F32(submission.output.data.clone())
    );
    assert_eq!(
        last_reference_stage_evidence().bound_parameters,
        expected_parameters
    );
    assert_eq!(
        (
            counts.selection.get(),
            counts.sources.get(),
            counts.native.get(),
            counts.construction.get(),
            counts.publication.get()
        ),
        (1, 1, 1, 1, 1)
    );
}

#[test]
fn generic_provider_rejects_foreign_native_target_before_selection_or_payloads() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let counts = Rc::new(Counts::default());
    let provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    let foreign = Provider::new(RuntimeShape::Cpu, Rc::new(Counts::default()), false);
    let request = eredu_runtime::NormalizedLoadRequest::default();
    reset_reference_stage_evidence("SafeTensors");
    assert!(eredu_core::load_model(&provider, root.path(), (request, foreign.target())).is_err());
    assert_eq!(
        (
            counts.selection.get(),
            counts.sources.get(),
            counts.native.get()
        ),
        (0, 0, 0)
    );
    assert!(last_reference_stage_evidence().payload_reads.is_empty());
}

#[test]
fn generic_provider_rejects_foreign_prepared_client_even_with_identical_device_and_counters() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let counts = Rc::new(Counts::default());
    let provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    let foreign = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    let prepared = eredu_core::load_model(
        &provider,
        root.path(),
        (
            eredu_runtime::NormalizedLoadRequest::default(),
            provider.target(),
        ),
    )
    .unwrap();
    assert!(foreign.create_session(prepared).is_err());
    assert_eq!(counts.submissions.get(), 0);
    assert_eq!(counts.client_drops.get(), 1);
}

#[test]
fn generic_publication_rejects_capability_mismatch_and_releases_client() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let counts = Rc::new(Counts::default());
    let mut provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    provider.realized = eredu_core::SessionCapabilities::new(true, false, false);
    let prepared = eredu_core::load_model(
        &provider,
        root.path(),
        (
            eredu_runtime::NormalizedLoadRequest::default(),
            provider.target(),
        ),
    )
    .unwrap();
    let result = eredu_core::ModelRuntime::from_prepared(provider, prepared);
    assert!(
        matches!(result, Err(error) if error.to_string().contains("exact scalar session mismatch"))
    );
    assert_eq!(counts.submissions.get(), 0);
    assert_eq!(counts.client_drops.get(), 1);
}

#[test]
fn generic_session_rejects_foreign_provider_before_submission_or_observation() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let counts = Rc::new(Counts::default());
    let provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
    let prepared = eredu_core::load_model(
        &provider,
        root.path(),
        (
            eredu_runtime::NormalizedLoadRequest::default(),
            provider.target(),
        ),
    )
    .unwrap();
    let mut session = provider.create_session(prepared).unwrap();
    let before = session.body.borrow().state().unwrap();
    for runtime in [RuntimeShape::Cpu, RuntimeShape::Wgpu] {
        let foreign = Provider::new(runtime, Rc::clone(&counts), false);
        assert!(session
            .prefill(&foreign, NumericTensor::token_ids(&[1]))
            .is_err());
        assert!(session
            .decode(&foreign, NumericTensor::token_ids(&[2]))
            .is_err());
        assert!(session
            .observe_output(&foreign, &NumericTensor::token_ids(&[0]))
            .is_err());
        assert!(session.authority.require_idle().is_ok());
        assert_state_exact(
            &session.body.borrow().state().unwrap(),
            &before,
            before.layout().len(),
            "foreign provider must not mutate session",
        );
        assert_eq!(counts.submissions.get(), 0);
    }
    let submission = session
        .prefill(&provider, NumericTensor::token_ids(&[1]))
        .unwrap();
    submission.completion.wait().unwrap();
    assert!(session
        .observe_output(&provider, &submission.output)
        .is_ok());
    assert_eq!(counts.submissions.get(), 1);
}

#[test]
fn generic_runtime_revalidates_original_admission_after_mutable_session_access() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    for through_parts in [false, true] {
        let counts = Rc::new(Counts::default());
        let provider = Provider::new(RuntimeShape::Cpu, Rc::clone(&counts), false);
        let prepared = eredu_core::load_model(
            &provider,
            root.path(),
            (
                eredu_runtime::NormalizedLoadRequest::default(),
                provider.target(),
            ),
        )
        .unwrap();
        let mut runtime = eredu_core::ModelRuntime::from_prepared(provider, prepared).unwrap();
        let session = if through_parts {
            runtime.parts_mut().1
        } else {
            runtime.session_mut()
        };
        session.reported_capabilities = eredu_core::SessionCapabilities::new(true, false, false);
        assert!(runtime.prefill(NumericTensor::token_ids(&[1])).is_err());
        assert!(runtime.decode(NumericTensor::token_ids(&[2])).is_err());
        assert!(runtime
            .observe_output(&NumericTensor::token_ids(&[0]))
            .is_err());
        assert_eq!(counts.submissions.get(), 0);
        assert!(runtime.session().authority.require_idle().is_ok());
    }
}
