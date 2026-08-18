use std::{convert::Infallible, path::Path};

use safemlx_lm::{
    load_realtime_model_with_options, Completion, RealtimeBackend, RealtimeModelLoadingBackend,
    RealtimeSampling, RealtimeSpeechConfig, SemanticStateTransaction, Submission, WorkDescriptor,
};

struct MockRealtimeBackend;

#[derive(Clone)]
struct MockSession;

impl SemanticStateTransaction for MockSession {
    type Branch = Self;
    type Error = Infallible;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, _: Self::Branch) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct Frame;

impl WorkDescriptor for Frame {
    type Error = Infallible;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
        output.push(1);
        Ok(())
    }
}

struct Complete;

impl Completion for Complete {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl RealtimeBackend for MockRealtimeBackend {
    type Model = u64;
    type ModelIdentity = u64;
    type Input = Frame;
    type Output = u64;
    type Session = MockSession;
    type Completion = Complete;
    type Error = Infallible;

    fn name(&self) -> &str {
        "mock-realtime"
    }

    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity {
        *model
    }

    fn speech_config(&self, _: &Self::Model) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(2, 1, 1, 1, 0, 0, vec![0, 1]).unwrap()
    }

    fn create_session(
        &self,
        _: &Self::Model,
        _: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error> {
        Ok(MockSession)
    }

    fn validate_session(&self, _: &Self::Model, _: &Self::Session) -> Result<(), Self::Error> {
        Ok(())
    }

    fn validate_input(&self, _: &Self::Model, _: &Self::Input) -> Result<(), Self::Error> {
        Ok(())
    }

    fn input_batch_size(&self, _: &Self::Input) -> usize {
        1
    }

    fn set_sampling(&self, _: &mut Self::Session, _: RealtimeSampling) -> Result<(), Self::Error> {
        Ok(())
    }

    fn submit_step(
        &self,
        model: &mut Self::Model,
        _: &mut Self::Session,
        _: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error> {
        Ok(Submission {
            output: *model,
            completion: Complete,
        })
    }
}

impl RealtimeModelLoadingBackend for MockRealtimeBackend {
    type LoadOptions = u64;

    fn prepare_realtime_model(
        &self,
        _: &Path,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error> {
        Ok(options)
    }
}

#[test]
fn downstream_realtime_loader_selects_backend_without_mlx_types() {
    let loaded =
        load_realtime_model_with_options(MockRealtimeBackend, "mock-artifact", 23).unwrap();
    assert_eq!(loaded.backend().name(), "mock-realtime");
    assert_eq!(*loaded.model(), 23);
}
