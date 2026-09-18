use super::super::super::*;
use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{TextGenerationDriver, TextGenerationInput, TokenFilter, TokenFilterController};
use safemlx::{Device, DeviceType, Stream};

struct All;
impl TokenFilterController for All {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(3),
                repetition_penalty: Some(1.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(1),
        managed_memory_capacity_bytes: Some(u64::MAX),
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    })
}
fn evidence() -> TextPreparationInput<'static, MlxModelInput> {
    TextPreparationInput::TokenIds {
        positions: 2,
        capacity_bytes: 8,
    }
}
fn runtime(pool: &WorkingMemoryPool) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &source).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    (runtime, artifact)
}
fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}
fn settle(pool: &WorkingMemoryPool) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == 0 && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn cause<'a>(mut error: &'a (dyn std::error::Error + 'static)) -> Option<&'a WorkingMemoryError> {
    loop {
        if let Some(found) = error.downcast_ref::<WorkingMemoryError>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

#[test]
fn ordinary_quote_seals_immediately_and_private_pending_quote_rejects_work() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    // Both actual models load before finite request preparation.
    let (runtime, artifact) = runtime(&pool);
    let (foreign, foreign_artifact) = self::runtime(&pool);
    let (preparation, mut quote) =
        admit_inner(&runtime, &evidence(), config(), &All, None, false, None, None)
            .map_err(sequence::AdmissionFailure::into_backend)
            .unwrap();
    let retained = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap();
    quote.validate(&runtime, quote.request()).unwrap();
    quote.validate_opening(&retained).unwrap();
    assert!(quote.predecessor().unwrap().is_none());
    let before = (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        paths::snapshot(),
    );
    assert_eq!(
        cause(&quote.validate(&foreign, quote.request()).unwrap_err()),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        cause(&quote.seal_installed_opening(&runtime).unwrap_err()),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    // Only this child test replaces the private opening owner. Production has
    // no constructor of an unsealed quote and no enabled resume adapter.
    quote.unique_for_test().unwrap().opening = OpeningSeal::pending(&retained);
    assert_eq!(
        cause(&quote.validate(&runtime, quote.request()).unwrap_err()),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        cause(&quote.validate_opening(&retained).unwrap_err()),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        cause(&quote.predecessor().unwrap_err()),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        cause(&quote.activate_disk_route().unwrap_err()),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    // Metadata queries remain usable without pretending that the draft can run.
    assert_eq!(quote.config(), config());
    assert_eq!(quote.local_prediction(0).unwrap(), 0);
    assert_eq!(quote.prediction_frontier(0).unwrap(), 0);
    let _ = (quote.contract(), quote.storage_contract(), quote.request());
    assert_eq!(
        cause(&quote.seal_installed_opening(&foreign).unwrap_err()),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    let lease = runtime
        .session()
        .authority
        .borrow_mut()
        .begin_submission()
        .unwrap();
    let busy = quote.seal_installed_opening(&runtime).unwrap_err();
    assert!(matches!(&busy, Error::Other(source)
        if source.downcast_ref::<eredu_core::SessionAuthorityError>() == Some(&eredu_core::SessionAuthorityError::Busy)));
    // No native work was submitted under this test's exclusion-only lease.
    assert!(lease.resolve());
    drop(lease);
    // An unchanged empty state is not evidence of completed installation.
    assert_eq!(
        cause(&quote.seal_installed_opening(&runtime).unwrap_err()),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        (
            pool.used_bytes().unwrap(),
            pool.peak_bytes().unwrap(),
            paths::snapshot()
        ),
        before
    );
    drop((
        quote,
        preparation,
        retained,
        runtime,
        artifact,
        foreign,
        foreign_artifact,
    ));
    settle(&pool);
}

#[test]
fn ordinary_successor_keeps_exact_predecessor_and_rejects_a_later_revision() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2, 3, 4, 5]),
            config(),
            All,
        )
        .unwrap();
    let mut outputs = Vec::new();
    for _ in 0..2 {
        outputs.push(driver.advance(&mut state).unwrap().unwrap().into_output());
        driver.take_completed_delivery(&mut state).unwrap();
    }
    let (preparation, quote, original) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, _, _) = boundary.copy_mechanism_parts();
        let retained = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap();
        assert_eq!(retained.admission().unwrap().position(), 6);
        let (preparation, quote) =
            admit_inner(runtime, &evidence(), config(), &All, None, false, None, None)
                .map_err(sequence::AdmissionFailure::into_backend)
                .unwrap();
        quote.validate_opening(&retained).unwrap();
        quote.validate_opening(&retained.clone()).unwrap();
        quote
            .predecessor()
            .unwrap()
            .unwrap()
            .validate_same_request(retained.admission().unwrap().request())
            .unwrap();
        assert_eq!(
            quote.opening.validate(&retained, 7),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        let wrong =
            OpeningSeal::ordinary(retained.revision().clone(), Some(quote.request().clone()));
        assert_eq!(
            wrong.validate(&retained, 6),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        (preparation, quote, retained)
    };
    outputs.push(driver.advance(&mut state).unwrap().unwrap().into_output());
    driver.take_completed_delivery(&mut state).unwrap();
    {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, _, _) = boundary.copy_mechanism_parts();
        let current = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap();
        assert_eq!(current.admission().unwrap().position(), 7);
        assert_ne!(current.revision(), original.revision());
        assert_eq!(
            cause(&quote.validate_opening(&current).unwrap_err()),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(
            cause(&quote.seal_installed_opening(runtime).unwrap_err()),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        quote.validate_opening(&original).unwrap();
    }
    drop((quote, preparation, original, outputs, state));
    drop(driver);
    drop((runtime, artifact));
    settle(&pool);
}
