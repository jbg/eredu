use super::*;
use crate::preparation_selection::tests::{BoundedIndependentAdapter, inspected_config};
use eredu_core::{capture::*, *};
use eredu_runtime::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Account {
    calls: Arc<AtomicUsize>,
    stop: usize,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(n <= self.stop, "producer after refusal");
        if n == self.stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    (
        HostMetadataFunding::new(Account {
            calls: calls.clone(),
            stop,
            retired: retired.clone(),
        })
        .unwrap(),
        calls,
        retired,
    )
}
fn fixture() -> (tempfile::TempDir, PreparedModelDiscovery) {
    let config = serde_json::json!({
        "model_type":"llama", "architectures":["LlamaForCausalLM"], "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":1, "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":2, "vocab_size":16,
        "rms_norm_eps":1e-5, "max_position_embeddings":32, "tie_word_embeddings":false
    });
    let args = crate::llama::model_args_from_config_value(&config).unwrap();
    let parameters = Arc::new(crate::decoder::dense_parameter_description(&args).unwrap());
    let (root, inspection) = inspected_config(config);
    let rank = ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
    let completion = CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let request = NormalizedLoadRequest::default()
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
        .with_parallel_execution(
            ParallelLoadRequest::new(
                rank,
                PipelineWireContract::new(PipelineActivationDtype::Float32),
                1,
                32,
                completion,
            )
            .unwrap(),
        )
        .unwrap();
    let selected = crate::preparation_selection::select_preparation(
        &inspection,
        &request,
        &BoundedIndependentAdapter::default(),
    )
    .unwrap();
    let plan =
        ModelPreparationPlan::from_retained_admission(inspection, selected.admission()).unwrap();
    let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
    let discovery = sources
        .prepare_discovery(
            ObservationMechanisms {
                activation_tensors: true,
                floating_to_f32: true,
                ..Default::default()
            },
            CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Preview],
                conditions: vec!["selected collector".into()],
                ..Default::default()
            },
        )
        .bind_partition_parameters(Some(parameters))
        .unwrap()
        .bind_partition_observation_hooks(Some(
            eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
                .with_publication(true),
        ))
        .unwrap();
    (root, discovery)
}

#[test]
fn selected_component_source_retains_original_account_and_refuses_every_destination() {
    let (_root, discovery) = fixture();
    let expected = discovery.component_partition_layouts(2).unwrap().unwrap();
    assert!(
        discovery.resolved_artifact_identity().is_none(),
        "layout source must not hash files"
    );
    for maximum in [2, 1] {
        let (funding, calls, retired) = account(usize::MAX);
        let result = discovery.compile_component_partition_source(
            maximum,
            None,
            CaptureSourceConstruction::new(Some(&funding)),
        );
        let count = calls.load(Ordering::SeqCst);
        assert!(count > 1);
        if maximum == 2 {
            assert_eq!(
                result.as_ref().unwrap().as_ref().unwrap().layouts(),
                &expected
            );
        } else {
            assert!(result.is_err());
        }
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(result);
        assert!(retired.load(Ordering::SeqCst));
        for stop in 1..count {
            let (funding, calls, retired) = account(stop);
            let error = discovery
                .compile_component_partition_source(
                    maximum,
                    None,
                    CaptureSourceConstruction::new(Some(&funding)),
                )
                .unwrap_err();
            assert_eq!(
                error.funding_error(),
                Some(HostMetadataFundingError::Unavailable)
            );
            assert_eq!(calls.load(Ordering::SeqCst), stop + 1);
            drop(funding);
            assert!(!retired.load(Ordering::SeqCst));
            drop(error);
            assert!(retired.load(Ordering::SeqCst));
        }
    }
}

#[test]
fn resolved_partition_catalog_uses_original_support_worker_and_every_destination() {
    let (_root, discovery) = fixture();
    let layouts = discovery.component_partition_layouts(2).unwrap().unwrap();
    // This test explicitly covers the already resolved source entry. Original
    // file hashing is a separate construction producer, never a hidden warmup.
    let expected = discovery
        .capture_with_partition_support(&layouts, |_| ObservationSupportStatus::Supported)
        .unwrap();
    assert!(discovery.resolved_artifact_identity().is_some());
    fn build(
        discovery: &PreparedModelDiscovery,
        layouts: &ComponentPartitionLayouts,
        construction: CaptureSourceConstruction<'_>,
    ) -> Result<CaptureDiscovery, CaptureDiscoverySourceError> {
        discovery.capture_with_partition_source(layouts, construction, |point, construction| {
            let members = layouts
                .capture_hook_members_source(&point.path, construction)?
                .unwrap();
            assert!(!members.is_empty());
            Ok(ObservationSupportStatus::Supported)
        })
    }
    let (funding, calls, _) = account(usize::MAX);
    let actual = build(
        &discovery,
        &layouts,
        CaptureSourceConstruction::new(Some(&funding)),
    )
    .unwrap();
    assert_eq!(actual, expected);
    let count = calls.load(Ordering::SeqCst);
    assert!(count > 20);
    drop(actual);
    for stop in 1..count {
        let (funding, calls, _) = account(stop);
        let error = build(
            &discovery,
            &layouts,
            CaptureSourceConstruction::new(Some(&funding)),
        )
        .unwrap_err();
        assert_eq!(
            error.funding_error(),
            Some(HostMetadataFundingError::Unavailable)
        );
        assert_eq!(calls.load(Ordering::SeqCst), stop + 1);
    }
}

#[test]
fn fresh_partition_source_constructs_identity_without_ordinary_discovery_warmup() {
    let (_root, discovery) = fixture();
    assert!(discovery.resolved_artifact_identity().is_none());
    let (funding, _, retired) = account(usize::MAX);
    let construction = CaptureSourceConstruction::new(Some(&funding));
    let source = discovery
        .compile_component_partition_source(2, None, construction)
        .unwrap()
        .unwrap();
    assert!(discovery.resolved_artifact_identity().is_none());
    let catalog = discovery
        .capture_with_partition_source(source.layouts(), construction, |_, _| {
            Ok(ObservationSupportStatus::Supported)
        })
        .unwrap();
    assert_eq!(
        catalog.artifact_identity,
        discovery.resolved_artifact_identity().unwrap().to_string()
    );
    assert!(source.metadata_funding().unwrap().same_account(&funding));
    let published = (catalog, source);
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    drop(published);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn capture_identity_refusal_preserves_retry_and_resolved_identity_survives_source_removal() {
    let (root, discovery) = fixture();
    let (refused, _, retired) = account(1);
    let error = discovery.prepare_capture_identity(&refused).unwrap_err();
    assert_eq!(error.funding_error(), Some(HostMetadataFundingError::Unavailable));
    assert!(discovery.resolved_artifact_identity().is_none());
    drop(refused);
    assert!(!retired.load(Ordering::SeqCst));
    drop(error);
    assert!(retired.load(Ordering::SeqCst));

    let (funding, _, retired) = account(usize::MAX);
    let identity = discovery.prepare_capture_identity(&funding).unwrap();
    assert_eq!(discovery.resolved_artifact_identity(), Some(identity));
    drop(root);
    assert_eq!(discovery.prepare_capture_identity(&funding).unwrap(), identity);
    assert_eq!(discovery.capture().unwrap().artifact_identity, identity.to_string());
    drop(funding);
    assert!(retired.load(Ordering::SeqCst), "cached digest owns no source account");
}
