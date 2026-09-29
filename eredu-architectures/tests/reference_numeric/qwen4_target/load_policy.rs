//! Generic load policy lowered before native construction, with causal history admission.
use super::*;
use eredu_architectures::qwen4_exp::prepared::TargetLoadError;
use eredu_runtime::*;

#[test]
fn qwen4_normalized_bounds_preserve_sources_and_reject_inadequate_history() {
    let (_directory, gguf, st) = super::gguf::fixtures(false);
    let policy = eredu_evaluation::qwen4_exp::bounded_policy();
    let request = NormalizedLoadRequest::default().with_bounded_execution(policy);
    for target in [&st, &gguf] {
        let before = target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        let plan = target
            .execution_plan_for_load(&request, &cold::Support)
            .unwrap();
        assert_eq!(
            plan.requirements().text().state_access(),
            ReplicatedTextStateAccess::AttentionWithStreams
        );
        let selected_request = plan.load_selection_request().unwrap().clone();
        let changed_request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
            selected_request.text().clone().with_prompt_cache(true),
            selected_request.weights(),
        )
        .unwrap();
        let capabilities = cold::capabilities(plan.requirements(), None);
        assert!(matches!(
            plan.clone().select(&changed_request, &capabilities, None),
            Err(
                eredu_architectures::qwen4_exp::prepared::TargetSelectionError::LoadRequestMismatch
            )
        ));
        plan.clone()
            .select(&selected_request, &capabilities, None)
            .unwrap();
        let streams = plan.requirements().text().append_streams();
        assert_eq!(streams.len(), 2);
        assert!(streams
            .iter()
            .all(|binding| binding.lanes == 2 && binding.limits.entries == 64));
        assert!(matches!(
            target.execution_plan_for_load(&NormalizedLoadRequest::default(), &cold::Support),
            Err(TargetLoadError::MissingBounds)
        ));
        let prediction = request
            .clone()
            .with_drafting(DraftingLoadRequest::embedded(2).unwrap());
        assert!(matches!(
            target.execution_plan_for_load(&prediction, &cold::Support),
            Err(TargetLoadError::PredictionPreparationRequired)
        ));
        let insufficient = BoundedExecutionPolicy::new(
            policy.invocation(),
            policy.selection(),
            policy.rows(),
            AppendStreamLoadPolicy::new(
                AppendStreamLimits {
                    entries: 63,
                    ..policy.append().limits()
                },
                policy.append().payload_bytes(),
                policy.append().scratch_bytes(),
                policy.append().catalog_bytes(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            target.execution_plan_for_load(
                &request.clone().with_bounded_execution(insufficient),
                &cold::Support
            ),
            Err(TargetLoadError::Budget {
                required: 64,
                limit: 63,
                ..
            })
        ));
        let different = BoundedExecutionPolicy::new(
            InvocationLimits::new(2, 32, 127).unwrap(),
            policy.selection(),
            policy.rows(),
            policy.append(),
        )
        .unwrap();
        assert!(matches!(
            target.execution_plan_for_load(
                &request.clone().with_bounded_execution(different),
                &cold::Support
            ),
            Err(TargetLoadError::RetainedLimits)
        ));
        let paged = request
            .clone()
            .with_state_residency(CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1).unwrap(),
            ));
        assert!(matches!(
            target.execution_plan_for_load(&paged, &cold::Support),
            Err(TargetLoadError::PageGeometry)
        ));
        assert_eq!(
            target
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            before
        );
    }
    assert!(matches!(
        TargetLimits::from_load_request(&request, eredu_nn::TensorElementType::I32),
        Err(TargetLoadError::StateRepresentation)
    ));
}

fn history_error(error: &(dyn std::error::Error + 'static), required: u64, limit: u64) {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(RequestError::History {
            required: actual,
            limit: bound,
        }) = error.downcast_ref::<RequestError>()
        {
            assert_eq!((*actual, *bound), (required, limit));
            return;
        }
        current = error.source();
    }
    panic!("missing typed history error: {error}");
}

#[test]
fn qwen4_history_ceiling_rejects_decode_and_transport_before_state_or_lookup_changes() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut execution = session::session(&ctx);
    for count in [32, 32, 32, 31] {
        let tokens = NumericTensor::token_ids(&vec![3usize; count]);
        execution.prefill(&tokens, None, &ctx).unwrap();
    }
    let token = NumericTensor::token_ids(&[4]);
    execution.decode(&token, &ctx).unwrap();
    let saved = execution.checkpoint(&ctx).unwrap();
    let rows = execution.execution_strategy().provider().rows.calls.len();
    let error = execution.decode(&token, &ctx).unwrap_err();
    history_error(&error, 129, 128);
    assert_eq!(
        execution.execution_strategy().provider().rows.calls.len(),
        rows
    );
    session::assert_checkpoint(&execution.checkpoint(&ctx).unwrap(), &saved);
    execution.rollback(saved.clone(), &ctx).unwrap();
    history_error(&execution.decode(&token, &ctx).unwrap_err(), 129, 128);
    session::assert_checkpoint(&execution.checkpoint(&ctx).unwrap(), &saved);

    let spec = specification();
    let architecture = model(spec.clone(), &ctx);
    let request = RequestContext::new(
        Some(OriginalTokenIds::Host(&[4])),
        None,
        1,
        1,
        128,
        configuration().vocabulary,
        spec.limits.invocation_tokens,
        configuration().attention.rotary.dimensions,
        None,
        0,
        &ctx,
    )
    .unwrap();
    let error = architecture
        .resume_request(
            NumericTensor::new([1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]),
            request.boundary(),
            128,
            &ctx,
        )
        .err()
        .expect("transport cannot exceed retained history");
    history_error(&error, 129, 128);
    let mut smaller = spec.clone();
    smaller.limits.history_tokens = 64;
    assert_ne!(smaller.geometry_fingerprint(), spec.geometry_fingerprint());
}

pub(super) fn request(residency: LayerWeightResidency) -> NormalizedLoadRequest {
    NormalizedLoadRequest::default()
        .with_bounded_execution(eredu_evaluation::qwen4_exp::bounded_policy())
        .with_weight_residency(WeightResidency::with_layers(residency))
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(false, true, true))
}

#[derive(Default)]
struct CountedRows(std::cell::Cell<usize>);
impl RowLookupMechanismSupport for CountedRows {
    fn storage(&self) -> Option<AddressableStorageCapabilities> {
        self.0.set(self.0.get() + 1);
        cold::Support.storage()
    }
    fn workspace(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        self.0.set(self.0.get() + 1);
        cold::Support.workspace(descriptor)
    }
}

fn with_history(request: &NormalizedLoadRequest, history: i32) -> NormalizedLoadRequest {
    let policy = request.bounded_execution().unwrap();
    request.clone().with_bounded_execution(
        BoundedExecutionPolicy::new(
            InvocationLimits::new(2, 32, history).unwrap(),
            policy.selection(),
            policy.rows(),
            policy.append(),
        )
        .unwrap(),
    )
}

#[test]
fn qwen4_normalized_headers_retain_request_specific_policy_without_source_access() {
    use eredu_architectures::qwen4_exp::prepared::{
        GgufTargetPlan, SafetensorsTargetPlan, TargetSelectionError,
    };
    let (directory, config, st_source) = super::safetensors_admission::fixture();
    let gguf_path = directory.path().join("target.gguf");
    let gguf = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(&gguf_path).unwrap()).unwrap();
    let gguf_source = super::super::qwen4_gguf::open_text_source(gguf.text_plan());
    let st = SafetensorsTargetPlan::prepare(
        st_source.as_ref(),
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
    )
    .unwrap();
    let physical = super::safetensors_admission::physical_sources(st_source.as_ref());
    let support = CountedRows::default();
    macro_rules! check {
        ($source:expr, $make:expr, $reopen:expr, $control_reads:expr) => {{
            let source = $source;
            let make = $make;
            let paths = source
                .source_keys()
                .iter()
                .map(|key| {
                    source
                        .source_provenance(key)
                        .unwrap()
                        .backing_shard
                        .unwrap()
                })
                .collect::<std::collections::BTreeSet<_>>();
            for path in &paths {
                std::fs::rename(path, path.with_extension("hidden")).unwrap();
            }
            let base = request(LayerWeightResidency::FullyResident);
            let queries = support.0.get();
            assert!(matches!(
                make(&NormalizedLoadRequest::default()),
                Err(TargetLoadError::MissingBounds)
            ));
            assert!(matches!(
                make(
                    &base
                        .clone()
                        .with_drafting(DraftingLoadRequest::embedded(2).unwrap())
                ),
                Err(TargetLoadError::PredictionPreparationRequired)
            ));
            assert!(matches!(
                make(&with_history(&base, 130)),
                Err(TargetLoadError::Preparation(_))
            ));
            assert!(matches!(
                make(
                    &base
                        .clone()
                        .with_state_residency(CacheResidencyPolicy::Paged(
                            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1).unwrap()
                        ))
                ),
                Err(TargetLoadError::PageGeometry)
            ));
            let policy = base.bounded_execution().unwrap();
            let insufficient_history = base.clone().with_bounded_execution(
                BoundedExecutionPolicy::new(
                    policy.invocation(),
                    policy.selection(),
                    policy.rows(),
                    AppendStreamLoadPolicy::new(
                        AppendStreamLimits {
                            entries: 63,
                            ..policy.append().limits()
                        },
                        policy.append().payload_bytes(),
                        policy.append().scratch_bytes(),
                        policy.append().catalog_bytes(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
            assert!(matches!(
                make(&insufficient_history),
                Err(TargetLoadError::Budget {
                    required: 64,
                    limit: 63,
                    ..
                })
            ));
            assert_eq!(
                support.0.get(),
                queries,
                "invalid policy fails before querying row mechanisms"
            );
            let policy = base.bounded_execution().unwrap();
            let insufficient_rows = RowLookupLoadPolicy::new(
                RowLookupLimits {
                    acquisition_bytes: 1,
                    ..policy.rows().limits()
                },
                policy.rows().bank(),
                policy.rows().retained_scalar_bytes(),
            )
            .unwrap();
            let insufficient = base.clone().with_bounded_execution(
                BoundedExecutionPolicy::new(
                    policy.invocation(),
                    policy.selection(),
                    insufficient_rows,
                    policy.append(),
                )
                .unwrap(),
            );
            let row_error = make(&insufficient)
                .err()
                .expect("row acquisition budget is enforced");
            let TargetLoadError::Preparation(
                eredu_architectures::qwen4_exp::prepared::PreparationError::Table(
                    eredu_architectures::qwen4_exp::checkpoint::NGramArtifactError::Lookup(
                        RowLookupError::Budget {
                            resource: "acquisition bytes",
                            limit: 1,
                            ..
                        },
                    ),
                ),
            ) = row_error
            else {
                panic!("unexpected row admission error: {row_error:?}");
            };
            let parallel = ParallelLoadRequest::new(
                eredu_core::ParallelRankTopology::new(
                    eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
                    0,
                )
                .unwrap(),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
                2,
                32,
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
            .unwrap();
            // Header preparation precedes joint prediction/partition construction.
            // It retains policy without prematurely selecting an ordinary executable.
            let partition_request =
                eredu_architectures::partitioned_execution::PartitionedSelectionRequest::new(
                    parallel.rank().topology(),
                    parallel.rank().global_rank(),
                    2,
                    32,
                    PipelineActivationDtype::Float32,
                )
                .unwrap()
                .with_completion_policy(parallel.completion());
            let partition = make(&base.clone().with_parallel_execution(parallel).unwrap())
                .unwrap()
                .partition(partition_request)
                .unwrap();
            assert_eq!(
                partition.requirements().topology().tensor_parallel_size(),
                2
            );
            assert_eq!(partition.requirements().topology().global_rank(), 0);
            let mut plans = Vec::new();
            for (history, batch, chunk, residency) in [
                (64, 2, 32, LayerWeightResidency::FullyResident),
                (
                    127,
                    2,
                    32,
                    LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                        eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1)
                            .unwrap(),
                    )),
                ),
                (
                    128,
                    2,
                    32,
                    LayerWeightResidency::DenseDiskStream(
                        DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
                    ),
                ),
                (128, 1, 16, LayerWeightResidency::FullyResident),
            ] {
                let request = request(residency).with_bounded_execution(
                    BoundedExecutionPolicy::new(
                        InvocationLimits::new(batch, chunk, history).unwrap(),
                        policy.selection(),
                        policy.rows(),
                        policy.append(),
                    )
                    .unwrap(),
                );
                let plan = make(&request).unwrap();
                let selected_request = plan.load_selection_request().unwrap().clone();
                assert_eq!(selected_request.weights(), request.weight_residency());
                assert_eq!(plan.requirements().text().append_streams().len(), 2);
                for stream in plan.requirements().text().append_streams() {
                    assert_eq!(stream.lanes, batch as u32);
                    assert_eq!(stream.limits.entries, 64);
                    assert_eq!(stream.limits.page_entries, 2);
                }
                let caps = cold::capabilities(plan.requirements(), None);
                let changed = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
                    selected_request.text().clone().with_prompt_cache(true),
                    selected_request.weights(),
                )
                .unwrap();
                assert!(matches!(
                    plan.clone().select(&changed, &caps, None),
                    Err(TargetSelectionError::LoadRequestMismatch)
                ));
                let selected = plan.clone().select(&selected_request, &caps, None).unwrap();
                let workspace = selected
                    .parameter_materialization_workspace(
                        &super::super::prepared_adapter::NumericPreparationProvider {
                            addressable: true,
                        },
                    )
                    .unwrap();
                assert!(workspace.ordinary_recipe_peak_bytes > 0);
                plans.push((plan, selected, request, workspace));
            }
            assert_ne!(
                plans[0].0.requirements().text().architecture_identity(),
                plans[1].0.requirements().text().architecture_identity()
            );
            let paged = request(LayerWeightResidency::FullyResident).with_state_residency(
                CacheResidencyPolicy::Paged(
                    PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
                ),
            );
            let paged_plan = make(&paged).unwrap();
            assert_eq!(
                paged_plan.load_selection_request().unwrap().text().state(),
                paged.state_residency()
            );
            assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
            for path in &paths {
                std::fs::rename(path.with_extension("hidden"), path).unwrap();
            }
            // A rename changes the store's admitted file identity. Construct the
            // readable source only after restoring the cold artifact paths.
            let source: eredu_checkpoint::store::SharedCheckpointSource = $reopen;
            for (plan, selected, request, cold_workspace) in plans {
                let queries = support.0.get();
                let before = source.source_diagnostics().unwrap().physical_reads;
                let bound = plan.clone().bind(source.clone(), None).unwrap();
                assert_eq!(bound.requirements(), plan.requirements());
                assert_eq!(
                    bound.load_selection_request(),
                    plan.load_selection_request()
                );
                let retained = bound.load_selection_request().unwrap();
                let changed = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
                    retained.text().clone().with_prompt_cache(true),
                    retained.weights(),
                )
                .unwrap();
                assert!(matches!(
                    bound.clone().select(
                        &changed,
                        &cold::capabilities(bound.requirements(), None),
                        None
                    ),
                    Err(TargetSelectionError::LoadRequestMismatch)
                ));
                let selected_bound = selected.bind(source.clone(), None).unwrap();
                assert_eq!(
                    selected_bound.realization().text().requirements(),
                    plan.requirements().text()
                );
                assert_eq!(
                    selected_bound
                        .parameter_materialization_workspace(
                            &super::super::prepared_adapter::NumericPreparationProvider {
                                addressable: true
                            },
                        )
                        .unwrap(),
                    cold_workspace
                );
                assert_eq!(
                    support.0.get(),
                    queries,
                    "binding must not query row support again"
                );
                assert_eq!(
                    source.source_diagnostics().unwrap().physical_reads,
                    before + 2 * $control_reads
                );
                assert_eq!(
                    bound.load_selection_request().unwrap().weights(),
                    request.weight_residency()
                );
            }
        }};
    }
    check!(
        st_source,
        |request: &NormalizedLoadRequest| st.execution_plan_for_load(
            request,
            eredu_nn::TensorElementType::F32,
            &support,
            physical.clone()
        ),
        Arc::new(
            eredu_checkpoint::store::SafetensorsWeightStore::open(
                directory.path().join("safetensors")
            )
            .unwrap()
        ),
        3
    );
    check!(
        gguf_source,
        |request: &NormalizedLoadRequest| gguf.execution_plan_for_load(
            request,
            eredu_nn::TensorElementType::F32,
            &support
        ),
        super::super::qwen4_gguf::open_text_source(gguf.text_plan()),
        0
    );
}
