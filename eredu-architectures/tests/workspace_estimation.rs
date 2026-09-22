//! Cold equation traces use the same modules as numerical conformance. The
//! deliberately specified allocation mechanism here is independent of MLX.
#[path = "../src/memory_fixture.rs"]
#[allow(dead_code)]
mod memory_fixture;

use eredu_architectures::{decoder, llama, qwen, readout::execute_readout};
use eredu_core::{
    AdmissionRequest, CacheStateStrategy, CapabilityError, EstimationCompleteness,
    ExecutionWorkspaceEstimate, InferenceGeometry, InputModalities, InputTokenCount,
    ModelCapabilities, Observed, OutputDemand, RuntimeStateEstimate, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_nn::{
    workspace::*, AttentionCache, AttentionRequest, EmbeddingOperator, Error, LinearOperator,
    NeuralBackend, NormalizationOperator, Tensor,
};
use eredu_runtime::working_memory::*;

#[path = "workspace_estimation/blockwise.rs"]
mod blockwise;
#[path = "workspace_estimation/compressed.rs"]
mod compressed;
#[path = "workspace_estimation/grouped.rs"]
mod grouped;
#[path = "workspace_estimation/hybrid.rs"]
mod hybrid;
#[path = "workspace_estimation/parallel.rs"]
mod parallel;
#[path = "workspace_estimation/prediction.rs"]
mod prediction;

#[derive(Debug)]
struct EquationMechanism {
    omit_attention: bool,
}
impl WorkspaceMechanisms for EquationMechanism {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory_fixture::placement())
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        // This independent fixture selects F32 storage for every floating
        // result. Metadata views alias it and need not be row-contiguous.
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32).then_some(
            WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                !matches!(
                    operation.kind,
                    WorkspaceOperationKindView::View(_)
                        | WorkspaceOperationKindView::Transpose(_)
                        | WorkspaceOperationKindView::Index { .. }
                        | WorkspaceOperationKindView::StaticSlice { .. }
                ),
            ),
        )
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "test mechanism has no disjoint host workspace".into(),
        }))
    }

    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.omit_attention && matches!(op.kind, WorkspaceOperationKind::Attention { .. }) {
            return Ok(None);
        }
        let aliases = matches!(
            op.kind,
            WorkspaceOperationKind::View(_)
                | WorkspaceOperationKind::Transpose(_)
                | WorkspaceOperationKind::Index { .. }
                | WorkspaceOperationKind::StaticSlice { .. }
        );
        let scratch_bytes = if matches!(op.kind, WorkspaceOperationKind::Attention { .. }) {
            let q = op.inputs[0].shape();
            let k = op.inputs[1].shape();
            // This test mechanism explicitly materializes one float32 QK matrix.
            q[0] as u64 * q[1] as u64 * q[2] as u64 * k[2] as u64 * 4
        } else {
            0
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: op.outputs.iter().map(|layout| if aliases { Ok(WorkspaceOutputStorage::AliasInput(0)) } else { layout.bytes().map(WorkspaceOutputStorage::Allocate) }).collect::<Result<_,_>>()?,
            scratch_bytes,
            assumptions: "test mechanism: F32 output allocations, aliasing metadata views, one materialized QK attention matrix".into(),
        }))
    }
}

#[derive(Default)]
struct AppendCache {
    keys: Option<WorkspaceTensor>,
    values: Option<WorkspaceTensor>,
    offset: i32,
}
impl AttentionCache<WorkspaceTensor> for AppendCache {
    fn offset(&self) -> i32 {
        self.offset
    }
    fn max_size(&self) -> Option<i32> {
        None
    }
    fn update_for_attention(
        &mut self,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(WorkspaceTensor, WorkspaceTensor), Error> {
        self.offset += keys.shape()[2];
        let keys = if let Some(old) = self.keys.take() {
            WorkspaceTensor::concatenate(&[old, keys], 2, context)?
        } else {
            keys
        };
        let values = if let Some(old) = self.values.take() {
            WorkspaceTensor::concatenate(&[old, values], 2, context)?
        } else {
            values
        };
        self.keys = Some(keys.clone());
        self.values = Some(values.clone());
        Ok((keys, values))
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        WorkspaceBackend::attention_with_sinks(request, context)
    }
}

fn config(family: &str, tied: bool, packed: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type":family,"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
        "vocab_size":37,"max_position_embeddings":128,"tie_word_embeddings":tied
    });
    if packed {
        config["quantization"] = serde_json::json!({"group_size":32,"bits":4});
    }
    config
}
fn geometry(chunk: u64, output: OutputDemand) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 2,
        cached_positions: 2,
        input_positions: 7,
        max_output_tokens: 3,
        prefill_chunk_positions: chunk,
        output,
    }
}
fn request(g: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
    }
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "equation trace fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::PersistentStateOnly,
    }
}

fn estimate<C: decoder::Config>(
    args: &C,
    g: InferenceGeometry,
    omit_attention: bool,
) -> Result<(RuntimeStateEstimate, Vec<WorkspaceTraceReport>), CapabilityError> {
    let run = || -> Result<Vec<WorkspaceTraceReport>, Error> {
        let context = WorkspaceContext::new(EquationMechanism { omit_attention });
        let mut pinned = decoder::StaticModules::<WorkspaceBackend>::new(args, &context)?;
        let mut blocks = (0..args.num_hidden_layers())
            .map(|index| {
                decoder::TransformerBlock::<WorkspaceBackend>::new(args, index as usize, &context)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut caches = (0..blocks.len())
            .map(|_| {
                let shape = [g.batch_size as i32, 2, g.cached_positions as i32, 8];
                let existing = || {
                    let layout = WorkspaceLayout::new(&shape, WorkspaceDtype::Float32)?;
                    let backing = WorkspaceExistingStorage::try_new_placed(
                        Some(layout.bytes()?),
                        crate::memory_fixture::placement(),
                        &context,
                    )?;
                    WorkspaceTensor::existing_with_storage(layout, &backing, &context)
                };
                Ok(AppendCache {
                    keys: Some(existing()?),
                    values: Some(existing()?),
                    offset: g.cached_positions as i32,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut spans = Vec::new();
        let mut position = 0;
        while position < g.input_positions {
            let count = g.prefill_chunk_positions.min(g.input_positions - position);
            spans.push((
                count,
                g.output.for_chunk(position + count == g.input_positions),
            ));
            position += count;
        }
        spans.extend((0..g.max_output_tokens).map(|_| (1, OutputDemand::LastPosition)));
        let mut reports = Vec::new();
        for (count, demand) in spans {
            context.begin_state_span(
                caches
                    .iter()
                    .flat_map(|cache| cache.keys.iter().chain(cache.values.iter())),
            )?;
            let ids = (0..g.batch_size * count)
                .map(|n| (3 + n * 5) % 37)
                .map(|n| n as i32)
                .collect::<Vec<_>>();
            let tokens = WorkspaceTensor::from_i32_slice(
                &ids,
                &[g.batch_size as i32, count as i32],
                &context,
            )?;
            let mask =
                WorkspaceBackend::causal_mask(count as i32, caches[0].offset, None, &context)?;
            let mut hidden = pinned.embeddings.forward(&tokens, &context)?;
            for (block, cache) in blocks.iter_mut().zip(&mut caches) {
                hidden = block.forward(
                    decoder::AttentionInput {
                        hidden: &hidden,
                        mask: Some(&mask),
                        cache: Some(cache),
                        allow_sliding_prefill: false,
                        rotary_position: None,
                    },
                    &context,
                )?;
            }
            let _scores = execute_readout(&hidden, demand, 1, &context, |hidden| {
                let normalized = pinned.norm.forward(hidden, &context)?;
                match &mut pinned.lm_head {
                    Some(head) => head.forward(&normalized, &context),
                    None => pinned.embeddings.as_linear(&normalized, &context),
                }
            })?;
            let roots = caches
                .iter()
                .flat_map(|cache| {
                    [
                        cache.keys.as_ref().unwrap().clone(),
                        cache.values.as_ref().unwrap().clone(),
                    ]
                })
                .collect::<Vec<_>>();
            reports.push(context.report(&roots)?);
        }
        Ok(reports)
    };
    let reports = run().map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let worst = reports
        .iter()
        .find(|report| report.inference_transient_bytes().is_none())
        .unwrap_or_else(|| {
            reports
                .iter()
                .max_by_key(|report| report.inference_transient_bytes())
                .unwrap()
        });
    let layout = StateMemoryLayout::new(
        decoder::cache_layout(args).unwrap(),
        vec![0; args.num_hidden_layers() as usize],
        args.hidden_size() as u64,
        1,
        EstimationCompleteness::Complete,
    )?;
    let state = eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        std::num::NonZeroU8::new(4).unwrap(),
    )?;
    let zero = || {
        WorkspaceBound::bounded(0,"test execution has no additional untraced operations; all tensor outputs and scratch are in the equation trace")
    };
    let outside = ExecutionWorkspaceEstimate {
        physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
            geometry: g,
            activations: crate::memory_fixture::requirements(0),
            attention: crate::memory_fixture::requirements(0),
            vocabulary: crate::memory_fixture::requirements(0),
            state_update: crate::memory_fixture::requirements(0),
            materialization: crate::memory_fixture::requirements(0),
            retained: crate::memory_fixture::requirements(0),
        }),
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    };
    Ok((with_equation_workspace(state, outside, worst)?, reports))
}

fn check_equations<C: decoder::Config>(args: &C) {
    for demand in [
        OutputDemand::StateOnly,
        OutputDemand::LastPosition,
        OutputDemand::Sequence,
    ] {
        for chunk in [1, 2, 3, 7] {
            let (state, reports) = estimate(args, geometry(chunk, demand), false).unwrap();
            assert!(
                state
                    .execution_workspace
                    .unwrap()
                    .peak_bytes()
                    .unwrap()
                    .unwrap()
                    > 0
            );
            let prompt_chunks = 7_u64.div_ceil(chunk) as usize;
            assert_eq!(reports.len(), prompt_chunks + 3);
            for (index, report) in reports.iter().enumerate() {
                let projections = report
                    .operations
                    .iter()
                    .filter(|op| {
                        matches!(op.kind, WorkspaceOperationKind::Projection(_))
                            && op.outputs[0].shape().last() == Some(&37)
                    })
                    .collect::<Vec<_>>();
                let expected = if index >= prompt_chunks {
                    OutputDemand::LastPosition
                } else {
                    demand.for_chunk(index + 1 == prompt_chunks)
                };
                assert_eq!(
                    projections.len(),
                    usize::from(expected != OutputDemand::StateOnly)
                );
                if let Some(projection) = projections.first() {
                    let positions = if index >= prompt_chunks {
                        1
                    } else {
                        chunk.min(7 - index as u64 * chunk)
                    };
                    assert_eq!(
                        projection.inputs[0].shape()[1] as u64,
                        expected.positions(positions)
                    );
                }
            }
        }
    }
}

#[test]
fn dense_and_packed_family_equations_trace_prefill_readout_and_three_cached_decodes() {
    for tied in [false, true] {
        for packed in [false, true] {
            check_equations(
                &llama::model_args_from_config_value(&config("llama", tied, packed)).unwrap(),
            );
            for family in ["qwen2", "qwen3"] {
                check_equations(
                    &qwen::model_args_from_config_value(&config(family, tied, packed)).unwrap(),
                );
            }
        }
    }
}

#[test]
fn equation_quotes_select_affordable_chunks_and_reserve_against_competing_admissions() {
    let args = llama::model_args_from_config_value(&config("llama", true, false)).unwrap();
    let g = geometry(7, OutputDemand::LastPosition);
    let smaller = estimate(&args, geometry(2, g.output), false).unwrap().0;
    let capacity = smaller.requested_state_bytes
        + smaller
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap();
    let full = estimate(&args, g, false).unwrap().0;
    assert!(
        capacity
            < full.requested_state_bytes
                + full
                    .execution_workspace
                    .as_ref()
                    .unwrap()
                    .peak_bytes()
                    .unwrap()
                    .unwrap()
    );
    let quotation = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let smaller_admission = eredu_core::Admission {
        requested_positions: smaller.assumptions.requested_positions,
        state: smaller,
        incremental_required_bytes: Some(capacity),
        memory_limits: Default::default(),
        additional_headroom: Default::default(),
    };
    let capacity = quotation
        .reservation_requirements(&smaller_admission, None)
        .unwrap()
        .get(quotation.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let pool = crate::memory_fixture::ledger(capacity, 0).unwrap();
    let identity = InferenceExecutionIdentity::default();
    let (admission, reservation) =
        plan_prefill(&identity, &pool, &capabilities(), request(g), g, |g| {
            estimate(&args, g, false).map(|result| result.0)
        })
        .unwrap();
    assert!(
        admission
            .state
            .execution_workspace
            .unwrap()
            .geometry
            .prefill_chunk_positions
            <= 2
    );
    assert!(crate::memory_fixture::used(&pool).unwrap() > 0);
    assert!(
        plan_prefill(&identity, &pool, &capabilities(), request(g), g, |g| {
            estimate(&args, g, false).map(|result| result.0)
        })
        .is_err()
    );
    drop(reservation);
    assert_eq!(crate::memory_fixture::used(&pool).unwrap(), 0);
}

#[test]
fn one_missing_native_primitive_prevents_strict_admission_at_every_chunk_size() {
    let args = llama::model_args_from_config_value(&config("llama", true, false)).unwrap();
    let g = geometry(7, OutputDemand::LastPosition);
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let mut attempts = Vec::new();
    let result = plan_prefill(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(g),
        g,
        |g| {
            attempts.push(g.prefill_chunk_positions);
            estimate(&args, g, true).map(|result| result.0)
        },
    );
    assert!(matches!(
        result,
        Err(PrefillPlanningError::Admission(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    assert_eq!(attempts, [7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(crate::memory_fixture::used(&pool).unwrap(), 0);
}

#[derive(Default)]
struct RecurrentState {
    convolution: Option<WorkspaceTensor>,
    recurrent: Option<WorkspaceTensor>,
    position: i32,
}
impl eredu_runtime::RuntimeLayerState<WorkspaceBackend> for RecurrentState {
    type RetainedValues<'a> =
        std::iter::Flatten<std::array::IntoIter<Option<&'a WorkspaceTensor>, 2>>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        [self.convolution.as_ref(), self.recurrent.as_ref()]
            .into_iter()
            .flatten()
    }
}
impl eredu_runtime::RuntimeStateComponents<WorkspaceBackend> for RecurrentState {
    fn position(&self) -> i32 {
        self.position
    }
    fn fixed_component(
        &mut self,
        role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<WorkspaceTensor>, eredu_runtime::StateError> {
        match role {
            eredu_core::cache::StateTensorRole::Convolution { slot: 0 } => {
                Ok(&mut self.convolution)
            }
            eredu_core::cache::StateTensorRole::Recurrent => Ok(&mut self.recurrent),
            _ => Err(eredu_runtime::StateError::InvalidResidency(
                "undeclared test state role".into(),
            )),
        }
    }
    fn advance_fixed(&mut self, tokens: i32) -> Result<(), eredu_runtime::StateError> {
        self.position += tokens;
        Ok(())
    }
}

#[test]
fn hybrid_recurrent_equations_trace_convolution_and_asymmetric_persistent_matrices() {
    use eredu_runtime::RuntimeLayerState;
    for family in ["qwen3_5_text", "qwen3_next"] {
        let args = qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":family,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"max_position_embeddings":128,"intermediate_size":64,
            "vocab_size":37,"tie_word_embeddings":true,"num_experts":0,
            "layer_types":["linear_attention","full_attention"],
            "linear_conv_kernel_dim":3,"linear_key_head_dim":4,"linear_value_head_dim":8,
            "linear_num_key_heads":2,"linear_num_value_heads":4
        })).unwrap().text;
        let context = WorkspaceContext::new(EquationMechanism {
            omit_attention: false,
        });
        let mut mixer =
            qwen::hybrid::LinearAttention::<WorkspaceBackend>::new(&args, 0, &context).unwrap();
        let mut state = RecurrentState::default();
        for sequence in [3, 2, 1, 1, 1] {
            context
                .begin_state_span(eredu_runtime::RuntimeLayerState::retained_values(&state))
                .unwrap();
            let hidden = WorkspaceTensor::full_f32(0.125, &[2, sequence, 32], &context).unwrap();
            let output = mixer.forward(&hidden, &mut state, &context).unwrap();
            assert_eq!(output.shape(), [2, sequence, 32]);
            assert_eq!(state.recurrent.as_ref().unwrap().shape(), [2, 4, 4, 8]);
            assert_eq!(state.convolution.as_ref().unwrap().shape(), [2, 2, 48]);
            let report = context
                .report(&state.retained_values().cloned().collect::<Vec<_>>())
                .unwrap();
            assert!(report.transient_bytes.unwrap() > 0);
            assert!(report.retained_bytes.unwrap() >= 2 * 4 * 4 * 8 * 4 + 2 * 2 * 48 * 4);
            assert_eq!(
                report
                    .operations
                    .iter()
                    .filter(|op| matches!(op.kind, WorkspaceOperationKind::GatedDeltaScan))
                    .count(),
                1
            );
            assert!(report
                .operations
                .iter()
                .any(|op| matches!(op.kind, WorkspaceOperationKind::Convolution { .. })));
        }
        assert_eq!(state.position, 8);
    }
}
