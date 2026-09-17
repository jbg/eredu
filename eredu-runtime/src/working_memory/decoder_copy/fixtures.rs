use super::*;
use crate::working_memory::{
    InferenceRequest, InferenceTextPreparation, WorkingMemoryFundingRun, WorkingMemoryReservation,
};
use crate::{PenaltyConfig, SamplingBackend, TokenDomain};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    StateMemoryLayout, TextGenerationConfig, TokenFilter, WorkspaceBound,
};
use std::cell::Cell;

const HOST_SOURCE_BYTES: u64 = 1024;

fn config(outputs: usize, adaptive: bool) -> TextGenerationConfig {
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: true,
        temperature: 0.7,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(outputs),
    });
    if adaptive {
        config.with_mirostat_v2(3.7, 0.23).unwrap()
    } else {
        config
    }
}

// A scalar portable sampling backend has no native arrays. Its real boxed
// history is constructed by the one-time stage and grown by the shared sampler.
// The generous source envelope covers this fixture's complete host work; it is
// not a synthetic assertion about native model workspace or snapshot support.
fn prepared_request(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    prepared_request_with_bytes(pool, capacity, outputs, adaptive, HOST_SOURCE_BYTES)
}

fn prepared_request_with_bytes(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
    source_bytes: u64,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: outputs as u64,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        outputs as u64,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable scalar sampler host envelope");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(source_bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation: WorkingMemoryReservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1 + outputs as u64,
                state,
                incremental_required_bytes: source_bytes,
                available_memory_bytes: None,
            },
            capacity,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
    let config = config(outputs, adaptive);
    let preparation = request.prepare_text(&execution, geometry, config).unwrap();
    (preparation, run, config)
}

pub(super) fn source(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    RunOwnedTextSampler,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let (preparation, run, config) = prepared_request(pool, capacity, outputs, adaptive);
    let (sampler, completion) = preparation
        .claim_sampling(config)
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    (sampler, preparation, run)
}

#[derive(Default)]
struct Context {
    callbacks: Cell<usize>,
}

struct Scalar;
impl SamplingBackend for Scalar {
    type Logits = u32;
    type Token = u32;
    type RandomState = ();
    type Context = Context;
    type Error = String;

    fn error(message: String) -> String {
        message
    }
    fn validate_token(token: &u32, _: TokenDomain, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn scale_temperature(logits: &u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_penalties(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn apply_top_k(logits: u32, _: i32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_top_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_min_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_token_filter(logits: &u32, _: &TokenFilter, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_mirostat(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn sample_raw(logits: &u32, _: f32, _: Option<&mut ()>, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn sample_processed(
        logits: &u32,
        _: f32,
        _: Option<&mut ()>,
        _: &Context,
    ) -> Result<u32, String> {
        Ok(*logits)
    }
    fn token_id(token: &u32, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn token_probability(_: &u32, _: u32, _: &Context) -> Result<f32, String> {
        Ok(0.125)
    }
}

pub(super) fn grow(sampler: &mut RunOwnedTextSampler, tokens: &[u32]) {
    let context = Context::default();
    for token in tokens {
        assert_eq!(
            sampler
                .prepare_sample()
                .unwrap()
                .sample::<Scalar>(token, 0.7, None, &context)
                .unwrap(),
            *token
        );
    }
    assert_eq!(context.callbacks.get(), tokens.len());
}

pub(super) fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

#[derive(Debug, Default)]
pub(super) struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}

impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if self.missing_tensor {
            return Ok(None);
        }
        let [layout] = operation.outputs.as_slice() else {
            panic!("closed copy has one output");
        };
        let capacity = layout.bytes()?.div_ceil(16) * 16;
        let effect = match operation.kind {
            WorkspaceOperationKind::Contiguous => WorkspaceOutputStorage::AllocateOrAliasInputs {
                bytes: capacity,
                inputs: vec![0],
            },
            WorkspaceOperationKind::DeepCopy => WorkspaceOutputStorage::Allocate(capacity),
            _ => panic!("only the closed copy program is allowed"),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![effect],
            scratch_bytes: 3,
            assumptions: "fixture: sixteen-byte padded result and three scratch bytes".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 5,
            assumptions: "fixture: five disjoint staging bytes per operation".into(),
        }))
    }
}
