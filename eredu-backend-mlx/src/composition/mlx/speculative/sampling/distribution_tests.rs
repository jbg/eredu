use super::*;
use eredu_core::capture::*;
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError},
    GenerationSampler, SamplingBackend,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

type Sampling = MlxSpeculativeSampling<Probe>;

#[derive(Default)]
struct Calls {
    process: Cell<usize>,
    sample: Cell<usize>,
    commit: Cell<usize>,
    submitted: Cell<usize>,
}

#[derive(Clone, Copy)]
enum Failure {
    None,
    Error,
    Panic,
}

#[derive(Clone)]
struct Probe {
    pool: MemoryLedger,
    sample_pool: Option<MemoryLedger>,
    calls: Rc<Calls>,
    failure: Failure,
    retain_sample: bool,
    retained: Rc<RefCell<Option<Array>>>,
}

impl Probe {
    fn new(pool: &MemoryLedger) -> Self {
        Self {
            pool: pool.clone(),
            sample_pool: None,
            calls: Rc::default(),
            failure: Failure::None,
            retain_sample: false,
            retained: Rc::default(),
        }
    }
}

#[derive(Debug)]
struct CallbackFailure;
impl std::fmt::Display for CallbackFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("distribution callback failed after submission")
    }
}
impl std::error::Error for CallbackFailure {}

impl SpeculativeSampler<MlxSamplingBackend> for Probe {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn process_logits(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        history: &[u32],
        context: &Stream,
    ) -> Result<MlxTensor, Exception> {
        self.calls.process.set(self.calls.process.get() + 1);
        assert!(self.pool.unquoted_owner_count().unwrap() > 0);
        if !matches!(self.failure, Failure::None) {
            let pending = logits.as_array().exp(context)?;
            let _completion = async_eval_with_event([&pending])?;
            self.calls.submitted.set(self.calls.submitted.get() + 1);
            if matches!(self.failure, Failure::Panic) {
                panic!("distribution callback panic after submission");
            }
            return Err(Exception::from_source(CallbackFailure));
        }
        SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut GenerationSampler::new(),
            logits,
            temperature,
            history,
            context,
        )
    }

    fn sample_processed(
        &self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        context: &Stream,
    ) -> Result<MlxTensor, Exception> {
        self.calls.sample.set(self.calls.sample.get() + 1);
        assert!(
            self.sample_pool
                .as_ref()
                .unwrap_or(&self.pool)
                .unquoted_owner_count()
                .unwrap()
                > 0
        );
        if self.retain_sample {
            let retained = logits.as_array().exp(context)?;
            retained.evaluated()?;
            self.retained.replace(Some(retained));
        }
        MlxSamplingBackend::sample_processed(logits, temperature, random, context)
    }

    fn commit_token(&mut self, _: &MlxTensor, _: u32, _: &Stream) -> Result<(), Exception> {
        self.calls.commit.set(self.calls.commit.get() + 1);
        Ok(())
    }
}

fn reserved_ledger() -> MemoryLedger {
    let probe = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let controls = crate::memory_fixture::host_total(
        &probe
            .reservation_requirements(&zero_admission(), None)
            .unwrap(),
    );
    crate::memory_fixture::ledger(controls, 0).unwrap()
}

fn zero_admission() -> eredu_core::Admission {
    use eredu_core::{
        cache::LayerCachePolicy, EstimationCompleteness, ExecutionWorkspaceEstimate,
        InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
        WorkspaceBound,
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless speculative distribution fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(eredu_core::Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(0),
    })
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn settle(pool: &MemoryLedger, expected: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == expected
    });
}

fn blocked(pool: &MemoryLedger) {
    settle(pool, 1);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

fn source<'a, T: std::error::Error + 'static>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    let mut current = error;
    loop {
        if let Some(found) = current.downcast_ref::<T>() {
            return Some(found);
        }
        current = current.source()?;
    }
}

fn assert_reserved(error: &Exception) {
    assert_eq!(
        source::<WorkingMemoryError>(error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
}

fn logits(probabilities: &[f32]) -> Array {
    Array::from_slice(
        &probabilities
            .iter()
            .map(|value| value.ln())
            .collect::<Vec<_>>(),
        &[1, probabilities.len() as i32],
    )
}

fn distribution(
    sampler: &mut Sampling,
    probabilities: &[f32],
    context: SpeculativeExecutionStreams<'_>,
) -> MlxSpeculativeDistribution {
    sampler
        .process_logits(
            &logits(probabilities),
            1.0,
            &[],
            SamplingPlacement::Target,
            context,
        )
        .unwrap()
}

fn assert_probabilities(
    sampler: &Sampling,
    distribution: &MlxSpeculativeDistribution,
    expected: &[f32],
    context: SpeculativeExecutionStreams<'_>,
) {
    for (token, expected) in expected.iter().enumerate() {
        let actual = sampler
            .probability_at(
                distribution,
                token as u32,
                SamplingPlacement::Target,
                context,
            )
            .unwrap();
        assert!(
            (actual - expected).abs() < 2e-6,
            "token {token}: {actual} versus {expected}"
        );
    }
}

fn capture_plan() -> AdmittedCapturePlan {
    use eredu_core::*;
    let point = ObservationPoint {
        path: MODEL_LOGITS_OBSERVATION_PATH.into(),
        node_id: "model".into(),
        meaning: "raw logits".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            [
                ("batch", SymbolicDimension::Batch),
                ("sequence", SymbolicDimension::Sequence),
                ("vocabulary", SymbolicDimension::Known(3)),
            ]
            .into_iter()
            .map(|(name, dimension)| TensorAxis {
                name: name.into(),
                dimension,
            })
            .collect(),
        ),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let usage = CaptureUsage {
        captures: 4,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "distribution".into(),
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 3 },
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &crate::composition::mlx::session::bounded_capture::capabilities(),
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 1,
            max_predictions: 2,
        },
    )
    .unwrap()
}

#[test]
fn reserved_distribution_processing_precedes_policy_callbacks_and_capture_ledger_mutation() {
    let stream = stream();
    let pool = reserved_ledger();
    let mut sampler = Sampling::new(Probe::new(&pool));
    sampler.enable_control_capture(capture_plan()).unwrap();
    let before = sampler
        .capture
        .as_ref()
        .unwrap()
        .borrow()
        .session
        .cumulative_usage();
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let reserved_snapshot = pool.snapshot().unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool);
    assert_reserved(
        &sampler
            .process_logits(
                &logits(&[0.2, 0.3, 0.5]),
                1.0,
                &[],
                SamplingPlacement::Target,
                context,
            )
            .unwrap_err(),
    );
    assert_eq!(sampler.inner().calls.process.get(), 0);
    assert_eq!(
        sampler
            .capture
            .as_ref()
            .unwrap()
            .borrow()
            .session
            .cumulative_usage(),
        before
    );
    assert!(sampler.take_control_captures().is_empty());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.snapshot().unwrap(), reserved_snapshot);
    drop(reservation);
    let distribution = distribution(&mut sampler, &[0.2, 0.3, 0.5], context);
    assert_eq!(sampler.inner().calls.process.get(), 1);
    assert_eq!(sampler.take_control_captures().len(), 1);
    assert!(
        sampler
            .capture
            .as_ref()
            .unwrap()
            .borrow()
            .session
            .cumulative_usage()
            .captures
            > before.captures
    );
    drop((sampler, distribution));
    settle(&pool, 0);
}

#[test]
fn foreign_distribution_cannot_enter_reserved_domain_or_invoke_sample_and_commit_callbacks() {
    let target = stream();
    let draft = stream();
    let pool_a = reserved_ledger();
    let pool_b = reserved_ledger();
    let context_a = SpeculativeExecutionStreams::single(&target).with_memory_ledger(&pool_a);
    let context_b = SpeculativeExecutionStreams::for_test(&target, &draft)
        .unwrap()
        .with_memory_ledger(&pool_b);
    let mut sampler = Sampling::new(Probe::new(&pool_a));
    let mut distribution = distribution(&mut sampler, &[0.2, 0.3, 0.5], context_a);
    let before = distribution
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let allocation = distribution.as_array().allocation_info().unwrap();
    let reservation = pool_b
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let reserved_snapshot = pool_b.snapshot().unwrap();
    assert_reserved(
        &sampler
            .probability_at(&distribution, 1, SamplingPlacement::Target, context_b)
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .positive_probability_difference(
                &distribution,
                &distribution,
                SamplingPlacement::Target,
                context_b,
            )
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .sample(
                &distribution,
                0.0,
                None,
                SamplingPlacement::Target,
                context_b,
            )
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .update_sampler_state(&distribution, 2, SamplingPlacement::Target, context_b)
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .prepare_verification(&mut [&mut distribution], 1.0, context_b)
            .unwrap_err(),
    );
    sampler
        .prepare_verification(&mut [], 1.0, context_b)
        .unwrap();
    sampler
        .prepare_verification(&mut [&mut distribution], 0.0, context_b)
        .unwrap();
    sampler
        .prepare_verification(
            &mut [&mut distribution],
            1.0,
            SpeculativeExecutionStreams::single(&target).with_memory_ledger(&pool_b),
        )
        .unwrap();
    assert_eq!(sampler.inner().calls.sample.get(), 0);
    assert_eq!(sampler.inner().calls.commit.get(), 0);
    assert_eq!(
        distribution
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        before
    );
    assert_eq!(
        distribution.as_array().allocation_info().unwrap(),
        allocation
    );
    assert_probabilities(&sampler, &distribution, &[0.2, 0.3, 0.5], context_a);
    assert_eq!(pool_b.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool_b.snapshot().unwrap(), reserved_snapshot);
    drop((reservation, distribution, sampler));
    settle(&pool_a, 0);
}

#[test]
fn lazy_distribution_clone_keeps_authority_after_source_and_sampler_teardown() {
    let stream = stream();
    let pool = reserved_ledger();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool);
    let mut sampler = Sampling::new(Probe::new(&pool));
    let raw = logits(&[0.2, 0.3, 0.5])
        .add(&Array::from_f32(0.25), &stream)
        .unwrap();
    let distribution = sampler
        .process_logits(&raw, 1.0, &[], SamplingPlacement::Target, context)
        .unwrap();
    assert!(distribution.as_array().allocation_info().unwrap().is_none());
    let surviving = distribution.clone();
    drop((sampler, raw, distribution));
    blocked(&pool);
    let values = surviving
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    for (actual, probability) in values.iter().zip([0.2_f32, 0.3, 0.5]) {
        assert!((actual - (probability.ln() + 0.25)).abs() < 2e-6);
    }
    drop(surviving);
    settle(&pool, 0);
}

#[test]
fn repeated_same_ledger_distribution_operations_reuse_one_authority_and_preserve_numerics() {
    let target = stream();
    let draft = stream();
    let pool = reserved_ledger();
    let context = SpeculativeExecutionStreams::for_test(&target, &draft)
        .unwrap()
        .with_memory_ledger(&pool);
    let mut sampler = Sampling::new(Probe::new(&pool));
    let left = distribution(&mut sampler, &[0.6, 0.3, 0.1], context);
    let mut right = distribution(&mut sampler, &[0.2, 0.2, 0.6], context);
    for _ in 0..3 {
        assert_probabilities(&sampler, &left, &[0.6, 0.3, 0.1], context);
        assert_eq!(
            sampler
                .sample(&left, 0.0, None, SamplingPlacement::Target, context)
                .unwrap(),
            0
        );
        sampler
            .update_sampler_state(&left, 0, SamplingPlacement::Target, context)
            .unwrap();
        sampler
            .prepare_verification(&mut [&mut right], 1.0, context)
            .unwrap();
        let residual = sampler
            .positive_probability_difference(&left, &right, SamplingPlacement::Target, context)
            .unwrap()
            .unwrap();
        assert_probabilities(&sampler, &residual, &[0.8, 0.2, 0.0], context);
        assert_eq!(
            sampler
                .sample(&residual, 0.0, None, SamplingPlacement::Target, context)
                .unwrap(),
            0
        );
        assert!(sampler
            .positive_probability_difference(&left, &left, SamplingPlacement::Target, context)
            .unwrap()
            .is_none());
        drop(residual);
        settle(&pool, 1);
    }
    assert_eq!(sampler.inner().calls.commit.get(), 3);
    assert_eq!(sampler.inner().calls.sample.get(), 6);
    drop((sampler, left, right));
    settle(&pool, 0);
}

#[test]
fn cross_domain_residual_and_verification_keep_all_source_and_destination_authorities() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let draft = stream();
        let target = Stream::new_with_device(&safemlx::Device::new(device, 0));
        let pool_a = reserved_ledger();
        let pool_b = reserved_ledger();
        let pool_c = reserved_ledger();
        let context_a = SpeculativeExecutionStreams::single(&draft).with_memory_ledger(&pool_a);
        let context_b = SpeculativeExecutionStreams::single(&draft).with_memory_ledger(&pool_b);
        let context_c = SpeculativeExecutionStreams::for_test(&target, &draft)
            .unwrap()
            .with_memory_ledger(&pool_c);
        let mut sampler_a = Sampling::new(Probe::new(&pool_a));
        let mut sampler_b = Sampling::new(Probe::new(&pool_b));
        let left = distribution(&mut sampler_a, &[0.6, 0.3, 0.1], context_a);
        let right = distribution(&mut sampler_b, &[0.2, 0.2, 0.6], context_b);
        let mut residual = sampler_a
            .positive_probability_difference(&left, &right, SamplingPlacement::Target, context_b)
            .unwrap()
            .unwrap();
        assert_probabilities(&sampler_a, &residual, &[0.8, 0.2, 0.0], context_b);
        sampler_a
            .prepare_verification(&mut [&mut residual], 1.0, context_c)
            .unwrap();
        assert_probabilities(&sampler_a, &residual, &[0.8, 0.2, 0.0], context_c);
        let clone = residual.clone();
        drop((sampler_a, sampler_b, left, right, residual));
        for pool in [&pool_a, &pool_b, &pool_c] {
            blocked(pool);
        }
        let probabilities = probabilities(clone.as_array(), &target)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!((probabilities[0] - 0.8).abs() < 2e-6);
        assert!((probabilities[1] - 0.2).abs() < 2e-6);
        assert_eq!(probabilities[2], 0.0);
        drop(clone);
        for pool in [&pool_a, &pool_b, &pool_c] {
            settle(pool, 0);
        }
    }
}

#[test]
fn callback_errors_and_unwinds_keep_authority_through_native_recovery() {
    let stream = stream();
    for failure in [Failure::Error, Failure::Panic] {
        let pool = reserved_ledger();
        let context = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool);
        let mut probe = Probe::new(&pool);
        probe.failure = failure;
        let calls = probe.calls.clone();
        let mut sampler = Sampling::new(probe);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sampler.process_logits(
                &logits(&[0.2, 0.3, 0.5]),
                1.0,
                &[],
                SamplingPlacement::Target,
                context,
            )
        }));
        match failure {
            Failure::Error => {
                assert!(source::<CallbackFailure>(&outcome.unwrap().unwrap_err()).is_some())
            }
            Failure::Panic => assert!(outcome.is_err()),
            Failure::None => unreachable!(),
        }
        assert_eq!(calls.process.get(), 1);
        assert_eq!(calls.submitted.get(), 1);
        blocked(&pool);
        drop(sampler);
        settle(&pool, 0);
    }
}

#[test]
fn earlier_sampler_clone_retains_cross_domain_arrays_stored_by_shared_sampling_callback() {
    let stream = stream();
    let pool_a = reserved_ledger();
    let pool_b = reserved_ledger();
    let context_a = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool_a);
    let context_b = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool_b);
    let mut probe = Probe::new(&pool_a);
    probe.sample_pool = Some(pool_b.clone());
    probe.retain_sample = true;
    let mut sampler = Sampling::new(probe);
    let earlier_clone = sampler.clone();
    let raw = logits(&[0.2, 0.3, 0.5]);
    let distribution = sampler
        .process_logits(&raw, 1.0, &[], SamplingPlacement::Target, context_a)
        .unwrap();
    assert_eq!(
        sampler
            .sample(
                &distribution,
                0.0,
                None,
                SamplingPlacement::Target,
                context_b
            )
            .unwrap(),
        2
    );
    drop((sampler, raw, distribution));
    blocked(&pool_a);
    blocked(&pool_b);
    {
        let payload = earlier_clone.inner().retained.borrow();
        let values = payload
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for (actual, expected) in values.iter().zip([0.2_f32, 0.3, 0.5]) {
            assert!((actual - expected).abs() < 2e-6);
        }
    }
    drop(earlier_clone);
    settle(&pool_a, 0);
    settle(&pool_b, 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
