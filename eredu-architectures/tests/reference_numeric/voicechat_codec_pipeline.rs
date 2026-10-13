//! Waveform agreement encloses local provider and spectral failures.
use super::voicechat_rnnt_coordination::{coordinator_pair, Coordinator};
use super::voicechat_speech::codec;
use super::*;
use eredu_codec::{
    ear::{AudioDecoderExecution, AudioDecoderKernel},
    ear_decoder::{DecoderConfig, DecoderState, NeuralDecoder},
    ear_pipeline::{DecoderPipelineExecution, PipelineAudioDecoder},
    prvq::{DistanceMetric, Geometry, Quantized, ResidualQuantizer},
    prvq_layered::QuantizerExecution,
    Error as CodecError,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use std::sync::atomic::Ordering;

struct Pipeline {
    decoder: NeuralDecoder<NumericBackend>,
    calls: usize,
    malformed_spectrum: bool,
}
impl DecoderPipelineExecution<NumericBackend> for Pipeline {
    type State = DecoderState<NumericTensor>;
    fn config(&self) -> &DecoderConfig {
        self.decoder.config()
    }
    fn validate_state(
        &self,
        state: &Self::State,
        _: i32,
        finished: bool,
    ) -> Result<(), CodecError> {
        let expected = if finished {
            eredu_codec::streaming::StreamPhase::Finished
        } else {
            eredu_codec::streaming::StreamPhase::AcceptingInput
        };
        if state.phase() != expected {
            return Err(CodecError::InvalidShape(
                "neural and waveform lifecycle differ".into(),
            ));
        }
        Ok(())
    }
    fn initial_state(&self) -> Result<Self::State, CodecError> {
        self.decoder.initial_state()
    }
    fn finish(&self, parent: &Self::State) -> Result<Self::State, CodecError> {
        self.decoder.finish(parent)
    }
    fn forward(
        &mut self,
        latent: &NumericTensor,
        state: &mut Self::State,
        _: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &NumericContext,
    ) -> Result<(NumericTensor, DistributedCommitOutcome), BackendFailure> {
        self.calls += 1;
        let (spectrum, next) = self
            .decoder
            .step(latent, state, context)
            .map_err(BackendFailure::from_error)?;
        *state = next;
        Ok((
            if self.malformed_spectrum {
                NumericTensor::zeros([1, 1, 1])
            } else {
                spectrum
            },
            DistributedCommitOutcome::Committed(epoch),
        ))
    }
}
struct Quantizer {
    inner: ResidualQuantizer<NumericBackend>,
    calls: usize,
    fail: bool,
}
impl QuantizerExecution<NumericBackend> for Quantizer {
    fn validate_coordinated_execution(&self) -> Result<(), CodecError> {
        Ok(())
    }
    fn geometry(&self) -> Geometry {
        self.inner.geometry()
    }
    fn decode_range(
        &mut self,
        codes: &NumericTensor,
        range: std::ops::Range<usize>,
        context: &NumericContext,
    ) -> Result<NumericTensor, CodecError> {
        self.calls += 1;
        if self.fail {
            return Err(CodecError::InvalidShape(
                "injected local codebook failure".into(),
            ));
        }
        self.inner.decode_range(codes, range, context)
    }
    fn quantize_range(
        &mut self,
        latent: &NumericTensor,
        range: std::ops::Range<usize>,
        metric: DistanceMetric,
        context: &NumericContext,
    ) -> Result<Quantized<NumericTensor>, CodecError> {
        self.inner.quantize_range(latent, range, metric, context)
    }
}
#[test]
fn waveform_pipeline_agrees_codes_provider_and_spectral_failures_without_adopting_parent() {
    let (coordinators, timeouts) = coordinator_pair();
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for (rank, coordinator) in coordinators.into_iter().enumerate() {
            workers.push(scope.spawn(move || {
                run_worker(rank, coordinator);
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}

fn run_worker(rank: usize, mut coordinator: Coordinator) {
    let context = NumericContext::default();
    let fixture: serde_json::Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_CHARACTER_TRANSITION_REFERENCE_JSON)
            .unwrap();
    let mut ordinary = codec(&fixture, &context);
    let selected = codec(&fixture, &context);
    let kernel = AudioDecoderKernel::new(selected.config().clone(), &context).unwrap();
    let mut neural = Pipeline {
        decoder: selected.neural,
        calls: 0,
        malformed_spectrum: false,
    };
    let mut quantizer = Quantizer {
        inner: selected.quantizer,
        calls: 0,
        fail: false,
    };
    let epoch = |n| DistributedCommitEpoch::new(n).unwrap();
    let codes = NumericTensor::from_i32_slice(&[0; 31], &[1, 1, 31], &context).unwrap();
    let parent = PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(1),
    )
    .initial_state(1)
    .unwrap();
    // Identical geometry with different decisions cannot reach codebooks.
    let different =
        NumericTensor::from_i32_slice(&[rank as i32; 31], &[1, 1, 31], &context).unwrap();
    {
        let mut adapter = PipelineAudioDecoder::new(
            &kernel,
            &mut neural,
            &mut quantizer,
            &mut coordinator,
            [1, 2, 3],
            epoch(1),
        );
        assert!(adapter.step(&different, &parent, true, &context).is_err());
        assert!(adapter.is_consumed());
        assert!(adapter.step(&codes, &parent, true, &context).is_err());
    }
    assert_eq!((neural.calls, quantizer.calls), (0, 0));
    // A rank-local provider failure reaches the same gate on its peer.
    quantizer.fail = rank == 0;
    assert!(PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(2)
    )
    .step(&codes, &parent, true, &context)
    .is_err());
    assert_eq!(neural.calls, 0);
    quantizer.fail = false;
    // A completed neural candidate is still unpublished if spectral work fails.
    neural.malformed_spectrum = rank == 0;
    assert!(PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(3)
    )
    .step(&codes, &parent, true, &context)
    .is_err());
    assert_eq!(parent.frames_seen(), 0);
    assert_eq!(parent.samples_emitted(), 0);
    assert!(parent
        .neural()
        .units()
        .iter()
        .all(|unit| unit.carry().is_none()));
    neural.malformed_spectrum = false;
    let (expected, expected_state) = ordinary
        .step(&codes, &ordinary.initial_state(1).unwrap(), true, &context)
        .unwrap();
    let (actual, state) = PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(4),
    )
    .step(&codes, &parent, true, &context)
    .unwrap();
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.data, expected.data);
    assert_eq!(state.frames_seen(), 1);
    let calls = neural.calls;
    // Drain and step cannot disagree silently across ranks.
    {
        let mut adapter = PipelineAudioDecoder::new(
            &kernel,
            &mut neural,
            &mut quantizer,
            &mut coordinator,
            [1, 2, 3],
            epoch(5),
        );
        if rank == 0 {
            assert!(adapter.finish(&state, true, &context).is_err());
        } else {
            assert!(adapter.step(&codes, &state, true, &context).is_err());
        }
    }
    assert_eq!(neural.calls, calls);
    let (expected_tail, _) = ordinary.finish(&expected_state, true, &context).unwrap();
    let (tail, finished) = PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(6),
    )
    .finish(&state, true, &context)
    .unwrap();
    assert_eq!(tail.unwrap().data, expected_tail.unwrap().data);
    assert!(finished.is_finished());
    assert_eq!(finished.frames_seen(), 1);
    let (tail, again) = PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(7),
    )
    .finish(&finished, true, &context)
    .unwrap();
    assert!(tail.is_none());
    assert_eq!(again.samples_emitted(), finished.samples_emitted());
    // Equal codebooks with different spectral geometry still reject before reads.
    let mut geometry = kernel.config().clone();
    if rank == 1 {
        geometry.spectral.hop_length = 4;
    }
    let different_kernel = AudioDecoderKernel::new(geometry, &context).unwrap();
    let reads = quantizer.calls;
    let mut adapter = PipelineAudioDecoder::new(
        &different_kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(8),
    );
    let different_parent = adapter.initial_state(1).unwrap();
    assert!(adapter
        .step(&codes, &different_parent, true, &context)
        .is_err());
    assert_eq!(quantizer.calls, reads);

    let wrong_phase = if rank == 0 {
        parent.clone().map_neural(|_| finished.neural().clone())
    } else {
        parent.clone()
    };
    let mut adapter = PipelineAudioDecoder::new(
        &kernel,
        &mut neural,
        &mut quantizer,
        &mut coordinator,
        [1, 2, 3],
        epoch(9),
    );
    assert!(adapter.step(&codes, &wrong_phase, true, &context).is_err());
    assert_eq!(quantizer.calls, reads);
}
