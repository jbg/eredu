//! Recorded PCM advancement over the shared perception/RNNT pipeline transition.
use super::{
    frontend::PcmPrecision,
    input::{self, InputError, InputExecution, InputFrame, InputState},
    perception_rnnt::PerceptionRnntLimits,
    perception_rnnt_pipeline::{self, PerceptionPipelineExecution},
    rnnt::RnntProjectionExecution,
    rnnt_pipeline::{RnntPipelineCursor, RnntPredictorExecution},
};
use eredu_core::DistributedCommitEpoch;
use eredu_media::audio::streaming::PcmFrameSource;
use eredu_nn::{NeuralBackend, RetainedTensorRoots, Tensor};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

const PROTOCOL: u64 = 0x50434d5049500001;

/// Borrowed authority for at most one ready PCM frame.
/// Empty/incomplete advances retain the same authority. Once a ready frame is
/// agreed, a retry or fork needs a fresh adapter and perception epoch; the RNNT
/// cursor stays caller-owned and is never restored with input state.
pub struct PipelineInput<'a, B: NeuralBackend, P, J, R, C> {
    perception: &'a mut P,
    projections: &'a mut J,
    predictor: &'a mut R,
    coordinator: &'a mut C,
    identity: [u64; 3],
    epoch: Option<DistributedCommitEpoch>,
    cursor: &'a mut RnntPipelineCursor,
    backend: PhantomData<B>,
}
impl<'a, B: NeuralBackend, P, J, R, C> PipelineInput<'a, B, P, J, R, C> {
    /// Retains explicit identity and borrowed component resources without execution.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        perception: &'a mut P,
        projections: &'a mut J,
        predictor: &'a mut R,
        coordinator: &'a mut C,
        identity: [u64; 3],
        perception_epoch: DistributedCommitEpoch,
        cursor: &'a mut RnntPipelineCursor,
    ) -> Self {
        Self {
            perception,
            projections,
            predictor,
            coordinator,
            identity,
            epoch: Some(perception_epoch),
            cursor,
            backend: PhantomData,
        }
    }
    /// Whether this invocation has already admitted a ready frame attempt.
    pub fn is_consumed(&self) -> bool {
        self.epoch.is_none()
    }
}
fn agreed<T, B, C>(
    coordinator: &mut C,
    descriptor: [u64; 8],
    local: Result<T, InputError>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<T, InputError>
where
    B: NeuralBackend,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    if !coordinator.agree(descriptor, local.is_ok(), context)? {
        return Err(local.err().unwrap_or(InputError::State(
            "PCM participants differ in preparation, availability or invocation identity",
        )));
    }
    local
}
impl<B, P, J, R, C> InputExecution<B> for PipelineInput<'_, B, P, J, R, C>
where
    B: NeuralBackend,
    P: PerceptionPipelineExecution<B>,
    P::State: RetainedTensorRoots<B::Tensor>,
    J: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    type PerceptionCache = P::State;
    fn advance_pcm(
        &mut self,
        previous: &InputState<B::Tensor, P::State>,
        limits: PerceptionRnntLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<InputFrame<B::Tensor, P::State>>, InputError> {
        let frontend = previous.frontend();
        let ready = frontend.has_ready_frame();
        let epoch = self.epoch;
        let descriptor = [
            PROTOCOL,
            self.identity[0],
            self.identity[1],
            self.identity[2],
            epoch.map_or(0, DistributedCommitEpoch::value),
            frontend.frames(),
            previous.neural().frames(),
            u64::from(ready),
        ];
        let preflight = if epoch.is_none() {
            Err(InputError::State(
                "PCM pipeline invocation has already consumed its perception epoch",
            ))
        } else if frontend.frames() != previous.neural().frames() {
            Err(InputError::State("PCM and neural input frontiers differ"))
        } else {
            Ok(())
        };
        agreed::<_, B, C>(self.coordinator, descriptor, preflight, context)?;
        let options = frontend.options();
        let precision = match options.pcm_precision {
            PcmPrecision::F32 => 0,
            PcmPrecision::F16 => 1,
            PcmPrecision::Bf16 => 2,
        };
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 1,
                frontend.queued_samples() as u64,
                options.buffer_frames as u64,
                options.queue_capacity_samples as u64,
                precision,
                u64::from(options.pad_and_drop_preencoded),
                u64::from(frontend.is_finished()),
                frontend.remaining_continuation_frames(),
            ],
            Ok(()),
            context,
        )?;
        if !ready {
            return Ok(None);
        }
        // No-ready calls spend only bounded agreement resources, without reserving
        // a neural epoch or materializing a mel tensor.
        let epoch = self.epoch.take().expect("agreed unconsumed invocation");
        let prepared = input::prepare_frontend(previous).and_then(|prepared| {
            let prepared = prepared.ok_or(InputError::State(
                "ready PCM frame disappeared during preparation",
            ))?;
            super::perception_rnnt::validate_advance(
                (self.perception.dimensions(), self.projections.dimensions()),
                &prepared.shape,
                prepared.frontend.drop_preencoded,
                previous.neural(),
                limits,
            )?;
            Ok(prepared)
        });
        let mut phase = descriptor;
        phase[0] = PROTOCOL + 2;
        let prepared = agreed::<_, B, C>(self.coordinator, phase, prepared, context)?;
        let step = &prepared.frontend;
        let source = match step.source {
            PcmFrameSource::Input => 0,
            PcmFrameSource::Continuation => 1,
        };
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 3,
                step.frame_index,
                step.start_sample,
                step.valid_samples as u64,
                source,
                prepared.shape[1] as u64,
                prepared.shape[2] as u64,
                step.drop_preencoded as u64,
            ],
            Ok(()),
            context,
        )?;
        // Compare the actual host-processed float values before allocating native
        // tensors. This catches differing PCM or preprocessing despite equal lengths.
        let mut hash = Sha256::new();
        for value in &step.mel.values {
            hash.update(value.to_bits().to_le_bytes());
        }
        let digest = hash.finalize();
        let mut words = [
            PROTOCOL + 4,
            self.identity[0],
            self.identity[1],
            self.identity[2],
            0,
            0,
            0,
            0,
        ];
        for (word, bytes) in words[4..].iter_mut().zip(digest.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().unwrap());
        }
        agreed::<_, B, C>(self.coordinator, words, Ok(()), context)?;
        let local = input::materialize::<B>(&prepared, context);
        phase[0] = PROTOCOL + 5;
        let mel = agreed::<_, B, C>(self.coordinator, phase, local, context)?;
        let neural = perception_rnnt_pipeline::advance(
            self.perception,
            self.projections,
            self.predictor,
            self.coordinator,
            self.identity,
            epoch,
            self.cursor,
            &mel,
            prepared.frontend.drop_preencoded,
            previous.neural(),
            limits,
            context,
        )?;
        let candidate = input::compose_frame(prepared, neural);
        phase[0] = PROTOCOL + 6;
        Ok(Some(agreed::<_, B, C>(
            self.coordinator,
            phase,
            candidate,
            context,
        )?))
    }
}
