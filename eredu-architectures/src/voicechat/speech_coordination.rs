//! Selected owner authority for ordinary speech refinement and its RNG transaction.
use super::super::{
    speech_sampling::SpeechSamplingError,
    tts_mog::{TtsError, GUMBEL_EPSILON},
    tts_rvq_layered::{RefinementPhase, RefinementSampling},
};
use eredu_nn::{NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    replicated_execution::ReplicatedCoordinator, DenseSamplingBackend, GumbelTruncation,
    RealtimeGenerationBranch, RealtimeRandomStateTransactionError, RealtimeScheduleState,
};

const PROTOCOL: u64 = 0x5350454543480001;

/// One explicit coordinated sampling invocation over retained communication.
///
/// The caller supplies recorded request, turn and invocation identity. Only the
/// coordinator's owner draws randomness; followers may have no RNG. Authority is
/// consumed on admission, including failure. Retry and replay require a fresh
/// authority and an owner-supplied unpublished branch. This object owns neither
/// a scheduler nor a commit epoch and is not part of a model snapshot.
pub struct SpeechSamplingCoordination<'a, T: Tensor> {
    coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
    identity: [u64; 3],
    consumed: bool,
    fenced: bool,
}
impl<'a, T: Tensor> SpeechSamplingCoordination<'a, T> {
    /// Borrows selected bounded communication and its recorded invocation identity.
    pub fn new(
        coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
        identity: [u64; 3],
    ) -> Self {
        Self {
            coordinator,
            identity,
            consumed: false,
            fenced: false,
        }
    }
    /// Whether this authority has admitted its one sampling operation.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    fn agree(
        &mut self,
        stage: u64,
        words: &[u64],
        success: bool,
        context: &T::Context,
    ) -> Result<(), TtsError> {
        if self.fenced {
            return Err(TtsError::Configuration(
                "speech sampling authority is fenced",
            ));
        }
        let result = (|| {
            // Agree length before visiting variable metadata, so mismatched
            // shapes or policies cannot cause different numbers of exchanges.
            if !self.coordinator.agree(
                [
                    PROTOCOL,
                    self.identity[0],
                    self.identity[1],
                    self.identity[2],
                    stage,
                    words.len() as u64,
                    0,
                    0,
                ],
                success,
                context,
            )? {
                return Err(TtsError::Configuration(
                    "speech sampling participants disagree or a participant failed",
                ));
            }
            for (index, chunk) in words.chunks(6).enumerate() {
                let mut descriptor = [u64::MAX; 8];
                descriptor[0] = PROTOCOL;
                descriptor[1] = index as u64;
                descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
                if !self.coordinator.agree(descriptor, true, context)? {
                    return Err(TtsError::Configuration(
                        "speech sampling metadata differs between participants",
                    ));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.fenced = true;
        }
        result
    }

    fn publish(&mut self, value: T, context: &T::Context) -> Result<T, TtsError> {
        if self.fenced {
            return Err(TtsError::Configuration(
                "speech sampling authority is fenced",
            ));
        }
        match self.coordinator.publish(value, context) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.fenced = true;
                Err(error.into())
            }
        }
    }
}

struct Sampler<'a, 'c, B: NeuralBackend, D: DenseSamplingBackend> {
    random: &'a mut Option<D::RandomState>,
    coordination: Option<&'a mut SpeechSamplingCoordination<'c, B::Tensor>>,
    latent: usize,
    context: &'a D::Context,
}

fn validate_draws<T: Tensor>(
    mixture: &T,
    gaussian: &T,
    shape: &[i32],
    latent: i32,
) -> Result<(), TtsError> {
    if mixture.shape() != &shape[..2]
        || mixture.element_type() != Some(TensorElementType::I32)
        || gaussian.shape() != [shape[0], shape[1], latent]
        || gaussian.element_type() != Some(TensorElementType::F32)
    {
        return Err(TtsError::Shape(
            "speech sampling returned incompatible decisions or noise",
        ));
    }
    Ok(())
}

impl<B, D> RefinementSampling<B> for Sampler<'_, '_, B, D>
where
    B: NeuralBackend,
    D: DenseSamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    D::Error: std::error::Error + Send + Sync + 'static,
{
    fn agree(&mut self, phase: RefinementPhase, success: bool) -> Result<(), TtsError> {
        if let Some(coordination) = self.coordination.as_deref_mut() {
            coordination.agree(phase.stage as u64, &phase.words, success, self.context)?;
        }
        Ok(())
    }

    fn sample(
        &mut self,
        slot: usize,
        logits: &B::Tensor,
        top_p: f64,
    ) -> Result<(B::Tensor, B::Tensor), TtsError> {
        let shape = logits.shape();
        let prepared = (|| {
            GumbelTruncation::TopP(top_p)
                .validate()
                .map_err(eredu_nn::Error::backend_source)?;
            if shape.len() != 3 {
                return Err(TtsError::Shape(
                    "speech logits require batch, time and mixtures",
                ));
            }
            let latent = i32::try_from(self.latent)
                .map_err(|_| TtsError::Configuration("speech latent width exceeds I32"))?;
            for dimensions in [shape, &[shape[0], shape[1], latent]] {
                eredu_runtime::dense_sampling::validate_dense_draw_shape(dimensions)
                    .map_err(eredu_nn::Error::backend_source)?;
            }
            Ok(latent)
        })();
        if let Some(coordination) = self.coordination.as_deref_mut() {
            let mut words = vec![slot as u64, top_p.to_bits(), self.latent as u64];
            words.extend(shape.iter().map(|n| *n as u64));
            let agreement = coordination.agree(102, &words, prepared.is_ok(), self.context);
            if prepared.is_err() {
                return Err(prepared.unwrap_err());
            }
            agreement?;
        }
        let latent = prepared?;
        let owner = self
            .coordination
            .as_ref()
            .is_none_or(|c| c.coordinator.is_owner());
        let drawn = (|| {
            let (mixture, gaussian) = if owner {
                let random = self.random.as_mut().ok_or(TtsError::Configuration(
                    "speech sampling requires a retained random stream",
                ))?;
                let uniform = D::uniform_f32(shape, random, self.context)
                    .map_err(eredu_nn::Error::backend_source)?;
                let mixture = D::gumbel_max(
                    logits,
                    &uniform,
                    GumbelTruncation::TopP(top_p),
                    GUMBEL_EPSILON,
                    self.context,
                )
                .map_err(eredu_nn::Error::backend_source)?;
                let gaussian =
                    D::standard_normal_f32(&[shape[0], shape[1], latent], random, self.context)
                        .map_err(eredu_nn::Error::backend_source)?;
                (mixture, gaussian)
            } else {
                (
                    B::Tensor::full_i32(0, &shape[..2], self.context)?,
                    B::Tensor::full_f32(0., &[shape[0], shape[1], latent], self.context)?,
                )
            };
            validate_draws(&mixture, &gaussian, shape, latent)?;
            Ok((mixture, gaussian))
        })();
        if let Some(coordination) = self.coordination.as_deref_mut() {
            let agreement = coordination.agree(103, &[slot as u64], drawn.is_ok(), self.context);
            let (mixture, gaussian) = drawn?;
            agreement?;
            let mixture = coordination.publish(mixture, self.context)?;
            let gaussian = coordination.publish(gaussian, self.context)?;
            let checked = validate_draws(&mixture, &gaussian, shape, latent);
            let agreement = coordination.agree(104, &[slot as u64], checked.is_ok(), self.context);
            checked?;
            agreement?;
            Ok((mixture, gaussian))
        } else {
            drawn
        }
    }
}

pub(super) fn sampled<B, D, M, Q, P, S, C, O>(
    branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
    needs_draw: bool,
    latent: usize,
    mode: u64,
    mut coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
    context: &D::Context,
    operation: impl FnOnce(&mut dyn RefinementSampling<B>) -> Result<O, TtsError>,
) -> Result<O, SpeechSamplingError>
where
    B: NeuralBackend,
    D: DenseSamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    D::RandomState: Clone,
    D::Error: std::error::Error + Send + Sync + 'static,
    Q: RealtimeScheduleState,
{
    let owner = coordination
        .as_ref()
        .is_none_or(|c| c.coordinator.is_owner());
    let local = if branch.has_submission_completion() {
        Err(SpeechSamplingError::Submitted)
    } else if needs_draw && owner && branch.random_state().is_none() {
        Err(SpeechSamplingError::MissingRandomState)
    } else {
        Ok(())
    };
    if let Some(authority) = coordination.as_deref_mut() {
        let fresh = !authority.consumed;
        authority.consumed = true;
        let agreement = authority.agree(
            100,
            &[mode, u64::from(needs_draw), latent as u64],
            fresh && local.is_ok(),
            context,
        );
        local?;
        agreement?;
    } else {
        local?;
    }
    branch
        .with_random_state_transaction(|random| {
            let mut sampler = Sampler::<B, D> {
                random,
                coordination,
                latent,
                context,
            };
            let output = operation(&mut sampler);
            let agreement = match sampler.coordination.as_deref_mut() {
                Some(authority) => authority.agree(105, &[mode], output.is_ok(), context),
                None => Ok(()),
            };
            // The operation agreement is inside the RNG transaction. A local
            // failure retains its source; peers return the coordinated rejection.
            let output = output?;
            agreement?;
            Ok(output)
        })
        .map_err(|error| match error {
            RealtimeRandomStateTransactionError::Submitted => SpeechSamplingError::Submitted,
            RealtimeRandomStateTransactionError::Operation(error) => {
                SpeechSamplingError::Refinement(error)
            }
        })
}
