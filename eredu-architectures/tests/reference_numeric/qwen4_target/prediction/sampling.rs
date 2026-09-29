//! Greedy numerical sampling, with explicit test-only draft interventions.
use super::*;
use eredu_core::generation::{FinishReason, GenerationCancellationToken};
use eredu_core::{
    SamplingPlacement, SpeculativeConstraint, SpeculativeDraftRandomPosition,
    SpeculativeOutputError, SpeculativePublisher, SpeculativeSampling,
};
#[derive(Clone, Default)]
pub(super) struct Sampling {
    pub drafts: Option<(Vec<u32>, bool)>,
    pub forced: Option<(u32, usize)>,
}

impl SpeculativeSampling for Sampling {
    type Logits = NumericTensor;
    type Distribution = u32;
    type Seed = ();
    type RandomState = usize;
    type DraftRandomness = usize;
    type RandomnessRoot = usize;
    type Context<'a> = &'a NumericContext;
    type Error = Error;

    fn control_force_next(
        &mut self,
        token: u32,
        vocabulary: usize,
        position: usize,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        use eredu_core::speculative::SpeculativeControlError as E;
        if token as usize >= vocabulary {
            return Err(E::InvalidToken(token));
        }
        if self.forced.is_some() {
            return Err(E::PendingToken);
        }
        self.forced = Some((token, position));
        Ok(())
    }
    fn control_pending_forced(&self) -> Option<u32> {
        self.forced.map(|(token, _)| token)
    }
    fn control_clear_forced(&mut self) -> bool {
        self.forced.take().is_some()
    }
    fn control_snapshot_bytes(&self, _: Option<&usize>, _: Option<&usize>) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64)
            .checked_add(16)?
            .checked_add(
                self.drafts
                    .as_ref()
                    .map_or(0, |(tokens, _)| tokens.capacity() as u64 * 4),
            )
    }
    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn randomness_root<'a>(_: Option<()>, _: &'a NumericContext) -> Result<usize, Error>
    where
        Self: 'a,
    {
        Ok(0)
    }

    fn target_randomness_from_root<'a>(
        root: &mut usize,
        _: &'a NumericContext,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        let value = *root;
        *root += 1;
        Ok(value)
    }

    fn draft_randomness_from_root<'a>(
        root: &mut usize,
        _: &'a NumericContext,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        let value = *root;
        *root += 1;
        Ok(value)
    }

    fn draft_randomness_at<'a>(
        root: &usize,
        position: SpeculativeDraftRandomPosition,
        _: &'a NumericContext,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        Ok(*root + position.get())
    }

    fn process_logits<'a>(
        &mut self,
        logits: &NumericTensor,
        _: f32,
        history: &[u32],
        placement: SamplingPlacement,
        _: &'a NumericContext,
    ) -> Result<u32, Error>
    where
        Self: 'a,
    {
        if placement == SamplingPlacement::Target {
            if let Some((token, position)) = self.forced {
                if position == history.len() {
                    return Ok(token);
                }
            }
        }
        if let (SamplingPlacement::Draft, Some((expected, reject))) = (placement, &self.drafts) {
            let next = expected[history.len()];
            Ok(if *reject { (next + 1) % 32 } else { next })
        } else {
            Ok(argmax(logits))
        }
    }

    fn sample<'a>(
        &self,
        distribution: &u32,
        _: f32,
        _: Option<&mut usize>,
        _: SamplingPlacement,
        _: &'a NumericContext,
    ) -> Result<u32, Error>
    where
        Self: 'a,
    {
        Ok(*distribution)
    }

    fn probability_at<'a>(
        &self,
        distribution: &u32,
        token: u32,
        _: SamplingPlacement,
        _: &'a NumericContext,
    ) -> Result<f32, Error>
    where
        Self: 'a,
    {
        Ok(if *distribution == token { 1.0 } else { 0.0 })
    }

    fn sample_unit_interval<'a>(
        &self,
        _: Option<&mut usize>,
        _: &'a NumericContext,
    ) -> Result<f32, Error>
    where
        Self: 'a,
    {
        Ok(0.5)
    }

    fn positive_probability_difference<'a>(
        &self,
        target: &u32,
        _: &u32,
        _: SamplingPlacement,
        _: &'a NumericContext,
    ) -> Result<Option<u32>, Error>
    where
        Self: 'a,
    {
        Ok(Some(*target))
    }

    fn update_sampler_state<'a>(
        &mut self,
        _: &u32,
        token: u32,
        placement: SamplingPlacement,
        _: &'a NumericContext,
    ) -> Result<(), Error>
    where
        Self: 'a,
    {
        if placement == SamplingPlacement::Target
            && self.forced.is_some_and(|(forced, _)| token == forced)
        {
            self.forced = None;
        }
        Ok(())
    }
}

pub(super) fn argmax(logits: &NumericTensor) -> u32 {
    assert_eq!(logits.shape[..2], [1, 1]);
    assert!(logits
        .data
        .iter()
        .all(|n| n.is_finite() || *n == f32::NEG_INFINITY));
    assert!(logits.data.iter().any(|n| n.is_finite()));
    logits
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}
#[derive(Clone)]
pub(super) struct Constraint;
impl SpeculativeConstraint for Constraint {
    fn control_snapshot_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        Ok(self.clone())
    }
    fn push_token(&mut self, _: u32) -> Result<bool, SpeculativeOutputError> {
        Ok(false)
    }
    fn finish(&mut self, _: FinishReason) -> Result<(), SpeculativeOutputError> {
        Ok(())
    }
}
pub(super) struct Publisher(pub std::rc::Rc<RefCell<Vec<u32>>>);
impl SpeculativePublisher<Constraint> for Publisher {
    fn publish_committed(
        &mut self,
        _: &mut Constraint,
        tokens: &[u32],
        _: &GenerationCancellationToken,
        _: bool,
    ) -> Result<bool, SpeculativeOutputError> {
        self.0.borrow_mut().extend_from_slice(tokens);
        Ok(false)
    }
    fn publish_cancelled(&mut self, _: &mut Constraint) -> Result<(), SpeculativeOutputError> {
        Ok(())
    }
}
