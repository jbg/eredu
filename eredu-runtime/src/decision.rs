//! Backend-neutral sequential prediction decisions and layered traversal handoff.

use crate::host_metadata::funded_vec;
use std::marker::PhantomData;

use eredu_core::OutputDemand;
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    layered::{LayeredTraversalHook, LayeredTraversalPoint, LayeredUnitAction},
    Sampler, SamplingBackend, TokenDomain,
};

/// How a complete sequential prediction plan obtains its tokens.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SequentialDecisionMode {
    /// Every prediction is supplied by the caller.
    TeacherForced,
    /// Every prediction is selected from backend-native logits.
    Autoregressive,
    /// Forced and sampler-selected predictions are interleaved.
    PartiallyForced,
}

/// Explicit observation contract at sequential decision boundaries.
///
/// Opportunistic tracing receives only values required by another consumer;
/// it does not require execution of a wholly forced tail.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub enum SequentialDecisionObservation {
    /// Preserve tracing of values that execution already produces.
    #[default]
    Opportunistic,
    /// Require a real final-position value, including for forced decisions.
    RequiredLastPosition,
    /// Require every position, including for forced decisions.
    RequiredSequence,
}

impl SequentialDecisionObservation {
    /// Vocabulary demand of the observation itself.
    pub const fn demand(self) -> OutputDemand {
        match self {
            Self::Opportunistic => OutputDemand::StateOnly,
            Self::RequiredLastPosition => OutputDemand::LastPosition,
            Self::RequiredSequence => OutputDemand::Sequence,
        }
    }
}

/// Combines independently required output geometries before projection.
pub const fn merge_output_demand(left: OutputDemand, right: OutputDemand) -> OutputDemand {
    match (left, right) {
        (OutputDemand::Sequence, _) | (_, OutputDemand::Sequence) => OutputDemand::Sequence,
        (OutputDemand::LastPosition, _) | (_, OutputDemand::LastPosition) => {
            OutputDemand::LastPosition
        }
        _ => OutputDemand::StateOnly,
    }
}

/// One prediction's forcing directive.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PredictionDirective<T> {
    /// Select a token from this prediction's logits.
    Sample,
    /// Supply this backend-native token without invoking the sampler.
    Force(T),
}

/// Validated forcing and diagnostic policy for one ordered prediction chain.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SequentialDecisionPlan<T> {
    directives: Vec<PredictionDirective<T>>,
    retain_diagnostics: bool,
    allow_fully_forced_tail_skip: bool,
}

impl<T> SequentialDecisionPlan<T> {
    /// Creates a non-empty ordered prediction plan.
    pub fn new(
        directives: impl IntoIterator<Item = PredictionDirective<T>>,
        retain_diagnostics: bool,
        allow_fully_forced_tail_skip: bool,
    ) -> Result<Self, SequentialDecisionPlanError> {
        Self::from_directives(directives.into_iter().collect(),retain_diagnostics,allow_fully_forced_tail_skip)
    }

    /// Adopts an already constructed ordered destination without another copy.
    pub fn from_directives(directives:Vec<PredictionDirective<T>>,
        retain_diagnostics:bool,allow_fully_forced_tail_skip:bool)
        ->Result<Self,SequentialDecisionPlanError> {
        if directives.is_empty() {
            return Err(SequentialDecisionPlanError::EmptyPlan);
        }
        Ok(Self {
            directives,
            retain_diagnostics,
            allow_fully_forced_tail_skip,
        })
    }

    /// Returns the number of ordered target and predictor decisions.
    pub fn len(&self) -> usize {
        self.directives.len()
    }

    /// Returns whether this plan contains no decisions.
    pub fn is_empty(&self) -> bool {
        self.directives.is_empty()
    }

    /// Returns the aggregate forcing mode.
    pub fn mode(&self) -> SequentialDecisionMode {
        let forced = self
            .directives
            .iter()
            .filter(|directive| matches!(directive, PredictionDirective::Force(_)))
            .count();
        match forced {
            0 => SequentialDecisionMode::Autoregressive,
            count if count == self.directives.len() => SequentialDecisionMode::TeacherForced,
            _ => SequentialDecisionMode::PartiallyForced,
        }
    }

    /// Returns a portable per-prediction forcing mask in decision order.
    pub fn forcing_mask(&self) -> impl ExactSizeIterator<Item = bool> + '_ {
        self.directives
            .iter()
            .map(|directive| matches!(directive, PredictionDirective::Force(_)))
    }

    /// Returns whether diagnostic logits are retained for every executed decision.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }

    /// Returns whether a proven fully forced tail may omit model calls.
    pub const fn allows_fully_forced_tail_skip(&self) -> bool {
        self.allow_fully_forced_tail_skip
    }
}

/// Origin of one resolved sequential token.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SequentialDecisionSource {
    /// The caller forced the token after logits were computed.
    Forced,
    /// A backend-native sampler selected the token.
    Sampled,
    /// The caller forced the token in a tail whose model calls were omitted.
    ForcedTailSkipped,
}

/// One resolved token in prediction order.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SequentialDecision<T> {
    prediction: usize,
    source: SequentialDecisionSource,
    token: T,
}

impl<T> SequentialDecision<T> {
    /// Returns the zero-based prediction ordinal.
    pub const fn prediction(&self) -> usize {
        self.prediction
    }

    /// Returns whether forcing or sampling produced this token.
    pub const fn source(&self) -> SequentialDecisionSource {
        self.source
    }

    /// Borrows the backend-native selected token.
    pub const fn token(&self) -> &T {
        &self.token
    }
}

/// Diagnostic logits retained at one executed decision boundary.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SequentialDecisionDiagnostic<L> {
    prediction: usize,
    logits: L,
}

impl<L> SequentialDecisionDiagnostic<L> {
    /// Returns the zero-based prediction ordinal.
    pub const fn prediction(&self) -> usize {
        self.prediction
    }

    /// Borrows the backend-native logits without host materialization.
    pub const fn logits(&self) -> &L {
        &self.logits
    }
}

/// Validated result of checking whether the unexecuted group tail is skippable.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FullyForcedTailDecision {
    /// At least one remaining unit must execute.
    Execute,
    /// Every remaining unit corresponds exactly to one forced decision.
    Skip {
        /// Number of forced predictions proven safe to omit.
        predictions: usize,
    },
}

/// Statically dispatched sequential token resolver.
///
/// The driver owns one sampler per prediction and one optional backend random
/// state. Forced tokens and diagnostic logits remain backend-native values.
pub struct SequentialDecisionDriver<B, S>
where
    B: SamplingBackend,
    S: Sampler<B>,
{
    plan: SequentialDecisionPlan<B::Token>,
    samplers: Vec<S>,
    temperatures: Vec<f32>,
    random: Option<B::RandomState>,
    decisions: Vec<SequentialDecision<B::Token>>,
    diagnostics: Vec<SequentialDecisionDiagnostic<B::Logits>>,
    // Returned sampling state is adopted by the enclosing funded transaction.
    host_funding: Option<eredu_core::HostMetadataFunding>,
}

/// Sampler instances and optional backend randomness advanced by one decision pass.
pub type SequentialSamplingState<S, R> = (Vec<S>, Option<R>);

impl<B, S> SequentialDecisionDriver<B, S>
where
    B: SamplingBackend,
    S: Sampler<B>,
{
    /// Creates a driver with exactly one sampler and temperature per prediction.
    pub fn new(
        plan: SequentialDecisionPlan<B::Token>,
        samplers: Vec<S>,
        temperatures: Vec<f32>,
        random: Option<B::RandomState>,
    ) -> Result<Self, SequentialDecisionPlanError> {
        Self::new_in(plan,samplers,temperatures,random,None)
    }

    /// Uses the same decision driver with paid result and diagnostic storage.
    /// Supplied policies, tokens and random state already carry their child sources.
    pub fn new_with_host_source(plan:SequentialDecisionPlan<B::Token>,samplers:Vec<S>,
        temperatures:Vec<f32>,random:Option<B::RandomState>,funding:&eredu_core::HostMetadataFunding)
        ->Result<Self,SequentialDecisionPlanError> {
        Self::new_in(plan,samplers,temperatures,random,Some(funding.clone()))
    }

    fn new_in(plan:SequentialDecisionPlan<B::Token>,samplers:Vec<S>,
        temperatures:Vec<f32>,random:Option<B::RandomState>,
        host_funding:Option<eredu_core::HostMetadataFunding>)->Result<Self,SequentialDecisionPlanError> {
        if samplers.len() != plan.len() {
            return Err(SequentialDecisionPlanError::SamplerCountMismatch {
                predictions: plan.len(),
                samplers: samplers.len(),
            });
        }
        if temperatures.len() != plan.len() {
            return Err(SequentialDecisionPlanError::TemperatureCountMismatch {
                predictions: plan.len(),
                temperatures: temperatures.len(),
            });
        }
        if let Some((prediction, temperature)) = temperatures
            .iter()
            .copied()
            .enumerate()
            .find(|(_, temperature)| !temperature.is_finite() || *temperature < 0.0)
        {
            return Err(SequentialDecisionPlanError::InvalidTemperature {
                prediction,
                bits: temperature.to_bits(),
            });
        }
        let (decisions,diagnostics)=if let Some(funding)=&host_funding {
            funding.reserve_metadata(Self::host_header_bytes())?;
            (funded_vec(plan.len(),Some(funding))?,
                funded_vec(if plan.retain_diagnostics {plan.len()}else{0},Some(funding))?)
        } else {(Vec::new(),Vec::new())};
        Ok(Self {plan,samplers,temperatures,random,decisions,diagnostics,host_funding})
    }

    fn clone_token(&self,value:&B::Token,context:&B::Context)->Result<B::Token,SequentialDecisionError<B::Error>> {
        match self.host_funding.as_ref() {
            Some(funding)=>B::clone_token_with_host_source(value,funding,context).map_err(SequentialDecisionError::HostClone),
            None=>Ok(value.clone()),
        }
    }
    pub(crate) fn host_header_bytes()->usize {
        std::mem::size_of::<Self>()*2+std::mem::size_of::<Result<Self,SequentialDecisionPlanError>>()
    }
    pub(crate) fn host_source_bytes(predictions:usize,diagnostics:bool,forced_tail:usize)->Option<usize> {
        use crate::host_metadata::funded_vec_bytes as vector;
        let mut bytes=Self::host_header_bytes()
            .checked_add(vector::<SequentialDecision<B::Token>>(predictions)?)?
            .checked_add(vector::<SequentialDecisionDiagnostic<B::Logits>>(if diagnostics {predictions}else{0})?)?;
        // The existing traversal can accept at most one terminal forced tail.
        if forced_tail!=0 {
            bytes=bytes.checked_add(vector::<TokenDomain>(forced_tail)?.checked_mul(2)?)?
                .checked_add(vector::<B::Token>(forced_tail)?)?;
        }
        Some(bytes)
    }

    /// Borrows the validated decision plan.
    pub const fn plan(&self) -> &SequentialDecisionPlan<B::Token> {
        &self.plan
    }

    /// Returns the next prediction ordinal expected at a traversal boundary.
    pub fn next_prediction(&self) -> usize {
        self.decisions.len()
    }

    /// Returns resolved decisions in canonical order.
    pub fn decisions(&self) -> &[SequentialDecision<B::Token>] {
        &self.decisions
    }

    /// Returns retained diagnostic logits in canonical order.
    pub fn diagnostics(&self) -> &[SequentialDecisionDiagnostic<B::Logits>] {
        &self.diagnostics
    }

    /// Borrows the backend random state after all decisions made so far.
    pub const fn random_state(&self) -> Option<&B::RandomState> {
        self.random.as_ref()
    }

    /// Validates whether `remaining_units` is exactly one fully forced tail.
    pub fn fully_forced_tail_decision(
        &self,
        prediction: usize,
        remaining_units: usize,
    ) -> Result<FullyForcedTailDecision, SequentialDecisionError<B::Error>> {
        self.require_next(prediction)?;
        let remaining_predictions = self.plan.len().saturating_sub(prediction);
        if !self.plan.allow_fully_forced_tail_skip
            || self.plan.retain_diagnostics
            || remaining_predictions != remaining_units
            || !self.plan.directives[prediction..]
                .iter()
                .all(|directive| matches!(directive, PredictionDirective::Force(_)))
        {
            return Ok(FullyForcedTailDecision::Execute);
        }
        Ok(FullyForcedTailDecision::Skip {
            predictions: remaining_predictions,
        })
    }

    /// Returns cloned forced tokens for a tail already proven skippable.
    pub fn forced_tail_tokens(
        &self,
        prediction: usize,
        count: usize,
        domains: impl IntoIterator<Item = TokenDomain>,
        context: &B::Context,
    ) -> Result<Vec<B::Token>, SequentialDecisionError<B::Error>> {
        self.require_next(prediction)?;
        if self.fully_forced_tail_decision(prediction, count)?
            != (FullyForcedTailDecision::Skip { predictions: count })
        {
            return Err(SequentialDecisionError::InvalidTailSkip { prediction, count });
        }
        let mut source=domains.into_iter();
        let mut domains=funded_vec(count,self.host_funding.as_ref())?;
        for domain in source.by_ref() {
            if domains.len()==count {return Err(SequentialDecisionError::TokenDomainCountMismatch {
                prediction,expected:count,actual:count.saturating_add(1)});}
            domains.push(domain);
        }
        if domains.len() != count {
            return Err(SequentialDecisionError::TokenDomainCountMismatch {
                prediction,
                expected: count,
                actual: domains.len(),
            });
        }
        let mut tokens=funded_vec(count,self.host_funding.as_ref())?;
        for (directive,domain) in self.plan.directives[prediction..prediction+count].iter().zip(domains) {
            let token=match directive {
                PredictionDirective::Force(token)=>B::validate_token(token,domain,context)
                    .map_err(SequentialDecisionError::Backend)?,
                PredictionDirective::Sample=>unreachable!("tail was proven fully forced"),
            };
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Records a proven and architecture-accepted forced tail.
    pub fn commit_forced_tail(
        &mut self,
        prediction: usize,
        tokens: Vec<B::Token>,
    ) -> Result<(), SequentialDecisionError<B::Error>> {
        self.require_next(prediction)?;
        let count = tokens.len();
        if self.fully_forced_tail_decision(prediction, count)?
            != (FullyForcedTailDecision::Skip { predictions: count })
        {
            return Err(SequentialDecisionError::InvalidTailSkip { prediction, count });
        }
        self.decisions
            .extend(
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(offset, token)| SequentialDecision {
                        prediction: prediction + offset,
                        source: SequentialDecisionSource::ForcedTailSkipped,
                        token,
                    }),
            );
        Ok(())
    }

    /// Demand of the actual next directive and diagnostic consumer.
    pub fn readout_demand(
        &self,
        prediction: usize,
    ) -> Result<OutputDemand, SequentialDecisionError<B::Error>> {
        self.require_next(prediction)?;
        Ok(
            if self.plan.retain_diagnostics
                || matches!(
                    self.plan.directives[prediction],
                    PredictionDirective::Sample
                )
            {
                OutputDemand::Sequence
            } else {
                OutputDemand::StateOnly
            },
        )
    }

    /// Resolves one executed prediction from backend-native logits.
    pub fn resolve(
        &mut self,
        prediction: usize,
        logits: &B::Logits,
        domain: TokenDomain,
        context: &B::Context,
    ) -> Result<B::Token, SequentialDecisionError<B::Error>> {
        self.resolve_optional(
            prediction,
            Some(logits),
            domain,
            OutputDemand::StateOnly,
            context,
        )
    }

    /// Resolves a decision without manufacturing logits for an unobserved
    /// forced token. Missing required scores fail before sampling or token work.
    pub fn resolve_optional(
        &mut self,
        prediction: usize,
        logits: Option<&B::Logits>,
        domain: TokenDomain,
        observation: OutputDemand,
        context: &B::Context,
    ) -> Result<B::Token, SequentialDecisionError<B::Error>> {
        let demand = merge_output_demand(self.readout_demand(prediction)?, observation);
        if logits.is_none() && demand != OutputDemand::StateOnly {
            return Err(SequentialDecisionError::MissingLogits { prediction });
        }
        if matches!(self.plan.directives[prediction],PredictionDirective::Sample) {
            if let Some(funding)=&self.host_funding {
                self.samplers[prediction].reserve_sample_with_host_source(funding)?;
            }
        }
        let (token, source) = match &self.plan.directives[prediction] {
            PredictionDirective::Force(token) => (self.clone_token(token,context)?, SequentialDecisionSource::Forced),
            PredictionDirective::Sample => (
                self.samplers[prediction]
                    .sample(
                        logits.expect("sampled decisions require logits"),
                        self.temperatures[prediction],
                        self.random.as_mut(),
                        context,
                    )
                    .map_err(SequentialDecisionError::Backend)?,
                SequentialDecisionSource::Sampled,
            ),
        };
        let token =
            B::validate_token(&token, domain, context).map_err(SequentialDecisionError::Backend)?;
        if self.plan.retain_diagnostics {
            self.diagnostics.push(SequentialDecisionDiagnostic {
                prediction,
                logits: match self.host_funding.as_ref() {
                    Some(funding)=>B::clone_logits_with_host_source(logits.expect("diagnostic decisions require logits"),funding,context)
                        .map_err(SequentialDecisionError::HostClone)?,
                    None=>logits.expect("diagnostic decisions require logits").clone(),
                },
            });
        }
        self.decisions.push(SequentialDecision {
            prediction,
            source,
            token: self.clone_token(&token,context)?,
        });
        Ok(token)
    }

    /// Validates that every planned prediction was resolved exactly once.
    pub fn finish(&self) -> Result<(), SequentialDecisionError<B::Error>> {
        if self.decisions.len() != self.plan.len() {
            return Err(SequentialDecisionError::Incomplete {
                resolved: self.decisions.len(),
                predictions: self.plan.len(),
            });
        }
        Ok(())
    }

    /// Finishes the existing decision sequence and returns its advanced
    /// sampler and backend-random states for transactional publication.
    ///
    /// Decisions and diagnostics remain inspectable until this method consumes
    /// the driver. A caller can therefore copy any required output metadata
    /// before atomically adopting these state components.
    pub fn finish_into_sampling_state(
        self,
    ) -> Result<SequentialSamplingState<S, B::RandomState>, SequentialDecisionError<B::Error>> {
        self.finish()?;
        Ok((self.samplers, self.random))
    }

    fn require_next(&self, prediction: usize) -> Result<(), SequentialDecisionError<B::Error>> {
        if prediction != self.decisions.len() || prediction >= self.plan.len() {
            return Err(SequentialDecisionError::OutOfOrder {
                expected: self.decisions.len(),
                actual: prediction,
                predictions: self.plan.len(),
            });
        }
        Ok(())
    }
}

/// Architecture-owned conversion between layered boundaries and predictions.
pub trait SequentialDecisionBoundary<B, C, E>
where
    B: SamplingBackend,
{
    /// Returns the prediction ordinal at this traversal point, if any.
    fn prediction_at(&self, point: LayeredTraversalPoint, forward: &C) -> Option<usize>;

    /// Installs merged demand before this boundary's unit/group executes.
    fn prepare_logits(
        &mut self,
        _prediction: usize,
        _point: LayeredTraversalPoint,
        _demand: OutputDemand,
        _forward: &mut C,
        _context: &B::Context,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Borrows an optional projection produced by a demand-aware boundary.
    /// Existing boundaries continue producing their ordinary scores.
    fn optional_logits(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        value: &B::Logits,
        forward: &mut C,
        context: &B::Context,
    ) -> Result<Option<B::Logits>, E> {
        self.logits(prediction, point, value, forward, context)
            .map(Some)
    }

    /// Produces backend-native logits for one target or predictor boundary.
    fn logits(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        value: &B::Logits,
        forward: &mut C,
        context: &B::Context,
    ) -> Result<B::Logits, E>;

    /// Returns the exact accepted token-id domain for this prediction.
    fn token_domain(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        forward: &C,
    ) -> Result<TokenDomain, E>;

    /// Supplies a forced or sampled backend-native token to subsequent units.
    fn accept(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        token: &B::Token,
        forward: &mut C,
        context: &B::Context,
    ) -> Result<(), E>;

    /// Converts a generic decision failure into the architecture error type.
    fn decision_error(&mut self, error: SequentialDecisionError<B::Error>) -> E;
}

/// Adapter that drives sequential decisions from shared layered traversal hooks.
pub struct SequentialDecisionTraversal<'a, B, S, D, C, E>
where
    B: SamplingBackend,
    S: Sampler<B>,
    D: SequentialDecisionBoundary<B, C, E>,
{
    driver: &'a mut SequentialDecisionDriver<B, S>,
    boundary: &'a mut D,
    observation: SequentialDecisionObservation,
    marker: PhantomData<fn(C) -> E>,
}

impl<'a, B, S, D, C, E> SequentialDecisionTraversal<'a, B, S, D, C, E>
where
    B: SamplingBackend,
    S: Sampler<B>,
    D: SequentialDecisionBoundary<B, C, E>,
{
    /// Couples a decision driver to one architecture-owned boundary mapping.
    pub fn new(driver: &'a mut SequentialDecisionDriver<B, S>, boundary: &'a mut D) -> Self {
        Self {
            driver,
            boundary,
            observation: SequentialDecisionObservation::Opportunistic,
            marker: PhantomData,
        }
    }

    /// Requires actual decision observations independently of tracing geometry.
    pub const fn with_observation(mut self, observation: SequentialDecisionObservation) -> Self {
        self.observation = observation;
        self
    }

    fn prepare(
        &mut self,
        point: LayeredTraversalPoint,
        forward: &mut C,
        context: &B::Context,
    ) -> Result<(), E> {
        let Some(prediction) = self.boundary.prediction_at(point, forward) else {
            return Ok(());
        };
        let demand = self
            .driver
            .readout_demand(prediction)
            .map_err(|error| self.boundary.decision_error(error))?;
        self.boundary.prepare_logits(
            prediction,
            point,
            merge_output_demand(demand, self.observation.demand()),
            forward,
            context,
        )
    }

    fn process(
        &mut self,
        point: LayeredTraversalPoint,
        value: &B::Logits,
        forward: &mut C,
        context: &B::Context,
    ) -> Result<(), E> {
        let Some(prediction) = self.boundary.prediction_at(point, forward) else {
            return Ok(());
        };
        let logits = self
            .boundary
            .optional_logits(prediction, point, value, forward, context)?;
        let domain = self.boundary.token_domain(prediction, point, forward)?;
        let token = self
            .driver
            .resolve_optional(
                prediction,
                logits.as_ref(),
                domain,
                self.observation.demand(),
                context,
            )
            .map_err(|error| self.boundary.decision_error(error))?;
        self.boundary
            .accept(prediction, point, &token, forward, context)
    }
}

impl<NB, B, S, D, C, E> LayeredTraversalHook<NB, C, E>
    for SequentialDecisionTraversal<'_, B, S, D, C, E>
where
    NB: NeuralBackend,
    B: SamplingBackend<Logits = NB::Tensor, Context = <NB::Tensor as Tensor>::Context>,
    S: Sampler<B>,
    D: SequentialDecisionBoundary<B, C, E>,
{
    fn after_group_begin(
        &mut self,
        group: usize,
        _value: &mut NB::Tensor,
        forward: &mut C,
        context: &<NB::Tensor as Tensor>::Context,
    ) -> Result<(), E> {
        self.prepare(LayeredTraversalPoint::Group { group }, forward, context)
    }

    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        remaining_units: usize,
        _value: &mut NB::Tensor,
        forward: &mut C,
        context: &<NB::Tensor as Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        let point = LayeredTraversalPoint::Unit { group, index };
        let Some(prediction) = self.boundary.prediction_at(point, forward) else {
            return Ok(LayeredUnitAction::Execute);
        };
        self.prepare(point, forward, context)?;
        if self.observation != SequentialDecisionObservation::Opportunistic {
            return Ok(LayeredUnitAction::Execute);
        }
        let tail = self
            .driver
            .fully_forced_tail_decision(prediction, remaining_units)
            .map_err(|error| self.boundary.decision_error(error))?;
        let FullyForcedTailDecision::Skip { predictions } = tail else {
            return Ok(LayeredUnitAction::Execute);
        };
        let mut domains = funded_vec(predictions,self.driver.host_funding.as_ref())
            .map_err(|cause|self.boundary.decision_error(SequentialDecisionError::HostMetadata(cause)))?;
        for offset in 0..predictions {
            let skipped_point = LayeredTraversalPoint::Unit {
                group,
                index: index + offset,
            };
            let actual = self.boundary.prediction_at(skipped_point, forward);
            if actual != Some(prediction + offset) {
                let error = SequentialDecisionError::TailBoundaryMismatch {
                    expected: prediction + offset,
                    actual,
                };
                return Err(self.boundary.decision_error(error));
            }
            domains.push(self.boundary.token_domain(
                prediction + offset,
                skipped_point,
                forward,
            )?);
        }
        let tokens = self
            .driver
            .forced_tail_tokens(prediction, predictions, domains, context)
            .map_err(|error| self.boundary.decision_error(error))?;
        for (offset, token) in tokens.iter().enumerate() {
            let skipped_point = LayeredTraversalPoint::Unit {
                group,
                index: index + offset,
            };
            self.boundary
                .accept(prediction + offset, skipped_point, token, forward, context)?;
        }
        self.driver
            .commit_forced_tail(prediction, tokens)
            .map_err(|error| self.boundary.decision_error(error))?;
        Ok(LayeredUnitAction::SkipRemainingGroup)
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut NB::Tensor,
        forward: &mut C,
        context: &<NB::Tensor as Tensor>::Context,
    ) -> Result<(), E> {
        self.process(
            LayeredTraversalPoint::Unit { group, index },
            value,
            forward,
            context,
        )
    }

    fn after_group(
        &mut self,
        group: usize,
        value: &mut NB::Tensor,
        forward: &mut C,
        context: &<NB::Tensor as Tensor>::Context,
    ) -> Result<(), E> {
        self.process(
            LayeredTraversalPoint::Group { group },
            value,
            forward,
            context,
        )
    }
}

/// Invalid construction of a sequential decision plan or driver.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SequentialDecisionPlanError {
    /// Exact host source refused construction before allocation.
    #[error(transparent)]
    HostMetadata(#[from] eredu_core::HostMetadataFundingError),
    /// No target or predictor decisions were declared.
    #[error("sequential decision plan must contain at least one prediction")]
    EmptyPlan,
    /// Sampler state cardinality did not match prediction cardinality.
    #[error("sequential decision plan has {predictions} predictions but {samplers} samplers")]
    SamplerCountMismatch {
        /// Planned prediction count.
        predictions: usize,
        /// Supplied sampler count.
        samplers: usize,
    },
    /// Temperature cardinality did not match prediction cardinality.
    #[error(
        "sequential decision plan has {predictions} predictions but {temperatures} temperatures"
    )]
    TemperatureCountMismatch {
        /// Planned prediction count.
        predictions: usize,
        /// Supplied temperature count.
        temperatures: usize,
    },
    /// One sampling temperature was negative or non-finite.
    #[error(
        "sequential decision temperature at prediction {prediction} is invalid (bits {bits:#010x})"
    )]
    InvalidTemperature {
        /// Invalid prediction ordinal.
        prediction: usize,
        /// Exact invalid floating-point bits.
        bits: u32,
    },
}

/// Failure while resolving an ordered sequential decision chain.
#[derive(Debug, thiserror::Error)]
pub enum SequentialDecisionError<E> {
    /// The exact retained native descriptor copy failed.
    #[error(transparent)]
    HostClone(eredu_core::BackendFailure),
    /// Exact host source refused the next actual decision destination.
    #[error(transparent)]
    HostMetadata(#[from] eredu_core::HostMetadataFundingError),
    /// An actual sampler, diagnostic, or required observation lacked scores.
    #[error("sequential decision {prediction} requires logits")]
    MissingLogits {
        /// Prediction rejected before any sampling or token validation.
        prediction: usize,
    },
    /// The sampler or sampling backend failed.
    #[error("sequential decision backend failed: {0}")]
    Backend(E),
    /// A traversal boundary did not match the next prediction.
    #[error(
        "sequential decision expected prediction {expected} of {predictions}, received {actual}"
    )]
    OutOfOrder {
        /// Next required prediction.
        expected: usize,
        /// Traversal-supplied prediction.
        actual: usize,
        /// Total prediction count.
        predictions: usize,
    },
    /// A requested tail skip was not proven safe by the plan.
    #[error(
        "prediction tail beginning at {prediction} with length {count} is not safely skippable"
    )]
    InvalidTailSkip {
        /// First proposed skipped prediction.
        prediction: usize,
        /// Proposed skipped prediction count.
        count: usize,
    },
    /// Architecture token-domain cardinality drifted across a skipped tail.
    #[error(
        "prediction tail beginning at {prediction} has {actual} token domains, expected {expected}"
    )]
    TokenDomainCountMismatch {
        /// First proposed skipped prediction.
        prediction: usize,
        /// Expected domain count.
        expected: usize,
        /// Supplied domain count.
        actual: usize,
    },
    /// Architecture boundary mapping disagreed within a proposed skipped tail.
    #[error("forced-tail boundary expected prediction {expected}, got {actual:?}")]
    TailBoundaryMismatch {
        /// Expected prediction ordinal.
        expected: usize,
        /// Architecture-reported ordinal.
        actual: Option<usize>,
    },
    /// Traversal completed before resolving every prediction.
    #[error("sequential decision traversal resolved {resolved} of {predictions} predictions")]
    Incomplete {
        /// Resolved prediction count.
        resolved: usize,
        /// Planned prediction count.
        predictions: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PenaltyConfig;
    use eredu_core::TokenFilter;

    struct Backend;

    impl SamplingBackend for Backend {
        type Logits = i32;
        type Token = i32;
        type RandomState = i32;
        type Context = ();
        type Error = String;

        fn error(message: String) -> Self::Error {
            message
        }

        fn validate_token(
            token: &Self::Token,
            domain: TokenDomain,
            _: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            usize::try_from(*token)
                .ok()
                .filter(|token| *token < domain.cardinality())
                .map(|_| *token)
                .ok_or_else(|| "token is outside its decision domain".into())
        }

        fn scale_temperature(
            logits: &Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_penalties(
            logits: &Self::Logits,
            _: &[u32],
            _: PenaltyConfig,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_top_k(
            logits: Self::Logits,
            _: i32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_top_p(
            logits: Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_min_p(
            logits: Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_token_filter(
            logits: &Self::Logits,
            _: &TokenFilter,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_mirostat(
            logits: &Self::Logits,
            _: &[u32],
            _: PenaltyConfig,
            _: f32,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn sample_raw(
            logits: &Self::Logits,
            _: f32,
            random: Option<&mut Self::RandomState>,
            _: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            if let Some(random) = random {
                *random += 1;
            }
            Ok(*logits)
        }

        fn sample_processed(
            logits: &Self::Logits,
            temperature: f32,
            random: Option<&mut Self::RandomState>,
            context: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            Self::sample_raw(logits, temperature, random, context)
        }

        fn token_id(token: &Self::Token, _: &Self::Context) -> Result<u32, Self::Error> {
            u32::try_from(*token).map_err(|error| error.to_string())
        }

        fn token_probability(
            _: &Self::Logits,
            _: u32,
            _: &Self::Context,
        ) -> Result<f32, Self::Error> {
            Ok(1.0)
        }
    }

    #[derive(Clone)]
    struct OffsetSampler(i32);

    impl Sampler<Backend> for OffsetSampler {
        fn sample(
            &mut self,
            logits: &i32,
            _: f32,
            random: Option<&mut i32>,
            _: &(),
        ) -> Result<i32, String> {
            if let Some(random) = random {
                *random += 1;
            }
            Ok(*logits + self.0)
        }
    }

    #[test]
    fn teacher_forcing_sampling_masks_and_diagnostics_share_one_driver() {
        let plan = SequentialDecisionPlan::new(
            [
                PredictionDirective::Force(7),
                PredictionDirective::Sample,
                PredictionDirective::Force(9),
            ],
            true,
            true,
        )
        .unwrap();
        assert_eq!(plan.mode(), SequentialDecisionMode::PartiallyForced);
        assert_eq!(plan.forcing_mask().collect::<Vec<_>>(), [true, false, true]);
        let mut driver = SequentialDecisionDriver::<Backend, _>::new(
            plan,
            vec![OffsetSampler(100), OffsetSampler(10), OffsetSampler(100)],
            vec![0.0; 3],
            Some(4),
        )
        .unwrap();

        let domain = TokenDomain::new(100);
        assert_eq!(driver.resolve(0, &1, domain, &()).unwrap(), 7);
        assert_eq!(driver.resolve(1, &2, domain, &()).unwrap(), 12);
        assert_eq!(driver.resolve(2, &3, domain, &()).unwrap(), 9);
        driver.finish().unwrap();
        assert_eq!(driver.random_state(), Some(&5));
        assert_eq!(
            driver
                .decisions()
                .iter()
                .map(|decision| (decision.source(), *decision.token()))
                .collect::<Vec<_>>(),
            [
                (SequentialDecisionSource::Forced, 7),
                (SequentialDecisionSource::Sampled, 12),
                (SequentialDecisionSource::Forced, 9),
            ]
        );
        assert_eq!(
            driver
                .diagnostics()
                .iter()
                .map(|diagnostic| (diagnostic.prediction(), *diagnostic.logits()))
                .collect::<Vec<_>>(),
            [(0, 1), (1, 2), (2, 3)]
        );
    }

    #[test]
    fn fully_forced_tail_skip_requires_exact_cardinality_and_no_diagnostics() {
        let plan = SequentialDecisionPlan::new(
            [
                PredictionDirective::Sample,
                PredictionDirective::Force(8),
                PredictionDirective::Force(9),
            ],
            false,
            true,
        )
        .unwrap();
        let mut driver = SequentialDecisionDriver::<Backend, _>::new(
            plan,
            vec![OffsetSampler(1); 3],
            vec![0.0; 3],
            None,
        )
        .unwrap();
        driver.resolve(0, &4, TokenDomain::new(100), &()).unwrap();
        assert_eq!(
            driver.fully_forced_tail_decision(1, 1).unwrap(),
            FullyForcedTailDecision::Execute
        );
        assert_eq!(
            driver.fully_forced_tail_decision(1, 2).unwrap(),
            FullyForcedTailDecision::Skip { predictions: 2 }
        );
        let tokens = driver
            .forced_tail_tokens(1, 2, [TokenDomain::new(100); 2], &())
            .unwrap();
        driver.commit_forced_tail(1, tokens).unwrap();
        driver.finish().unwrap();
        assert_eq!(
            driver
                .decisions()
                .iter()
                .map(SequentialDecision::source)
                .collect::<Vec<_>>(),
            [
                SequentialDecisionSource::Sampled,
                SequentialDecisionSource::ForcedTailSkipped,
                SequentialDecisionSource::ForcedTailSkipped,
            ]
        );
        assert!(driver.diagnostics().is_empty());

        let diagnostic_plan = SequentialDecisionPlan::new(
            [PredictionDirective::Force(1), PredictionDirective::Force(2)],
            true,
            true,
        )
        .unwrap();
        let diagnostic = SequentialDecisionDriver::<Backend, _>::new(
            diagnostic_plan,
            vec![OffsetSampler(0); 2],
            vec![0.0; 2],
            None,
        )
        .unwrap();
        assert_eq!(
            diagnostic.fully_forced_tail_decision(0, 2).unwrap(),
            FullyForcedTailDecision::Execute
        );
    }

    #[test]
    fn driver_rejects_cardinality_temperature_and_order_drift() {
        let plan =
            SequentialDecisionPlan::new([PredictionDirective::Sample], false, false).unwrap();
        assert!(matches!(
            SequentialDecisionDriver::<Backend, _>::new(
                plan.clone(),
                Vec::<OffsetSampler>::new(),
                vec![0.0],
                None
            ),
            Err(SequentialDecisionPlanError::SamplerCountMismatch { .. })
        ));
        assert!(matches!(
            SequentialDecisionDriver::<Backend, _>::new(
                plan.clone(),
                vec![OffsetSampler(0)],
                vec![f32::NAN],
                None
            ),
            Err(SequentialDecisionPlanError::InvalidTemperature { .. })
        ));
        let mut driver = SequentialDecisionDriver::<Backend, _>::new(
            plan,
            vec![OffsetSampler(0)],
            vec![0.0],
            None,
        )
        .unwrap();
        assert!(matches!(
            driver.resolve(1, &0, TokenDomain::new(100), &()),
            Err(SequentialDecisionError::OutOfOrder { .. })
        ));
    }

    #[test]
    fn token_domains_reject_sampled_executed_forcing_and_skipped_forcing_before_commit() {
        let domain = TokenDomain::new(10);

        let forced_plan =
            SequentialDecisionPlan::new([PredictionDirective::Force(10)], false, false).unwrap();
        let mut forced = SequentialDecisionDriver::<Backend, _>::new(
            forced_plan,
            vec![OffsetSampler(0)],
            vec![0.0],
            None,
        )
        .unwrap();
        assert!(matches!(
            forced.resolve(0, &0, domain, &()),
            Err(SequentialDecisionError::Backend(_))
        ));
        assert!(forced.decisions().is_empty());

        let sampled_plan =
            SequentialDecisionPlan::new([PredictionDirective::Sample], false, false).unwrap();
        let mut sampled = SequentialDecisionDriver::<Backend, _>::new(
            sampled_plan,
            vec![OffsetSampler(10)],
            vec![0.0],
            Some(4),
        )
        .unwrap();
        assert!(matches!(
            sampled.resolve(0, &0, domain, &()),
            Err(SequentialDecisionError::Backend(_))
        ));
        assert!(sampled.decisions().is_empty());

        let tail_plan = SequentialDecisionPlan::new(
            [
                PredictionDirective::Force(1),
                PredictionDirective::Force(10),
            ],
            false,
            true,
        )
        .unwrap();
        let tail = SequentialDecisionDriver::<Backend, _>::new(
            tail_plan,
            vec![OffsetSampler(0); 2],
            vec![0.0; 2],
            None,
        )
        .unwrap();
        assert!(matches!(
            tail.forced_tail_tokens(0, 2, [domain; 2], &()),
            Err(SequentialDecisionError::Backend(_))
        ));
        assert!(tail.decisions().is_empty());
    }
    #[test]
    fn absent_scores_reject_before_sampling_and_only_unobserved_forcing_can_proceed() {
        for (directive, diagnostics, observation) in [
            (PredictionDirective::Sample, false, OutputDemand::StateOnly),
            (PredictionDirective::Force(7), true, OutputDemand::StateOnly),
            (
                PredictionDirective::Force(7),
                false,
                OutputDemand::LastPosition,
            ),
            (PredictionDirective::Force(7), false, OutputDemand::Sequence),
        ] {
            let plan = SequentialDecisionPlan::new([directive], diagnostics, true).unwrap();
            let mut driver = SequentialDecisionDriver::<Backend, _>::new(
                plan,
                vec![OffsetSampler(0)],
                vec![0.0],
                Some(19),
            )
            .unwrap();
            assert!(matches!(
                driver.resolve_optional(0, None, TokenDomain::new(10), observation, &()),
                Err(SequentialDecisionError::MissingLogits { prediction: 0 })
            ));
            assert_eq!(driver.random_state(), Some(&19));
            assert!(driver.decisions().is_empty() && driver.diagnostics().is_empty());
            assert!(driver.finish().is_err());
            driver
                .resolve_optional(0, Some(&3), TokenDomain::new(10), observation, &())
                .unwrap();
            driver.finish().unwrap();
        }
        let plan =
            SequentialDecisionPlan::new([PredictionDirective::Force(7)], false, false).unwrap();
        let mut driver = SequentialDecisionDriver::<Backend, _>::new(
            plan,
            vec![OffsetSampler(0)],
            vec![0.0],
            Some(19),
        )
        .unwrap();
        assert_eq!(
            driver
                .resolve_optional(0, None, TokenDomain::new(10), OutputDemand::StateOnly, &())
                .unwrap(),
            7
        );
        assert_eq!(driver.random_state(), Some(&19));
        assert!(driver.diagnostics().is_empty());
        driver.finish().unwrap();
    }
}
