//! Per-call destination for the shared sampler's actual scalar frontiers.
//! Mirostat uses one prepaid nested completion in the same construction bank;
//! the existing final SamplingEvent role is still consumed exactly once.
use super::MlxSamplingBackend;
use crate::{
    backend::{
        error::Error, nn::workspace::ResidentCompletionRecipe, random::RandomState,
        submission_recovery::prediction::PredictionRole, MlxCompletion,
    },
    MlxTensor,
};
use eredu_core::{Completion, Submission, TokenFilter};
use eredu_runtime::{PenaltyConfig, SamplingBackend, TokenDomain};
use safemlx::{Array, OperationEvent, OriginalScopeObserver, PreparedResidentGraph, Stream};
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::size_of,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadPhase {
    Constructing,
    ReadingToken,
    TokenRead(u32),
    ReadingProbability,
    Complete,
}

pub(crate) struct OriginalSamplingContext<'a> {
    stream: &'a Stream,
    graph: RefCell<Option<PreparedResidentGraph>>,
    completion: RefCell<Option<MlxCompletion>>,
    random_root: RefCell<Option<Array>>,
    random_slot: RefCell<Option<safemlx::PreparedArrayClone>>,
    token_root: RefCell<Option<Array>>,
    token_slot: RefCell<Option<safemlx::PreparedArrayClone>>,
    phase: Cell<ReadPhase>,
    adaptive: bool,
    readout: Cell<Option<u32>>,
    synchronization: RefCell<Option<crate::backend::runtime::distributed::topology::original_source::control::BoundSamplingSource>>,
    // All arrays, clone shells and the bank retire before their actual custody.
    observer: OriginalScopeObserver,
    role: RefCell<Option<PredictionRole>>,
}
impl<'a> OriginalSamplingContext<'a> {
    pub(crate) fn new(
        recipe: ResidentCompletionRecipe,
        role: PredictionRole,
        observer: OriginalScopeObserver,
        stream: &'a Stream,
        logits: &Array,
    ) -> Result<Self, Error> {
        Self::new_with_source(recipe,role,observer,stream,Some(logits),None)
    }
    pub(crate) fn new_with_source(
        recipe: ResidentCompletionRecipe, role: PredictionRole, observer: OriginalScopeObserver,
        stream: &'a Stream, logits: Option<&Array>,
        synchronization: Option<crate::backend::runtime::distributed::topology::original_source::control::BoundSamplingSource>,
    ) -> Result<Self,Error> {
        let same = role.sampling_recipe().is_some_and(|actual| {
            actual.traversal == recipe.traversal
                && actual.graph == recipe.graph
                && actual.validation_roots == recipe.validation_roots
                && actual.nested_completions == recipe.nested_completions
        });
        let adaptive = recipe.nested_completions == 1;
        if !same
            || recipe.validation_roots != 0
            || recipe.nested_completions > 1
            || !(1..=2).contains(&recipe.traversal.roots())
            || (adaptive && recipe.traversal.roots() != 2)
        {
            return Err(Error::PredictionScopeUnavailable);
        }
        match logits {
            Some(logits) => OperationEvent::validate_traversal_leaf(logits, &observer)?,
            None if synchronization.as_ref().is_some_and(|source| !source.plan().is_sampling_rank()) => {},
            None => return Err(Error::PredictionScopeUnavailable),
        }

        let random_slot = (recipe.traversal.roots() == 2)
            .then(safemlx::PreparedArrayClone::try_prepare_for_inspection)
            .transpose()
            .map_err(Error::OriginalSamplingClone)?;
        // The final event owns the token after Mirostat's probability read.
        // This second raw-H shell is prepared before any Graph constructor.
        let token_slot = adaptive
            .then(safemlx::PreparedArrayClone::try_prepare_for_inspection)
            .transpose()
            .map_err(Error::OriginalSamplingClone)?;
        let mut graph = OperationEvent::prepare_resident_graph(recipe.graph, &observer)?;
        if adaptive {
            let traversal = recipe
                .nested_traversal()
                .ok_or(Error::PredictionScopeUnavailable)?;
            graph.configure_nested_completions(&traversal, 1)?;
        }
        Ok(Self {
            stream,
            graph: RefCell::new(Some(graph)),
            completion: RefCell::new(None),
            random_root: RefCell::new(None),
            random_slot: RefCell::new(random_slot),
            token_root: RefCell::new(None),
            token_slot: RefCell::new(token_slot),
            phase: Cell::new(ReadPhase::Constructing),
            adaptive,
            readout: Cell::new(None),
            synchronization: RefCell::new(synchronization),
            observer,
            role: RefCell::new(Some(role)),
        })
    }
    fn same_scope(&self) -> Result<(), Error> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(&self.observer) {
            return Err(Error::PredictionScopeUnavailable);
        }
        Ok(())
    }
    fn constructing(&self) -> Result<(), Error> {
        if self.phase.get() != ReadPhase::Constructing {
            return Err(Error::PredictionScopeUnavailable);
        }
        self.same_scope()
    }
    fn retain_random(&self, random: Option<&RandomState>) -> Result<(), Error> {
        self.constructing()?;
        let slot = self
            .random_slot
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take();
        match (random, slot) {
            (None, None) => Ok(()),
            (Some(random), Some(mut slot)) => {
                let root = slot
                    .fill_in_original_scope(random.as_array(), &self.observer)
                    .map_err(Error::from)?;
                let previous = self
                    .random_root
                    .try_borrow_mut()
                    .map_err(|_| Error::PredictionScopeReentrant)?
                    .replace(root);
                if previous.is_some() {
                    return Err(Error::PredictionScopeUnavailable);
                }
                Ok(())
            }
            _ => Err(Error::PredictionScopeUnavailable),
        }
    }
    fn read(&self, token: &MlxTensor) -> Result<u32, Error> {
        self.constructing()?;
        self.phase.set(ReadPhase::ReadingToken);
        let random = self
            .random_root
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take();
        if self.adaptive {
            OperationEvent::validate_nested_completion(2)?;
            let random = random.ok_or(Error::PredictionScopeUnavailable)?;
            let mut slot = self
                .token_slot
                .try_borrow_mut()
                .map_err(|_| Error::PredictionScopeReentrant)?
                .take()
                .ok_or(Error::PredictionScopeUnavailable)?;
            let retained_token = slot.fill_in_original_scope(token.as_array(), &self.observer)?;
            // The finite helper suspends and resumes this same bank on every
            // exit. Only successful completion retires its exact Scope records;
            // failed/pending prefixes stay owned by ordinary recovery.
            OperationEvent::complete_nested([&retained_token, &random], self.stream)?;
            // The advanced key escapes to the next Sampling role. Authenticate
            // its completed parent Event here before that role consumes it.
            self.observer.validate_completed_array(&random)?;
            let evaluated = retained_token.completed_in_original_scope(&self.observer)?;
            let values = evaluated
                .try_as_slice::<u32>()
                .map_err(Error::OriginalSamplingData)?;
            if values.len() != 1 {
                return Err(Error::PredictionScopeUnavailable);
            }
            let value = values[0];
            drop(evaluated);
            self.token_root.replace(Some(retained_token));
            self.readout.set(Some(value));
            self.phase.set(ReadPhase::TokenRead(value));
            return Ok(value);
        }
        let source = self.complete(token.as_array(), random)?;
        let evaluated = token
            .as_array()
            .completed_in_original_scope(source.observer())?;
        let values = evaluated
            .try_as_slice::<u32>()
            .map_err(Error::OriginalSamplingData)?;
        if values.len() != 1 {
            return Err(Error::PredictionScopeUnavailable);
        }
        self.readout.set(Some(values[0]));
        self.phase.set(ReadPhase::Complete);
        Ok(values[0])
    }
    fn probability(&self, logits: &MlxTensor, token: u32) -> Result<f32, Error> {
        if !self.adaptive || self.phase.get() != ReadPhase::TokenRead(token) {
            return Err(Error::PredictionScopeUnavailable);
        }
        self.same_scope()?;
        self.phase.set(ReadPhase::ReadingProbability);
        // The token frontier evaluated these processed logits in the parent
        // Sampling role. Detach only its authenticated completed Event before
        // the probability graph crosses into the final SamplingEvent child.
        // This performs no evaluation or progress and preserves all refusals.
        self.observer.validate_completed_array(logits.as_array())?;
        // These are exactly the ordinary softmax/static-index constructors.
        // The same Graph bank still owns their finite reserved destinations.
        let selected = super::backend::token_probability_array(logits, token, self.stream)?;
        let retained_token = self
            .token_root
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take()
            .ok_or(Error::PredictionScopeUnavailable)?;
        let source = self.complete(&selected, Some(retained_token))?;
        let evaluated = selected.completed_in_original_scope(source.observer())?;
        let values = evaluated
            .try_as_slice::<f32>()
            .map_err(Error::OriginalSamplingData)?;
        if values.len() != 1 {
            return Err(Error::PredictionScopeUnavailable);
        }
        self.phase.set(ReadPhase::Complete);
        Ok(values[0])
    }
    fn complete(
        &self,
        output: &Array,
        companion: Option<Array>,
    ) -> Result<crate::backend::adapter::completion::SamplingEventSource, Error> {
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take();
        drop(graph);
        let role = self
            .role
            .try_borrow_mut()
            .map_err(|_| Error::PredictionScopeReentrant)?
            .take()
            .ok_or(Error::PredictionScopeUnavailable)?;
        let completion =
            MlxCompletion::sampling_borrowed_with_scope(output, companion, self.stream, role)?;
        let active = CompletionLoan {
            value: Some(completion),
            destination: &self.completion,
        };
        let completion = active.value.as_ref().expect("submitted sampler completion");
        completion.wait()?;
        let source = completion
            .original_sampling_source()
            .ok_or(Error::PredictionScopeUnavailable)?;
        source.ensure_usable()?;
        Ok(source)
    }
    pub(crate) fn finish(self, output: Array) -> Result<Submission<Array, MlxCompletion>, Error> {
        if self.phase.get() != ReadPhase::Complete
            || self.graph.borrow().is_some()
            || self.role.borrow().is_some()
        {
            return Err(Error::PredictionScopeUnavailable);
        }
        let completion = self
            .completion
            .borrow_mut()
            .take()
            .ok_or(Error::PredictionScopeUnavailable)?;
        let source = completion
            .original_sampling_source()
            .ok_or(Error::PredictionScopeUnavailable)?;
        source.ensure_usable()?;
        source.observer().validate_completed_array(&output)?;
        Ok(Submission { output, completion })
    }
}
struct CompletionLoan<'a> {
    value: Option<MlxCompletion>,
    destination: &'a RefCell<Option<MlxCompletion>>,
}
impl Drop for CompletionLoan<'_> {
    fn drop(&mut self) {
        let previous = self.destination.replace(self.value.take());
        debug_assert!(previous.is_none());
        drop(previous);
    }
}

pub(crate) struct OriginalSamplingBackend<'a>(PhantomData<&'a ()>);
impl<'a> SamplingBackend for OriginalSamplingBackend<'a> {
    type Logits = MlxTensor;
    type Token = MlxTensor;
    type RandomState = RandomState;
    type Context = OriginalSamplingContext<'a>;
    type Error = Error;
    fn error(message: String) -> Error {
        Error::from(MlxSamplingBackend::error(message))
    }
    fn validate_token(
        token: &MlxTensor,
        domain: TokenDomain,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::validate_token(token, domain, context.stream).map_err(Error::from)
    }
    fn scale_temperature(
        logits: &MlxTensor,
        temperature: f32,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::scale_temperature(logits, temperature, context.stream)
            .map_err(Error::from)
    }
    fn apply_penalties(
        logits: &MlxTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_penalties(logits, history, penalties, context.stream)
            .map_err(Error::from)
    }
    fn apply_top_k(
        logits: MlxTensor,
        top_k: i32,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_top_k(logits, top_k, context.stream).map_err(Error::from)
    }
    fn apply_top_p(
        logits: MlxTensor,
        top_p: f32,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_top_p(logits, top_p, context.stream).map_err(Error::from)
    }
    fn apply_min_p(
        logits: MlxTensor,
        min_p: f32,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_min_p(logits, min_p, context.stream).map_err(Error::from)
    }
    fn apply_token_filter(
        logits: &MlxTensor,
        filter: &TokenFilter,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_token_filter(logits, filter, context.stream).map_err(Error::from)
    }
    fn apply_mirostat(
        logits: &MlxTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        MlxSamplingBackend::apply_mirostat(
            logits,
            history,
            penalties,
            temperature,
            mu,
            context.stream,
        )
        .map_err(Error::from)
    }
    fn sample_raw(
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        let mut random = random;
        let output = MlxSamplingBackend::sample_raw(
            logits,
            temperature,
            random.as_deref_mut(),
            context.stream,
        )
        .map_err(Error::from)?;
        context.retain_random(random.as_deref())?;
        Ok(output)
    }
    fn sample_processed(
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        context: &Self::Context,
    ) -> Result<MlxTensor, Error> {
        context.constructing()?;
        let mut random = random;
        let output = MlxSamplingBackend::sample_processed(
            logits,
            temperature,
            random.as_deref_mut(),
            context.stream,
        )
        .map_err(Error::from)?;
        context.retain_random(random.as_deref())?;
        Ok(output)
    }
    fn token_probability(
        logits: &MlxTensor,
        token: u32,
        context: &Self::Context,
    ) -> Result<f32, Error> {
        context.probability(logits, token)
    }
    fn token_id(token: &MlxTensor, context: &Self::Context) -> Result<u32, Error> {
        context.read(token)
    }
}

pub(crate) fn control_bytes() -> Option<u64> {
    let bytes = [
        size_of::<OriginalSamplingContext<'static>>(),
        size_of::<Option<OriginalSamplingContext<'static>>>(),
        size_of::<Result<OriginalSamplingContext<'static>, Error>>(),
        size_of::<Result<Option<OriginalSamplingContext<'static>>, Error>>(),
        size_of::<CompletionLoan<'static>>(),
        size_of::<Result<crate::backend::adapter::completion::SamplingEventSource, Error>>(),
        size_of::<Option<Array>>(),
        size_of::<Option<safemlx::PreparedArrayClone>>(),
        size_of::<Result<safemlx::PreparedArrayClone, safemlx::PreparedArrayCloneCause>>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<std::cell::RefMut<'static, Option<Array>>>(),
        size_of::<std::cell::RefMut<'static, Option<safemlx::PreparedArrayClone>>>(),
        Array::inspection_clone_handle_bytes().checked_mul(2)?,
        safemlx::PreparedArrayClone::control_bytes()?.checked_mul(2)?,
        size_of::<OriginalSamplingBackend<'static>>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<ResidentCompletionRecipe>(),
        size_of::<Option<ResidentCompletionRecipe>>(),
        size_of::<Result<ResidentCompletionRecipe, Error>>(),
        size_of::<Option<PreparedResidentGraph>>(),
        size_of::<Option<PredictionRole>>(),
        size_of::<std::cell::RefMut<'static, Option<PreparedResidentGraph>>>(),
        size_of::<std::cell::RefMut<'static, Option<PredictionRole>>>(),
        size_of::<std::cell::RefMut<'static, Option<MlxCompletion>>>(),
        size_of::<ReadPhase>(),
        size_of::<Result<u32, Error>>(),
        size_of::<Result<f32, Error>>(),
        size_of::<Result<&'static [f32], safemlx::error::AsSliceError>>(),
        OperationEvent::nested_completion_control_bytes::<2>()?,
        size_of::<Result<(), Error>>(),
        size_of::<safemlx::EvaluatedArray<'static>>(),
        size_of::<Result<safemlx::EvaluatedArray<'static>, safemlx::error::Exception>>(),
        size_of::<Result<&'static [u32], safemlx::error::AsSliceError>>(),
        super::backend::filter_control_bytes()?,
        super::backend::penalty_control_bytes()?,
        super::backend::mirostat_control_bytes()?,
        crate::backend::adapter::completion::SamplingEventSource::control_bytes()?
            .try_into()
            .ok()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    u64::try_from(bytes).ok()
}

mod synchronization;
