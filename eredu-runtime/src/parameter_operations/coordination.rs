//! Live parameter operations with separately admitted rejection/commit control.
use super::*;
use crate::CommunicationSessionIdentity;
use eredu_core::{
    consensus::BoundedConsensusTransport,
    parameters::{
        ParameterCoordinationError as Error, ParameterCoordinationStage as Stage,
        ParameterCoordinationUsage,
    },
    BackendFailure, BoundedCompletion, BoundedCompletionWait, BoundedSubmissionOutcome, Completion,
};
use sha2::{Digest, Sha256};
use std::sync::{Mutex, MutexGuard, TryLockError};

const MAGIC: u32 = 0x4552_504f;
const VERSION: u32 = 1;
/// Fixed control words, including exact setup and operation digests.
pub const PARAMETER_CONTROL_WORDS: usize = 38;
// Admission, preparation, publication and possible restoration; every decision
// has a second completed-validation exchange. Unused allowance is not refunded.
const CONTROL_ROUNDS: u64 = 8;
const PAYLOAD_HEADER: usize = 8;

/// Ordinary native portable-word transport and exact retained session facts.
pub trait ParameterOperationTransport: BoundedConsensusTransport {
    /// Actual world rank, not an application-supplied label.
    fn parameter_rank(&self) -> usize;
    /// Actual retained setup identity.
    fn parameter_setup(&self) -> CommunicationSessionIdentity;
    /// Selected bounded completion policy, queried without submission.
    fn parameter_wait(&self) -> Result<BoundedCompletionWait, ParameterError>;
    /// Exact conservative native/host bound for one equally sized gather.
    /// Runtime adds its framing and decoding buffers separately.
    fn estimate_parameter_gather(&self, local_words: usize)
        -> Result<CaptureUsage, ParameterError>;
    /// Rejects a failed native session before new work.
    fn ensure_parameter_active(&self) -> Result<(), BackendFailure>;
    /// Fences the shared execution on ambiguous failure. Exact submitted native
    /// completions still retain or quarantine resources independently.
    fn fail_parameter_operation(&self, error: &Error);
}

/// Common operation category. Local immutable admission remains separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterOperationKind {
    /// Prepared loaded metadata, before publishing discovery.
    Catalog,
    /// Selected effective entries.
    Query,
    /// Bounded native contractions.
    Projection,
    /// Complete coordinated edit activation.
    Activation,
    /// Restoration of an active edit.
    Removal,
}
impl ParameterOperationKind {
    fn word(self) -> u32 {
        match self {
            Self::Catalog => 0,
            Self::Query => 1,
            Self::Projection => 2,
            Self::Activation => 3,
            Self::Removal => 4,
        }
    }
    fn transaction(self) -> bool {
        matches!(self, Self::Activation | Self::Removal)
    }
}

/// Descriptive common loaded-state binding; this value alone grants no authority.
/// Composition must supply retained source/execution/branch/version facts and
/// separately validate its local immutable parameter plan and idle session lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterOperationBinding {
    digest: [u8; 32],
    catalog_only: bool,
}
impl ParameterOperationBinding {
    /// Binds a common branch and current parameter version to exact loaded facts.
    pub fn new(
        artifact: &str,
        execution: &str,
        branch: &str,
        version: u64,
    ) -> Result<Self, ParameterError> {
        let mut digest = Sha256::new();
        digest.update(b"eredu-parameter-operation-binding-v1\0");
        for value in [artifact, execution, branch] {
            if value.is_empty() || value.len() > 4096 {
                return Err(Error::Admission("loaded binding identity").into());
            }
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(version.to_le_bytes());
        Ok(Self {
            digest: digest.finalize().into(),
            catalog_only: false,
        })
    }
}

/// A distinct loaded model within the actual retained communication setup.
/// Issued once by the shared coordinator at model construction, in common
/// construction order. Model reset, snapshots and retries retain this value.
/// It is descriptive identity, not native submission or edit authority.
#[derive(Debug)]
pub struct ParameterModelIdentity {
    setup: CommunicationSessionIdentity,
    ordinal: u64,
}
impl ParameterModelIdentity {
    /// Begins loaded metadata discovery before fallible source identity hashing.
    /// This binding admits catalogue exchange only; reads and edits require the
    /// fully resolved artifact/execution binding returned by `binding`.
    pub fn catalog_binding(&self, version: u64) -> ParameterOperationBinding {
        let mut digest = Sha256::new();
        digest.update(b"eredu-parameter-model-catalog-v1\0");
        digest.update(self.setup.bytes());
        digest.update(self.ordinal.to_le_bytes());
        digest.update(version.to_le_bytes());
        ParameterOperationBinding {
            digest: digest.finalize().into(),
            catalog_only: true,
        }
    }
    /// Binds current loaded facts to this particular model instance. Different
    /// models with identical weights and versions still produce different bindings.
    pub fn binding(
        &self,
        artifact: &str,
        execution: &str,
        version: u64,
    ) -> Result<ParameterOperationBinding, ParameterError> {
        let mut digest = Sha256::new();
        digest.update(b"eredu-parameter-model-binding-v1\0");
        digest.update(self.setup.bytes());
        digest.update(self.ordinal.to_le_bytes());
        for value in [artifact, execution] {
            if value.is_empty() || value.len() > 4096 {
                return Err(Error::Admission("loaded model binding identity").into());
            }
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        digest.update(version.to_le_bytes());
        Ok(ParameterOperationBinding {
            digest: digest.finalize().into(),
            catalog_only: false,
        })
    }
}

/// Completed local read preparation. Failed preparation supplies an error,
/// without inventing geometry or a payload ceiling. All ready peers must agree
/// these exact facts before any source work or variable-size gather.
pub struct ParameterReadPreparation<P> {
    value: P,
    geometry: [u8; 32],
    max_words: usize,
}
impl<P> ParameterReadPreparation<P> {
    /// Retains the prepared operation and its global geometry/payload bound.
    /// These descriptive facts do not replace local admission or reservations.
    pub fn new(value: P, geometry: [u8; 32], max_words: usize) -> Self {
        Self {
            value,
            geometry,
            max_words,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    usage: ParameterCoordinationUsage,
    models: u64,
    fenced: bool,
}

/// One setup-owned monotone operation sequence. Native clones share this owner;
/// model snapshots neither clone its authority nor rewind its usage or attempts.
#[derive(Debug)]
pub struct ParameterOperationCoordinator {
    setup: CommunicationSessionIdentity,
    wait: BoundedCompletionWait,
    per_operation: CaptureUsage,
    state: Mutex<State>,
}
impl ParameterOperationCoordinator {
    /// Admits fixed control storage from exact native setup facts. Every operation
    /// reserves this finite allowance before control allocation or submission,
    /// even when local caller limits reject all parameter/payload work.
    pub fn new<T: ParameterOperationTransport>(transport: &T) -> Result<Self, ParameterError> {
        let setup = transport.parameter_setup();
        let count = setup.participant_count();
        if count == 0
            || count != transport.participant_count()
            || u32::try_from(count).is_err()
            || transport.parameter_rank() >= count
        {
            return Err(Error::Admission("retained topology").into());
        }
        let wait = transport.parameter_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(Error::Admission("bounded completion disposition").into());
        }
        let native = transport.estimate_parameter_gather(PARAMETER_CONTROL_WORDS)?;
        let gathered = mul(PARAMETER_CONTROL_WORDS as u64, count as u64)?;
        let framing = CaptureUsage {
            retained_bytes: mul(add(gathered, PARAMETER_CONTROL_WORDS as u64)?, 4)?,
            host_bytes: mul(gathered, 4)?,
            ..Default::default()
        };
        let per_operation = native.checked_add(framing)?.checked_mul(CONTROL_ROUNDS)?;
        Ok(Self {
            setup,
            wait,
            per_operation,
            state: Mutex::new(State::default()),
        })
    }
    /// Reserved mandatory-control cost of each attempt, including unused rounds.
    pub const fn per_operation(&self) -> CaptureUsage {
        self.per_operation
    }
    /// Issues one model identity without communication or native parameter work.
    /// Every rank pairs corresponding constructions in the same order. Operations
    /// selecting different construction ordinals fail before parameter work;
    /// identities cannot be recycled after destruction or a failed operation.
    pub fn register_model(&self) -> Result<ParameterModelIdentity, ParameterError> {
        let mut state = self.state.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => Error::Busy,
            TryLockError::Poisoned(_) => Error::Fenced,
        })?;
        if state.fenced {
            return Err(Error::Fenced.into());
        }
        let ordinal = state.models;
        state.models = ordinal.checked_add(1).ok_or(ParameterError::Overflow)?;
        Ok(ParameterModelIdentity {
            setup: self.setup,
            ordinal,
        })
    }
    /// Session-lifetime control charges, including failed or abandoned attempts.
    pub fn usage(&self) -> Result<ParameterCoordinationUsage, ParameterError> {
        match self.state.try_lock() {
            Ok(state) => Ok(state.usage),
            // An unwind is terminal authority, but not a refund or a reason to
            // hide already-reserved work. Reading counters cannot revive it.
            Err(TryLockError::Poisoned(state)) => Ok(state.into_inner().usage),
            Err(TryLockError::WouldBlock) => Err(Error::Busy.into()),
        }
    }

    /// Runs a bounded read through admission, completed source preparation,
    /// payload transport and completed result delivery. No rank receives output
    /// unless every rank validated the same operation and its own assembled value.
    /// `local` already includes exact local authority/geometry/native-work budget
    /// checks; errors must be passed here rather than returned before agreement.
    /// `reservation` prices payload transport before `produce` can run. The caller
    /// also prepays producer output and final assembly storage in its local check.
    /// The common `intent` identifies the request independently of successful
    /// local preparation. Geometry and the common payload ceiling are compared
    /// only after all ranks report ready; a failed rank supplies no guessed facts.
    #[allow(clippy::too_many_arguments)]
    pub fn read<T, P, O>(
        &self,
        transport: &T,
        binding: ParameterOperationBinding,
        intent: &[u8; 32],
        kind: ParameterOperationKind,
        local: Result<ParameterReadPreparation<P>, ParameterError>,
        reservation: &mut impl CaptureReservation,
        produce: impl FnOnce(P) -> Result<Vec<u32>, ParameterError>,
        assemble: impl FnOnce(&[Vec<u32>]) -> Result<O, ParameterError>,
    ) -> Result<O, ParameterError>
    where
        T: ParameterOperationTransport,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let mut operation = self.begin(transport, binding, intent, kind)?;
        let local = local.and_then(|prepared| {
            operation.geometry = prepared.geometry;
            operation.max_words = prepared.max_words;
            if kind.transaction() {
                return Err(Error::Admission("transaction used as read").into());
            }
            if binding.catalog_only && kind != ParameterOperationKind::Catalog {
                return Err(
                    Error::Admission("catalogue binding cannot read parameter values").into(),
                );
            }
            let cost = payload_cost(transport, prepared.max_words)?;
            if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
                reservation.reserve(cost)?
            {
                return Err(CaptureError::Limit { budget, cumulative }.into());
            }
            Ok(prepared.value)
        });
        let prepared = match operation.agree(Stage::Admission, local) {
            Ok(value) => value,
            Err(error) => return operation.rejected(error),
        };
        let max_words = operation.max_words;
        let produced = produce(prepared).and_then(|words| {
            if words.len() > max_words {
                return Err(Error::Protocol("producer exceeded prepaid payload").into());
            }
            Ok(words)
        });
        let words = match operation.agree(Stage::Preparation, produced) {
            Ok(words) => words,
            Err(error) => return operation.rejected(error),
        };
        let output = operation
            .payload(&words, max_words)
            .and_then(|received| assemble(&received));
        match operation.agree(Stage::Delivery, output) {
            Ok(output) => {
                operation.finished = true;
                Ok(output)
            }
            Err(error) => operation.rejected(error),
        }
    }

    /// Prepares all replacements before any publication. On a completed peer
    /// publication rejection every rank restores its original values and cache/
    /// version compatibility before retry is allowed. Restoration failure or an
    /// ambiguous exchange fences the owner; it never exposes a usable partial edit.
    /// Callbacks retain native resources through exact completion and may not
    /// release their original handles before this method returns.
    pub fn transaction<T, P, R>(
        &self,
        transport: &T,
        binding: ParameterOperationBinding,
        intent: &[u8; 32],
        kind: ParameterOperationKind,
        local: Result<P, ParameterError>,
        prepare: impl FnOnce(P) -> Result<R, ParameterError>,
        mut publish: impl FnMut(&mut R) -> Result<(), ParameterError>,
        mut restore: impl FnMut(&mut R) -> Result<(), ParameterError>,
    ) -> Result<R, ParameterError>
    where
        T: ParameterOperationTransport,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let mut operation = self.begin(transport, binding, intent, kind)?;
        let local = local.and_then(|value| {
            if !kind.transaction() {
                return Err(Error::Admission("read used as transaction").into());
            }
            if binding.catalog_only {
                return Err(
                    Error::Admission("catalogue binding cannot edit parameter values").into(),
                );
            }
            Ok(value)
        });
        let input = match operation.agree(Stage::Admission, local) {
            Ok(input) => input,
            Err(error) => return operation.rejected(error),
        };
        let mut prepared = match operation.agree(Stage::Preparation, prepare(input)) {
            Ok(value) => value,
            Err(error) => return operation.rejected(error),
        };
        match operation.agree(Stage::Publication, publish(&mut prepared)) {
            Ok(()) => {
                operation.finished = true;
                Ok(prepared)
            }
            Err(error) if completed_rejection(&error) => {
                match operation.agree(Stage::Restoration, restore(&mut prepared)) {
                    Ok(()) => operation.rejected(error),
                    Err(restoration) => {
                        operation.fence(&Error::Protocol("parameter restoration failed"));
                        Err(restoration)
                    }
                }
            }
            Err(error) => Err(error),
        }
    }

    fn begin<'a, T: ParameterOperationTransport>(
        &'a self,
        transport: &'a T,
        binding: ParameterOperationBinding,
        intent: &[u8; 32],
        kind: ParameterOperationKind,
    ) -> Result<Operation<'a, T>, ParameterError> {
        let mut state = self.state.try_lock().map_err(|error| {
            let error = match error {
                TryLockError::WouldBlock => Error::Busy,
                TryLockError::Poisoned(_) => Error::Fenced,
            };
            transport.fail_parameter_operation(&error);
            ParameterError::from(error)
        })?;
        if state.fenced {
            return Err(Error::Fenced.into());
        }
        let attempt = state.usage.attempts;
        let Some(attempts) = attempt.checked_add(1) else {
            state.fenced = true;
            transport.fail_parameter_operation(&Error::Protocol("operation sequence exhausted"));
            return Err(ParameterError::Overflow);
        };
        let reserved = match state.usage.reserved.checked_add(self.per_operation) {
            Ok(usage) => usage,
            Err(error) => {
                state.fenced = true;
                transport
                    .fail_parameter_operation(&Error::Protocol("control accounting exhausted"));
                return Err(error.into());
            }
        };
        state.usage = ParameterCoordinationUsage { attempts, reserved };
        let mut digest = Sha256::new();
        digest.update(b"eredu-live-parameter-operation-v1\0");
        digest.update(binding.digest);
        digest.update([u8::from(binding.catalog_only)]);
        digest.update(intent);
        digest.update(kind.word().to_le_bytes());
        let mut operation = Operation {
            transport,
            owner: self,
            state,
            attempt,
            digest: digest.finalize().into(),
            kind,
            geometry: [0; 32],
            max_words: 0,
            round: 0,
            finished: false,
        };
        if transport.parameter_setup() != self.setup
            || transport.participant_count() != self.setup.participant_count()
            || transport.parameter_rank() >= self.setup.participant_count()
        {
            operation.fence(&Error::Protocol("retained transport setup"));
            return Err(Error::Protocol("retained transport setup").into());
        }
        if let Err(error) = transport.ensure_parameter_active() {
            let error = Error::Backend(error);
            operation.fence(&error);
            return Err(error.into());
        }
        Ok(operation)
    }
}

struct Operation<'a, T: ParameterOperationTransport> {
    transport: &'a T,
    owner: &'a ParameterOperationCoordinator,
    state: MutexGuard<'a, State>,
    attempt: u64,
    digest: [u8; 32],
    kind: ParameterOperationKind,
    geometry: [u8; 32],
    max_words: usize,
    round: u32,
    finished: bool,
}
impl<T: ParameterOperationTransport> Drop for Operation<'_, T> {
    fn drop(&mut self) {
        if !self.finished && !self.state.fenced {
            self.fence(&Error::Protocol(
                "operation abandoned before completed decision",
            ));
        }
    }
}
impl<T: ParameterOperationTransport> Operation<'_, T> {
    fn fence(&mut self, error: &Error) {
        self.state.fenced = true;
        self.transport.fail_parameter_operation(error);
    }
    fn rejected<O>(&mut self, error: ParameterError) -> Result<O, ParameterError> {
        if completed_rejection(&error) && !self.state.fenced {
            self.finished = true;
        }
        Err(error)
    }
}
impl<T> Operation<'_, T>
where
    T: ParameterOperationTransport,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn gather(&mut self, words: &[u32]) -> Result<Vec<u32>, ParameterError> {
        let result = (|| {
            self.transport.ensure_parameter_active()?;
            let submitted = self
                .transport
                .submit_all_gather_words(words)
                .map_err(BackendFailure::from_error)?;
            let output = match submitted
                .wait_bounded(self.owner.wait)
                .map_err(BackendFailure::from_error)?
            {
                BoundedSubmissionOutcome::Completed(output) => output,
                BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                    return Err(Error::Deadline { cancellation })
                }
            };
            self.transport
                .resolve_all_gather_words(output)
                .map_err(BackendFailure::from_error)
                .map_err(Error::from)
        })();
        result.map_err(|error| {
            self.fence(&error);
            error.into()
        })
    }

    fn agree<V>(
        &mut self,
        stage: Stage,
        local: Result<V, ParameterError>,
    ) -> Result<V, ParameterError> {
        if self.state.fenced {
            return Err(Error::Fenced.into());
        }
        if self.round >= 4 {
            self.fence(&Error::Protocol("control allowance exhausted"));
            return Err(Error::Protocol("control allowance exhausted").into());
        }
        self.round += 1;
        let rank = self.transport.parameter_rank();
        let mut frame = [0; PARAMETER_CONTROL_WORDS];
        frame[..12].copy_from_slice(&[
            MAGIC,
            VERSION,
            self.attempt as u32,
            (self.attempt >> 32) as u32,
            self.kind.word(),
            stage_word(stage),
            self.owner.setup.participant_count() as u32,
            rank as u32,
            u32::from(local.is_err()),
            0,
            0,
            self.round,
        ]);
        for (word, bytes) in frame[12..20]
            .iter_mut()
            .zip(self.owner.setup.bytes().chunks_exact(4))
        {
            *word = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        for (word, bytes) in frame[20..28].iter_mut().zip(self.digest.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        for (word, bytes) in frame[28..36].iter_mut().zip(self.geometry.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        frame[36] = self.max_words as u32;
        frame[37] = ((self.max_words as u64) >> 32) as u32;
        let first = self.exchange_control(&frame, false);
        if self.state.fenced {
            return Err(first.err().unwrap_or_else(|| Error::Fenced.into()));
        }
        frame[1] |= 1 << 16;
        frame[8] = u32::from(first.is_err());
        if let Ok(rejected) = first {
            frame[9] = u32::from(rejected.is_some());
            frame[10] = rejected.unwrap_or(0) as u32;
        }
        let confirmation = self.exchange_control(&frame, true);
        match (first, confirmation) {
            (Ok(rejected), Ok(None)) => match rejected {
                None => local,
                Some(peer) => Err(match local {
                    Err(source) => Error::LocalRejected {
                        rank,
                        stage,
                        source: Box::new(source),
                    },
                    Ok(_) => Error::PeerRejected { rank: peer, stage },
                }
                .into()),
            },
            (Err(error), _) | (_, Err(error)) => {
                self.fence(&Error::Protocol("control decision did not complete"));
                Err(error)
            }
            (_, Ok(Some(_))) => {
                let error = Error::Protocol("peer rejected control validation");
                self.fence(&error);
                Err(error.into())
            }
        }
    }
    fn exchange_control(
        &mut self,
        frame: &[u32; PARAMETER_CONTROL_WORDS],
        confirmation: bool,
    ) -> Result<Option<usize>, ParameterError> {
        let words = self.gather(frame)?;
        let count = self.owner.setup.participant_count();
        if words.len() != count * PARAMETER_CONTROL_WORDS {
            return Err(Error::Protocol("control frame length").into());
        }
        let rank = self.transport.parameter_rank();
        if words[rank * PARAMETER_CONTROL_WORDS..(rank + 1) * PARAMETER_CONTROL_WORDS] != frame[..]
        {
            return Err(Error::Protocol("local control echo").into());
        }
        let mut rejected = None;
        let compare_prepared = if confirmation {
            frame[8] == 0 && frame[9] == 0
        } else {
            frame[5] != stage_word(Stage::Admission)
                || words
                    .chunks_exact(PARAMETER_CONTROL_WORDS)
                    .all(|peer| peer[8] == 0)
        };
        for (rank, peer) in words.chunks_exact(PARAMETER_CONTROL_WORDS).enumerate() {
            if peer[..7] != frame[..7]
                || peer[7] != rank as u32
                || peer[11..28] != frame[11..28]
                || peer[8] > 1
            {
                return Err(Error::Protocol("setup, operation, phase or rank").into());
            }
            if compare_prepared && (!confirmation || peer[8] == 0) && peer[28..] != frame[28..] {
                return Err(Error::Protocol("prepared geometry or payload bound").into());
            }
            if !confirmation && peer[9..11] != [0, 0] {
                return Err(Error::Protocol("reserved control words").into());
            }
            if confirmation && peer[8] == 0 && frame[8] == 0 && peer[9..11] != frame[9..11] {
                return Err(Error::Protocol("completed control decision").into());
            }
            if peer[8] == 1 {
                rejected.get_or_insert(rank);
            }
        }
        Ok(rejected)
    }
    fn payload(
        &mut self,
        words: &[u32],
        max_words: usize,
    ) -> Result<Vec<Vec<u32>>, ParameterError> {
        let stride = max_words
            .checked_add(PAYLOAD_HEADER)
            .ok_or(ParameterError::Overflow)?;
        let mut frame = vec![0; stride];
        frame[..PAYLOAD_HEADER].copy_from_slice(&[
            MAGIC,
            VERSION,
            self.attempt as u32,
            (self.attempt >> 32) as u32,
            self.transport.parameter_rank() as u32,
            words.len() as u32,
            self.kind.word(),
            0,
        ]);
        frame[PAYLOAD_HEADER..PAYLOAD_HEADER + words.len()].copy_from_slice(words);
        let gathered = self.gather(&frame)?;
        if gathered.len() != stride * self.owner.setup.participant_count() {
            return Err(Error::Protocol("payload length").into());
        }
        let rank = self.transport.parameter_rank();
        if gathered[rank * stride..(rank + 1) * stride] != frame {
            return Err(Error::Protocol("local payload echo").into());
        }
        let mut output = Vec::with_capacity(self.owner.setup.participant_count());
        for (rank, peer) in gathered.chunks_exact(stride).enumerate() {
            if peer[..4] != frame[..4]
                || peer[4] != rank as u32
                || peer[6..8] != frame[6..8]
                || peer[5] as usize > max_words
            {
                return Err(Error::Protocol("payload identity or extent").into());
            }
            let end = PAYLOAD_HEADER + peer[5] as usize;
            if peer[end..].iter().any(|word| *word != 0) {
                return Err(Error::Protocol("nonzero payload padding").into());
            }
            output.push(peer[PAYLOAD_HEADER..end].to_vec());
        }
        Ok(output)
    }
}
fn payload_cost<T: ParameterOperationTransport>(
    transport: &T,
    max_words: usize,
) -> Result<CaptureUsage, ParameterError> {
    let stride = max_words
        .checked_add(PAYLOAD_HEADER)
        .ok_or(ParameterError::Overflow)?;
    u32::try_from(stride).map_err(|_| ParameterError::Overflow)?;
    let total = mul(stride as u64, transport.participant_count() as u64)?;
    let framing = CaptureUsage {
        retained_bytes: add(mul(add(stride as u64, total)?, 4)?, 256)?,
        host_bytes: add(
            mul(total, 8)?,
            mul(transport.participant_count() as u64, 64)?,
        )?,
        encoded_bytes: mul(total, 4)?,
        ..Default::default()
    };
    Ok(transport
        .estimate_parameter_gather(stride)?
        .checked_add(framing)?)
}
fn stage_word(stage: Stage) -> u32 {
    match stage {
        Stage::Admission => 0,
        Stage::Preparation => 1,
        Stage::Delivery => 2,
        Stage::Publication => 3,
        Stage::Restoration => 4,
    }
}
fn completed_rejection(error: &ParameterError) -> bool {
    matches!(error, ParameterError::Coordination(error) if matches!(&**error, Error::PeerRejected { .. } | Error::LocalRejected { .. }))
}

#[cfg(test)]
mod tests;
