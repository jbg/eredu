//! Backend-neutral payload retention for delayed realtime coordinates.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

use eredu_core::{RealtimeChannelId, RealtimeChannelLayout, RealtimeSlotCoordinate};

use eredu_core::RealtimeChannelPayload;

/// Process-local identity of the canonical session state owning payload tensors.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimePayloadOwnerIdentity(u64);

impl RealtimePayloadOwnerIdentity {
    /// Creates a nonzero owner identity.
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Returns the exact process-local owner value.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Process-local coordinate-history generation bound to every retained payload.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimePayloadGeneration(u64);

impl RealtimePayloadGeneration {
    /// Creates a nonzero history-generation identity.
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Returns the exact process-local history-generation value.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Exact semantic identity carried by every realtime coordinate payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimePayloadContract {
    schedule: RealtimeChannelLayout,
    batch: NonZeroUsize,
    generation: RealtimePayloadGeneration,
    owner: RealtimePayloadOwnerIdentity,
}

impl RealtimePayloadContract {
    /// Creates a contract bound to one schedule, batch, payload geometry, generation, and owner.
    pub fn new(
        schedule: RealtimeChannelLayout,
        batch: usize,
        generation: RealtimePayloadGeneration,
        owner: RealtimePayloadOwnerIdentity,
    ) -> Result<Self, RealtimePayloadContractError> {
        let batch = NonZeroUsize::new(batch).ok_or(RealtimePayloadContractError::EmptyBatch)?;
        for channel in schedule.channels() {
            if channel.payload.elements(batch.get()).is_none() {
                return Err(RealtimePayloadContractError::GeometryOverflow {
                    channel: channel.id,
                    batch: batch.get(),
                });
            }
        }
        Ok(Self {
            schedule,
            batch,
            generation,
            owner,
        })
    }

    /// Returns the exact normalized channel layout.
    pub const fn schedule(&self) -> &RealtimeChannelLayout {
        &self.schedule
    }

    /// Returns the positive batch cardinality.
    pub const fn batch(&self) -> NonZeroUsize {
        self.batch
    }

    /// Returns the request or session generation identity.
    pub const fn generation(&self) -> RealtimePayloadGeneration {
        self.generation
    }

    /// Returns the exact state or resource owner identity.
    pub const fn owner(&self) -> RealtimePayloadOwnerIdentity {
        self.owner
    }

    /// Resolves exact categorical or dense geometry for an admitted channel.
    pub fn slot_geometry(
        &self,
        slot: RealtimeChannelId,
    ) -> Result<&RealtimeChannelPayload, RealtimePayloadContractError> {
        self.schedule
            .channel(slot)
            .map(|channel| &channel.payload)
            .ok_or(RealtimePayloadContractError::InvalidSlot { slot })
    }

    /// Validates another contract against every exact semantic identity field.
    pub fn validate(&self, contract: &Self) -> Result<(), RealtimePayloadContractError> {
        if self.schedule != contract.schedule {
            return Err(RealtimePayloadContractError::ScheduleMismatch);
        }
        if self.batch != contract.batch {
            return Err(RealtimePayloadContractError::BatchMismatch);
        }
        if self.generation != contract.generation {
            return Err(RealtimePayloadContractError::GenerationMismatch);
        }
        if self.owner != contract.owner {
            return Err(RealtimePayloadContractError::OwnerMismatch);
        }
        Ok(())
    }
}

/// One opaque payload bound to an exact coordinate and semantic contract.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimePayloadEnvelope<P> {
    contract: RealtimePayloadContract,
    coordinate: RealtimeSlotCoordinate,
    payload: P,
}

impl<P> RealtimePayloadEnvelope<P> {
    /// Binds a payload to one admitted coordinate and its contract-derived payload geometry.
    pub fn new(
        contract: RealtimePayloadContract,
        coordinate: RealtimeSlotCoordinate,
        payload: P,
    ) -> Result<Self, RealtimePayloadContractError> {
        contract.slot_geometry(coordinate.slot())?;
        Ok(Self {
            contract,
            coordinate,
            payload,
        })
    }

    /// Returns the exact semantic payload contract.
    pub const fn contract(&self) -> &RealtimePayloadContract {
        &self.contract
    }

    /// Returns the exact delayed-frame coordinate.
    pub const fn coordinate(&self) -> RealtimeSlotCoordinate {
        self.coordinate
    }

    /// Returns the exact declared geometry retained with this coordinate.
    pub fn geometry(&self) -> &RealtimeChannelPayload {
        self.contract
            .slot_geometry(self.coordinate.slot())
            .expect("envelope channel was validated")
    }

    /// Returns the opaque payload.
    pub const fn payload(&self) -> &P {
        &self.payload
    }

    /// Validates handoff against another exact contract and coordinate.
    pub fn validate(
        &self,
        contract: &RealtimePayloadContract,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<(), RealtimePayloadContractError> {
        self.contract.validate(contract)?;
        if self.coordinate != coordinate {
            return Err(RealtimePayloadContractError::CoordinateMismatch);
        }
        contract.slot_geometry(coordinate.slot())?;
        Ok(())
    }

    /// Consumes the envelope into its contract, coordinate, and payload.
    pub fn into_parts(self) -> (RealtimePayloadContract, RealtimeSlotCoordinate, P) {
        (self.contract, self.coordinate, self.payload)
    }
}

/// Stable failure while creating or validating exact realtime payload identity.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimePayloadContractError {
    /// A payload contract cannot represent an empty batch.
    #[error("realtime payload contract batch is empty")]
    EmptyBatch,
    /// Batch expansion exceeds representable payload geometry.
    #[error("realtime channel {channel:?} geometry overflows for batch {batch}")]
    GeometryOverflow {
        /// Affected channel.
        channel: RealtimeChannelId,
        /// Requested batch cardinality.
        batch: usize,
    },
    /// The normalized channel layout differs.
    #[error("realtime payload contract schedule does not match")]
    ScheduleMismatch,
    /// The positive batch cardinality differs.
    #[error("realtime payload contract batch does not match")]
    BatchMismatch,
    /// The request or session generation identity differs.
    #[error("realtime payload contract generation does not match")]
    GenerationMismatch,
    /// The state or resource owner identity differs.
    #[error("realtime payload contract owner does not match")]
    OwnerMismatch,
    /// The delayed-frame coordinate differs.
    #[error("realtime payload coordinate does not match")]
    CoordinateMismatch,
    /// A coordinate names an undeclared channel.
    #[error("realtime payload channel {slot:?} is not declared")]
    InvalidSlot {
        /// Unknown channel.
        slot: RealtimeChannelId,
    },
}

/// Payloads retained at exact channel coordinates for one declared layout.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimePayloadHistory<P> {
    schedule: RealtimeChannelLayout,
    contract: Option<RealtimePayloadContract>,
    payloads: BTreeMap<RealtimeSlotCoordinate, RealtimePayloadEnvelope<P>>,
}

impl<P> RealtimePayloadHistory<P> {
    /// Creates an empty pre-first-frame history bound only to one exact schedule.
    ///
    /// A payload contract must be bound before any coordinate payload can be
    /// published. This unbound form exists because session batch is not known
    /// until the first frame is accepted.
    pub fn new(schedule: RealtimeChannelLayout) -> Self {
        Self {
            schedule,
            contract: None,
            payloads: BTreeMap::new(),
        }
    }

    /// Creates an empty history bound to one complete payload contract.
    pub fn with_contract(contract: RealtimePayloadContract) -> Self {
        Self {
            schedule: contract.schedule().clone(),
            contract: Some(contract),
            payloads: BTreeMap::new(),
        }
    }

    /// Returns the exact normalized schedule bound to this history.
    pub const fn schedule(&self) -> &RealtimeChannelLayout {
        &self.schedule
    }

    /// Returns the exact payload contract after first-frame binding.
    pub const fn contract(&self) -> Option<&RealtimePayloadContract> {
        self.contract.as_ref()
    }

    /// Binds the first complete contract or validates an already-bound history.
    ///
    /// Failed validation never changes the current contract or payloads.
    pub fn bind_or_validate_contract(
        &mut self,
        contract: &RealtimePayloadContract,
    ) -> Result<(), RealtimePayloadHistoryError> {
        if &self.schedule != contract.schedule() {
            return Err(RealtimePayloadHistoryError::PayloadContract(
                RealtimePayloadContractError::ScheduleMismatch,
            ));
        }
        if let Some(current) = &self.contract {
            current
                .validate(contract)
                .map_err(RealtimePayloadHistoryError::PayloadContract)
        } else {
            debug_assert!(self.payloads.is_empty());
            self.contract = Some(contract.clone());
            Ok(())
        }
    }

    /// Validates whether a branch history may replace this canonical history.
    pub fn validate_successor(&self, successor: &Self) -> Result<(), RealtimePayloadHistoryError> {
        self.validate_schedule(&successor.schedule)?;
        match (&self.contract, &successor.contract) {
            (Some(current), Some(candidate)) => current
                .validate(candidate)
                .map_err(RealtimePayloadHistoryError::PayloadContract),
            (None, Some(_)) if self.payloads.is_empty() => Ok(()),
            (None, None) => Ok(()),
            (Some(_), None) | (None, Some(_)) => Err(RealtimePayloadHistoryError::UnboundContract),
        }
    }

    /// Returns the number of retained coordinate payloads.
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Returns whether no payloads are retained.
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Rejects handoff to any materially different normalized schedule.
    pub fn validate_schedule(
        &self,
        schedule: &RealtimeChannelLayout,
    ) -> Result<(), RealtimePayloadHistoryError> {
        if &self.schedule == schedule {
            Ok(())
        } else {
            Err(RealtimePayloadHistoryError::ScheduleMismatch)
        }
    }

    /// Resolves an absolute coordinate by adding the exact configured slot delay.
    ///
    /// This is only coordinate arithmetic; it does not assign temporal-input or
    /// prediction-target meaning to the resulting slot.
    pub fn delayed_coordinate(
        &self,
        schedule: &RealtimeChannelLayout,
        base_position: usize,
        slot: RealtimeChannelId,
    ) -> Result<RealtimeSlotCoordinate, RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        let delay = self.slot_delay(slot)?;
        let position = base_position.checked_add(delay).ok_or(
            RealtimePayloadHistoryError::CoordinateOverflow {
                base_position,
                delay,
            },
        )?;
        Ok(RealtimeSlotCoordinate::new(position, slot))
    }

    /// Inserts one payload at one exact validated coordinate.
    pub fn insert(
        &mut self,
        schedule: &RealtimeChannelLayout,
        coordinate: RealtimeSlotCoordinate,
        payload: P,
    ) -> Result<(), RealtimePayloadHistoryError> {
        self.insert_many(schedule, [(coordinate, payload)])
    }

    /// Calculates one delayed coordinate and inserts its payload atomically.
    pub fn insert_delayed(
        &mut self,
        schedule: &RealtimeChannelLayout,
        base_position: usize,
        slot: RealtimeChannelId,
        payload: P,
    ) -> Result<RealtimeSlotCoordinate, RealtimePayloadHistoryError> {
        let coordinate = self.delayed_coordinate(schedule, base_position, slot)?;
        self.insert(schedule, coordinate, payload)?;
        Ok(coordinate)
    }

    /// Inserts a complete set of coordinate payloads as one publication.
    ///
    /// Invalid slots and duplicates are detected before any payload is visible.
    pub fn insert_many(
        &mut self,
        schedule: &RealtimeChannelLayout,
        payloads: impl IntoIterator<Item = (RealtimeSlotCoordinate, P)>,
    ) -> Result<(), RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        let contract = self
            .contract
            .as_ref()
            .ok_or(RealtimePayloadHistoryError::UnboundContract)?
            .clone();
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        let mut pending = BTreeSet::new();
        for (coordinate, _) in &payloads {
            self.validate_coordinate(*coordinate)?;
            if self.payloads.contains_key(coordinate) || !pending.insert(*coordinate) {
                return Err(RealtimePayloadHistoryError::DuplicatePayload {
                    coordinate: *coordinate,
                });
            }
        }
        let envelopes = payloads
            .into_iter()
            .map(|(coordinate, payload)| {
                RealtimePayloadEnvelope::new(contract.clone(), coordinate, payload)
                    .map(|envelope| (coordinate, envelope))
                    .map_err(RealtimePayloadHistoryError::PayloadContract)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.payloads.extend(envelopes);
        Ok(())
    }

    /// Atomically publishes validated coordinate payloads with explicit overwrite semantics.
    ///
    /// This is reserved for schedule-authorized placement and target updates.
    /// Ordinary admission should use [`Self::insert_many`] so accidental
    /// duplicate producers still fail closed.
    pub fn overwrite_many(
        &mut self,
        schedule: &RealtimeChannelLayout,
        payloads: impl IntoIterator<Item = (RealtimeSlotCoordinate, P)>,
    ) -> Result<(), RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        let contract = self
            .contract
            .as_ref()
            .ok_or(RealtimePayloadHistoryError::UnboundContract)?
            .clone();
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        for (coordinate, _) in &payloads {
            self.validate_coordinate(*coordinate)?;
        }
        let envelopes = payloads
            .into_iter()
            .map(|(coordinate, payload)| {
                RealtimePayloadEnvelope::new(contract.clone(), coordinate, payload)
                    .map(|envelope| (coordinate, envelope))
                    .map_err(RealtimePayloadHistoryError::PayloadContract)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.payloads.extend(envelopes);
        Ok(())
    }

    /// Returns a payload at one exact validated coordinate, when present.
    pub fn get(
        &self,
        schedule: &RealtimeChannelLayout,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<Option<&P>, RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        self.validate_coordinate(coordinate)?;
        Ok(self
            .payloads
            .get(&coordinate)
            .map(RealtimePayloadEnvelope::payload))
    }

    /// Returns the typed envelope at one exact validated coordinate, when present.
    pub fn envelope(
        &self,
        schedule: &RealtimeChannelLayout,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<Option<&RealtimePayloadEnvelope<P>>, RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        self.validate_coordinate(coordinate)?;
        Ok(self.payloads.get(&coordinate))
    }

    /// Resolves one required payload or fails with its exact missing coordinate.
    pub fn required(
        &self,
        schedule: &RealtimeChannelLayout,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<&P, RealtimePayloadHistoryError> {
        self.get(schedule, coordinate)?
            .ok_or(RealtimePayloadHistoryError::MissingPayload { coordinate })
    }

    /// Resolves required coordinates in caller order, retaining duplicate reads.
    pub fn resolve_required(
        &self,
        schedule: &RealtimeChannelLayout,
        coordinates: impl IntoIterator<Item = RealtimeSlotCoordinate>,
    ) -> Result<Vec<&P>, RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        coordinates
            .into_iter()
            .map(|coordinate| self.required(schedule, coordinate))
            .collect()
    }

    /// Prunes coordinates older than the deterministic delayed-history window.
    ///
    /// For next frontier `n` and declared delay-plus-feedback lookback `d`,
    /// positions before `n - (d + 1)` are removed. The additional position retains the oldest
    /// payload referenced by the just-submitted transition until its completion
    /// has captured [`Self::retained_values`]. Warm-up clamps the minimum
    /// position to zero. The return value is the number of payloads removed.
    pub fn prune_for_next_frontier(
        &mut self,
        schedule: &RealtimeChannelLayout,
        next_frontier: usize,
    ) -> Result<usize, RealtimePayloadHistoryError> {
        self.validate_schedule(schedule)?;
        let retained_positions = schedule.retention_lookback().checked_add(1).ok_or(
            RealtimePayloadHistoryError::RetentionWindowOverflow {
                max_delay: schedule.retention_lookback(),
            },
        )?;
        let minimum = next_frontier.saturating_sub(retained_positions);
        let previous = self.payloads.len();
        self.payloads
            .retain(|coordinate, _| coordinate.position() >= minimum);
        Ok(previous - self.payloads.len())
    }

    /// Iterates retained payloads in stable coordinate order.
    pub fn retained_values(&self) -> impl Iterator<Item = &P> {
        self.payloads.values().map(RealtimePayloadEnvelope::payload)
    }

    /// Iterates retained typed envelopes in stable coordinate order.
    pub fn envelopes(&self) -> impl Iterator<Item = &RealtimePayloadEnvelope<P>> {
        self.payloads.values()
    }

    /// Iterates exact coordinates and retained payloads in stable order.
    pub fn entries(&self) -> impl Iterator<Item = (RealtimeSlotCoordinate, &P)> {
        self.payloads
            .iter()
            .map(|(coordinate, envelope)| (*coordinate, envelope.payload()))
    }

    fn validate_coordinate(
        &self,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<(), RealtimePayloadHistoryError> {
        self.slot_delay(coordinate.slot()).map(|_| ())
    }

    fn slot_delay(&self, slot: RealtimeChannelId) -> Result<usize, RealtimePayloadHistoryError> {
        self.schedule
            .channel(slot)
            .map(|channel| channel.delay)
            .ok_or(RealtimePayloadHistoryError::InvalidSlot { slot })
    }
}

/// Stable failure while retaining or resolving delayed-coordinate payloads.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimePayloadHistoryError {
    /// Coordinate payload operations require a complete exact contract.
    #[error("realtime payload history has no bound payload contract")]
    UnboundContract,
    /// The complete payload identity differs from the bound history.
    #[error(transparent)]
    PayloadContract(RealtimePayloadContractError),
    /// The caller supplied a materially different normalized schedule.
    #[error("realtime payload history does not match the normalized schedule")]
    ScheduleMismatch,
    /// A coordinate names an undeclared channel.
    #[error("realtime payload channel {slot:?} is not declared")]
    InvalidSlot {
        /// Unknown channel.
        slot: RealtimeChannelId,
    },
    /// A coordinate already has a payload in the current or pending publication.
    #[error("realtime coordinate {coordinate:?} already contains a payload")]
    DuplicatePayload {
        /// Duplicate coordinate.
        coordinate: RealtimeSlotCoordinate,
    },
    /// A required coordinate has no retained payload.
    #[error("realtime coordinate {coordinate:?} has no retained payload")]
    MissingPayload {
        /// Missing coordinate.
        coordinate: RealtimeSlotCoordinate,
    },
    /// Adding a schedule delay exceeded the coordinate representation.
    #[error("realtime payload coordinate overflowed from base {base_position} plus delay {delay}")]
    CoordinateOverflow {
        /// Undelayed base position.
        base_position: usize,
        /// Exact configured slot delay.
        delay: usize,
    },
    /// Maximum delay cannot be represented as a retained-position count.
    #[error("realtime payload retention window overflowed for maximum delay {max_delay}")]
    RetentionWindowOverflow {
        /// Exact configured maximum delay.
        max_delay: usize,
    },
}

#[cfg(test)]
mod tests {
    use crate::{RealtimeIngressContract, TokenDomain};
    use eredu_core::{RealtimeFrameConvention, RealtimeSpeechConfig};

    use super::*;

    fn schedule() -> RealtimeChannelLayout {
        RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                2,
                1,
                1,
                1,
                0,
                1,
                RealtimeFrameConvention::FeedbackAlignedHistory,
                vec![2, 0, 3],
            )
            .unwrap(),
            TokenDomain::new(32),
            TokenDomain::new(16),
        )
        .unwrap()
        .channels()
        .clone()
    }

    fn coordinate(position: usize, slot: RealtimeChannelId) -> RealtimeSlotCoordinate {
        RealtimeSlotCoordinate::new(position, slot)
    }

    fn payload_contract(
        schedule: RealtimeChannelLayout,
        batch: usize,
        text_domain: TokenDomain,
        audio_domain: TokenDomain,
        generation: u64,
        owner: u64,
    ) -> RealtimePayloadContract {
        let mut channels = schedule.channels().to_vec();
        for channel in &mut channels {
            channel.payload = RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: if channel.id.index() == 0 {
                    text_domain.cardinality()
                } else {
                    audio_domain.cardinality()
                },
            };
        }
        RealtimePayloadContract::new(
            RealtimeChannelLayout::new(schedule.policy().into(), channels).unwrap(),
            batch,
            RealtimePayloadGeneration::new(generation).unwrap(),
            RealtimePayloadOwnerIdentity::new(owner).unwrap(),
        )
        .unwrap()
    }

    fn exact_payload_contract() -> RealtimePayloadContract {
        payload_contract(
            schedule(),
            2,
            TokenDomain::new(32),
            TokenDomain::new(16),
            7,
            3,
        )
    }

    fn payload_history<P>() -> RealtimePayloadHistory<P> {
        RealtimePayloadHistory::with_contract(exact_payload_contract())
    }

    #[test]
    fn payload_contract_requires_nonempty_batch_and_typed_nonempty_identities() {
        assert_eq!(
            RealtimePayloadContract::new(
                schedule(),
                0,
                RealtimePayloadGeneration::new(7).unwrap(),
                RealtimePayloadOwnerIdentity::new(3).unwrap()
            ),
            Err(RealtimePayloadContractError::EmptyBatch)
        );
        assert_eq!(RealtimePayloadGeneration::new(0), None);
        assert_eq!(RealtimePayloadOwnerIdentity::new(0), None);

        let contract = exact_payload_contract();
        assert_eq!(contract.schedule(), &schedule());
        assert_eq!(contract.batch().get(), 2);
        assert_eq!(contract.generation().value(), 7);
        assert_eq!(contract.owner().value(), 3);
    }

    #[test]
    fn payload_envelope_derives_slot_domain_and_validates_exact_coordinate() {
        let contract = exact_payload_contract();
        let text_coordinate = coordinate(5, RealtimeChannelId::new(0));
        let text = RealtimePayloadEnvelope::new(contract.clone(), text_coordinate, "text").unwrap();
        assert_eq!(text.contract(), &contract);
        assert_eq!(text.coordinate(), text_coordinate);
        assert_eq!(
            text.geometry(),
            &RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: 32
            }
        );
        assert_eq!(text.payload(), &"text");
        assert_eq!(text.validate(&contract, text_coordinate), Ok(()));

        let audio_coordinate = coordinate(5, RealtimeChannelId::new(2));
        let audio =
            RealtimePayloadEnvelope::new(contract.clone(), audio_coordinate, "audio").unwrap();
        assert_eq!(
            audio.geometry(),
            &RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: 16
            }
        );
        assert_eq!(audio.validate(&contract, audio_coordinate), Ok(()));
        assert_eq!(
            audio.validate(&contract, coordinate(6, RealtimeChannelId::new(2))),
            Err(RealtimePayloadContractError::CoordinateMismatch)
        );

        assert_eq!(
            RealtimePayloadEnvelope::new(
                contract,
                coordinate(5, RealtimeChannelId::new(3)),
                "invalid",
            ),
            Err(RealtimePayloadContractError::InvalidSlot {
                slot: RealtimeChannelId::new(3),
            })
        );
    }

    #[test]
    fn history_requires_first_contract_binding_and_stores_only_typed_envelopes() {
        let schedule = schedule();
        let text = coordinate(0, RealtimeChannelId::new(0));
        let audio = coordinate(0, RealtimeChannelId::new(1));
        let mut history = RealtimePayloadHistory::new(schedule.clone());
        assert!(history.contract().is_none());
        assert_eq!(history.get(&schedule, text), Ok(None));
        assert_eq!(
            history.insert(&schedule, text, 3),
            Err(RealtimePayloadHistoryError::UnboundContract)
        );
        assert!(history.is_empty());

        let contract = exact_payload_contract();
        history.bind_or_validate_contract(&contract).unwrap();
        history
            .insert_many(&schedule, [(text, 3), (audio, 5)])
            .unwrap();
        assert_eq!(history.contract(), Some(&contract));
        let envelopes = history.envelopes().collect::<Vec<_>>();
        assert_eq!(envelopes.len(), 2);
        assert_eq!(envelopes[0].contract(), &contract);
        assert_eq!(envelopes[0].coordinate(), text);
        assert_eq!(
            envelopes[0].geometry(),
            &RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: 32
            }
        );
        assert_eq!(envelopes[0].payload(), &3);
        assert_eq!(envelopes[1].contract(), &contract);
        assert_eq!(envelopes[1].coordinate(), audio);
        assert_eq!(
            envelopes[1].geometry(),
            &RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: 16
            }
        );
        assert_eq!(
            history.envelope(&schedule, audio).unwrap(),
            Some(envelopes[1])
        );
    }

    #[test]
    fn bound_history_rejects_every_contract_perturbation_without_mutation() {
        let exact = exact_payload_contract();
        let schedule = schedule();
        let text = coordinate(0, RealtimeChannelId::new(0));
        let mut history = RealtimePayloadHistory::with_contract(exact.clone());
        history.insert(&schedule, text, 3).unwrap();
        let other_schedule = RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                2,
                1,
                1,
                1,
                0,
                1,
                RealtimeFrameConvention::AbsoluteDelayedSlots,
                vec![2, 0, 3],
            )
            .unwrap(),
            TokenDomain::new(32),
            TokenDomain::new(16),
        )
        .unwrap()
        .channels()
        .clone();
        let perturbations = [
            (
                payload_contract(
                    other_schedule,
                    2,
                    TokenDomain::new(32),
                    TokenDomain::new(16),
                    7,
                    3,
                ),
                RealtimePayloadContractError::ScheduleMismatch,
            ),
            (
                payload_contract(
                    schedule.clone(),
                    3,
                    TokenDomain::new(32),
                    TokenDomain::new(16),
                    7,
                    3,
                ),
                RealtimePayloadContractError::BatchMismatch,
            ),
            (
                payload_contract(
                    schedule.clone(),
                    2,
                    TokenDomain::new(33),
                    TokenDomain::new(16),
                    7,
                    3,
                ),
                RealtimePayloadContractError::ScheduleMismatch,
            ),
            (
                payload_contract(
                    schedule.clone(),
                    2,
                    TokenDomain::new(32),
                    TokenDomain::new(17),
                    7,
                    3,
                ),
                RealtimePayloadContractError::ScheduleMismatch,
            ),
            (
                payload_contract(
                    schedule.clone(),
                    2,
                    TokenDomain::new(32),
                    TokenDomain::new(16),
                    8,
                    3,
                ),
                RealtimePayloadContractError::GenerationMismatch,
            ),
            (
                payload_contract(
                    schedule.clone(),
                    2,
                    TokenDomain::new(32),
                    TokenDomain::new(16),
                    7,
                    4,
                ),
                RealtimePayloadContractError::OwnerMismatch,
            ),
        ];

        for (candidate, expected) in perturbations {
            assert_eq!(
                history.bind_or_validate_contract(&candidate),
                Err(RealtimePayloadHistoryError::PayloadContract(expected))
            );
            assert_eq!(history.contract(), Some(&exact));
            assert_eq!(history.required(&schedule, text), Ok(&3));
            assert_eq!(history.len(), 1);
        }
    }

    #[test]
    fn history_clone_and_prune_preserve_envelope_identity() {
        let schedule = schedule();
        let contract = exact_payload_contract();
        let mut history = RealtimePayloadHistory::with_contract(contract.clone());
        for position in 0..=6 {
            history
                .insert(
                    &schedule,
                    coordinate(position, RealtimeChannelId::new(0)),
                    position,
                )
                .unwrap();
        }
        let mut branch = history.clone();
        assert_eq!(branch.contract(), Some(&contract));
        assert!(branch
            .envelopes()
            .all(|envelope| envelope.contract() == &contract));

        assert_eq!(branch.prune_for_next_frontier(&schedule, 7), Ok(2));
        assert_eq!(branch.len(), 5);
        assert_eq!(history.len(), 7);
        assert!(branch.envelopes().all(|envelope| {
            envelope.coordinate().position() >= 2 && envelope.contract() == &contract
        }));
    }

    #[test]
    fn exact_insert_get_and_required_resolution_preserve_coordinate_order() {
        let schedule = schedule();
        let mut history = payload_history();
        let text = coordinate(2, RealtimeChannelId::new(0));
        let audio_zero = coordinate(2, RealtimeChannelId::new(1));
        let audio_one = coordinate(2, RealtimeChannelId::new(2));
        history
            .insert_many(
                &schedule,
                [(audio_one, "a1"), (text, "text"), (audio_zero, "a0")],
            )
            .unwrap();

        assert_eq!(history.get(&schedule, text).unwrap(), Some(&"text"));
        assert_eq!(history.required(&schedule, audio_one).unwrap(), &"a1");
        assert_eq!(
            history
                .resolve_required(&schedule, [audio_one, text, audio_one])
                .unwrap(),
            vec![&"a1", &"text", &"a1"]
        );
        assert_eq!(
            history.retained_values().copied().collect::<Vec<_>>(),
            vec!["text", "a0", "a1"]
        );
        assert_eq!(history.schedule(), &schedule);
    }

    #[test]
    fn invalid_or_duplicate_publications_are_atomic() {
        let schedule = schedule();
        let mut history = payload_history();
        let retained = coordinate(0, RealtimeChannelId::new(0));
        history.insert(&schedule, retained, 7).unwrap();
        let invalid = coordinate(1, RealtimeChannelId::new(3));

        assert_eq!(
            history.insert_many(
                &schedule,
                [(coordinate(1, RealtimeChannelId::new(1)), 8), (invalid, 9)]
            ),
            Err(RealtimePayloadHistoryError::InvalidSlot {
                slot: RealtimeChannelId::new(3),
            })
        );
        assert_eq!(history.len(), 1);
        assert_eq!(
            history.insert_many(&schedule, [(invalid, 8), (invalid, 9)]),
            Err(RealtimePayloadHistoryError::InvalidSlot {
                slot: RealtimeChannelId::new(3),
            })
        );
        let pending = coordinate(1, RealtimeChannelId::new(1));
        assert_eq!(
            history.insert_many(&schedule, [(pending, 8), (pending, 9)]),
            Err(RealtimePayloadHistoryError::DuplicatePayload {
                coordinate: pending,
            })
        );
        assert_eq!(history.len(), 1);
        assert_eq!(
            history.insert(&schedule, retained, 10),
            Err(RealtimePayloadHistoryError::DuplicatePayload {
                coordinate: retained,
            })
        );
        assert_eq!(history.required(&schedule, retained), Ok(&7));
    }

    #[test]
    fn mismatch_missing_and_arithmetic_fail_without_mutation() {
        let schedule = schedule();
        let other = RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                2,
                1,
                1,
                1,
                0,
                1,
                RealtimeFrameConvention::AbsoluteDelayedSlots,
                vec![2, 0, 3],
            )
            .unwrap(),
            TokenDomain::new(32),
            TokenDomain::new(16),
        )
        .unwrap()
        .channels()
        .clone();
        let mut history = payload_history();
        let retained = coordinate(0, RealtimeChannelId::new(0));
        history.insert(&schedule, retained, 7).unwrap();

        assert_eq!(
            history.insert(&other, coordinate(1, RealtimeChannelId::new(0)), 8),
            Err(RealtimePayloadHistoryError::ScheduleMismatch)
        );
        let missing = coordinate(1, RealtimeChannelId::new(1));
        assert_eq!(
            history.required(&schedule, missing),
            Err(RealtimePayloadHistoryError::MissingPayload {
                coordinate: missing,
            })
        );
        assert_eq!(
            history.insert_delayed(&schedule, usize::MAX, RealtimeChannelId::new(0), 9,),
            Err(RealtimePayloadHistoryError::CoordinateOverflow {
                base_position: usize::MAX,
                delay: 2,
            })
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history.required(&schedule, retained), Ok(&7));
    }

    #[test]
    fn pruning_uses_next_frontier_and_maximum_delay_deterministically() {
        let schedule = schedule();
        let mut history = payload_history();
        for position in 0..=5 {
            history
                .insert(
                    &schedule,
                    coordinate(position, RealtimeChannelId::new(0)),
                    position,
                )
                .unwrap();
        }

        assert_eq!(history.prune_for_next_frontier(&schedule, 3), Ok(0));
        assert_eq!(history.prune_for_next_frontier(&schedule, 6), Ok(1));
        assert_eq!(
            history
                .entries()
                .map(|(coordinate, payload)| (coordinate.position(), *payload))
                .collect::<Vec<_>>(),
            vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        );
    }
}
