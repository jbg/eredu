use super::*;

/// Borrowed protection membership for one atomic capacity request.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyProtection<'a> {
    values: Protection<'a>,
}
#[derive(Clone, Copy, Debug)]
enum Protection<'a> {
    Set(&'a BTreeSet<OffloadUnitId>),
    Sorted(&'a [OffloadUnitId]),
}
impl<'a> ResidencyProtection<'a> {
    /// Adapts an existing ordinary protection set without copying its IDs.
    pub fn set(values: &'a BTreeSet<OffloadUnitId>) -> Self {
        Self {
            values: Protection::Set(values),
        }
    }
    /// Validates strict sorted membership before lending independent IDs.
    pub fn sorted(values: &'a [OffloadUnitId]) -> Option<Self> {
        values
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            .then_some(Self {
                values: Protection::Sorted(values),
            })
    }
    fn contains(self, id: &OffloadUnitId) -> bool {
        match self.values {
            Protection::Set(values) => values.contains(id),
            Protection::Sorted(values) => values.binary_search(id).is_ok(),
        }
    }
}

#[derive(Clone, Copy)]
enum Requests<'a> {
    Owned(&'a [(OffloadUnitId, u64)]),
    Indexed(&'a [OffloadUnitId], &'a [ResidencyReservationRow]),
}
impl<'a> Requests<'a> {
    fn len(self) -> usize {
        match self {
            Self::Owned(rows) => rows.len(),
            Self::Indexed(_, rows) => rows.len(),
        }
    }
    fn id(self, index: usize) -> &'a OffloadUnitId {
        match self {
            Self::Owned(rows) => &rows[index].0,
            Self::Indexed(ids, rows) => &ids[rows[index].input],
        }
    }
    fn bytes(self, index: usize) -> u64 {
        match self {
            Self::Owned(rows) => rows[index].1,
            Self::Indexed(_, rows) => rows[index].bytes,
        }
    }
    fn validate_indices(self) -> Result<(), Failure> {
        if let Self::Indexed(ids, rows) = self {
            if let Some(row) = rows.iter().find(|row| row.input >= ids.len()) {
                return Err(Failure::Destination {
                    field: "reservation input",
                    required: row.input.saturating_add(1),
                    available: ids.len(),
                });
            }
        }
        Ok(())
    }
}

impl ResidencyLedger {
    /// Atomically reserves using final caller-prepared scalar destinations.
    ///
    /// All planning, output capacity and accounting checks precede mutation.
    /// Success returns those same buffers for exact backend storage release;
    /// refusal owns the full source/buffer prefix. Native work is never invoked.
    pub fn reserve_copies_in(
        &mut self,
        ids: &[OffloadUnitId],
        requests: &[ResidencyReservationRow],
        tier: MemoryTier,
        protected: ResidencyProtection<'_>,
        mut storage: ResidencyAdmissionStorage,
    ) -> Result<ResidencyAdmissionStorage, PreparedResidencyAdmissionFailure> {
        match self.reserve_in(
            Requests::Indexed(ids, requests),
            tier,
            protected,
            &mut storage,
            None,
        ) {
            Ok(()) => Ok(storage),
            Err(cause) => Err(PreparedResidencyAdmissionFailure { cause, storage }),
        }
    }

    pub(in crate::residency) fn reserve_copies_ordinary(
        &mut self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        protected: &BTreeSet<OffloadUnitId>,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        // Preserve the target refusal before ordinary scratch allocation.
        validate_ledger_tier(tier, "capacity reservation")?;
        let mut storage = ResidencyAdmissionStorage::ordinary(self.plan_source(), requests.len());
        let mut output = Vec::new();
        let result = self.reserve_in(
            Requests::Owned(requests),
            tier,
            ResidencyProtection::set(protected),
            &mut storage,
            Some(&mut output),
        );
        match result {
            Ok(()) => Ok(output),
            Err(cause) => Err(owned_error(cause, &storage)),
        }
    }

    fn reserve_in(
        &mut self,
        requests: Requests<'_>,
        tier: MemoryTier,
        protected: ResidencyProtection<'_>,
        storage: &mut ResidencyAdmissionStorage,
        mut ordinary_output: Option<&mut Vec<EvictedResidencyCopy>>,
    ) -> Result<(), Failure> {
        if tier == MemoryTier::Disk {
            return Err(Failure::InvalidTarget {
                operation: "capacity reservation",
            });
        }
        if !self.matches_plan_source(&storage.source) {
            return Err(Failure::Source);
        }
        requests.validate_indices()?;
        storage.reset();
        match batch::validate(
            self,
            requests.len(),
            |index| requests.id(index),
            tier,
            &mut storage.order,
        ) {
            Ok(()) => {}
            Err(ResidencyBatchFailure::Destination {
                required,
                available,
            }) => {
                return Err(Failure::Destination {
                    field: "batch index",
                    required,
                    available,
                });
            }
            Err(ResidencyBatchFailure::Ledger(ResidencyBatchError::InvalidTargetTier {
                operation,
            })) => {
                return Err(Failure::InvalidTarget { operation });
            }
            Err(ResidencyBatchFailure::Ledger(ResidencyBatchError::DuplicateBatchUnit)) => {
                return Err(Failure::Duplicate);
            }
            Err(ResidencyBatchFailure::Ledger(ResidencyBatchError::UnknownUnit { id })) => {
                storage.store_primary(id)?;
                return Err(Failure::Unknown);
            }
        }
        let mut required = 0u64;
        for index in 0..requests.len() {
            let id = requests.id(index);
            let plan = storage
                .source
                .ordinal(id)
                .expect("validated actual source ID");
            let bytes = requests.bytes(index);
            if bytes == 0 {
                return Err(Failure::Zero { plan });
            }
            let unit = self.units.get(id).expect("validated batch ID");
            if unit.copy(tier).is_some() {
                return Err(Failure::Exists { plan, tier });
            }
            required = required.checked_add(bytes).ok_or(Failure::Overflow {
                context: "batch capacity reservation",
            })?;
        }
        if requests.len() == 0 {
            return Ok(());
        }

        let mut selected = 0usize;
        if let Some(budget) = self.budget(tier) {
            let needed = self
                .tier_bytes(tier)
                .checked_add(required)
                .ok_or(Failure::Overflow {
                    context: "budget reservation",
                })?;
            let release = needed.saturating_sub(budget);
            if release != 0 {
                for (plan, unit) in self.units.values().enumerate() {
                    let Some(copy) = unit.copy(tier) else {
                        continue;
                    };
                    if copy.lifecycle != CopyLifecycle::Resident
                        || unit.spec.policy() == ResidencyPolicy::Pinned
                        || copy.pins != 0
                        || protected.contains(unit.spec.id())
                        || self.window_contains(unit.spec.id(), tier)
                    {
                        continue;
                    }
                    ensure_capacity(
                        storage.prepared,
                        "candidate",
                        storage.candidates.len(),
                        storage.candidates.capacity(),
                    )?;
                    storage.candidates.push(Candidate {
                        plan,
                        priority: match unit.spec.policy() {
                            ResidencyPolicy::Windowed => 0,
                            ResidencyPolicy::Cacheable => 1,
                            ResidencyPolicy::Pinned => unreachable!("excluded above"),
                        },
                        frequency: match self.plan.config().eviction_policy() {
                            CacheEvictionPolicy::LeastRecentlyUsed => 0,
                            CacheEvictionPolicy::LeastFrequentlyUsed => copy.frequency,
                        },
                        last_used: copy.last_used,
                        bytes: copy.bytes,
                    });
                }
                storage.candidates.sort_unstable_by_key(|row| {
                    (row.priority, row.frequency, row.last_used, row.plan)
                });
                let mut releasable = 0u64;
                for candidate in &storage.candidates {
                    releasable =
                        releasable
                            .checked_add(candidate.bytes)
                            .ok_or(Failure::Overflow {
                                context: "eviction capacity planning",
                            })?;
                    selected += 1;
                    if releasable >= release {
                        break;
                    }
                }
                if releasable < release {
                    for (plan, unit) in self.units.values().enumerate() {
                        let Some(copy) = unit.copy(tier) else {
                            continue;
                        };
                        if copy.lifecycle != CopyLifecycle::Resident {
                            continue;
                        }
                        let pinned = unit.spec.policy() == ResidencyPolicy::Pinned;
                        let active_window = self.window_contains(unit.spec.id(), tier);
                        let request_protected = protected.contains(unit.spec.id());
                        if !(pinned || copy.pins != 0 || active_window || request_protected) {
                            continue;
                        }
                        ensure_capacity(
                            storage.prepared,
                            "blocker",
                            storage.blockers.len(),
                            storage.blockers.capacity(),
                        )?;
                        storage.blockers.push(ResidencyBlockerRow {
                            plan,
                            pinned,
                            in_use: copy.pins,
                            active_window,
                            request_protected,
                        });
                    }
                    return Err(Failure::Budget {
                        plan: storage
                            .source
                            .ordinal(requests.id(0))
                            .expect("validated first request"),
                        tier,
                        required,
                        budget,
                        resident: self.tier_bytes(tier),
                    });
                }
            }
        }

        // Preflight every removal and output before the first ledger mutation.
        let mut remaining = self.tier_bytes(tier);
        for row in storage.candidates.iter().take(selected) {
            let id = storage
                .source
                .id(row.plan)
                .expect("candidate belongs to exact source");
            let copy = self.units.get(id).and_then(|unit| unit.copy(tier)).ok_or(
                Failure::Inconsistent {
                    plan: row.plan,
                    tier,
                    operation: "capacity eviction planning",
                },
            )?;
            remaining = remaining
                .checked_sub(copy.bytes)
                .ok_or(Failure::Inconsistent {
                    plan: row.plan,
                    tier,
                    operation: "copy removal accounting",
                })?;
            ensure_capacity(
                storage.prepared,
                "eviction",
                storage.evicted.len(),
                storage.evicted.capacity(),
            )?;
            storage.evicted.push(ResidencyEvictedRow {
                plan: row.plan,
                tier,
                bytes: copy.bytes,
            });
        }
        let charged = remaining.checked_add(required).ok_or(Failure::Overflow {
            context: "resident byte reservation",
        })?;
        for index in 0..requests.len() {
            let id = requests.id(index);
            if self.units.get(id).is_none() {
                return Err(Failure::Inconsistent {
                    plan: storage.source.ordinal(id).expect("validated source"),
                    tier,
                    operation: "reservation insertion",
                });
            }
        }
        if let Some(output) = ordinary_output.as_mut() {
            output.reserve(storage.evicted.len());
            for row in &storage.evicted {
                output.push(EvictedResidencyCopy {
                    id: storage
                        .source
                        .id(row.plan)
                        .expect("exact eviction source")
                        .clone(),
                    tier: row.tier,
                    bytes: row.bytes,
                });
            }
        }

        // Exclusive self is retained throughout planning and commit. All IDs,
        // slots, capacities and subtraction/addition facts were checked above.
        for row in &storage.evicted {
            let id = storage.source.id(row.plan).expect("preflight source");
            self.remove_copy_committed(id, tier, row.bytes, true);
        }
        self.set_tier_bytes(tier, charged);
        for index in 0..requests.len() {
            let tick = self.next_tick();
            *self
                .units
                .get_mut(requests.id(index))
                .and_then(|unit| unit.slot_mut(tier))
                .expect("preflight insertion") = Some(LedgerCopy {
                lifecycle: CopyLifecycle::Reserved,
                bytes: requests.bytes(index),
                pins: 0,
                last_used: tick,
                frequency: 0,
                in_flight: None,
            });
        }
        self.update_resident_telemetry(tier);
        Ok(())
    }
}

fn ensure_capacity(
    prepared: bool,
    field: &'static str,
    len: usize,
    capacity: usize,
) -> Result<(), Failure> {
    if prepared && len == capacity {
        Err(Failure::Destination {
            field,
            required: len.saturating_add(1),
            available: capacity,
        })
    } else {
        Ok(())
    }
}

fn owned_error(cause: Failure, storage: &ResidencyAdmissionStorage) -> ResidencyLedgerError {
    let id = |plan| storage.source.id(plan).expect("ordinary source ID").clone();
    match cause {
        Failure::NotInitialized => ResidencyLedgerError::NotInitialized,
        Failure::InvalidTarget { operation } => {
            ResidencyLedgerError::InvalidTargetTier { operation }
        }
        Failure::Duplicate => ResidencyLedgerError::DuplicateBatchUnit,
        Failure::Unknown => ResidencyLedgerError::UnknownUnit {
            id: OffloadUnitId(storage.primary.clone()),
        },
        Failure::Zero { plan } => ResidencyLedgerError::ZeroReservation { id: id(plan) },
        Failure::Exists { plan, tier } => {
            ResidencyLedgerError::CopyAlreadyExists { id: id(plan), tier }
        }
        Failure::Overflow { context } => ResidencyLedgerError::ArithmeticOverflow { context },
        Failure::Budget {
            plan,
            tier,
            required,
            budget,
            resident,
        } => ResidencyLedgerError::BudgetExhausted {
            requested: id(plan),
            tier,
            required_bytes: required,
            budget_bytes: budget,
            resident_bytes: resident,
            blocking_units: storage
                .blockers
                .iter()
                .map(|row| ResidencyBlocker {
                    id: id(row.plan),
                    pinned: row.pinned,
                    in_use: row.in_use,
                    active_window: row.active_window,
                    request_protected: row.request_protected,
                })
                .collect(),
        },
        Failure::Inconsistent {
            plan,
            tier,
            operation,
        } => ResidencyLedgerError::StateInconsistent {
            id: id(plan),
            tier,
            operation,
        },
        Failure::Source | Failure::Destination { .. } => {
            unreachable!("ordinary wrapper owns matching, growable storage")
        }
    }
}
