//! Compact declarations for large, lazily materialized parameter namespaces.
use super::*;

/// A uniform, source-backed range whose entries acquire host/device storage on
/// demand. The descriptor occupies constant space independently of cardinality.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct OffloadUnitRange {
    prefix: OffloadUnitId,
    start: u64,
    end: u64,
    bytes: u64,
    policy: ResidencyPolicy,
}

impl OffloadUnitRange {
    /// Declares canonical IDs `prefix.<decimal member>` in the half-open range.
    /// Source-backed entries cannot be pinned before acquisition.
    pub fn new(
        prefix: OffloadUnitId,
        start: u64,
        end: u64,
        bytes: u64,
        policy: ResidencyPolicy,
    ) -> Result<Self, OffloadError> {
        if start >= end || bytes == 0 || policy == ResidencyPolicy::Pinned {
            return Err(OffloadError::InvalidUnitRange { prefix });
        }
        (end - start)
            .checked_mul(bytes)
            .ok_or(OffloadError::ByteTotalOverflow {
                tier: MemoryTier::Disk,
            })?;
        Ok(Self {
            prefix,
            start,
            end,
            bytes,
            policy,
        })
    }
    /// Exact namespace prefix.
    pub fn prefix(&self) -> &OffloadUnitId {
        &self.prefix
    }
    /// First admitted member.
    pub const fn start(&self) -> u64 {
        self.start
    }
    /// Exclusive member frontier.
    pub const fn end(&self) -> u64 {
        self.end
    }
    /// Materialized payload size of each member, before native allocation overhead.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Eviction policy shared by every member.
    pub const fn policy(&self) -> ResidencyPolicy {
        self.policy
    }
    /// Total source-backed payload, without enumerating entries.
    pub const fn total_bytes(&self) -> u64 {
        (self.end - self.start) * self.bytes
    }
    /// Constructs an exact admitted member identity.
    pub fn member_id(&self, member: u64) -> Option<OffloadUnitId> {
        (self.start <= member && member < self.end)
            .then(|| OffloadUnitId(format!("{}.{member}", self.prefix)))
    }
    /// Resolves an ID without accepting alternate spellings or adjacent namespaces.
    pub fn member(&self, id: &OffloadUnitId) -> Option<u64> {
        let suffix = id
            .as_str()
            .strip_prefix(self.prefix.as_str())?
            .strip_prefix('.')?;
        let member = suffix.parse::<u64>().ok()?;
        (self.member_id(member).as_ref() == Some(id)).then_some(member)
    }
    pub(super) fn unit(&self, id: &OffloadUnitId) -> Option<OffloadUnitSpec> {
        self.member(id).map(|_| OffloadUnitSpec {
            id: id.clone(),
            bytes: self.bytes,
            policy: self.policy,
            tier: MemoryTier::Disk,
        })
    }
}

#[derive(Deserialize)]
struct SerializedRange {
    prefix: OffloadUnitId,
    start: u64,
    end: u64,
    bytes: u64,
    policy: ResidencyPolicy,
}
impl<'de> Deserialize<'de> for OffloadUnitRange {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = SerializedRange::deserialize(deserializer)?;
        Self::new(
            value.prefix,
            value.start,
            value.end,
            value.bytes,
            value.policy,
        )
        .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn range(start: u64, end: u64) -> OffloadUnitRange {
        OffloadUnitRange::new(
            OffloadUnitId::new("rows").unwrap(),
            start,
            end,
            16,
            ResidencyPolicy::Cacheable,
        )
        .unwrap()
    }

    #[test]
    fn huge_range_keeps_only_live_residency_metadata_and_never_evicts_pins() {
        let range = range(0, 1_000_000_000_000);
        let plan = OffloadPlan::with_ranges(
            OffloadConfig::new(Some(32), Some(32), 1).unwrap(),
            [],
            [range.clone()],
        )
        .unwrap();
        assert!(plan.units().is_empty());
        assert_eq!(plan.planned_bytes().disk, 16_000_000_000_000);
        let mut ledger = ResidencyLedger::new(plan);
        ledger.mark_initialized();
        let protected = BTreeSet::new();
        let pinned = range.member_id(0).unwrap();
        for row in 0..1000 {
            let id = range.member_id(row).unwrap();
            assert!(ledger.contains(&id));
            assert!(!ledger.is_resident(&id, MemoryTier::Device).unwrap());
            let evictions = ledger
                .reserve_copy(&id, MemoryTier::Device, 16, &protected)
                .unwrap();
            assert!(evictions.iter().all(|evicted| evicted.id != pinned));
            ledger
                .publish_reserved(&id, MemoryTier::Device, 16, None)
                .unwrap();
            if row == 0 {
                ledger.pin(&id, MemoryTier::Device, 1).unwrap();
            }
            assert!(ledger.unit_reports().len() <= 2);
            assert_eq!(ledger.spec(&id).unwrap().bytes(), 16);
        }
        let before = ledger.unit_reports();
        let oversized = range.member_id(1001).unwrap();
        assert!(ledger
            .reserve_copy(&oversized, MemoryTier::Device, 33, &protected)
            .is_err());
        assert_eq!(ledger.unit_reports(), before);
        assert!(!ledger.units.contains_key(&oversized));
        ledger.unpin(&pinned, MemoryTier::Device);
        ledger.evict(&pinned, MemoryTier::Device).unwrap();
        ledger
            .evict(&range.member_id(999).unwrap(), MemoryTier::Device)
            .unwrap();
        assert!(ledger.unit_reports().is_empty());
        assert!(ledger.contains(&pinned));
    }

    #[test]
    fn ranges_validate_collisions_canonical_ids_and_serialized_geometry() {
        let r = range(2, 5);
        for id in ["rows.02", "rows.5", "rows.1", "rows.+2", "rows.2.0"] {
            assert!(r.member(&OffloadUnitId::new(id).unwrap()).is_none());
        }
        assert_eq!(r.member(&OffloadUnitId::new("rows.2").unwrap()), Some(2));
        assert!(
            OffloadPlan::with_ranges(OffloadConfig::default(), [], [range(0, 3), range(2, 5)])
                .is_err()
        );
        let explicit = OffloadUnitSpec::new(
            r.member_id(2).unwrap(),
            16,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )
        .unwrap();
        assert!(OffloadPlan::with_ranges(OffloadConfig::default(), [explicit], [r]).is_err());
        let plan =
            OffloadPlan::with_ranges(OffloadConfig::default(), [], [range(0, u64::MAX / 16)])
                .unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.len() < 512);
        assert_eq!(serde_json::from_str::<OffloadPlan>(&json).unwrap(), plan);
        let mut malformed = serde_json::to_value(plan).unwrap();
        malformed["ranges"][0]["bytes"] = 0.into();
        assert!(serde_json::from_value::<OffloadPlan>(malformed).is_err());
    }
}
