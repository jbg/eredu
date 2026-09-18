//! Finite compact-bank iteration shared by execution and source quotation.
use crate::ParameterBankAccess;
use std::{mem::size_of, ops::Range};

/// Invalid retained geometry for independently addressable grouped execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AddressableChunkPlanError {
    /// A nonempty route axis and member domain are required.
    #[error("addressable chunk source has no routes or members")]
    EmptyDomain,
    /// Bulk execution needs the selected physical member-byte maximum.
    #[error("addressable bulk chunk source has no selected member-byte maximum")]
    MissingMemberBytes,
    /// The selected access class has no existing chunk policy.
    #[error("addressable chunk access class is unsupported")]
    Access,
    /// Source-derived route or byte arithmetic overflowed.
    #[error("addressable chunk source arithmetic overflowed")]
    Overflow,
}

/// Descriptive iteration bounds from the existing compact-bank policy.
///
/// This contains no selected IDs, cache lease, or native construction authority.
/// Actual member demands and their scratch-byte sum still require validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressableChunkPlan {
    rows: usize,
    chunk_rows: usize,
    routes: usize,
    members: usize,
}
impl AddressableChunkPlan {
    /// Uses the actual selected member-byte maximum only for bulk planning.
    /// Incremental execution preserves the existing one-row chunk policy.
    pub fn new(rows: usize, routes: usize, members: usize, access: ParameterBankAccess,
        maximum_member_bytes: Option<u64>, bulk_target_bytes: u64)
        -> Result<Self, AddressableChunkPlanError> {
        use AddressableChunkPlanError as E;
        if routes == 0 || members == 0 { return Err(E::EmptyDomain); }
        let chunk_rows = match access {
            ParameterBankAccess::Bulk => {
                let per_row = maximum_member_bytes.ok_or(E::MissingMemberBytes)?
                    .checked_mul(u64::try_from(routes).map_err(|_| E::Overflow)?)
                    .ok_or(E::Overflow)?;
                usize::try_from((bulk_target_bytes / per_row.max(1)).max(1))
                    .unwrap_or(usize::MAX)
            }
            ParameterBankAccess::Incremental => 1,
            _ => return Err(E::Access),
        };
        // Every real route slot must be representable, even when duplicate IDs
        // later reduce the number of acquired members.
        rows.checked_mul(routes).ok_or(E::Overflow)?;
        Ok(Self { rows, chunk_rows, routes, members })
    }

    /// Projects the exact retained iteration into the neural recorder without
    /// reconstructing policy or moving runtime ownership into a lower stratum.
    pub const fn workspace_source(self)->eredu_nn::workspace::WorkspaceAddressableChunkPlan {
        eredu_nn::workspace::WorkspaceAddressableChunkPlan {
            rows:self.rows,chunk_rows:self.chunk_rows,routes:self.routes,members:self.members,
        }
    }

    /// Narrows only the row population of this already selected immutable plan.
    /// The physical member domain, route width and compact chunk policy remain
    /// unchanged; this descriptive projection creates no acquisition authority.
    pub fn for_rows(self, rows: usize) -> Option<Self> {
        (rows <= self.rows).then_some(Self { rows, ..self })
    }

    /// Exact number of sequential acquisitions for this invocation.
    pub fn len(self) -> usize { self.rows.div_ceil(self.chunk_rows) }
    /// Whether the invocation contains no token rows.
    pub fn is_empty(self) -> bool { self.rows == 0 }
    /// The exact token range consumed by one existing driver iteration.
    pub fn range(self, index: usize) -> Option<Range<usize>> {
        if index >= self.len() { return None; }
        let start = index.checked_mul(self.chunk_rows)?;
        Some(start..start.saturating_add(self.chunk_rows).min(self.rows))
    }
    /// Allocation-free exact ranges in their ordinary token order.
    pub fn ranges(self) -> impl ExactSizeIterator<Item = Range<usize>> {
        (0..self.len()).map(move |index| {
            self.range(index).expect("ordinal from the same finite chunk plan")
        })
    }
    /// Maximum distinct members for this exact chunk, before actual ID discovery.
    pub fn maximum_members(self, index: usize) -> Option<usize> {
        self.range(index)?.len().checked_mul(self.routes).map(|n| n.min(self.members))
    }
    /// Fixed source-construction and range-query controls, with no heap children.
    pub const fn control_bytes() -> usize {
        size_of::<Self>() + size_of::<Result<Self, AddressableChunkPlanError>>()
            + size_of::<AddressableChunkPlanError>() + size_of::<Option<u64>>()
            + size_of::<Range<usize>>() + size_of::<Option<Range<usize>>>()
            + size_of::<[usize; 8]>() + size_of::<[u64; 3]>()
    }
}

/// One selected member chunk passed by the shared grouped driver to movement.
/// The source contains no discovered IDs, native role, cache lease or byte grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressableChunkCensus {
    bank: usize,
    unit: usize,
    plan: AddressableChunkPlan,
    index: usize,
    access: ParameterBankAccess,
}
impl AddressableChunkCensus {
    /// Selects one actual ordinal from the retained ordinary iteration.
    pub fn new(bank: usize, unit: usize, plan: AddressableChunkPlan, index: usize,
        access: ParameterBankAccess) -> Option<Self> {
        plan.range(index)?;
        Some(Self { bank, unit, plan, index, access })
    }
    /// Selected logical bank identity.
    pub const fn bank(self) -> usize { self.bank }
    /// Selected logical execution unit.
    pub const fn unit(self) -> usize { self.unit }
    /// The unchanged ordinary iteration policy.
    pub const fn plan(self) -> AddressableChunkPlan { self.plan }
    /// Current ordinal in that policy.
    pub const fn index(self) -> usize { self.index }
    /// Selected residency access class.
    pub const fn access(self) -> ParameterBankAccess { self.access }
    /// Exact row count of this ordinal.
    pub fn rows(self) -> usize { self.plan.range(self.index).expect("validated chunk ordinal").len() }
    /// Selection width per token row.
    pub const fn routes(self) -> usize { self.plan.routes }
    /// Complete owner-local identity domain.
    pub const fn members(self) -> usize { self.plan.members }
    /// Row count of the enclosing invocation.
    pub const fn total_rows(self) -> usize { self.plan.rows }
    /// Exact number of integer identity slots.
    pub fn values(self) -> Option<usize> { self.rows().checked_mul(self.routes()) }
    /// Distinct selected-member ceiling for this chunk.
    pub fn maximum_members(self) -> usize {
        self.plan.maximum_members(self.index).expect("validated chunk route count")
    }
}

#[cfg(test)]
mod row_tests {
    use super::*;
    #[test]
    fn completed_rows_only_narrow_the_retained_chunk_program() {
        for access in [ParameterBankAccess::Bulk,ParameterBankAccess::Incremental] {
            let source=AddressableChunkPlan::new(17,2,8,access,Some(128),512).unwrap();
            for rows in [0,1,2,3,16,17] {
                let actual=source.for_rows(rows).unwrap();
                let ordinary=AddressableChunkPlan::new(rows,2,8,access,Some(128),512).unwrap();
                assert_eq!(actual,ordinary);
                assert!(actual.len()<=source.len());
                assert_eq!(actual.workspace_source().chunk_rows,source.workspace_source().chunk_rows);
                assert_eq!(actual.workspace_source().members,8);
                assert_eq!(actual.workspace_source().routes,2);
                let covered=actual.ranges().map(|range|range.len()).sum::<usize>();
                assert_eq!(covered,rows);
                assert_eq!(AddressableChunkCensus::new(7,11,actual,0,access).is_some(),rows!=0);
                assert!(actual.for_rows(rows+1).is_none());
            }
            assert!(source.for_rows(18).is_none());
            assert!(source.for_rows(usize::MAX).is_none());
            assert_eq!(source.workspace_source().rows,17);
        }
    }
}
