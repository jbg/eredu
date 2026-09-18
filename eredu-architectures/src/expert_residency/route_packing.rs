//! One retained semantic owner table and one ordinary/paid packing worker.
use super::{ExpertRealizationPlan, ExpertRoutePackingPlan};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Exact architecture ownership borrowed from the selected constructor.
/// No tensor value, native stream, or communication authority is present.
#[derive(Debug, Clone, Copy)]
pub struct ExpertRoutePackingSource<'a> {
    owners: &'a [usize],
    owner_local_experts: &'a [usize],
    peers: usize,
}
impl<S> ExpertRealizationPlan<S> {
    /// Borrows the table established by the ordinary selected constructor.
    /// The dense owner-local numbering is never recomputed during a forward.
    pub fn route_packing_source(&self) -> ExpertRoutePackingSource<'_> {
        ExpertRoutePackingSource {
            owners: &self.owners,
            owner_local_experts: &self.owner_local_experts,
            peers: self.expert_parallel_size,
        }
    }
}

/// Exact finite destination population derived from selected route geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRoutePackingGeometry {
    source_tokens: usize,
    routes_per_token: usize,
    routes: usize,
    peers: usize,
    payload_bytes: usize,
}
impl ExpertRoutePackingGeometry {
    /// Source rows before selection expansion.
    pub const fn source_tokens(self) -> usize { self.source_tokens }
    /// Selected route slots for each source token.
    pub const fn routes_per_token(self) -> usize { self.routes_per_token }
    /// Actual selected route slots, including zero-row invocations.
    pub const fn routes(self) -> usize { self.routes }
    /// Selected peer population, including peers with no selected routes.
    pub const fn peers(self) -> usize { self.peers }
    /// Requested storage of the six concrete vector allocations in the worker.
    /// Immutable owner tables remain in their original constructor account.
    pub const fn payload_bytes(self) -> usize { self.payload_bytes }
}

/// Fixed failure before any malformed selection can index a destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ExpertRoutePackingCause {
    /// Selected route dimensions or value count differ.
    #[error("expert route IDs do not match source token and route cardinality")]
    Geometry,
    /// An actual selected ID is outside the retained bank.
    #[error("route position {position} selects invalid global expert {expert}")]
    Expert { position: usize, expert: usize },
    /// The actual signed routing source contains a negative identity.
    #[error("route position {position} selects negative global expert {expert}")]
    NegativeExpert { position: usize, expert: i32 },
    /// A checked finite destination calculation overflowed.
    #[error("expert route destination geometry overflowed")]
    Overflow,
    /// The exact host metadata account refused construction.
    #[error("expert route metadata destination was refused: {0}")]
    Funding(#[source] HostMetadataFundingError),
}

/// Move-only metadata result. Fields retire the packed arrays before funding.
#[derive(Debug)]
pub struct FundedExpertRoutePacking {
    plan: ExpertRoutePackingPlan,
    funding: HostMetadataFunding,
}
impl FundedExpertRoutePacking {
    pub(crate) fn borrowed_plan(&self) -> &ExpertRoutePackingPlan { &self.plan }
    /// Only projections that change their final extent need an owned shape.
    /// The enclosing packing keeps its account alive through the reshape call.
    pub(crate) fn output_shape(&self, input: &[i32], output: i32)
        -> Result<Vec<i32>, FundedExpertRoutePackingFailure>
    {
        let result = (|| {
            let frames = [size_of::<Vec<i32>>(), size_of::<(&Self, &[i32], i32)>(),
                size_of::<Result<Vec<i32>, FundedExpertRoutePackingFailure>>(),
                size_of::<FundedExpertRoutePackingFailure>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(usize, Option<usize>)>(),
                eredu_nn::Error::retained_source_construction_bytes::<FundedExpertRoutePackingFailure>()
                    .ok_or(ExpertRoutePackingCause::Overflow)?];
            let controls = frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(ExpertRoutePackingCause::Overflow)?;
            self.funding.reserve_metadata(controls).map_err(ExpertRoutePackingCause::Funding)?;
            if input.is_empty() || output < 0 { return Err(ExpertRoutePackingCause::Geometry); }
            let bytes = input.len().checked_mul(size_of::<i32>())
                .filter(|bytes| *bytes <= isize::MAX as usize)
                .ok_or(ExpertRoutePackingCause::Overflow)?;
            self.funding.reserve_metadata(bytes).map_err(ExpertRoutePackingCause::Funding)?;
            let mut shape = Vec::with_capacity(input.len());
            shape.extend_from_slice(input);
            *shape.last_mut().expect("nonempty source geometry") = output;
            Ok(shape)
        })();
        result.map_err(|cause| FundedExpertRoutePackingFailure { cause, funding: self.funding.clone() })
    }
    /// Original row count before route expansion.
    pub fn source_tokens(&self) -> usize { self.plan.source_tokens() }
    /// Selected route slots for each source token.
    pub fn routes_per_token(&self) -> usize { self.plan.routes_per_token() }
    /// Immutable destination-major send counts, including idle peers.
    pub fn send_counts(&self) -> &[usize] { self.plan.send_counts() }
    /// Original route position for each destination-major row.
    pub fn packed_route_positions(&self) -> &[usize] { self.plan.packed_route_positions() }
    /// Source token for each destination-major row.
    pub fn packed_token_indices(&self) -> &[usize] { self.plan.packed_token_indices() }
    /// Global expert for each destination-major row.
    pub fn packed_global_experts(&self) -> &[usize] { self.plan.packed_global_experts() }
    /// Owner-local expert for each destination-major row.
    pub fn packed_owner_local_experts(&self) -> &[usize] { self.plan.packed_owner_local_experts() }
    /// The actual retained account, without creating another allowance.
    pub fn funding(&self) -> &HostMetadataFunding { &self.funding }
}

/// A refused attempt retains every prior reservation on its original account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct FundedExpertRoutePackingFailure {
    #[source]
    cause: ExpertRoutePackingCause,
    funding: HostMetadataFunding,
}
impl FundedExpertRoutePackingFailure {
    /// Fixed source of the refusal.
    pub const fn cause(&self) -> ExpertRoutePackingCause { self.cause }
    /// Retained cumulative account; refusing an attempt does not refund it.
    pub fn funding(&self) -> &HostMetadataFunding { &self.funding }
}

#[derive(Clone, Copy)]
enum RouteIndices<'a> {
    Native(&'a [i32]),
    Host(&'a [usize]),
}
impl RouteIndices<'_> {
    fn len(self) -> usize { match self { Self::Native(v) => v.len(), Self::Host(v) => v.len() } }
    fn get(self, position: usize) -> Result<usize, ExpertRoutePackingCause> {
        match self {
            Self::Native(values) => usize::try_from(values[position]).map_err(|_|
                ExpertRoutePackingCause::NegativeExpert { position, expert: values[position] }),
            Self::Host(values) => Ok(values[position]),
        }
    }
    fn validated(self, position: usize) -> usize {
        match self { Self::Native(v) => v[position] as usize, Self::Host(v) => v[position] }
    }
}

struct PackingBuffers {
    send_counts: Vec<usize>,
    cursor: Vec<usize>,
    packed_route_positions: Vec<usize>,
    packed_token_indices: Vec<usize>,
    packed_global_experts: Vec<usize>,
    packed_owner_local_experts: Vec<usize>,
}
#[derive(Default)]
struct PackingCursor {
    offset: usize,
    position: usize,
    expert: usize,
    owner: usize,
    destination: usize,
}

impl<'source> ExpertRoutePackingSource<'source> {
    /// The same retained owner table supplies finite dynamic-region row ceilings.
    /// It does not choose routing IDs or issue tensor/native storage authority.
    pub fn region_source(self, source_tokens: usize, routes_per_token: usize, local_rank: usize)
        -> Result<super::ExpertRouteRegionSource<'source>, super::ExpertRouteRegionCause> {
        super::region::source(self.owners, self.owner_local_experts,
            self.geometry(source_tokens, routes_per_token)?, local_rank)
    }

    /// Borrowed immutable constructor-table payload retained by this source.
    /// It is distinct from each invocation's packing destinations.
    pub fn retained_table_bytes(self) -> Option<usize> {
        self.owners.len().checked_add(self.owner_local_experts.len())
            .and_then(|entries| entries.checked_mul(size_of::<usize>()))
    }

    /// Computes storage solely from this selected source and actual row/route geometry.
    /// It does not read selected tensor values or authorize their evaluation.
    pub fn geometry(self, source_tokens: usize, routes_per_token: usize)
        -> Result<ExpertRoutePackingGeometry, ExpertRoutePackingCause>
    {
        if routes_per_token == 0 || self.peers == 0 {
            return Err(ExpertRoutePackingCause::Geometry);
        }
        let routes = source_tokens.checked_mul(routes_per_token)
            .ok_or(ExpertRoutePackingCause::Overflow)?;
        // send_counts and insertion cursor, then four destination-major arrays.
        let entries = self.peers.checked_mul(2)
            .and_then(|peers| routes.checked_mul(4).and_then(|rows| rows.checked_add(peers)))
            .ok_or(ExpertRoutePackingCause::Overflow)?;
        let payload_bytes = entries.checked_mul(size_of::<usize>())
            .ok_or(ExpertRoutePackingCause::Overflow)?;
        // Match Vec's supported allocation range before any actual allocation.
        if payload_bytes > isize::MAX as usize {
            return Err(ExpertRoutePackingCause::Overflow);
        }
        Ok(ExpertRoutePackingGeometry { source_tokens, routes_per_token, routes,
            peers: self.peers, payload_bytes })
    }

    /// Fixed preparation, validation, reservation and result transports. Payload
    /// allocations are separately derived by `geometry`, never supplied as bytes.
    pub fn preparation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(), size_of::<ExpertRoutePackingGeometry>(),
            size_of::<ExpertRoutePackingPlan>(), size_of::<FundedExpertRoutePacking>(),
            size_of::<FundedExpertRoutePackingFailure>(),
            size_of::<Result<FundedExpertRoutePacking, FundedExpertRoutePackingFailure>>(),
            size_of::<Result<ExpertRoutePackingGeometry, ExpertRoutePackingCause>>(),
            size_of::<Result<(), ExpertRoutePackingCause>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<HostMetadataFunding>(), size_of::<&[usize]>(),
            size_of::<RouteIndices<'_>>(),
            size_of::<PackingBuffers>(), size_of::<PackingCursor>(),
            size_of::<(Self, ExpertRoutePackingGeometry, RouteIndices<'_>)>(),
            size_of::<(usize, usize, Result<usize, ExpertRoutePackingCause>)>(),
            size_of::<std::ops::Range<usize>>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Builds an admitted host packing from already completed route values.
    /// The account pays before validation/work, and remains attached on failure.
    /// This supplies no native readout, collective, or numerical storage grant.
    pub fn prepare(self, source_tokens: usize, routes_per_token: usize,
        global_experts: &[usize], funding: HostMetadataFunding)
        -> Result<FundedExpertRoutePacking, FundedExpertRoutePackingFailure>
    {
        self.prepare_indices(source_tokens, routes_per_token, RouteIndices::Host(global_experts), funding)
    }

    /// Uses the exact completed native i32 ID slice without making a converted copy.
    pub fn prepare_i32(self, source_tokens: usize, routes_per_token: usize,
        global_experts: &[i32], funding: HostMetadataFunding)
        -> Result<FundedExpertRoutePacking, FundedExpertRoutePackingFailure>
    {
        self.prepare_indices(source_tokens, routes_per_token, RouteIndices::Native(global_experts), funding)
    }

    fn prepare_indices(self, source_tokens: usize, routes_per_token: usize,
        global_experts: RouteIndices<'_>, funding: HostMetadataFunding)
        -> Result<FundedExpertRoutePacking, FundedExpertRoutePackingFailure>
    {
        let build = || {
            let controls = Self::preparation_control_bytes()
                .ok_or(ExpertRoutePackingCause::Overflow)?;
            funding.reserve_metadata(controls).map_err(ExpertRoutePackingCause::Funding)?;
            let geometry = self.geometry(source_tokens, routes_per_token)?;
            self.validate(geometry, global_experts)?;
            funding.reserve_metadata(geometry.payload_bytes)
                .map_err(ExpertRoutePackingCause::Funding)?;
            Ok::<_, ExpertRoutePackingCause>(self.pack_validated(geometry, global_experts))
        };
        match build() {
            Ok(plan) => Ok(FundedExpertRoutePacking { plan, funding }),
            Err(cause) => Err(FundedExpertRoutePackingFailure { cause, funding }),
        }
    }

    pub(super) fn pack(self, source_tokens: usize, routes_per_token: usize,
        global_experts: &[usize]) -> Result<ExpertRoutePackingPlan, ExpertRoutePackingCause>
    {
        self.pack_indices(source_tokens, routes_per_token, RouteIndices::Host(global_experts))
    }

    /// Ordinary construction from the completed native ID slice. Managed callers
    /// use `prepare_i32` with their original metadata account instead.
    pub(crate) fn pack_i32(self, source_tokens: usize, routes_per_token: usize,
        global_experts: &[i32]) -> Result<ExpertRoutePackingPlan, ExpertRoutePackingCause>
    {
        self.pack_indices(source_tokens, routes_per_token, RouteIndices::Native(global_experts))
    }

    fn pack_indices(self, source_tokens: usize, routes_per_token: usize,
        global_experts: RouteIndices<'_>) -> Result<ExpertRoutePackingPlan, ExpertRoutePackingCause>
    {
        let geometry = self.geometry(source_tokens, routes_per_token)?;
        self.validate(geometry, global_experts)?;
        Ok(self.pack_validated(geometry, global_experts))
    }

    fn validate(self, geometry: ExpertRoutePackingGeometry, global_experts: RouteIndices<'_>)
        -> Result<(), ExpertRoutePackingCause>
    {
        if global_experts.len() != geometry.routes {
            return Err(ExpertRoutePackingCause::Geometry);
        }
        for position in 0..global_experts.len() {
            let expert = global_experts.get(position)?;
            if expert >= self.owners.len() {
                return Err(ExpertRoutePackingCause::Expert { position, expert });
            }
        }
        Ok(())
    }

    fn pack_validated(self, geometry: ExpertRoutePackingGeometry, global_experts: RouteIndices<'_>)
        -> ExpertRoutePackingPlan
    {
        let mut output = PackingBuffers {
            send_counts: vec![0usize; geometry.peers],
            cursor: vec![0usize; geometry.peers],
            packed_route_positions: vec![0usize; geometry.routes],
            packed_token_indices: vec![0usize; geometry.routes],
            packed_global_experts: vec![0usize; geometry.routes],
            packed_owner_local_experts: vec![0usize; geometry.routes],
        };
        let mut work = PackingCursor::default();
        while work.position < global_experts.len() {
            output.send_counts[self.owners[global_experts.validated(work.position)]] += 1;
            work.position += 1;
        }
        while work.owner < geometry.peers {
            output.cursor[work.owner] = work.offset;
            work.offset += output.send_counts[work.owner];
            work.owner += 1;
        }
        work.position = 0;
        while work.position < global_experts.len() {
            work.expert = global_experts.validated(work.position);
            work.owner = self.owners[work.expert];
            work.destination = output.cursor[work.owner];
            output.cursor[work.owner] += 1;
            output.packed_route_positions[work.destination] = work.position;
            output.packed_token_indices[work.destination] = work.position / geometry.routes_per_token;
            output.packed_global_experts[work.destination] = work.expert;
            output.packed_owner_local_experts[work.destination] = self.owner_local_experts[work.expert];
            work.position += 1;
        }
        ExpertRoutePackingPlan {
            source_tokens: geometry.source_tokens, routes_per_token: geometry.routes_per_token,
            send_counts: output.send_counts,
            packed_route_positions: output.packed_route_positions,
            packed_token_indices: output.packed_token_indices,
            packed_global_experts: output.packed_global_experts,
            packed_owner_local_experts: output.packed_owner_local_experts,
        }
    }
}
