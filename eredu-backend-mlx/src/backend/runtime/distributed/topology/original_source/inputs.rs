//! Shared source-only initialized leaf worker for finite control protocols.
use super::*;
use safemlx::{
    PreparedInputArena, PreparedInputLayout, PreparedInputLeaf, PreparedInputPlan,
    PreparedSubmissionGraphQuota,
};
#[derive(Clone)]
struct Custody {
    source: RetainedCommunicationSource,
    native: Option<super::preparation::SharedPreparationCustody>,
    funding: HostMetadataFunding,
}
// Provisional arrays/leaves retire before the arena/preparation and H on every
// failed prefix. No native roots are stored in the account-only custody.
struct Progress<const N: usize> {
    arrays: [Option<Array>; N],
    leaves: [Option<PreparedInputLeaf>; N],
    arena: Option<PreparedInputArena>,
    prepared: Option<PreparedSubmissionGraphQuota<Custody>>,
    rejected: Option<Custody>,
}
/// A completed leaf minted only by this source's initialized input producer.
/// Numerical data and its descriptor retire before the retained source funding.
pub(in crate::backend::runtime::distributed::topology::original_source) struct CompletedCommunicationInput
{
    value: Array,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
impl CompletedCommunicationInput {
    pub(super) fn value(&self) -> &Array {
        &self.value
    }
    pub(super) fn belongs_to(&self, source: &OriginalCommunicationSource<'_>) -> bool {
        self.source.same_source(source.source()) && self.funding.same_account(source.funding())
    }
}
pub(super) fn construct_retained<const N: usize>(
    source: &OriginalCommunicationSource<'_>,
    plans: [PreparedInputPlan<'_>; N],
) -> Result<[CompletedCommunicationInput; N], Error> {
    reserve(
        source.funding(),
        &[
            size_of::<[CompletedCommunicationInput; N]>(),
            size_of::<Result<[CompletedCommunicationInput; N], Error>>(),
            size_of::<[Array; N]>(),
        ],
    )?;
    let values = construct(source, plans)?;
    Ok(values.map(|value| CompletedCommunicationInput {
        value,
        source: source.source().clone(),
        funding: source.funding().clone(),
    }))
}
pub(super) fn construct<const N: usize>(
    source: &OriginalCommunicationSource<'_>,
    plans: [PreparedInputPlan<'_>; N],
) -> Result<[Array; N], Error> {
    construct_inner(source, plans, None)
}
pub(super) fn construct_admitted<const N: usize>(
    source: &OriginalCommunicationSource<'_>,
    plans: [PreparedInputPlan<'_>; N],
    custody: eredu_runtime::working_memory::CommunicationPreparationCustody,
) -> Result<[Array; N], Error> {
    construct_inner(source, plans, Some(custody))
}
pub(super) fn storage_bytes<const N: usize>(
    plans: &[PreparedInputPlan<'_>; N],
) -> Result<usize, eredu_runtime::working_memory::WorkingMemoryError> {
    let layouts: [PreparedInputLayout; N] = std::array::from_fn(|i| plans[i].layout());
    let source = layout_storage_bytes(&layouts)?;
    let shared = super::preparation::SharedPreparationCustody::allocation_bytes()
        .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)?;
    source
        .checked_add(shared)
        .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
}
/// Same leaf/arena quotation before protocol words exist. Facts carry no input
/// constructor or native authority; the actual worker still takes borrowed plans.
pub(super) fn source_layout_storage_bytes<const N: usize>(
    layouts: &[PreparedInputLayout; N],
) -> Result<usize, eredu_runtime::working_memory::WorkingMemoryError> {
    layout_storage_bytes(layouts)?
        .checked_add(size_of::<[usize; 3]>())
        .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)?
        .checked_add(
            construct_control_bytes::<N>()
                .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)?,
        )
        .ok_or(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
}
fn layout_storage_bytes<const N: usize>(
    layouts: &[PreparedInputLayout; N],
) -> Result<usize, eredu_runtime::working_memory::WorkingMemoryError> {
    use eredu_runtime::working_memory::WorkingMemoryError as W;
    let wrapper = PreparedInputLeaf::array_layout().map_err(|_| W::UnknownBound)?;
    let mut metadata = wrapper.metadata_bytes().checked_mul(N).ok_or(W::Overflow)?;
    let mut backing = 0usize;
    let mut controls = wrapper.control_bytes().checked_mul(N).ok_or(W::Overflow)?;
    for layout in layouts {
        metadata = metadata
            .checked_add(layout.metadata_bytes())
            .ok_or(W::Overflow)?;
        backing = backing
            .checked_add(layout.backing_bytes())
            .ok_or(W::Overflow)?;
        controls = controls
            .checked_add(layout.control_bytes())
            .ok_or(W::Overflow)?;
    }
    let arena = PreparedInputArena::layout::<Custody>(metadata).map_err(|_| W::UnknownBound)?;
    arena
        .total_bytes()
        .and_then(|n| n.checked_add(backing))
        .and_then(|n| n.checked_add(controls))
        .ok_or(W::Overflow)
}

fn construct_control_bytes<const N: usize>() -> Option<usize> {
    let parts = [
        size_of::<Option<eredu_runtime::working_memory::CommunicationPreparationCustody>>(),
        size_of::<Option<super::preparation::SharedPreparationCustody>>(),
        size_of::<Progress<N>>(),
        size_of::<Custody>(),
        size_of::<[Array; N]>(),
        size_of::<[PreparedInputPlan<'_>; N]>(),
        size_of::<[Option<Array>; N]>(),
        size_of::<Result<[Array; N], Error>>(),
        size_of::<(&OriginalCommunicationSource<'_>, [PreparedInputPlan<'_>; N])>(),
        size_of::<Result<safemlx::PreparedInputArrayLayout, safemlx::PreparedInputCause>>(),
        size_of::<Result<safemlx::SubmissionGraphQuotaLayout, safemlx::SubmissionGraphQuotaCause>>(
        ),
        size_of::<
            Result<
                PreparedSubmissionGraphQuota<Custody>,
                safemlx::SubmissionGraphQuotaError<Custody>,
            >,
        >(),
        size_of::<
            Result<
                PreparedInputArena,
                safemlx::SubmissionGraphQuotaError<PreparedSubmissionGraphQuota<Custody>>,
            >,
        >(),
        size_of::<Result<PreparedInputLeaf, safemlx::PreparedInputCause>>(),
        size_of::<Result<Array, safemlx::PreparedInputCause>>(),
        size_of::<std::array::IntoIter<PreparedInputPlan<'_>, N>>(),
        size_of::<(usize, PreparedInputPlan<'_>)>(),
        failure_control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn construct_inner<const N: usize>(
    source: &OriginalCommunicationSource<'_>,
    plans: [PreparedInputPlan<'_>; N],
    native: Option<eredu_runtime::working_memory::CommunicationPreparationCustody>,
) -> Result<[Array; N], Error> {
    source
        .funding()
        .reserve_metadata(construct_control_bytes::<N>().ok_or_else(overflow)?)
        .map_err(Error::WorkspacePlanning)?;
    source.validate()?;
    let fail = |cause| failure(cause, source.source(), source.funding());
    let wrapper = PreparedInputLeaf::array_layout().map_err(|cause| fail(Cause::Input(cause)))?;
    let mut metadata = wrapper
        .metadata_bytes()
        .checked_mul(N)
        .ok_or_else(overflow)?;
    let mut backing = 0usize;
    let mut controls = wrapper
        .control_bytes()
        .checked_mul(N)
        .ok_or_else(overflow)?;
    for plan in &plans {
        metadata = metadata
            .checked_add(plan.metadata_bytes())
            .ok_or_else(overflow)?;
        backing = backing
            .checked_add(plan.backing_bytes())
            .ok_or_else(overflow)?;
        controls = controls
            .checked_add(plan.control_bytes())
            .ok_or_else(overflow)?;
    }
    let arena = PreparedInputArena::layout::<Custody>(metadata)
        .map_err(|cause| fail(Cause::SourceGraph(cause)))?;
    if native.is_none() {
        reserve(
            source.funding(),
            &[arena.total_bytes().ok_or_else(overflow)?, backing, controls],
        )?;
    }
    let custody = Custody {
        source: source.source().clone(),
        native: native.map(super::preparation::SharedPreparationCustody::new),
        funding: source.funding().clone(),
    };
    let mut progress: Progress<N> = Progress {
        arrays: std::array::from_fn(|_| None),
        leaves: std::array::from_fn(|_| None),
        arena: None,
        prepared: None,
        rejected: None,
    };
    let prepared = match PreparedSubmissionGraphQuota::try_new(metadata, custody.clone()) {
        Ok(value) => value,
        Err(error) => {
            let (cause, owner) = error.into_parts();
            progress.rejected = Some(owner);
            return Err(fail(Cause::SourceGraph(cause)));
        }
    };
    progress.arena = Some(match PreparedInputArena::try_allocate(prepared) {
        Ok(value) => value,
        Err(error) => {
            let (cause, owner) = error.into_parts();
            progress.prepared = Some(owner);
            return Err(fail(Cause::SourceGraph(cause)));
        }
    });
    for (index, plan) in plans.into_iter().enumerate() {
        progress.leaves[index] = Some(
            plan.construct(progress.arena.as_ref().expect("prepared source arena"))
                .map_err(|cause| fail(Cause::Input(cause)))?,
        );
        progress.arrays[index] = Some(
            progress.leaves[index]
                .as_ref()
                .expect("completed source leaf")
                .try_source_array()
                .map_err(|cause| fail(Cause::Input(cause)))?,
        );
    }
    Ok(std::array::from_fn(|index| {
        progress.arrays[index]
            .take()
            .expect("completed source array")
    }))
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn reserve(funding: &HostMetadataFunding, bytes: &[usize]) -> Result<(), Error> {
    funding
        .reserve_metadata(
            bytes
                .iter()
                .copied()
                .try_fold(size_of_val(bytes), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)
}
