//! Native retained-storage facts converted once into portable metadata roots.

use super::*;
use eredu_nn::workspace::WorkspaceMetadataError;
use safemlx::{Array, Dtype};
use std::{collections::TryReserveError, mem::size_of};
mod owned;
mod preparation;
mod representation;
pub(crate) use owned::OwnedArrayProjection;
mod paged_sources;
mod roots;
pub(crate) use paged_sources::{
    OriginalPagedAppendClaim, OriginalPagedVisibleClaim, OriginalPagedAttentionBlock, OriginalPagedBlockSource,
    OriginalPagedDiskWriteSource, OriginalPagedHostEviction, OriginalPagedHostReturn,
    OriginalPagedDiscard, OriginalPagedScanClaim, OriginalPagedScanSource, PagedAppendInput, PagedHostStoreDeclaration,
    PagedScanInput, PagedScopeRetention, ProjectedPagedSources,
};
use preparation::projected_dtype;
pub use preparation::{ProjectionSourceError, ProjectionSourceLayout};

#[derive(Debug)]
enum ProjectionRow<A> {
    Known {
        identity: safemlx::AllocationIdentity,
        bytes: u64,
        storage: WorkspaceExistingStorage,
        array: A,
    },
    Unknown(A),
}
impl<A> ProjectionRow<A> {
    fn identity(&self) -> safemlx::AllocationIdentity {
        match self {
            Self::Known { identity, .. } => *identity,
            Self::Unknown(_) => unreachable!("known projection prefix"),
        }
    }
    fn map<B, E>(self, clone: &mut impl FnMut(A) -> Result<B, E>) -> Result<ProjectionRow<B>, E> {
        Ok(match self {
            Self::Known {
                identity,
                bytes,
                storage,
                array,
            } => ProjectionRow::Known {
                identity,
                bytes,
                storage,
                array: clone(array)?,
            },
            Self::Unknown(array) => ProjectionRow::Unknown(clone(array)?),
        })
    }
}

/// One physical-identity row may own both a completed array descriptor and an
/// immutable host handle. Their shared allocation is represented once, even
/// when the two native source kinds arrive through different lexical loans.
#[derive(Debug)]
struct ProjectionNativeSource {
    array: Option<Array>,
    host: Option<std::sync::Arc<safemlx::ImmutableHostTransferBuffer>>,
}
impl ProjectionNativeSource {
    fn array(value: Array) -> Self {
        Self {
            array: Some(value),
            host: None,
        }
    }
    fn host(value: std::sync::Arc<safemlx::ImmutableHostTransferBuffer>) -> Self {
        Self {
            array: None,
            host: Some(value),
        }
    }
}

/// Fixed inventory planning/refusal, without native state or formatted text.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionInventoryError {
    /// Requested row/control layout cannot be represented.
    #[error("native projection inventory layout overflow")]
    Overflow,
    /// A new physical root or unknown witness exceeds the declared source count.
    #[error("native projection inventory exhausted")]
    Exhausted,
    /// A caller attempted to overwrite a retained handoff destination.
    #[error("native projection storage destination is already occupied")]
    Destination,
    /// Actual final row allocation failed before native witness cloning.
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    /// The participating context refused its admitted metadata allowance.
    #[error(transparent)]
    Metadata(#[from] Error),
}

/// Two actual row extents for the borrowed-to-retained handoff, plus fixed
/// inventory controls. Portable root/context/layout metadata and native clone
/// shells are separate owning producers; this is not complete copy admission.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionInventoryLayout {
    sources: usize,
    requested_bytes: usize,
}
impl ProjectionInventoryLayout {
    /// Prices one known-or-unknown row per supplied source. Aliases use one root.
    pub fn new(sources: usize) -> Option<Self> {
        let borrowed = WorkspaceContext::metadata_vec_bytes::<ProjectionRow<&Array>>(sources)?;
        let retained =
            WorkspaceContext::metadata_vec_bytes::<ProjectionRow<ProjectionNativeSource>>(sources)?;
        let parts = [
            borrowed,
            retained,
            size_of::<ExistingArrayProjection<'_>>(),
            size_of::<ProjectedNativeStorage>(),
            size_of::<Option<ProjectedNativeStorage>>(),
            size_of::<&mut Option<ProjectedNativeStorage>>(),
            size_of::<Result<(), ProjectionSourceError>>(),
            size_of::<std::iter::Enumerate<std::vec::IntoIter<ProjectionRow<&Array>>>>(),
            size_of::<Self>(),
            size_of::<ProjectionInventoryError>(),
            size_of::<Vec<ProjectionRow<&Array>>>(),
            size_of::<Vec<ProjectionRow<ProjectionNativeSource>>>(),
            size_of::<Result<ExistingArrayProjection<'_>, ProjectionInventoryError>>(),
            size_of::<Result<ProjectedNativeStorage, Error>>(),
            size_of::<Result<usize, usize>>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<std::vec::IntoIter<ProjectionRow<&Array>>>(),
            size_of::<ProjectionRow<&Array>>(),
            size_of::<ProjectionRow<ProjectionNativeSource>>(),
            Error::retained_source_construction_bytes::<ProjectionInventoryError>()?,
            size_of::<Result<ProjectionRow<ProjectionNativeSource>, std::convert::Infallible>>(),
            size_of::<Result<ProjectionRow<ProjectionNativeSource>, Error>>(),
        ];
        let requested_bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        Some(Self {
            sources,
            requested_bytes,
        })
    }
    /// Exact row requests and named inventory controls; no allocation or grant.
    pub fn requested_bytes(self) -> usize {
        self.requested_bytes
    }
    /// Reserves the declared borrowed inventory once. Final owned rows are
    /// reserved from the same source count before the handoff clones any array.
    pub fn construct(
        self,
        context: &WorkspaceContext,
    ) -> Result<ExistingArrayProjection<'_>, ProjectionInventoryError> {
        context
            .charge_metadata(
                ExistingArrayProjection::inventory_control_bytes()
                    .ok_or_else(|| Error::from(WorkspaceMetadataError::Overflow))?,
            )
            .map_err(Error::from)?;
        let rows = context.metadata_vec(self.sources)?;
        Ok(ExistingArrayProjection {
            context,
            rows,
            known: 0,
            maximum: Some(self.sources),
            metadata_started: true,
        })
    }
}

/// Temporary native owners and their exact portable storage roots. Keep this
/// inventory through registry binding and admission, then discard it; an
/// immutable quote must not retain the old native state through these handles.
pub struct ProjectedNativeStorage {
    rows: Vec<ProjectionRow<ProjectionNativeSource>>,
    known: usize,
    // Exact canonical block sources retire after all native array witnesses.
    paged_sources: Vec<crate::backend::runtime::cache::kv::ProjectedPagedSource>,
    copy_sources: Vec<crate::backend::runtime::cache::residency::PinnedCacheSource>,
    // Native handles and metadata rows retire before their actual paying H.
    _funding: Option<HostMetadataFunding>,
    _host_preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl std::fmt::Debug for ProjectedNativeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedNativeStorage")
            .field("rows", &self.rows)
            .field("known", &self.known)
            .field("paged_sources", &self.paged_sources.len())
            .finish_non_exhaustive()
    }
}
impl ProjectedNativeStorage {
    pub(crate) fn paged_sources(
        &self,
    ) -> &[crate::backend::runtime::cache::kv::ProjectedPagedSource] {
        &self.paged_sources
    }
    /// False if any projected source lacked certified immutable/completed backing. Known
    /// entries alone must never claim complete decoder coverage.
    pub fn is_complete(&self) -> bool {
        self.known == self.rows.len()
    }
    /// Unique physical roots in native identity order, with retained witnesses.
    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = (safemlx::AllocationIdentity, u64, &WorkspaceExistingStorage)>
    {
        self.rows[..self.known].iter().map(|row| match row {
            ProjectionRow::Known {
                identity,
                bytes,
                storage,
                ..
            } => (*identity, *bytes, storage),
            ProjectionRow::Unknown(_) => unreachable!("known projection prefix"),
        })
    }
    /// Borrows the retained native witness without polling or querying it.
    pub fn native_array(&self, identity: safemlx::AllocationIdentity) -> Option<&Array> {
        let index = self.rows[..self.known]
            .binary_search_by_key(&identity, ProjectionRow::identity)
            .ok()?;
        match &self.rows[index] {
            ProjectionRow::Known { array, .. } => array.array.as_ref(),
            ProjectionRow::Unknown(_) => unreachable!("known projection prefix"),
        }
    }
    /// Borrows the actual retained immutable host witness. A physical identity
    /// or portable storage root alone cannot create this source owner.
    pub(crate) fn native_host(
        &self,
        identity: safemlx::AllocationIdentity,
    ) -> Option<&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>> {
        let index = self.rows[..self.known]
            .binary_search_by_key(&identity, ProjectionRow::identity)
            .ok()?;
        match &self.rows[index] {
            ProjectionRow::Known { array, .. } => array.host.as_ref(),
            ProjectionRow::Unknown(_) => unreachable!("known projection prefix"),
        }
    }
}

/// Actual resident state projection paired with temporary physical witnesses.
/// Both are produced by one traversal, preserving aliases across all roles.
pub struct ProjectedResidentState {
    /// Exact portable state projection sharing the inventory's metadata roots.
    pub state: eredu_runtime::DeviceState<
        WorkspaceBackend,
        eredu_runtime::working_memory::WorkspaceResidentLayerState,
    >,
    /// Temporary native witnesses and their unique physical-root mapping.
    pub storage: ProjectedNativeStorage,
}

impl ProjectedResidentState {
    /// Separates numerical metadata from temporary native storage witnesses.
    pub fn into_parts(
        self,
    ) -> (
        eredu_runtime::DeviceState<
            WorkspaceBackend,
            eredu_runtime::working_memory::WorkspaceResidentLayerState,
        >,
        ProjectedNativeStorage,
    ) {
        (self.state, self.storage)
    }
}

/// One retained native inventory, including aliases across different layers or
/// state roles. Borrowed native owners must outlive the complete projection.
pub struct ExistingArrayProjection<'a> {
    context: &'a WorkspaceContext,
    rows: Vec<ProjectionRow<&'a Array>>,
    known: usize,
    maximum: Option<usize>,
    metadata_started: bool,
}
impl<'a> ExistingArrayProjection<'a> {
    /// Checked context metadata for a counted projection and retained handoff.
    /// `sources` bounds distinct physical/unknown rows; `imports` counts actual
    /// descriptor visits, including reprojections of an existing source. Shared
    /// shape/root births and the caller's returned tensor vector are separate.
    pub fn metadata_control_bytes(sources: usize, imports: usize) -> Option<usize> {
        let inventory = ProjectionInventoryLayout::new(sources)?.requested_bytes();
        let imports = Self::import_control_bytes()?.checked_mul(imports)?;
        let clones = Self::clone_control_bytes()?.checked_mul(sources)?;
        // A participating worker stops on its first error. Context capacity
        // refusals are inline; all other typed sources use this shared query.
        let error = WorkspaceContext::metadata_source_bytes::<ProjectionSourceError>()?
            .max(WorkspaceContext::metadata_source_bytes::<
                ProjectionInventoryError,
            >()?)
            .max(WorkspaceContext::metadata_source_bytes::<
                safemlx::ArrayDescriptorError,
            >()?);
        inventory
            .checked_add(imports)?
            .checked_add(clones)?
            .checked_add(error)
    }

    pub(super) fn inventory_control_bytes() -> Option<usize> {
        Some(ProjectionInventoryLayout::new(0)?.requested_bytes())
    }

    fn start_metadata(&mut self) -> Result<(), ProjectionSourceError> {
        if !self.metadata_started {
            self.context
                .charge_metadata(
                    Self::inventory_control_bytes().ok_or(ProjectionSourceError::Overflow)?,
                )
                .map_err(Error::from)?;
            self.metadata_started = true;
        }
        Ok(())
    }

    /// The root visitor supplies an exact initial count before any descriptor
    /// loan. Later distinct rows still use the same checked growth worker.
    pub(super) fn reserve_sources(&mut self, sources: usize) -> Result<(), ProjectionSourceError> {
        self.start_metadata()?;
        if self.maximum.is_none() && self.rows.is_empty() && self.rows.capacity() < sources {
            self.rows = self.context.metadata_vec(sources)?;
        }
        Ok(())
    }

    /// Dynamically growing inventory using the supplied context's metadata
    /// policy. A counted root traversal reserves its initial rows before import.
    pub fn new(context: &'a WorkspaceContext) -> Self {
        Self {
            context,
            rows: Vec::new(),
            known: 0,
            maximum: None,
            metadata_started: false,
        }
    }
    /// Uses the exact finite source-count inventory producer. This does not pay
    /// for portable root/context metadata or native clone shells.
    pub fn with_source_count(
        context: &'a WorkspaceContext,
        sources: usize,
    ) -> Result<Self, ProjectionInventoryError> {
        ProjectionInventoryLayout::new(sources)
            .ok_or(ProjectionInventoryError::Overflow)?
            .construct(context)
    }
    /// Context shared by every projected role and layer.
    pub fn context(&self) -> &'a WorkspaceContext {
        self.context
    }
    /// Whether all successfully projected sources had certified physical backing.
    pub fn is_complete(&self) -> bool {
        self.known == self.rows.len()
    }
    /// Exact imported roots, borrowing the same first native witness per identity.
    pub(crate) fn storage_roots(
        &self,
    ) -> impl Iterator<Item = (safemlx::AllocationIdentity, u64, &WorkspaceExistingStorage)> {
        self.rows[..self.known].iter().map(|row| match row {
            ProjectionRow::Known {
                identity,
                bytes,
                storage,
                ..
            } => (*identity, *bytes, storage),
            ProjectionRow::Unknown(_) => unreachable!("known projection prefix"),
        })
    }
    /// Legacy infallible ordinary handoff. Checked copy preparation consumes
    /// `try_into_storage` or `into_prepared_storage` so every clone can refuse.
    pub fn into_storage(self) -> ProjectedNativeStorage {
        let mut rows = Vec::with_capacity(self.maximum.unwrap_or(self.rows.len()));
        for row in self.rows {
            rows.push(
                row.map(&mut |array: &Array| {
                    Ok::<_, std::convert::Infallible>(ProjectionNativeSource::array(array.clone()))
                })
                .unwrap(),
            );
        }
        ProjectedNativeStorage {
            rows,
            known: self.known,
            paged_sources: Vec::new(),
            copy_sources: Vec::new(),
            _host_preparation: None,
            _funding: self.context.metadata_funding(),
        }
    }
    /// The same sorted handoff using closed inspection clones. Reserve final
    /// rows before the first clone; a failure drops every retained prefix.
    pub fn try_into_storage(self) -> Result<ProjectedNativeStorage, Error> {
        let context = self.context;
        self.into_prepared_storage()
            .map_err(|cause| cause.in_context(context))
    }

    /// Imports one array without evaluation or native housekeeping. The retained
    /// borrow keeps its backing alive. The actual runtime owner holds the shape
    /// loan through layout construction without copying a descriptor snapshot.
    /// Runtime contention remains a typed source error. Checked contexts admit
    /// each metadata destination and retained error before its allocation.
    pub fn project(&mut self, array: &'a Array) -> Result<WorkspaceTensor, Error> {
        self.charge_import()
            .map_err(|cause| cause.in_context(self.context))?;
        let metadata = array
            .try_descriptor()
            .map_err(|cause| self.context.metadata_source(cause))?;
        let dtype = projected_dtype(metadata.facts().dtype())
            .map_err(|cause| cause.in_context(self.context))?;
        let storage = self
            .storage_for(array, metadata.facts().allocation())
            .map_err(|cause| cause.in_context(self.context))?;
        WorkspaceTensor::existing_with_storage(
            self.context
                .layout(metadata.shape(), dtype)?
                .with_representation(representation::from_descriptor(&metadata)),
            &storage,
            self.context,
        )
    }
}

/// Projects one complete inventory of retained arrays without evaluation,
/// polling, waiting or native allocation. Completed native backing capacities
/// and alias identities are preserved; unfinished/foreign storage stays unknown.
///
/// Keep the source state exclusively retained while using this projection to
/// quote and admit work. This is existing-residency evidence, not a bound on
/// future allocation. The selected workspace mechanisms price future equations.
pub fn project_existing_arrays<'a>(
    arrays: impl IntoIterator<Item = &'a Array>,
    context: &WorkspaceContext,
) -> Result<Vec<WorkspaceTensor>, Error> {
    let mut projection = ExistingArrayProjection::new(context);
    arrays
        .into_iter()
        .map(|array| projection.project(array))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Stream, ops::indexing::TryIndexOp};

    #[derive(Debug)]
    struct NoOperations;
    impl WorkspaceMechanisms for NoOperations {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            panic!("state projection must not execute an equation");
        }
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn native_borrowed_projection_preserves_exact_contiguity_without_new_storage() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let root = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let view = root.transpose_axes(&[1, 0], &stream).unwrap();
        view.evaluated().unwrap();
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::with_source_count(&context, 2).unwrap();
        let row = projection.project(&root).unwrap();
        let column = projection.project(&view).unwrap();
        assert!(row.layout().representation().unwrap().row_contiguous());
        assert!(!column.layout().representation().unwrap().row_contiguous());
        assert_eq!(row.layout().representation().unwrap().dtype(),
            eredu_nn::workspace::WorkspaceFloatingType::Float32);
        assert_eq!(column.layout().shape(), &[3, 2]);
        let column_rep = column.layout().representation().unwrap();
        assert!(!column_rep.last_axis_contiguous());
        assert_eq!(column_rep.dense_axis_at(2, 0), Some(1));
        assert_eq!(column_rep.dense_axis_at(2, 1), Some(0));
        let storage = projection.into_storage();
        assert!(storage.is_complete());
        assert_eq!(storage.iter().len(), 1, "layout evidence adds no backing");
        assert_eq!(context.report(&[row, column]).unwrap().operations.len(), 0);

        // Repeated extents cannot identify a permutation; actual strides do.
        let cache = Array::from_slice(&(0..32).map(|n| n as f32).collect::<Vec<_>>(), &[1, 2, 2, 8]);
        let permuted = cache.transpose_axes(&[0, 2, 1, 3], &stream).unwrap();
        let sparse = cache.try_index_device((.., .., 1.., ..), &stream).unwrap();
        permuted.evaluated().unwrap();
        sparse.evaluated().unwrap();
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::with_source_count(&context, 3).unwrap();
        let root = projection.project(&cache).unwrap();
        let retained = projection.project_prepared(&permuted).unwrap();
        let sparse = projection.project(&sparse).unwrap();
        let rep = retained.layout().representation().unwrap();
        assert!(!rep.row_contiguous());
        assert!(rep.last_axis_contiguous());
        assert_eq!((0..4).map(|position| rep.dense_axis_at(4, position)).collect::<Vec<_>>(),
            vec![Some(0), Some(2), Some(1), Some(3)]);
        let sparse_rep = sparse.layout().representation().unwrap();
        assert!(!sparse_rep.row_contiguous());
        assert!(sparse_rep.last_axis_contiguous());
        assert_eq!(sparse_rep.dense_axis_at(4, 0), None, "strided slice is not dense");
        assert_eq!((0..4).map(|axis| sparse_rep.element_stride_at(4, axis)).collect::<Vec<_>>(),
            vec![Some(1), Some(16), Some(1), Some(1)]);

        let storage = projection.into_storage();
        assert!(storage.is_complete());
        assert_eq!(storage.iter().len(), 1, "all views retain the original backing");
        assert_eq!(context.report(&[root, retained, sparse]).unwrap().operations.len(), 0);

        let root = Array::from_slice(&(0..12).map(|n| n as f32).collect::<Vec<_>>(), &[12]);
        for (strides, offset, expected) in [
            ([6, 2], 0, [Some(6), Some(2)]),
            ([6, -2], 4, [None, None]),
            ([0, 1], 0, [None, None]),
        ] {
            let view = root.as_strided(&[2, 3][..], &strides[..], offset, &stream).unwrap();
            view.evaluated().unwrap();
            let context = WorkspaceContext::new(NoOperations);
            let mut projection = ExistingArrayProjection::new(&context);
            let value = projection.project(&view).unwrap();
            let representation = value.layout().representation().unwrap();
            assert!(!representation.row_contiguous());
            assert_eq!([representation.element_stride_at(2, 0),
                representation.element_stride_at(2, 1)], expected);
            assert_eq!(projection.into_storage().iter().len(), 1);
            assert!(context.report(&[value]).unwrap().operations.is_empty());
        }
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn native_storage_projection_preserves_capacity_aliases_and_unknown_lazy_state() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let root = Array::from_slice(&(0..120).map(|n| n as f32).collect::<Vec<_>>(), &[2, 20, 3]);
        let view = root.try_index_device((.., 18.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        let info = root.allocation_info().unwrap().unwrap();
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::with_source_count(&context, 3).unwrap();
        let projected = [&root, &view, &view]
            .into_iter()
            .map(|array| projection.project(array).unwrap())
            .collect::<Vec<_>>();
        let storage = projection.into_storage();
        assert!(storage.is_complete());
        assert_eq!(storage.iter().len(), 1);
        let (identity, bytes, portable) = storage.iter().next().unwrap();
        assert_eq!(identity, info.identity());
        assert_eq!(bytes, info.bytes() as u64);
        assert_eq!(portable.capacity_bytes(), Some(bytes));
        let borrowed =
            WorkspaceBorrowedStorage::new(&context, storage.iter().map(|(_, _, root)| root))
                .unwrap();
        assert_eq!(borrowed.total_bytes(), bytes);
        assert!(borrowed.roots()[0].same_storage(portable));
        assert_eq!(
            storage
                .native_array(identity)
                .unwrap()
                .allocation_info()
                .unwrap(),
            Some(info)
        );
        assert_eq!(projected[1].layout().shape(), view.shape());
        context.set_borrowed_storage(borrowed.clone()).unwrap();
        context.begin_state_span(&projected).unwrap();
        let report = context.report(&projected[1..]).unwrap();
        assert!(report.operations.is_empty());
        assert_eq!(
            report.state.as_ref().unwrap().retained_bytes,
            Some(info.bytes() as u64)
        );
        assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        assert_eq!(report.inference_transient_bytes(), Some(0));
        let residual = report.residual.as_ref().unwrap();
        assert!(residual.borrowed_storage.same_identity(&borrowed));
        assert_eq!(residual.total_bytes, Some(0));
        assert_eq!(residual.retained_bytes, Some(0));
        let report = context.report(&[]).unwrap();
        assert_eq!(
            report.inference_transient_bytes(),
            Some(info.bytes() as u64)
        );

        let lazy = root.square(&stream).unwrap();
        let mut empty = ExistingArrayProjection::with_source_count(&context, 0).unwrap();
        let refusal = empty.project(&lazy).unwrap_err();
        assert!(matches!(
            std::error::Error::source(&refusal)
                .unwrap()
                .downcast_ref::<ProjectionInventoryError>(),
            Some(ProjectionInventoryError::Exhausted)
        ));
        assert_eq!(empty.storage_roots().count(), 0);
        assert!(empty.is_complete()); // no failed projection was published

        assert_eq!(lazy.allocation_info().unwrap(), None);
        let mut projection = ExistingArrayProjection::new(&context);
        let unpriced = vec![projection.project(&lazy).unwrap()];
        let unknown_storage = projection.into_storage();
        assert!(!unknown_storage.is_complete());
        assert_eq!(unknown_storage.iter().len(), 0);
        assert_eq!(lazy.allocation_info().unwrap(), None);
        context.begin_state_span(&unpriced).unwrap();
        assert_eq!(
            context
                .report(&unpriced)
                .unwrap()
                .inference_transient_bytes(),
            None
        );
        assert_eq!(
            context.report(&[]).unwrap().inference_transient_bytes(),
            None
        );
        // Completion outside inventory cannot upgrade its captured evidence.
        lazy.evaluated().unwrap();
        assert!(!unknown_storage.is_complete());
        let mut projection = ExistingArrayProjection::new(&context);
        projection.project(&lazy).unwrap();
        assert!(projection.into_storage().is_complete());
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn native_projection_witnesses_retire_without_retaining_backing_in_metadata() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        #[derive(Debug)]
        struct Retired(Arc<AtomicBool>);
        impl Drop for Retired {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let retired = Arc::new(AtomicBool::new(false));
        let array = Array::from_slice(&[1.25_f32, 2.5, 3.75], &[3]);
        array
            .retain_allocation_owner(Retired(retired.clone()))
            .unwrap();
        let info = array.allocation_info().unwrap().unwrap();
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::new(&context);
        let metadata = projection.project(&array).unwrap();
        let storage = projection.into_storage();
        let borrowed =
            WorkspaceBorrowedStorage::new(&context, storage.iter().map(|(_, _, root)| root))
                .unwrap();
        context.set_borrowed_storage(borrowed.clone()).unwrap();
        drop(array);
        safemlx::reclaim_allocation_owners();
        assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(
            storage
                .native_array(info.identity())
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[1.25, 2.5, 3.75]
        );
        drop(storage);
        safemlx::reclaim_allocation_owners();
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(borrowed.total_bytes(), info.bytes() as u64);
        context.begin_state_span([&metadata]).unwrap();
        assert_eq!(
            context
                .report(&[metadata])
                .unwrap()
                .state
                .unwrap()
                .retained_bytes,
            Some(info.bytes() as u64)
        );
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn descriptor_projection_busy_error_preserves_source_and_releases_no_foreign_guard() {
        use std::{sync::mpsc, time::Duration};
        let source = Array::from_slice(&[137_i32, -139, 149, 151], &[2, 2]);
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::new(&context);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(10))
                .unwrap()
                .enter()
                .unwrap();
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            drop(guard);
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let error = projection.project(&source).unwrap_err();
        // Release before assertions so a failed assertion cannot strand owner.
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<safemlx::ArrayDescriptorError>(),
            Some(&safemlx::ArrayDescriptorError::RuntimeBusy)
        );
        assert_eq!(projection.storage_roots().count(), 0);
        let projected = projection.project(&source).unwrap();
        assert_eq!(projected.layout().shape(), [2, 2]);
        assert!(projection.is_complete());
        assert_eq!(projection.storage_roots().count(), 1);
        assert_eq!(
            source.evaluated().unwrap().as_slice::<i32>(),
            &[137, -139, 149, 151]
        );
    }

    #[test]
    #[ignore = "requires native CPU execution"]
    fn descriptor_projection_unsupported_dtype_releases_actual_owner_before_error_escapes() {
        use std::time::Duration;
        let unsupported = Array::from_slice(&[157_i64, -163], &[2]);
        let supported = Array::from_slice(&[167_i32, 173], &[2]);
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::new(&context);
        let error = projection.project(&unsupported).unwrap_err();
        assert!(error.to_string().contains("Int64"));
        assert_eq!(projection.storage_roots().count(), 0);
        let entered = std::thread::spawn(|| {
            let guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(10))
                .unwrap()
                .enter()
                .unwrap();
            drop(guard);
        });
        entered.join().unwrap();
        let projected = projection.project(&supported).unwrap();
        assert_eq!(projected.layout().shape(), [2]);
        assert_eq!(projection.storage_roots().count(), 1);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires native Metal device access"]
    fn descriptor_projection_preserves_completed_metal_backing_and_nonzero_view() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let source = Array::from_slice(&[2_f32, -3., 5., -7., 11., -13.], &[2, 3]);
        let value = source.square(&stream).unwrap();
        value.evaluated().unwrap();
        let view = value.try_index_device((.., 1..), &stream).unwrap();
        view.evaluated().unwrap();
        let info = value.try_allocation_info().unwrap().unwrap();
        let context = WorkspaceContext::new(NoOperations);
        let mut projection = ExistingArrayProjection::new(&context);
        let root = projection.project(&value).unwrap();
        let projected = projection.project(&view).unwrap();
        assert_eq!(root.layout().shape(), [2, 3]);
        assert_eq!(projected.layout().shape(), [2, 2]);
        assert!(projection.is_complete());
        let roots = projection.storage_roots().collect::<Vec<_>>();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].0, info.identity());
        assert_eq!(roots[0].1, info.bytes() as u64);
        assert_eq!(roots[0].2.capacity_bytes(), Some(info.bytes() as u64));
        assert_eq!(
            value.evaluated().unwrap().as_slice::<f32>(),
            &[4., 9., 25., 49., 121., 169.]
        );
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
