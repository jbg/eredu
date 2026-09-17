//! Fixed projection controls over the same retained native descriptors.
use super::*;
use eredu_nn::workspace::{WorkspaceImportError, WorkspaceLayoutError, WorkspaceLayoutView};

/// Fixed source/import refusal. Construction retains the enclosing host custody.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionSourceError {
    /// The no-hooks native descriptor loan was refused.
    #[error(transparent)]
    Descriptor(#[from] safemlx::ArrayDescriptorError),
    /// The native element representation has no workspace equivalent.
    #[error("existing state dtype {0:?} has no workspace representation")]
    Dtype(Dtype),
    /// The selected source has no certified completed backing.
    #[error("native snapshot projection has unknown backing")]
    UnknownBacking,
    /// Native physical capacity does not fit the portable representation.
    #[error("native projection capacity overflows")]
    Overflow,
    /// One physical source changed capacity between two aliases.
    #[error("retained native allocation capacity changed during projection")]
    CapacityChanged,
    /// The final inventory rows could not be prepared.
    #[error(transparent)]
    Inventory(#[from] ProjectionInventoryError),
    /// Logical shape validation failed before copying it.
    #[error(transparent)]
    Layout(#[from] WorkspaceLayoutError),
    /// Final logical shape or imported metadata refused construction.
    #[error(transparent)]
    Import(#[from] WorkspaceImportError),
    /// The context refused a participating metadata constructor.
    #[error(transparent)]
    Metadata(#[from] Error),
    /// An exactly sized descriptor handle could not be allocated or filled.
    #[error(transparent)]
    Clone(#[from] safemlx::PreparedArrayCloneCause),
    /// Fixed host source observation refused without housekeeping or allocation.
    #[error(transparent)]
    Host(#[from] safemlx::HostTransferMetadataError),
}
impl ProjectionSourceError {
    pub(super) fn ordinary(self) -> Error {
        match self {
            Self::Metadata(cause)
            | Self::Inventory(ProjectionInventoryError::Metadata(cause))
            | Self::Import(WorkspaceImportError::Metadata(cause)) => cause,
            Self::Descriptor(cause) => Error::backend_source(cause),
            Self::Inventory(cause) => Error::backend_retained_source(cause),
            cause => Error::backend(cause),
        }
    }
    pub(super) fn in_context(self, context: &WorkspaceContext) -> Error {
        if !context.uses_checked_metadata() {
            return self.ordinary();
        }
        match self {
            Self::Metadata(cause)
            | Self::Inventory(ProjectionInventoryError::Metadata(cause))
            | Self::Import(WorkspaceImportError::Metadata(cause)) => cause,
            Self::Descriptor(cause) => context.metadata_source(cause),
            Self::Inventory(cause) => context.metadata_source(cause),
            cause => context.metadata_source(cause),
        }
    }
}
pub(super) fn projected_dtype(dtype: Dtype) -> Result<WorkspaceDtype, ProjectionSourceError> {
    Ok(match dtype {
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16 => WorkspaceDtype::Float32,
        Dtype::Int32 => WorkspaceDtype::Int32,
        Dtype::Uint32 => WorkspaceDtype::Uint32,
        Dtype::Uint8 => WorkspaceDtype::Uint8,
        Dtype::Bool => WorkspaceDtype::Bool,
        dtype => return Err(ProjectionSourceError::Dtype(dtype)),
    })
}

/// Actual descriptor scalar identity; no layout-contiguity credit is inferred.
/// Integer logical layouts already retain their exact scalar representation.
pub(super) fn projected_representation(dtype: Dtype) -> Option<WorkspaceRepresentation> {
    let floating = match dtype {
        Dtype::Float32 => WorkspaceFloatingType::Float32,
        Dtype::Float16 => WorkspaceFloatingType::Float16,
        Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
        _ => return None,
    };
    Some(WorkspaceRepresentation::new(floating, false))
}

/// One operand's owning import/root/native-handle requests. Counting every
/// operand is conservative when physical aliases share one root and handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionSourceLayout {
    bytes: usize,
    rank: usize,
}
impl ProjectionSourceLayout {
    /// Borrows the actual descriptor without allocations, evaluation or polling.
    /// Uses the same dtype and logical-shape validation as prepared projection.
    pub fn inspect(array: &Array) -> Result<Self, ProjectionSourceError> {
        Self::with_layout(array, |layout| Self::for_rank(layout.shape().len()))?
            .ok_or(ProjectionSourceError::Overflow)
    }
    /// Loans the exact projected dtype and shape while the native descriptor is
    /// held. The callback owns its controls; no shape or descriptor can escape.
    pub(crate) fn with_layout<T>(
        array: &Array,
        visit: impl FnOnce(WorkspaceLayoutView<'_>) -> T,
    ) -> Result<T, ProjectionSourceError> {
        let descriptor = array.try_descriptor()?;
        let dtype = projected_dtype(descriptor.facts().dtype())?;
        let layout = WorkspaceLayoutView::new(descriptor.shape(), dtype)?
            .with_representation(representation::from_descriptor(&descriptor));
        let info = descriptor
            .facts()
            .allocation()
            .ok_or(ProjectionSourceError::UnknownBacking)?;
        u64::try_from(info.bytes()).map_err(|_| ProjectionSourceError::Overflow)?;
        Ok(visit(layout))
    }
    pub(crate) fn host_layout_control_bytes() -> Option<usize> {
        let frames = [
            safemlx::HostTransferDescriptor::<4>::control_bytes()?,
            size_of::<safemlx::HostTransferDescriptor<4>>(),
            size_of::<Result<safemlx::HostTransferDescriptor<4>, safemlx::HostTransferMetadataError>>(
            ),
            size_of::<WorkspaceLayoutView<'_>>(),
            size_of::<Result<WorkspaceLayoutView<'_>, eredu_nn::workspace::WorkspaceLayoutError>>(),
            size_of::<&safemlx::ImmutableHostTransferBuffer>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn with_host_layout<T>(
        host: &safemlx::ImmutableHostTransferBuffer,
        visit: impl FnOnce(WorkspaceLayoutView<'_>) -> T,
    ) -> Result<T, ProjectionSourceError> {
        let descriptor = host.try_fixed_descriptor::<4>()?;
        let dtype = projected_dtype(descriptor.dtype())?;
        let layout = WorkspaceLayoutView::new(descriptor.shape(), dtype)?
            .with_representation(projected_representation(descriptor.dtype()));
        u64::try_from(descriptor.allocation().bytes())
            .map_err(|_| ProjectionSourceError::Overflow)?;
        Ok(visit(layout))
    }
    fn for_rank(rank: usize) -> Option<Self> {
        let parts = [
            WorkspaceExistingStorage::construction_bytes()?,
            WorkspaceTensor::imported_construction_bytes(rank)?,
            ExistingArrayProjection::import_control_bytes()?,
            ExistingArrayProjection::clone_control_bytes()?,
            size_of::<Self>(),
            size_of::<Result<Self, ProjectionSourceError>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        Some(Self { bytes, rank })
    }
    /// Requested source import controls/backing, excluding borrowed native data.
    pub fn requested_bytes(self) -> usize {
        self.bytes
    }
    /// Actual logical shape rank observed under the native descriptor loan.
    pub fn rank(self) -> usize {
        self.rank
    }
}

impl ProjectedNativeStorage {
    /// Actual borrowed row iterator and output controls. The empty owner only
    /// identifies the iterator type; it allocates no row or native handle.
    pub(crate) fn iteration_control_bytes() -> Option<usize> {
        let empty = Self {
            rows: Vec::new(),
            known: 0,
            paged_sources: Vec::new(),
            copy_sources: Vec::new(),
            _host_preparation: None,
            _funding: None,
        };
        let iterator = std::mem::size_of_val(&empty.iter());
        iterator.checked_add(size_of::<
            Option<(safemlx::AllocationIdentity, u64, &WorkspaceExistingStorage)>,
        >())
    }
}

impl<'a> ExistingArrayProjection<'a> {
    /// Actual descriptor and fixed import transports, excluding logical shape
    /// and portable backing-root constructors owned by their shared queries.
    pub(super) fn import_frame_bytes() -> Option<usize> {
        let parts = [
            Array::descriptor_control_bytes()?,
            representation::control_bytes()?,
            size_of::<ProjectionSourceError>(),
            size_of::<Result<WorkspaceTensor, ProjectionSourceError>>(),
            size_of::<Result<WorkspaceDtype, ProjectionSourceError>>(),
            size_of::<(
                Dtype,
                WorkspaceFloatingType,
                Option<WorkspaceRepresentation>,
            )>(),
            size_of::<Result<WorkspaceExistingStorage, ProjectionSourceError>>(),
            size_of::<Result<WorkspaceLayoutView<'_>, WorkspaceLayoutError>>(),
            size_of::<Option<safemlx::ArrayAllocationInfo>>(),
            size_of::<(usize, u64, &Array, &WorkspaceContext)>(),
            size_of::<Result<(), ProjectionSourceError>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    pub(super) fn import_control_bytes() -> Option<usize> {
        Self::import_frame_bytes()?
            .checked_add(storage_control_bytes::<&Array>(size_of::<&Array>())?)
    }

    /// One real prepared C handle and its fallible fill/row transports.
    pub(super) fn clone_control_bytes() -> Option<usize> {
        let parts = [
            Array::inspection_clone_handle_bytes(),
            safemlx::PreparedArrayClone::control_bytes()?,
            size_of::<Result<ProjectedNativeStorage, ProjectionSourceError>>(),
            size_of::<Result<ProjectionRow<ProjectionNativeSource>, ProjectionSourceError>>(),
            size_of::<Result<Array, ProjectionSourceError>>(),
            size_of::<(&Array, &WorkspaceContext)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    pub(super) fn charge_import(&mut self) -> Result<(), ProjectionSourceError> {
        self.start_metadata()?;
        self.context
            .charge_metadata(Self::import_frame_bytes().ok_or(ProjectionSourceError::Overflow)?)
            .map_err(Error::from)?;
        Ok(())
    }

    pub(super) fn storage_for(
        &mut self,
        array: &'a Array,
        allocation: Option<safemlx::ArrayAllocationInfo>,
    ) -> Result<WorkspaceExistingStorage, ProjectionSourceError> {
        storage_for(
            &mut self.rows,
            &mut self.known,
            self.maximum,
            allocation,
            self.context,
            || Ok(array),
            |_| Ok(()),
        )
    }

    /// Same native identity mapping and import semantics with fixed failure
    /// causes. Missing completed backing refuses before metadata construction.
    pub fn project_prepared(
        &mut self,
        array: &'a Array,
    ) -> Result<WorkspaceTensor, ProjectionSourceError> {
        self.charge_import()?;
        let descriptor = array.try_descriptor()?;
        let dtype = projected_dtype(descriptor.facts().dtype())?;
        let layout = WorkspaceLayoutView::new(descriptor.shape(), dtype)?
            .with_representation(representation::from_descriptor(&descriptor));
        let allocation = descriptor
            .facts()
            .allocation()
            .ok_or(ProjectionSourceError::UnknownBacking)?;
        let storage = self.storage_for(array, Some(allocation))?;
        Ok(WorkspaceTensor::import_existing(
            layout,
            &storage,
            self.context,
        )?)
    }
    /// Final native descriptor handles use the exact fixed clone constructor.
    /// Each partial handle/row drops before the caller's enclosing host custody.
    pub fn into_prepared_storage(self) -> Result<ProjectedNativeStorage, ProjectionSourceError> {
        let mut destination = None;
        self.retain_prepared_storage(&mut destination)?;
        Ok(destination.expect("successful handoff installs its destination"))
    }

    /// Same worker with an enclosing retirement destination. A borrowed manager
    /// callback supplies an owner outside its guard so all accepted native
    /// handles survive a later refusal or unwind until after that guard exits.
    pub(crate) fn retain_prepared_storage(
        mut self,
        destination: &mut Option<ProjectedNativeStorage>,
    ) -> Result<(), ProjectionSourceError> {
        if destination.is_some() {
            return Err(ProjectionInventoryError::Destination.into());
        }
        self.start_metadata()?;
        let rows = self
            .context
            .metadata_vec(self.maximum.unwrap_or(self.rows.len()))?;
        *destination = Some(ProjectedNativeStorage {
            rows,
            known: 0,
            paged_sources: Vec::new(),
            copy_sources: Vec::new(),
            _host_preparation: None,
            _funding: self.context.metadata_funding(),
        });
        let retained = destination.as_mut().expect("installed destination");
        for (index, row) in self.rows.into_iter().enumerate() {
            let row = row.map(&mut |array: &Array| {
                clone_array(array, self.context).map(ProjectionNativeSource::array)
            })?;
            retained.rows.push(row);
            retained.known += usize::from(index < self.known);
        }
        Ok(())
    }
}

/// Shared identity/capacity/root insertion for borrowed and owning destinations.
pub(super) fn storage_for<A, F, V>(
    rows: &mut Vec<ProjectionRow<A>>,
    known: &mut usize,
    maximum: Option<usize>,
    allocation: Option<safemlx::ArrayAllocationInfo>,
    context: &WorkspaceContext,
    retain: F,
    complete_witness: V,
) -> Result<WorkspaceExistingStorage, ProjectionSourceError>
where
    F: FnOnce() -> Result<A, ProjectionSourceError>,
    V: FnOnce(&mut A) -> Result<(), ProjectionSourceError>,
{
    context
        .charge_metadata(
            storage_control_bytes::<A>(size_of::<(F, V)>())
                .ok_or(ProjectionSourceError::Overflow)?,
        )
        .map_err(Error::from)?;
    let check = |length: usize, maximum: Option<usize>| -> Result<(), ProjectionSourceError> {
        if maximum.is_some_and(|maximum| length == maximum) {
            Err(ProjectionInventoryError::Exhausted.into())
        } else {
            Ok(())
        }
    };
    match allocation {
        Some(info) => {
            let bytes = u64::try_from(info.bytes()).map_err(|_| ProjectionSourceError::Overflow)?;
            match rows[..*known].binary_search_by_key(&info.identity(), ProjectionRow::identity) {
                Ok(index) => match &mut rows[index] {
                    ProjectionRow::Known {
                        bytes: previous,
                        storage,
                        array,
                        ..
                    } => {
                        if *previous != bytes {
                            return Err(ProjectionSourceError::CapacityChanged);
                        }
                        complete_witness(array)?;
                        Ok(storage.clone())
                    }
                    ProjectionRow::Unknown(_) => unreachable!("known projection prefix"),
                },
                Err(index) => {
                    check(rows.len(), maximum)?;
                    context.reserve_metadata_vec(rows, 1)?;
                    let storage = WorkspaceExistingStorage::try_new(Some(bytes), context)?;
                    rows.insert(
                        index,
                        ProjectionRow::Known {
                            identity: info.identity(),
                            bytes,
                            storage: storage.clone(),
                            array: retain()?,
                        },
                    );
                    *known += 1;
                    Ok(storage)
                }
            }
        }
        None => {
            check(rows.len(), maximum)?;
            context.reserve_metadata_vec(rows, 1)?;
            let storage = WorkspaceExistingStorage::try_new(None, context)?;
            rows.push(ProjectionRow::Unknown(retain()?));
            Ok(storage)
        }
    }
}
pub(super) fn storage_control_bytes<A>(retention: usize) -> Option<usize> {
    size_of::<(
        &mut Vec<ProjectionRow<A>>,
        &mut usize,
        Option<usize>,
        Option<safemlx::ArrayAllocationInfo>,
        &WorkspaceContext,
        Result<usize, usize>,
        usize,
        u64,
        A,
        Result<A, ProjectionSourceError>,
        Result<(), ProjectionSourceError>,
        Result<WorkspaceExistingStorage, ProjectionSourceError>,
    )>()
    .checked_add(retention)
}
pub(super) fn clone_array(
    array: &Array,
    context: &WorkspaceContext,
) -> Result<Array, ProjectionSourceError> {
    context
        .charge_metadata(
            ExistingArrayProjection::clone_control_bytes()
                .ok_or(ProjectionSourceError::Overflow)?,
        )
        .map_err(Error::from)?;
    let mut clone = safemlx::PreparedArrayClone::try_prepare_for_inspection()?;
    clone
        .fill_for_inspection(array)
        .map_err(ProjectionSourceError::Clone)
}
