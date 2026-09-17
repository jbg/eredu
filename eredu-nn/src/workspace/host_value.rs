//! Opaque stored-Host effects, distinct from Device tensors and native grants.
use super::*;
use std::{mem::size_of, rc::Rc};

/// Identity of one store effect or declared read source in its metadata trace.
/// This is neither a physical allocation key nor native source permission.
#[derive(Clone, Debug)]
pub struct WorkspaceStoredHostIdentity(Rc<()>);
impl WorkspaceStoredHostIdentity {
    /// Compare the exact store occurrence, including aliases of its handle.
    pub fn same_store(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Fixed geometry of one separately funded Host value. There is deliberately
/// no Tensor/storage alias here: native Host publication owns its allocation,
/// and the internal copy-completion Array stays in the Model completion union.
#[derive(Clone, Debug)]
pub struct WorkspaceStoredHostValue {
    identity: WorkspaceStoredHostIdentity,
    source: WorkspaceTensor,
    shape: [i32; 4],
    rank: usize,
    dtype: WorkspaceFloatingType,
    context: WorkspaceContext,
}
impl WorkspaceContext {
    /// Trace a Host-store effect and return only its exact opaque value handle.
    /// The selected source constructor separately accounts the Host allocation
    /// and native publication. This effect creates no Device output storage.
    pub fn store_host_value(
        &self,
        input: &WorkspaceTensor,
        dtype: WorkspaceFloatingType,
    ) -> Result<WorkspaceStoredHostValue, Error> {
        self.validate_values([input])?;
        let dimensions = input.layout().shape();
        if dimensions.is_empty()
            || dimensions.len() > 4
            || dimensions.iter().any(|n| *n <= 0)
            || input.layout.dtype != WorkspaceDtype::Float32
            || input
                .layout
                .representation()
                .is_some_and(|value| value.dtype() != dtype)
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.charge_metadata(size_of::<(
            &Self,
            &WorkspaceTensor,
            WorkspaceFloatingType,
            WorkspaceStoredHostValue,
            WorkspaceStoredHostIdentity,
            [i32; 4],
            Vec<WorkspaceLayout>,
            Vec<WorkspaceTensor>,
            Result<WorkspaceStoredHostValue, Error>,
        )>())?;
        let identity = WorkspaceStoredHostIdentity(self.metadata_rc(())?);
        let mut shape = [0; 4];
        shape[..dimensions.len()].copy_from_slice(dimensions);
        let outputs = self.execute(
            WorkspaceOperationKind::HostStoreFloating(identity.clone(), dtype),
            &[input],
            Vec::new(),
        )?;
        debug_assert!(outputs.is_empty());
        Ok(WorkspaceStoredHostValue {
            identity,
            source: input.clone(),
            shape,
            rank: dimensions.len(),
            dtype,
            context: self.clone(),
        })
    }
}
impl WorkspaceStoredHostValue {
    /// Declare the same conditional store in a later invocation. A selected
    /// policy may defer the one-use native store until then; its copy/compaction
    /// and completion work belongs to that invocation's budget. The retained
    /// source and store identity are unchanged, and no new Host value is made.
    pub fn declare_deferred_store(&self, context: &WorkspaceContext) -> Result<(), Error> {
        if !self.context.shares_trace(context) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        context.validate_values([&self.source])?;
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspaceContext,
            WorkspaceOperationKind,
            Vec<WorkspaceTensor>,
            Result<(), Error>,
        )>())?;
        let outputs = context.execute(
            WorkspaceOperationKind::HostStoreFloating(self.identity.clone(), self.dtype),
            &[&self.source],
            Vec::new(),
        )?;
        debug_assert!(outputs.is_empty());
        Ok(())
    }

    /// Shape of the exact stored value, independent of any Device array handle.
    pub fn shape(&self) -> &[i32] {
        &self.shape[..self.rank]
    }
    /// Actual native floating representation retained by the store producer.
    pub fn floating_type(&self) -> WorkspaceFloatingType {
        self.dtype
    }
    /// Identity matched by the selected source program, never an allocation key.
    pub fn identity(&self) -> &WorkspaceStoredHostIdentity {
        &self.identity
    }
    /// Trace independent Device materialization from this same stored value.
    /// The actual native consumer must authenticate its completed Host source;
    /// constructing or cloning this metadata handle grants no such permission.
    pub fn load(&self, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
        load_host_value(
            &self.identity,
            self.shape(),
            self.dtype,
            &self.context,
            context,
        )
    }
}

/// Opaque geometry of a separately prepared Host read. It owns no Tensor or
/// allocation identifier and declares no Host-store effect. A native source
/// binder must independently retain and authenticate its actual file/backing.
#[derive(Clone, Debug)]
pub struct WorkspaceHostReadValue {
    identity: WorkspaceStoredHostIdentity,
    shape: [i32; 4],
    rank: usize,
    dtype: WorkspaceFloatingType,
    context: WorkspaceContext,
}
impl WorkspaceContext {
    /// Declares a finite read-source handle from provider-described geometry.
    /// This metadata grants no read, allocation, source or execution authority.
    pub fn declare_host_read_value(
        &self,
        shape: &[i32],
        dtype: WorkspaceFloatingType,
    ) -> Result<WorkspaceHostReadValue, Error> {
        self.charge_metadata(size_of::<(
            &Self,
            &[i32],
            WorkspaceFloatingType,
            [i32; 4],
            WorkspaceHostReadValue,
            WorkspaceStoredHostIdentity,
            Result<WorkspaceHostReadValue, Error>,
        )>())?;
        if shape.is_empty() || shape.len() > 4 || shape.iter().any(|n| *n <= 0) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut dimensions = [0; 4];
        dimensions[..shape.len()].copy_from_slice(shape);
        Ok(WorkspaceHostReadValue {
            identity: WorkspaceStoredHostIdentity(self.metadata_rc(())?),
            shape: dimensions,
            rank: shape.len(),
            dtype,
            context: self.clone(),
        })
    }
}
impl WorkspaceHostReadValue {
    /// Exact logical extent without an invented Tensor storage identity.
    pub fn shape(&self) -> &[i32] {
        &self.shape[..self.rank]
    }
    /// The actual source scalar representation, separate from extent class.
    pub fn floating_type(&self) -> WorkspaceFloatingType {
        self.dtype
    }
    /// Whether this metadata belongs to the exact current quotation trace.
    pub fn shares_context(&self, context: &WorkspaceContext) -> bool {
        self.context.shares_trace(context)
    }
    /// Uses the same separately qualified Host-load effect as stored values.
    pub fn load(&self, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
        load_host_value(
            &self.identity,
            self.shape(),
            self.dtype,
            &self.context,
            context,
        )
    }
}
fn load_host_value(
    identity: &WorkspaceStoredHostIdentity,
    shape: &[i32],
    dtype: WorkspaceFloatingType,
    source: &WorkspaceContext,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    if !source.shares_trace(context) {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    context.charge_metadata(size_of::<(
        &WorkspaceStoredHostIdentity,
        &[i32],
        WorkspaceFloatingType,
        &WorkspaceContext,
        &WorkspaceContext,
        WorkspaceOperationKind,
        Result<WorkspaceTensor, Error>,
    )>())?;
    WorkspaceTensor::operation(
        WorkspaceOperationKind::HostLoadStoredFloating(identity.clone(), dtype),
        &[],
        shape,
        WorkspaceDtype::Float32,
        context,
    )
}
