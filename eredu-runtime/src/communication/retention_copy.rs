//! Fallible copies of already validated declarations for completion retention.
//! These helpers preserve declaration order and never rebuild validation sets.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::{size_of, size_of_val}};

fn frames<T>() -> Option<usize> {
    let values = [size_of::<T>(), size_of::<Result<T, TryReserveError>>(),
        size_of::<TryReserveError>(), size_of::<&T>()];
    values.into_iter().try_fold(size_of_val(&values), usize::checked_add)
}
fn vector_bytes<T>(length: usize) -> Option<usize> {
    let values = [Layout::array::<T>(length).ok()?.size(), size_of::<Vec<T>>(),
        size_of::<Result<Vec<T>, TryReserveError>>(), size_of::<Result<(), TryReserveError>>(),
        size_of::<(&[T], usize)>(), size_of::<std::slice::Iter<'_, T>>()];
    values.into_iter().try_fold(size_of_val(&values), usize::checked_add)
}
fn vector<T>(length: usize) -> Result<Vec<T>, TryReserveError> {
    let mut output = Vec::new();
    output.try_reserve_exact(length)?;
    Ok(output)
}
fn copy_slice<T: Copy>(values: &[T]) -> Result<Vec<T>, TryReserveError> {
    let mut output = vector(values.len())?;
    output.extend_from_slice(values);
    Ok(output)
}
fn string_bytes(value: &str) -> Option<usize> {
    let values = [value.len(), size_of::<String>(), size_of::<Result<String, TryReserveError>>(),
        size_of::<Result<(), TryReserveError>>(), size_of::<&str>()];
    values.into_iter().try_fold(size_of_val(&values), usize::checked_add)
}
fn copy_string(value: &str) -> Result<String, TryReserveError> {
    let mut output = String::new();
    output.try_reserve_exact(value.len())?;
    output.push_str(value);
    Ok(output)
}
fn dtype_bytes(value: &TensorDtype) -> Option<usize> {
    let bytes=frames::<TensorDtype>()?;
    match value { TensorDtype::Encoded(name)=>bytes.checked_add(string_bytes(name)?), _=>Some(bytes) }
}
fn copy_dtype(value: &TensorDtype) -> Result<TensorDtype, TryReserveError> {
    use TensorDtype::*;
    Ok(match value {
        Bool=>Bool,F32=>F32,F16=>F16,Bf16=>Bf16,I8=>I8,U8=>U8,U16=>U16,U32=>U32,
        U64=>U64,I16=>I16,I32=>I32,I64=>I64,F64=>F64,Complex64=>Complex64,
        Encoded(name)=>Encoded(copy_string(name)?),
    })
}
impl CommunicationOperationRequirement {
    fn retention_copy_bytes(&self) -> Option<usize> {
        let bytes=frames::<Self>()?.checked_add(vector_bytes::<TensorDtype>(self.dtypes.len())?)?;
        self.dtypes.iter().try_fold(bytes, |bytes,dtype|bytes.checked_add(dtype_bytes(dtype)?))
    }
    fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        let mut dtypes=vector(self.dtypes.len())?;
        for dtype in &self.dtypes { dtypes.push(copy_dtype(dtype)?); }
        Ok(Self { operation: self.operation, dtypes,
            limits: self.limits, exact_completion: self.exact_completion })
    }
}
impl CommunicationGroupRequirements {
    /// Requested storage and fixed transport controls of `try_copy_for_retention`.
    /// This is a source-specific host census, not allocation or admission authority.
    pub fn retention_copy_bytes(&self) -> Option<usize> {
        let total = frames::<Self>()?.checked_add(
            vector_bytes::<CommunicationOperationRequirement>(self.operations.len())?)?;
        self.operations.iter().try_fold(total, |total, value|
            total.checked_add(value.retention_copy_bytes()?))
    }
    /// Copies this validated ordered declaration using fallible destinations.
    /// Callers retain their own source/account owner through the returned copy.
    pub fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        let mut operations = vector(self.operations.len())?;
        for value in &self.operations { operations.push(value.try_copy_for_retention()?); }
        Ok(Self { operations })
    }
}
impl BoundaryRoleContract {
    fn retention_copy_bytes(&self) -> Option<usize> {
        frames::<Self>()?.checked_add(string_bytes(&self.role)?)?
            .checked_add(dtype_bytes(&self.dtype)?)?
            .checked_add(vector_bytes::<BoundaryDimensionContract>(self.shape.len())?)
    }
    fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        Ok(Self { role: copy_string(&self.role)?, dtype: copy_dtype(&self.dtype)?,
            shape: copy_slice(&self.shape)? })
    }
}
impl RoleExactBoundaryContract {
    fn retention_copy_bytes(&self) -> Option<usize> {
        let total = frames::<Self>()?.checked_add(string_bytes(&self.schema)?)?
            .checked_add(vector_bytes::<BoundaryRoleContract>(self.roles.len())?)?;
        self.roles.iter().try_fold(total, |total, role|
            total.checked_add(role.retention_copy_bytes()?))
    }
    fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        let schema = copy_string(&self.schema)?;
        let mut roles = vector(self.roles.len())?;
        for role in &self.roles { roles.push(role.try_copy_for_retention()?); }
        Ok(Self { protocol: self.protocol, schema, roles })
    }
}
impl CommunicationRouteDescriptor {
    /// Requested storage and fixed transport controls of `try_copy_for_retention`,
    /// including each actual UTF-8 schema/role and ordered shape destination.
    pub fn retention_copy_bytes(&self) -> Option<usize> {
        let total = frames::<Self>()?.checked_add(self.requirement.retention_copy_bytes()?)?
            .checked_add(size_of::<Option<RoleExactBoundaryContract>>())?
            .checked_add(size_of::<Result<Option<RoleExactBoundaryContract>, TryReserveError>>())?;
        match &self.boundary { Some(value) => total.checked_add(value.retention_copy_bytes()?), None => Some(total) }
    }
    /// Fallibly copies the exact validated route, including its role contract.
    /// No new declaration, ordering decision, or validation set is constructed.
    pub fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        Ok(Self { id: self.id, submission_order: self.submission_order,
            source: self.source, destination: self.destination,
            requirement: self.requirement.try_copy_for_retention()?,
            boundary: self.boundary.as_ref().map(RoleExactBoundaryContract::try_copy_for_retention).transpose()? })
    }
}

impl CommunicationGroupDescriptor {
    /// Exact destinations for copying this retained ordered group declaration.
    pub fn retention_copy_bytes(&self) -> Option<usize> {
        frames::<Self>()?.checked_add(vector_bytes::<usize>(self.members.len())?)?
            .checked_add(self.requirements.retention_copy_bytes()?)
    }
    /// Copies the established declaration without rebuilding membership or
    /// operation requirements. The caller retains and funds its source owner.
    pub fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        Ok(Self { id: self.id, creation_order: self.creation_order,
            members: copy_slice(&self.members)?, local_index: self.local_index,
            requirements: self.requirements.try_copy_for_retention()? })
    }
}
impl CommunicationManifest {
    /// Exact host destinations for a complete retained-manifest copy. Native
    /// resources and setup identity remain with the original source owner.
    pub fn retention_copy_bytes(&self) -> Option<usize> {
        let mut total = frames::<Self>()?
            .checked_add(vector_bytes::<CommunicationGroupDescriptor>(self.groups.len())?)?
            .checked_add(vector_bytes::<CommunicationRouteDescriptor>(self.routes.len())?)?;
        for group in &self.groups { total = total.checked_add(group.retention_copy_bytes()?)?; }
        for route in &self.routes { total = total.checked_add(route.retention_copy_bytes()?)?; }
        Some(total)
    }
    /// Fallibly copies the exact immutable declarations and completion policy.
    /// No validation set, setup transcript or new native resource is created.
    pub fn try_copy_for_retention(&self) -> Result<Self, TryReserveError> {
        let mut groups = vector(self.groups.len())?;
        for group in &self.groups { groups.push(group.try_copy_for_retention()?); }
        let mut routes = vector(self.routes.len())?;
        for route in &self.routes { routes.push(route.try_copy_for_retention()?); }
        Ok(Self { world_size: self.world_size, rank: self.rank,
            groups, routes, completion: self.completion })
    }
}

#[cfg(test)]
mod tests;
