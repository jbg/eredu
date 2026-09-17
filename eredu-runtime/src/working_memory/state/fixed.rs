//! Finite declared role storage for the ordinary workspace state worker.
use super::*;
use std::mem::size_of;

/// Immutable role declarations and shared value snapshots. Cloning never
/// allocates; mutation explicitly copies the value slots under the trace budget.
#[derive(Debug, Clone, Default)]
pub(super) struct FixedSlots {
    roles: Option<std::rc::Rc<Vec<StateTensorRole>>>,
    values: Option<std::rc::Rc<Vec<Option<WorkspaceTensor>>>>,
}

impl FixedSlots {
    pub(super) fn new(
        policy: &LayerCachePolicy,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if policy.fixed_state().is_empty() {
            return Ok(Self::default());
        }
        let mut roles = context.metadata_vec(policy.fixed_state().len())?;
        for declaration in policy.fixed_state() {
            if let Err(index) = roles.binary_search(&declaration.role) {
                roles.insert(index, declaration.role);
            }
        }
        let mut values = context.metadata_vec(roles.len())?;
        values.resize_with(roles.len(), || None);
        Ok(Self {
            roles: Some(context.metadata_rc(roles)?),
            values: Some(context.metadata_rc(values)?),
        })
    }
    pub(super) fn len(&self) -> usize {
        self.roles.as_ref().map_or(0, |roles| roles.len())
    }
    pub(super) fn clear(&mut self) {
        self.values = None;
        self.roles = None;
    }
    pub(super) fn index(&self, role: &StateTensorRole) -> Option<usize> {
        self.roles.as_ref()?.binary_search(role).ok()
    }
    pub(super) fn get(&self, role: &StateTensorRole) -> Option<&Option<WorkspaceTensor>> {
        self.index(role)
            .map(|i| &self.values.as_ref().expect("declared values")[i])
    }
    pub(super) fn get_mut(
        &mut self,
        role: &StateTensorRole,
        context: &WorkspaceContext,
    ) -> Result<Option<&mut Option<WorkspaceTensor>>, Error> {
        let Some(index) = self.index(role) else {
            return Ok(None);
        };
        Ok(Some(&mut self.values_mut(context)?[index]))
    }
    pub(super) fn values(&self) -> std::slice::Iter<'_, Option<WorkspaceTensor>> {
        self.values
            .as_deref()
            .map_or(&[][..], |values| values.as_slice())
            .iter()
    }
    pub(super) fn values_mut(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<&mut [Option<WorkspaceTensor>], Error> {
        let Some(values) = &mut self.values else {
            return Ok(&mut []);
        };
        if std::rc::Rc::get_mut(values).is_none() {
            let mut copied = context.metadata_vec(values.len())?;
            copied.extend(values.iter().cloned());
            // Publish only after both the vector and shared shell succeed.
            *values = context.metadata_rc(copied)?;
        }
        Ok(std::rc::Rc::get_mut(values)
            .expect("exclusive value slots")
            .as_mut_slice())
    }
}

impl FixedSlots {
    fn unique_count(policy: &LayerCachePolicy) -> usize {
        let declarations = policy.fixed_state();
        declarations
            .iter()
            .enumerate()
            .filter(|(index, declaration)| {
                !declarations[..*index]
                    .iter()
                    .any(|previous| previous.role == declaration.role)
            })
            .count()
    }

    fn construction_bytes(policy: &LayerCachePolicy) -> Option<usize> {
        let count = policy.fixed_state().len();
        let (vectors, owners) = if count == 0 {
            (0usize, 0usize)
        } else {
            (
                WorkspaceContext::metadata_vec_bytes::<StateTensorRole>(count)?.checked_add(
                    WorkspaceContext::metadata_vec_bytes::<Option<WorkspaceTensor>>(
                        Self::unique_count(policy),
                    )?,
                )?,
                WorkspaceContext::metadata_rc_bytes::<Vec<StateTensorRole>>()?.checked_add(
                    WorkspaceContext::metadata_rc_bytes::<Vec<Option<WorkspaceTensor>>>()?,
                )?,
            )
        };
        let parts = [
            vectors,
            owners,
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<usize, usize>>(),
            size_of::<(usize, usize, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

impl WorkspaceConcatStateFactory {
    /// Actual role/value constructor controls. Empty declarations construct no
    /// vectors or shared shells; values use the same unique role population as
    /// the sorted insertion worker. This query creates no allocation authority.
    pub fn fixed_role_constructor_storage_bytes(policy: &LayerCachePolicy) -> Option<usize> {
        if !crate::working_memory::qualified_storage::qualified() {
            return None;
        }
        FixedSlots::construction_bytes(policy)
    }

    /// Constructor controls plus exact import duplicate marks. Tensor/context,
    /// state-table and operation/report owners are separate producers.
    pub fn fixed_role_projection_storage_bytes(policy: &LayerCachePolicy) -> Option<usize> {
        let parts = [
            Self::fixed_role_constructor_storage_bytes(policy)?,
            Self::projection_error_control_bytes(policy)?,
            WorkspaceContext::metadata_vec_bytes::<bool>(FixedSlots::unique_count(policy))?,
            size_of::<Vec<bool>>(),
            size_of::<Option<usize>>(),
            size_of::<std::slice::Iter<'_, StateTensorRole>>(),
            size_of::<std::slice::Iter<'_, Option<WorkspaceTensor>>>(),
            size_of::<std::slice::IterMut<'_, Option<WorkspaceTensor>>>(),
            size_of::<(usize, usize, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
