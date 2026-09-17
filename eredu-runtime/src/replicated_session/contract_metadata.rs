//! Destinations for the shared prepared-contract validation worker.
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

/// A legacy contract rejection or a typed failure of a counted constructor.
#[derive(Debug, thiserror::Error)]
pub enum PreparedTextContractError {
    /// An existing source/geometry producer rejected the contract.
    #[error("{0}")]
    Contract(String),
    /// A participating metadata destination refused or failed.
    #[error(transparent)]
    Metadata(#[from] Error),
}
impl From<String> for PreparedTextContractError {
    fn from(message: String) -> Self {
        Self::Contract(message)
    }
}
impl PreparedTextContractError {
    pub(super) fn into_legacy(self) -> String {
        match self {
            Self::Contract(message) => message,
            Self::Metadata(error) => error.to_string(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ContractMetadata<'a>(Option<&'a WorkspaceContext>);
impl<'a> ContractMetadata<'a> {
    pub(super) fn is_checked(self) -> bool {
        self.0.is_some()
    }

    pub(super) fn context(self) -> Option<&'a WorkspaceContext> {
        self.0
    }

    pub(super) fn new(context: Option<&'a WorkspaceContext>) -> Self {
        Self(context.filter(|value| value.uses_checked_metadata()))
    }
    pub(super) fn message(self, args: std::fmt::Arguments<'_>) -> PreparedTextContractError {
        match self.0 {
            Some(context) => context.metadata_error(args).into(),
            None => PreparedTextContractError::Contract(args.to_string()),
        }
    }
    pub(super) fn architecture_error<E: std::error::Error + 'static>(
        self,
        error: E,
        prefix: &str,
    ) -> PreparedTextContractError {
        if self.is_checked() {
            let mut cause: &dyn std::error::Error = &error;
            loop {
                if let Some(cause) = cause.downcast_ref::<WorkspaceMetadataError>() {
                    return PreparedTextContractError::Metadata((*cause).into());
                }
                match cause.source() {
                    Some(source) => cause = source,
                    None => break,
                }
            }
        }
        self.message(format_args!("{prefix}{error}"))
    }
    pub(super) fn vector<T>(self, capacity: usize) -> Result<Vec<T>, PreparedTextContractError> {
        match self.0 {
            Some(context) => context.metadata_vec(capacity).map_err(Into::into),
            None => Ok(Vec::with_capacity(capacity)),
        }
    }
    pub(super) fn controls<T>(self) -> Result<(), PreparedTextContractError> {
        if let Some(context) = self.0 {
            let parts = [
                std::mem::size_of::<T>(),
                std::mem::size_of::<Self>(),
                std::mem::size_of::<Result<T, PreparedTextContractError>>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)
                .map_err(Error::from)?;
            context.charge_metadata(bytes).map_err(Error::from)?;
        }
        Ok(())
    }
}

pub(super) struct DebugRows<I>(pub(super) I);
impl<I: Iterator + Clone> std::fmt::Debug for DebugRows<I>
where
    I::Item: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.0.clone()).finish()
    }
}

/// Owns the original description once; companion rows retain only indices into
/// that description, never copied names, shapes, owners, recipes or provenance.
pub(super) struct ValidatedParameterCatalog<'source> {
    pub(super) description: std::borrow::Cow<'source, crate::ArchitectureParameterDescription>,
    pub(super) companions: Vec<(usize, usize)>,
}
#[derive(Clone, Copy)]
pub(super) struct CompanionView<'a> {
    member: &'a crate::ParameterMemberSpec,
    owner: &'a crate::ParameterGroupOwner,
}
impl<'a> CompanionView<'a> {
    pub(super) fn name(self) -> &'a str {
        self.member.target()
    }
    pub(super) fn primary(self) -> &'a str {
        self.member
            .linear_companion_of()
            .expect("validated companion primary")
    }
    pub(super) fn role(self) -> eredu_nn::LinearCompanionRole {
        self.member
            .linear_companion()
            .expect("validated companion role")
    }
    pub(super) fn logical_shape(self) -> &'a [usize] {
        self.member.global_shape()
    }
    pub(super) fn owner(self) -> &'a crate::ParameterGroupOwner {
        self.owner
    }
}
impl ValidatedParameterCatalog<'_> {
    pub(super) fn row(&self, (group, member): (usize, usize)) -> CompanionView<'_> {
        let group = &self.description.groups()[group];
        CompanionView {
            member: &group.group().members()[member],
            owner: group.owner(),
        }
    }
    pub(super) fn for_primary<'a>(
        &'a self,
        primary: &str,
    ) -> impl ExactSizeIterator<Item = CompanionView<'a>> + Clone + 'a {
        let start = self
            .companions
            .partition_point(|&index| self.row(index).primary() < primary);
        let end = self
            .companions
            .partition_point(|&index| self.row(index).primary() <= primary);
        self.companions[start..end]
            .iter()
            .map(move |&index| self.row(index))
    }
    pub(super) fn unknown_primaries<'a>(
        &'a self,
        tasks: &'a [crate::ReplicatedTextMaterializationTask],
    ) -> impl Iterator<Item = &'a str> + Clone + 'a {
        self.companions
            .iter()
            .enumerate()
            .filter_map(move |(position, &index)| {
                let primary = self.row(index).primary();
                if position != 0 && self.row(self.companions[position - 1]).primary() == primary {
                    return None;
                }
                (!tasks.iter().any(|task| task.name() == primary)).then_some(primary)
            })
    }
}
