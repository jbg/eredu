//! One role-validation worker for ordinary and original source construction.
use super::{ArchitecturePartitionError as Error, PartitionOwnership};
use eredu_collections::ordered_map::{Map, TryInsertError};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

impl PartitionOwnership {
    /// Creates validated boundary and static-module ownership.
    pub fn new(
        input: bool,
        output: bool,
        roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, Error> {
        Self::from_owned(
            input,
            output,
            roles.into_iter().map(Into::into).collect(),
            None,
        )
    }
    /// Validates already funded role strings through the original worker.
    /// The caller retains the actual account with every escaping source/error.
    pub fn new_with_funding(
        input: bool,
        output: bool,
        roles: Vec<String>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::from_owned(input, output, roles, Some(funding))
    }
    fn from_owned(
        input: bool,
        output: bool,
        roles: Vec<String>,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self, Error> {
        controls(funding)?;
        validate(&roles, funding)?;
        Ok(Self {
            input,
            output,
            static_roles: roles,
            replicated_static_roles: Vec::new(),
        })
    }
    /// Retains auxiliary storage roles without granting invocation ownership.
    pub fn with_replicated_static_roles(
        self,
        roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, Error> {
        self.with_owned_replicated(roles.into_iter().map(Into::into).collect(), None)
    }
    /// Attaches already funded replica roles through the same validation worker.
    /// The enclosing source retains the account through the result's lifetime.
    pub fn with_replicated_static_roles_funded(
        self,
        roles: Vec<String>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        self.with_owned_replicated(roles, Some(funding))
    }
    fn with_owned_replicated(
        mut self,
        roles: Vec<String>,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self, Error> {
        controls(funding)?;
        validate(&roles, funding)?;
        self.replicated_static_roles = roles;
        Ok(self)
    }
}
fn reserve(funding: Option<&HostMetadataFunding>, bytes: usize) -> Result<(), Error> {
    if let Some(funding) = funding {
        funding.reserve_metadata(bytes)?;
    }
    Ok(())
}
fn controls(funding: Option<&HostMetadataFunding>) -> Result<(), Error> {
    let parts = [
        size_of::<PartitionOwnership>(),
        size_of::<Vec<String>>(),
        size_of::<Map<&str, ()>>(),
        size_of::<std::slice::Iter<'_, String>>(),
        size_of::<Option<&HostMetadataFunding>>(),
        size_of::<Result<PartitionOwnership, Error>>(),
        size_of::<(bool, bool, &str, Layout)>(),
        HostMetadataFunding::reservation_control_bytes(),
    ];
    reserve(
        funding,
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)?,
    )
}
fn validate(roles: &[String], funding: Option<&HostMetadataFunding>) -> Result<(), Error> {
    let mut unique = Map::new();
    for role in roles {
        if role.trim().is_empty() {
            return Err(Error::EmptyStaticRole);
        }
        let key = role.as_str();
        reserve(
            funding,
            unique
                .insertion_control_bytes(&key)
                .ok_or(HostMetadataFundingError::Overflow)?,
        )?;
        let previous = unique
            .try_insert_with(key, (), |layout| reserve(funding, layout.size()))
            .map_err(|cause| match cause {
                TryInsertError::Funding(cause) => cause,
                TryInsertError::SizeOverflow => {
                    Error::MetadataFunding(HostMetadataFundingError::Overflow)
                }
            })?;
        if previous.is_some() {
            let fixed = size_of::<(
                Vec<u8>,
                String,
                Layout,
                std::collections::TryReserveError,
                Result<String, std::string::FromUtf8Error>,
                Error,
            )>();
            reserve(
                funding,
                fixed
                    .checked_add(role.len())
                    .ok_or(HostMetadataFundingError::Overflow)?,
            )?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(role.len())?;
            if bytes.capacity() != role.len() {
                return Err(Error::MetadataCapacity);
            }
            bytes.extend_from_slice(role.as_bytes());
            return Err(Error::DuplicateStaticRole(
                String::from_utf8(bytes).expect("copied UTF-8"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Debug)]
    struct Account {
        calls: Arc<AtomicUsize>,
        stop: usize,
    }
    impl eredu_core::HostMetadataAccount for Account {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(call <= self.stop, "producer after refusal");
            if call == self.stop {
                Err(HostMetadataFundingError::Unavailable)
            } else {
                Ok(())
            }
        }
    }
    fn funding(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        // The account's shared shell is itself a real constructor reservation.
        let funding = HostMetadataFunding::new(Account {
            calls: calls.clone(),
            stop: stop + 1,
        })
        .unwrap();
        (funding, calls)
    }
    #[test]
    fn ownership_preserves_order_and_every_reached_refusal() {
        for roles in [vec!["z", "a", "middle"], vec!["a", "a"], vec!["a", " "]] {
            let expected = PartitionOwnership::new(true, false, roles.clone());
            let (funding, calls) = funding(usize::MAX - 1);
            let actual = PartitionOwnership::new_with_funding(
                true,
                false,
                roles.iter().map(|s| s.to_string()).collect(),
                &funding,
            );
            assert_eq!(actual, expected);
            let count = calls.load(Ordering::SeqCst) - 1;
            for stop in 0..count {
                let (funding, calls) = self::funding(stop);
                let result = PartitionOwnership::new_with_funding(
                    true,
                    false,
                    roles.iter().map(|s| s.to_string()).collect(),
                    &funding,
                );
                assert!(
                    matches!(
                        result,
                        Err(Error::MetadataFunding(
                            HostMetadataFundingError::Unavailable
                        ))
                    ),
                    "cut {stop}"
                );
                assert_eq!(calls.load(Ordering::SeqCst), stop + 2);
            }
        }
    }
    #[test]
    fn replica_roles_do_not_grant_ordinary_invocation_ownership() {
        let (funding, _) = funding(usize::MAX - 1);
        let source =
            PartitionOwnership::new_with_funding(false, false, vec!["embedding".into()], &funding)
                .unwrap()
                .with_replicated_static_roles_funded(
                    vec!["projector".into(), "embedding".into()],
                    &funding,
                )
                .unwrap();
        assert!(source.owns_static_role("embedding"));
        assert!(!source.owns_static_role("projector"));
        assert!(source.stores_static_role("projector"));
        assert!(!source.owns_input());
        assert!(!source.owns_output());
        assert_eq!(source.replicated_static_roles(), ["projector", "embedding"]);
    }
}
