//! The canonical identity reducer and its original metadata destinations.
use super::*;
use crate::{HostMetadataFunding, HostMetadataFundingError};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::atomic::{AtomicBool, Ordering},
};

/// A failed identity preparation retains the account used before its producers.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ArtifactIdentityPreparationError {
    #[source]
    cause: ArtifactError,
    _funding: HostMetadataFunding,
}
impl ArtifactIdentityPreparationError {
    /// Fixed funding refusal, without constructing another diagnostic owner.
    pub fn funding_error(&self) -> Option<HostMetadataFundingError> {
        match self.cause {
            ArtifactError::IdentityFunding(error) => Some(error),
            _ => None,
        }
    }
}

// A single original resolver at a time, including ordinary callers. Refusal can
// release the claim without installing an error shell or poisoning the source.
// Waiters retain only their already priced call frames, never a waiter list.
pub(super) struct Resolution<'a>(&'a AtomicBool);
impl<'a> Resolution<'a> {
    pub(super) fn claim(active: &'a AtomicBool) -> Self {
        while active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::thread::yield_now();
        }
        Self(active)
    }
}
impl Drop for Resolution<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
pub(super) struct Destination<'a>(pub(super) Option<&'a HostMetadataFunding>);
impl eredu_checkpoint::artifact::ArtifactFingerprintAllocation for Destination<'_> {
    type Error = HostMetadataFundingError;
    fn reserve(&self, bytes: usize) -> Result<(), Self::Error> {
        self.0
            .map_or(Ok(()), |funding| funding.reserve_metadata(bytes))
    }
}
impl Destination<'_> {
    fn controls<T>(self) -> Result<(), ArtifactError> {
        use eredu_checkpoint::artifact::ArtifactFingerprintAllocation;
        self.reserve(size_of::<(T, Self, Result<T, ArtifactError>, usize)>())?;
        Ok(())
    }
    fn vector<T>(self, count: usize) -> Result<Vec<T>, ArtifactError> {
        use eredu_checkpoint::artifact::ArtifactFingerprintAllocation;
        self.controls::<(Vec<T>, Layout, usize, std::collections::TryReserveError)>()?;
        self.reserve(
            Layout::array::<T>(count)
                .map_err(|_| ArtifactError::IdentityOverflow)?
                .size(),
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(ArtifactError::IdentityAllocation)?;
        if size_of::<T>() != 0 && values.capacity() != count {
            return Err(ArtifactError::IdentityCapacity);
        }
        Ok(values)
    }
    fn grow<T>(self, values: &mut Vec<T>) -> Result<(), ArtifactError> {
        use eredu_checkpoint::artifact::ArtifactFingerprintAllocation;
        self.controls::<(
            &mut Vec<T>,
            Layout,
            usize,
            std::collections::TryReserveError,
        )>()?;
        if values.len() < values.capacity() {
            return Ok(());
        }
        let target = values
            .capacity()
            .checked_mul(2)
            .ok_or(ArtifactError::IdentityOverflow)?
            .max(4);
        self.reserve(
            Layout::array::<T>(target)
                .map_err(|_| ArtifactError::IdentityOverflow)?
                .size(),
        )?;
        values
            .try_reserve_exact(target - values.len())
            .map_err(ArtifactError::IdentityAllocation)?;
        if size_of::<T>() != 0 && values.capacity() != target {
            return Err(ArtifactError::IdentityCapacity);
        }
        Ok(())
    }
    fn text(self, value: fmt::Arguments<'_>) -> Result<String, ArtifactError> {
        use fmt::Write;
        struct Count(usize);
        impl fmt::Write for Count {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        self.controls::<(Count, fmt::Arguments<'_>, String, [u8; 4], fmt::Error)>()?;
        let mut count = Count(0);
        count
            .write_fmt(value)
            .map_err(|_| ArtifactError::IdentityOverflow)?;
        let mut output = String::from_utf8(self.vector(count.0)?).expect("empty UTF-8");
        output
            .write_fmt(value)
            .expect("closed identity diagnostic writer");
        debug_assert_eq!(output.len(), count.0);
        Ok(output)
    }
    fn invalid(self, value: fmt::Arguments<'_>) -> ArtifactError {
        match self.text(value) {
            Ok(text) => ArtifactError::InvalidArtifactIdentity(text),
            Err(error) => error,
        }
    }
}

impl DeferredArtifactIdentity {
    /// Resolves the same exact files and canonical identity under the original
    /// source account. Successful fixed digests share the existing cache; failed
    /// funded attempts retain their diagnostics and leave a later attempt possible.
    /// Ordinary callers use the same file pass, reducer and resolution claim.
    pub fn resolve_with_metadata(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<ArtifactIdentity, ArtifactIdentityPreparationError> {
        let destination = Destination(Some(funding));
        let result = (|| {
            destination.controls::<(
                &Self,
                Resolution<'_>,
                ArtifactIdentity,
                ArtifactError,
                ArtifactIdentityPreparationError,
                Option<&Result<ArtifactIdentity, Arc<ArtifactError>>>,
                Vec<ArtifactMemberFingerprint>,
                Result<bool, bool>,
            )>()?;
            if let Some(result) = self.0.identity.get() {
                return result.clone().map_err(ArtifactError::SharedIdentity);
            }
            let _resolution = Resolution::claim(&self.0.resolving);
            if let Some(result) = self.0.identity.get() {
                return result.clone().map_err(ArtifactError::SharedIdentity);
            }
            #[cfg(unix)]
            {
                let (domain, source) = self
                    .0
                    .source
                    .as_ref()
                    .expect("unresolved identity retains sources");
                let members = source.fingerprint_with_allocations(&destination).map_err(
                    |error| {
                        use eredu_checkpoint::artifact::ArtifactFingerprintPreparationError as E;
                        match error {
                            E::Source(error) => ArtifactError::ArtifactFingerprint(error),
                            E::Funding(error) => ArtifactError::IdentityFunding(error),
                            E::Overflow => ArtifactError::IdentityOverflow,
                            E::Allocation(error) => ArtifactError::IdentityAllocation(error),
                            E::Capacity => ArtifactError::IdentityCapacity,
                        }
                    },
                )?;
                let identity = fingerprint(
                    domain,
                    members.into_iter().map(ArtifactMemberIdentity::from),
                    destination,
                )?;
                assert!(
                    self.0.identity.set(Ok(identity)).is_ok(),
                    "exclusive original resolver"
                );
                Ok(identity)
            }
            #[cfg(not(unix))]
            {
                Err(ArtifactError::IdentityPlatform)
            }
        })();
        result.map_err(|cause| ArtifactIdentityPreparationError {
            cause,
            _funding: funding.clone(),
        })
    }
}

pub(super) fn fingerprint<I: IntoIterator<Item = ArtifactMemberIdentity>>(
    domain: &str,
    members: I,
    destination: Destination<'_>,
) -> Result<ArtifactIdentity, ArtifactError> {
    destination.controls::<(
        &str,
        I,
        I::IntoIter,
        ArtifactMemberIdentity,
        Option<ArtifactMemberIdentity>,
        Vec<ArtifactMemberIdentity>,
        usize,
        Option<usize>,
        sha2::Sha256,
        [u8; 32],
        sha2::digest::Output<sha2::Sha256>,
    )>()?;
    if domain.is_empty() {
        return Err(destination.invalid(format_args!("artifact identity domain must not be empty")));
    }
    let incoming = members.into_iter();
    let mut members = destination.vector(incoming.size_hint().0)?;
    for member in incoming {
        destination.grow(&mut members)?;
        members.push(member);
    }
    if members.is_empty() {
        return Err(
            destination.invalid(format_args!("artifact identity requires at least one file"))
        );
    }
    if members.iter().any(|member| member.logical_role.is_empty()) {
        return Err(destination.invalid(format_args!("artifact member has an empty logical role")));
    }
    destination.controls::<(
        &mut [ArtifactMemberIdentity],
        std::ops::Range<usize>,
        usize,
        usize,
        usize,
        usize,
        std::cmp::Ordering,
        std::slice::Windows<'_, ArtifactMemberIdentity>,
    )>()?;
    // Canonical ascending roles with fixed iterative controls. No opaque sort
    // scratch or input-dependent recursive frames are entered in either policy.
    sort(&mut members);
    if let Some(pair) = members
        .windows(2)
        .find(|pair| pair[0].logical_role == pair[1].logical_role)
    {
        return Err(destination.invalid(format_args!(
            "duplicate artifact logical role {:?}",
            pair[0].logical_role
        )));
    }
    let mut hasher = sha2::Sha256::new();
    hash_identity_component(&mut hasher, b"eredu-checkpoint-artifact-v2");
    hash_identity_component(&mut hasher, domain.as_bytes());
    use sha2::Digest as _;
    hasher.update((members.len() as u64).to_le_bytes());
    for member in members {
        hash_identity_component(&mut hasher, member.logical_role.as_bytes());
        hasher.update(member.length.to_le_bytes());
        hasher.update(member.digest);
    }
    Ok(ArtifactIdentity(hasher.finalize().into()))
}
fn sort(values: &mut [ArtifactMemberIdentity]) {
    fn sift(values: &mut [ArtifactMemberIdentity], mut root: usize, end: usize) {
        while root < end / 2 {
            let left = root * 2 + 1;
            let right = left + 1;
            let child = if right < end && values[left].logical_role < values[right].logical_role {
                right
            } else {
                left
            };
            if values[root].logical_role >= values[child].logical_role {
                break;
            }
            values.swap(root, child);
            root = child;
        }
    }
    for root in (0..values.len() / 2).rev() {
        sift(values, root, values.len());
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift(values, 0, end);
    }
}

#[cfg(all(test, unix))]
mod tests;
