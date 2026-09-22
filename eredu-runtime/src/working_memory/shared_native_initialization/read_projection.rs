//! Projection admission retaining the actual original read owner and account.
use super::*;
use eredu_checkpoint::store::{
    EncodedProjectionBuildError, EncodedReadProjectionPlan, PreparedEncodedRead,
};

impl<C> InitializedSharedNative<PreparedEncodedRead<C>> {
    /// Transfer the read and this owner's actual account together. Keeping the
    /// account in the read permits subsequent consuming constructors without
    /// releasing source metadata admission or exposing an uncharged output.
    pub fn into_owned_read(self) -> PreparedEncodedRead<(C, SharedNativeInitializationCustody)> {
        self.output
            .with_custody(SharedNativeInitializationCustody(self.account, None))
    }
}

impl<C> SharedNativeInitializer for EncodedReadProjectionPlan<'_, C> {
    type Output = PreparedEncodedRead<(C, SharedNativeInitializationCustody)>;
    type Error = EncodedProjectionBuildError<(C, SharedNativeInitializationCustody)>;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.required_bytes::<SharedNativeInitializationCustody>()
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        self.construct(custody)
    }
}

#[cfg(test)]
mod tests;
