//! Price the actual native callback and work types before role admission.
use crate::backend::runtime::residency::manager::OriginalMaterializedLoan;
use crate::backend::error::Error;
use crate::composition::mlx::speculative::embedded_native::{
    ActiveEmbeddedNativeInvocation, EmbeddedNativeLayout,
};
use eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody;
use safemlx::{OriginalBufferBudget, SubmissionScope};

pub(super) fn handler_controls<Q: 'static, W, T, B, F, G, H>(
    layout: &EmbeddedNativeLayout,
    work: &W,
    handlers: &(F, G, H),
) -> Option<u64>
where
    F: FnOnce(&mut W, &Q, &SubmissionScope, OriginalMaterializedLoan<'_>) -> Result<B, Error>,
    G: FnOnce(&mut W, &Q, &ActiveEmbeddedNativeInvocation<'_>) -> Result<T, Error>,
    H: FnOnce(
        &mut W,
        &T,
        &OriginalBufferBudget,
        &OriginalSpeculativeBudgetCustody,
    ) -> Result<(), Error>,
{
    let _ = work;
    layout.control_bytes::<Q, W, T, B, F, G, H>(handlers)
}

/// Borrows the actual source and prepays the two concrete escaping error
/// transports. It never wraps or moves the large model execution callback.
pub(super) struct PhaseSource<'a> {
    source: &'a crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
    funding: &'a eredu_nn::workspace::HostMetadataFunding,
    backend: std::cell::RefCell<Option<crate::composition::mlx::model::PreparedPlanningError<Error>>>,
    admission: std::cell::RefCell<Option<crate::composition::mlx::model::PreparedPlanningError<
        eredu_runtime::working_memory::SpeculativeRequestError>>>,
}
impl<'a> PhaseSource<'a> {
    pub(super) fn new(
        source: &'a crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
        funding: &'a eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, eredu_nn::workspace::HostMetadataFundingError> {
        use crate::composition::mlx::model::PreparedPlanningError;
        funding.reserve_metadata(std::mem::size_of::<(Self,Result<Self,eredu_nn::workspace::HostMetadataFundingError>,
            Option<&dyn std::error::Error>, &Error, bool,
            std::cell::RefMut<'_,Option<PreparedPlanningError<Error>>>,
            std::cell::RefMut<'_,Option<PreparedPlanningError<eredu_runtime::working_memory::SpeculativeRequestError>>>,
        )>())?;
        let backend=PreparedPlanningError::prepare(funding)?;
        let admission=PreparedPlanningError::prepare(funding)?;
        Ok(Self {source,funding,backend:std::cell::RefCell::new(Some(backend)),
            admission:std::cell::RefCell::new(Some(admission))})
    }
    pub(super) fn retain_error(&self, cause: Error) -> Error {
        use crate::composition::mlx::model::planning_error_has_funding;
        if planning_error_has_funding::<Error>(&cause,self.funding)
            || planning_error_has_funding::<eredu_runtime::working_memory::SpeculativeRequestError>(&cause,self.funding) {
            return cause;
        }
        let prepared=self.backend.borrow_mut().take();
        match prepared {
            Some(prepared)=>prepared.retain(cause),
            // A distinct second cause needs a distinct paid owner. The escaped
            // first diagnostic keeps its original receipt and is never reused.
            None=>crate::composition::mlx::model::retain_planning_error(cause,self.funding.clone()),
        }
    }
    pub(super) fn retain_admission_error(&self,cause:eredu_runtime::working_memory::SpeculativeRequestError)->Error {
        let prepared=self.admission.borrow_mut().take();
        match prepared {
            Some(prepared)=>prepared.retain(cause),
            None=>crate::composition::mlx::model::retain_planning_error(cause,self.funding.clone()),
        }
    }
    pub(super) fn retain_startup_error<E: std::error::Error + Send + Sync + 'static>(
        &self, cause: E,
    ) -> Error {
        let cause=crate::composition::mlx::model::retain_planning_error(cause,self.funding.clone());
        self.retain_error(cause)
    }
}
impl std::ops::Deref for PhaseSource<'_> {
    type Target = crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources;
    fn deref(&self) -> &Self::Target { self.source }
}
