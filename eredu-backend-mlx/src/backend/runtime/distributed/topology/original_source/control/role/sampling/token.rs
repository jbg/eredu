//! Sampling and expert integer leaves share the registered initialized publisher.
use super::*;
use crate::backend::initialized_input;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, OriginalHostSourceBank, WorkingMemoryError,
};

fn caller_control_bytes() -> Result<usize, WorkingMemoryError> {
    let parts = [
        size_of::<BankLoan<'_>>(),
        size_of::<(&Owner, PreparedInputPlan<'_>)>(),
        size_of::<Option<OriginalHostSourceBank>>(),
        size_of::<std::cell::RefMut<'_, Option<OriginalHostSourceBank>>>(),
        size_of::<Result<usize, WorkingMemoryError>>(),
        size_of::<usize>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
}
fn storage_bytes(plan: &PreparedInputPlan<'_>) -> Result<(u64, u64), WorkingMemoryError> {
    initialized_input::storage_bytes(plan, caller_control_bytes()?)
}
impl OriginalParallelSource {
    pub(crate) fn sampling_source_facts(
        &self,
        outputs: u64,
    ) -> Result<HostSourceConstructionFacts, Error> {
        reserve(
            self.funding(),
            &[
                size_of::<[u32; 1]>(),
                size_of::<[usize; 1]>(),
                size_of::<PreparedInputPlan<'_>>(),
                size_of::<HostSourceConstructionFacts>(),
                size_of::<Result<HostSourceConstructionFacts, WorkingMemoryError>>(),
                size_of::<Result<HostSourceConstructionFacts, Error>>(),
                size_of::<Result<PreparedInputPlan<'_>, safemlx::PreparedInputCause>>(),
                size_of::<Result<(u64, u64), WorkingMemoryError>>(),
                size_of::<(&Self, u64)>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let source = self.communication_source()?;
        let inputs = self
            .agreement_inputs()
            .ok_or_else(|| failure(Cause::SamplingIdentity, source.source(), self.funding()))?;
        let values = [0u32];
        let shape = [1usize];
        let plan = inputs
            .runtime()
            .u32(&values, &shape)
            .map_err(|cause| failure(Cause::Input(cause), source.source(), self.funding()))?;
        let count = outputs.checked_mul(2).ok_or_else(overflow)?;
        let (bytes, _) = storage_bytes(&plan).map_err(Error::PrefillControl)?;
        HostSourceConstructionFacts::new(
            bytes.checked_mul(count).ok_or_else(overflow)?,
            usize::try_from(count).map_err(|_| overflow())?,
            0,
        )
        .map_err(Error::PrefillControl)
    }
}
impl OriginalParallelSource {
    /// Physical source ceiling for actual integer rows. The prepared input
    /// producer uses the same I32 descriptor/allocator/control geometry for
    /// initialized zeros and a borrowed I32 copy; contents only choose filling.
    /// This cold plan is never constructed or used as route-ID evidence.
    pub(crate) fn expert_input_source_facts(
        &self,
        maximum_rows: usize,
        attempts: usize,
    ) -> Result<HostSourceConstructionFacts, Error> {
        reserve(
            self.funding(),
            &[
                size_of::<(&Self, usize, usize)>(),
                size_of::<[usize; 2]>(),
                size_of::<PreparedInputPlan<'_>>(),
                size_of::<HostSourceConstructionFacts>(),
                size_of::<Result<HostSourceConstructionFacts, Error>>(),
                size_of::<Result<PreparedInputPlan<'_>, safemlx::PreparedInputCause>>(),
                size_of::<Result<(u64, u64), WorkingMemoryError>>(),
                safemlx::PreparedInputRuntime::zeros_plan_control_bytes(),
            ],
        )?;
        let source = self.communication_source()?;
        let inputs = self
            .agreement_inputs()
            .ok_or_else(|| failure(Cause::SamplingIdentity, source.source(), self.funding()))?;
        // The shared empty-row source is its actual one-row seed followed by
        // an independently priced empty Slice. Rank two also covers the
        // movement worker's rank-one index source from this same producer.
        let shape = [maximum_rows.max(1), 1];
        let plan = inputs
            .runtime()
            .zeros(safemlx::Dtype::Int32, &shape)
            .map_err(|cause| failure(Cause::Input(cause), source.source(), self.funding()))?;
        let (bytes, _) = storage_bytes(&plan).map_err(Error::PrefillControl)?;
        HostSourceConstructionFacts::new(
            bytes
                .checked_mul(u64::try_from(attempts).map_err(|_| overflow())?)
                .ok_or_else(overflow)?,
            attempts,
            0,
        )
        .map_err(Error::PrefillControl)
    }
}
impl OriginalParallelControlOwner {
    pub(crate) fn with_token_sources(
        self,
        bank: Option<OriginalHostSourceBank>,
    ) -> Result<Self, Error> {
        let owner = self.owner();
        reserve(
            &owner.custody.funding,
            &[
                size_of::<Self>(),
                size_of::<Option<OriginalHostSourceBank>>(),
                size_of::<eredu_runtime::working_memory::OriginalHostSourceCustody>(),
                size_of::<(
                    &OriginalHostSourceBank,
                    &eredu_runtime::working_memory::OriginalHostSourceCustody,
                )>(),
                size_of::<Result<Self, Error>>(),
                size_of::<std::cell::Ref<'_, Option<OriginalHostSourceBank>>>(),
                size_of::<std::cell::RefMut<'_, Option<OriginalHostSourceBank>>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let bank = bank.ok_or(Error::PredictionScopeUnavailable)?;
        if !bank.belongs_to_source(&owner.native.host_source_custody())
            || owner.token_sources.borrow().is_some()
        {
            return Err(control_error(
                ControlCause::Identity,
                &owner.custody.source,
                &owner.custody.funding,
            ));
        }
        *owner.token_sources.borrow_mut() = Some(bank);
        Ok(self)
    }
}
struct BankLoan<'a> {
    bank: Option<OriginalHostSourceBank>,
    destination: &'a RefCell<Option<OriginalHostSourceBank>>,
}
impl Drop for BankLoan<'_> {
    fn drop(&mut self) {
        let previous = self.destination.replace(self.bank.take());
        debug_assert!(previous.is_none());
        drop(previous);
    }
}
pub(in super::super) fn construct(
    owner: &Owner,
    plan: PreparedInputPlan<'_>,
) -> Result<Array, Error> {
    let bank = owner
        .token_sources
        .try_borrow_mut()
        .map_err(|_| Error::PredictionScopeReentrant)?
        .take()
        .ok_or(Error::PredictionScopeUnavailable)?;
    let mut loan = BankLoan {
        bank: Some(bank),
        destination: &owner.token_sources,
    };
    let custody = owner.native.host_source_custody();
    initialized_input::construct(
        loan.bank.as_mut().expect("exclusive source bank"),
        &custody,
        plan,
        caller_control_bytes().map_err(Error::PrefillControl)?,
    )
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[test]
fn initialized_integer_source_census_matches_actual_copy_and_empty_seed() {
    if !crate::tests::support::native_process::enter("original-integer-source") {
        return;
    }
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let runtime = safemlx::PreparedInputRuntime::prepare().unwrap();
    // Cross source backing pages, and include the actual empty-route seed.
    // Both plans come from the same live allocator and native layout producer.
    for rows in [1usize, 2, 65, 16385] {
        let values = vec![17i32; rows];
        let shape = [rows, 1];
        let copy = runtime.i32(&values, &shape).unwrap();
        let fill = runtime.zeros(safemlx::Dtype::Int32, &shape).unwrap();
        assert_eq!(storage_bytes(&copy).unwrap(), storage_bytes(&fill).unwrap());
        let flat = [rows];
        let flat = runtime.i32(&values, &flat).unwrap();
        let (actual, backing) = storage_bytes(&flat).unwrap();
        let (ceiling, maximum_backing) = storage_bytes(&fill).unwrap();
        assert!(actual <= ceiling);
        assert_eq!(backing, maximum_backing);
    }
}
