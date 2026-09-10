//! Scalar bank binding consumes the exact selected member recipes and one cache budget.
use super::*;
use eredu_architectures::{
    prepared_execution::PartitionBankMechanisms, RoutedGroupedPlan, SelectedRoutedBank,
};
use std::rc::Rc;

#[derive(Default)]
struct Pool {
    resident: Vec<(ParameterBankKey, u64)>,
    bytes: u64,
    budget: u64,
}

pub(super) struct Bank {
    inner: NumericGroupedBankMechanism,
    pool: Rc<RefCell<Pool>>,
}

impl AddressableGroupedBank<NumericBackend> for Bank {
    type Acquisition = NumericBankAcquisition;
    type Report = NumericBankReport;
    type Error = Error;
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.inner.member_bytes(key)
    }
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        context: &NumericContext,
    ) -> Result<Self::Acquisition, Error> {
        let mut pool = self.pool.borrow_mut();
        let requested = request
            .entries()
            .iter()
            .map(|(key, _)| self.inner.bytes[key])
            .sum::<u64>();
        if requested > pool.budget {
            return Err(Error::backend(
                "scalar bank acquisition exceeds shared byte budget",
            ));
        }
        let mut selected = Vec::new();
        for (key, _) in request.entries() {
            if let Some(index) = pool
                .resident
                .iter()
                .position(|(resident, _)| resident == key)
            {
                let (_, bytes) = pool.resident.remove(index);
                pool.bytes -= bytes;
            }
            selected.push((*key, self.inner.bytes[key]));
        }
        while pool.bytes + requested > pool.budget {
            let (_, bytes) = pool.resident.remove(0);
            pool.bytes -= bytes;
            REFERENCE_STAGE_EVIDENCE.with(|e| e.borrow_mut().bank_evictions += 1);
        }
        pool.bytes += requested;
        pool.resident.extend(selected);
        REFERENCE_STAGE_EVIDENCE.with(|e| {
            let mut e = e.borrow_mut();
            e.peak_bank_bytes = e.peak_bank_bytes.max(pool.bytes);
            e.bank_acquisitions
                .extend(request.entries().iter().map(|(key, _)| *key));
        });
        self.inner.acquire(request, context)
    }
    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &GroupedGatedProductSpec,
        context: &NumericContext,
    ) -> Result<NumericExpertBank, Error> {
        self.inner.gated_product_groups(acquisition, spec, context)
    }
    fn linear_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        context: &NumericContext,
    ) -> Result<grouped_linear::NumericLinearGroups, Error> {
        self.inner.linear_groups(acquisition, spec, context)
    }
    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &GroupedRelu2Spec,
        context: &NumericContext,
    ) -> Result<NumericRelu2Groups, Error> {
        self.inner.relu2_groups(acquisition, spec, context)
    }
    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &NumericTensor,
        context: &NumericContext,
    ) -> Result<(), Error> {
        REFERENCE_STAGE_EVIDENCE.with(|e| e.borrow_mut().bank_completions += 1);
        self.inner.complete(acquisition, output, context)
    }
    fn report(&self) -> Result<Self::Report, Error> {
        self.inner.report()
    }
}

pub(super) fn bind(
    banks: &BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
    options: ParameterBankLoadOptions,
    checkpoint: &dyn eredu_checkpoint::store::CheckpointSource,
    context: &NumericContext,
) -> Result<
    BTreeMap<
        eredu_runtime::RoutedBankId,
        PartitionBankMechanisms<Bank, NumericIndexedMovement, ()>,
    >,
    Error,
> {
    let pool = Rc::new(RefCell::new(Pool {
        budget: options.offload().device_budget_bytes().unwrap(),
        ..Pool::default()
    }));
    banks
        .iter()
        .filter(|(_, bank)| !bank.addressable_members().is_empty())
        .map(|(id, bank)| {
            let mut gated = BTreeMap::new();
            let mut linear = BTreeMap::new();
            let mut bytes = BTreeMap::new();
            for member in bank.addressable_members() {
                let (values, selected_bytes) =
                    payload::addressable_values(member, checkpoint, context)?;
                let unit = member.key().unit();
                let group = member.placement().owner_group().as_str();
                match bank.plan() {
                    RoutedGroupedPlan::Gated(plan) => {
                        let spec = plan.unit_spec(group, unit).unwrap();
                        let spec = spec
                            .clone()
                            .with_group_geometry(1, spec.intermediate_dimensions())?;
                        let gu = &values["gate_up_proj"];
                        let down = &values["down_proj"];
                        let width = spec.intermediate_dimensions() as usize;
                        let gu = NumericTensor::new(gu.shape[1..].to_vec(), gu.data.clone());
                        gated.insert(
                            member.key(),
                            NumericExpertBank {
                                experts: vec![NumericExpert {
                                    gate: gu.axis_slice(0, 0, width),
                                    up: gu.axis_slice(0, width, 2 * width),
                                    down: NumericTensor::new(
                                        down.shape[1..].to_vec(),
                                        down.data.clone(),
                                    ),
                                    gate_bias: None,
                                    up_bias: None,
                                    down_bias: None,
                                }],
                                parameters: Vec::new(),
                                policy: spec.policy(),
                                spec,
                            },
                        );
                    }
                    RoutedGroupedPlan::Linear(plan) => {
                        let spec = plan
                            .unit_spec(group, unit)
                            .unwrap()
                            .clone()
                            .with_group_count(1)?;
                        linear.insert(
                            member.key(),
                            grouped_linear::NumericLinearGroups::from_bound_weight(
                                spec,
                                values["weight"].clone(),
                            )?,
                        );
                    }
                    _ => {
                        return Err(Error::backend(
                            "scalar partition payload binder has no selected equation mechanism",
                        ))
                    }
                }
                bytes.insert(member.key(), selected_bytes);
            }
            let bank = Bank {
                inner: NumericGroupedBankMechanism {
                    banks: gated,
                    linear,
                    bytes: bytes.clone(),
                    report: NumericBankReport::default(),
                    resident: Vec::new(),
                    capacity: usize::MAX,
                },
                pool: Rc::clone(&pool),
            };
            Ok((
                *id,
                PartitionBankMechanisms::new(bytes, bank, NumericIndexedMovement, ()),
            ))
        })
        .collect()
}
