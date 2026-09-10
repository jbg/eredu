//! Equation-erased providers selected by portable preparation.
use super::*;

/// One resident bank with its selected equation and per-unit route proof.
pub enum PlannedResidentBank {
    /// SwiGLU and other gated-product policies.
    Gated(PlannedResidentGatedProduct),
    /// ReLU-squared feed-forward banks.
    Relu2(PlannedResidentRelu2),
    /// Selected activated projections with owned output rows.
    Linear {
        /// Group owning the logical invocations.
        owner_group: eredu_runtime::ExecutionGroupId,
        /// Selected local projection geometry.
        plan: ExpertRealizationPlan<eredu_nn::GroupedLinearSpec>,
        /// Route cardinality per logical invocation.
        routes_by_unit: BTreeMap<usize, usize>,
    },
}
impl PlannedResidentBank {
    pub(crate) fn from_partitioned(
        bank: &SelectedRoutedBank,
        exchange: bool,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes = bank
            .routes_by_unit
            .iter()
            .map(|(unit, count)| (*unit, if exchange { 1 } else { *count }))
            .collect::<BTreeMap<_, _>>();
        let maximum = routes.values().copied().max().unwrap_or(0);
        Ok(match &bank.plan {
            RoutedGroupedPlan::Gated(plan) => {
                Self::Gated(PlannedResidentGatedProduct::new_partitioned_with_routes(
                    bank.owner_group.clone(),
                    plan.clone(),
                    routes,
                )?)
            }
            RoutedGroupedPlan::Relu2(plan) => Self::Relu2(PlannedResidentRelu2::new_partitioned(
                bank.owner_group.clone(),
                plan.clone(),
                maximum,
            )?),
            RoutedGroupedPlan::Linear(plan) => {
                validate_routes_by_unit::<LinearOperation>(plan, &routes)
                    .map_err(|e| RoutedTextExecutionError::Contract(e.to_string()))?;
                if plan
                    .unit_specs()
                    .keys()
                    .any(|(owner, _)| owner != &bank.owner_group)
                {
                    return Err(RoutedTextExecutionError::Contract(
                        "partitioned linear bank names a different owner".into(),
                    ));
                }
                Self::Linear {
                    owner_group: bank.owner_group.clone(),
                    plan: plan.clone(),
                    routes_by_unit: routes,
                }
            }
        })
    }

    pub(super) fn from_selected(
        bank: SelectedRoutedBank,
    ) -> Result<Self, RoutedTextExecutionError> {
        let SelectedRoutedBank {
            owner_group,
            plan,
            routes_by_unit,
            routes_per_token,
            ..
        } = bank;
        Ok(match plan {
            RoutedGroupedPlan::Gated(plan) => {
                validate_replicated_plan(&plan)?;
                Self::Gated(PlannedResidentGatedProduct {
                    owner_group,
                    plan,
                    routes_by_unit,
                })
            }
            RoutedGroupedPlan::Relu2(plan) => {
                validate_replicated_plan(&plan)?;
                Self::Relu2(PlannedResidentRelu2 {
                    owner_group,
                    plan,
                    routes_per_token,
                })
            }
            RoutedGroupedPlan::Linear(plan) => {
                validate_replicated_plan(&plan)?;
                Self::Linear {
                    owner_group,
                    plan,
                    routes_by_unit,
                }
            }
        })
    }
}

/// One addressable bank, using the same chunking, lease, and completion machinery.
pub enum PlannedAddressableBank<B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Movement: IndexedMovement<B>,
{
    /// Gated-product equation.
    Gated(PlannedAddressableGatedProduct<B, Bank, Movement>),
    /// ReLU-squared equation.
    Relu2(PlannedAddressableRelu2<B, Bank, Movement>),
    /// Activated linear projection equation.
    Linear(PlannedAddressableLinear<B, Bank, Movement>),
}
impl<B, Bank, Movement> PlannedAddressableBank<B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    pub(crate) fn from_partitioned(
        selected: &SelectedRoutedBank,
        exchange: bool,
        bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes = if exchange {
            1
        } else {
            selected.routes_per_token
        };
        Ok(match &selected.plan {
            RoutedGroupedPlan::Gated(plan) => {
                Self::Gated(PlannedAddressableGatedProduct::new_partitioned(
                    selected.owner_group.clone(),
                    plan.clone(),
                    selected.catalog.clone(),
                    bytes,
                    bank,
                    movement,
                    options,
                    routes,
                )?)
            }
            RoutedGroupedPlan::Relu2(plan) => {
                Self::Relu2(PlannedAddressableRelu2::new_partitioned(
                    selected.owner_group.clone(),
                    plan.clone(),
                    selected.catalog.clone(),
                    bytes,
                    bank,
                    movement,
                    options,
                    routes,
                )?)
            }
            RoutedGroupedPlan::Linear(plan) => {
                Self::Linear(PlannedAddressableLinear::new_partitioned(
                    selected.owner_group.clone(),
                    plan.clone(),
                    selected.catalog.clone(),
                    bytes,
                    bank,
                    movement,
                    options,
                    routes,
                )?)
            }
        })
    }

    pub(super) fn from_selected(
        selected: SelectedRoutedBank,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<Self, RoutedTextExecutionError> {
        let SelectedRoutedBank {
            owner_group,
            plan,
            catalog,
            routes_by_unit,
            addressable_members,
            ..
        } = selected;
        let bytes = addressable_members
            .iter()
            .map(|member| (member.key(), member.selected_bytes()))
            .collect();
        Ok(match plan {
            RoutedGroupedPlan::Gated(plan) => {
                Self::Gated(PlannedAddressableGatedProduct::from_validated_routes(
                    owner_group,
                    plan,
                    catalog,
                    bytes,
                    bank,
                    movement,
                    options,
                    routes_by_unit,
                )?)
            }
            RoutedGroupedPlan::Relu2(plan) => {
                Self::Relu2(PlannedAddressableRelu2::from_validated_routes(
                    owner_group,
                    plan,
                    catalog,
                    bytes,
                    bank,
                    movement,
                    options,
                    routes_by_unit,
                )?)
            }
            RoutedGroupedPlan::Linear(plan) => {
                Self::Linear(PlannedAddressableLinear::from_validated_routes(
                    owner_group,
                    plan,
                    catalog,
                    bytes,
                    bank,
                    movement,
                    options,
                    routes_by_unit,
                )?)
            }
        })
    }
    /// Reports one bank without erasing its storage's report type.
    pub fn bank_report(&self) -> Result<Bank::Report, RoutedTextExecutionError> {
        match self {
            Self::Gated(p) => p.bank_report(),
            Self::Relu2(p) => p.bank_report(),
            Self::Linear(p) => p.bank_report(),
        }
    }
}

impl<B> RoutedExpertProvider<B> for PlannedResidentBank
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_grouped(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_grouped(p, resident, request, context)
            }
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract(
                "selected linear bank received a different grouped equation".into(),
            )),
        }
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_compact_grouped(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_compact_grouped(p, resident, request, context)
            }
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract(
                "selected linear bank received a different grouped equation".into(),
            )),
        }
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_relu2_routed(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_relu2_routed(p, resident, request, context)
            }
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract(
                "selected linear bank received a different grouped equation".into(),
            )),
        }
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_linear_routed(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_linear_routed(p, resident, request, context)
            }
            Self::Linear {
                owner_group,
                plan,
                routes_by_unit,
            } => {
                let routes = routes_by_unit.get(&request.layer).copied().ok_or_else(|| {
                    RoutedTextExecutionError::Contract(
                        "linear bank invocation has no route cardinality".into(),
                    )
                })?;
                validate_route_cardinality(request.routes, routes)?;
                let selected = plan
                    .unit_spec(owner_group.as_str(), request.layer)
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(
                            "linear bank invocation has no specification".into(),
                        )
                    })?;
                if selected != resident.spec() {
                    return Err(RoutedTextExecutionError::Contract(
                        "resident linear bank differs from selected plan".into(),
                    ));
                }
                resident
                    .forward_grouped(request.input, request.routes, context)
                    .map_err(|e| RoutedTextExecutionError::Mechanism(e.to_string()))
            }
        }
    }
}

impl<B, Bank, Movement> RoutedExpertProvider<B> for PlannedAddressableBank<B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_grouped(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_grouped(p, resident, request, context)
            }
            Self::Linear(p) => {
                RoutedExpertProvider::<B>::forward_grouped(p, resident, request, context)
            }
        }
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_compact_grouped(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_compact_grouped(p, resident, request, context)
            }
            Self::Linear(p) => {
                RoutedExpertProvider::<B>::forward_compact_grouped(p, resident, request, context)
            }
        }
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_relu2_routed(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_relu2_routed(p, resident, request, context)
            }
            Self::Linear(p) => {
                RoutedExpertProvider::<B>::forward_relu2_routed(p, resident, request, context)
            }
        }
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match self {
            Self::Gated(p) => {
                RoutedExpertProvider::<B>::forward_linear_routed(p, resident, request, context)
            }
            Self::Relu2(p) => {
                RoutedExpertProvider::<B>::forward_linear_routed(p, resident, request, context)
            }
            Self::Linear(p) => {
                RoutedExpertProvider::<B>::forward_linear_routed(p, resident, request, context)
            }
        }
    }
}

impl<A> PreparedRoutedTextArchitecture<A> {
    /// Constructs resident execution for every selected bank through the shared session driver.
    pub fn construct_resident_session<B, M>(
        self,
        mechanisms: M,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<
                eredu_runtime::RoutedBankProviders<PlannedResidentBank>,
            >,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + GroupedNeuralBackend,
        M: eredu_runtime::ReplicatedTextSessionMechanisms<A, B>,
        A: eredu_runtime::LayeredArchitecture<B, M::State>
            + eredu_runtime::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let (mut modules, residency, banks) = self.into_parts();
        if residency != eredu_runtime::ParameterBankResidency::WithLayer {
            return Err("selected banks are not resident with their layers".into());
        }
        let providers = banks
            .into_iter()
            .map(|(id, bank)| PlannedResidentBank::from_selected(bank).map(|p| (id, p)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let providers =
            eredu_runtime::RoutedBankProviders::new(providers).map_err(|e| e.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(providers),
            context,
        )
        .map_err(|e| e.to_string())
    }
    /// Constructs addressable execution for every selected bank through the shared session driver.
    pub fn construct_addressable_session<B, M, Bank, Movement>(
        self,
        mechanisms: M,
        native_banks: BTreeMap<RoutedBankId, (Bank, Movement)>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<
                eredu_runtime::RoutedBankProviders<PlannedAddressableBank<B, Bank, Movement>>,
            >,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + GroupedNeuralBackend,
        M: eredu_runtime::ReplicatedTextSessionMechanisms<A, B>,
        A: eredu_runtime::LayeredArchitecture<B, M::State>
            + eredu_runtime::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B>,
        Movement::Error: std::fmt::Display,
    {
        let (mut modules, residency, banks) = self.into_parts();
        let eredu_runtime::ParameterBankResidency::IndependentCache(options) = residency else {
            return Err("selected banks are not independently addressable".into());
        };
        if native_banks.keys().ne(banks.keys()) {
            return Err("materialized bank identities differ from selected banks".into());
        }
        let mut native_banks = native_banks;
        let providers = banks
            .into_iter()
            .map(|(id, selected)| {
                let (bank, movement) = native_banks.remove(&id).expect("checked bank identities");
                PlannedAddressableBank::from_selected(selected, bank, movement, options)
                    .map(|p| (id, p))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let providers =
            eredu_runtime::RoutedBankProviders::new(providers).map_err(|e| e.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(providers),
            context,
        )
        .map_err(|e| e.to_string())
    }
}

impl RoutedBankRequirements {
    /// Validates one bank's equation, ownership, addressable catalog, and route counts.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: RoutedGroupedPlan,
        catalog: ExpertResidencyCatalog,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextRequirementsError> {
        macro_rules! validate {
            ($operation:ty, $plan:expr) => {{
                validate_routes_by_unit::<$operation>($plan, &routes_by_unit)?;
                validate_plan_catalog::<$operation>(&owner_group, $plan, &catalog)?;
            }};
        }
        match &plan {
            RoutedGroupedPlan::Gated(plan) => validate!(GatedProductOperation, plan),
            RoutedGroupedPlan::Relu2(plan) => validate!(Relu2Operation, plan),
            RoutedGroupedPlan::Linear(plan) => validate!(LinearOperation, plan),
        }
        let routes_per_token = routes_by_unit.values().copied().max().unwrap_or(0);
        if routes_per_token == 0 {
            return Err(RoutedTextRequirementsError::Invalid(
                "routed bank has no positive route count".into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            catalog,
            routes_by_unit,
            routes_per_token,
        })
    }
}

impl RoutedTextRequirements {
    /// Admits an identified collection against the exact text parameter topology and sources.
    pub fn new(
        text: eredu_runtime::ReplicatedTextRequirements,
        banks: impl IntoIterator<Item = (RoutedBankId, RoutedBankRequirements)>,
        source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
    ) -> Result<Self, RoutedTextRequirementsError> {
        let mut admitted = BTreeMap::new();
        let mut operations = BTreeSet::new();
        for (id, bank) in banks {
            if bank
                .catalog
                .units()
                .iter()
                .any(|unit| unit.identity().bank() != id.value() as usize)
            {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "catalog keys differ from routed bank {id:?}"
                )));
            }
            match &bank.plan {
                RoutedGroupedPlan::Gated(plan) => {
                    validate_catalog_parameter_topology::<GatedProductOperation>(
                        &text,
                        plan,
                        &bank.catalog,
                        source,
                    )?;
                    operations.insert(eredu_runtime::GroupedOperationRequirement::GatedProduct);
                }
                RoutedGroupedPlan::Relu2(plan) => {
                    validate_catalog_parameter_topology::<Relu2Operation>(
                        &text,
                        plan,
                        &bank.catalog,
                        source,
                    )?;
                    operations.insert(eredu_runtime::GroupedOperationRequirement::Relu2);
                }
                RoutedGroupedPlan::Linear(plan) => {
                    validate_catalog_parameter_topology::<LinearOperation>(
                        &text,
                        plan,
                        &bank.catalog,
                        source,
                    )?;
                    operations.insert(eredu_runtime::GroupedOperationRequirement::Linear);
                }
            }
            if admitted.insert(id, bank).is_some() {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "duplicate routed bank {id:?}"
                )));
            }
        }
        if admitted.is_empty() {
            return Err(RoutedTextRequirementsError::Invalid(
                "routed architecture has no banks".into(),
            ));
        }
        Ok(Self {
            text: text.with_grouped_operations(operations),
            banks: admitted,
        })
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for PlannedResidentBank
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear { .. } => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
}

impl<B, Bank, Movement> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PlannedAddressableBank<B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear(_) => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear(_) => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        match self {
            Self::Gated(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(p, resident, request, partitions, context),
            Self::Relu2(p) => eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(p, resident, request, partitions, context),
            Self::Linear(_) => Err(RoutedTextExecutionError::Contract("selected linear bank received a feed-forward partial invocation".into())),
        }
    }
}
