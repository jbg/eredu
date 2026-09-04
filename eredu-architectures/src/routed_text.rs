//! Architecture-owned routed execution over generic grouped-bank mechanisms.

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use eredu_nn::{
    GroupSelection, GroupedGatedProductOperator, GroupedNeuralBackend, GroupedRelu2Operator, Tensor,
};
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, ParameterBankAcquisition, ParameterBankKey,
    RoutedExpertProvider, RoutedExpertRequest,
};

use crate::{ExpertRealizationPlan, ExpertResidencyCatalog};

/// Architecture-owned grouped equation and exact per-unit realization plan.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutedGroupedPlan {
    /// Gated-product grouped banks.
    Gated(ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>),
    /// ReLU-squared grouped banks.
    Relu2(ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>),
}

impl RoutedGroupedPlan {
    /// Returns the architecture-global routed member count.
    pub fn global_group_count(&self) -> usize {
        match self {
            Self::Gated(plan) => plan.global_expert_count(),
            Self::Relu2(plan) => plan.global_expert_count(),
        }
    }

    /// Returns the selected expert-axis width.
    pub fn expert_parallel_size(&self) -> usize {
        match self {
            Self::Gated(plan) => plan.expert_parallel_size(),
            Self::Relu2(plan) => plan.expert_parallel_size(),
        }
    }

    /// Returns this rank's coordinate in the selected expert axis.
    pub fn expert_parallel_rank(&self) -> usize {
        match self {
            Self::Gated(plan) => plan.expert_parallel_rank(),
            Self::Relu2(plan) => plan.expert_parallel_rank(),
        }
    }

    /// Returns the gated-product plan when that equation was selected.
    pub const fn gated(&self) -> Option<&ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> {
        match self {
            Self::Gated(plan) => Some(plan),
            Self::Relu2(_) => None,
        }
    }

    /// Returns the ReLU-squared plan when that equation was selected.
    pub const fn relu2(&self) -> Option<&ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> {
        match self {
            Self::Relu2(plan) => Some(plan),
            Self::Gated(_) => None,
        }
    }
}

impl From<ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> for RoutedGroupedPlan {
    fn from(plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>) -> Self {
        Self::Gated(plan)
    }
}

impl From<ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> for RoutedGroupedPlan {
    fn from(plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>) -> Self {
        Self::Relu2(plan)
    }
}

/// Converts one architecture-owned grouped specification into the opaque
/// routed plan retained by the neutral execution driver.
pub trait RoutedGroupedSpec: Clone {
    /// Erases the concrete grouped equation after architecture preparation.
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan;
}

impl RoutedGroupedSpec for eredu_nn::GroupedGatedProductSpec {
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan {
        RoutedGroupedPlan::Gated(plan)
    }
}

impl RoutedGroupedSpec for eredu_nn::GroupedRelu2Spec {
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan {
        RoutedGroupedPlan::Relu2(plan)
    }
}

/// Checked routed architecture and the selected neutral execution values that
/// must be paired with it.
pub struct PreparedRoutedTextArchitecture<A> {
    text: crate::replicated_text::PreparedReplicatedTextArchitecture<A>,
    bank_residency: eredu_runtime::ParameterBankResidency,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    routes_by_unit: BTreeMap<usize, usize>,
    addressable_members: Vec<eredu_runtime::AddressableBankMember>,
    addressable_quantization: Option<eredu_checkpoint::WeightQuantization>,
}

impl<A> PreparedRoutedTextArchitecture<A> {
    /// Returns the checked shared text architecture handoff.
    pub const fn text(&self) -> &crate::replicated_text::PreparedReplicatedTextArchitecture<A> {
        &self.text
    }

    /// Returns selected bank placement.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }

    /// Returns the canonical routed execution group.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the maximum selected route cardinality across grouped banks.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }

    /// Returns exact route cardinality keyed by architecture-global routed unit.
    ///
    /// Most families use one uniform cardinality. Composite families such as
    /// Inkling also execute an always-on shared bank under a distinct provider
    /// unit, so retaining this map is required to preserve its selected contract.
    pub const fn routes_by_unit(&self) -> &BTreeMap<usize, usize> {
        &self.routes_by_unit
    }

    /// Returns the exact architecture-global grouped plan.
    pub const fn plan(&self) -> &ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec> {
        &self.plan
    }

    /// Returns exact atomic addressable recipes and byte geometry.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }

    /// Returns generic source bindings and selected byte geometry for storage.
    pub fn addressable_members(&self) -> &[eredu_runtime::AddressableBankMember] {
        &self.addressable_members
    }

    /// Returns the selected uniform load-time bank transform, when present.
    pub const fn addressable_quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.addressable_quantization
    }

    /// Consumes the checked handoff into text modules and routed composition values.
    pub fn into_parts(
        self,
    ) -> (
        crate::replicated_text::PreparedReplicatedTextModules<A>,
        eredu_runtime::ParameterBankResidency,
        eredu_runtime::ExecutionGroupId,
        ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        ExpertResidencyCatalog,
    ) {
        (
            self.text.into_modules(),
            self.bank_residency,
            self.owner_group,
            self.plan,
            self.catalog,
        )
    }

    /// Constructs resident routed execution through the shared text session core.
    pub fn construct_resident_session<B, M>(
        self,
        mechanisms: M,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<PlannedResidentGatedProduct>,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_nn::GroupedNeuralBackend,
        M: eredu_runtime::ReplicatedTextSessionMechanisms<A, B>,
        A: eredu_runtime::LayeredArchitecture<B, M::State>
            + eredu_runtime::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let routes_by_unit = self.routes_by_unit.clone();
        let (mut modules, residency, owner_group, plan, catalog) = self.into_parts();
        if residency != eredu_runtime::ParameterBankResidency::WithLayer {
            return Err("selected routed bank residency is not with its execution unit".into());
        }
        let provider = PlannedResidentGatedProduct::new_with_routes(
            owner_group,
            plan,
            catalog,
            routes_by_unit,
        )
        .map_err(|error| error.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(provider),
            context,
        )
        .map_err(|error| error.to_string())
    }

    /// Constructs addressable routed execution through the shared text session core.
    pub fn construct_addressable_session<B, M, Bank, Movement>(
        self,
        mechanisms: M,
        bank: Bank,
        movement: Movement,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<
                PlannedAddressableGatedProduct<B, Bank, Movement>,
            >,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_nn::GroupedNeuralBackend,
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
        let routes_by_unit = self.routes_by_unit.clone();
        let selected_member_bytes = self
            .addressable_members
            .iter()
            .map(|member| (member.key(), member.selected_bytes()))
            .collect();
        let (mut modules, residency, owner_group, plan, catalog) = self.into_parts();
        let eredu_runtime::ParameterBankResidency::IndependentCache(options) = residency else {
            return Err("selected routed bank residency is not independently addressable".into());
        };
        let provider = PlannedAddressableGatedProduct::new_with_routes(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_by_unit,
        )
        .map_err(|error| error.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(provider),
            context,
        )
        .map_err(|error| error.to_string())
    }
}

/// Checked ReLU-squared routed architecture paired with its neutral plan.
pub struct PreparedRelu2RoutedTextArchitecture<A> {
    text: crate::replicated_text::PreparedReplicatedTextArchitecture<A>,
    bank_residency: eredu_runtime::ParameterBankResidency,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    addressable_members: Vec<eredu_runtime::AddressableBankMember>,
    addressable_quantization: Option<eredu_checkpoint::WeightQuantization>,
}

impl<A> PreparedRelu2RoutedTextArchitecture<A> {
    /// Returns the checked shared text architecture handoff.
    pub const fn text(&self) -> &crate::replicated_text::PreparedReplicatedTextArchitecture<A> {
        &self.text
    }

    /// Returns selected bank placement.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }

    /// Returns the canonical routed execution group.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the exact architecture-global grouped plan.
    pub const fn plan(&self) -> &ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec> {
        &self.plan
    }

    /// Returns exact atomic addressable recipes and byte geometry.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }

    /// Returns generic source bindings and selected byte geometry for storage.
    pub fn addressable_members(&self) -> &[eredu_runtime::AddressableBankMember] {
        &self.addressable_members
    }

    /// Returns the selected uniform load-time bank transform, when present.
    pub const fn addressable_quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.addressable_quantization
    }

    fn into_parts(
        self,
    ) -> (
        crate::replicated_text::PreparedReplicatedTextModules<A>,
        eredu_runtime::ParameterBankResidency,
        eredu_runtime::ExecutionGroupId,
        ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
        ExpertResidencyCatalog,
    ) {
        (
            self.text.into_modules(),
            self.bank_residency,
            self.owner_group,
            self.plan,
            self.catalog,
        )
    }

    /// Constructs resident ReLU-squared execution through the shared text session core.
    pub fn construct_resident_session<B, M>(
        self,
        mechanisms: M,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<PlannedResidentRelu2>,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_nn::GroupedNeuralBackend,
        M: eredu_runtime::ReplicatedTextSessionMechanisms<A, B>,
        A: eredu_runtime::LayeredArchitecture<B, M::State>
            + eredu_runtime::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let routes_per_token = self.routes_per_token;
        let (mut modules, residency, owner_group, plan, catalog) = self.into_parts();
        if residency != eredu_runtime::ParameterBankResidency::WithLayer {
            return Err("selected routed bank residency is not with its execution unit".into());
        }
        let provider = PlannedResidentRelu2::new(owner_group, plan, catalog, routes_per_token)
            .map_err(|error| error.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(provider),
            context,
        )
        .map_err(|error| error.to_string())
    }

    /// Constructs addressable ReLU-squared execution through the shared text session core.
    pub fn construct_addressable_session<B, M, Bank, Movement>(
        self,
        mechanisms: M,
        bank: Bank,
        movement: Movement,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        eredu_runtime::ReplicatedTextSession<
            A,
            B,
            M,
            eredu_runtime::RoutedReplicatedTextExecution<
                PlannedAddressableRelu2<B, Bank, Movement>,
            >,
        >,
        String,
    >
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_nn::GroupedNeuralBackend,
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
        let routes_per_token = self.routes_per_token;
        let selected_member_bytes = self
            .addressable_members
            .iter()
            .map(|member| (member.key(), member.selected_bytes()))
            .collect();
        let (mut modules, residency, owner_group, plan, catalog) = self.into_parts();
        let eredu_runtime::ParameterBankResidency::IndependentCache(options) = residency else {
            return Err("selected routed bank residency is not independently addressable".into());
        };
        let provider = PlannedAddressableRelu2::new(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_per_token,
        )
        .map_err(|error| error.to_string())?;
        eredu_runtime::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            eredu_runtime::RoutedReplicatedTextExecution::new(provider),
            context,
        )
        .map_err(|error| error.to_string())
    }
}

/// Failure while pairing one selected routed realization with concrete modules.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextPreparationError {
    /// The admitted graph no longer matches the selected routed class.
    #[error("replicated routed text preparation is ineligible")]
    Ineligible,
    /// Requirements, artifact provenance, selected formats, or modules disagreed.
    #[error("invalid replicated routed text preparation: {0}")]
    Invalid(String),
}

fn transformed_member_parameter_bytes(
    metadata: &eredu_checkpoint::recipe::RecipeMetadata,
    quantization: eredu_checkpoint::WeightQuantization,
) -> Result<u64, String> {
    let shape = metadata.shape();
    if shape.len() < 2 {
        return Err("addressable transform target is not a matrix".into());
    }
    let columns = *shape.last().expect("matrix shape is non-empty");
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| "addressable transform row geometry overflowed".to_owned())?;
    let group = usize::try_from(quantization.group_size())
        .map_err(|_| "addressable transform group size is negative".to_owned())?;
    let bits = usize::try_from(quantization.bits())
        .map_err(|_| "addressable transform bit width is negative".to_owned())?;
    if group == 0 || !columns.is_multiple_of(group) || !columns.is_multiple_of(32) {
        return Err(format!(
            "addressable transform width {columns} is incompatible with group {group}"
        ));
    }
    let packed = columns
        .checked_mul(bits)
        .and_then(|value| value.checked_div(8))
        .ok_or_else(|| "addressable packed byte geometry overflowed".to_owned())?;
    let affine_scalar_bytes = usize::try_from(
        metadata
            .dtype()
            .bit_width()
            .map_err(|error| error.to_string())?
            / 8,
    )
    .map_err(|_| "addressable affine scalar width is not representable".to_owned())?;
    let companion_width = if matches!(quantization, eredu_checkpoint::WeightQuantization::MxFp4) {
        1
    } else {
        affine_scalar_bytes
    };
    let companions = columns
        .checked_div(group)
        .and_then(|count| count.checked_mul(companion_width))
        .and_then(|scales| {
            let biases = if quantization.has_biases() { scales } else { 0 };
            biases.checked_add(scales)
        })
        .ok_or_else(|| "addressable companion byte geometry overflowed".to_owned())?;
    u64::try_from(
        rows.checked_mul(
            packed
                .checked_add(companions)
                .ok_or_else(|| "addressable selected row byte geometry overflowed".to_owned())?,
        )
        .ok_or_else(|| "addressable selected byte geometry overflowed".to_owned())?,
    )
    .map_err(|_| "addressable selected byte geometry is not representable".to_owned())
}

fn selected_member_geometry(
    unit: &crate::ExpertResidencyUnit,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<(u64, u64, Option<eredu_checkpoint::WeightQuantization>), String> {
    let mut selected_quantization = None;
    let mut source_bytes = 0u64;
    let mut selected_bytes = 0u64;
    for parameter in unit.parameters() {
        let metadata = parameter.metadata().ok_or_else(|| {
            format!(
                "addressable member {:?} omitted admitted recipe metadata",
                unit.identity()
            )
        })?;
        source_bytes = source_bytes
            .checked_add(metadata.byte_len())
            .ok_or_else(|| "addressable source byte geometry overflowed".to_owned())?;
        let realization = selected
            .parameters()
            .iter()
            .find(|candidate| candidate.name() == parameter.logical_target())
            .ok_or_else(|| {
                format!(
                    "addressable target {:?} has no selected parameter realization",
                    parameter.logical_target()
                )
            })?;
        let parameter_bytes = if matches!(
            realization.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        ) {
            if !matches!(
                parameter.role(),
                crate::ExpertParameterRole::QuantizableProjection { .. }
            ) {
                return Err(format!(
                    "addressable transform target {:?} is not a quantizable projection",
                    parameter.logical_target()
                ));
            }
            let quantization = realization
                .executable()
                .weight_quantization()
                .ok_or_else(|| {
                    format!(
                        "addressable transform target {:?} has no packed realization",
                        parameter.logical_target()
                    )
                })?;
            if matches!(
                quantization,
                eredu_checkpoint::WeightQuantization::GgufIQuant { .. }
            ) {
                return Err(
                    "load-time addressable transformation cannot select GGUF IQuant".into(),
                );
            }
            if selected_quantization
                .replace(quantization)
                .is_some_and(|prior| prior != quantization)
            {
                return Err("addressable bank selected heterogeneous load-time transforms".into());
            }
            transformed_member_parameter_bytes(metadata, quantization)?
        } else {
            metadata.byte_len()
        };
        selected_bytes = selected_bytes
            .checked_add(parameter_bytes)
            .ok_or_else(|| "addressable selected byte geometry overflowed".to_owned())?;
    }
    if unit.byte_len() != Some(source_bytes) {
        return Err(format!(
            "addressable member {:?} source bytes differ from its admitted catalog",
            unit.identity()
        ));
    }
    Ok((source_bytes, selected_bytes, selected_quantization))
}

pub(crate) fn project_addressable_members(
    catalog: &ExpertResidencyCatalog,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<
    (
        Vec<eredu_runtime::AddressableBankMember>,
        Option<eredu_checkpoint::WeightQuantization>,
    ),
    RoutedTextPreparationError,
> {
    let mut selected_quantization = None;
    let mut members = Vec::with_capacity(catalog.units().len());
    for unit in catalog.units() {
        let (_source_bytes, selected_bytes, member_quantization) =
            selected_member_geometry(unit, selected)
                .map_err(RoutedTextPreparationError::Invalid)?;
        if let Some(quantization) = member_quantization {
            if selected_quantization
                .replace(quantization)
                .is_some_and(|prior| prior != quantization)
            {
                return Err(RoutedTextPreparationError::Invalid(
                    "addressable bank selected heterogeneous load-time transforms".into(),
                ));
            }
        }
        let mut bindings = Vec::with_capacity(unit.parameters().len());
        for parameter in unit.parameters() {
            let metadata = parameter.metadata().ok_or_else(|| {
                RoutedTextPreparationError::Invalid(format!(
                    "addressable member {:?} omitted admitted recipe metadata",
                    unit.identity()
                ))
            })?;
            let mut binding = eredu_runtime::WeightBinding::from_recipe(
                parameter.binding_name(),
                parameter.recipe().clone(),
                metadata.byte_len(),
            )
            .and_then(|binding| binding.with_logical_target(parameter.logical_target()))
            .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
            if let crate::ExpertParameterRole::QuantizableProjection {
                scales_binding,
                biases_binding,
            } = parameter.role()
            {
                binding = binding
                    .with_quantization_companions(scales_binding, biases_binding)
                    .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
            }
            bindings.push(binding);
        }
        members.push(
            eredu_runtime::AddressableBankMember::new(unit.identity(), bindings, selected_bytes)
                .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?,
        );
    }
    Ok((members, selected_quantization))
}

fn validate_selected_routed_handoff(
    expected: &RoutedTextRequirements,
    selected: &SelectedRoutedTextRealization,
) -> Result<(), RoutedTextPreparationError> {
    let expected_plan = select_grouped_formats(expected.plan.clone(), selected.text())
        .map_err(RoutedTextPreparationError::Invalid)?;
    if selected.text().requirements() != expected.text()
        || selected.owner_group() != expected.owner_group()
        || selected.plan() != &expected_plan
        || selected.catalog() != expected.catalog()
        || selected.routes_per_token() != expected.routes_per_token()
        || selected.routes_by_unit != expected.routes_by_unit
    {
        return Err(RoutedTextPreparationError::Invalid(
            "selected realization differs from admitted routed requirements".into(),
        ));
    }
    Ok(())
}

fn addressable_parameter_targets<O>(
    residency: eredu_runtime::ParameterBankResidency,
    _plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
) -> std::collections::BTreeSet<String>
where
    O: RoutedGroupedOperationValidation,
{
    if matches!(
        residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        catalog
            .logical_targets()
            .into_iter()
            .map(str::to_owned)
            .collect()
    } else {
        std::collections::BTreeSet::new()
    }
}

fn prepare_qwen_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::qwen::RoutedLayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe() => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe() => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::qwen::RoutedLayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_qwen_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::qwen::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::qwen::RoutedLayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::qwen(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_gpt_oss_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::gpt_oss::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::GptOss(args) => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::GptOss(args) => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::gpt_oss::new_layered_model::<B>(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_gpt_oss_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::gpt_oss::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::gpt_oss::new_layered_model::<B>(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::gpt_oss(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_lfm2_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::lfm2::LayeredModel<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Lfm2(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::lfm2::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_lfm2_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::lfm2::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::lfm2::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::lfm2(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_kimi_linear_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::kimi_linear::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::kimi_linear::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_kimi_linear_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::kimi_linear::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::kimi_linear::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::kimi_linear(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_qwen_hybrid_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::qwen::hybrid::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                &args.text
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                &args.text
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::qwen::hybrid::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_qwen_hybrid_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::qwen::hybrid::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::qwen::hybrid::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::qwen_hybrid_text(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_deepseek_v3_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::deepseek::v3::Model<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::deepseek::v3::Model::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_deepseek_v3_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::deepseek::v3_architecture_fingerprint(&selected_args);
    let architecture = crate::deepseek::v3::Model::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::deepseek_v3(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_deepseek_v4_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::deepseek::v4::Model<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::HyperNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    let routes_by_unit = expected.routes_by_unit.clone();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::deepseek::v4::Model::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_deepseek_v4_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::deepseek::v4_architecture_fingerprint(&selected_args);
    let architecture = crate::deepseek::v4::Model::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::deepseek_v4(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

fn prepare_nemotron_h_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRelu2RoutedTextArchitecture<crate::nemotron_h::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    let routes_per_token = selected.routes_per_token();
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text)?;
    let RoutedGroupedPlan::Relu2(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::nemotron_h::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_nemotron_h_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::nemotron_h::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::nemotron_h::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::nemotron_h(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets = addressable_parameter_targets::<Relu2Operation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            expected.text,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRelu2RoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        addressable_members,
        addressable_quantization,
    })
}

/// Receives one statically paired ReLU-squared routed architecture.
pub trait Relu2RoutedTextArchitectureVisitor<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Called immediately before architecture construction begins.
    fn construction_started(&mut self) {}

    /// Binds one architecture-owned prepared handoff to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedRelu2RoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Constructs an admitted ReLU-squared routed family and invokes a generic visitor.
pub fn visit_relu2_routed_text_architecture<B, S, V>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: Relu2RoutedTextArchitectureVisitor<B, S>,
{
    visitor.construction_started();
    let prepared = prepare_nemotron_h_routed_text_architecture::<B, S>(
        inspection,
        selected,
        store.clone(),
        context,
    )
    .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, store)
        .map_err(RoutedTextDispatchError::Backend)
}

/// Receives one statically paired gated routed architecture.
pub trait GatedRoutedTextArchitectureVisitor<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Called immediately before architecture construction begins.
    fn construction_started(&mut self) {}

    /// Binds one architecture-owned prepared handoff to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Constructs an admitted pooling-attention routed family and invokes generic mechanisms.
pub fn visit_pooling_routed_text_architecture<B, S, V>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::HyperNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: GatedRoutedTextArchitectureVisitor<B, S>,
{
    visitor.construction_started();
    let prepared = prepare_deepseek_v4_routed_text_architecture::<B, S>(
        inspection,
        selected,
        store.clone(),
        context,
    )
    .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, store)
        .map_err(RoutedTextDispatchError::Backend)
}

/// Constructs the exact admitted gated routed family and invokes a generic mechanism visitor.
pub fn visit_gated_routed_text_architecture<B, S, V>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::CompressedAttentionCache<B::Tensor>,
    V: GatedRoutedTextArchitectureVisitor<B, S>,
{
    visitor.construction_started();
    match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe()
            ) =>
        {
            prepare_qwen_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe()
            ) =>
        {
            prepare_qwen_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::GptOss(_)
            ) =>
        {
            prepare_gpt_oss_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::GptOss(_)
            ) =>
        {
            prepare_gpt_oss_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::Lfm2(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_lfm2_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::Lfm2(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_lfm2_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_kimi_linear_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::KimiLinear(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_kimi_linear_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                    if args.vision.is_none()
                        && args.text.mtp_num_hidden_layers == 0
                        && args.text.is_moe()
            ) =>
        {
            prepare_qwen_hybrid_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::QwenHybrid(args)
                    if args.vision.is_none()
                        && args.text.mtp_num_hidden_layers == 0
                        && args.text.is_moe()
            ) =>
        {
            prepare_qwen_hybrid_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                    if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers()
            ) =>
        {
            prepare_deepseek_v3_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::DeepSeekV3(args)
                    if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers()
            ) =>
        {
            prepare_deepseek_v3_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        _ => Err(RoutedTextDispatchError::Architecture(
            RoutedTextPreparationError::Ineligible.to_string(),
        )),
    }
}

/// Failure while architecture-owned routed dispatch invokes a mechanism visitor.
#[derive(Debug, thiserror::Error)]
pub enum RoutedTextDispatchError<E> {
    /// Architecture admission and selected realization disagreed.
    #[error("{0}")]
    Architecture(String),
    /// Backend mechanism binding failed.
    #[error("routed text mechanism binding failed: {0}")]
    Backend(E),
}

/// Exact architecture and artifact requirements for replicated routed text.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedTextRequirements {
    text: eredu_runtime::ReplicatedTextRequirements,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: RoutedGroupedPlan,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    routes_by_unit: BTreeMap<usize, usize>,
}

impl RoutedTextRequirements {
    /// Returns the shared replicated text requirements.
    pub const fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        &self.text
    }

    /// Returns the canonical execution group that owns routed units.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the architecture-global routed identity and grouped geometry plan.
    pub const fn plan(&self) -> &RoutedGroupedPlan {
        &self.plan
    }

    /// Returns exact independently addressable checkpoint recipes.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }

    /// Returns the maximum distinct routed bank members selected for one token row.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }

    fn routes_for_unit(&self, unit: usize) -> Option<usize> {
        self.routes_by_unit.get(&unit).copied()
    }
}

fn uniform_routes_by_unit<S>(
    plan: &ExpertRealizationPlan<S>,
    routes_per_token: usize,
) -> BTreeMap<usize, usize> {
    plan.unit_specs()
        .keys()
        .map(|(_, unit)| (*unit, routes_per_token))
        .collect()
}

fn validate_routes_by_unit<O>(
    plan: &ExpertRealizationPlan<O::Spec>,
    routes_by_unit: &BTreeMap<usize, usize>,
) -> Result<(), RoutedTextRequirementsError>
where
    O: RoutedGroupedOperationValidation,
{
    let planned = plan
        .unit_specs()
        .keys()
        .map(|(_, unit)| *unit)
        .collect::<std::collections::BTreeSet<_>>();
    let routed = routes_by_unit
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if planned != routed {
        return Err(RoutedTextRequirementsError::Invalid(format!(
            "route-cardinality units {routed:?} differ from grouped-bank units {planned:?}"
        )));
    }
    for ((_, unit), spec) in plan.unit_specs() {
        let routes = routes_by_unit[unit];
        let groups = usize::try_from(O::group_count(spec)).unwrap_or_default();
        if routes == 0 || routes > groups {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "grouped-bank unit {unit} selects {routes} routes from {groups} groups"
            )));
        }
    }
    Ok(())
}

pub(crate) fn gated_routed_text_requirements(
    text: eredu_runtime::ReplicatedTextRequirements,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
    gated_routed_text_requirements_with_routes(
        text,
        owner_group,
        plan,
        catalog,
        routes_by_unit,
        recipe_source,
    )
}

pub(crate) fn gated_routed_text_requirements_with_routes(
    text: eredu_runtime::ReplicatedTextRequirements,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    catalog: ExpertResidencyCatalog,
    routes_by_unit: BTreeMap<usize, usize>,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let routes_per_token = routes_by_unit.values().copied().max().unwrap_or(0);
    if routes_per_token == 0 {
        return Err(RoutedTextRequirementsError::Invalid(
            "routed architecture has no positive routes-per-token cardinality".into(),
        ));
    }
    validate_routes_by_unit::<GatedProductOperation>(&plan, &routes_by_unit)?;
    let text =
        text.with_grouped_operations([eredu_runtime::GroupedOperationRequirement::GatedProduct]);
    validate_plan_catalog::<GatedProductOperation>(&owner_group, &plan, &catalog)?;
    validate_catalog_parameter_topology::<GatedProductOperation>(
        &text,
        &plan,
        &catalog,
        recipe_source,
    )?;
    Ok(RoutedTextRequirements {
        text,
        owner_group,
        plan: RoutedGroupedPlan::Gated(plan),
        catalog,
        routes_per_token,
        routes_by_unit,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_gated_routed_architecture_handoff<B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    expected: RoutedTextRequirements,
    selected: SelectedRoutedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<A>, String>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>,
    A::StaticModules: Clone,
{
    validate_selected_routed_handoff(&expected, &selected).map_err(|error| error.to_string())?;
    let (text, bank_residency, owner_group, plan, catalog) = selected.into_parts();
    let (addressable_members, addressable_quantization) =
        project_addressable_members(&catalog, &text).map_err(|error| error.to_string())?;
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err("selected grouped equation differs from the composite architecture".into());
    };
    let routes_per_token = expected.routes_per_token;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared = crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
        architecture,
        source_architecture,
        expected.text,
        text,
        capability_estimate,
        effective_model_type,
        prompt_cache_architecture_identity,
        targets.iter().map(String::as_str),
        context,
    )?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        routes_by_unit: expected.routes_by_unit,
        addressable_members,
        addressable_quantization,
    })
}

/// Caller policy for one replicated routed text session.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedTextSelectionRequest {
    text: eredu_runtime::ReplicatedTextSelectionRequest,
    weights: eredu_runtime::WeightResidency,
}

impl RoutedTextSelectionRequest {
    /// Pairs shared text-session policy with ordinary and banked weight placement.
    pub fn new(
        text: eredu_runtime::ReplicatedTextSelectionRequest,
        weights: eredu_runtime::WeightResidency,
    ) -> Result<Self, RoutedTextSelectionError> {
        if text.residency() != weights.layers() {
            return Err(RoutedTextSelectionError {
                issues: vec![
                    "ordinary weight residency differs between text and banked policy".into(),
                ],
            });
        }
        if let Some(options) = weights.parameter_bank_cache() {
            options
                .validate()
                .map_err(|error| RoutedTextSelectionError {
                    issues: vec![error.to_string()],
                })?;
        }
        Ok(Self { text, weights })
    }

    /// Returns shared state, transform, session, topology, and completion policy.
    pub const fn text(&self) -> &eredu_runtime::ReplicatedTextSelectionRequest {
        &self.text
    }

    /// Returns ordinary and independently addressable weight placement.
    pub const fn weights(&self) -> eredu_runtime::WeightResidency {
        self.weights
    }
}

/// Authoritative routed realization selected before architecture construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedRoutedTextRealization {
    text: eredu_runtime::SelectedReplicatedTextRealization,
    bank_residency: eredu_runtime::ParameterBankResidency,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: RoutedGroupedPlan,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    routes_by_unit: BTreeMap<usize, usize>,
}

impl SelectedRoutedTextRealization {
    /// Returns the selected shared text-session realization.
    pub const fn text(&self) -> &eredu_runtime::SelectedReplicatedTextRealization {
        &self.text
    }

    /// Returns with-unit or independently cached bank placement.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }

    /// Returns the canonical routed execution group.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the selected architecture-global identity and grouped geometry plan.
    pub const fn plan(&self) -> &RoutedGroupedPlan {
        &self.plan
    }

    /// Returns the selected exact bank recipe catalog.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }

    /// Returns the exact selected expert cardinality for every routed token row.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }

    /// Returns exact route cardinality keyed by architecture-global routed unit.
    pub const fn routes_by_unit(&self) -> &BTreeMap<usize, usize> {
        &self.routes_by_unit
    }

    /// Consumes the realization into its shared text contract and routed values.
    pub fn into_parts(
        self,
    ) -> (
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ParameterBankResidency,
        eredu_runtime::ExecutionGroupId,
        RoutedGroupedPlan,
        ExpertResidencyCatalog,
    ) {
        (
            self.text,
            self.bank_residency,
            self.owner_group,
            self.plan,
            self.catalog,
        )
    }
}

/// Complete fail-closed routed selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("replicated routed text realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct RoutedTextSelectionError {
    issues: Vec<String>,
}

impl RoutedTextSelectionError {
    /// Returns every missing mechanism in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

fn maximum_selected_compact_bytes(
    requirements: &RoutedTextRequirements,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<u64, String> {
    let mut by_unit = BTreeMap::<(String, usize), Vec<u64>>::new();
    for unit in requirements.catalog().units() {
        let (_, selected_bytes, _) = selected_member_geometry(unit, selected)?;
        by_unit
            .entry((
                unit.owner_group().as_str().to_owned(),
                unit.identity().unit(),
            ))
            .or_default()
            .push(selected_bytes);
    }
    let mut maximum = 0u64;
    for ((group, unit), mut members) in by_unit {
        let routes = requirements.routes_for_unit(unit).ok_or_else(|| {
            format!("routed unit {group:?}/{unit} has no admitted route cardinality")
        })?;
        if members.len() < routes {
            return Err(format!(
                "routed unit {group:?}/{unit} has {} bank members for {} routes per token",
                members.len(),
                routes
            ));
        }
        members.sort_unstable_by(|left, right| right.cmp(left));
        let bytes = members
            .into_iter()
            .take(routes)
            .try_fold(0u64, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| "per-row compact-bank byte geometry overflowed".to_owned())?;
        maximum = maximum.max(bytes);
    }
    if maximum == 0 {
        return Err("routed requirements contain no compact-bank geometry".into());
    }
    Ok(maximum)
}

/// Selects one routed realization without constructing modules or opening payloads.
pub fn select_routed_text_realization(
    requirements: &RoutedTextRequirements,
    request: &RoutedTextSelectionRequest,
    capabilities: &eredu_runtime::BackendMechanismCapabilities,
) -> Result<SelectedRoutedTextRealization, RoutedTextSelectionError> {
    let text = eredu_runtime::select_replicated_text_realization(
        requirements.text(),
        request.text(),
        capabilities,
    );
    let bank_residency = request.weights().parameter_banks();
    let mut issues = text
        .as_ref()
        .err()
        .map(|error| error.issues().to_vec())
        .unwrap_or_default();
    if let eredu_runtime::ParameterBankResidency::IndependentCache(options) = bank_residency {
        if !capabilities.indexed_movement() {
            issues.push("indexed selection and movement".into());
        }
        match capabilities.addressable_storage() {
            Some(storage) => {
                if !storage.bulk_access() {
                    issues.push("addressable bulk access".into());
                }
                if !storage.incremental_access() {
                    issues.push("addressable incremental access".into());
                }
                if !storage.lease_completion() {
                    issues.push("addressable lease completion".into());
                }
                if !storage.tiers().disk() {
                    issues.push("addressable disk storage".into());
                }
                if options.offload().host_budget_bytes() != Some(0) && !storage.tiers().host() {
                    issues.push("addressable host storage".into());
                }
                if !storage.tiers().device() {
                    issues.push("addressable device storage".into());
                }
                if options.compact_bank_scratch_bytes() > storage.maximum_compact_bytes() {
                    issues.push(format!(
                        "compact bank limit {} exceeds backend maximum {}",
                        options.compact_bank_scratch_bytes(),
                        storage.maximum_compact_bytes()
                    ));
                }
            }
            None => issues.push("independently addressable storage".into()),
        }
        if let Ok(selected_text) = &text {
            match maximum_selected_compact_bytes(requirements, selected_text) {
                Ok(bytes) if bytes > options.compact_bank_scratch_bytes() => issues.push(format!(
                    "one routed token row requires {bytes} selected compact-bank bytes for {} routes, exceeding limit {}",
                    requirements.routes_per_token(),
                    options.compact_bank_scratch_bytes()
                )),
                Ok(_) => {}
                Err(issue) => issues.push(issue),
            }
        }
    }
    if !issues.is_empty() {
        return Err(RoutedTextSelectionError { issues });
    }
    let text = text.expect("an empty diagnostic implies successful text selection");
    let plan = select_grouped_formats(requirements.plan.clone(), &text).map_err(|issue| {
        RoutedTextSelectionError {
            issues: vec![issue],
        }
    })?;
    Ok(SelectedRoutedTextRealization {
        text,
        bank_residency,
        owner_group: requirements.owner_group.clone(),
        plan,
        catalog: requirements.catalog.clone(),
        routes_per_token: requirements.routes_per_token,
        routes_by_unit: requirements.routes_by_unit.clone(),
    })
}

fn select_grouped_formats(
    plan: RoutedGroupedPlan,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<RoutedGroupedPlan, String> {
    match plan {
        RoutedGroupedPlan::Gated(plan) => plan
            .try_map_unit_specs(|spec| select_gated_formats(spec, text))
            .map(RoutedGroupedPlan::Gated),
        RoutedGroupedPlan::Relu2(plan) => plan
            .try_map_unit_specs(|spec| select_relu2_formats(spec, text))
            .map(RoutedGroupedPlan::Relu2),
    }
}

fn select_projection_format(
    projection: &eredu_nn::GroupedProjectionSpec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedProjectionSpec, String> {
    let weight = projection.weight().id.to_string();
    let executable = text
        .parameters()
        .iter()
        .find(|parameter| parameter.name() == weight)
        .map(eredu_runtime::SelectedParameterRealization::executable)
        .ok_or_else(|| format!("grouped projection {weight:?} has no selected realization"))?;
    if executable == projection.format().encoding() {
        return Ok(projection.clone());
    }
    eredu_nn::GroupedProjectionSpec::new(
        projection.weight().clone(),
        projection.bias().cloned(),
        crate::linear_format::standard_expert_format(&weight, executable)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn select_gated_formats(
    spec: eredu_nn::GroupedGatedProductSpec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedGatedProductSpec, String> {
    let layout = match spec.layout() {
        eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
            eredu_nn::GatedProductGroupLayout::Packed {
                gate_up: select_projection_format(gate_up, text)?,
                down: select_projection_format(down, text)?,
            }
        }
        eredu_nn::GatedProductGroupLayout::Independent(groups) => {
            eredu_nn::GatedProductGroupLayout::Independent(
                groups
                    .iter()
                    .map(|group| {
                        Ok(eredu_nn::GatedProductGroupParameters::new(
                            select_projection_format(group.gate(), text)?,
                            select_projection_format(group.up(), text)?,
                            select_projection_format(group.down(), text)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            )
        }
        _ => return Err("unsupported grouped gated-product layout".into()),
    };
    eredu_nn::GroupedGatedProductSpec::new(
        spec.group_count(),
        spec.input_dimensions(),
        spec.intermediate_dimensions(),
        spec.output_dimensions(),
        spec.policy(),
        layout,
    )
    .map_err(|error| error.to_string())
}

fn select_relu2_formats(
    spec: eredu_nn::GroupedRelu2Spec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedRelu2Spec, String> {
    eredu_nn::GroupedRelu2Spec::new(
        spec.group_count(),
        spec.hidden_dimensions(),
        spec.intermediate_dimensions(),
        select_projection_format(spec.up(), text)?,
        select_projection_format(spec.down(), text)?,
    )
    .map_err(|error| error.to_string())
}

/// Failure while deriving routed requirements from an admitted artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextRequirementsError {
    /// The admitted graph is outside replicated routed text execution.
    #[error("architecture is not an eligible replicated routed text graph")]
    Ineligible,
    /// Artifact provenance, architecture geometry, or routed topology is invalid.
    #[error("invalid replicated routed text requirements: {0}")]
    Invalid(String),
}

/// Derives routed requirements entirely from admitted architecture facts.
///
/// Backend support and caller residency policy do not participate in this step.
pub fn routed_text_requirements(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let architecture = inspection.architecture_plan();
    if architecture.has_processor() || architecture.gguf_media_projector().is_some() {
        return Err(RoutedTextRequirementsError::Ineligible);
    }
    enum Family<'a> {
        Qwen(&'a crate::qwen::ModelArgs),
        GptOss(&'a crate::gpt_oss::ModelArgs),
        NemotronH(&'a crate::nemotron_h::ModelArgs),
        Lfm2(&'a crate::lfm2::ModelArgs),
        KimiLinear(&'a crate::kimi_linear::ModelArgs),
        QwenHybrid(&'a crate::qwen::hybrid::HybridConfig),
        DeepSeekV3(&'a crate::deepseek::V3Args),
        DeepSeekV4(&'a crate::deepseek::V4Args),
    }
    let family = match (
        architecture.safetensors_architecture(),
        architecture.gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe() => {
                Family::Qwen(args)
            }
            crate::configuration::SafetensorsModelConfig::GptOss(args) => Family::GptOss(args),
            crate::configuration::SafetensorsModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::NemotronH(args)
            }
            crate::configuration::SafetensorsModelConfig::Lfm2(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::Lfm2(args)
            }
            crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::KimiLinear(args)
            }
            crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                Family::QwenHybrid(&args.text)
            }
            crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::DeepSeekV3(args)
            }
            crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                Family::DeepSeekV4(args)
            }
            _ => return Err(RoutedTextRequirementsError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe() => {
                Family::Qwen(args)
            }
            crate::configuration::GgufModelConfig::GptOss(args) => Family::GptOss(args),
            crate::configuration::GgufModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::NemotronH(args)
            }
            crate::configuration::GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
                Family::Lfm2(args)
            }
            crate::configuration::GgufModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::KimiLinear(args)
            }
            crate::configuration::GgufModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                Family::QwenHybrid(&args.text)
            }
            crate::configuration::GgufModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::DeepSeekV3(args)
            }
            crate::configuration::GgufModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                Family::DeepSeekV4(args)
            }
            _ => return Err(RoutedTextRequirementsError::Ineligible),
        },
        _ => return Err(RoutedTextRequirementsError::Ineligible),
    };
    let routes_per_token = usize::try_from(match &family {
        Family::Qwen(args) => args.num_experts_per_tok,
        Family::GptOss(args) => args.num_experts_per_tok,
        Family::NemotronH(args) => args.num_experts_per_tok,
        Family::Lfm2(args) => args.num_experts_per_tok,
        Family::KimiLinear(args) => args.num_experts_per_token,
        Family::QwenHybrid(args) => args.num_experts_per_tok,
        Family::DeepSeekV3(args) => args.num_experts_per_tok,
        Family::DeepSeekV4(args) => args.num_experts_per_tok,
    })
    .ok()
    .filter(|routes| *routes > 0)
    .ok_or_else(|| {
        RoutedTextRequirementsError::Invalid(
            "routed architecture has no positive routes-per-token cardinality".into(),
        )
    })?;
    let expected_routed_units = match &family {
        Family::Qwen(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| {
                (
                    ("text_decoder".to_owned(), unit),
                    format!("{}.layers.{unit}", args.parameter_root),
                )
            })
            .collect(),
        Family::GptOss(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| {
                (
                    ("text_decoder".to_owned(), unit),
                    format!("{}.layers.{unit}", args.parameter_root),
                )
            })
            .collect(),
        Family::NemotronH(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| **policy == crate::nemotron_h::LayerPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::Lfm2(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| policy.feed_forward == crate::lfm2::FeedForwardPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::KimiLinear(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| {
                policy.feed_forward == crate::kimi_linear::FeedForwardPolicy::SparseMoe
            })
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::QwenHybrid(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::DeepSeekV3(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| **policy == crate::deepseek::LayerPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::DeepSeekV4(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| (("target".to_owned(), unit), format!("layers.{unit}")))
            .collect(),
    };
    let recipe_source = crate::replicated_text::inspection_recipe_source(inspection)
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
    if let Family::NemotronH(args) = family {
        let text =
            crate::replicated_text::nemotron_h_replicated_text_requirements(inspection, args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
                .with_grouped_operations([eredu_runtime::GroupedOperationRequirement::Relu2]);
        let plan = crate::nemotron_h::replicated_expert_realization_plan(args)
            .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
        let catalog = crate::nemotron_h::expert_residency_catalog(recipe_source.as_ref(), args)
            .map_err(RoutedTextRequirementsError::Invalid)?;
        let owner_group = eredu_runtime::ExecutionGroupId::new("target")
            .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
        validate_expected_routed_schedule(&expected_routed_units, &plan, &catalog)?;
        validate_plan_catalog::<Relu2Operation>(&owner_group, &plan, &catalog)?;
        validate_catalog_parameter_topology::<Relu2Operation>(
            &text,
            &plan,
            &catalog,
            recipe_source.as_ref(),
        )?;
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        return Ok(RoutedTextRequirements {
            text,
            owner_group,
            plan: RoutedGroupedPlan::Relu2(plan),
            catalog,
            routes_per_token,
            routes_by_unit,
        });
    }
    let (text, plan, catalog, owner_group_name) = match family {
        Family::Qwen(args) => (
            crate::replicated_text::qwen_replicated_text_requirements(inspection, args),
            crate::qwen::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::qwen::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "text_decoder",
        ),
        Family::GptOss(args) => (
            crate::replicated_text::gpt_oss_replicated_text_requirements(inspection, args),
            crate::gpt_oss::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::gpt_oss::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "text_decoder",
        ),
        Family::Lfm2(args) => (
            crate::replicated_text::lfm2_replicated_text_requirements(inspection, args),
            crate::lfm2::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::lfm2::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::KimiLinear(args) => (
            crate::replicated_text::kimi_linear_replicated_text_requirements(inspection, args),
            crate::kimi_linear::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::kimi_linear::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::QwenHybrid(args) => (
            crate::replicated_text::qwen_hybrid_replicated_text_requirements(inspection, args),
            crate::qwen::hybrid::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::qwen::hybrid::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::DeepSeekV3(args) => (
            crate::replicated_text::deepseek_v3_replicated_text_requirements(inspection, args),
            crate::deepseek::v3_replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::deepseek::v3_expert_residency_catalog(recipe_source.as_ref(), args, None)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::DeepSeekV4(args) => (
            crate::replicated_text::deepseek_v4_replicated_text_requirements(inspection, args),
            crate::deepseek::v4_replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::deepseek::v4_expert_residency_catalog(recipe_source.as_ref(), args, None)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::NemotronH(_) => {
            unreachable!("Nemotron-H returned through ReLU-squared requirements")
        }
    };
    let text = text
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
        .with_grouped_operations([eredu_runtime::GroupedOperationRequirement::GatedProduct]);
    let plan = plan?;
    let catalog = catalog?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(owner_group_name)
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
    validate_expected_routed_schedule(&expected_routed_units, &plan, &catalog)?;
    validate_plan_catalog::<GatedProductOperation>(&owner_group, &plan, &catalog)?;
    validate_catalog_parameter_topology::<GatedProductOperation>(
        &text,
        &plan,
        &catalog,
        recipe_source.as_ref(),
    )?;
    let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
    Ok(RoutedTextRequirements {
        text,
        owner_group,
        plan: RoutedGroupedPlan::Gated(plan),
        catalog,
        routes_per_token,
        routes_by_unit,
    })
}

fn validate_expected_routed_schedule<S>(
    expected: &BTreeMap<(String, usize), String>,
    plan: &ExpertRealizationPlan<S>,
    catalog: &ExpertResidencyCatalog,
) -> Result<(), RoutedTextRequirementsError> {
    let actual = plan
        .unit_specs()
        .keys()
        .map(|(group, unit)| ((group.as_str().to_owned(), *unit), ()))
        .collect::<BTreeMap<_, _>>();
    let expected_addresses = expected
        .keys()
        .cloned()
        .map(|address| (address, ()))
        .collect::<BTreeMap<_, _>>();
    if actual != expected_addresses {
        return Err(RoutedTextRequirementsError::Invalid(format!(
            "routed plan addresses {:?} differ from architecture schedule {:?}",
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        )));
    }
    for ((group, unit), expected_path) in expected {
        let members = catalog
            .units()
            .iter()
            .filter(|member| member.owner_group().as_str() == group && member.owner_unit() == *unit)
            .collect::<Vec<_>>();
        if members.is_empty()
            || members
                .iter()
                .any(|member| member.unit_path() != expected_path)
        {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "routed catalog address {group:?}/{unit} does not use architecture path {expected_path:?}"
            )));
        }
    }
    Ok(())
}

fn validate_catalog_parameter_topology<O: RoutedGroupedOperationValidation>(
    text: &eredu_runtime::ReplicatedTextRequirements,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<(), RoutedTextRequirementsError> {
    for unit in catalog.units() {
        for parameter in unit.parameters() {
            let requirement = text
                .parameters()
                .iter()
                .find(|candidate| candidate.name() == parameter.logical_target())
                .ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} is absent from replicated parameter topology",
                        parameter.logical_target()
                    ))
                })?;
            let spec = plan
                .unit_spec(unit.owner_group().as_str(), unit.identity().unit())
                .ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} has no routed unit specification",
                        parameter.logical_target()
                    ))
                })?;
            let expected_shapes = O::member_parameter_shapes(spec, unit.identity().member())
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
            if let Some(expected_member_shape) = expected_shapes.get(parameter.logical_target()) {
                let occurrences = catalog
                    .units()
                    .iter()
                    .flat_map(|unit| unit.parameters())
                    .filter(|candidate| candidate.logical_target() == parameter.logical_target())
                    .count();
                let mut expected_logical = expected_member_shape.clone();
                if occurrences > 1 {
                    let first = expected_logical.first_mut().ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(
                            "grouped member shape has no leading expert axis".into(),
                        )
                    })?;
                    *first = occurrences;
                }
                if requirement.logical_shape() != expected_logical {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} logical shape {:?} differs from grouped shape {expected_logical:?}",
                        parameter.logical_target(),
                        requirement.logical_shape()
                    )));
                }
                let mut expected_physical = requirement
                    .physical_shape()
                    .ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(format!(
                            "bank target {:?} has no admitted physical geometry",
                            parameter.logical_target()
                        ))
                    })?
                    .to_vec();
                if occurrences > 1 {
                    let first = expected_physical.first_mut().ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(
                            "packed bank physical shape has no expert axis".into(),
                        )
                    })?;
                    if *first != occurrences {
                        return Err(RoutedTextRequirementsError::Invalid(format!(
                            "bank target {:?} physical expert axis {} differs from catalog cardinality {occurrences}",
                            parameter.logical_target(), *first
                        )));
                    }
                    *first = 1;
                }
                let metadata = parameter.metadata().ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} has no inferred recipe geometry",
                        parameter.logical_target()
                    ))
                })?;
                if metadata.shape() != expected_physical {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} recipe shape {:?} differs from admitted member shape {expected_physical:?}",
                        parameter.logical_target(), metadata.shape()
                    )));
                }
            }
            match requirement.owner() {
                eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
                    group,
                    unit: owner,
                } if group == unit.owner_group().as_str() && *owner == unit.owner_unit() => {}
                actual => {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} owner {actual:?} differs from {:?}/{}",
                        parameter.logical_target(),
                        unit.owner_group().as_str(),
                        unit.owner_unit()
                    )));
                }
            }
            if !matches!(
                requirement.role(),
                eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    | eredu_runtime::ReplicatedTextParameterRole::LinearBias
                    | eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
            ) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} has non-linear topology role {:?}",
                    parameter.logical_target(),
                    requirement.role()
                )));
            }
            let derived = text.derived_recipes().get(parameter.logical_target());
            let allowed = derived.map_or_else(
                || {
                    requirement
                        .sources()
                        .iter()
                        .chain(requirement.aliases())
                        .cloned()
                        .collect::<std::collections::BTreeSet<_>>()
                },
                |recipe| {
                    recipe
                        .source_keys()
                        .into_iter()
                        .map(str::to_owned)
                        .collect()
                },
            );
            let actual = parameter
                .recipe()
                .source_keys()
                .into_iter()
                .map(str::to_owned)
                .collect::<std::collections::BTreeSet<_>>();
            if actual.is_empty() || !actual.is_subset(&allowed) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} recipe sources {actual:?} differ from admitted sources {allowed:?}",
                    parameter.logical_target()
                )));
            }
            let member = unit.identity().member();
            let member_selection = eredu_checkpoint::store::TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let exact_match = if let Some(recipe) = derived {
                recipe
                    .select_bounded(recipe_source, member_selection.clone())
                    .map_err(|error| {
                        RoutedTextRequirementsError::Invalid(format!(
                            "bank target {:?} cannot select admitted member {member}: {error}",
                            parameter.logical_target()
                        ))
                    })?
                    == *parameter.recipe()
            } else {
                requirement
                    .sources()
                    .iter()
                    .chain(requirement.aliases())
                    .filter_map(|source| {
                        let complete = eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            source,
                            eredu_checkpoint::store::TensorSelection::Full,
                        );
                        if &complete == parameter.recipe() {
                            return Some(true);
                        }
                        complete
                            .select_bounded(recipe_source, member_selection.clone())
                            .ok()
                            .map(|selected| selected == *parameter.recipe())
                    })
                    .any(|matches| matches)
            };
            if !exact_match {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} member {member} recipe differs from the exact admitted member recipe",
                    parameter.logical_target()
                )));
            }
        }
    }
    Ok(())
}

/// Invalid routed plan or failure from a generic execution mechanism.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextExecutionError {
    /// Architecture plan, catalog, unit address, or grouped geometry disagreed.
    #[error("invalid routed text execution contract: {0}")]
    Contract(String),
    /// Indexed movement, bank acquisition, grouped construction, or completion failed.
    #[error("routed text execution mechanism failed: {0}")]
    Mechanism(String),
}

fn validate_route_cardinality<T: Tensor>(
    routes: &GroupSelection<T>,
    routes_per_token: usize,
) -> Result<(), RoutedTextExecutionError> {
    let actual = routes
        .group_indices()
        .shape()
        .last()
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    if actual != Some(routes_per_token) {
        return Err(RoutedTextExecutionError::Contract(format!(
            "route cardinality {actual:?} differs from selected routes per token {routes_per_token}"
        )));
    }
    Ok(())
}

/// Resident grouped execution validated against one architecture plan.
pub struct PlannedResidentGatedProduct {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    routes_by_unit: BTreeMap<usize, usize>,
}

impl PlannedResidentGatedProduct {
    /// Validates and retains one replicated routed-unit plan.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        catalog: ExpertResidencyCatalog,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_with_routes(owner_group, plan, catalog, routes_by_unit)
    }

    fn new_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        catalog: ExpertResidencyCatalog,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        validate_plan_catalog::<GatedProductOperation>(&owner_group, &plan, &catalog)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_routes_by_unit::<GatedProductOperation>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        Ok(Self {
            owner_group,
            plan,
            routes_by_unit,
        })
    }

    /// Validates and retains one exact rank-local expert-partition plan.
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_partitioned_with_routes(owner_group, plan, routes_by_unit)
    }

    /// Validates one rank-local plan with exact per-unit route cardinality.
    pub fn new_partitioned_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        if plan.local_global_group_indices().is_empty() {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned routed rank owns no experts".into(),
            ));
        }
        validate_routes_by_unit::<GatedProductOperation>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        if plan
            .unit_specs()
            .keys()
            .any(|(group, _)| group != &owner_group)
        {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned routed plan names a different owner group".into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_by_unit,
        })
    }
}

impl<B> RoutedExpertProvider<B> for PlannedResidentGatedProduct
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        resident_bank
            .forward_grouped(request.input, request.routes, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }

    fn forward_compact_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        validate_route_cardinality(request.routes, 1)?;
        resident_bank
            .forward_grouped(request.input, request.routes, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for PlannedResidentGatedProduct
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        B::gated_product_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
        .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        validate_route_cardinality(request.routes, 1)?;
        B::gated_product_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
        .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

/// Resident ReLU-squared execution validated against one architecture plan.
pub struct PlannedResidentRelu2 {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    routes_per_token: usize,
}

/// Provider installed on a pipeline rank whose exact local routed catalog is empty.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyPartitionRoutedExpertProvider;

impl<B> RoutedExpertProvider<B> for EmptyPartitionRoutedExpertProvider
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received gated-product work".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received ReLU-squared work".into(),
        ))
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for EmptyPartitionRoutedExpertProvider
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel gated-product work".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel ReLU-squared work".into(),
        ))
    }
}

impl PlannedResidentRelu2 {
    /// Validates and retains one replicated routed-unit plan.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
        catalog: ExpertResidencyCatalog,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        validate_plan_catalog::<Relu2Operation>(&owner_group, &plan, &catalog)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        if routes_per_token == 0 {
            return Err(RoutedTextExecutionError::Contract(
                "routes per token must be positive".into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_per_token,
        })
    }

    /// Validates and retains one exact rank-local ReLU-squared expert plan.
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        if plan.local_global_group_indices().is_empty()
            || plan
                .unit_specs()
                .keys()
                .any(|(group, _)| group != &owner_group)
            || routes_per_token == 0
        {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned ReLU-squared plan has no local experts, changes owner, or has no routes"
                    .into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_per_token,
        })
    }
}

impl<B> RoutedExpertProvider<B> for PlannedResidentRelu2
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        validate_route_cardinality(request.routes, self.routes_per_token)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        resident_bank
            .forward_grouped(request.input, request.routes, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for PlannedResidentRelu2
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        validate_route_cardinality(request.routes, self.routes_per_token)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        B::relu2_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
        .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }
}

/// Gated-product grouped-operation projection for the neutral routed driver.
#[derive(Debug, Clone, Copy)]
pub struct GatedProductOperation;

/// ReLU-squared grouped-operation projection for the neutral routed driver.
#[derive(Debug, Clone, Copy)]
pub struct Relu2Operation;

/// Architecture-level geometry shared by resident and addressable grouped execution.
pub trait RoutedGroupedOperationValidation {
    /// Architecture-owned grouped specification.
    type Spec: Clone + PartialEq;

    /// Returns the number of groups described by a specification.
    fn group_count(spec: &Self::Spec) -> i32;

    /// Returns the per-group intermediate width.
    fn intermediate_dimensions(spec: &Self::Spec) -> i32;

    /// Returns every exact parameter target needed by one global member.
    fn member_parameter_targets(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<Vec<String>, eredu_nn::Error>;

    /// Returns exact per-member logical shapes for primary projections and biases.
    fn member_parameter_shapes(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error>;

    /// Projects an architecture-global specification to one compact bank.
    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error>;
}

/// One grouped equation consumable by addressable routed composition.
pub trait RoutedGroupedOperation<B>: RoutedGroupedOperationValidation
where
    B: GroupedNeuralBackend,
{
    /// Constructs and executes the exact grouped equation over acquired storage.
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display;
}

/// Grouped equation that preserves tensor-parallel reduction structure.
pub trait TensorParallelRoutedGroupedOperation<B>: RoutedGroupedOperation<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    /// Constructs and executes one acquired rank-local grouped partial.
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display;
}

impl RoutedGroupedOperationValidation for GatedProductOperation {
    type Spec = eredu_nn::GroupedGatedProductSpec;

    fn group_count(spec: &Self::Spec) -> i32 {
        spec.group_count()
    }

    fn intermediate_dimensions(spec: &Self::Spec) -> i32 {
        spec.intermediate_dimensions()
    }

    fn member_parameter_targets(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        let projections = match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
                vec![gate_up, down]
            }
            eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                let group = groups.get(member).ok_or_else(|| {
                    eredu_nn::Error::backend(format!(
                        "gated-product member {member} has no parameter group"
                    ))
                })?;
                vec![group.gate(), group.up(), group.down()]
            }
            _ => return Err(eredu_nn::Error::backend("unknown gated-product layout")),
        };
        Ok(projections
            .into_iter()
            .flat_map(eredu_nn::GroupedProjectionSpec::parameters)
            .map(|parameter| parameter.id.to_string())
            .collect())
    }

    fn member_parameter_shapes(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error> {
        let input = usize::try_from(spec.input_dimensions()).map_err(eredu_nn::Error::backend)?;
        let intermediate =
            usize::try_from(spec.intermediate_dimensions()).map_err(eredu_nn::Error::backend)?;
        let output = usize::try_from(spec.output_dimensions()).map_err(eredu_nn::Error::backend)?;
        let mut shapes = BTreeMap::new();
        let mut add = |projection: &eredu_nn::GroupedProjectionSpec, shape: Vec<usize>| {
            shapes.insert(projection.weight().id.to_string(), shape.clone());
            if let Some(bias) = projection.bias() {
                shapes.insert(bias.id.to_string(), shape[..shape.len() - 1].to_vec());
            }
            if let Some(companion_shape) = grouped_companion_shape(projection.format(), &shape) {
                if let Some(scale) = projection.format().scale() {
                    shapes.insert(scale.id.to_string(), companion_shape.clone());
                }
                if let Some(bias) = projection.format().affine_bias() {
                    shapes.insert(bias.id.to_string(), companion_shape);
                }
            }
        };
        match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
                add(gate_up, vec![1, intermediate * 2, input]);
                add(down, vec![1, output, intermediate]);
            }
            eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                let group = groups.get(member).ok_or_else(|| {
                    eredu_nn::Error::backend(format!(
                        "gated-product member {member} has no parameter group"
                    ))
                })?;
                add(group.gate(), vec![intermediate, input]);
                add(group.up(), vec![intermediate, input]);
                add(group.down(), vec![output, intermediate]);
            }
            _ => return Err(eredu_nn::Error::backend("unknown gated-product layout")),
        }
        Ok(shapes)
    }

    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error> {
        spec.clone()
            .with_group_geometry(group_count, spec.intermediate_dimensions())
    }
}

impl<B> RoutedGroupedOperation<B> for GatedProductOperation
where
    B: GroupedNeuralBackend,
{
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .gated_product_groups(acquisition, spec, context)
            .map_err(|error| error.to_string())?;
        groups
            .forward_grouped(input, routes, context)
            .map_err(|error| error.to_string())
    }
}

impl<B> TensorParallelRoutedGroupedOperation<B> for GatedProductOperation
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .gated_product_groups(acquisition, spec, context)
            .map_err(|error| error.to_string())?;
        B::gated_product_groups_tensor_parallel(&mut groups, input, routes, partitions, context)
            .map_err(|error| error.to_string())
    }
}

impl RoutedGroupedOperationValidation for Relu2Operation {
    type Spec = eredu_nn::GroupedRelu2Spec;

    fn group_count(spec: &Self::Spec) -> i32 {
        spec.group_count()
    }

    fn intermediate_dimensions(spec: &Self::Spec) -> i32 {
        spec.intermediate_dimensions()
    }

    fn member_parameter_targets(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        Ok([spec.up(), spec.down()]
            .into_iter()
            .flat_map(eredu_nn::GroupedProjectionSpec::parameters)
            .map(|parameter| parameter.id.to_string())
            .collect())
    }

    fn member_parameter_shapes(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error> {
        let hidden = usize::try_from(spec.hidden_dimensions()).map_err(eredu_nn::Error::backend)?;
        let intermediate =
            usize::try_from(spec.intermediate_dimensions()).map_err(eredu_nn::Error::backend)?;
        let mut shapes = BTreeMap::new();
        for (projection, shape) in [
            (spec.up(), vec![1, intermediate, hidden]),
            (spec.down(), vec![1, hidden, intermediate]),
        ] {
            shapes.insert(projection.weight().id.to_string(), shape.clone());
            if let Some(bias) = projection.bias() {
                shapes.insert(bias.id.to_string(), shape[..shape.len() - 1].to_vec());
            }
            if let Some(companion_shape) = grouped_companion_shape(projection.format(), &shape) {
                if let Some(scale) = projection.format().scale() {
                    shapes.insert(scale.id.to_string(), companion_shape.clone());
                }
                if let Some(bias) = projection.format().affine_bias() {
                    shapes.insert(bias.id.to_string(), companion_shape);
                }
            }
        }
        Ok(shapes)
    }

    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error> {
        spec.clone().with_group_count(group_count)
    }
}

fn grouped_companion_shape(
    format: &eredu_nn::LinearFormatSpec,
    weight_shape: &[usize],
) -> Option<Vec<usize>> {
    let rows = *weight_shape.get(weight_shape.len().checked_sub(2)?)?;
    let columns = *weight_shape.last()?;
    let (row_groups, column_groups) = match format.encoding() {
        eredu_checkpoint::LinearFormat::Affine(config) => (
            rows,
            columns.div_ceil(usize::try_from(config.group_size).ok()?),
        ),
        eredu_checkpoint::LinearFormat::MxFp4 => (rows, columns.div_ceil(32)),
        eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) => (
            rows.div_ceil(usize::try_from(config.block_rows).ok()?),
            columns.div_ceil(usize::try_from(config.block_columns).ok()?),
        ),
        eredu_checkpoint::LinearFormat::Dense
        | eredu_checkpoint::LinearFormat::GgufIQuant { .. } => return None,
    };
    let mut shape = weight_shape[..weight_shape.len() - 2].to_vec();
    shape.extend([row_groups, column_groups]);
    Some(shape)
}

impl<B> RoutedGroupedOperation<B> for Relu2Operation
where
    B: GroupedNeuralBackend,
{
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .relu2_groups(acquisition, spec, context)
            .map_err(|error| error.to_string())?;
        groups
            .forward_grouped(input, routes, context)
            .map_err(|error| error.to_string())
    }
}

impl<B> TensorParallelRoutedGroupedOperation<B> for Relu2Operation
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, String>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .relu2_groups(acquisition, spec, context)
            .map_err(|error| error.to_string())?;
        B::relu2_groups_tensor_parallel(&mut groups, input, routes, partitions, context)
            .map_err(|error| error.to_string())
    }
}

/// Addressable grouped execution driven by architecture identities.
pub struct PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Movement: IndexedMovement<B>,
    O: RoutedGroupedOperation<B>,
{
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<O::Spec>,
    catalog: ExpertResidencyCatalog,
    bank: Bank,
    movement: Movement,
    compact_bank_scratch_bytes: u64,
    bulk_compact_bank_target_bytes: u64,
    routes_by_unit: BTreeMap<usize, usize>,
    operation: PhantomData<fn() -> (O, B)>,
}

/// Addressable gated-product execution through the shared neutral driver.
pub type PlannedAddressableGatedProduct<B, Bank, Movement> =
    PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>;

/// Addressable ReLU-squared execution through the shared neutral driver.
pub type PlannedAddressableRelu2<B, Bank, Movement> =
    PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>;

impl<B, A, G, W, E>
    crate::partitioned_execution::PreparedRoutedPartitionedArchitecture<B, A, G, W, E>
where
    B: GroupedNeuralBackend,
{
    /// Returns checkpoint sources excluded by this rank's architecture-owned
    /// expert assignment. Backends consume the resulting names without
    /// interpreting expert ownership.
    pub fn unowned_expert_checkpoint_sources(&self) -> BTreeSet<String> {
        let local = self
            .plan()
            .local_global_group_indices()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        self.catalog()
            .units()
            .iter()
            .filter(|unit| {
                unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
                    && !local.contains(&unit.identity().member())
            })
            .flat_map(|unit| unit.parameters())
            .flat_map(|parameter| parameter.recipe().source_keys())
            .map(str::to_owned)
            .collect()
    }

    /// Returns logical parameter targets materialized exclusively by the
    /// selected independent addressable bank.
    pub fn addressable_logical_targets(&self) -> BTreeSet<String> {
        self.catalog()
            .logical_targets()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// Returns whether one architecture-global unit invokes a routed bank.
    pub fn unit_is_routed(&self, unit: usize) -> bool {
        self.plan()
            .unit_spec(self.owner_group().as_str(), unit)
            .is_some()
    }

    /// Erases the grouped equation after preparation so backend composition
    /// can retain the selected runtime plan without interpreting it.
    pub fn routed_grouped_plan(&self) -> RoutedGroupedPlan
    where
        E: RoutedGroupedSpec,
    {
        E::into_routed_grouped_plan(self.plan().clone())
    }

    /// Selects the exact cross-stage routed collective schedule from the
    /// retained architecture plan.
    #[allow(clippy::too_many_arguments)]
    pub fn collective_wave_schedule_with_tensor_order(
        &self,
        tensor_reductions: &BTreeMap<usize, (usize, usize)>,
        unit_count: usize,
        tensor_partitions: usize,
        tensor_rank: usize,
        pipeline_stages: usize,
        hidden_width: usize,
        output_width: usize,
    ) -> Result<crate::partitioned_execution::RoutedExpertCollectiveWaveSchedule, String>
    where
        E: crate::partitioned_execution::RoutedCollectiveSpec,
    {
        crate::partitioned_execution::routed_expert_collective_wave_schedule_with_tensor_order(
            self.plan(),
            self.owner_group(),
            tensor_reductions,
            unit_count,
            tensor_partitions,
            tensor_rank,
            pipeline_stages,
            hidden_width,
            output_width,
        )
    }
}

impl<B, A, G, W>
    crate::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        B,
        A,
        G,
        W,
        eredu_nn::GroupedGatedProductSpec,
    >
where
    B: GroupedNeuralBackend,
{
    /// Constructs the architecture-owned resident provider without exposing
    /// its expert realization to the backend binder.
    pub fn resident_gated_product_provider(
        &self,
    ) -> Result<PlannedResidentGatedProduct, RoutedTextExecutionError> {
        PlannedResidentGatedProduct::new_partitioned(
            self.owner_group().clone(),
            self.plan().clone(),
            self.provider_routes_per_token(),
        )
    }

    /// Binds generic addressable-bank mechanisms to the retained architecture
    /// plan without transferring expert policy into the backend.
    pub fn addressable_gated_product_provider<Bank, Movement>(
        &self,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<PlannedAddressableGatedProduct<B, Bank, Movement>, RoutedTextExecutionError>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B>,
        Movement::Error: std::fmt::Display,
    {
        PlannedAddressableGatedProduct::new_partitioned(
            self.owner_group().clone(),
            self.plan().clone(),
            self.catalog().clone(),
            selected_member_bytes,
            bank,
            movement,
            options,
            self.provider_routes_per_token(),
        )
    }
}

impl<B, A, G, W>
    crate::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        B,
        A,
        G,
        W,
        eredu_nn::GroupedRelu2Spec,
    >
where
    B: GroupedNeuralBackend,
{
    /// Constructs the architecture-owned resident ReLU-squared provider
    /// without exposing its expert realization to the backend binder.
    pub fn resident_relu2_provider(
        &self,
    ) -> Result<PlannedResidentRelu2, RoutedTextExecutionError> {
        PlannedResidentRelu2::new_partitioned(
            self.owner_group().clone(),
            self.plan().clone(),
            self.provider_routes_per_token(),
        )
    }

    /// Binds generic addressable-bank mechanisms to the retained architecture
    /// ReLU-squared plan without exposing expert policy to the backend.
    pub fn addressable_relu2_provider<Bank, Movement>(
        &self,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<PlannedAddressableRelu2<B, Bank, Movement>, RoutedTextExecutionError>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B>,
        Movement::Error: std::fmt::Display,
    {
        PlannedAddressableRelu2::new_partitioned(
            self.owner_group().clone(),
            self.plan().clone(),
            self.catalog().clone(),
            selected_member_bytes,
            bank,
            movement,
            options,
            self.provider_routes_per_token(),
        )
    }
}

impl<O, B, Bank, Movement> PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    O: RoutedGroupedOperation<B>,
{
    /// Validates plan/catalog coherence before any bank acquisition.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_with_routes(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_by_unit,
        )
    }

    /// Validates and retains one exact rank-local addressable expert plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        options
            .validate()
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        validate_routes_by_unit::<O>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        if plan.local_global_group_indices().is_empty()
            || plan
                .unit_specs()
                .keys()
                .any(|(group, _)| group != &owner_group)
        {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned addressable plan has no local experts or names a different owner group"
                    .into(),
            ));
        }
        let selected_units = selected_member_bytes
            .keys()
            .map(|key| key.unit())
            .collect::<BTreeSet<_>>();
        if selected_units.is_empty() {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned addressable catalog selected no local units".into(),
            ));
        }
        let expected_entries = selected_units
            .len()
            .checked_mul(plan.local_global_group_indices().len())
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(
                    "partitioned addressable catalog cardinality overflowed".into(),
                )
            })?;
        if selected_member_bytes.len() != expected_entries {
            return Err(RoutedTextExecutionError::Contract(format!(
                "partitioned addressable catalog selected {} entries, expected {expected_entries}",
                selected_member_bytes.len()
            )));
        }
        for ((group, unit), spec) in plan
            .unit_specs()
            .iter()
            .filter(|((_, unit), _)| selected_units.contains(unit))
        {
            let local_count = usize::try_from(O::group_count(spec)).map_err(|_| {
                RoutedTextExecutionError::Contract(
                    "partitioned addressable group count is not representable".into(),
                )
            })?;
            if local_count != plan.local_global_group_indices().len() {
                return Err(RoutedTextExecutionError::Contract(format!(
                    "partitioned addressable unit {:?}/{unit} has {local_count} local groups, expected {}",
                    group.as_str(),
                    plan.local_global_group_indices().len()
                )));
            }
            for global in plan.local_global_group_indices() {
                let key = ParameterBankKey::new(*unit, *global);
                let selected = selected_member_bytes.get(&key).copied().ok_or_else(|| {
                    RoutedTextExecutionError::Contract(format!(
                        "partitioned addressable unit {:?}/{unit} is missing global expert {global}",
                        group.as_str()
                    ))
                })?;
                let catalog_unit = catalog.unit(key).ok_or_else(|| {
                    RoutedTextExecutionError::Contract(format!(
                        "partitioned addressable catalog is missing {key:?}"
                    ))
                })?;
                if catalog_unit.owner_group() != group || bank.member_bytes(key) != Some(selected) {
                    return Err(RoutedTextExecutionError::Contract(format!(
                        "partitioned addressable member {key:?} differs from selected ownership or byte geometry"
                    )));
                }
            }
        }
        Ok(Self {
            owner_group,
            plan,
            catalog,
            bank,
            movement,
            compact_bank_scratch_bytes: options.compact_bank_scratch_bytes(),
            bulk_compact_bank_target_bytes: options.prefill_compact_bank_target_bytes(),
            routes_by_unit,
            operation: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        options
            .validate()
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_routes_by_unit::<O>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_catalog::<O, B, Bank>(
            &owner_group,
            &plan,
            &catalog,
            &selected_member_bytes,
            &bank,
        )?;
        Ok(Self {
            owner_group,
            plan,
            catalog,
            bank,
            movement,
            compact_bank_scratch_bytes: options.compact_bank_scratch_bytes(),
            bulk_compact_bank_target_bytes: options.prefill_compact_bank_target_bytes(),
            routes_by_unit,
            operation: PhantomData,
        })
    }

    /// Returns generic storage telemetry without exposing backend types to the architecture.
    pub fn bank_report(&self) -> Result<Bank::Report, RoutedTextExecutionError> {
        self.bank
            .report()
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }

    fn key_for(
        &self,
        owner_unit: usize,
        selected_identity: usize,
    ) -> Result<ParameterBankKey, RoutedTextExecutionError> {
        let global_identity = self
            .plan
            .local_global_group_indices()
            .get(selected_identity)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "selected owner-local group {selected_identity} is outside the rank-local expert plan"
                ))
            })?;
        self.catalog
            .units()
            .iter()
            .filter(|unit| {
                unit.owner_group() == &self.owner_group
                    && unit.identity().unit() == owner_unit
                    && unit.identity().member() == global_identity
            })
            .map(|unit| unit.identity())
            .next()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "selected group {selected_identity} has no bank key for {:?}/{owner_unit}",
                    self.owner_group.as_str()
                ))
            })
    }

    fn acquire_chunk(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Bank::Acquisition, O::Spec, GroupSelection<B::Tensor>), RoutedTextExecutionError>
    {
        let group_count = usize::try_from(O::group_count(spec)).map_err(|_| {
            RoutedTextExecutionError::Contract("grouped bank count is not representable".into())
        })?;
        let demands = self
            .movement
            .index_demands(routes.group_indices(), group_count, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        if demands.is_empty() {
            return Err(RoutedTextExecutionError::Contract(
                "routed selection contains no bank members".into(),
            ));
        }
        if demands.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(RoutedTextExecutionError::Mechanism(
                "indexed demand identities are not strictly ordered".into(),
            ));
        }
        let mut entries = Vec::with_capacity(demands.len());
        let mut mapping = Vec::with_capacity(demands.len());
        let mut scratch_bytes = 0u64;
        for (compact, (identity, demand)) in demands.into_iter().enumerate() {
            let key = self.key_for(owner_unit, identity)?;
            let bytes = self.bank.member_bytes(key).ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "selected bank member {key:?} has no byte geometry"
                ))
            })?;
            scratch_bytes = scratch_bytes.checked_add(bytes).ok_or_else(|| {
                RoutedTextExecutionError::Contract(
                    "selected compact-bank byte geometry overflowed".into(),
                )
            })?;
            entries.push((key, demand));
            mapping.push((identity, compact));
        }
        if scratch_bytes > self.compact_bank_scratch_bytes {
            return Err(RoutedTextExecutionError::Contract(format!(
                "selected compact bank requires {scratch_bytes} bytes, limit is {}",
                self.compact_bank_scratch_bytes
            )));
        }
        let compact_indices = self
            .movement
            .remap_indices(routes.group_indices(), &mapping, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        let compact_routes = GroupSelection::new(
            compact_indices,
            routes.selected_scores().clone(),
            routes.coefficients().clone(),
        );
        let compact_count = i32::try_from(entries.len()).map_err(|_| {
            RoutedTextExecutionError::Contract("compact bank group count exceeds i32".into())
        })?;
        let compact_spec = O::compact_spec(spec, compact_count)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        let acquisition = self
            .bank
            .acquire(ParameterBankAcquisition::new(&entries, access), context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        Ok((acquisition, compact_spec, compact_routes))
    }

    fn execute_chunk(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RoutedTextExecutionError> {
        let (acquisition, compact_spec, compact_routes) =
            self.acquire_chunk(spec, owner_unit, routes, access, context)?;
        let output = O::execute(
            &mut self.bank,
            &acquisition,
            &compact_spec,
            input,
            &compact_routes,
            context,
        )
        .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        self.bank
            .complete(acquisition, &output, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        Ok(output)
    }

    fn execute(
        &mut self,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RoutedTextExecutionError> {
        let spec = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .cloned()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let input_shape = request.input.shape();
        let route_shape = request.routes.group_indices().shape();
        let row_count = |shape: &[i32]| {
            shape[..shape.len().saturating_sub(1)]
                .iter()
                .try_fold(1usize, |total, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|dimension| total.checked_mul(dimension))
                })
        };
        let input_rows = (input_shape.len() >= 2)
            .then(|| row_count(input_shape))
            .flatten();
        let route_rows = (route_shape.len() >= 2)
            .then(|| row_count(route_shape))
            .flatten();
        if input_rows.is_none()
            || input_rows != route_rows
            || request.routes.selected_scores().shape() != route_shape
            || request.routes.coefficients().shape() != route_shape
        {
            return Err(RoutedTextExecutionError::Contract(format!(
                "routed input and selection shapes disagree: input={input_shape:?}, routes={route_shape:?}"
            )));
        }
        let hidden = *input_shape.last().expect("validated hidden axis");
        let selections_per_row = *route_shape.last().expect("validated selection axis");
        let rows = input_rows.expect("validated routed row geometry");
        let flat_rows = i32::try_from(rows).map_err(|_| {
            RoutedTextExecutionError::Contract("routed row count exceeds i32".into())
        })?;
        let input = request
            .input
            .reshape(&[flat_rows, hidden], context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        let flatten_routes = |value: &B::Tensor| {
            value
                .reshape(&[flat_rows, selections_per_row], context)
                .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
        };
        let group_indices = flatten_routes(request.routes.group_indices())?;
        let selected_scores = flatten_routes(request.routes.selected_scores())?;
        let coefficients = flatten_routes(request.routes.coefficients())?;
        let access = request.pass.parameter_bank_access();
        let chunk_rows = if access == eredu_runtime::ParameterBankAccess::Bulk {
            let max_member_bytes = self
                .catalog
                .units()
                .iter()
                .filter(|unit| {
                    unit.owner_group() == &self.owner_group
                        && unit.identity().unit() == request.layer
                })
                .filter_map(|unit| self.bank.member_bytes(unit.identity()))
                .max()
                .ok_or_else(|| {
                    RoutedTextExecutionError::Contract(format!(
                        "execution unit {:?}/{} has no selected bank byte geometry",
                        self.owner_group.as_str(),
                        request.layer
                    ))
                })?;
            let per_row = max_member_bytes
                .checked_mul(u64::try_from(selections_per_row).map_err(|_| {
                    RoutedTextExecutionError::Contract("selection cardinality exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    RoutedTextExecutionError::Contract(
                        "per-row compact-bank byte geometry overflowed".into(),
                    )
                })?;
            usize::try_from(
                self.bulk_compact_bank_target_bytes
                    .checked_div(per_row.max(1))
                    .unwrap_or(0)
                    .max(1),
            )
            .unwrap_or(usize::MAX)
        } else {
            1
        };
        let mut outputs = Vec::new();
        let mut start = 0usize;
        while start < rows {
            let end = start.saturating_add(chunk_rows).min(rows);
            let select = |movement: &mut Movement, value: &B::Tensor| {
                movement
                    .select_rows(value, start, end, context)
                    .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
            };
            let chunk_input = select(&mut self.movement, &input)?;
            let chunk_indices = select(&mut self.movement, &group_indices)?;
            let chunk_scores = select(&mut self.movement, &selected_scores)?;
            let chunk_coefficients = select(&mut self.movement, &coefficients)?;
            let chunk_routes = GroupSelection::new(chunk_indices, chunk_scores, chunk_coefficients);
            outputs.push(self.execute_chunk(
                &spec,
                request.layer,
                &chunk_input,
                &chunk_routes,
                access,
                context,
            )?);
            start = end;
        }
        let output = if outputs.len() == 1 {
            outputs.pop().expect("one routed output")
        } else {
            self.movement
                .concatenate_rows(&outputs, context)
                .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?
        };
        output
            .reshape(input_shape, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
    }
}

impl<B, Bank, Movement> RoutedExpertProvider<B>
    for PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>
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
        _: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute(request, context)
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<O, B, Bank, Movement> PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    O: TensorParallelRoutedGroupedOperation<B>,
{
    fn execute_chunk_tensor_parallel(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, RoutedTextExecutionError> {
        let (acquisition, compact_spec, compact_routes) =
            self.acquire_chunk(spec, owner_unit, routes, access, context)?;
        let output = O::execute_tensor_parallel(
            &mut self.bank,
            &acquisition,
            &compact_spec,
            input,
            &compact_routes,
            partitions,
            context,
        )
        .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        self.bank
            .complete(acquisition, output.reducible(), context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        Ok(output)
    }

    fn execute_tensor_parallel(
        &mut self,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, RoutedTextExecutionError> {
        let spec = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .cloned()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let input_shape = request.input.shape();
        let route_shape = request.routes.group_indices().shape();
        let row_count = |shape: &[i32]| {
            shape[..shape.len().saturating_sub(1)]
                .iter()
                .try_fold(1usize, |total, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|dimension| total.checked_mul(dimension))
                })
        };
        let input_rows = (input_shape.len() >= 2)
            .then(|| row_count(input_shape))
            .flatten();
        let route_rows = (route_shape.len() >= 2)
            .then(|| row_count(route_shape))
            .flatten();
        if input_rows.is_none()
            || input_rows != route_rows
            || request.routes.selected_scores().shape() != route_shape
            || request.routes.coefficients().shape() != route_shape
        {
            return Err(RoutedTextExecutionError::Contract(format!(
                "routed input and selection shapes disagree: input={input_shape:?}, routes={route_shape:?}"
            )));
        }
        let hidden = *input_shape.last().expect("validated hidden axis");
        let selections_per_row = *route_shape.last().expect("validated selection axis");
        let rows = input_rows.expect("validated routed row geometry");
        let flat_rows = i32::try_from(rows).map_err(|_| {
            RoutedTextExecutionError::Contract("routed row count exceeds i32".into())
        })?;
        let input = request
            .input
            .reshape(&[flat_rows, hidden], context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        let flatten_routes = |value: &B::Tensor| {
            value
                .reshape(&[flat_rows, selections_per_row], context)
                .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
        };
        let group_indices = flatten_routes(request.routes.group_indices())?;
        let selected_scores = flatten_routes(request.routes.selected_scores())?;
        let coefficients = flatten_routes(request.routes.coefficients())?;
        let access = request.pass.parameter_bank_access();
        let chunk_rows = if access == eredu_runtime::ParameterBankAccess::Bulk {
            let max_member_bytes = self
                .catalog
                .units()
                .iter()
                .filter(|unit| {
                    unit.owner_group() == &self.owner_group
                        && unit.identity().unit() == request.layer
                })
                .filter_map(|unit| self.bank.member_bytes(unit.identity()))
                .max()
                .ok_or_else(|| {
                    RoutedTextExecutionError::Contract(format!(
                        "execution unit {:?}/{} has no selected bank byte geometry",
                        self.owner_group.as_str(),
                        request.layer
                    ))
                })?;
            let per_row = max_member_bytes
                .checked_mul(u64::try_from(selections_per_row).map_err(|_| {
                    RoutedTextExecutionError::Contract("selection cardinality exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    RoutedTextExecutionError::Contract(
                        "per-row compact-bank byte geometry overflowed".into(),
                    )
                })?;
            usize::try_from(
                self.bulk_compact_bank_target_bytes
                    .checked_div(per_row.max(1))
                    .unwrap_or(0)
                    .max(1),
            )
            .unwrap_or(usize::MAX)
        } else {
            1
        };
        let mut reducible = Vec::new();
        let mut post_reduce = Vec::new();
        let mut start = 0usize;
        while start < rows {
            let end = start.saturating_add(chunk_rows).min(rows);
            let select = |movement: &mut Movement, value: &B::Tensor| {
                movement
                    .select_rows(value, start, end, context)
                    .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
            };
            let chunk_input = select(&mut self.movement, &input)?;
            let chunk_indices = select(&mut self.movement, &group_indices)?;
            let chunk_scores = select(&mut self.movement, &selected_scores)?;
            let chunk_coefficients = select(&mut self.movement, &coefficients)?;
            let chunk_routes = GroupSelection::new(chunk_indices, chunk_scores, chunk_coefficients);
            let output = self.execute_chunk_tensor_parallel(
                &spec,
                request.layer,
                &chunk_input,
                &chunk_routes,
                access,
                partitions,
                context,
            )?;
            let (chunk_reducible, chunk_post_reduce) = output.into_parts();
            reducible.push(chunk_reducible);
            post_reduce.push(chunk_post_reduce);
            start = end;
        }
        let concatenate = |movement: &mut Movement, mut values: Vec<B::Tensor>| {
            if values.len() == 1 {
                Ok(values.pop().expect("one routed tensor-parallel output"))
            } else {
                movement
                    .concatenate_rows(&values, context)
                    .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))
            }
        };
        let reducible = concatenate(&mut self.movement, reducible)?
            .reshape(input_shape, context)
            .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?;
        let post_reduce = if post_reduce.iter().all(Option::is_none) {
            None
        } else if post_reduce.iter().all(Option::is_some) {
            Some(
                concatenate(
                    &mut self.movement,
                    post_reduce.into_iter().flatten().collect(),
                )?
                .reshape(input_shape, context)
                .map_err(|error| RoutedTextExecutionError::Mechanism(error.to_string()))?,
            )
        } else {
            return Err(RoutedTextExecutionError::Contract(
                "tensor-parallel compact chunks disagree on post-reduction bias".into(),
            ));
        };
        Ok(eredu_nn::TensorParallelGroupedOutput::new(
            reducible,
            post_reduce,
        ))
    }
}

impl<B, Bank, Movement> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.execute_tensor_parallel(request, partitions, context)
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<B, Bank, Movement> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.execute_tensor_parallel(request, partitions, context)
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
    }
}

impl<B, Bank, Movement> RoutedExpertProvider<B>
    for PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>
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
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute(request, context)
    }
}

fn validate_replicated_plan<S>(
    plan: &ExpertRealizationPlan<S>,
) -> Result<(), RoutedTextExecutionError> {
    if plan.expert_parallel_size() != 1
        || plan.expert_parallel_rank() != 0
        || plan.local_global_group_indices() != (0..plan.global_expert_count()).collect::<Vec<_>>()
    {
        return Err(RoutedTextExecutionError::Contract(
            "routed text execution requires a complete replicated group plan".into(),
        ));
    }
    Ok(())
}

fn validate_catalog<O, B, Bank>(
    owner_group: &eredu_runtime::ExecutionGroupId,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
    selected_member_bytes: &BTreeMap<ParameterBankKey, u64>,
    bank: &Bank,
) -> Result<(), RoutedTextExecutionError>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    O: RoutedGroupedOperation<B>,
{
    validate_plan_catalog::<O>(owner_group, plan, catalog)
        .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
    if selected_member_bytes.len() != catalog.units().len() {
        return Err(RoutedTextExecutionError::Contract(
            "selected addressable member geometry differs from the architecture catalog".into(),
        ));
    }
    for unit in catalog.units() {
        let expected = selected_member_bytes
            .get(&unit.identity())
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "bank member {:?} has no selected byte geometry",
                    unit.identity()
                ))
            })?;
        if bank.member_bytes(unit.identity()) != Some(expected) {
            return Err(RoutedTextExecutionError::Contract(format!(
                "bank member {:?} byte geometry differs from admitted {expected} bytes",
                unit.identity(),
            )));
        }
    }
    Ok(())
}

fn validate_plan_catalog<O>(
    owner_group: &eredu_runtime::ExecutionGroupId,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
) -> Result<(), RoutedTextRequirementsError>
where
    O: RoutedGroupedOperationValidation,
{
    for ((group, request_unit), spec) in plan.unit_specs() {
        if group != owner_group {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "routed unit group {:?} differs from selected group {:?}",
                group.as_str(),
                owner_group.as_str()
            )));
        }
        let member_count = usize::try_from(O::group_count(spec)).map_err(|_| {
            RoutedTextRequirementsError::Invalid(format!(
                "grouped bank for {:?}/{request_unit} has invalid group count {}",
                group.as_str(),
                O::group_count(spec)
            ))
        })?;
        for member in 0..member_count {
            let mut matches = catalog
                .units()
                .iter()
                .filter(|unit| {
                    unit.owner_group() == group
                        && unit.identity().unit() == *request_unit
                        && unit.identity().member() == member
                })
                .collect::<Vec<_>>();
            if matches.is_empty() {
                matches = catalog
                    .units()
                    .iter()
                    .filter(|unit| {
                        unit.owner_group() == group
                            && unit.owner_unit() == *request_unit
                            && unit.identity().member() == member
                    })
                    .collect();
            }
            let [unit] = matches.as_slice() else {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog must contain one member {member} for {:?}/{request_unit}",
                    group.as_str()
                )));
            };
            if unit.identity() != ParameterBankKey::new(*request_unit, member) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog member {:?} does not use request unit {request_unit} as its key namespace",
                    unit.identity()
                )));
            }
            let expected = O::member_parameter_targets(spec, member)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
            let mut actual = std::collections::BTreeSet::new();
            for parameter in unit.parameters() {
                let target = parameter.logical_target();
                let binding = parameter.binding_name();
                let prefix = target.strip_suffix(binding).ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank binding {binding:?} does not name the suffix of logical target {target:?}"
                    ))
                })?;
                actual.insert(target.to_owned());
                if let crate::ExpertParameterRole::QuantizableProjection {
                    scales_binding,
                    biases_binding,
                } = parameter.role()
                {
                    for companion in [scales_binding, biases_binding] {
                        let target = format!("{prefix}{companion}");
                        if expected.contains(&target) {
                            actual.insert(target);
                        }
                    }
                }
            }
            if actual != expected {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog member {:?} targets {actual:?}, expected grouped targets {expected:?}",
                    unit.identity()
                )));
            }
        }
        let mut unit_facts = catalog
            .units()
            .iter()
            .filter(|unit| unit.owner_group() == group && unit.identity().unit() == *request_unit);
        if let Some(first) = unit_facts.next() {
            if unit_facts.any(|unit| {
                unit.unit_path() != first.unit_path()
                    || unit.owner_unit() != first.owner_unit()
                    || unit.distribution() != first.distribution()
            }) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog members for {:?}/{request_unit} disagree on path, owner, or distribution",
                    group.as_str()
                )));
            }
        }
    }
    for unit in catalog.units() {
        let spec = plan.unit_spec(unit.owner_group().as_str(), unit.identity().unit());
        let member_count = spec
            .and_then(|spec| usize::try_from(O::group_count(spec)).ok())
            .unwrap_or_default();
        if unit.owner_group() != owner_group
            || spec.is_none()
            || unit.identity().member() >= member_count
        {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "bank catalog member {:?} is outside the replicated routed plan",
                unit.identity()
            )));
        }
        if unit.byte_len().is_none() {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "bank catalog member {:?} has no admitted byte geometry",
                unit.identity()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExpertParameterRecipe, ExpertParameterRole, ExpertResidencyDistribution,
        ExpertResidencyUnit,
    };
    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype, RecipeMetadata},
        store::{StoreError, TensorMetadata, TensorSelection},
        LinearFormat, SourceTensorEncoding, StoredDtype,
    };
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ParallelRankTopology,
        ParallelTopology,
    };
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
        GroupedProjectionSpec, LinearFormatSpec, ParameterSpec,
    };
    use eredu_runtime::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
        ArchitectureMergeDestination, ExecutionGraph, ExecutionUnitLayout,
        ParameterTransformConstraint, ReplicatedTextParameterOwner,
        ReplicatedTextParameterPresence, ReplicatedTextParameterRequirement,
        ReplicatedTextParameterRole, ReplicatedTextPhysicalSource, ReplicatedTextStateAccess,
        StateLayout,
    };

    fn grouped_spec() -> GroupedGatedProductSpec {
        let projection = |name| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(name).unwrap(),
                None,
                LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            )
            .unwrap()
        };
        GroupedGatedProductSpec::new(
            2,
            4,
            4,
            4,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("test.experts.gate_up_proj"),
                down: projection("test.experts.down_proj"),
            },
        )
        .unwrap()
    }

    struct TestRecipeCatalog;

    impl RecipeCatalog for TestRecipeCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let shape = if key == "source.down" {
                vec![2, 4, 4]
            } else {
                vec![2, 8, 4]
            };
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: shape.clone(),
                physical_shape: shape.clone(),
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: (shape.iter().product::<usize>() * 2) as u64,
                backing_shard: None,
            })
        }
    }

    struct WrongAffineCompanionCatalog;

    impl RecipeCatalog for WrongAffineCompanionCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let shape = match key {
                "source.gate.weight" => vec![2, 8, 4],
                "source.gate.scales" => vec![2, 8, 2],
                "source.gate.biases" => vec![2, 8, 1],
                "source.down.weight" => vec![2, 4, 4],
                "source.down.scales" | "source.down.biases" => vec![2, 4, 1],
                _ => return Err(StoreError::UnknownTensor { key: key.into() }),
            };
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: shape.clone(),
                physical_shape: shape.clone(),
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: (shape.iter().product::<usize>() * 2) as u64,
                backing_shard: None,
            })
        }
    }

    fn plan(
        owner: &eredu_runtime::ExecutionGroupId,
    ) -> ExpertRealizationPlan<GroupedGatedProductSpec> {
        ExpertRealizationPlan::balanced(
            2,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            BTreeMap::from([((owner.clone(), 0), grouped_spec())]),
        )
        .unwrap()
    }

    fn catalog(
        owner: &eredu_runtime::ExecutionGroupId,
        key_namespace: usize,
        include_down: bool,
        gate_source: &str,
    ) -> ExpertResidencyCatalog {
        ExpertResidencyCatalog::new((0..2).map(|member| {
            let selection = TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let mut parameters = vec![ExpertParameterRecipe::new(
                "gate_up_proj",
                "test.experts.gate_up_proj",
                DerivedWeightRecipe::source(gate_source, selection.clone()),
                ExpertParameterRole::Preserved,
            )
            .unwrap()];
            if include_down {
                parameters.push(
                    ExpertParameterRecipe::new(
                        "down_proj",
                        "test.experts.down_proj",
                        DerivedWeightRecipe::source("source.down", selection),
                        ExpertParameterRole::Preserved,
                    )
                    .unwrap(),
                );
            }
            ExpertResidencyUnit::new(
                ParameterBankKey::new(key_namespace, member),
                owner.clone(),
                0,
                "decoder.layers.0.mlp",
                ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .unwrap()
        }))
        .unwrap()
        .with_inferred_byte_geometry(&TestRecipeCatalog)
        .unwrap()
    }

    fn text_requirements() -> eredu_runtime::ReplicatedTextRequirements {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let parameter = |name: &str, source: &str| {
            let shape = if name.ends_with("gate_up_proj") {
                vec![2, 8, 4]
            } else {
                vec![2, 4, 4]
            };
            ReplicatedTextParameterRequirement::new(
                name,
                vec![source.into()],
                vec![ReplicatedTextPhysicalSource::new(
                    source,
                    "/checkpoint/model.safetensors",
                    source,
                )
                .unwrap()],
                Vec::new(),
                Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                Some(shape.clone()),
                shape,
                LinearFormat::Dense,
                ReplicatedTextParameterRole::LinearWeight,
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: "decoder".into(),
                    unit: 0,
                },
                ReplicatedTextParameterPresence::Required,
                ParameterTransformConstraint::Linear { packed_axis: 2 },
            )
            .unwrap()
        };
        eredu_runtime::ReplicatedTextRequirements::new(
            "test.routed-catalog",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            ReplicatedTextStateAccess::KeyValue,
            vec![
                parameter("test.experts.gate_up_proj", "source.gate"),
                parameter("test.experts.down_proj", "source.down"),
            ],
        )
        .unwrap()
    }

    fn affine_plan_and_catalog(
        owner: &eredu_runtime::ExecutionGroupId,
    ) -> (
        ExpertRealizationPlan<GroupedGatedProductSpec>,
        ExpertResidencyCatalog,
    ) {
        let affine =
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(16, 4).unwrap());
        let projection = |weight: &str, scales: &str, biases: &str| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(weight).unwrap(),
                None,
                LinearFormatSpec::affine(
                    affine,
                    ParameterSpec::trainable(scales).unwrap(),
                    ParameterSpec::trainable(biases).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        let spec = GroupedGatedProductSpec::new(
            2,
            4,
            4,
            4,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection(
                    "test.experts.gate_up_proj",
                    "test.experts.gate_up_proj_scales",
                    "test.experts.gate_up_proj_biases",
                ),
                down: projection(
                    "test.experts.down_proj",
                    "test.experts.down_proj_scales",
                    "test.experts.down_proj_biases",
                ),
            },
        )
        .unwrap();
        let plan = ExpertRealizationPlan::balanced(
            2,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            BTreeMap::from([((owner.clone(), 0), spec)]),
        )
        .unwrap();
        let bindings = [
            ("gate_up_proj", "source.gate.weight"),
            ("gate_up_proj_scales", "source.gate.scales"),
            ("gate_up_proj_biases", "source.gate.biases"),
            ("down_proj", "source.down.weight"),
            ("down_proj_scales", "source.down.scales"),
            ("down_proj_biases", "source.down.biases"),
        ];
        let catalog = ExpertResidencyCatalog::new((0..2).map(|member| {
            let selection = TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let parameters = bindings
                .iter()
                .map(|(binding, source)| {
                    ExpertParameterRecipe::new(
                        *binding,
                        format!("test.experts.{binding}"),
                        DerivedWeightRecipe::source(*source, selection.clone()),
                        ExpertParameterRole::Preserved,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            ExpertResidencyUnit::new(
                ParameterBankKey::new(0, member),
                owner.clone(),
                0,
                "decoder.layers.0.mlp",
                ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .unwrap()
        }))
        .unwrap()
        .with_inferred_byte_geometry(&WrongAffineCompanionCatalog)
        .unwrap();
        (plan, catalog)
    }

    fn affine_text_requirements() -> eredu_runtime::ReplicatedTextRequirements {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let parameter =
            |name: &str, source: &str, shape: Vec<usize>, role: ReplicatedTextParameterRole| {
                ReplicatedTextParameterRequirement::new(
                    name,
                    vec![source.into()],
                    vec![ReplicatedTextPhysicalSource::new(
                        source,
                        "/checkpoint/model.safetensors",
                        source,
                    )
                    .unwrap()],
                    Vec::new(),
                    Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                    Some(shape.clone()),
                    shape,
                    LinearFormat::Dense,
                    role,
                    ReplicatedTextParameterOwner::ExecutionUnit {
                        group: "decoder".into(),
                        unit: 0,
                    },
                    ReplicatedTextParameterPresence::Required,
                    ParameterTransformConstraint::None,
                )
                .unwrap()
            };
        eredu_runtime::ReplicatedTextRequirements::new(
            "test.affine-routed-catalog",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            ReplicatedTextStateAccess::KeyValue,
            vec![
                parameter(
                    "test.experts.gate_up_proj",
                    "source.gate.weight",
                    vec![2, 8, 4],
                    ReplicatedTextParameterRole::LinearWeight,
                ),
                parameter(
                    "test.experts.gate_up_proj_scales",
                    "source.gate.scales",
                    vec![2, 8, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.gate_up_proj_biases",
                    "source.gate.biases",
                    vec![2, 8, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.down_proj",
                    "source.down.weight",
                    vec![2, 4, 4],
                    ReplicatedTextParameterRole::LinearWeight,
                ),
                parameter(
                    "test.experts.down_proj_scales",
                    "source.down.scales",
                    vec![2, 4, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.down_proj_biases",
                    "source.down.biases",
                    vec![2, 4, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn routed_catalog_rejects_wrong_keys_missing_targets_and_unadmitted_sources() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let plan = plan(&owner);
        let wrong_key = validate_plan_catalog::<GatedProductOperation>(
            &owner,
            &plan,
            &catalog(&owner, 1, true, "source.gate"),
        )
        .unwrap_err();
        assert!(
            wrong_key.to_string().contains("key namespace"),
            "{wrong_key}"
        );

        let missing = validate_plan_catalog::<GatedProductOperation>(
            &owner,
            &plan,
            &catalog(&owner, 0, false, "source.gate"),
        )
        .unwrap_err();
        assert!(
            missing.to_string().contains("expected grouped targets"),
            "{missing}"
        );

        let wrong_source = validate_catalog_parameter_topology::<GatedProductOperation>(
            &text_requirements(),
            &plan,
            &catalog(&owner, 0, true, "source.unadmitted"),
            &TestRecipeCatalog,
        )
        .unwrap_err();
        assert!(
            wrong_source
                .to_string()
                .contains("differ from admitted sources"),
            "{wrong_source}"
        );

        let derived = DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::source("source.gate", TensorSelection::Full),
                DerivedWeightRecipe::source("source.gate.extra", TensorSelection::Full),
            ],
        };
        let derived_text = text_requirements()
            .with_derived_recipes(
                BTreeMap::from([("test.experts.gate_up_proj".into(), derived)]),
                BTreeMap::from([(
                    "test.experts.gate_up_proj".into(),
                    RecipeMetadata {
                        shape: vec![2, 16, 4],
                        dtype: RecipeDtype::F16,
                        byte_len: 256,
                    },
                )]),
            )
            .unwrap();
        let removed_source = validate_catalog_parameter_topology::<GatedProductOperation>(
            &derived_text,
            &plan,
            &catalog(&owner, 0, true, "source.gate"),
            &TestRecipeCatalog,
        )
        .expect_err("derived recipe source deletion was accepted");
        assert!(removed_source
            .to_string()
            .contains("differs from the exact admitted member recipe"));
    }

    #[test]
    fn routed_schedule_rejects_coordinated_omission_and_wrong_catalog_path() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let plan = plan(&owner);
        let catalog = catalog(&owner, 0, true, "source.gate");
        let omitted = BTreeMap::from([
            (("decoder".to_owned(), 0), "decoder.layers.0.mlp".to_owned()),
            (("decoder".to_owned(), 1), "decoder.layers.1.mlp".to_owned()),
        ]);
        let error = validate_expected_routed_schedule(&omitted, &plan, &catalog)
            .expect_err("coordinated plan/catalog omission was accepted");
        assert!(
            error.to_string().contains("architecture schedule"),
            "{error}"
        );

        let wrong_path = BTreeMap::from([(
            ("decoder".to_owned(), 0),
            "decoder.layers.0.wrong".to_owned(),
        )]);
        let error = validate_expected_routed_schedule(&wrong_path, &plan, &catalog)
            .expect_err("wrong catalog path was accepted");
        assert!(error.to_string().contains("architecture path"), "{error}");
    }

    #[test]
    fn routed_catalog_rejects_wrong_affine_companion_geometry() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let (plan, catalog) = affine_plan_and_catalog(&owner);
        let error = validate_catalog_parameter_topology::<GatedProductOperation>(
            &affine_text_requirements(),
            &plan,
            &catalog,
            &WrongAffineCompanionCatalog,
        )
        .expect_err("wrong affine scale shape was accepted");
        assert!(error.to_string().contains("recipe shape"), "{error}");
    }
}
