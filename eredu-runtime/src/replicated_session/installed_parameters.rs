use super::*;
use crate::parameter_operations::{
    prepare_bank_parameter_slots, PreparedBankParameter, PreparedBankParameterMember,
    PreparedParameterLocation, PreparedParameterSlot,
};
use eredu_checkpoint::quantization::WeightMaterializationReport;
use eredu_core::parameters::ParameterError;

/// Materialization facts for one bound prediction module.
pub struct PredictionModuleParameters {
    /// Module ordinal in the selected prediction executable.
    pub ordinal: usize,
    /// Exact physical parameter slots.
    pub slots: Vec<PreparedParameterSlot>,
    /// Exact materialization requests.
    pub tasks: Vec<ReplicatedTextMaterializationTask>,
    /// Observed checkpoint transformations.
    pub report: WeightMaterializationReport,
}

#[derive(Clone)]
pub(super) struct InstalledTextParameters {
    slots: Vec<PreparedParameterSlot>,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    banks: Vec<PreparedBankParameter>,
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Complete published catalog, including auxiliary modules and addressable banks.
    pub fn installed_parameter_slots(&self) -> &[PreparedParameterSlot] {
        match &self.installed_parameters {
            Some(parameters) => &parameters.slots,
            None => self.prepared_parameter_slots(),
        }
    }

    /// Materialization tasks for the installed target and prediction modules.
    pub fn installed_parameter_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        match &self.installed_parameters {
            Some(parameters) => &parameters.tasks,
            None => self.parameter_materialization_tasks(),
        }
    }

    /// Selected bank slots used when lending the corresponding native parameters.
    pub fn installed_bank_parameters(&self) -> &[PreparedBankParameter] {
        self.installed_parameters
            .as_ref()
            .map_or(&[], |parameters| &parameters.banks)
    }

    fn copy_installed_parameters(&self) -> InstalledTextParameters {
        self.installed_parameters
            .clone()
            .unwrap_or_else(|| InstalledTextParameters {
                slots: self.prepared_parameter_slots().to_vec(),
                tasks: self.parameter_materialization_tasks().to_vec(),
                banks: Vec::new(),
            })
    }

    /// Binds the exact native bank members to the selected portable parameter declarations.
    pub fn install_bank_parameters(
        &mut self,
        members: impl IntoIterator<Item = PreparedBankParameterMember>,
    ) -> Result<(), ParameterError> {
        let banks = prepare_bank_parameter_slots(self.parameter_declarations(), members)?;
        let mut parameters = self.copy_installed_parameters();
        if !parameters.banks.is_empty() {
            return Err(ParameterError::Invalid(
                "parameter banks are already installed".into(),
            ));
        }
        parameters
            .slots
            .extend(banks.iter().map(|parameter| parameter.slot.clone()));
        parameters
            .slots
            .sort_unstable_by(|a, b| a.parameter.id.cmp(&b.parameter.id));
        parameters.banks = banks;
        self.installed_parameters = Some(parameters);
        Ok(())
    }

    /// Validates and publishes the auxiliary parameter catalog as one operation.
    pub fn install_prediction_parameters(
        &mut self,
        modules: impl IntoIterator<Item = PredictionModuleParameters>,
    ) -> Result<(), ParameterError> {
        let mut parameters = self.copy_installed_parameters();
        let mut report = WeightMaterializationReport::default();
        for module in modules {
            for mut slot in module.slots {
                if parameters
                    .slots
                    .iter()
                    .any(|existing| existing.parameter.id == slot.parameter.id)
                {
                    return Err(ParameterError::Invalid(format!(
                        "prediction parameter {} has multiple prepared owners",
                        slot.parameter.id.as_str()
                    )));
                }
                slot.location = PreparedParameterLocation::Prediction {
                    module: module.ordinal,
                };
                parameters.slots.push(slot);
            }
            for task in module.tasks {
                match parameters
                    .tasks
                    .iter()
                    .find(|existing| existing.name() == task.name())
                {
                    Some(existing) if existing != &task => {
                        return Err(ParameterError::Invalid(
                            "prediction materialization task disagrees with retained owner".into(),
                        ));
                    }
                    Some(_) => {}
                    None => parameters.tasks.push(task),
                }
            }
            report.merge(module.report);
        }
        parameters
            .slots
            .sort_unstable_by(|a, b| a.parameter.id.cmp(&b.parameter.id));
        self.installed_parameters = Some(parameters);
        if report.transformed_weights > 0 {
            self.record_auxiliary_materialization(report);
        }
        Ok(())
    }
}
