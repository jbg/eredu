//! Owned consumer projection of family neural declarations, without native resources.
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearFormatSpec, LogicalParameterDeclaration,
    LogicalParameterRole, ParameterSpec,
};
use eredu_runtime::ReplicatedTextParameterOwner;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) struct Parameter {
    pub spec: ParameterSpec,
    pub shape: Vec<usize>,
    pub role: LogicalParameterRole,
    pub format: Option<LinearFormatSpec>,
    pub owner: ReplicatedTextParameterOwner,
}
/// Complete assistant parameter facts projected from its typed construction scopes.
/// Physical aliases and checkpoint recipes remain separate source extensions.
#[derive(Debug, Clone, Default)]
pub struct AssistantParameterDeclarations {
    pub(crate) parameters: BTreeMap<String, Parameter>,
}
impl AssistantParameterDeclarations {
    /// Creates an empty projection; complete preparation validates all scopes together.
    pub fn new() -> Self {
        Self::default()
    }
    /// Retains one independently acquired scope without deriving ownership from identities.
    pub fn append(
        &mut self,
        owner: ReplicatedTextParameterOwner,
        scope: &impl DeclareLogicalParameters,
    ) -> Result<(), Error> {
        for declaration in scope.logical_parameters()? {
            let key = declaration.parameter.id.to_string();
            if self.parameters.contains_key(&key) {
                return Err(Error::backend(format!(
                    "duplicate assistant parameter {key:?}"
                )));
            }
            self.parameters.insert(
                key,
                Parameter {
                    spec: declaration.parameter.clone(),
                    shape: declaration.shape,
                    role: declaration.role,
                    format: declaration.linear_format.cloned(),
                    owner: owner.clone(),
                },
            );
        }
        Ok(())
    }
    /// Projects exact physical weights and companions using the shared encoding visitor.
    pub fn physical_parameters(
        &self,
    ) -> Result<BTreeMap<String, eredu_runtime::ParameterMemberSpec>, Error> {
        self.validate()?;
        let mut members = BTreeMap::new();
        for parameter in self.logical_parameters()? {
            let member = eredu_runtime::ParameterMemberSpec::from_logical_declaration(
                &parameter,
                eredu_runtime::MemberSharding::Replicated,
            );
            let expanded = match parameter.linear_format {
                Some(format) => eredu_runtime::expand_linear_format_parameter(&member, format)
                    .map_err(Error::backend_source)?,
                None => vec![member],
            };
            for member in expanded {
                let name = member.target().to_owned();
                if members.insert(name.clone(), member).is_some() {
                    return Err(Error::backend(format!(
                        "conflicting assistant physical parameter {name:?}"
                    )));
                }
            }
        }
        Ok(members)
    }

    pub(crate) fn validate_projection(&self, target: &Self) -> Result<(), Error> {
        self.validate()?;
        target.validate()?;
        if self.parameters.len() != target.parameters.len() {
            return Err(Error::backend(
                "assistant projection changes logical parameter identities",
            ));
        }
        for (name, source) in &self.parameters {
            let selected = target
                .parameters
                .get(name)
                .ok_or_else(|| Error::backend("assistant projection omits a logical parameter"))?;
            let mut source_spec = source.spec.clone();
            if source.format.is_some() && selected.format.is_some() {
                source_spec.linear_row_layout = selected.spec.linear_row_layout;
            }
            if source_spec != selected.spec
                || source.shape != selected.shape
                || source.role != selected.role
                || source.owner != selected.owner
                || source.format.is_some() != selected.format.is_some()
            {
                return Err(Error::backend(format!(
                    "assistant projection changes logical declaration {name:?}"
                )));
            }
        }
        Ok(())
    }

    /// Rejects unresolved sharing, incompatible geometry and conflicting companion references.
    pub fn validate(&self) -> Result<(), Error> {
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for AssistantParameterDeclarations {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(self
            .parameters
            .values()
            .map(|p| LogicalParameterDeclaration {
                parameter: &p.spec,
                shape: p.shape.clone(),
                role: p.role,
                linear_format: p.format.as_ref(),
            })
            .collect())
    }
}
