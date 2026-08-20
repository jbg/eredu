//! Typed multimodal component graphs and residency classes.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Logical value domain crossing a component boundary.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComponentDomain {
    /// Integer token identities.
    TokenIds,
    /// Prepacked image or video patches.
    PatchMatrix,
    /// Prepared audio features or codebook identities.
    AudioFeatures,
    /// Decoder-width hidden activations.
    HiddenStates,
    /// Vocabulary logits.
    Logits,
    /// Ordered target block states consumed by a draft model.
    TargetStates,
}

/// Semantic execution-unit kind without family equations.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComponentKind {
    /// Token embedding, normalization, or another immutable text root.
    StaticText,
    /// Vision encoder or projector unit.
    Vision,
    /// Audio encoder or projector unit.
    Audio,
    /// Ordered text/media assembly.
    Assembly,
    /// One decoder layer.
    Decoder,
    /// Embedded multi-token prediction unit.
    Prediction,
    /// External assistant unit.
    Assistant,
    /// Final vocabulary projection.
    OutputProjection,
}

/// Immutable-weight residency accounting class.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComponentResidencyClass {
    /// Small static text modules.
    Static,
    /// Modality-specific tower and prepared-media workspace.
    Media,
    /// Independently resident or streamed decoder unit.
    Decoder,
    /// Independently leased routed experts.
    Experts,
    /// Embedded or external draft modules.
    Draft,
}

/// One typed unit in a composite execution graph.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentSpec {
    /// Stable architecture-owned identity.
    pub id: String,
    /// Semantic component kind.
    pub kind: ComponentKind,
    /// External input domains in declared order.
    pub external_inputs: Vec<ComponentDomain>,
    /// Dependency unit identities in declared order.
    pub dependencies: Vec<String>,
    /// Required output domain for each dependency.
    pub dependency_inputs: Vec<ComponentDomain>,
    /// Unit output domain.
    pub output: ComponentDomain,
    /// Weight-residency accounting class.
    pub residency: ComponentResidencyClass,
}

/// Validated typed component graph with one or more observable outputs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentGraph {
    units: Vec<ComponentSpec>,
    execution_order: Vec<usize>,
    outputs: Vec<usize>,
}

impl ComponentGraph {
    /// Validates identities, dependency domains, acyclicity, and outputs.
    pub fn new(
        units: Vec<ComponentSpec>,
        outputs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, ComponentGraphError> {
        if units.is_empty() {
            return Err(ComponentGraphError::Empty);
        }
        let mut by_id = BTreeMap::new();
        for (index, unit) in units.iter().enumerate() {
            if unit.id.trim().is_empty() {
                return Err(ComponentGraphError::EmptyId);
            }
            if unit.dependencies.len() != unit.dependency_inputs.len() {
                return Err(ComponentGraphError::DependencyArity {
                    unit: unit.id.clone(),
                    dependencies: unit.dependencies.len(),
                    domains: unit.dependency_inputs.len(),
                });
            }
            if by_id.insert(unit.id.clone(), index).is_some() {
                return Err(ComponentGraphError::DuplicateId(unit.id.clone()));
            }
        }
        let mut edges = vec![Vec::new(); units.len()];
        let mut indegree = vec![0_usize; units.len()];
        for (unit_index, unit) in units.iter().enumerate() {
            let mut seen = BTreeSet::new();
            for (slot, dependency) in unit.dependencies.iter().enumerate() {
                let dependency_index = by_id.get(dependency).copied().ok_or_else(|| {
                    ComponentGraphError::UnknownDependency {
                        unit: unit.id.clone(),
                        dependency: dependency.clone(),
                    }
                })?;
                if !seen.insert(dependency_index) {
                    return Err(ComponentGraphError::DuplicateDependency {
                        unit: unit.id.clone(),
                        dependency: dependency.clone(),
                    });
                }
                let actual = units[dependency_index].output;
                let expected = unit.dependency_inputs[slot];
                if actual != expected {
                    return Err(ComponentGraphError::DomainMismatch {
                        unit: unit.id.clone(),
                        dependency: dependency.clone(),
                        expected,
                        actual,
                    });
                }
                edges[dependency_index].push(unit_index);
                indegree[unit_index] += 1;
            }
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, degree)| (*degree == 0).then_some(index))
            .collect::<VecDeque<_>>();
        let mut execution_order = Vec::with_capacity(units.len());
        while let Some(index) = ready.pop_front() {
            execution_order.push(index);
            for dependent in &edges[index] {
                indegree[*dependent] -= 1;
                if indegree[*dependent] == 0 {
                    ready.push_back(*dependent);
                }
            }
        }
        if execution_order.len() != units.len() {
            return Err(ComponentGraphError::Cycle);
        }
        let outputs = outputs
            .into_iter()
            .map(|output| {
                let output = output.as_ref();
                by_id
                    .get(output)
                    .copied()
                    .ok_or_else(|| ComponentGraphError::UnknownOutput(output.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if outputs.is_empty() {
            return Err(ComponentGraphError::NoOutputs);
        }
        if outputs.iter().copied().collect::<BTreeSet<_>>().len() != outputs.len() {
            return Err(ComponentGraphError::DuplicateOutput);
        }
        Ok(Self {
            units,
            execution_order,
            outputs,
        })
    }

    /// Returns units in architecture declaration order.
    pub fn units(&self) -> &[ComponentSpec] {
        &self.units
    }

    /// Returns units in dependency-safe execution order.
    pub fn execution_order(&self) -> impl Iterator<Item = &ComponentSpec> {
        self.execution_order.iter().map(|index| &self.units[*index])
    }

    /// Returns graph outputs in declared order.
    pub fn outputs(&self) -> impl Iterator<Item = &ComponentSpec> {
        self.outputs.iter().map(|index| &self.units[*index])
    }

    /// Counts components in one residency class.
    pub fn residency_count(&self, class: ComponentResidencyClass) -> usize {
        self.units
            .iter()
            .filter(|unit| unit.residency == class)
            .count()
    }
}

/// Invalid component graph declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ComponentGraphError {
    /// No components were declared.
    #[error("component graph cannot be empty")]
    Empty,
    /// One component identity was empty.
    #[error("component identity cannot be empty")]
    EmptyId,
    /// A component identity occurred more than once.
    #[error("component {0:?} was declared more than once")]
    DuplicateId(String),
    /// Dependency and expected-domain counts differ.
    #[error("component {unit:?} has {dependencies} dependencies but {domains} dependency domains")]
    DependencyArity {
        /// Component identity.
        unit: String,
        /// Dependency count.
        dependencies: usize,
        /// Expected-domain count.
        domains: usize,
    },
    /// A dependency identity was not declared.
    #[error("component {unit:?} depends on unknown component {dependency:?}")]
    UnknownDependency {
        /// Consumer component.
        unit: String,
        /// Missing dependency.
        dependency: String,
    },
    /// The same dependency was listed twice.
    #[error("component {unit:?} repeats dependency {dependency:?}")]
    DuplicateDependency {
        /// Consumer component.
        unit: String,
        /// Repeated dependency.
        dependency: String,
    },
    /// Producer and consumer domains disagree.
    #[error("component {unit:?} expects {expected:?} from {dependency:?}, got {actual:?}")]
    DomainMismatch {
        /// Consumer component.
        unit: String,
        /// Producer component.
        dependency: String,
        /// Required domain.
        expected: ComponentDomain,
        /// Produced domain.
        actual: ComponentDomain,
    },
    /// Dependency graph contains a cycle.
    #[error("component graph contains a dependency cycle")]
    Cycle,
    /// No observable output was declared.
    #[error("component graph requires at least one output")]
    NoOutputs,
    /// One output identity was repeated.
    #[error("component graph repeats an output")]
    DuplicateOutput,
    /// An output identity was not declared.
    #[error("unknown component output {0:?}")]
    UnknownOutput(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(id: &str, output: ComponentDomain) -> ComponentSpec {
        ComponentSpec {
            id: id.into(),
            kind: ComponentKind::StaticText,
            external_inputs: vec![],
            dependencies: vec![],
            dependency_inputs: vec![],
            output,
            residency: ComponentResidencyClass::Static,
        }
    }

    #[test]
    fn component_graph_validates_domains_and_preserves_outputs() {
        let embedding = ComponentSpec {
            external_inputs: vec![ComponentDomain::TokenIds],
            ..unit("embedding", ComponentDomain::HiddenStates)
        };
        let decoder = ComponentSpec {
            id: "decoder.0".into(),
            kind: ComponentKind::Decoder,
            dependencies: vec!["embedding".into()],
            dependency_inputs: vec![ComponentDomain::HiddenStates],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Decoder,
            external_inputs: vec![],
        };
        let graph = ComponentGraph::new(vec![decoder, embedding], ["decoder.0"]).unwrap();
        assert_eq!(
            graph
                .execution_order()
                .map(|unit| unit.id.as_str())
                .collect::<Vec<_>>(),
            ["embedding", "decoder.0"]
        );
        assert_eq!(graph.residency_count(ComponentResidencyClass::Decoder), 1);
    }

    #[test]
    fn component_graph_rejects_domain_mismatch_and_cycles() {
        let mut decoder = unit("decoder", ComponentDomain::HiddenStates);
        decoder.dependencies = vec!["tokens".into()];
        decoder.dependency_inputs = vec![ComponentDomain::HiddenStates];
        let tokens = unit("tokens", ComponentDomain::TokenIds);
        assert!(matches!(
            ComponentGraph::new(vec![tokens, decoder], ["decoder"]),
            Err(ComponentGraphError::DomainMismatch { .. })
        ));
        let mut left = unit("left", ComponentDomain::HiddenStates);
        left.dependencies = vec!["right".into()];
        left.dependency_inputs = vec![ComponentDomain::HiddenStates];
        let mut right = unit("right", ComponentDomain::HiddenStates);
        right.dependencies = vec!["left".into()];
        right.dependency_inputs = vec![ComponentDomain::HiddenStates];
        assert_eq!(
            ComponentGraph::new(vec![left, right], ["left"]).unwrap_err(),
            ComponentGraphError::Cycle
        );
    }
}
