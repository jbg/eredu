//! Decoder equation order, parameter ownership and cache references.
use super::Config;
use eredu_nn::Error;
use std::{collections::BTreeMap, ops::Range};

/// One independently leased equation in decoder execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderOperation {
    /// Attention and feed-forward body using the logical layer's independent cache.
    Body {
        /// Independent logical attention-cache slot.
        layer: usize,
    },
    /// Stateless normalization after a completed logical layer.
    Terminal {
        /// Logical layer whose output is normalized.
        layer: usize,
    },
}
impl DecoderOperation {
    /// Logical layer whose policy or boundary this equation implements.
    pub fn layer(self) -> usize {
        match self {
            Self::Body { layer } | Self::Terminal { layer } => layer,
        }
    }
}

/// Validated projections of a decoder's body and normalization equations.
#[derive(Debug, Clone)]
pub struct DecoderSchedule {
    operations: Vec<DecoderOperation>,
    body_operations: Vec<usize>,
    owners: Vec<usize>,
}
impl DecoderSchedule {
    /// Resolves explicit family sharing independently of cache and operation identities.
    pub fn new(config: &impl Config) -> Result<Self, Error> {
        let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
        let mut operations = Vec::new();
        let mut body_operations = Vec::with_capacity(layers);
        let mut terminals = BTreeMap::new();
        for layer in 0..layers {
            body_operations.push(operations.len());
            operations.push(DecoderOperation::Body { layer });
            if let Some(name) = config.block_output_normalization(layer) {
                let key = config.parameter_alias(&name).unwrap_or(name);
                terminals.entry(key).or_insert(operations.len());
                operations.push(DecoderOperation::Terminal { layer });
            }
        }
        let mut owners = Vec::with_capacity(operations.len());
        for operation in &operations {
            owners.push(match *operation {
                DecoderOperation::Body { layer } => {
                    let owner = config.block_parameter_owner(layer);
                    if owner >= layers || config.block_parameter_owner(owner) != owner {
                        return Err(Error::backend(format!(
                            "decoder layer {layer} has invalid body owner {owner}"
                        )));
                    }
                    body_operations[owner]
                }
                DecoderOperation::Terminal { layer } => {
                    let name = config
                        .block_output_normalization(layer)
                        .expect("declared terminal");
                    let key = config.parameter_alias(&name).unwrap_or(name);
                    terminals[&key]
                }
            });
        }
        Ok(Self {
            operations,
            body_operations,
            owners,
        })
    }
    /// Complete operation count, independent of cache count.
    pub fn len(&self) -> usize {
        self.operations.len()
    }
    /// Whether no equation is declared.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
    /// Exact equation at one architecture operation address.
    pub fn operation(&self, index: usize) -> Result<DecoderOperation, Error> {
        self.operations
            .get(index)
            .copied()
            .ok_or_else(|| Error::backend(format!("unknown decoder operation {index}")))
    }
    /// Canonical physical owner of an invocation's leased resource.
    pub fn owner(&self, index: usize) -> usize {
        self.owners[index]
    }
    /// Body operation associated with a logical layer/cache slot.
    pub fn body_operation(&self, layer: usize) -> usize {
        self.body_operations[layer]
    }
    /// Whether a body delegates its final output boundary to a following terminal.
    pub fn has_terminal(&self, layer: usize) -> bool {
        matches!(self.operations.get(self.body_operations[layer] + 1), Some(DecoderOperation::Terminal { layer: after }) if *after == layer)
    }
    /// Independently leased resources and their exact logical consumers.
    pub fn resources(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut resources = BTreeMap::<usize, Vec<usize>>::new();
        for (operation, &owner) in self.owners.iter().enumerate() {
            resources.entry(owner).or_default().push(operation);
        }
        resources
    }
    /// Projects explicit cache ownership for source-free selection and execution.
    pub fn state_partition_plan(
        &self,
        layout: &eredu_runtime::ExecutionUnitLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new(
            self.operations.iter().enumerate().filter_map(
                |(ordinal, operation)| match *operation {
                    DecoderOperation::Body { layer } => {
                        Some(eredu_runtime::ArchitectureStatePartitionRule::invocation(
                            layout.address(ordinal).expect("declared decoder operation"),
                            layer..layer + 1,
                        ))
                    }
                    DecoderOperation::Terminal { .. } => None,
                },
            ),
        )
    }
    /// Complete logical cache range used by a contiguous operation selection.
    pub fn state_range(&self, operations: Range<usize>) -> Option<Range<usize>> {
        let mut slots = operations.filter_map(|index| match self.operations[index] {
            DecoderOperation::Body { layer } => Some(layer),
            DecoderOperation::Terminal { .. } => None,
        });
        let first = slots.next()?;
        Some(slots.fold(first..first + 1, |range, slot| {
            range.start.min(slot)..range.end.max(slot + 1)
        }))
    }
    /// Stable equation identity, independent of canonical parameter names.
    pub fn path(&self, config: &impl Config, operation: usize) -> Result<String, Error> {
        let equation = self.operation(operation)?;
        let layer = equation.layer();
        Ok(match equation {
            DecoderOperation::Body { .. } => format!("{}.layers.{layer}", config.parameter_root()),
            DecoderOperation::Terminal { .. } => {
                format!("{}.layers.{layer}.output_norm", config.parameter_root())
            }
        })
    }
    /// Original logical input/output timing across body and terminal leases.
    pub fn boundaries(
        &self,
        config: &impl Config,
        operation: usize,
    ) -> eredu_runtime::UnitObservationBoundaries {
        let equation = self.operations[operation];
        let path = format!("{}.layers.{}", config.parameter_root(), equation.layer());
        match equation {
            DecoderOperation::Body { layer } => eredu_runtime::UnitObservationBoundaries::new(
                Some(format!("{path}.input")),
                (!self.has_terminal(layer)).then(|| format!("{path}.output")),
            ),
            DecoderOperation::Terminal { .. } => {
                eredu_runtime::UnitObservationBoundaries::new(None, Some(format!("{path}.output")))
            }
        }
    }
}
